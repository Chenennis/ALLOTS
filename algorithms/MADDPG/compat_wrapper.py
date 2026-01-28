"""
Compatibility Wrapper for MADDPG with Device Churn

This wrapper enables FOMADDPG to work with FOgym's dynamic device sets
by using the compatibility layer (SlotMapper, ObsAdapter, ActAdapter).

Author: FOenv Team
Date: 2026-01-13
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from adapters.multi_manager_wrapper import MultiManagerCompatWrapper
from algorithms.MADDPG.fomaddpg.fomaddpg import FOMADDPG


class MADDPGCompatAgent:
    """
    Compatibility wrapper for MADDPG with device churn.
    
    This class wraps FOMADDPG and handles:
    - Variable-length observations → Fixed padded format
    - Fixed padded actions → Variable-length action_sets
    - Mask-aware training (no gradient leakage)
    """
    
    def __init__(
        self,
        manager_ids: List[str],
        N_max: int,
        x_dim: int,
        g_dim: int,
        p: int,
        # MADDPG hyperparameters (from paper)
        lr_actor: float = 1e-2,
        lr_critic: float = 1e-2,
        hidden_dim: int = 64,
        gamma: float = 0.95,
        tau: float = 0.01,
        noise_scale: float = 0.1,
        buffer_capacity: int = 1000000,
        batch_size: int = 1024,
        device: str = "cpu"
    ):
        """
        Initialize MADDPG compatibility wrapper.
        
        Args:
            manager_ids: List of manager IDs
            N_max: Maximum device slots per manager
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            lr_actor: Actor learning rate (MADDPG paper: 1e-2)
            lr_critic: Critic learning rate (MADDPG paper: 1e-2)
            hidden_dim: Hidden layer dimension (MADDPG paper: 64)
            gamma: Discount factor (MADDPG paper: 0.95)
            tau: Soft update coefficient (MADDPG paper: 0.01)
            noise_scale: Exploration noise (MADDPG paper: 0.1)
            buffer_capacity: Replay buffer size
            batch_size: Batch size (MADDPG paper: 1024)
            device: Computing device
        """
        self.manager_ids = manager_ids
        self.n_managers = len(manager_ids)
        self.N_max = N_max
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.device = device
        
        # Create compatibility wrapper
        self.compat_wrapper = MultiManagerCompatWrapper(
            use_stable_mapping=True,  # 使用稳定映射
            manager_ids=manager_ids,
            N_max=N_max,
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            verbose=False
        )
        
        # Get fixed dimensions
        state_dim, action_dim = self.compat_wrapper.get_state_action_dims()
        
        # Create MADDPG agent
        self.maddpg = FOMADDPG(
            n_agents=self.n_managers,
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=hidden_dim,
            max_action=1.0,  # Actions will be clipped in FOgym
            gamma=gamma,
            tau=tau,
            noise_scale=noise_scale,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            device=device
        )
        
        print(f"MADDPGCompatAgent initialized:")
        print(f"  Managers: {self.n_managers}")
        print(f"  State dim per manager: {state_dim} (g={g_dim}, X_pad={N_max}×{x_dim}, mask={N_max})")
        print(f"  Action dim per manager: {action_dim} (N_max={N_max} × p={p})")
        print(f"  Hyperparameters: lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}")
    
    def select_actions(
        self,
        raw_obs: Dict[str, Dict],
        explore: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Select actions for all managers.
        
        Args:
            raw_obs: Raw observations from FOgym
                     {manager_id: {'g': [g_dim], 'device_ids': [...], 'device_states': {...}}}
            explore: Whether to add exploration noise
            
        Returns:
            padded_actions: {manager_id: A_pad [N_max, p]}
        """
        # Adapt observations to fixed format
        adapted_obs = self.compat_wrapper.adapt_obs_all(raw_obs, format='separate')
        
        # Stack obs_vec for all managers
        states = np.stack([
            adapted_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)  # [n_managers, state_dim]
        
        # Get masks for all managers
        masks = np.stack([
            adapted_obs[mid]['mask']
            for mid in self.manager_ids
        ], axis=0)  # [n_managers, N_max]
        
        # Select actions using MADDPG
        actions = self.maddpg.select_actions(states, add_noise=explore)  # [n_managers, action_dim]
        
        # Reshape and apply masks
        padded_actions = {}
        for i, manager_id in enumerate(self.manager_ids):
            # Reshape to [N_max, p]
            A_pad = actions[i].reshape(self.N_max, self.p)
            
            # Apply mask to zero out inactive slots
            mask = masks[i][:, np.newaxis]  # [N_max, 1]
            A_pad = A_pad * mask
            
            padded_actions[manager_id] = A_pad
        
        return padded_actions
    
    def store_transition(
        self,
        obs: Dict[str, Dict],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, Dict],
        done: bool
    ):
        """
        Store transition in replay buffer.
        
        Args:
            obs: Current observations
            actions: Padded actions {manager_id: [N_max, p]}
            rewards: Rewards {manager_id: float}
            next_obs: Next observations
            done: Episode done flag
        """
        # Adapt observations
        adapted_obs = self.compat_wrapper.adapt_obs_all(obs, format='separate')
        adapted_next_obs = self.compat_wrapper.adapt_obs_all(next_obs, format='separate')
        
        # Stack states
        states = np.stack([
            adapted_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)  # [n_managers, state_dim]
        
        next_states = np.stack([
            adapted_next_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)  # [n_managers, state_dim]
        
        # Flatten actions
        actions_array = np.stack([
            actions[mid].flatten()
            for mid in self.manager_ids
        ], axis=0)  # [n_managers, action_dim]
        
        # Stack rewards and dones
        rewards_array = np.array([
            rewards.get(mid, 0.0)
            for mid in self.manager_ids
        ])  # [n_managers]
        
        dones_array = np.array([done] * self.n_managers)  # [n_managers]
        
        # Store in MADDPG replay buffer
        self.maddpg.store_experience(
            states=states,
            actions=actions_array,
            rewards=rewards_array,
            next_states=next_states,
            dones=dones_array
        )
    
    def update(self) -> Dict[str, float]:
        """
        Update MADDPG policy.
        
        Returns:
            metrics: Training metrics (actor_loss, critic_loss, etc.)
        """
        return self.maddpg.update()
    
    def get_buffer_size(self) -> int:
        """Get current replay buffer size."""
        return len(self.maddpg.replay_buffer)
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'maddpg': {
                'agents': [
                    {
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'actor_target': agent.actor_target.state_dict(),
                        'critic_target': agent.critic_target.state_dict(),
                        'actor_optimizer': agent.actor_optimizer.state_dict(),
                        'critic_optimizer': agent.critic_optimizer.state_dict(),
                    }
                    for agent in self.maddpg.agents
                ]
            }
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, agent in enumerate(self.maddpg.agents):
            agent_ckpt = checkpoint['maddpg']['agents'][i]
            agent.actor.load_state_dict(agent_ckpt['actor'])
            agent.critic.load_state_dict(agent_ckpt['critic'])
            agent.actor_target.load_state_dict(agent_ckpt['actor_target'])
            agent.critic_target.load_state_dict(agent_ckpt['critic_target'])
            agent.actor_optimizer.load_state_dict(agent_ckpt['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(agent_ckpt['critic_optimizer'])


if __name__ == "__main__":
    # Simple test
    print("Testing MADDPGCompatAgent...")
    
    manager_ids = ['manager_1', 'manager_2', 'manager_3', 'manager_4']
    agent = MADDPGCompatAgent(
        manager_ids=manager_ids,
        N_max=44,
        x_dim=6,
        g_dim=26,
        p=5,
        device="cpu"
    )
    
    # Test observation
    raw_obs = {
        'manager_1': {
            'g': np.random.randn(26),
            'device_ids': ['dev_1', 'dev_2'],
            'device_states': {
                'dev_1': np.random.randn(6),
                'dev_2': np.random.randn(6)
            }
        },
        'manager_2': {
            'g': np.random.randn(26),
            'device_ids': ['dev_3'],
            'device_states': {
                'dev_3': np.random.randn(6)
            }
        },
        'manager_3': {
            'g': np.random.randn(26),
            'device_ids': [],
            'device_states': {}
        },
        'manager_4': {
            'g': np.random.randn(26),
            'device_ids': ['dev_4', 'dev_5', 'dev_6'],
            'device_states': {
                'dev_4': np.random.randn(6),
                'dev_5': np.random.randn(6),
                'dev_6': np.random.randn(6)
            }
        }
    }
    
    # Select actions
    actions = agent.select_actions(raw_obs, explore=True)
    print(f"\nActions generated for {len(actions)} managers")
    for mid, act in actions.items():
        print(f"  {mid}: shape={act.shape}")
    
    print("\n>>> MADDPGCompatAgent test passed!")
