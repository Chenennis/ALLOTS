"""
Compatibility Wrapper for MATD3 with Device Churn

MATD3 = Multi-Agent Twin Delayed Deep Deterministic Policy Gradient
Extends MADDPG with: Twin Critics, Delayed Policy Updates, Target Policy Smoothing

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
from algorithms.MATD3.fomatd3.fomatd3 import FOMATD3


class MATD3CompatAgent:
    """
    Compatibility wrapper for MATD3 with device churn.
    
    MATD3 improvements over MADDPG:
    - Twin critics (Q1, Q2) to reduce overestimation
    - Delayed policy updates (update actor less frequently)
    - Target policy smoothing (add noise to target actions)
    """
    
    def __init__(
        self,
        manager_ids: List[str],
        N_max: int,
        x_dim: int,
        g_dim: int,
        p: int,
        # MATD3 hyperparameters
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.1,
        noise_clip: float = 0.2,
        policy_delay: int = 2,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize MATD3 compatibility wrapper.
        
        Args:
            manager_ids: List of manager IDs
            N_max: Maximum device slots per manager
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            lr_actor: Actor learning rate (TD3: 1e-4)
            lr_critic: Critic learning rate (TD3: 1e-3)
            hidden_dim: Hidden layer dimension (TD3: 256)
            gamma: Discount factor (TD3: 0.99)
            tau: Soft update coefficient (TD3: 0.005)
            noise_scale: Exploration noise (TD3: 0.1)
            noise_clip: Target policy smoothing noise clip (TD3: 0.2)
            policy_delay: Delayed policy update frequency (TD3: 2)
            buffer_capacity: Replay buffer size
            batch_size: Batch size (TD3: 64, smaller than MADDPG)
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
        
        # MATD3 uses centralized state (all managers concatenated)
        centralized_state_dim = state_dim * self.n_managers
        
        # Create MATD3 agent
        self.matd3 = FOMATD3(
            n_agents=self.n_managers,
            state_dim=centralized_state_dim,  # Use centralized state dim
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=hidden_dim,
            max_action=1.0,
            gamma=gamma,
            tau=tau,
            noise_scale=noise_scale,
            noise_clip=noise_clip,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            policy_delay=policy_delay,
            device=device
        )
        
        print(f"MATD3CompatAgent initialized:")
        print(f"  Managers: {self.n_managers}")
        print(f"  State dim per manager: {state_dim} (g={g_dim}, X_pad={N_max}×{x_dim}, mask={N_max})")
        print(f"  Action dim per manager: {action_dim} (N_max={N_max} × p={p})")
        print(f"  Hyperparameters: lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}")
        print(f"  TD3 features: policy_delay={policy_delay}, noise_clip={noise_clip}")
    
    def select_actions(
        self,
        raw_obs: Dict[str, Dict],
        explore: bool = True
    ) -> Dict[str, np.ndarray]:
        """Select actions for all managers."""
        # Adapt observations to fixed format
        adapted_obs = self.compat_wrapper.adapt_obs_all(raw_obs, format='separate')
        
        # Concatenate all obs_vec to create centralized state
        centralized_state = np.concatenate([
            adapted_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)  # [centralized_state_dim]
        
        # Get masks for all managers
        masks = np.stack([
            adapted_obs[mid]['mask']
            for mid in self.manager_ids
        ], axis=0)  # [n_managers, N_max]
        
        # Select actions using MATD3 (pass centralized state)
        actions = self.matd3.select_actions(centralized_state, add_noise=explore)  # [n_managers, action_dim]
        
        # Reshape and apply masks
        padded_actions = {}
        for i, manager_id in enumerate(self.manager_ids):
            # Reshape to [N_max, p]
            A_pad = actions[i].reshape(self.N_max, self.p)
            
            # Apply mask
            mask = masks[i][:, np.newaxis]
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
        """Store transition in replay buffer."""
        # Adapt observations
        adapted_obs = self.compat_wrapper.adapt_obs_all(obs, format='separate')
        adapted_next_obs = self.compat_wrapper.adapt_obs_all(next_obs, format='separate')
        
        # Concatenate to centralized states
        states = np.concatenate([
            adapted_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)  # [centralized_state_dim]
        
        next_states = np.concatenate([
            adapted_next_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)  # [centralized_state_dim]
        
        # Flatten actions
        actions_array = np.stack([
            actions[mid].flatten()
            for mid in self.manager_ids
        ], axis=0)
        
        # Stack rewards and dones
        rewards_array = np.array([
            rewards.get(mid, 0.0)
            for mid in self.manager_ids
        ])
        
        dones_array = np.array([done] * self.n_managers)
        
        # Store in MATD3 replay buffer (uses FOReplayBuffer.add)
        # states and next_states are already centralized and flattened
        self.matd3.replay_buffer.add(
            state=states,
            actions=actions_array,
            rewards=rewards_array,
            next_state=next_states,
            dones=dones_array
        )
    
    def update(self) -> Dict[str, float]:
        """Update MATD3 policy."""
        return self.matd3.update()
    
    def get_buffer_size(self) -> int:
        """Get current replay buffer size."""
        return len(self.matd3.replay_buffer)
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'matd3': {
                'agents': [
                    {
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'target_actor': agent.target_actor.state_dict(),
                        'target_critic': agent.target_critic.state_dict(),
                    }
                    for agent in self.matd3.agents
                ]
            }
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, agent in enumerate(self.matd3.agents):
            agent_ckpt = checkpoint['matd3']['agents'][i]
            agent.actor.load_state_dict(agent_ckpt['actor'])
            agent.critic.load_state_dict(agent_ckpt['critic'])
            agent.target_actor.load_state_dict(agent_ckpt['target_actor'])
            agent.target_critic.load_state_dict(agent_ckpt['target_critic'])


if __name__ == "__main__":
    # Simple test
    print("Testing MATD3CompatAgent...")
    
    manager_ids = ['manager_1', 'manager_2', 'manager_3', 'manager_4']
    agent = MATD3CompatAgent(
        manager_ids=manager_ids,
        N_max=44,
        x_dim=6,
        g_dim=26,
        p=5,
        device="cpu"
    )
    
    # Test observation
    raw_obs = {
        f'manager_{i+1}': {
            'g': np.random.randn(26),
            'device_ids': [f'dev_{i}_{j}' for j in range(np.random.randint(1, 5))],
            'device_states': {
                f'dev_{i}_{j}': np.random.randn(6)
                for j in range(np.random.randint(1, 5))
            }
        }
        for i in range(4)
    }
    
    # Select actions
    actions = agent.select_actions(raw_obs, explore=True)
    print(f"\nActions generated for {len(actions)} managers")
    
    print("\n>>> MATD3CompatAgent test passed!")
