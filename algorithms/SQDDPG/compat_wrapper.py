"""
Compatibility Wrapper for SQDDPG with Device Churn

SQDDPG = Shapley Q-value Deep Deterministic Policy Gradient
Extends MADDPG with Shapley-based credit assignment for multi-agent cooperation.

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
from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG


class SQDDPGCompatAgent:
    """
    Compatibility wrapper for SQDDPG with device churn.
    
    SQDDPG uses Shapley values to credit assignment among agents (managers),
    promoting cooperation and fair reward distribution.
    """
    
    def __init__(
        self,
        manager_ids: List[str],
        N_max: int,
        x_dim: int,
        g_dim: int,
        p: int,
        # SQDDPG hyperparameters (from paper)
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
        Initialize SQDDPG compatibility wrapper.
        
        Args:
            manager_ids: List of manager IDs
            N_max: Maximum device slots per manager
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            lr_actor: Actor learning rate (SQDDPG: 1e-2)
            lr_critic: Critic learning rate (SQDDPG: 1e-2)
            hidden_dim: Hidden layer dimension (SQDDPG: 64)
            gamma: Discount factor (SQDDPG: 0.95)
            tau: Soft update coefficient (SQDDPG: 0.01)
            noise_scale: Exploration noise (SQDDPG: 0.1)
            buffer_capacity: Replay buffer size
            batch_size: Batch size (SQDDPG: 1024)
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
        
        # Create SQDDPG agent
        self.sqddpg = FOSQDDPG(
            n_agents=self.n_managers,
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden_dim=hidden_dim,
            gamma=gamma,
            tau=tau,
            noise_scale=noise_scale,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            sample_size=5,  # Shapley value sampling
            device=device
        )
        
        print(f"SQDDPGCompatAgent initialized:")
        print(f"  Managers: {self.n_managers}")
        print(f"  State dim per manager: {state_dim} (g={g_dim}, X_pad={N_max}×{x_dim}, mask={N_max})")
        print(f"  Action dim per manager: {action_dim} (N_max={N_max} × p={p})")
        print(f"  Hyperparameters: lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}")
        print(f"  SQDDPG features: Shapley value credit assignment")
    
    def select_actions(
        self,
        raw_obs: Dict[str, Dict],
        explore: bool = True
    ) -> Dict[str, np.ndarray]:
        """Select actions for all managers."""
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
        
        # Select actions using SQDDPG
        actions = self.sqddpg.select_actions(states, add_noise=explore)  # [n_managers, action_dim]
        
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
        
        # Stack states
        states = np.stack([
            adapted_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)
        
        next_states = np.stack([
            adapted_next_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ], axis=0)
        
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
        
        dones_array = np.array([done] * self.n_managers, dtype=bool)
        
        # Store in SQDDPG replay buffer
        self.sqddpg.store_experience(
            states=states,
            actions=actions_array,
            rewards=rewards_array,
            next_states=next_states,
            dones=dones_array
        )
    
    def update(self) -> Dict[str, float]:
        """Update SQDDPG policy."""
        result = self.sqddpg.update()
        if result is None:
            # Buffer not large enough yet
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'status': 'warmup'}
        return result
    
    def get_buffer_size(self) -> int:
        """Get current replay buffer size."""
        return len(self.sqddpg.replay_buffer)
    
    def save(self, path: str):
        """Save model."""
        self.sqddpg.save_models(path)
    
    def load(self, path: str):
        """Load model."""
        self.sqddpg.load_models(path)


if __name__ == "__main__":
    # Simple test
    print("Testing SQDDPGCompatAgent...")
    
    manager_ids = ['manager_1', 'manager_2', 'manager_3', 'manager_4']
    agent = SQDDPGCompatAgent(
        manager_ids=manager_ids,
        N_max=44,
        x_dim=6,
        g_dim=26,
        p=5,
        device="cpu"
    )
    
    # Test observation
    raw_obs = {}
    for i in range(4):
        n_devices = np.random.randint(1, 5)
        device_ids = [f'dev_{i}_{j}' for j in range(n_devices)]
        raw_obs[f'manager_{i+1}'] = {
            'g': np.random.randn(26),
            'device_ids': device_ids,
            'device_states': {
                dev_id: np.random.randn(6)
                for dev_id in device_ids
            }
        }
    
    # Select actions
    actions = agent.select_actions(raw_obs, explore=True)
    print(f"\nActions generated for {len(actions)} managers")
    
    print("\n>>> SQDDPGCompatAgent test passed!")
