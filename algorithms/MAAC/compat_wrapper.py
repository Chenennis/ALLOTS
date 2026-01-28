"""
Compatibility Wrapper for MAAC with Device Churn

This wrapper enables FOMAAC to work with FOgym's dynamic device sets
by using the compatibility layer (SlotMapper, ObsAdapter, ActAdapter).

Author: FOenv Team
Date: 2026-01-25
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
from algorithms.MAAC.fomaac.fomaac import FOMAAC


class MAACCompatAgent:
    """
    Compatibility wrapper for MAAC with device churn.
    
    This class wraps FOMAAC and handles:
    - Variable-length observations → Fixed padded format
    - Multi-head attention over agents' state-action pairs
    - Fixed padded actions → Variable-length action_sets
    - Mask-aware training
    """
    
    def __init__(
        self,
        manager_ids: List[str],
        N_max: int,
        x_dim: int,
        g_dim: int,
        p: int,
        # MAAC hyperparameters
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        hidden_dim: int = 128,
        attend_heads: int = 4,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.1,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize MAAC compatibility wrapper.
        
        Args:
            manager_ids: List of manager IDs
            N_max: Maximum device slots per manager
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: Hidden layer dimension
            attend_heads: Number of attention heads
            gamma: Discount factor
            tau: Soft update coefficient
            noise_scale: Exploration noise
            buffer_capacity: Replay buffer size
            batch_size: Batch size
            device: Computing device
        """
        self.manager_ids = manager_ids
        self.n_managers = len(manager_ids)
        self.N_max = N_max
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.device = device
        
        # Create compatibility wrapper with stable mapping
        self.compat_wrapper = MultiManagerCompatWrapper(
            use_stable_mapping=True,  # Use stable slot mapping
            manager_ids=manager_ids,
            N_max=N_max,
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            verbose=False
        )
        
        # Create FOMAAC agent
        self.maac = FOMAAC(
            n_agents=self.n_managers,
            N_max=N_max,
            device_dim=x_dim,
            global_dim=g_dim,
            action_dim=p,
            hidden_dim=hidden_dim,
            attend_heads=attend_heads,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            noise_scale=noise_scale,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            device=device
        )
        
        print(f"MAACCompatAgent initialized:")
        print(f"  Managers: {self.n_managers}")
        print(f"  N_max: {N_max}, x_dim: {x_dim}, g_dim: {g_dim}, p: {p}")
        print(f"  Attention heads: {attend_heads}, hidden_dim: {hidden_dim}")
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
        
        # Prepare inputs for MAAC
        device_states = []
        global_feats = []
        masks = []
        
        for mid in self.manager_ids:
            obs = adapted_obs[mid]
            # Reshape X_pad from flat to [N_max, x_dim]
            X_flat = obs['X_pad']
            X_reshaped = X_flat.reshape(self.N_max, self.x_dim)
            
            device_states.append(X_reshaped)
            global_feats.append(obs['g'])
            masks.append(obs['mask'])
        
        device_states = np.array(device_states)  # [n_managers, N_max, x_dim]
        global_feats = np.array(global_feats)    # [n_managers, g_dim]
        masks = np.array(masks)                  # [n_managers, N_max]
        
        # Select actions using MAAC
        actions = self.maac.select_actions(
            device_states, global_feats, masks, add_noise=explore
        )  # [n_managers, N_max, p]
        
        # Convert to dict format
        padded_actions = {}
        for i, manager_id in enumerate(self.manager_ids):
            padded_actions[manager_id] = actions[i]  # [N_max, p]
        
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
        
        # Prepare data
        device_states = []
        global_feats = []
        masks = []
        action_list = []
        reward_list = []
        next_device_states = []
        next_global_feats = []
        next_masks = []
        
        for mid in self.manager_ids:
            obs_data = adapted_obs[mid]
            next_obs_data = adapted_next_obs[mid]
            
            # Reshape X_pad
            device_states.append(obs_data['X_pad'].reshape(self.N_max, self.x_dim))
            global_feats.append(obs_data['g'])
            masks.append(obs_data['mask'])
            action_list.append(actions[mid])
            reward_list.append(rewards.get(mid, 0.0))
            
            next_device_states.append(next_obs_data['X_pad'].reshape(self.N_max, self.x_dim))
            next_global_feats.append(next_obs_data['g'])
            next_masks.append(next_obs_data['mask'])
        
        self.maac.store_experience(
            device_states=np.array(device_states),
            global_feats=np.array(global_feats),
            masks=np.array(masks),
            actions=np.array(action_list),
            rewards=reward_list,
            next_device_states=np.array(next_device_states),
            next_global_feats=np.array(next_global_feats),
            next_masks=np.array(next_masks),
            done=done
        )
    
    def update(self) -> Dict[str, float]:
        """
        Update MAAC policy.
        
        Returns:
            metrics: Training metrics (actor_loss, critic_loss)
        """
        return self.maac.update()
    
    def get_buffer_size(self) -> int:
        """Get replay buffer size."""
        return len(self.maac.replay_buffer)
    
    def save(self, path: str):
        """Save model."""
        self.maac.save(path)
    
    def load(self, path: str):
        """Load model."""
        self.maac.load(path)


if __name__ == "__main__":
    # Simple test
    print("Testing MAACCompatAgent...")
    
    manager_ids = ['manager_1', 'manager_2', 'manager_3', 'manager_4']
    agent = MAACCompatAgent(
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
    
    print("\n>>> MAACCompatAgent test passed!")
