"""
Compatibility Wrapper for AGILE with Device Churn

This wrapper enables FOAGILE to work with FOgym's dynamic device sets
by using the compatibility layer (SlotMapper, ObsAdapter, ActAdapter).

Author: FOenv Team
Date: 2026-01-24
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
from algorithms.AGILE.foagile.foagile import FOAGILE


class AGILECompatAgent:
    """
    Compatibility wrapper for AGILE with device churn.
    
    This class wraps FOAGILE and handles:
    - Variable-length observations → Fixed padded format with graph structure
    - Graph attention for learning device relationships
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
        # AGILE hyperparameters
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        hidden_dim: int = 64,
        num_heads: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.1,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize AGILE compatibility wrapper.
        
        Args:
            manager_ids: List of manager IDs
            N_max: Maximum device slots per manager
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads in GAT
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
        
        # Create FOAGILE agent
        self.agile = FOAGILE(
            n_agents=self.n_managers,
            N_max=N_max,
            device_dim=x_dim,
            global_dim=g_dim,
            action_dim=p,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            noise_scale=noise_scale,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            device=device
        )
        
        print(f"AGILECompatAgent initialized:")
        print(f"  Managers: {self.n_managers}")
        print(f"  N_max: {N_max}, x_dim: {x_dim}, g_dim: {g_dim}, p: {p}")
        print(f"  GAT heads: {num_heads}, hidden_dim: {hidden_dim}")
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
        
        # Prepare inputs for AGILE
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
        
        # Select actions using AGILE
        actions = self.agile.select_actions(
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
        
        # Store for each manager
        for i, mid in enumerate(self.manager_ids):
            obs_data = adapted_obs[mid]
            next_obs_data = adapted_next_obs[mid]
            
            # Reshape X_pad
            device_states = obs_data['X_pad'].reshape(self.N_max, self.x_dim)
            next_device_states = next_obs_data['X_pad'].reshape(self.N_max, self.x_dim)
            
            self.agile.store_experience(
                agent_idx=i,
                device_states=device_states,
                global_feat=obs_data['g'],
                mask=obs_data['mask'],
                actions=actions[mid],
                reward=rewards.get(mid, 0.0),
                next_device_states=next_device_states,
                next_global_feat=next_obs_data['g'],
                next_mask=next_obs_data['mask'],
                done=done
            )
    
    def update(self) -> Dict[str, float]:
        """
        Update AGILE policy.
        
        Returns:
            metrics: Training metrics (actor_loss, critic_loss, etc.)
        """
        return self.agile.update()
    
    def get_buffer_size(self) -> int:
        """Get minimum replay buffer size across all agents."""
        return min(len(buf) for buf in self.agile.replay_buffers)
    
    def save(self, path: str):
        """Save model."""
        self.agile.save(path)
    
    def load(self, path: str):
        """Load model."""
        self.agile.load(path)


if __name__ == "__main__":
    # Simple test
    print("Testing AGILECompatAgent...")
    
    manager_ids = ['manager_1', 'manager_2', 'manager_3', 'manager_4']
    agent = AGILECompatAgent(
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
    
    print("\n>>> AGILECompatAgent test passed!")
