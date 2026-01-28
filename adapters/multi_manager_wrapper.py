"""
MultiManagerCompatWrapper: Unified compatibility layer for all managers

This wrapper coordinates multiple managers, each with dynamic device sets,
and provides fixed-dimension interface to baseline algorithms.

UPDATED 2026-01-18: Now uses SimpleSlotMapper by default for baseline algorithms.
This removes the stable slot mapping advantage, making EA's native design more valuable.

Author: FOenv Team
Date: 2026-01-13 (Updated: 2026-01-18)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .slot_mapper import SlotMapper
from .simple_slot_mapper import SimpleSlotMapper
from .obs_adapter import ObsAdapter
from .act_adapter import ActAdapter

logger = logging.getLogger(__name__)


class MultiManagerCompatWrapper:
    """
    Compatibility wrapper for multi-manager environments with device churn.
    
    Manages SlotMapper, ObsAdapter, ActAdapter for each manager,
    and provides unified interface for baseline algorithms.
    
    IMPORTANT: By default uses SimpleSlotMapper which does NOT maintain
    stable device-to-slot binding. This provides a fair comparison against
    EA which natively supports stable mapping.
    
    Set use_stable_mapping=True to use the original stable SlotMapper.
    """
    
    def __init__(
        self,
        manager_ids: List[str],
        N_max: int,
        x_dim: int,
        g_dim: int,
        p: int,
        verbose: bool = False,
        use_stable_mapping: bool = False  # NEW: Default to simple (unstable) mapping
    ):
        """
        Initialize MultiManagerCompatWrapper.
        
        Args:
            manager_ids: List of manager IDs
            N_max: Maximum number of device slots per manager
            x_dim: Dimension of device state features
            g_dim: Dimension of global features per manager
            p: Action dimension per device
            verbose: If True, log churn events and mapping updates
            use_stable_mapping: If True, use stable SlotMapper (like EA).
                               If False (default), use SimpleSlotMapper.
        """
        self.manager_ids = manager_ids
        self.n_managers = len(manager_ids)
        self.N_max = N_max
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.verbose = verbose
        self.use_stable_mapping = use_stable_mapping
        
        # Create adapters for each manager
        self.slot_mappers: Dict[str, SlotMapper] = {}
        self.obs_adapters: Dict[str, ObsAdapter] = {}
        self.act_adapters: Dict[str, ActAdapter] = {}
        
        # Choose mapper type based on use_stable_mapping
        MapperClass = SlotMapper if use_stable_mapping else SimpleSlotMapper
        mapper_type = "stable" if use_stable_mapping else "simple"
        
        for manager_id in manager_ids:
            self.slot_mappers[manager_id] = MapperClass(N_max, manager_id)
            self.obs_adapters[manager_id] = ObsAdapter(
                self.slot_mappers[manager_id], N_max, x_dim
            )
            self.act_adapters[manager_id] = ActAdapter(
                self.slot_mappers[manager_id], N_max, p
            )
        
        if verbose:
            logger.info(f"MultiManagerCompatWrapper initialized: "
                       f"{self.n_managers} managers, N_max={N_max}, "
                       f"x_dim={x_dim}, g_dim={g_dim}, p={p}, "
                       f"mapper_type={mapper_type}")
    
    def adapt_obs_all(
        self,
        raw_obs: Dict[str, Dict],
        format: str = "separate"
    ) -> Dict[str, np.ndarray]:
        """
        Adapt observations for all managers.
        
        Args:
            raw_obs: Dict mapping manager_id to observation dict containing:
                     - 'g': global features [g_dim]
                     - 'device_ids': List of active device IDs
                     - 'device_states': Dict or List of device states
            format: Output format:
                    - 'separate': Return dict with separate obs per manager
                    - 'concat': Return concatenated obs_all
                    
        Returns:
            If format='separate':
                Dict mapping manager_id to obs_fixed containing:
                - 'g': global features [g_dim]
                - 'X_pad': padded device states [N_max, x_dim]
                - 'mask': device mask [N_max]
                - 'obs_vec': flattened obs [g_dim + N_max*x_dim + N_max]
            
            If format='concat':
                Single dict with:
                - 'obs_all': concatenated obs [n_managers * (g_dim + N_max*x_dim + N_max)]
                - 'separate': Same as format='separate' for individual access
        """
        adapted_obs = {}
        
        for manager_id in self.manager_ids:
            if manager_id not in raw_obs:
                raise ValueError(f"Missing observation for {manager_id}")
            
            obs = raw_obs[manager_id]
            
            # Extract components
            g = np.asarray(obs['g'], dtype=np.float32)  # [g_dim]
            device_ids = obs['device_ids']
            
            # Handle device_states as dict or list
            if isinstance(obs['device_states'], dict):
                device_states = obs['device_states']
            else:
                # Convert list to dict
                device_states = {
                    dev_id: state
                    for dev_id, state in zip(device_ids, obs['device_states'])
                }
            
            # Adapt to padded format
            X_pad, mask = self.obs_adapters[manager_id].to_padded(
                device_ids, device_states
            )
            
            # Create flattened obs vector
            obs_vec = np.concatenate([
                g,                          # [g_dim]
                X_pad.flatten(),            # [N_max * x_dim]
                mask                        # [N_max]
            ], axis=0)
            
            adapted_obs[manager_id] = {
                'g': g,
                'X_pad': X_pad,
                'mask': mask,
                'obs_vec': obs_vec,
                'n_devices': int(mask.sum())
            }
        
        if format == 'concat':
            # Concatenate all obs_vec
            obs_all = np.concatenate([
                adapted_obs[mid]['obs_vec']
                for mid in self.manager_ids
            ], axis=0)
            
            return {
                'obs_all': obs_all,
                'separate': adapted_obs
            }
        else:
            return adapted_obs
    
    def adapt_actions_all(
        self,
        padded_actions: Dict[str, np.ndarray],
        apply_mask: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert padded actions to environment-compatible action_sets.
        
        Args:
            padded_actions: Dict mapping manager_id to A_pad [N_max, p]
            apply_mask: If True, zero out inactive slot actions
            
        Returns:
            env_actions: Dict mapping manager_id to action_set
                        action_set: Dict mapping device_id to action [p]
        """
        env_actions = {}
        
        for manager_id in self.manager_ids:
            if manager_id not in padded_actions:
                raise ValueError(f"Missing actions for {manager_id}")
            
            A_pad = padded_actions[manager_id]
            
            # Convert to aligned action_set
            action_set = self.act_adapters[manager_id].to_aligned_action_set(
                A_pad, apply_mask=apply_mask
            )
            
            env_actions[manager_id] = action_set
        
        return env_actions
    
    def get_state_action_dims(self) -> Tuple[int, int]:
        """
        Get fixed dimensions for baseline algorithms.
        
        Returns:
            state_dim: Total state dimension per manager
            action_dim: Total action dimension per manager
        """
        # Per-manager dimensions
        state_dim = self.g_dim + self.N_max * self.x_dim + self.N_max  # g + X_pad + mask
        action_dim = self.N_max * self.p
        
        return state_dim, action_dim
    
    def get_centralized_dims(self) -> Tuple[int, int]:
        """
        Get centralized dimensions (for CTDE algorithms).
        
        Returns:
            centralized_state_dim: Sum of all managers' state_dim
            centralized_action_dim: Sum of all managers' action_dim
        """
        state_dim, action_dim = self.get_state_action_dims()
        
        return state_dim * self.n_managers, action_dim * self.n_managers
    
    def log_churn_event(self, episode: int, manager_id: str, severity: float,
                       devices_joined: int, devices_left: int):
        """Log churn event (if verbose=True)."""
        if self.verbose:
            mapper = self.slot_mappers[manager_id]
            logger.info(
                f"[Ep {episode}] CHURN: {manager_id}, "
                f"severity={severity:.2%}, "
                f"joined={devices_joined}, left={devices_left}, "
                f"active={len(mapper.get_active_devices())}/{self.N_max}"
            )
    
    def verify_no_overflow(self):
        """
        Verify no manager exceeded N_max.
        
        Raises:
            RuntimeError: If any manager has too many devices
        """
        for manager_id, mapper in self.slot_mappers.items():
            active_count = len(mapper.get_active_devices())
            free_count = len(mapper.free_slots)
            
            if active_count > self.N_max:
                raise RuntimeError(
                    f"Manager {manager_id} overflow: "
                    f"{active_count} devices > N_max={self.N_max}"
                )
            
            if active_count + free_count != self.N_max:
                raise RuntimeError(
                    f"Manager {manager_id} slot accounting error: "
                    f"active={active_count}, free={free_count}, "
                    f"sum={active_count + free_count} != N_max={self.N_max}"
                )
    
    def get_masks_all(self) -> Dict[str, np.ndarray]:
        """Get masks for all managers."""
        return {
            manager_id: mapper.get_mask()
            for manager_id, mapper in self.slot_mappers.items()
        }
    
    def __repr__(self) -> str:
        state_dim, action_dim = self.get_state_action_dims()
        return (
            f"MultiManagerCompatWrapper("
            f"managers={self.n_managers}, "
            f"N_max={self.N_max}, "
            f"state_dim={state_dim}, "
            f"action_dim={action_dim})"
        )
