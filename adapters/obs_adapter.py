"""
ObsAdapter: Converts variable-length observations to fixed-length padded tensors

Author: FOenv Team
Date: 2026-01-13
"""

import numpy as np
from typing import Dict, List, Tuple

from .slot_mapper import SlotMapper


class ObsAdapter:
    """
    Adapts variable-length device observations to fixed-length padded format.
    
    Workflow:
    1. Update slot mapping with current active devices
    2. Create zero-padded device state matrix X_pad [N_max, x_dim]
    3. Fill active slots with actual device states
    4. Return (X_pad, mask) where mask indicates active slots
    """
    
    def __init__(self, slot_mapper: SlotMapper, N_max: int, x_dim: int):
        """
        Initialize ObsAdapter.
        
        Args:
            slot_mapper: SlotMapper instance for this manager
            N_max: Maximum number of device slots
            x_dim: Dimension of device state features
        """
        self.slot_mapper = slot_mapper
        self.N_max = N_max
        self.x_dim = x_dim
    
    def to_padded(
        self,
        device_ids: List[str],
        device_states: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert variable-length device observations to padded format.
        
        Args:
            device_ids: List of active device IDs
            device_states: Dict mapping device_id to state vector [x_dim]
            
        Returns:
            X_pad: Padded device states [N_max, x_dim]
            mask: Binary mask [N_max], 1.0 for active slots
            
        Raises:
            ValueError: If device state dimensions don't match x_dim
        """
        # Update mapping
        self.slot_mapper.update_mapping(device_ids)
        
        # Create zero-padded matrix
        X_pad = np.zeros((self.N_max, self.x_dim), dtype=np.float32)
        
        # Fill active slots
        for device_id in device_ids:
            slot = self.slot_mapper.get_slot(device_id)
            assert slot is not None, f"Device {device_id} should have a slot after update_mapping"
            
            # Get device state
            state = device_states.get(device_id)
            if state is None:
                raise ValueError(f"Missing state for device {device_id}")
            
            # Verify dimension
            state = np.asarray(state, dtype=np.float32)
            if state.shape[0] != self.x_dim:
                raise ValueError(
                    f"Device {device_id} state dimension mismatch: "
                    f"expected {self.x_dim}, got {state.shape[0]}"
                )
            
            # Fill slot
            X_pad[slot] = state
        
        # Get mask
        mask = self.slot_mapper.get_mask()
        
        return X_pad, mask
    
    def to_padded_from_list(
        self,
        device_ids: List[str],
        device_states_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method: convert from list of states (aligned with device_ids).
        
        Args:
            device_ids: List of device IDs [n_devices]
            device_states_list: List of state vectors [n_devices x x_dim]
            
        Returns:
            X_pad: Padded device states [N_max, x_dim]
            mask: Binary mask [N_max]
        """
        # Convert list to dict
        device_states = {
            dev_id: state
            for dev_id, state in zip(device_ids, device_states_list)
        }
        
        return self.to_padded(device_ids, device_states)
