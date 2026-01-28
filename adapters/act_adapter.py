"""
ActAdapter: Converts fixed-length padded actions to aligned action_set

Author: FOenv Team
Date: 2026-01-13
"""

import numpy as np
from typing import Dict, List

from .slot_mapper import SlotMapper


class ActAdapter:
    """
    Adapts fixed-length padded actions to variable-length aligned action_set.
    
    Workflow:
    1. Apply mask to ensure inactive slots have zero actions
    2. Convert padded actions to device-aligned action_set
    3. Verify no gradient leakage from padding slots
    """
    
    def __init__(self, slot_mapper: SlotMapper, N_max: int, p: int):
        """
        Initialize ActAdapter.
        
        Args:
            slot_mapper: SlotMapper instance for this manager
            N_max: Maximum number of device slots
            p: Action dimension per device (e.g., 5 for FlexOffer)
        """
        self.slot_mapper = slot_mapper
        self.N_max = N_max
        self.p = p
    
    def to_aligned_action_set(
        self,
        A_pad: np.ndarray,
        apply_mask: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Convert padded actions to aligned action_set.
        
        Args:
            A_pad: Padded action matrix [N_max, p] or flattened [N_max*p]
            apply_mask: If True, zero out inactive slot actions (recommended)
            
        Returns:
            action_set: Dict mapping device_id to action vector [p]
            
        Raises:
            ValueError: If action dimensions don't match
        """
        # Handle flattened actions
        if A_pad.ndim == 1:
            if A_pad.shape[0] != self.N_max * self.p:
                raise ValueError(
                    f"Flattened action dimension mismatch: "
                    f"expected {self.N_max * self.p}, got {A_pad.shape[0]}"
                )
            A_pad = A_pad.reshape(self.N_max, self.p)
        
        # Verify shape
        if A_pad.shape != (self.N_max, self.p):
            raise ValueError(
                f"Action shape mismatch: expected ({self.N_max}, {self.p}), "
                f"got {A_pad.shape}"
            )
        
        # Apply mask to zero out inactive slots
        if apply_mask:
            mask = self.slot_mapper.get_mask()  # [N_max]
            mask_expanded = mask[:, np.newaxis]  # [N_max, 1]
            A_pad = A_pad * mask_expanded  # Broadcast multiplication
        
        # Convert to action_set
        action_set = {}
        for slot in range(self.N_max):
            device_id = self.slot_mapper.get_device(slot)
            if device_id is not None:
                action_set[device_id] = A_pad[slot].copy()
        
        # Verify all active devices have actions
        active_devices = self.slot_mapper.get_active_devices()
        for device_id in active_devices:
            assert device_id in action_set, (
                f"Active device {device_id} missing in action_set"
            )
        
        return action_set
    
    def mask_actions_inplace(self, A_pad: np.ndarray) -> np.ndarray:
        """
        Apply mask to actions in-place (for use before env.step).
        
        Args:
            A_pad: Padded action matrix [N_max, p] or [batch, N_max, p]
            
        Returns:
            A_pad: Same array with inactive slots zeroed (modified in-place)
        """
        mask = self.slot_mapper.get_mask()  # [N_max]
        
        if A_pad.ndim == 2:
            # [N_max, p]
            mask_expanded = mask[:, np.newaxis]
            A_pad *= mask_expanded
        elif A_pad.ndim == 3:
            # [batch, N_max, p]
            mask_expanded = mask[np.newaxis, :, np.newaxis]
            A_pad *= mask_expanded
        else:
            raise ValueError(f"Unsupported action shape: {A_pad.shape}")
        
        return A_pad
