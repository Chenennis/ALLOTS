"""
SlotMapper: Maintains stable device-to-slot binding over time

This class ensures that devices keep their assigned slots while active,
and only allocate/free slots when devices join/leave (churn).

Author: FOenv Team
Date: 2026-01-13
"""

import numpy as np
from typing import Dict, List, Optional, Set


class SlotMapper:
    """
    Manages persistent device-to-slot mapping for one Manager.
    
    Key invariants:
    - If a device remains active across timesteps, it MUST keep the same slot
    - Only JOIN/LEAVE events modify the mapping
    - No heuristic reordering during runtime
    """
    
    def __init__(self, N_max: int, manager_id: str = "manager_0"):
        """
        Initialize SlotMapper.
        
        Args:
            N_max: Maximum number of device slots
            manager_id: ID of the manager this mapper serves (for logging)
        """
        self.N_max = N_max
        self.manager_id = manager_id
        
        # Mapping state
        self.slot_of_device: Dict[str, int] = {}  # device_id -> slot_index
        self.device_of_slot: List[Optional[str]] = [None] * N_max  # slot_index -> device_id
        self.free_slots: Set[int] = set(range(N_max))  # Available slots
        
        # Mask: 1.0 for active slots, 0.0 for inactive
        self.mask = np.zeros(N_max, dtype=np.float32)
        
    def update_mapping(self, active_device_ids: List[str]):
        """
        Update mapping based on current active devices.
        
        This implements JOIN/LEAVE logic:
        1. LEAVE: Devices not in active_device_ids but in slot_of_device → free their slots
        2. JOIN: Devices in active_device_ids but not in slot_of_device → allocate slots
        3. STAY: Devices in both → keep their slots unchanged
        
        Args:
            active_device_ids: List of currently active device IDs
            
        Raises:
            RuntimeError: If N_max is too small (no free slots for joining devices)
        """
        active_set = set(active_device_ids)
        
        # LEAVE: Remove devices that are no longer active
        for device_id, slot in list(self.slot_of_device.items()):
            if device_id not in active_set:
                # Device left - free its slot
                del self.slot_of_device[device_id]
                self.device_of_slot[slot] = None
                self.mask[slot] = 0.0
                self.free_slots.add(slot)
        
        # JOIN: Allocate slots for new devices
        for device_id in active_device_ids:
            if device_id not in self.slot_of_device:
                # New device joined - allocate a slot
                if not self.free_slots:
                    raise RuntimeError(
                        f"N_max={self.N_max} too small for {self.manager_id}. "
                        f"Cannot allocate slot for device {device_id}. "
                        f"Active devices: {len(active_device_ids)}, Free slots: 0"
                    )
                
                # Allocate smallest available slot (deterministic)
                slot = min(self.free_slots)
                self.free_slots.remove(slot)
                
                self.slot_of_device[device_id] = slot
                self.device_of_slot[slot] = device_id
                self.mask[slot] = 1.0
        
        # Verify invariants
        self._verify_invariants(active_device_ids)
    
    def _verify_invariants(self, active_device_ids: List[str]):
        """
        Verify mapping consistency (for debugging).
        
        Checks:
        1. mask.sum() == len(active_device_ids)
        2. Each active device has a valid slot
        3. No duplicate slots
        4. Bidirectional mapping consistency
        """
        active_count = len(active_device_ids)
        mask_sum = int(self.mask.sum())
        
        assert mask_sum == active_count, (
            f"Mask sum mismatch: mask.sum()={mask_sum}, "
            f"len(active_device_ids)={active_count}"
        )
        
        # Verify each active device has a slot
        for device_id in active_device_ids:
            assert device_id in self.slot_of_device, (
                f"Active device {device_id} not in slot_of_device"
            )
            
            slot = self.slot_of_device[device_id]
            assert 0 <= slot < self.N_max, (
                f"Invalid slot {slot} for device {device_id}"
            )
            
            # Verify bidirectional consistency
            assert self.device_of_slot[slot] == device_id, (
                f"Bidirectional mapping broken: "
                f"slot_of_device[{device_id}]={slot}, "
                f"but device_of_slot[{slot}]={self.device_of_slot[slot]}"
            )
            
            # Verify mask
            assert self.mask[slot] == 1.0, (
                f"Mask inconsistency: device {device_id} at slot {slot} has mask=0"
            )
        
        # Verify no duplicate slots
        assigned_slots = set(self.slot_of_device.values())
        assert len(assigned_slots) == active_count, (
            f"Duplicate slots detected: {len(assigned_slots)} unique slots "
            f"for {active_count} devices"
        )
    
    def get_slot(self, device_id: str) -> Optional[int]:
        """Get slot index for a device ID (None if not active)."""
        return self.slot_of_device.get(device_id)
    
    def get_device(self, slot: int) -> Optional[str]:
        """Get device ID at a slot (None if slot is free)."""
        if 0 <= slot < self.N_max:
            return self.device_of_slot[slot]
        return None
    
    def get_mask(self) -> np.ndarray:
        """Get current mask (copy to prevent modification)."""
        return self.mask.copy()
    
    def get_active_devices(self) -> List[str]:
        """Get list of currently active device IDs."""
        return list(self.slot_of_device.keys())
    
    def __repr__(self) -> str:
        active_count = len(self.slot_of_device)
        free_count = len(self.free_slots)
        return (
            f"SlotMapper({self.manager_id}, "
            f"active={active_count}/{self.N_max}, "
            f"free={free_count})"
        )
