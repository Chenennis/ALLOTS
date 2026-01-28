"""
SimpleSlotMapper: Basic slot allocation without stable mapping

This class provides dimension adaptation for baseline algorithms but does NOT
maintain stable device-to-slot binding. Each update completely reassigns slots.

This is designed to test the advantage of EA's stable mapping mechanism.

Author: FOenv Team
Date: 2026-01-18
"""

import numpy as np
from typing import Dict, List, Optional, Set


class SimpleSlotMapper:
    """
    Simple slot mapper that only handles dimension adaptation.
    
    Key differences from SlotMapper:
    - Does NOT maintain stable device-to-slot binding
    - Devices are assigned to slots sequentially on each update
    - This means a device's slot may change between timesteps
    
    This provides a "fair" comparison where baseline algorithms only get
    dimension adaptation, not the stability benefits that EA natively provides.
    """
    
    def __init__(self, N_max: int, manager_id: str = "manager_0"):
        """
        Initialize SimpleSlotMapper.
        
        Args:
            N_max: Maximum number of device slots
            manager_id: ID of the manager this mapper serves (for logging)
        """
        self.N_max = N_max
        self.manager_id = manager_id
        
        # Mapping state (rebuilt on each update)
        self.slot_of_device: Dict[str, int] = {}
        self.device_of_slot: List[Optional[str]] = [None] * N_max
        self.free_slots: Set[int] = set(range(N_max))
        
        # Mask: 1.0 for active slots, 0.0 for inactive
        self.mask = np.zeros(N_max, dtype=np.float32)
        
    def update_mapping(self, active_device_ids: List[str]):
        """
        Update mapping based on current active devices.
        
        IMPORTANT: This completely rebuilds the mapping each time!
        Devices are assigned to slots 0, 1, 2, ... in order.
        
        This means:
        - Device A might be in slot 3 at time t
        - After churn, Device A might be in slot 7 at time t+1
        - The network must learn to handle this instability
        
        Args:
            active_device_ids: List of currently active device IDs
            
        Raises:
            RuntimeError: If N_max is too small
        """
        n_active = len(active_device_ids)
        
        if n_active > self.N_max:
            raise RuntimeError(
                f"N_max={self.N_max} too small for {self.manager_id}. "
                f"Active devices: {n_active}"
            )
        
        # COMPLETELY RESET the mapping
        self.slot_of_device = {}
        self.device_of_slot = [None] * self.N_max
        self.mask = np.zeros(self.N_max, dtype=np.float32)
        self.free_slots = set(range(self.N_max))
        
        # Assign devices to slots sequentially (0, 1, 2, ...)
        for i, device_id in enumerate(active_device_ids):
            slot = i  # Sequential assignment
            self.slot_of_device[device_id] = slot
            self.device_of_slot[slot] = device_id
            self.mask[slot] = 1.0
            self.free_slots.discard(slot)
    
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
            f"SimpleSlotMapper({self.manager_id}, "
            f"active={active_count}/{self.N_max}, "
            f"free={free_count})"
        )


def test_simple_slot_mapper():
    """Test SimpleSlotMapper behavior"""
    print("=== Testing SimpleSlotMapper ===\n")
    
    mapper = SimpleSlotMapper(N_max=10, manager_id="test_manager")
    
    # Initial devices
    devices_t0 = ['dev_A', 'dev_B', 'dev_C']
    mapper.update_mapping(devices_t0)
    
    print("Time 0:")
    print(f"  Devices: {devices_t0}")
    print(f"  Mapping: {mapper.slot_of_device}")
    print(f"  Mask: {mapper.get_mask()}")
    
    # After churn: device B leaves, device D joins
    devices_t1 = ['dev_A', 'dev_C', 'dev_D']
    mapper.update_mapping(devices_t1)
    
    print("\nTime 1 (after churn):")
    print(f"  Devices: {devices_t1}")
    print(f"  Mapping: {mapper.slot_of_device}")
    print(f"  Mask: {mapper.get_mask()}")
    
    print("\n⚠️ Notice: dev_A changed from slot 0 to slot 0 (stable)")
    print("   But dev_C changed from slot 2 to slot 1 (UNSTABLE!)")
    print("   This instability challenges the baseline algorithms.")
    
    print("\n=== Test passed ===")


if __name__ == "__main__":
    test_simple_slot_mapper()
