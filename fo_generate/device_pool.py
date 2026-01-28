"""
Device Pool Management Module for FOgym Churn Environment

This module manages the Universe, Active, and Inactive device sets for churn support.

Key Concepts:
- Universe: All devices ever created for this manager (Active ∪ Inactive)
- Active: Currently participating devices
- Inactive: Temporarily offline devices (can be reactivated)

Author: FOenv Team
Date: 2026-01-12
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

from fo_generate.unified_mdp_env import DeviceMDPInterface, DeviceType

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """
    Metadata for a device in the pool
    
    Attributes:
        device_id: Unique device identifier
        device_type: Type of device (Battery, EV, etc.)
        manager_id: Manager controlling this device
        user_id: User owning this device
        device_config: Original configuration dict
        is_active: Whether device is currently active
        join_count: Number of times device joined (0 = original)
        leave_count: Number of times device left
        total_timesteps_active: Total timesteps the device was active
    """
    device_id: str
    device_type: str
    manager_id: str
    user_id: str
    device_config: Dict[str, Any]
    is_active: bool = True
    join_count: int = 0
    leave_count: int = 0
    total_timesteps_active: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'manager_id': self.manager_id,
            'user_id': self.user_id,
            'device_config': self.device_config,
            'is_active': self.is_active,
            'join_count': self.join_count,
            'leave_count': self.leave_count,
            'total_timesteps_active': self.total_timesteps_active,
        }


class DevicePool:
    """
    Manages Universe, Active, and Inactive device sets for a Manager
    
    This class provides:
    - Add/remove devices from Active set
    - Track device statistics
    - Ensure consistency between Universe, Active, Inactive
    - Support for device querying and filtering
    """
    
    def __init__(self, manager_id: str):
        """
        Initialize device pool
        
        Args:
            manager_id: Manager identifier
        """
        self.manager_id = manager_id
        
        # Universe: All devices ever created (device_id → DeviceInfo)
        self.universe: Dict[str, DeviceInfo] = {}
        
        # Active set: Currently active device IDs
        self.active_ids: Set[str] = set()
        
        # Inactive set: Temporarily offline device IDs
        self.inactive_ids: Set[str] = set()
        
        # Statistics
        self.total_churn_events = 0
        self.total_devices_created = 0
        
        logger.info(f"DevicePool initialized for manager {manager_id}")
    
    def add_device(self, device_info: DeviceInfo, is_active: bool = True) -> None:
        """
        Add a device to the Universe
        
        Args:
            device_info: Device metadata
            is_active: Whether device starts as active
        """
        device_id = device_info.device_id
        
        if device_id in self.universe:
            logger.warning(f"Device {device_id} already exists in Universe, skipping")
            return
        
        # Add to universe
        device_info.is_active = is_active
        self.universe[device_id] = device_info
        
        # Add to appropriate set
        if is_active:
            self.active_ids.add(device_id)
        else:
            self.inactive_ids.add(device_id)
        
        self.total_devices_created += 1
        
        logger.debug(f"Added device {device_id} (type={device_info.device_type}, "
                    f"active={is_active}) to pool")
    
    def activate_device(self, device_id: str) -> bool:
        """
        Move device from Inactive to Active
        
        Args:
            device_id: Device identifier
        
        Returns:
            True if successful, False otherwise
        """
        if device_id not in self.universe:
            logger.error(f"Device {device_id} not in Universe, cannot activate")
            return False
        
        if device_id in self.active_ids:
            logger.warning(f"Device {device_id} already active")
            return False
        
        if device_id not in self.inactive_ids:
            logger.error(f"Device {device_id} not in Inactive set, cannot activate")
            return False
        
        # Move from inactive to active
        self.inactive_ids.remove(device_id)
        self.active_ids.add(device_id)
        
        # Update metadata
        self.universe[device_id].is_active = True
        self.universe[device_id].join_count += 1
        
        logger.debug(f"Activated device {device_id} (join_count={self.universe[device_id].join_count})")
        return True
    
    def deactivate_device(self, device_id: str) -> bool:
        """
        Move device from Active to Inactive
        
        Args:
            device_id: Device identifier
        
        Returns:
            True if successful, False otherwise
        """
        if device_id not in self.universe:
            logger.error(f"Device {device_id} not in Universe, cannot deactivate")
            return False
        
        if device_id in self.inactive_ids:
            logger.warning(f"Device {device_id} already inactive")
            return False
        
        if device_id not in self.active_ids:
            logger.error(f"Device {device_id} not in Active set, cannot deactivate")
            return False
        
        # Move from active to inactive
        self.active_ids.remove(device_id)
        self.inactive_ids.add(device_id)
        
        # Update metadata
        self.universe[device_id].is_active = False
        self.universe[device_id].leave_count += 1
        
        logger.debug(f"Deactivated device {device_id} (leave_count={self.universe[device_id].leave_count})")
        return True
    
    def remove_device(self, device_id: str) -> bool:
        """
        Permanently remove device from Universe
        
        This is rarely used; typically devices are moved to Inactive instead.
        
        Args:
            device_id: Device identifier
        
        Returns:
            True if successful, False otherwise
        """
        if device_id not in self.universe:
            logger.warning(f"Device {device_id} not in Universe, cannot remove")
            return False
        
        # Remove from sets
        self.active_ids.discard(device_id)
        self.inactive_ids.discard(device_id)
        
        # Remove from universe
        del self.universe[device_id]
        
        logger.info(f"Permanently removed device {device_id} from Universe")
        return True
    
    def get_active_devices(self) -> List[DeviceInfo]:
        """Get list of active devices"""
        return [self.universe[device_id] for device_id in self.active_ids]
    
    def get_inactive_devices(self) -> List[DeviceInfo]:
        """Get list of inactive devices"""
        return [self.universe[device_id] for device_id in self.inactive_ids]
    
    def get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device metadata"""
        return self.universe.get(device_id)
    
    def get_active_device_ids(self) -> List[str]:
        """Get list of active device IDs (sorted for consistency)"""
        return sorted(self.active_ids)
    
    def get_inactive_device_ids(self) -> List[str]:
        """Get list of inactive device IDs (sorted for consistency)"""
        return sorted(self.inactive_ids)
    
    def n_active(self) -> int:
        """Number of active devices"""
        return len(self.active_ids)
    
    def n_inactive(self) -> int:
        """Number of inactive devices"""
        return len(self.inactive_ids)
    
    def n_universe(self) -> int:
        """Total number of devices in Universe"""
        return len(self.universe)
    
    def update_active_timesteps(self, timesteps: int = 1) -> None:
        """
        Update active timestep counter for all active devices
        
        Args:
            timesteps: Number of timesteps to add (default 1)
        """
        for device_id in self.active_ids:
            self.universe[device_id].total_timesteps_active += timesteps
    
    def get_device_types_distribution(self, active_only: bool = True) -> Dict[str, int]:
        """
        Get distribution of device types
        
        Args:
            active_only: Only count active devices
        
        Returns:
            Dictionary mapping device_type → count
        """
        devices = self.get_active_devices() if active_only else self.universe.values()
        distribution = {}
        for device_info in devices:
            device_type = device_info.device_type
            distribution[device_type] = distribution.get(device_type, 0) + 1
        return distribution
    
    def validate_consistency(self) -> bool:
        """
        Validate internal consistency
        
        Returns:
            True if consistent, False otherwise
        """
        # Check: Active ∩ Inactive = ∅
        if self.active_ids & self.inactive_ids:
            logger.error(f"Active and Inactive sets overlap: {self.active_ids & self.inactive_ids}")
            return False
        
        # Check: Active ∪ Inactive ⊆ Universe
        all_ids = self.active_ids | self.inactive_ids
        universe_ids = set(self.universe.keys())
        if not all_ids.issubset(universe_ids):
            logger.error(f"Active/Inactive not subset of Universe: {all_ids - universe_ids}")
            return False
        
        # Check: is_active flag consistency
        for device_id in self.active_ids:
            if not self.universe[device_id].is_active:
                logger.error(f"Device {device_id} in active_ids but is_active=False")
                return False
        
        for device_id in self.inactive_ids:
            if self.universe[device_id].is_active:
                logger.error(f"Device {device_id} in inactive_ids but is_active=True")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'manager_id': self.manager_id,
            'n_universe': self.n_universe(),
            'n_active': self.n_active(),
            'n_inactive': self.n_inactive(),
            'total_churn_events': self.total_churn_events,
            'total_devices_created': self.total_devices_created,
            'device_types_active': self.get_device_types_distribution(active_only=True),
            'device_types_universe': self.get_device_types_distribution(active_only=False),
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"DevicePool(manager={self.manager_id}, "
                f"active={self.n_active()}, inactive={self.n_inactive()}, "
                f"universe={self.n_universe()})")


class DevicePoolManager:
    """
    Manages DevicePool for all Managers in the environment
    """
    
    def __init__(self):
        """Initialize pool manager"""
        self.pools: Dict[str, DevicePool] = {}
        logger.info("DevicePoolManager initialized")
    
    def create_pool(self, manager_id: str) -> DevicePool:
        """
        Create a DevicePool for a manager
        
        Args:
            manager_id: Manager identifier
        
        Returns:
            Created DevicePool
        """
        if manager_id in self.pools:
            logger.warning(f"Pool for manager {manager_id} already exists")
            return self.pools[manager_id]
        
        pool = DevicePool(manager_id)
        self.pools[manager_id] = pool
        logger.info(f"Created pool for manager {manager_id}")
        return pool
    
    def get_pool(self, manager_id: str) -> Optional[DevicePool]:
        """Get pool for a manager"""
        return self.pools.get(manager_id)
    
    def get_all_pools(self) -> Dict[str, DevicePool]:
        """Get all pools"""
        return self.pools.copy()
    
    def get_total_active_devices(self) -> int:
        """Get total number of active devices across all managers"""
        return sum(pool.n_active() for pool in self.pools.values())
    
    def get_total_universe_devices(self) -> int:
        """Get total number of devices in all universe pools"""
        return sum(pool.n_universe() for pool in self.pools.values())
    
    def validate_all_pools(self) -> bool:
        """Validate consistency of all pools"""
        all_valid = True
        for manager_id, pool in self.pools.items():
            if not pool.validate_consistency():
                logger.error(f"Pool for manager {manager_id} is inconsistent")
                all_valid = False
        return all_valid
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        return {
            'n_managers': len(self.pools),
            'total_active': self.get_total_active_devices(),
            'total_universe': self.get_total_universe_devices(),
            'pools': {manager_id: pool.get_statistics() 
                     for manager_id, pool in self.pools.items()},
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"DevicePoolManager(managers={len(self.pools)}, "
                f"total_active={self.get_total_active_devices()}, "
                f"total_universe={self.get_total_universe_devices()})")


if __name__ == "__main__":
    # Test device pool
    print("=== Testing DevicePool ===\n")
    
    # Create pool
    pool = DevicePool(manager_id="M1")
    print(f"Initial pool: {pool}\n")
    
    # Add devices
    for i in range(5):
        device_info = DeviceInfo(
            device_id=f"device_{i}",
            device_type="Battery",
            manager_id="M1",
            user_id=f"user_{i // 2}",
            device_config={'capacity': 10.0},
            is_active=(i < 3),  # First 3 active, last 2 inactive
        )
        pool.add_device(device_info, is_active=device_info.is_active)
    
    print(f"After adding devices: {pool}")
    print(f"  Active: {pool.get_active_device_ids()}")
    print(f"  Inactive: {pool.get_inactive_device_ids()}")
    print()
    
    # Test activation
    print("Activating device_3:")
    pool.activate_device("device_3")
    print(f"  Pool: {pool}")
    print(f"  Active: {pool.get_active_device_ids()}")
    print()
    
    # Test deactivation
    print("Deactivating device_1:")
    pool.deactivate_device("device_1")
    print(f"  Pool: {pool}")
    print(f"  Inactive: {pool.get_inactive_device_ids()}")
    print()
    
    # Test statistics
    print("Pool statistics:")
    stats = pool.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test validation
    print(f"Pool consistency: {pool.validate_consistency()}")
    print()
    
    # Test PoolManager
    print("=== Testing DevicePoolManager ===\n")
    manager = DevicePoolManager()
    
    for manager_id in ["M1", "M2", "M3"]:
        pool = manager.create_pool(manager_id)
        for i in range(10):
            device_info = DeviceInfo(
                device_id=f"{manager_id}_device_{i}",
                device_type=["Battery", "EV", "HeatPump"][i % 3],
                manager_id=manager_id,
                user_id=f"user_{i}",
                device_config={},
            )
            pool.add_device(device_info)
    
    print(f"PoolManager: {manager}")
    print(f"All pools valid: {manager.validate_all_pools()}")
    print()
    
    print("Manager statistics:")
    stats = manager.get_statistics()
    print(f"  Total managers: {stats['n_managers']}")
    print(f"  Total active: {stats['total_active']}")
    print(f"  Total universe: {stats['total_universe']}")
    
    print("\n=== All tests passed ===")
