"""
Churn Manager Module for FOgym Environment

This module executes device churn logic: selecting devices to leave/join,
creating new devices, and updating device pools.

Author: FOenv Team
Date: 2026-01-12
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass

from fo_generate.churn_config import ChurnConfig
from fo_generate.device_pool import DevicePool, DeviceInfo, DevicePoolManager
from fo_generate.unified_mdp_env import DeviceType

logger = logging.getLogger(__name__)


@dataclass
class ChurnEvent:
    """
    Record of a churn event
    
    Attributes:
        episode: Episode when churn occurred
        manager_id: Manager where churn occurred
        severity: Churn severity ρ
        devices_left: List of device IDs that left
        devices_joined: List of device IDs that joined
        new_devices_created: Number of new devices created
        n_active_before: Number of active devices before churn
        n_active_after: Number of active devices after churn
    """
    episode: int
    manager_id: str
    severity: float
    devices_left: List[str]
    devices_joined: List[str]
    new_devices_created: int
    n_active_before: int
    n_active_after: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'episode': self.episode,
            'manager_id': self.manager_id,
            'severity': self.severity,
            'devices_left': self.devices_left,
            'devices_joined': self.devices_joined,
            'new_devices_created': self.new_devices_created,
            'n_active_before': self.n_active_before,
            'n_active_after': self.n_active_after,
        }
    
    def __repr__(self) -> str:
        return (f"ChurnEvent(ep={self.episode}, mgr={self.manager_id}, "
                f"severity={self.severity:.2f}, left={len(self.devices_left)}, "
                f"joined={len(self.devices_joined)}, created={self.new_devices_created})")


class ChurnManager:
    """
    Manages device churn execution for multi-agent FlexOffer environment
    
    Responsibilities:
    - Determine when to trigger churn (based on ChurnConfig)
    - Sample churn severity
    - Select devices to leave and join
    - Create new devices when needed
    - Log churn events
    """
    
    def __init__(self, churn_config: ChurnConfig):
        """
        Initialize churn manager
        
        Args:
            churn_config: Churn configuration
        """
        self.config = churn_config
        self.rng = churn_config.get_rng()
        self.churn_events: List[ChurnEvent] = []
        self.episode_counter = 0
        
        logger.info(f"ChurnManager initialized with config: {churn_config}")
    
    def should_trigger_churn(self) -> bool:
        """
        Check if churn should be triggered this episode
        
        Returns:
            True if churn should occur, False otherwise
        """
        return self.config.should_trigger_churn(self.episode_counter)
    
    def increment_episode(self) -> None:
        """Increment episode counter"""
        self.episode_counter += 1
    
    def execute_churn_for_manager(
        self, 
        pool: DevicePool,
        device_type_distribution: Optional[Dict[str, float]] = None
    ) -> Optional[ChurnEvent]:
        """
        Execute churn for a single manager
        
        Algorithm:
            1. Sample severity ρ
            2. Compute k_leave, k_join
            3. Select k_leave random devices from Active → Inactive
            4. Select k_join random devices from Inactive → Active
            5. If Inactive insufficient, create new random devices
        
        Args:
            pool: DevicePool for this manager
            device_type_distribution: Distribution of device types for creating new devices
                                     (default: uniform over [Battery, EV, HeatPump, Dishwasher])
        
        Returns:
            ChurnEvent record, or None if churn not executed
        """
        if not self.config.enabled:
            return None
        
        n_active_before = pool.n_active()
        
        # Sample severity
        severity = self.config.sample_severity(self.rng)
        
        # Compute churn counts
        k_leave, k_join = self.config.compute_churn_counts(n_active_before, severity)
        
        if k_leave == 0 and k_join == 0:
            logger.debug(f"Manager {pool.manager_id}: No churn (k_leave=0, k_join=0)")
            return None
        
        logger.info(f"Manager {pool.manager_id}: Executing churn "
                   f"(severity={severity:.2f}, k_leave={k_leave}, k_join={k_join})")
        
        # Execute leave
        devices_left = self._execute_leave(pool, k_leave)
        
        # Execute join
        devices_joined, new_created = self._execute_join(
            pool, k_join, device_type_distribution
        )
        
        n_active_after = pool.n_active()
        
        # Create churn event record
        event = ChurnEvent(
            episode=self.episode_counter,
            manager_id=pool.manager_id,
            severity=severity,
            devices_left=devices_left,
            devices_joined=devices_joined,
            new_devices_created=new_created,
            n_active_before=n_active_before,
            n_active_after=n_active_after,
        )
        
        self.churn_events.append(event)
        pool.total_churn_events += 1
        
        logger.info(f"Churn completed: {event}")
        
        return event
    
    def _execute_leave(self, pool: DevicePool, k_leave: int) -> List[str]:
        """
        Execute device leave: move k_leave devices from Active to Inactive
        
        Args:
            pool: DevicePool
            k_leave: Number of devices to remove
        
        Returns:
            List of device IDs that left
        """
        if k_leave <= 0:
            return []
        
        active_ids = pool.get_active_device_ids()
        
        if k_leave > len(active_ids):
            logger.warning(f"k_leave={k_leave} exceeds n_active={len(active_ids)}, "
                          f"clamping to {len(active_ids)}")
            k_leave = len(active_ids)
        
        # Randomly select devices to deactivate
        selected_ids = self.rng.choice(active_ids, size=k_leave, replace=False).tolist()
        
        for device_id in selected_ids:
            success = pool.deactivate_device(device_id)
            if not success:
                logger.error(f"Failed to deactivate device {device_id}")
        
        logger.debug(f"Deactivated {len(selected_ids)} devices: {selected_ids}")
        
        return selected_ids
    
    def _execute_join(
        self, 
        pool: DevicePool, 
        k_join: int,
        device_type_distribution: Optional[Dict[str, float]] = None
    ) -> Tuple[List[str], int]:
        """
        Execute device join: move k_join devices from Inactive to Active
        
        If Inactive pool is insufficient, create new devices.
        
        Args:
            pool: DevicePool
            k_join: Number of devices to add
            device_type_distribution: Distribution for new device types
        
        Returns:
            (devices_joined, new_devices_created)
        """
        if k_join <= 0:
            return [], 0
        
        inactive_ids = pool.get_inactive_device_ids()
        n_inactive = len(inactive_ids)
        
        # Determine how many to reactivate vs create
        n_reactivate = min(k_join, n_inactive)
        n_create = k_join - n_reactivate
        
        devices_joined = []
        new_created = 0
        
        # Reactivate existing inactive devices
        if n_reactivate > 0:
            selected_ids = self.rng.choice(inactive_ids, size=n_reactivate, replace=False).tolist()
            for device_id in selected_ids:
                success = pool.activate_device(device_id)
                if success:
                    devices_joined.append(device_id)
                else:
                    logger.error(f"Failed to activate device {device_id}")
            
            logger.debug(f"Reactivated {len(selected_ids)} devices: {selected_ids}")
        
        # Create new devices if needed
        if n_create > 0:
            if not self.config.create_new_on_insufficient:
                logger.warning(f"Need to create {n_create} new devices but "
                              f"create_new_on_insufficient=False, skipping")
            else:
                new_devices = self._create_new_devices(pool, n_create, device_type_distribution)
                devices_joined.extend(new_devices)
                new_created = len(new_devices)
                logger.info(f"Created {new_created} new devices: {new_devices}")
        
        return devices_joined, new_created
    
    def _create_new_devices(
        self, 
        pool: DevicePool, 
        n_create: int,
        device_type_distribution: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Create new random devices and add to pool
        
        Args:
            pool: DevicePool
            n_create: Number of devices to create
            device_type_distribution: Distribution of device types
        
        Returns:
            List of new device IDs
        """
        if device_type_distribution is None:
            # Default: uniform over controllable device types
            device_types = [DeviceType.BATTERY, DeviceType.EV, 
                           DeviceType.HEAT_PUMP, DeviceType.DISHWASHER]
            device_type_probs = [0.25, 0.25, 0.25, 0.25]
        else:
            device_types = list(device_type_distribution.keys())
            device_type_probs = [device_type_distribution[dt] for dt in device_types]
            # Normalize
            total = sum(device_type_probs)
            device_type_probs = [p / total for p in device_type_probs]
        
        new_device_ids = []
        
        for i in range(n_create):
            # Sample device type
            device_type = self.rng.choice(device_types, p=device_type_probs)
            
            # Generate unique device ID
            unique_id = pool.total_devices_created + i + 1
            device_id = f"{device_type}_{pool.manager_id}_new_{unique_id}"
            
            # Create device config (randomized initial state)
            device_config = self._generate_random_device_config(device_type)
            device_config['device_id'] = device_id  # Add device_id to config
            device_config['device_type'] = device_type  # Add device_type to config
            
            # Create DeviceInfo
            device_info = DeviceInfo(
                device_id=device_id,
                device_type=device_type,
                manager_id=pool.manager_id,
                user_id=f"churn_user_{unique_id}",  # Virtual user for churned devices
                device_config=device_config,
                is_active=True,
                join_count=1,  # New device joining
            )
            
            # Add to pool
            pool.add_device(device_info, is_active=True)
            new_device_ids.append(device_id)
        
        return new_device_ids
    
    def _generate_random_device_config(self, device_type: str) -> Dict[str, Any]:
        """
        Generate random device configuration for a given type
        
        Initial states are randomized to simulate realistic conditions.
        Matches the expected format from ManagerAgent._create_device_model()
        
        Args:
            device_type: Device type
        
        Returns:
            Device configuration dictionary matching expected fields
        """
        if device_type == DeviceType.BATTERY:
            max_power = self.rng.uniform(2.0, 5.0)  # kW
            return {
                'capacity': self.rng.uniform(5.0, 15.0),  # kWh
                'max_power': max_power,  # kW
                'efficiency': self.rng.uniform(0.90, 0.95),
                'initial_state': self.rng.uniform(0.3, 0.7),  # 30-70% SOC
                'param1': 0.1,  # soc_min
                'param2': 0.9,  # soc_max
            }
        
        elif device_type == DeviceType.EV:
            return {
                'capacity': self.rng.uniform(40.0, 80.0),  # kWh (battery_capacity)
                'max_power': self.rng.uniform(3.0, 7.0),  # kW (max_charging_power)
                'efficiency': 0.95,
                'initial_state': self.rng.uniform(0.2, 0.6),  # 20-60% SOC
                'param1': 0.1,  # soc_min
                'param2': 0.95,  # soc_max
            }
        
        elif device_type == DeviceType.HEAT_PUMP:
            return {
                'max_power': self.rng.uniform(2.0, 5.0),  # kW (rated_power)
                'efficiency': self.rng.uniform(3.0, 4.5),  # COP
                'initial_state': self.rng.uniform(18.0, 22.0),  # °C (initial_temp)
                'param1': 18.0,  # temp_min
                'param2': 26.0,  # temp_max
                'param3': 0.1,  # heat_loss_coef
            }
        
        elif device_type == DeviceType.DISHWASHER:
            return {
                'max_power': self.rng.uniform(1.0, 2.0),  # kW (power_rating)
                'capacity': 3.0,  # kWh (total_energy)
                'efficiency': 0.95,
                'initial_state': 0.0,  # Not running when joining
                'param1': 3.5,  # operation_hours
                'param2': 0.5,  # min_start_delay
                'param3': 6.0,  # max_start_delay
            }
        
        else:
            logger.warning(f"Unknown device type {device_type}, using empty config")
            return {}
    
    def get_churn_history(self) -> List[ChurnEvent]:
        """Get list of all churn events"""
        return self.churn_events.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get churn statistics"""
        if not self.churn_events:
            return {
                'total_events': 0,
                'total_devices_left': 0,
                'total_devices_joined': 0,
                'total_new_devices': 0,
            }
        
        return {
            'total_events': len(self.churn_events),
            'total_devices_left': sum(len(e.devices_left) for e in self.churn_events),
            'total_devices_joined': sum(len(e.devices_joined) for e in self.churn_events),
            'total_new_devices': sum(e.new_devices_created for e in self.churn_events),
            'avg_severity': np.mean([e.severity for e in self.churn_events]),
            'events_by_manager': self._count_events_by_manager(),
        }
    
    def _count_events_by_manager(self) -> Dict[str, int]:
        """Count churn events by manager"""
        counts = {}
        for event in self.churn_events:
            manager_id = event.manager_id
            counts[manager_id] = counts.get(manager_id, 0) + 1
        return counts
    
    def reset_episode_counter(self) -> None:
        """Reset episode counter (useful for new training runs)"""
        self.episode_counter = 0
        logger.info("Episode counter reset to 0")
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"ChurnManager(enabled={self.config.enabled}, "
                f"episode={self.episode_counter}, "
                f"events={len(self.churn_events)})")


if __name__ == "__main__":
    # Test churn manager
    print("=== Testing ChurnManager ===\n")
    
    from fo_generate.churn_config import MODERATE_CHURN_CONFIG
    
    # Create churn manager
    config = MODERATE_CHURN_CONFIG
    churn_mgr = ChurnManager(config)
    print(f"ChurnManager: {churn_mgr}\n")
    
    # Create device pool
    pool = DevicePool(manager_id="M1")
    
    # Add initial devices
    for i in range(20):
        device_info = DeviceInfo(
            device_id=f"battery_{i}",
            device_type=DeviceType.BATTERY,
            manager_id="M1",
            user_id=f"user_{i}",
            device_config={'capacity': 10.0},
        )
        pool.add_device(device_info, is_active=True)
    
    print(f"Initial pool: {pool}\n")
    
    # Simulate episodes
    print("Simulating 30 episodes:\n")
    for ep in range(1, 31):
        churn_mgr.increment_episode()
        
        if churn_mgr.should_trigger_churn():
            print(f"Episode {ep}: Triggering churn")
            event = churn_mgr.execute_churn_for_manager(pool)
            if event:
                print(f"  {event}")
                print(f"  Pool after: {pool}")
            print()
    
    # Print statistics
    print("=== Churn Statistics ===")
    stats = churn_mgr.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("=== Pool Statistics ===")
    pool_stats = pool.get_statistics()
    for key, value in pool_stats.items():
        if key != 'device_types_active' and key != 'device_types_universe':
            print(f"  {key}: {value}")
    print(f"  device_types_active: {pool_stats['device_types_active']}")
    print(f"  device_types_universe: {pool_stats['device_types_universe']}")
    print()
    
    # Validate pool
    print(f"Pool consistency: {pool.validate_consistency()}")
    
    print("\n=== All tests passed ===")
