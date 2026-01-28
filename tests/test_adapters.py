"""
Unit tests for compatibility layer adapters

Run with: python tests/test_adapters.py

Author: FOenv Team
Date: 2026-01-13
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import unittest
from adapters.slot_mapper import SlotMapper
from adapters.obs_adapter import ObsAdapter
from adapters.act_adapter import ActAdapter
from adapters.multi_manager_wrapper import MultiManagerCompatWrapper


class TestSlotMapper(unittest.TestCase):
    """Test SlotMapper: device-to-slot mapping stability"""
    
    def test_basic_mapping(self):
        """Test basic slot allocation"""
        mapper = SlotMapper(N_max=10, manager_id="test_manager")
        
        # Initial state: all slots free
        self.assertEqual(len(mapper.free_slots), 10)
        self.assertEqual(mapper.mask.sum(), 0)
        
        # Add 3 devices
        devices = ['device_1', 'device_2', 'device_3']
        mapper.update_mapping(devices)
        
        # Check: 3 active, 7 free
        self.assertEqual(len(mapper.free_slots), 7)
        self.assertEqual(mapper.mask.sum(), 3)
        self.assertEqual(len(mapper.slot_of_device), 3)
        
        # All devices should have slots
        for dev in devices:
            self.assertIsNotNone(mapper.get_slot(dev))
    
    def test_mapping_stability(self):
        """Test that devices keep their slots while active"""
        mapper = SlotMapper(N_max=10, manager_id="test_manager")
        
        # Step 1: 3 devices join
        devices_t1 = ['dev_A', 'dev_B', 'dev_C']
        mapper.update_mapping(devices_t1)
        
        slots_t1 = {dev: mapper.get_slot(dev) for dev in devices_t1}
        
        # Step 2: Same devices (no churn)
        mapper.update_mapping(devices_t1)
        slots_t2 = {dev: mapper.get_slot(dev) for dev in devices_t1}
        
        # Slots must be identical (stability)
        self.assertEqual(slots_t1, slots_t2)
        
        # Step 3: One device leaves
        devices_t3 = ['dev_A', 'dev_C']  # dev_B left
        mapper.update_mapping(devices_t3)
        
        # dev_A and dev_C must keep their original slots
        self.assertEqual(mapper.get_slot('dev_A'), slots_t1['dev_A'])
        self.assertEqual(mapper.get_slot('dev_C'), slots_t1['dev_C'])
        
        # dev_B should have no slot
        self.assertIsNone(mapper.get_slot('dev_B'))
        
        # Step 4: New device joins
        devices_t4 = ['dev_A', 'dev_C', 'dev_D']
        mapper.update_mapping(devices_t4)
        
        # dev_A and dev_C still keep original slots
        self.assertEqual(mapper.get_slot('dev_A'), slots_t1['dev_A'])
        self.assertEqual(mapper.get_slot('dev_C'), slots_t1['dev_C'])
        
        # dev_D gets a new slot (likely the freed one from dev_B)
        self.assertIsNotNone(mapper.get_slot('dev_D'))
    
    def test_slot_reuse(self):
        """Test that freed slots are correctly reused"""
        mapper = SlotMapper(N_max=5, manager_id="test_manager")
        
        # Add 3 devices
        mapper.update_mapping(['dev_1', 'dev_2', 'dev_3'])
        slot_1 = mapper.get_slot('dev_1')
        
        # Remove dev_1
        mapper.update_mapping(['dev_2', 'dev_3'])
        
        # slot_1 should be free
        self.assertIn(slot_1, mapper.free_slots)
        
        # Add new device
        mapper.update_mapping(['dev_2', 'dev_3', 'dev_4'])
        
        # dev_4 should get the freed slot (smallest available)
        self.assertEqual(mapper.get_slot('dev_4'), slot_1)
    
    def test_overflow(self):
        """Test N_max overflow detection"""
        mapper = SlotMapper(N_max=3, manager_id="test_manager")
        
        # Try to add 4 devices (exceeds N_max=3)
        with self.assertRaises(RuntimeError) as cm:
            mapper.update_mapping(['dev_1', 'dev_2', 'dev_3', 'dev_4'])
        
        self.assertIn("N_max", str(cm.exception))


class TestObsAdapter(unittest.TestCase):
    """Test ObsAdapter: variable to fixed-length observations"""
    
    def test_to_padded(self):
        """Test observation padding"""
        N_max, x_dim = 5, 3
        mapper = SlotMapper(N_max, "test_manager")
        adapter = ObsAdapter(mapper, N_max, x_dim)
        
        # 2 devices with states
        device_ids = ['dev_A', 'dev_B']
        device_states = {
            'dev_A': np.array([1.0, 2.0, 3.0]),
            'dev_B': np.array([4.0, 5.0, 6.0])
        }
        
        X_pad, mask = adapter.to_padded(device_ids, device_states)
        
        # Check shapes
        self.assertEqual(X_pad.shape, (N_max, x_dim))
        self.assertEqual(mask.shape, (N_max,))
        
        # Check mask
        self.assertEqual(mask.sum(), 2)  # 2 active devices
        
        # Check padded values
        slot_A = mapper.get_slot('dev_A')
        slot_B = mapper.get_slot('dev_B')
        
        np.testing.assert_array_equal(X_pad[slot_A], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(X_pad[slot_B], [4.0, 5.0, 6.0])
        
        # Inactive slots should be zero
        for i in range(N_max):
            if mask[i] == 0:
                np.testing.assert_array_equal(X_pad[i], [0.0, 0.0, 0.0])


class TestActAdapter(unittest.TestCase):
    """Test ActAdapter: fixed to aligned actions"""
    
    def test_to_aligned_action_set(self):
        """Test action alignment"""
        N_max, p = 5, 2
        mapper = SlotMapper(N_max, "test_manager")
        adapter = ActAdapter(mapper, N_max, p)
        
        # Set up mapping
        mapper.update_mapping(['dev_A', 'dev_B'])
        slot_A = mapper.get_slot('dev_A')
        slot_B = mapper.get_slot('dev_B')
        
        # Create padded actions (including noise on inactive slots)
        A_pad = np.random.randn(N_max, p).astype(np.float32)
        A_pad[slot_A] = [1.0, 2.0]
        A_pad[slot_B] = [3.0, 4.0]
        
        # Convert to action_set (with masking)
        action_set = adapter.to_aligned_action_set(A_pad, apply_mask=True)
        
        # Check: only active devices in action_set
        self.assertEqual(len(action_set), 2)
        self.assertIn('dev_A', action_set)
        self.assertIn('dev_B', action_set)
        
        # Check action values
        np.testing.assert_array_equal(action_set['dev_A'], [1.0, 2.0])
        np.testing.assert_array_equal(action_set['dev_B'], [3.0, 4.0])
    
    def test_mask_prevents_leakage(self):
        """Test that masking zeros out inactive slot actions"""
        N_max, p = 5, 2
        mapper = SlotMapper(N_max, "test_manager")
        adapter = ActAdapter(mapper, N_max, p)
        
        # Only 1 active device
        mapper.update_mapping(['dev_A'])
        
        # Create actions with nonzero values on inactive slots
        A_pad = np.ones((N_max, p), dtype=np.float32)  # All 1.0
        
        # Apply masking in-place
        A_pad_masked = adapter.mask_actions_inplace(A_pad.copy())
        
        # Check: only slot for dev_A is nonzero
        slot_A = mapper.get_slot('dev_A')
        for i in range(N_max):
            if i == slot_A:
                np.testing.assert_array_equal(A_pad_masked[i], [1.0, 1.0])
            else:
                np.testing.assert_array_equal(A_pad_masked[i], [0.0, 0.0])


class TestRoundTripAlignment(unittest.TestCase):
    """Test round-trip: raw obs -> padded -> actions -> action_set"""
    
    def test_round_trip(self):
        """Test complete workflow alignment"""
        N_max, x_dim, p = 10, 4, 3
        mapper = SlotMapper(N_max, "test_manager")
        obs_adapter = ObsAdapter(mapper, N_max, x_dim)
        act_adapter = ActAdapter(mapper, N_max, p)
        
        # Step 1: Raw obs to padded
        device_ids = ['dev_1', 'dev_2', 'dev_3']
        device_states = {
            f'dev_{i+1}': np.random.randn(x_dim).astype(np.float32)
            for i in range(3)
        }
        
        X_pad, mask = obs_adapter.to_padded(device_ids, device_states)
        
        # Step 2: Generate dummy actions
        A_pad = np.random.randn(N_max, p).astype(np.float32)
        
        # Step 3: Convert to action_set
        action_set = act_adapter.to_aligned_action_set(A_pad, apply_mask=True)
        
        # Check: all active devices have actions
        self.assertEqual(set(action_set.keys()), set(device_ids))
        
        # Check: device_ids match
        for dev_id in device_ids:
            slot = mapper.get_slot(dev_id)
            self.assertIsNotNone(slot)
            # Action should come from the correct slot
            np.testing.assert_array_equal(
                action_set[dev_id],
                A_pad[slot]
            )


class TestMultiManagerWrapper(unittest.TestCase):
    """Test MultiManagerCompatWrapper"""
    
    def test_adapt_obs_all(self):
        """Test observation adaptation for multiple managers"""
        N_max, x_dim, g_dim, p = 5, 3, 4, 2
        manager_ids = ['manager_1', 'manager_2']
        
        wrapper = MultiManagerCompatWrapper(
            manager_ids, N_max, x_dim, g_dim, p
        )
        
        # Raw observations
        raw_obs = {
            'manager_1': {
                'g': np.array([1.0, 2.0, 3.0, 4.0]),
                'device_ids': ['dev_A', 'dev_B'],
                'device_states': {
                    'dev_A': np.array([1.0, 2.0, 3.0]),
                    'dev_B': np.array([4.0, 5.0, 6.0])
                }
            },
            'manager_2': {
                'g': np.array([5.0, 6.0, 7.0, 8.0]),
                'device_ids': ['dev_C'],
                'device_states': {
                    'dev_C': np.array([7.0, 8.0, 9.0])
                }
            }
        }
        
        # Adapt observations
        adapted = wrapper.adapt_obs_all(raw_obs, format='separate')
        
        # Check both managers
        self.assertIn('manager_1', adapted)
        self.assertIn('manager_2', adapted)
        
        # Check manager_1
        obs_1 = adapted['manager_1']
        self.assertEqual(obs_1['X_pad'].shape, (N_max, x_dim))
        self.assertEqual(obs_1['mask'].sum(), 2)  # 2 devices
        
        # Check manager_2
        obs_2 = adapted['manager_2']
        self.assertEqual(obs_2['X_pad'].shape, (N_max, x_dim))
        self.assertEqual(obs_2['mask'].sum(), 1)  # 1 device
    
    def test_get_dims(self):
        """Test dimension calculation"""
        N_max, x_dim, g_dim, p = 10, 6, 26, 5
        manager_ids = ['manager_1', 'manager_2', 'manager_3', 'manager_4']
        
        wrapper = MultiManagerCompatWrapper(
            manager_ids, N_max, x_dim, g_dim, p
        )
        
        state_dim, action_dim = wrapper.get_state_action_dims()
        
        # state_dim = g_dim + N_max*x_dim + N_max
        expected_state = g_dim + N_max * x_dim + N_max
        self.assertEqual(state_dim, expected_state)
        
        # action_dim = N_max * p
        expected_action = N_max * p
        self.assertEqual(action_dim, expected_action)
        
        # Centralized dims
        cent_state, cent_action = wrapper.get_centralized_dims()
        self.assertEqual(cent_state, state_dim * 4)
        self.assertEqual(cent_action, action_dim * 4)


def run_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running Compatibility Layer Unit Tests")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(">>> ALL TESTS PASSED!")
    else:
        print(">>> SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
