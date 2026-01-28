"""
Compatibility Layer for Fixed-Dimension MARL Baselines with Device Churn

This module provides adapters to enable fixed-dimension MARL algorithms 
(MADDPG, MATD3, SQDDPG, MAPPO, MAIPPO) to work with FOgym's dynamic device sets.

UPDATED 2026-01-18: 
- Added SimpleSlotMapper for fair comparison (no stable mapping)
- MultiManagerCompatWrapper now uses SimpleSlotMapper by default

Author: FOenv Team
Date: 2026-01-13 (Updated: 2026-01-18)
"""

from .slot_mapper import SlotMapper
from .simple_slot_mapper import SimpleSlotMapper
from .obs_adapter import ObsAdapter
from .act_adapter import ActAdapter
from .multi_manager_wrapper import MultiManagerCompatWrapper

__all__ = [
    'SlotMapper',
    'SimpleSlotMapper',
    'ObsAdapter',
    'ActAdapter',
    'MultiManagerCompatWrapper',
]
