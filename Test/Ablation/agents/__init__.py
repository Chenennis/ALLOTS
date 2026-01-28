"""
Ablation Agent Variants for SALSA/EA Algorithm

This module provides three ablation variants:
1. EAAgentNoPairSet - w/o Pair-Set Critic (separate state/action encoding)
2. EAAgentNoTDConsistent - w/o TD-Consistent Bootstrapping (padding-based next-set)
3. EAAgentNoCredit - w/o Per-Device Credit Assignment (uniform weights)

Author: FOenv Team
Date: 2026-01-20
"""

from Test.Ablation.agents.ea_no_pairset import EAAgentNoPairSet
from Test.Ablation.agents.ea_no_tdconsistent import EAAgentNoTDConsistent
from Test.Ablation.agents.ea_no_credit import EAAgentNoCredit

__all__ = [
    'EAAgentNoPairSet',
    'EAAgentNoTDConsistent',
    'EAAgentNoCredit',
]
