"""
FOAGILE: AGILE algorithm adapted for FOgym environment
(Action-Graph Integrated Learning for Dynamic Environments)

Based on "Know Your Action Set: Learning Action Relations for Reinforcement Learning" (ICLR 2022)
Adapted for multi-agent flexible orchestration with device churn.
"""

from .foagile import FOAGILE

__all__ = ['FOAGILE']
