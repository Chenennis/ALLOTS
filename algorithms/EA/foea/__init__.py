"""
FOea - Environment-Adaptive Multi-Agent Reinforcement Learning

This package implements EA (Environment-Adaptive) algorithm for FlexOffer environments
with device churn support.

Author: FOenv Team
Date: 2026-01-12
"""

from algorithms.EA.foea.actor import SetToSetActor
from algorithms.EA.foea.critic import PairSetCritic
from algorithms.EA.foea.replay_buffer import ChurnAwareReplayBuffer
from algorithms.EA.foea.ea_agent import EAAgent
from algorithms.EA.foea.ea_trainer import EATrainer

__all__ = [
    'SetToSetActor',
    'PairSetCritic',
    'ChurnAwareReplayBuffer',
    'EAAgent',
    'EATrainer',
]
