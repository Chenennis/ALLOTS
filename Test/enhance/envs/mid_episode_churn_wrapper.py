"""
Mid-Episode Churn Wrapper for Ablation Study

This wrapper enables churn events to occur during episode steps (not just at reset),
allowing proper testing of TD-Consistent bootstrapping module.

Key Features:
- Triggers churn at specified steps within an episode
- Properly updates observations after churn
- Reports churn events in info dict
- Does NOT modify the original FOgym environment

Author: FOenv Team
Date: 2026-01-21
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class MidEpisodeChurnWrapper:
    """
    Wrapper that enables mid-episode churn for FOgym environment.
    
    This wrapper intercepts step() calls and triggers churn at specified
    steps within an episode, allowing TD-Consistent module to be properly tested.
    
    Usage:
        env = MultiAgentFlexOfferEnv(...)
        wrapped_env = MidEpisodeChurnWrapper(env, churn_steps=[6, 12, 18])
    """
    
    def __init__(
        self,
        env,
        churn_steps: Optional[List[int]] = None,
        churn_prob: float = 0.0,
        verbose: bool = False
    ):
        """
        Initialize the wrapper.
        
        Args:
            env: Original FOgym MultiAgentFlexOfferEnv
            churn_steps: List of step indices where churn should occur (e.g., [6, 12, 18])
                        If None, uses churn_prob for probabilistic triggering
            churn_prob: Probability of churn at each step (used if churn_steps is None)
            verbose: Whether to log churn events
        """
        self.env = env
        self.churn_steps = churn_steps or [6, 12, 18]  # Default: every 6 hours
        self.churn_prob = churn_prob
        self.verbose = verbose
        
        # Track current step within episode
        self._episode_step = 0
        self._churn_events_this_episode = []
        
        # Validate that environment has churn support
        if not hasattr(env, 'churn_manager') or env.churn_manager is None:
            logger.warning("Environment does not have churn_manager! Mid-episode churn will be disabled.")
            self._churn_enabled = False
        else:
            self._churn_enabled = True
            logger.info(f"MidEpisodeChurnWrapper initialized with churn_steps={self.churn_steps}")
    
    def reset(self, **kwargs):
        """Reset environment and internal state."""
        self._episode_step = 0
        self._churn_events_this_episode = []
        return self.env.reset(**kwargs)
    
    def step(self, actions: Dict[str, np.ndarray]):
        """
        Execute one step, potentially triggering mid-episode churn.
        
        Args:
            actions: Dict of manager_id -> action array
            
        Returns:
            observations, rewards, dones, truncated, infos
        """
        # Execute the original step
        obs, rewards, dones, truncated, infos = self.env.step(actions)
        
        # Increment step counter
        self._episode_step += 1
        
        # Check if we should trigger mid-episode churn
        if self._should_trigger_churn():
            churn_events = self._execute_mid_episode_churn()
            
            if churn_events:
                # Update observations (device set may have changed!)
                obs = self.env._get_observations()
                
                # Add churn info
                for event in churn_events:
                    if event.manager_id in infos:
                        infos[event.manager_id]['mid_episode_churn'] = True
                        infos[event.manager_id]['churn_event'] = event.to_dict()
                
                self._churn_events_this_episode.extend(churn_events)
                
                if self.verbose:
                    logger.info(f"Mid-episode churn at step {self._episode_step}: "
                               f"{len(churn_events)} events")
        
        # Add step info
        for manager_id in infos:
            infos[manager_id]['episode_step'] = self._episode_step
            infos[manager_id]['mid_episode_churn'] = infos[manager_id].get('mid_episode_churn', False)
        
        return obs, rewards, dones, truncated, infos
    
    def _should_trigger_churn(self) -> bool:
        """Check if churn should be triggered at current step."""
        if not self._churn_enabled:
            return False
        
        # Check if current step is in churn_steps
        if self._episode_step in self.churn_steps:
            return True
        
        # Or use probabilistic triggering
        if self.churn_prob > 0 and np.random.random() < self.churn_prob:
            return True
        
        return False
    
    def _execute_mid_episode_churn(self) -> List[Any]:
        """
        Execute churn for all managers.
        
        Returns:
            List of ChurnEvent objects
        """
        churn_events = []
        
        if not self._churn_enabled:
            return churn_events
        
        try:
            # Use the environment's internal churn execution
            churn_events = self.env._execute_churn()
            
            if self.verbose and churn_events:
                for event in churn_events:
                    logger.info(f"  {event}")
                    
        except Exception as e:
            logger.error(f"Error executing mid-episode churn: {e}")
        
        return churn_events
    
    # Delegate all other attributes to the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def manager_ids(self):
        return self.env.manager_ids
    
    @property
    def manager_agents(self):
        return self.env.manager_agents
    
    def get_churn_events_this_episode(self) -> List[Any]:
        """Get all churn events that occurred in the current episode."""
        return self._churn_events_this_episode
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def close(self):
        return self.env.close()


def test_mid_episode_churn_wrapper():
    """Test the wrapper functionality."""
    print("=" * 60)
    print("Testing MidEpisodeChurnWrapper")
    print("=" * 60)
    
    # This test would require an actual FOgym environment
    # For now, just test the basic structure
    
    class MockEnv:
        def __init__(self):
            self.churn_manager = "mock"
            self.device_pool_manager = "mock"
            self.current_step = 0
            
        def reset(self, **kwargs):
            self.current_step = 0
            return {"obs": "test"}, {}
        
        def step(self, actions):
            self.current_step += 1
            obs = {"obs": f"step_{self.current_step}"}
            rewards = {"manager_1": 1.0}
            dones = {"__all__": self.current_step >= 24}
            infos = {"manager_1": {}}
            return obs, rewards, dones, False, infos
        
        def _get_observations(self):
            return {"obs": f"step_{self.current_step}_updated"}
        
        def _execute_churn(self):
            return []  # No actual churn in mock
    
    mock_env = MockEnv()
    wrapper = MidEpisodeChurnWrapper(mock_env, churn_steps=[6, 12, 18], verbose=True)
    
    print("\n1. Testing reset...")
    obs, info = wrapper.reset()
    print(f"   Reset successful: {obs}")
    
    print("\n2. Testing steps with churn triggers...")
    for i in range(24):
        obs, rewards, dones, truncated, infos = wrapper.step({"manager_1": None})
        if i + 1 in [6, 12, 18]:
            print(f"   Step {i+1}: Churn should trigger")
    
    print("\n3. Wrapper test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_mid_episode_churn_wrapper()
