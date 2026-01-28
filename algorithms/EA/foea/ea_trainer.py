"""
EA Trainer for Multi-Agent Environment with Churn

This module implements the training loop for EA algorithm with churn-enabled
FlexOffer environments.

Author: FOenv Team
Date: 2026-01-12
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

from algorithms.EA.foea.ea_agent import EAAgent
from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv

logger = logging.getLogger(__name__)


class EATrainer:
    """
    EA Trainer for Multi-Agent FlexOffer Environment
    
    Handles:
        - Training loop with churn support
        - Per-manager EA agents
        - Episode management
        - Metrics tracking and logging
        - Model checkpointing
    """
    
    def __init__(
        self,
        env: MultiAgentFlexOfferEnv,
        agent_config: Dict,
        train_config: Dict,
        output_dir: str = "results/ea",
    ):
        """
        Initialize EA Trainer
        
        Args:
            env: Multi-agent FlexOffer environment
            agent_config: Agent configuration dict
            train_config: Training configuration dict
            output_dir: Output directory for results
        """
        self.env = env
        self.agent_config = agent_config
        self.train_config = train_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract dimensions from environment
        self.manager_ids = env.manager_ids
        self.num_managers = len(self.manager_ids)
        
        # Create EA agents (one per manager)
        self.agents: Dict[str, EAAgent] = {}
        for manager_id in self.manager_ids:
            self.agents[manager_id] = EAAgent(
                **agent_config,
                num_managers=self.num_managers,
            )
        
        # Training state
        self.episode = 0
        self.total_timesteps = 0
        self.best_reward = -float('inf')
        
        # Metrics history
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
        logger.info(f"EATrainer initialized with {self.num_managers} managers")
        logger.info(f"Output directory: {self.output_dir}")
    
    def train(
        self,
        num_episodes: int,
        warmup_episodes: int = 10,
        eval_interval: int = 50,
        save_interval: int = 100,
    ):
        """
        Main training loop
        
        Args:
            num_episodes: Number of episodes to train
            warmup_episodes: Episodes before starting updates
            eval_interval: Interval for evaluation
            save_interval: Interval for saving checkpoints
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        logger.info(f"Warmup: {warmup_episodes} episodes")
        
        for ep in range(num_episodes):
            self.episode = ep
            
            # Run episode
            episode_data = self._run_episode(explore=(ep >= warmup_episodes))
            
            # Log episode results
            self._log_episode(episode_data)
            
            # Evaluation
            if (ep + 1) % eval_interval == 0:
                eval_data = self._evaluate()
                self._log_evaluation(eval_data)
            
            # Save checkpoint
            if (ep + 1) % save_interval == 0:
                self._save_checkpoint(f"checkpoint_ep{ep+1}.pt")
        
        # Final save
        self._save_checkpoint("final.pt")
        self._save_training_history()
        
        logger.info("Training completed")
    
    def _run_episode(self, explore: bool = True) -> Dict:
        """
        Run one episode
        
        Args:
            explore: Whether to use exploration noise
        
        Returns:
            Episode data dict
        """
        obs, info = self.env.reset()
        done = False
        episode_reward = {mid: 0.0 for mid in self.manager_ids}
        episode_length = 0
        update_metrics = []
        
        while not done:
            # Select actions for all managers
            actions = {}
            for manager_id in self.manager_ids:
                # Extract components from observation (assuming structured obs)
                g, X, mask = self._extract_obs_components(obs[manager_id], manager_id)
                
                # Select action
                agent = self.agents[manager_id]
                manager_idx = self.manager_ids.index(manager_id)
                A = agent.select_action(g, X, mask, manager_idx, explore=explore)
                
                # Convert padded action to FOgym action dict format
                actions[manager_id] = self._convert_action(A, mask)
            
            # Environment step
            next_obs, rewards, dones, truncated, next_info = self.env.step(actions)
            done = dones['__all__']
            
            # Store transitions for all managers
            for manager_id in self.manager_ids:
                g, X, mask = self._extract_obs_components(obs[manager_id], manager_id)
                g_next, X_next, mask_next = self._extract_obs_components(next_obs[manager_id], manager_id)
                
                agent = self.agents[manager_id]
                manager_idx = self.manager_ids.index(manager_id)
                
                # Get stored action (re-convert for consistency)
                A = agent.select_action(g, X, mask, manager_idx, explore=False)  # Deterministic
                
                agent.store_transition(
                    manager_id=manager_idx,
                    g=g,
                    X=X,
                    mask=mask,
                    A=A,
                    r=rewards[manager_id],
                    g_next=g_next,
                    X_next=X_next,
                    mask_next=mask_next,
                    done=done,
                )
                
                episode_reward[manager_id] += rewards[manager_id]
            
            # Update agents (if after warmup)
            if explore:
                for manager_id in self.manager_ids:
                    metrics = self.agents[manager_id].update(
                        batch_size=self.train_config.get('batch_size', 256)
                    )
                    if metrics:
                        update_metrics.append(metrics)
            
            obs = next_obs
            episode_length += 1
            self.total_timesteps += 1
        
        return {
            'episode': self.episode,
            'rewards': episode_reward,
            'total_reward': sum(episode_reward.values()),
            'length': episode_length,
            'update_metrics': update_metrics,
        }
    
    def _evaluate(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate agent performance (deterministic)
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation results dict
        """
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            episode_data = self._run_episode(explore=False)
            eval_rewards.append(episode_data['total_reward'])
            eval_lengths.append(episode_data['length'])
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
        }
    
    def _extract_obs_components(self, obs: np.ndarray, manager_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract g, X, mask from observation
        
        This is a placeholder implementation. In practice, you would need to:
        1. Parse the observation structure
        2. Extract global features (g)
        3. Extract device states (X)
        4. Construct mask from active devices
        
        Args:
            obs: Observation array from environment
            manager_id: Manager identifier
        
        Returns:
            (g, X, mask) tuple
        """
        # PLACEHOLDER: This needs to be implemented based on your observation structure
        # For now, return dummy values matching expected dimensions
        
        g_dim = self.agent_config['g_dim']
        N_max = self.agent_config['N_max']
        x_dim = self.agent_config['x_dim']
        
        # In real implementation, parse obs to get these
        g = np.random.randn(g_dim)  # Extract global features
        X = np.random.randn(N_max, x_dim)  # Extract device states
        mask = np.ones(N_max)  # Extract active device mask
        
        # Example parsing (adapt to your observation structure):
        # manager = self.env.manager_agents[manager_id]
        # n_devices = len(manager.controllable_devices)
        # mask = np.zeros(N_max)
        # mask[:n_devices] = 1
        # ... extract device features ...
        
        return g, X, mask
    
    def _convert_action(self, A: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Convert padded action [N_max, p] to FOgym action format
        
        Args:
            A: Padded actions [N_max, p]
            mask: Active device mask [N_max]
        
        Returns:
            Action array in FOgym format
        """
        # PLACEHOLDER: Convert based on your action space
        # For now, flatten active actions
        n_active = int(mask.sum())
        return A[:n_active].flatten()
    
    def _log_episode(self, episode_data: Dict):
        """Log episode results"""
        ep = episode_data['episode']
        total_reward = episode_data['total_reward']
        length = episode_data['length']
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        
        if ep % 10 == 0:
            logger.info(f"Episode {ep}: reward={total_reward:.2f}, length={length}")
            
            # Log per-manager rewards
            for mid, r in episode_data['rewards'].items():
                logger.debug(f"  {mid}: {r:.2f}")
    
    def _log_evaluation(self, eval_data: Dict):
        """Log evaluation results"""
        logger.info(f"Evaluation - Mean reward: {eval_data['mean_reward']:.2f} " +
                   f"(±{eval_data['std_reward']:.2f})")
        
        # Update best reward
        if eval_data['mean_reward'] > self.best_reward:
            self.best_reward = eval_data['mean_reward']
            self._save_checkpoint("best.pt")
            logger.info(f"New best reward: {self.best_reward:.2f}")
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / filename
        
        checkpoint = {
            'episode': self.episode,
            'total_timesteps': self.total_timesteps,
            'best_reward': self.best_reward,
            'agents': {mid: None for mid in self.manager_ids},  # Placeholder
        }
        
        # Save individual agent states
        for manager_id in self.manager_ids:
            agent_path = self.output_dir / f"agent_{manager_id}_{filename}"
            self.agents[manager_id].save(str(agent_path))
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_metrics': self.training_metrics,
        }
        
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved: {history_path}")


if __name__ == "__main__":
    # Test trainer (placeholder)
    print("EATrainer module loaded successfully")
    print("Use train.py script to run actual training")
