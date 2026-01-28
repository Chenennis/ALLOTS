"""
Compatibility Wrapper for MAPPO with Device Churn

MAPPO = Multi-Agent Proximal Policy Optimization (Centralized Critic)
On-policy algorithm with centralized value function.

Author: FOenv Team
Date: 2026-01-13
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from adapters.multi_manager_wrapper import MultiManagerCompatWrapper


class RolloutBuffer:
    """Simple rollout buffer for on-policy algorithms"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, log_prob, value, reward, done):
        """Store a transition"""
        self.buffer.append((obs, action, log_prob, value, reward, done))
    
    def get_all(self):
        """Get all transitions"""
        if len(self.buffer) == 0:
            return None
        
        batch = list(self.buffer)
        obs, actions, log_probs, values, rewards, dones = zip(*batch)
        
        return {
            'obs': np.array(obs),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'rewards': np.array(rewards),
            'dones': np.array(dones)
        }
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class SimpleMAPPOAgent:
    """Simplified MAPPO agent for compatibility"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        """Initialize simplified MAPPO"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Create simple actor and critic networks
        self.actor = self._build_actor().to(device)
        self.critic = self._build_critic().to(device)
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        
        self.rollout_buffer = RolloutBuffer()
    
    def _build_actor(self):
        """Build actor network"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.action_dim * 2)  # mean and log_std
        )
    
    def _build_critic(self):
        """Build centralized critic network"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim * self.n_agents, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    
    def select_action(self, state: np.ndarray, centralized_state: np.ndarray, deterministic: bool = False):
        """Select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        cent_state_tensor = torch.FloatTensor(centralized_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Actor output
            actor_out = self.actor(state_tensor)
            mean, log_std = torch.chunk(actor_out, 2, dim=-1)
            std = torch.exp(log_std.clamp(-20, 2))
            
            # Sample action
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            
            # Compute log prob
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            
            # Compute value
            value = self.critic(cent_state_tensor)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0, 0]
    
    def update(self, ppo_epochs: int = 4) -> Dict[str, float]:
        """Update policy using PPO"""
        batch = self.rollout_buffer.get_all()
        if batch is None or len(self.rollout_buffer) < 32:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Compute returns and advantages
        returns = self._compute_returns(batch['rewards'], batch['dones'], batch['values'])
        advantages = returns - batch['values']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(ppo_epochs):
            # Actor forward
            actor_out = self.actor(obs)
            mean, log_std = torch.chunk(actor_out, 2, dim=-1)
            std = torch.exp(log_std.clamp(-20, 2))
            dist = torch.distributions.Normal(mean, std)
            
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            
            # Policy loss (PPO clip)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Value loss
            cent_obs = obs.view(obs.size(0) // self.n_agents, -1)  # Flatten for centralized critic
            values = self.critic(cent_obs).squeeze(-1)
            values = values.repeat_interleave(self.n_agents)  # Expand back
            value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        self.rollout_buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / ppo_epochs,
            'value_loss': total_value_loss / ppo_epochs
        }
    
    def _compute_returns(self, rewards, dones, values):
        """Compute returns using GAE"""
        returns = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            returns[t] = gae + values[t]
        
        return returns


class MAPPOCompatAgent:
    """Compatibility wrapper for MAPPO with device churn"""
    
    def __init__(
        self,
        manager_ids: List[str],
        N_max: int,
        x_dim: int,
        g_dim: int,
        p: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        """Initialize MAPPO compatibility wrapper"""
        self.manager_ids = manager_ids
        self.n_managers = len(manager_ids)
        self.N_max = N_max
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.device = device
        
        # Create compatibility wrapper
        self.compat_wrapper = MultiManagerCompatWrapper(
            use_stable_mapping=True,  # 使用稳定映射
            manager_ids=manager_ids,
            N_max=N_max,
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            verbose=False
        )
        
        # Get fixed dimensions
        state_dim, action_dim = self.compat_wrapper.get_state_action_dims()
        
        # Create simplified MAPPO agent
        self.mappo = SimpleMAPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=self.n_managers,
            lr=lr,
            gamma=gamma,
            device=device
        )
        
        print(f"MAPPOCompatAgent initialized:")
        print(f"  Managers: {self.n_managers}")
        print(f"  State dim per manager: {state_dim}")
        print(f"  Action dim per manager: {action_dim}")
        print(f"  PPO hyperparameters: lr={lr}, gamma={gamma}")
    
    def select_actions(
        self,
        raw_obs: Dict[str, Dict],
        explore: bool = True
    ) -> Dict[str, np.ndarray]:
        """Select actions for all managers"""
        # Adapt observations
        adapted_obs = self.compat_wrapper.adapt_obs_all(raw_obs, format='separate')
        
        # Get centralized state
        centralized_state = np.concatenate([
            adapted_obs[mid]['obs_vec']
            for mid in self.manager_ids
        ])
        
        # Get masks
        masks = np.stack([
            adapted_obs[mid]['mask']
            for mid in self.manager_ids
        ], axis=0)
        
        # Select actions for each manager
        padded_actions = {}
        for i, manager_id in enumerate(self.manager_ids):
            state = adapted_obs[manager_id]['obs_vec']
            
            action, log_prob, value = self.mappo.select_action(
                state, centralized_state, deterministic=not explore
            )
            
            # Reshape and apply mask
            A_pad = action.reshape(self.N_max, self.p)
            mask = masks[i][:, np.newaxis]
            A_pad = A_pad * mask
            
            padded_actions[manager_id] = A_pad
            
            # Store for rollout buffer
            if not hasattr(self, '_temp_data'):
                self._temp_data = {}
            self._temp_data[manager_id] = {
                'obs': state,
                'action': action,
                'log_prob': log_prob,
                'value': value
            }
        
        return padded_actions
    
    def store_transition(
        self,
        obs: Dict[str, Dict],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, Dict],
        done: bool
    ):
        """Store transition in rollout buffer"""
        for manager_id in self.manager_ids:
            if hasattr(self, '_temp_data') and manager_id in self._temp_data:
                temp = self._temp_data[manager_id]
                reward = rewards.get(manager_id, 0.0)
                
                self.mappo.rollout_buffer.push(
                    temp['obs'],
                    temp['action'],
                    temp['log_prob'],
                    temp['value'],
                    reward,
                    done
                )
    
    def update(self) -> Dict[str, float]:
        """Update MAPPO policy"""
        return self.mappo.update(ppo_epochs=4)
    
    def get_buffer_size(self) -> int:
        """Get rollout buffer size"""
        return len(self.mappo.rollout_buffer)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'actor': self.mappo.actor.state_dict(),
            'critic': self.mappo.critic.state_dict(),
            'optimizer': self.mappo.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.mappo.actor.load_state_dict(checkpoint['actor'])
        self.mappo.critic.load_state_dict(checkpoint['critic'])
        self.mappo.optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":
    print("Testing MAPPOCompatAgent...")
    
    manager_ids = ['manager_1', 'manager_2', 'manager_3', 'manager_4']
    agent = MAPPOCompatAgent(
        manager_ids=manager_ids,
        N_max=44,
        x_dim=6,
        g_dim=26,
        p=5,
        device="cpu"
    )
    
    # Test observation
    raw_obs = {}
    for i in range(4):
        n_devices = np.random.randint(1, 5)
        device_ids = [f'dev_{i}_{j}' for j in range(n_devices)]
        raw_obs[f'manager_{i+1}'] = {
            'g': np.random.randn(26),
            'device_ids': device_ids,
            'device_states': {
                dev_id: np.random.randn(6)
                for dev_id in device_ids
            }
        }
    
    # Select actions
    actions = agent.select_actions(raw_obs, explore=True)
    print(f"\nActions generated for {len(actions)} managers")
    
    print("\n>>> MAPPOCompatAgent test passed!")
