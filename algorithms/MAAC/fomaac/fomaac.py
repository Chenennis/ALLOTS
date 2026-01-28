"""
FOMAAC: Multi-Actor-Attention-Critic for FOgym

Adapted from MAAC (Iqbal & Sha, ICML 2019) for:
- Continuous action space (instead of discrete)
- Device-level observations with padding and masks
- DDPG-style deterministic policy gradient

Key Features:
- Multi-head attention mechanism in centralized critic
- Each agent attends to other agents' state-action encodings
- Shared attention modules for efficient learning

Author: FOenv Team
Date: 2026-01-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class ContinuousPolicy(nn.Module):
    """
    Continuous action policy network.
    
    Replaces DiscretePolicy with continuous output using tanh activation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        norm_in: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = nn.Identity()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize output layer with small weights
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            actions: Continuous actions in [-1, 1] range [batch, output_dim]
        """
        x = self.in_fn(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class AttentionCriticContinuous(nn.Module):
    """
    Attention-based centralized critic for continuous actions.
    
    Each agent's Q-value is computed by:
    1. Encoding its own state-action pair
    2. Attending to other agents' state-action encodings
    3. Combining with attention-weighted values
    """
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        attend_heads: int = 4,
        norm_in: bool = True
    ):
        super().__init__()
        
        assert hidden_dim % attend_heads == 0
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.attend_heads = attend_heads
        self.attend_dim = hidden_dim // attend_heads
        
        # State-action encoder for each agent
        self.sa_encoders = nn.ModuleList()
        for _ in range(n_agents):
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('bn', nn.BatchNorm1d(state_dim + action_dim, affine=False))
            encoder.add_module('fc1', nn.Linear(state_dim + action_dim, hidden_dim))
            encoder.add_module('act1', nn.LeakyReLU())
            self.sa_encoders.append(encoder)
        
        # State encoder for Q computation
        self.state_encoders = nn.ModuleList()
        for _ in range(n_agents):
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('bn', nn.BatchNorm1d(state_dim, affine=False))
            encoder.add_module('fc1', nn.Linear(state_dim, hidden_dim))
            encoder.add_module('act1', nn.LeakyReLU())
            self.state_encoders.append(encoder)
        
        # Attention components (shared across agents)
        self.key_extractors = nn.ModuleList([
            nn.Linear(hidden_dim, self.attend_dim, bias=False)
            for _ in range(attend_heads)
        ])
        self.query_extractors = nn.ModuleList([
            nn.Linear(hidden_dim, self.attend_dim, bias=False)
            for _ in range(attend_heads)
        ])
        self.value_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, self.attend_dim),
                nn.LeakyReLU()
            )
            for _ in range(attend_heads)
        ])
        
        # Q-value head for each agent
        self.q_heads = nn.ModuleList()
        for _ in range(n_agents):
            q_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.q_heads.append(q_head)
    
    def forward(
        self,
        states: List[torch.Tensor],
        actions: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute Q-values for all agents.
        
        Args:
            states: List of state tensors [batch, state_dim] for each agent
            actions: List of action tensors [batch, action_dim] for each agent
            
        Returns:
            q_values: List of Q-value tensors [batch, 1] for each agent
        """
        # Encode state-action pairs
        sa_inputs = [torch.cat([s, a], dim=1) for s, a in zip(states, actions)]
        sa_encodings = [enc(sa) for enc, sa in zip(self.sa_encoders, sa_inputs)]
        
        # Encode states for Q computation
        s_encodings = [enc(s) for enc, s in zip(self.state_encoders, states)]
        
        # Compute attention for each agent
        q_values = []
        for a_i in range(self.n_agents):
            # Get all keys and values from other agents
            other_keys = []
            other_values = []
            
            for head_idx in range(self.attend_heads):
                head_keys = []
                head_values = []
                for j, sa_enc in enumerate(sa_encodings):
                    if j != a_i:
                        head_keys.append(self.key_extractors[head_idx](sa_enc))
                        head_values.append(self.value_extractors[head_idx](sa_enc))
                other_keys.append(head_keys)
                other_values.append(head_values)
            
            # Compute attention-weighted values
            attended_values = []
            for head_idx in range(self.attend_heads):
                query = self.query_extractors[head_idx](s_encodings[a_i])
                keys = torch.stack(other_keys[head_idx], dim=2)  # [batch, attend_dim, n_agents-1]
                values = torch.stack(other_values[head_idx], dim=2)  # [batch, attend_dim, n_agents-1]
                
                # Scaled dot-product attention
                attn_logits = torch.bmm(query.unsqueeze(1), keys)  # [batch, 1, n_agents-1]
                attn_logits = attn_logits / np.sqrt(self.attend_dim)
                attn_weights = F.softmax(attn_logits, dim=2)
                
                # Weighted sum of values
                attended = torch.bmm(values, attn_weights.transpose(1, 2)).squeeze(2)  # [batch, attend_dim]
                attended_values.append(attended)
            
            # Concatenate attention heads
            attended_concat = torch.cat(attended_values, dim=1)  # [batch, hidden_dim]
            
            # Compute Q-value
            q_input = torch.cat([s_encodings[a_i], attended_concat], dim=1)
            q = self.q_heads[a_i](q_input)
            q_values.append(q)
        
        return q_values


class ReplayBuffer:
    """Simple replay buffer for MAAC."""
    
    def __init__(self, capacity: int, n_agents: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_states: List[np.ndarray],
        done: bool
    ):
        """Store a transition."""
        self.buffer.append((states, actions, rewards, next_states, done))
    
    def sample(self, batch_size: int, device: str = 'cpu'):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        states = [[] for _ in range(self.n_agents)]
        actions = [[] for _ in range(self.n_agents)]
        rewards = [[] for _ in range(self.n_agents)]
        next_states = [[] for _ in range(self.n_agents)]
        dones = []
        
        for s, a, r, ns, d in batch:
            for i in range(self.n_agents):
                states[i].append(s[i])
                actions[i].append(a[i])
                rewards[i].append(r[i])
                next_states[i].append(ns[i])
            dones.append(d)
        
        # Convert to tensors
        states = [torch.FloatTensor(np.array(s)).to(device) for s in states]
        actions = [torch.FloatTensor(np.array(a)).to(device) for a in actions]
        rewards = [torch.FloatTensor(r).unsqueeze(1).to(device) for r in rewards]
        next_states = [torch.FloatTensor(np.array(ns)).to(device) for ns in next_states]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class FOMAAC:
    """
    FOMAAC: Multi-Actor-Attention-Critic for FOgym.
    
    Continuous action version of MAAC with:
    - DDPG-style deterministic policy gradient
    - Multi-head attention in centralized critic
    - Target networks with soft updates
    """
    
    def __init__(
        self,
        n_agents: int,
        N_max: int,
        device_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        attend_heads: int = 4,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.1,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        device: str = 'cpu'
    ):
        self.n_agents = n_agents
        self.N_max = N_max
        self.device_dim = device_dim
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.device = device
        
        # State dimension: flattened device states + global features
        self.state_dim = N_max * device_dim + global_dim
        # Total action dimension
        self.total_action_dim = N_max * action_dim
        
        # Actor networks (one per agent)
        self.actors = nn.ModuleList([
            ContinuousPolicy(self.state_dim, self.total_action_dim, hidden_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        self.target_actors = nn.ModuleList([
            ContinuousPolicy(self.state_dim, self.total_action_dim, hidden_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        # Copy weights to target
        for actor, target in zip(self.actors, self.target_actors):
            target.load_state_dict(actor.state_dict())
        
        # Critic network (shared, with attention)
        self.critic = AttentionCriticContinuous(
            n_agents, self.state_dim, self.total_action_dim,
            hidden_dim, attend_heads
        ).to(device)
        
        self.target_critic = AttentionCriticContinuous(
            n_agents, self.state_dim, self.total_action_dim,
            hidden_dim, attend_heads
        ).to(device)
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizers = [
            Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, n_agents)
        
        print(f"FOMAAC initialized:")
        print(f"  n_agents: {n_agents}, N_max: {N_max}")
        print(f"  state_dim: {self.state_dim}, action_dim: {self.total_action_dim}")
        print(f"  attend_heads: {attend_heads}, hidden_dim: {hidden_dim}")
    
    def select_actions(
        self,
        device_states: np.ndarray,
        global_feats: np.ndarray,
        masks: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Select actions for all agents.
        
        Args:
            device_states: [n_agents, N_max, device_dim]
            global_feats: [n_agents, global_dim]
            masks: [n_agents, N_max] - 1 for active, 0 for inactive
            add_noise: Whether to add exploration noise
            
        Returns:
            actions: [n_agents, N_max, action_dim]
        """
        actions = []
        
        for i in range(self.n_agents):
            # Flatten state
            flat_devices = device_states[i].flatten()  # [N_max * device_dim]
            state = np.concatenate([flat_devices, global_feats[i]])  # [state_dim]
            
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                self.actors[i].eval()
                action = self.actors[i](state_t).cpu().numpy()[0]
                self.actors[i].train()
            
            if add_noise:
                noise = np.random.normal(0, self.noise_scale, action.shape)
                action = np.clip(action + noise, -1, 1)
            
            # Reshape to [N_max, action_dim]
            action = action.reshape(self.N_max, self.action_dim)
            
            # Apply mask
            action = action * masks[i][:, np.newaxis]
            
            actions.append(action)
        
        return np.array(actions)
    
    def store_experience(
        self,
        device_states: np.ndarray,
        global_feats: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        rewards: List[float],
        next_device_states: np.ndarray,
        next_global_feats: np.ndarray,
        next_masks: np.ndarray,
        done: bool
    ):
        """Store a transition in replay buffer."""
        states = []
        flat_actions = []
        next_states = []
        
        for i in range(self.n_agents):
            # Flatten state
            flat_devices = device_states[i].flatten()
            state = np.concatenate([flat_devices, global_feats[i]])
            states.append(state)
            
            # Flatten action
            flat_action = actions[i].flatten()
            flat_actions.append(flat_action)
            
            # Flatten next state
            flat_next_devices = next_device_states[i].flatten()
            next_state = np.concatenate([flat_next_devices, next_global_feats[i]])
            next_states.append(next_state)
        
        self.replay_buffer.push(states, flat_actions, rewards, next_states, done)
    
    def update(self) -> Dict[str, float]:
        """Update actor and critic networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )
        
        # Update critic
        with torch.no_grad():
            next_actions = [
                self.target_actors[i](next_states[i])
                for i in range(self.n_agents)
            ]
            target_q_values = self.target_critic(next_states, next_actions)
            target_q = [
                rewards[i] + self.gamma * (1 - dones) * target_q_values[i]
                for i in range(self.n_agents)
            ]
        
        current_q = self.critic(states, actions)
        
        critic_loss = sum(
            F.mse_loss(current_q[i], target_q[i])
            for i in range(self.n_agents)
        )
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        
        # Update actors
        actor_losses = []
        for i in range(self.n_agents):
            # Get current actions from all actors
            curr_actions = []
            for j in range(self.n_agents):
                if j == i:
                    curr_actions.append(self.actors[j](states[j]))
                else:
                    curr_actions.append(actions[j])
            
            # Compute Q-value for agent i
            q_values = self.critic(states, curr_actions)
            actor_loss = -q_values[i].mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()
            
            actor_losses.append(actor_loss.item())
        
        # Soft update targets
        self._soft_update()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': np.mean(actor_losses)
        }
    
    def _soft_update(self):
        """Soft update target networks."""
        for actor, target in zip(self.actors, self.target_actors):
            for p, tp in zip(actor.parameters(), target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'target_actors': [a.state_dict() for a in self.target_actors],
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, a in enumerate(self.actors):
            a.load_state_dict(checkpoint['actors'][i])
        for i, a in enumerate(self.target_actors):
            a.load_state_dict(checkpoint['target_actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])


if __name__ == "__main__":
    print("Testing FOMAAC...")
    
    maac = FOMAAC(
        n_agents=4,
        N_max=44,
        device_dim=6,
        global_dim=26,
        action_dim=5,
        device='cpu'
    )
    
    # Test action selection
    device_states = np.random.randn(4, 44, 6)
    global_feats = np.random.randn(4, 26)
    masks = np.random.randint(0, 2, (4, 44)).astype(float)
    
    actions = maac.select_actions(device_states, global_feats, masks)
    print(f"Actions shape: {actions.shape}")
    
    print(">>> FOMAAC test passed!")
