"""
FOAGILE: AGILE-based Multi-Agent RL for Flexible Orchestration

This implementation adapts AGILE's action-graph concept to handle device churn
in multi-agent environments. Key features:
- Graph Attention Network (GAT) to learn device relationships
- Per-device action generation with relational features
- Mask-aware training for variable device sets

Author: FOenv Team
Date: 2026-01-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) for learning device relationships.
    Based on "Graph Attention Networks" (Veličković et al., 2018)
    """
    
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Linear(dim_in, dim_out, bias=False)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(2 * dim_out, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self._attention = None
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            h: Node features [batch, num_nodes, dim_in]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            mask: Node mask [batch, num_nodes] (1=active, 0=inactive)
        
        Returns:
            out: Transformed features [batch, num_nodes, dim_out]
        """
        batch_size, num_nodes, _ = h.shape
        
        # Linear transformation
        z = self.W(h)  # [batch, num_nodes, dim_out]
        
        # Prepare attention input: concat([z_i, z_j]) for all pairs
        z_repeat = z.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # [batch, N, N, dim_out]
        z_repeat_t = z.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # [batch, N, N, dim_out]
        a_input = torch.cat([z_repeat, z_repeat_t], dim=-1)  # [batch, N, N, 2*dim_out]
        
        # Compute attention logits
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [batch, N, N]
        
        # Mask attention based on adjacency and active nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply mask if provided (mask out inactive nodes)
        if mask is not None:
            # Mask rows and columns for inactive nodes
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)  # [batch, N, N]
            attention = torch.where(mask_2d > 0, attention, zero_vec)
        
        # Normalize attention weights
        attention = F.softmax(attention, dim=-1)
        self._attention = attention.detach()
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attention = F.dropout(attention, self.dropout)
        
        # Aggregate neighbor features
        h_prime = torch.matmul(attention, z)  # [batch, num_nodes, dim_out]
        
        if self.concat:
            return F.elu(h_prime)
        return h_prime
    
    def get_attention(self):
        return self._attention


class ActionGraphEncoder(nn.Module):
    """
    Encodes device features using Graph Attention to capture device relationships.
    This is the core of AGILE's "action graph" concept adapted for devices.
    """
    
    def __init__(
        self,
        device_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.device_dim = device_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        
        # Pre-process: combine device state with global context
        self.input_dim = device_dim + global_dim
        
        # Pre-linear projection
        self.pre_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head GAT layer 1
        self.gat_heads = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, concat=True)
            for _ in range(num_heads)
        ])
        
        # GAT layer 2 (output)
        self.gat_out = GraphAttentionLayer(
            hidden_dim * num_heads, hidden_dim, dropout=dropout, concat=False
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Summary vector projection
        self.summary_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        device_states: torch.Tensor,
        global_feat: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            device_states: [batch, N_max, device_dim]
            global_feat: [batch, global_dim]
            mask: [batch, N_max]
        
        Returns:
            node_feat: [batch, N_max, hidden_dim] - per-device relational features
            summary_vec: [batch, hidden_dim] - aggregated graph summary
        """
        batch_size, N_max, _ = device_states.shape
        
        # Expand global features to each device
        g_expanded = global_feat.unsqueeze(1).expand(-1, N_max, -1)  # [batch, N_max, global_dim]
        
        # Concatenate device states with global context
        x = torch.cat([device_states, g_expanded], dim=-1)  # [batch, N_max, input_dim]
        
        # Pre-process
        x = self.pre_mlp(x)  # [batch, N_max, hidden_dim]
        
        # Build adjacency matrix (fully connected among active devices)
        adj = mask.unsqueeze(2) * mask.unsqueeze(1)  # [batch, N_max, N_max]
        # Remove self-loops
        eye = torch.eye(N_max, device=x.device).unsqueeze(0)
        adj = adj * (1 - eye)
        
        # Multi-head GAT layer 1
        x = torch.cat([head(x, adj, mask) for head in self.gat_heads], dim=-1)
        x = F.elu(x)
        
        # GAT layer 2
        node_feat = self.gat_out(x, adj, mask)
        node_feat = self.norm(node_feat)
        
        # Compute summary vector (mean pooling over active devices)
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [batch, 1]
        masked_feat = node_feat * mask.unsqueeze(-1)
        summary_vec = masked_feat.sum(dim=1) / mask_sum  # [batch, hidden_dim]
        summary_vec = self.summary_mlp(summary_vec)
        
        return node_feat, summary_vec


class AGILEActor(nn.Module):
    """
    AGILE-based Actor network that uses graph attention for action generation.
    """
    
    def __init__(
        self,
        N_max: int,
        device_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 2
    ):
        super().__init__()
        self.N_max = N_max
        self.action_dim = action_dim
        
        # Graph encoder
        self.graph_encoder = ActionGraphEncoder(
            device_dim=device_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Action head: combines node features with summary
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(
        self,
        device_states: torch.Tensor,
        global_feat: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate actions for all devices.
        
        Args:
            device_states: [batch, N_max, device_dim]
            global_feat: [batch, global_dim]
            mask: [batch, N_max]
        
        Returns:
            actions: [batch, N_max, action_dim]
        """
        # Get relational features
        node_feat, summary_vec = self.graph_encoder(device_states, global_feat, mask)
        
        # Expand summary to each device
        summary_expanded = summary_vec.unsqueeze(1).expand(-1, self.N_max, -1)
        
        # Combine node features with summary
        combined = torch.cat([node_feat, summary_expanded], dim=-1)
        
        # Generate actions
        actions = self.action_head(combined)
        
        # Apply mask
        actions = actions * mask.unsqueeze(-1)
        
        return actions


class AGILECritic(nn.Module):
    """
    Critic network for AGILE that evaluates state-action pairs.
    Uses graph attention to understand device relationships.
    """
    
    def __init__(
        self,
        N_max: int,
        device_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 2
    ):
        super().__init__()
        self.N_max = N_max
        
        # State encoder with graph attention
        self.graph_encoder = ActionGraphEncoder(
            device_dim=device_dim + action_dim,  # Include actions
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        device_states: torch.Tensor,
        global_feat: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate state-action pair.
        
        Args:
            device_states: [batch, N_max, device_dim]
            global_feat: [batch, global_dim]
            actions: [batch, N_max, action_dim]
            mask: [batch, N_max]
        
        Returns:
            Q-value: [batch, 1]
        """
        # Concatenate states and actions
        sa = torch.cat([device_states, actions], dim=-1)
        
        # Get graph summary
        _, summary_vec = self.graph_encoder(sa, global_feat, mask)
        
        # Compute Q-value
        q = self.value_head(summary_vec)
        
        return q


class ReplayBuffer:
    """Experience replay buffer for AGILE."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        device_states: np.ndarray,
        global_feat: np.ndarray,
        mask: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_device_states: np.ndarray,
        next_global_feat: np.ndarray,
        next_mask: np.ndarray,
        done: bool
    ):
        self.buffer.append((
            device_states, global_feat, mask, actions,
            reward, next_device_states, next_global_feat, next_mask, done
        ))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        device_states = np.stack([t[0] for t in batch])
        global_feat = np.stack([t[1] for t in batch])
        mask = np.stack([t[2] for t in batch])
        actions = np.stack([t[3] for t in batch])
        rewards = np.array([t[4] for t in batch])
        next_device_states = np.stack([t[5] for t in batch])
        next_global_feat = np.stack([t[6] for t in batch])
        next_mask = np.stack([t[7] for t in batch])
        dones = np.array([t[8] for t in batch])
        
        return (device_states, global_feat, mask, actions, rewards,
                next_device_states, next_global_feat, next_mask, dones)
    
    def __len__(self):
        return len(self.buffer)


class FOAGILE:
    """
    FOAGILE: AGILE-based Multi-Agent RL algorithm for FOgym.
    
    Uses Graph Attention Networks to learn device relationships
    and generate coordinated actions.
    """
    
    def __init__(
        self,
        n_agents: int,
        N_max: int,
        device_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 2,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.1,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        device: str = "cpu"
    ):
        self.n_agents = n_agents
        self.N_max = N_max
        self.device_dim = device_dim
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Create actor and critic for each agent
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        self.replay_buffers = []
        
        for i in range(n_agents):
            # Actor
            actor = AGILEActor(
                N_max=N_max,
                device_dim=device_dim,
                global_dim=global_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(self.device)
            actor_target = AGILEActor(
                N_max=N_max,
                device_dim=device_dim,
                global_dim=global_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            # Critic
            critic = AGILECritic(
                N_max=N_max,
                device_dim=device_dim,
                global_dim=global_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(self.device)
            critic_target = AGILECritic(
                N_max=N_max,
                device_dim=device_dim,
                global_dim=global_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(self.device)
            critic_target.load_state_dict(critic.state_dict())
            
            # Optimizers
            actor_optim = optim.Adam(actor.parameters(), lr=lr_actor)
            critic_optim = optim.Adam(critic.parameters(), lr=lr_critic)
            
            # Buffer
            buffer = ReplayBuffer(capacity=buffer_capacity)
            
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.actor_optimizers.append(actor_optim)
            self.critic_optimizers.append(critic_optim)
            self.replay_buffers.append(buffer)
    
    def select_actions(
        self,
        device_states: np.ndarray,  # [n_agents, N_max, device_dim]
        global_feats: np.ndarray,   # [n_agents, global_dim]
        masks: np.ndarray,          # [n_agents, N_max]
        add_noise: bool = True
    ) -> np.ndarray:
        """Select actions for all agents."""
        actions = []
        
        for i in range(self.n_agents):
            self.actors[i].eval()
            
            with torch.no_grad():
                ds = torch.FloatTensor(device_states[i:i+1]).to(self.device)
                gf = torch.FloatTensor(global_feats[i:i+1]).to(self.device)
                m = torch.FloatTensor(masks[i:i+1]).to(self.device)
                
                action = self.actors[i](ds, gf, m).cpu().numpy()[0]
            
            if add_noise:
                noise = self.noise_scale * np.random.randn(*action.shape)
                action = np.clip(action + noise, -1, 1)
                # Apply mask
                action = action * masks[i:i+1].T
            
            actions.append(action)
        
        return np.array(actions)  # [n_agents, N_max, action_dim]
    
    def store_experience(
        self,
        agent_idx: int,
        device_states: np.ndarray,
        global_feat: np.ndarray,
        mask: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_device_states: np.ndarray,
        next_global_feat: np.ndarray,
        next_mask: np.ndarray,
        done: bool
    ):
        """Store experience for a specific agent."""
        self.replay_buffers[agent_idx].push(
            device_states, global_feat, mask, actions,
            reward, next_device_states, next_global_feat, next_mask, done
        )
    
    def update(self) -> Dict[str, float]:
        """Update all agents."""
        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0
        }
        
        for i in range(self.n_agents):
            if len(self.replay_buffers[i]) < self.batch_size:
                continue
            
            # Sample batch
            batch = self.replay_buffers[i].sample(self.batch_size)
            (device_states, global_feat, mask, actions, rewards,
             next_device_states, next_global_feat, next_mask, dones) = batch
            
            # Convert to tensors
            ds = torch.FloatTensor(device_states).to(self.device)
            gf = torch.FloatTensor(global_feat).to(self.device)
            m = torch.FloatTensor(mask).to(self.device)
            a = torch.FloatTensor(actions).to(self.device)
            r = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
            next_ds = torch.FloatTensor(next_device_states).to(self.device)
            next_gf = torch.FloatTensor(next_global_feat).to(self.device)
            next_m = torch.FloatTensor(next_mask).to(self.device)
            d = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
            
            # Update critic
            self.critics[i].train()
            with torch.no_grad():
                next_actions = self.actor_targets[i](next_ds, next_gf, next_m)
                target_q = self.critic_targets[i](next_ds, next_gf, next_actions, next_m)
                target_q = r + self.gamma * (1 - d) * target_q
            
            current_q = self.critics[i](ds, gf, a, m)
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_optimizers[i].step()
            
            # Update actor
            self.actors[i].train()
            actor_actions = self.actors[i](ds, gf, m)
            actor_loss = -self.critics[i](ds, gf, actor_actions, m).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            
            # Soft update targets
            self._soft_update(self.actors[i], self.actor_targets[i])
            self._soft_update(self.critics[i], self.critic_targets[i])
            
            metrics['actor_loss'] += actor_loss.item()
            metrics['critic_loss'] += critic_loss.item()
        
        metrics['actor_loss'] /= max(1, self.n_agents)
        metrics['critic_loss'] /= max(1, self.n_agents)
        
        return metrics
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'critics': [c.state_dict() for c in self.critics],
            'actor_targets': [at.state_dict() for at in self.actor_targets],
            'critic_targets': [ct.state_dict() for ct in self.critic_targets],
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.actor_targets[i].load_state_dict(checkpoint['actor_targets'][i])
            self.critic_targets[i].load_state_dict(checkpoint['critic_targets'][i])
