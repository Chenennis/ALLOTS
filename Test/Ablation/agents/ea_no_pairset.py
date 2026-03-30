"""
EA Agent WITHOUT Pair-Set Critic (Ablation Variant)

This ablation variant replaces the paired state-action token encoding with
separate state and action encoders that are combined afterward.

CHANGE from full SALSA:
  INSTEAD OF: p_t_d = concat(x_t_d, a_t_d); v_t_d = psi_Q(concat(p_t_d, g_t))
  USE:        state_embed = psi_state(concat(x_t_d, g_t))
              action_embed = psi_action(a_t_d)
              v_t_d = psi_combine(concat(state_embed, action_embed))

Purpose: Test if explicit per-device state-action pairing is necessary for
         assignment-sensitive learning.

Author: FOenv Team
Date: 2026-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import copy
import logging

from algorithms.EA.foea.actor import SetToSetActor, masked_mean
from algorithms.EA.foea.replay_buffer import ChurnAwareReplayBuffer

logger = logging.getLogger(__name__)


# ============================================================================
# ABLATION: Separate State-Action Critic (No Pair-Set)
# ============================================================================

class SeparateCritic(nn.Module):
    """
    Critic with SEPARATE state and action encoding (NO Pair-Set).
    
    Architecture:
        1. State Encoder: psi_state([x_j, g, emb]) → state_embed_j
        2. Action Encoder: psi_action(a_j) → action_embed_j
        3. Combine: psi_combine([state_embed_j, action_embed_j]) → v_j
        4. Pooling: v_bar = masked_mean(v, mask)
        5. Per-device Q Head: ρ_dev(v_j) → q_j
        6. Global Q Head: ρ_glob([v_bar, g]) → q_glob
        7. Final Q: masked_mean(q_j, mask) + q_glob
    
    KEY DIFFERENCE: State and action are encoded SEPARATELY first,
    then combined. This removes the explicit per-device state-action pairing.
    """
    
    def __init__(
        self,
        x_dim: int = 6,
        g_dim: int = 50,
        p: int = 5,
        N_max: int = 60,
        num_managers: int = 10,
        emb_dim: int = 16,
        token_dim: int = 128,
        hidden_dim: int = 256,
        activation: str = 'relu',
    ):
        super(SeparateCritic, self).__init__()
        
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.N_max = N_max
        self.num_managers = num_managers
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Manager ID embedding
        self.manager_embedding = nn.Embedding(num_managers, emb_dim)
        
        # ABLATION: Separate State Encoder psi_state: [x_j, g, emb] → state_embed
        state_input_dim = x_dim + g_dim + emb_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, token_dim // 2),
            self.activation,
        )
        
        # ABLATION: Separate Action Encoder psi_action: a_j → action_embed
        self.action_encoder = nn.Sequential(
            nn.Linear(p, hidden_dim // 2),
            self.activation,
            nn.Linear(hidden_dim // 2, token_dim // 2),
            self.activation,
        )
        
        # ABLATION: Combine Encoder psi_combine: [state_embed, action_embed] → v_j
        combine_input_dim = token_dim  # token_dim/2 + token_dim/2
        self.combine_encoder = nn.Sequential(
            nn.Linear(combine_input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, token_dim),
            self.activation,
        )
        
        # Per-device Q Head ρ_dev: v_j → q_j
        self.device_q_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )
        
        # Global Q Head ρ_glob: [v_bar, g] → q_glob
        glob_input_dim = token_dim + g_dim
        self.global_q_head = nn.Sequential(
            nn.Linear(glob_input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        A: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor,
        return_per_device: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with SEPARATE state-action encoding.
        """
        B = g.size(0)
        N_max = X.size(1)
        
        # Get manager embedding
        manager_emb = self.manager_embedding(manager_id)  # [B, emb_dim]
        
        # Expand g and manager_emb to all devices
        g_expanded = g.unsqueeze(1).expand(B, N_max, self.g_dim)
        manager_emb_expanded = manager_emb.unsqueeze(1).expand(B, N_max, self.emb_dim)
        
        # ABLATION: Encode state and action SEPARATELY
        # State encoding: [x_j, g, emb]
        state_input = torch.cat([X, g_expanded, manager_emb_expanded], dim=-1)
        state_embed = self.state_encoder(state_input)  # [B, N_max, token_dim/2]
        
        # Action encoding: a_j
        action_embed = self.action_encoder(A)  # [B, N_max, token_dim/2]
        
        # Combine state and action embeddings
        combined = torch.cat([state_embed, action_embed], dim=-1)  # [B, N_max, token_dim]
        v = self.combine_encoder(combined)  # [B, N_max, token_dim]
        
        # Pooling: v_bar = masked_mean(v, mask)
        v_bar = masked_mean(v, mask, dim=1, keepdim=False)  # [B, token_dim]
        
        # Per-device Q values
        q_dev = self.device_q_head(v)  # [B, N_max, 1]
        q_dev = q_dev.squeeze(-1)  # [B, N_max]
        
        # Masked mean of per-device Q values
        q_dev_expanded = q_dev.unsqueeze(-1)
        q_dev_mean = masked_mean(q_dev_expanded, mask, dim=1, keepdim=True)  # [B, 1]
        
        # Global Q value
        glob_input = torch.cat([v_bar, g], dim=-1)
        q_glob = self.global_q_head(glob_input)  # [B, 1]
        
        # Final Q value
        Q = q_dev_mean + q_glob  # [B, 1]
        
        if return_per_device:
            return Q, q_dev
        return Q


class TwinSeparateCritics(nn.Module):
    """Twin Separate Critics (Q1 and Q2) for MATD3-style training."""
    
    def __init__(self, **kwargs):
        super(TwinSeparateCritics, self).__init__()
        self.Q1 = SeparateCritic(**kwargs)
        self.Q2 = SeparateCritic(**kwargs)
    
    def forward(self, g, X, A, mask, manager_id) -> Tuple[torch.Tensor, torch.Tensor]:
        Q1 = self.Q1(g, X, A, mask, manager_id)
        Q2 = self.Q2(g, X, A, mask, manager_id)
        return Q1, Q2
    
    def Q1_forward(self, g, X, A, mask, manager_id, return_per_device=False):
        return self.Q1(g, X, A, mask, manager_id, return_per_device=return_per_device)
    
    def min_Q(self, g, X, A, mask, manager_id) -> torch.Tensor:
        Q1, Q2 = self.forward(g, X, A, mask, manager_id)
        return torch.min(Q1, Q2)


# ============================================================================
# EA Agent with NO Pair-Set Critic
# ============================================================================

class EAAgentNoPairSet:
    """
    EA Agent WITHOUT Pair-Set Critic (Ablation Variant).
    
    Uses SeparateCritic instead of PairSetCritic.
    All other components (Actor, TD-Consistent, Per-Device Credit) remain the same.
    """
    
    def __init__(
        self,
        x_dim: int = 6,
        g_dim: int = 50,
        p: int = 5,
        N_max: int = 60,
        num_managers: int = 4,
        emb_dim: int = 16,
        token_dim: int = 128,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        policy_delay: int = 2,
        noise_scale: float = 0.1,
        noise_clip: float = 0.2,
        advantage_tau: float = 1.0,
        buffer_capacity: int = 100000,
        credit_warmup_start: int = 0,
        credit_warmup_end: int = 0,
        credit_max_weight: float = 0.15,
        device: str = 'cpu',
    ):
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.N_max = N_max
        self.num_managers = num_managers
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        self.advantage_tau = advantage_tau
        self.device = device

        self.current_episode = 0
        self.credit_warmup_start = credit_warmup_start
        self.credit_warmup_end = credit_warmup_end
        self.credit_max_weight = credit_max_weight
        
        # Create actor (same as original)
        self.actor = SetToSetActor(
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            N_max=N_max,
            num_managers=num_managers,
            emb_dim=emb_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        
        # ABLATION: Use SeparateCritic instead of PairSetCritic
        self.critics = TwinSeparateCritics(
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            N_max=N_max,
            num_managers=num_managers,
            emb_dim=emb_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        
        # Create target networks
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critics_target = copy.deepcopy(self.critics).to(device)
        
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critics_target.parameters():
            param.requires_grad = False
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=lr_critic)
        
        self.replay_buffer = ChurnAwareReplayBuffer(
            capacity=buffer_capacity,
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            N_max=N_max,
        )
        
        self.total_steps = 0
        self.actor_updates = 0
        self.critic_updates = 0
        
        logger.info(f"EAAgentNoPairSet (Ablation) initialized on device {device}")
        logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        logger.info(f"Critics parameters: {sum(p.numel() for p in self.critics.parameters())}")

    def set_episode(self, episode: int):
        self.current_episode = episode

    def _get_credit_weight(self) -> float:
        if self.credit_warmup_start >= self.credit_warmup_end:
            return self.credit_max_weight
        if self.current_episode < self.credit_warmup_start:
            return 0.0
        elif self.current_episode < self.credit_warmup_end:
            progress = (self.current_episode - self.credit_warmup_start) / \
                       (self.credit_warmup_end - self.credit_warmup_start)
            return self.credit_max_weight * progress
        else:
            return self.credit_max_weight

    def select_action(self, g, X, mask, manager_id, explore=True):
        with torch.no_grad():
            g_t = torch.FloatTensor(g).unsqueeze(0).to(self.device)
            X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            manager_id_t = torch.LongTensor([manager_id]).to(self.device)
            
            if explore:
                A_t = self.actor.get_actions(g_t, X_t, mask_t, manager_id_t, noise_scale=self.noise_scale)
            else:
                A_t = self.actor(g_t, X_t, mask_t, manager_id_t)
            
            A = A_t.squeeze(0).cpu().numpy()
        return A
    
    def store_transition(self, manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done):
        self.replay_buffer.add(
            manager_id=manager_id, g=g, X=X, mask=mask, A=A, r=r,
            g_next=g_next, X_next=X_next, mask_next=mask_next, done=done,
        )
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}
        
        batch = self.replay_buffer.sample(batch_size, device=self.device)
        
        manager_id = batch['manager_id']
        g, X, mask, A = batch['g'], batch['X'], batch['mask'], batch['A']
        r, g_next, X_next = batch['r'], batch['g_next'], batch['X_next']
        mask_next, done = batch['mask_next'], batch['done']
        
        # Update critics (TD-Consistent: uses mask_next)
        critic_loss, q1_value, q2_value = self._update_critics(
            manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done
        )
        self.critic_updates += 1
        
        # Delayed policy update (Per-Device Credit: uses softmax weights)
        actor_metrics = None
        if self.total_steps % self.policy_delay == 0:
            actor_metrics = self._update_actor(manager_id, g, X, mask)
            self._soft_update_targets()
            self.actor_updates += 1
        
        self.total_steps += 1
        
        metrics = {
            'critic_loss': critic_loss, 'q1_value': q1_value, 'q2_value': q2_value,
            'actor_updates': self.actor_updates, 'critic_updates': self.critic_updates,
        }
        if actor_metrics:
            metrics.update({k: v for k, v in actor_metrics.items()})
        return metrics
    
    def _update_critics(self, manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done):
        """TD-Consistent bootstrapping (same as full SALSA)."""
        with torch.no_grad():
            A_next = self.actor_target(g_next, X_next, mask_next, manager_id)
            noise = (torch.randn_like(A_next) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            noise = noise * mask_next.unsqueeze(-1)
            A_next = (A_next + noise).clamp(-1.0, 1.0)
            A_next = A_next * mask_next.unsqueeze(-1)
            
            target_Q = self.critics_target.min_Q(g_next, X_next, A_next, mask_next, manager_id)
            y = r + self.gamma * (1 - done) * target_Q
        
        Q1, Q2 = self.critics(g, X, A, mask, manager_id)
        critic_loss = nn.MSELoss()(Q1, y) + nn.MSELoss()(Q2, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item(), Q1.mean().item(), Q2.mean().item()
    
    def _update_actor(self, manager_id, g, X, mask):
        """Per-Device Credit Assignment (same as full SALSA)."""
        A_pi = self.actor(g, X, mask, manager_id)
        Q1, q_per_device = self.critics.Q1_forward(g, X, A_pi, mask, manager_id, return_per_device=True)
        
        primary_loss = -Q1.squeeze(-1).mean()
        
        active_mask = mask.bool()
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        baseline = (q_per_device * mask).sum(dim=-1, keepdim=True) / mask_sum
        per_device_advantage = q_per_device - baseline
        
        with torch.no_grad():
            adv_for_norm = per_device_advantage.clone()
            adv_for_norm[~active_mask] = 0
            adv_std = adv_for_norm[active_mask].std().clamp(min=1e-6) if active_mask.any() else torch.tensor(1.0, device=mask.device)
        
        normalized_advantage = per_device_advantage / adv_std
        masked_advantage = normalized_advantage.clone()
        masked_advantage[~active_mask] = -1e9
        
        weights = F.softmax(masked_advantage / self.advantage_tau, dim=-1)
        weights = weights * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        weighted_q = (weights * q_per_device * mask).sum(dim=-1)
        uniform_q = (q_per_device * mask).sum(dim=-1) / mask_sum.squeeze(-1)
        advantage_reg = weighted_q - uniform_q
        
        credit_weight = self._get_credit_weight()
        actor_loss = (1.0 - credit_weight) * primary_loss - credit_weight * advantage_reg.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        with torch.no_grad():
            active_advantages = per_device_advantage[active_mask]
            mean_adv = active_advantages.mean().item() if active_advantages.numel() > 0 else 0.0
            std_adv = active_advantages.std().item() if active_advantages.numel() > 1 else 0.0
        
        return {
            'actor_loss': actor_loss.item(),
            'mean_advantage': mean_adv,
            'advantage_std': std_adv,
        }
    
    def _soft_update_targets(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critics_state_dict': self.critics.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critics_target_state_dict': self.critics_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'actor_updates': self.actor_updates,
            'critic_updates': self.critic_updates,
        }, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critics.load_state_dict(checkpoint['critics_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critics_target.load_state_dict(checkpoint['critics_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.actor_updates = checkpoint['actor_updates']
        self.critic_updates = checkpoint['critic_updates']
        logger.info(f"Agent loaded from {filepath}")
