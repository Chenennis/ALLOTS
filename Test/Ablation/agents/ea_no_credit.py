"""
EA Agent WITHOUT Per-Device Credit Assignment (Ablation Variant)

This ablation variant uses UNIFORM weights instead of softmax advantage weights
for the actor loss.

CHANGE from full SALSA:
  INSTEAD OF: A_hat_d = q_d - b
              w_d = softmax(A_hat_d / tau_w)
              L_pi = -mean(sum(w_d * q_d) / |D|)
  
  USE:        w_d = 1.0 / |D|  # Uniform weight for all devices
              L_pi = -mean(q_d for d in D)  # Simple mean

Purpose: Test if per-device credit assignment accelerates post-churn adaptation.

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

from algorithms.EA.foea.actor import SetToSetActor
from algorithms.EA.foea.critic import TwinCritics
from algorithms.EA.foea.replay_buffer import ChurnAwareReplayBuffer

logger = logging.getLogger(__name__)


class EAAgentNoCredit:
    """
    EA Agent WITHOUT Per-Device Credit Assignment (Ablation Variant).
    
    KEY DIFFERENCE: Uses UNIFORM weights instead of softmax advantage weights.
    This means:
    - All active devices contribute equally to the actor loss
    - No advantage-based prioritization of devices
    - Standard SQDDPG-style loss: L = -mean(Q)
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
        advantage_tau: float = 1.0,  # Ignored in this variant
        buffer_capacity: int = 100000,
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
        # advantage_tau is accepted but NOT used in this ablation
        self.advantage_tau = advantage_tau
        self.device = device
        
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
        
        # Critic (same as original Pair-Set)
        self.critics = TwinCritics(
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
        
        logger.info(f"EAAgentNoCredit (Ablation) initialized on device {device}")
        logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        logger.info(f"Critics parameters: {sum(p.numel() for p in self.critics.parameters())}")
    
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
        
        # Update critics (TD-Consistent: uses mask_next, same as full SALSA)
        critic_loss, q1_value, q2_value = self._update_critics(
            manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done
        )
        self.critic_updates += 1
        
        # ABLATION: Delayed policy update WITHOUT per-device credit (uniform weights)
        actor_metrics = None
        if self.total_steps % self.policy_delay == 0:
            actor_metrics = self._update_actor_uniform(manager_id, g, X, mask)
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
    
    def _update_actor_uniform(self, manager_id, g, X, mask):
        """
        ABLATION: Update actor with UNIFORM weights (no per-device credit).
        
        Simple SQDDPG-style loss: L = -mean(Q)
        All devices contribute equally, no advantage-based weighting.
        """
        A_pi = self.actor(g, X, mask, manager_id)
        Q1, q_per_device = self.critics.Q1_forward(g, X, A_pi, mask, manager_id, return_per_device=True)
        
        # ABLATION: Simple loss - just maximize mean Q
        # This is equivalent to w_d = 1/|D| for all devices
        actor_loss = -Q1.squeeze(-1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Compute statistics for logging (advantage is computed but NOT used for loss)
        with torch.no_grad():
            active_mask = mask.bool()
            mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            baseline = (q_per_device * mask).sum(dim=-1, keepdim=True) / mask_sum
            per_device_advantage = q_per_device - baseline
            
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
