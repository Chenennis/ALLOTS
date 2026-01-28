"""
EA Agent for Environment-Adaptive Multi-Agent RL

This module implements the EA agent that wraps actor, critics, and replay buffer,
providing a unified interface for training and evaluation.

Author: FOenv Team
Date: 2026-01-12
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


class EAAgent:
    """
    Environment-Adaptive Agent
    
    Integrates:
        - Set-to-Set Actor
        - Twin Pair-Set Critics
        - Churn-Aware Replay Buffer
        - MATD3-style training loop
        - Churn-consistent TD targets
    """
    
    def __init__(
        self,
        x_dim: int = 6,
        g_dim: int = 50,
        p: int = 5,
        N_max: int = 60,
        num_managers: int = 4,
        # Network architecture
        emb_dim: int = 16,
        token_dim: int = 128,
        hidden_dim: int = 256,
        # Training hyperparameters
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        policy_delay: int = 2,
        noise_scale: float = 0.1,
        noise_clip: float = 0.2,
        # Per-device advantage weighting
        advantage_tau: float = 1.0,  # Temperature for softmax advantage weighting
        # Buffer
        buffer_capacity: int = 100000,
        # Device
        device: str = 'cpu',
    ):
        """
        Initialize EA Agent
        
        Args:
            x_dim: Device state dimension
            g_dim: Global feature dimension
            p: Action dimension per device
            N_max: Maximum number of devices
            num_managers: Number of managers
            emb_dim: Manager embedding dimension
            token_dim: Token dimension
            hidden_dim: Hidden dimension
            gamma: Discount factor
            tau: Soft update rate
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            policy_delay: Policy update frequency
            noise_scale: Exploration noise scale
            noise_clip: Target policy smoothing noise clip
            advantage_tau: Temperature for softmax advantage weighting (lower = sharper)
            buffer_capacity: Replay buffer capacity
            device: PyTorch device
        """
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
        self.advantage_tau = advantage_tau  # Temperature for softmax advantage weighting
        self.device = device
        
        # Create actor and critics
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
        
        # Freeze target networks
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critics_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ChurnAwareReplayBuffer(
            capacity=buffer_capacity,
            x_dim=x_dim,
            g_dim=g_dim,
            p=p,
            N_max=N_max,
        )
        
        # Training state
        self.total_steps = 0
        self.actor_updates = 0
        self.critic_updates = 0
        
        # Progressive Credit warmup settings
        self.current_episode = 0
        self.credit_warmup_start = 100     # Start enabling Credit at episode 100
        self.credit_warmup_end = 300       # Full Credit at episode 300
        self.credit_max_weight = 0.05      # Maximum Credit weight (reduced from 0.15)
        
        logger.info(f"EAAgent initialized on device {device}")
        logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        logger.info(f"Critics parameters: {sum(p.numel() for p in self.critics.parameters())}")
    
    def select_action(
        self,
        g: np.ndarray,
        X: np.ndarray,
        mask: np.ndarray,
        manager_id: int,
        explore: bool = True
    ) -> np.ndarray:
        """
        Select action for a single manager
        
        Args:
            g: [g_dim] Global features
            X: [N_max, x_dim] Device states
            mask: [N_max] Active device mask
            manager_id: Manager ID
            explore: Whether to add exploration noise
        
        Returns:
            A: [N_max, p] Actions (masked)
        """
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            g_t = torch.FloatTensor(g).unsqueeze(0).to(self.device)  # [1, g_dim]
            X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # [1, N_max, x_dim]
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)  # [1, N_max]
            manager_id_t = torch.LongTensor([manager_id]).to(self.device)  # [1]
            
            # Get actions
            if explore:
                A_t = self.actor.get_actions(g_t, X_t, mask_t, manager_id_t, noise_scale=self.noise_scale)
            else:
                A_t = self.actor(g_t, X_t, mask_t, manager_id_t)
            
            # Convert back to numpy
            A = A_t.squeeze(0).cpu().numpy()  # [N_max, p]
        
        return A
    
    def set_episode(self, episode: int):
        """
        Set current episode for progressive Credit warmup
        
        Args:
            episode: Current episode number (0-indexed)
        """
        self.current_episode = episode
    
    def _get_credit_weight(self) -> float:
        """
        Get progressive Credit weight based on current episode
        
        Progressive schedule:
        - Episode 0-99: weight = 0.0 (no Credit, let Critic converge)
        - Episode 100-299: weight = linear ramp up to 0.05
        - Episode 300+: weight = 0.05 (stable Credit)
        
        Returns:
            Credit weight for actor loss
        """
        if self.current_episode < self.credit_warmup_start:
            return 0.0
        elif self.current_episode < self.credit_warmup_end:
            # Linear ramp from 0 to credit_max_weight
            progress = (self.current_episode - self.credit_warmup_start) / \
                       (self.credit_warmup_end - self.credit_warmup_start)
            return self.credit_max_weight * progress
        else:
            return self.credit_max_weight
    
    def store_transition(
        self,
        manager_id: int,
        g: np.ndarray,
        X: np.ndarray,
        mask: np.ndarray,
        A: np.ndarray,
        r: float,
        g_next: np.ndarray,
        X_next: np.ndarray,
        mask_next: np.ndarray,
        done: bool
    ):
        """Store a transition in replay buffer"""
        self.replay_buffer.add(
            manager_id=manager_id,
            g=g,
            X=X,
            mask=mask,
            A=A,
            r=r,
            g_next=g_next,
            X_next=X_next,
            mask_next=mask_next,
            done=done,
        )
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Perform one update step
        
        Args:
            batch_size: Batch size for sampling
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size, device=self.device)
        
        manager_id = batch['manager_id']
        g = batch['g']
        X = batch['X']
        mask = batch['mask']
        A = batch['A']
        r = batch['r']
        g_next = batch['g_next']
        X_next = batch['X_next']
        mask_next = batch['mask_next']  # KEY: Use next mask for churn consistency!
        done = batch['done']
        
        # Update critics
        critic_loss, q1_value, q2_value = self._update_critics(
            manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done
        )
        
        self.critic_updates += 1
        
        # Delayed policy update
        actor_metrics = None
        if self.total_steps % self.policy_delay == 0:
            actor_metrics = self._update_actor(manager_id, g, X, mask)
            self._soft_update_targets()
            self.actor_updates += 1
        
        self.total_steps += 1
        
        # Return metrics
        metrics = {
            'critic_loss': critic_loss,
            'q1_value': q1_value,
            'q2_value': q2_value,
            'actor_updates': self.actor_updates,
            'critic_updates': self.critic_updates,
        }
        
        if actor_metrics is not None:
            metrics['actor_loss'] = actor_metrics['actor_loss']
            metrics['mean_advantage'] = actor_metrics['mean_advantage']
            metrics['advantage_std'] = actor_metrics['advantage_std']
        
        return metrics
    
    def _update_critics(
        self,
        manager_id: torch.Tensor,
        g: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        A: torch.Tensor,
        r: torch.Tensor,
        g_next: torch.Tensor,
        X_next: torch.Tensor,
        mask_next: torch.Tensor,
        done: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Update critic networks with Smart TD Target (方案D)
        
        Key Innovation: Intelligent TD target that uses:
        - Normal mask_next when no churn detected (theoretically correct)
        - Conservative blend when churn detected (stable learning)
        
        Returns:
            (critic_loss, q1_value, q2_value)
        """
        with torch.no_grad():
            # ===== SMART TD TARGET (方案D) =====
            # Detect churn: compare device counts between mask and mask_next
            mask_sum = mask.sum(dim=-1)  # [B]
            mask_next_sum = mask_next.sum(dim=-1)  # [B]
            churn_detected = (mask_sum != mask_next_sum)  # [B] - True where churn happened
            
            # Compute actions for BOTH masks
            # Action with current mask (conservative)
            A_next_curr = self.actor_target(g_next, X_next, mask, manager_id)
            noise_curr = (torch.randn_like(A_next_curr) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            noise_curr = noise_curr * mask.unsqueeze(-1)
            A_next_curr = (A_next_curr + noise_curr).clamp(-1.0, 1.0)
            A_next_curr = A_next_curr * mask.unsqueeze(-1)
            
            # Action with next mask (theoretically correct)
            A_next_real = self.actor_target(g_next, X_next, mask_next, manager_id)
            noise_real = (torch.randn_like(A_next_real) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            noise_real = noise_real * mask_next.unsqueeze(-1)
            A_next_real = (A_next_real + noise_real).clamp(-1.0, 1.0)
            A_next_real = A_next_real * mask_next.unsqueeze(-1)
            
            # Compute Q values for both
            Q_target_curr = self.critics_target.min_Q(g_next, X_next, A_next_curr, mask, manager_id)
            Q_target_real = self.critics_target.min_Q(g_next, X_next, A_next_real, mask_next, manager_id)
            
            # Smart TD target selection:
            # - No churn: use Q_target_real (theoretically correct)
            # - Churn detected: use conservative blend
            churn_blend_weight = 0.7  # Weight for current mask when churn detected
            
            # Expand churn_detected for broadcasting [B] -> [B, 1]
            churn_mask = churn_detected.unsqueeze(-1).float()
            
            # Blend: churn → 0.7*Q_curr + 0.3*Q_real; no_churn → Q_real
            target_Q = churn_mask * (churn_blend_weight * Q_target_curr + (1 - churn_blend_weight) * Q_target_real) \
                     + (1 - churn_mask) * Q_target_real
            
            # Compute TD target
            y = r + self.gamma * (1 - done) * target_Q
        
        # Get current Q estimates
        Q1, Q2 = self.critics(g, X, A, mask, manager_id)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(Q1, y) + nn.MSELoss()(Q2, y)
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Log churn statistics (for debugging)
        churn_ratio = churn_detected.float().mean().item()
        
        return critic_loss.item(), Q1.mean().item(), Q2.mean().item()
    
    def _update_actor(
        self,
        manager_id: torch.Tensor,
        g: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor
    ) -> dict:
        """
        Update actor network with SQDDPG-style Q maximization + per-device advantage weighting
        
        Inspired by SQDDPG's effective simple loss (-Q.mean()), this implementation:
        1. PRIMARY: Maximize overall Q1 (SQDDPG style - proven effective)
        2. SECONDARY: Add per-device advantage weighting for fine-grained attention
        
        KEY INNOVATION: Per-device advantage weighting encourages the actor to focus
        more on devices with higher improvement potential, while the main Q maximization
        ensures overall policy improvement.
        
        Loss = -Q1.mean() + lambda * advantage_weighted_regularization
        
        Returns:
            dict with actor_loss, mean_advantage, advantage_std
        """
        # Compute actor actions
        A_pi = self.actor(g, X, mask, manager_id)
        
        # Get both aggregated Q and per-device Q values
        Q1, q_per_device = self.critics.Q1_forward(
            g, X, A_pi, mask, manager_id, return_per_device=True
        )  # Q1: [B, 1], q_per_device: [B, N_max]
        
        # ==== PRIMARY LOSS: SQDDPG-style Q maximization ====
        # This is the main driver of learning - maximize overall Q value
        primary_loss = -Q1.squeeze(-1).mean()
        
        # ==== SECONDARY: Per-device advantage weighting ====
        # Compute per-device advantage: A_d = q_d - baseline
        active_mask = mask.bool()
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        baseline = (q_per_device * mask).sum(dim=-1, keepdim=True) / mask_sum  # [B, 1]
        
        per_device_advantage = q_per_device - baseline  # [B, N_max]
        
        # Normalize advantage for computing weights (ensures softmax stability)
        with torch.no_grad():
            adv_for_norm = per_device_advantage.clone()
            adv_for_norm[~active_mask] = 0
            adv_std = adv_for_norm[active_mask].std().clamp(min=1e-6) if active_mask.any() else torch.tensor(1.0, device=mask.device)
        
        normalized_advantage = per_device_advantage / adv_std  # [B, N_max]
        
        # Apply mask to advantage (set inactive devices to large negative for softmax)
        masked_advantage = normalized_advantage.clone()
        masked_advantage[~active_mask] = -1e9  # Large negative for inactive devices
        
        # Compute softmax weights over device dimension: w_d = softmax(A_d / tau)
        weights = F.softmax(masked_advantage / self.advantage_tau, dim=-1)  # [B, N_max]
        weights = weights * mask  # Zero out inactive devices
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # Renormalize
        
        # Per-device weighted Q (focuses on high-advantage devices)
        weighted_q = (weights * q_per_device * mask).sum(dim=-1)  # [B]
        
        # Advantage-weighted regularization term
        # This term encourages actor to improve actions for high-advantage devices
        # By comparing weighted_q to uniform-weighted q (baseline), we get a signal
        # that's positive when the actor is doing well on high-advantage devices
        uniform_q = (q_per_device * mask).sum(dim=-1) / mask_sum.squeeze(-1)  # [B]
        advantage_reg = weighted_q - uniform_q  # [B] - positive when focusing on good devices
        
        # Get progressive Credit weight
        credit_weight = self._get_credit_weight()
        
        # Combined loss:
        # - primary_loss (1-credit_weight): SQDDPG-style maximize Q1 (main learning signal)
        # - secondary term (credit_weight): encourage improvement on high-advantage devices
        # Progressive warmup: credit_weight = 0 initially, ramps up to 0.05 after episode 300
        actor_loss = (1.0 - credit_weight) * primary_loss - credit_weight * advantage_reg.mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Compute statistics for logging
        with torch.no_grad():
            active_advantages = per_device_advantage[active_mask]
            mean_adv = active_advantages.mean().item() if active_advantages.numel() > 0 else 0.0
            std_adv = active_advantages.std().item() if active_advantages.numel() > 1 else 0.0
        
        return {
            'actor_loss': actor_loss.item(),
            'mean_advantage': mean_adv,
            'advantage_std': std_adv,
            'q1_mean': Q1.mean().item(),
            'weighted_q': weighted_q.mean().item(),
            'credit_weight': credit_weight,  # Track progressive Credit weight
        }
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save agent state"""
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
        """Load agent state"""
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


def test_ea_agent():
    """Test EA Agent"""
    print("=== Testing EA Agent ===\n")
    
    # Parameters
    x_dim = 6
    g_dim = 50
    p = 5
    N_max = 60
    num_managers = 4
    
    # Create agent
    agent = EAAgent(
        x_dim=x_dim,
        g_dim=g_dim,
        p=p,
        N_max=N_max,
        num_managers=num_managers,
        buffer_capacity=1000,
        device='cpu',
    )
    
    print(f"Agent created")
    print(f"Buffer size: {len(agent.replay_buffer)}\n")
    
    # Test action selection
    print("Testing action selection...")
    g = np.random.randn(g_dim)
    X = np.random.randn(N_max, x_dim)
    mask = np.zeros(N_max)
    mask[:30] = 1  # 30 active devices
    manager_id = 0
    
    A_explore = agent.select_action(g, X, mask, manager_id, explore=True)
    A_greedy = agent.select_action(g, X, mask, manager_id, explore=False)
    
    print(f"  Explored action shape: {A_explore.shape}")
    print(f"  Greedy action shape: {A_greedy.shape}")
    print(f"  Action difference: {np.abs(A_explore - A_greedy).mean():.4f}")
    print(f"  Active action norm: {np.linalg.norm(A_explore[:30]):.3f}")
    print(f"  Inactive action norm: {np.linalg.norm(A_explore[30:]):.6f} (should be ~0)")
    
    # Store some transitions
    print(f"\nStoring 500 transitions...")
    for i in range(500):
        manager_id = i % 4
        g = np.random.randn(g_dim)
        X = np.random.randn(N_max, x_dim)
        n_active = np.random.randint(10, 50)
        mask = np.zeros(N_max)
        mask[:n_active] = 1
        
        A = np.random.randn(N_max, p) * mask[:, None]
        r = np.random.randn()
        g_next = np.random.randn(g_dim)
        X_next = np.random.randn(N_max, x_dim)
        
        # Simulate churn
        if np.random.rand() < 0.1:
            n_active_next = np.random.randint(10, 50)
        else:
            n_active_next = n_active
        mask_next = np.zeros(N_max)
        mask_next[:n_active_next] = 1
        
        done = False
        
        agent.store_transition(manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done)
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # Test updates
    print(f"\nTesting updates...")
    for i in range(10):
        metrics = agent.update(batch_size=32)
        if metrics:
            print(f"  Update {i+1}: critic_loss={metrics['critic_loss']:.4f}, " +
                  f"q1={metrics['q1_value']:.3f}, q2={metrics['q2_value']:.3f}", end="")
            if 'actor_loss' in metrics:
                print(f", actor_loss={metrics['actor_loss']:.4f}")
            else:
                print()
    
    print(f"\nTotal steps: {agent.total_steps}")
    print(f"Actor updates: {agent.actor_updates}")
    print(f"Critic updates: {agent.critic_updates}")
    
    print("\n=== EA Agent test passed ===")


if __name__ == "__main__":
    test_ea_agent()
