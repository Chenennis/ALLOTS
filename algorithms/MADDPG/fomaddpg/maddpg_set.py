"""
MADDPG-Set: MADDPG with Set-Based Actor from ALLOTS.

Actor:  reuses EA's SetToSetActor (phi_k -> pool -> phi_r -> action head)
        shared across all managers via manager_id embedding.
Critic: standard MADDPG centralized MLP (flattened global state+action -> Q).

Does NOT include: Pair-Set tokens, TD-consistent bootstrapping, per-device credit.
This isolates the benefit of set-based encoding from ALLOTS's other innovations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Tuple, Optional, List
from collections import deque
import random
import logging

from algorithms.EA.foea.actor import SetToSetActor, masked_mean

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """Standard MADDPG centralized critic: flattened global state+action -> Q.

    Input:  concatenated [s_flat_0, ..., s_flat_{M-1}, a_flat_0, ..., a_flat_{M-1}]
            where each s_flat_i = [X_i.flatten(), g_i, mask_i]
            and   each a_flat_i = A_i.flatten()
    Output: scalar Q-value
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """state_action: [B, input_dim] -> [B, 1]"""
        return self.net(state_action)


# ---------------------------------------------------------------------------
# Structured Replay Buffer
# ---------------------------------------------------------------------------

class StructuredReplayBuffer:
    """Replay buffer that stores per-manager structured data so the
    set-based actor can reconstruct (g, X, mask) during training."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        all_g: np.ndarray,          # [M, g_dim]
        all_X: np.ndarray,          # [M, N_max, x_dim]
        all_mask: np.ndarray,       # [M, N_max]
        all_A: np.ndarray,          # [M, N_max, p]
        reward: float,              # scalar (global)
        all_g_next: np.ndarray,     # [M, g_dim]
        all_X_next: np.ndarray,     # [M, N_max, x_dim]
        all_mask_next: np.ndarray,  # [M, N_max]
        done: float,
    ):
        self.buffer.append((
            all_g.copy(), all_X.copy(), all_mask.copy(), all_A.copy(),
            float(reward),
            all_g_next.copy(), all_X_next.copy(), all_mask_next.copy(),
            float(done),
        ))

    def sample(self, batch_size: int, device: str = 'cpu'):
        """Return batched tensors on *device*.

        Returns:
            g      [B, M, g_dim]
            X      [B, M, N_max, x_dim]
            mask   [B, M, N_max]
            A      [B, M, N_max, p]
            reward [B, 1]
            g_next [B, M, g_dim]
            X_next [B, M, N_max, x_dim]
            mask_next [B, M, N_max]
            done   [B, 1]
        """
        batch = random.sample(self.buffer, batch_size)
        g, X, mask, A, r, gn, Xn, mn, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(g)).to(device),
            torch.FloatTensor(np.array(X)).to(device),
            torch.FloatTensor(np.array(mask)).to(device),
            torch.FloatTensor(np.array(A)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(gn)).to(device),
            torch.FloatTensor(np.array(Xn)).to(device),
            torch.FloatTensor(np.array(mn)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MADDPGSetAgent:
    """
    MADDPG with Set-Based Actor.

    Architecture
    ------------
    * Actor : EA's SetToSetActor — **shared** across managers via manager_id
      embedding.  Processes per-manager (g, X, mask) → A[N_max, p].
    * Critic: centralized MLP that takes the flattened observations and
      actions of *all* managers → scalar Q.

    Training
    --------
    Standard MADDPG (DDPG) policy gradient:
      critic_loss = MSE(Q, r + γ(1-d)·Q_target)
      actor_loss  = -Q(s, π(s)).mean()
    Soft-updated target networks for both actor and critic.

    No pair-set critic, no TD-consistent bootstrapping, no per-device credit.
    """

    def __init__(
        self,
        x_dim: int = 6,
        g_dim: int = 26,
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
        noise_scale: float = 0.1,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        device: str = 'cpu',
    ):
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.N_max = N_max
        self.num_managers = num_managers
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.device = device
        self.current_episode = 0

        # -- Actor (shared across all managers) --
        self.actor = SetToSetActor(
            x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max,
            num_managers=num_managers, emb_dim=emb_dim,
            token_dim=token_dim, hidden_dim=hidden_dim,
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        # -- Critic --
        # input = concat of per-manager [X_flat, g, mask, A_flat]
        per_mgr_dim = N_max * x_dim + g_dim + N_max + N_max * p
        critic_input_dim = num_managers * per_mgr_dim
        self.critic = CentralizedCritic(critic_input_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = StructuredReplayBuffer(buffer_capacity)
        self.total_steps = 0

        logger.info(
            f"MADDPGSetAgent  M={num_managers}  N_max={N_max}  "
            f"critic_in={critic_input_dim}  actor_params="
            f"{sum(p.numel() for p in self.actor.parameters())}"
        )

    # ---- helpers ----

    def _build_critic_input(
        self, X: torch.Tensor, g: torch.Tensor,
        mask: torch.Tensor, A: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten per-manager data and concatenate for the centralized critic.

        Args:
            X:    [B, M, N_max, x_dim]
            g:    [B, M, g_dim]
            mask: [B, M, N_max]
            A:    [B, M, N_max, p]

        Returns:
            [B, critic_input_dim]
        """
        B = X.size(0)
        parts = []
        for i in range(self.num_managers):
            parts.append(X[:, i].reshape(B, -1))          # [B, N_max*x_dim]
            parts.append(g[:, i])                          # [B, g_dim]
            parts.append(mask[:, i])                       # [B, N_max]
            parts.append(A[:, i].reshape(B, -1))           # [B, N_max*p]
        return torch.cat(parts, dim=1)  # [B, M * per_mgr_dim]

    def _manager_ids(self, batch_size: int) -> torch.Tensor:
        """[M] manager id vector on device."""
        return torch.arange(self.num_managers, device=self.device)

    # ---- public API ----

    def select_action(
        self, g: np.ndarray, X: np.ndarray,
        mask: np.ndarray, manager_id: int,
        explore: bool = True,
    ) -> np.ndarray:
        """Select action for a single manager.

        Args:
            g:    [g_dim]       global features
            X:    [N_max, x_dim] device states (padded)
            mask: [N_max]        active-device mask
            manager_id: int

        Returns:
            A: [N_max, p]  actions (masked)
        """
        self.actor.eval()
        with torch.no_grad():
            g_t = torch.FloatTensor(g).unsqueeze(0).to(self.device)
            X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            mid_t = torch.LongTensor([manager_id]).to(self.device)

            A = self.actor(g_t, X_t, mask_t, mid_t).squeeze(0).cpu().numpy()

        if explore:
            noise = np.random.normal(0, self.noise_scale, A.shape)
            noise *= mask[:, np.newaxis]
            A = np.clip(A + noise, -1, 1)
            A *= mask[:, np.newaxis]

        self.actor.train()
        return A

    def store_transition(
        self,
        all_g: np.ndarray,          # [M, g_dim]
        all_X: np.ndarray,          # [M, N_max, x_dim]
        all_mask: np.ndarray,       # [M, N_max]
        all_A: np.ndarray,          # [M, N_max, p]
        reward: float,
        all_g_next: np.ndarray,
        all_X_next: np.ndarray,
        all_mask_next: np.ndarray,
        done: float,
    ):
        """Store one global transition (all managers' data)."""
        self.replay_buffer.push(
            np.asarray(all_g), np.asarray(all_X), np.asarray(all_mask),
            np.asarray(all_A), reward,
            np.asarray(all_g_next), np.asarray(all_X_next),
            np.asarray(all_mask_next), done,
        )

    def update(self) -> Dict[str, float]:
        """Full MADDPG (DDPG) update using set-based actor.

        1. Sample a batch of structured transitions.
        2. Compute target Q  =  r + γ(1-d)·Q_target(s', π_target(s')).
        3. Critic loss = MSE(Q_current, target Q).
        4. Actor loss  = -Q(s, π(s)).mean()   (policy gradient).
        5. Soft-update target networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # ---------- 1. sample ----------
        (g, X, mask, A, reward, g_next, X_next, mask_next, done) = \
            self.replay_buffer.sample(self.batch_size, self.device)
        # shapes: g[B,M,g_dim]  X[B,M,N,x]  mask[B,M,N]  A[B,M,N,p]
        #         reward[B,1]   done[B,1]

        B = g.size(0)
        mids = self._manager_ids(B)  # [M]

        # ---------- 2. target Q ----------
        with torch.no_grad():
            # compute next actions for all managers with target actor
            A_next = torch.zeros_like(A)  # [B, M, N, p]
            for i in range(self.num_managers):
                mid_batch = torch.full((B,), i, dtype=torch.long, device=self.device)
                A_next[:, i] = self.actor_target(
                    g_next[:, i], X_next[:, i], mask_next[:, i], mid_batch,
                )

            critic_in_next = self._build_critic_input(X_next, g_next, mask_next, A_next)
            q_target = self.critic_target(critic_in_next)       # [B, 1]
            y = reward + self.gamma * (1.0 - done) * q_target   # [B, 1]

        # ---------- 3. critic loss ----------
        critic_in = self._build_critic_input(X, g, mask, A)
        q_current = self.critic(critic_in)                       # [B, 1]
        critic_loss = F.mse_loss(q_current, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # ---------- 4. actor loss ----------
        # re-compute actions for all managers with current actor
        A_policy = torch.zeros_like(A)
        for i in range(self.num_managers):
            mid_batch = torch.full((B,), i, dtype=torch.long, device=self.device)
            A_policy[:, i] = self.actor(g[:, i], X[:, i], mask[:, i], mid_batch)

        critic_in_policy = self._build_critic_input(X, g, mask, A_policy)
        q_policy = self.critic(critic_in_policy)                 # [B, 1]
        actor_loss = -q_policy.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # ---------- 5. soft update ----------
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        self.total_steps += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': q_current.mean().item(),
            'training_step': self.total_steps,
        }

    # ---- target update ----

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # ---- episode hook ----

    def set_episode(self, episode: int):
        self.current_episode = episode

    # ---- save / load ----

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
