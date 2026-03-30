"""
MAAC-Set: MAAC with Set-Based Actor from ALLOTS.

Actor:  reuses EA's SetToSetActor (one per manager, not shared).
Critic: reuses MAAC's AttentionCriticContinuous — each manager attends to
        other managers' state-action encodings via multi-head attention.

Does NOT include: Pair-Set tokens, TD-consistent bootstrapping, per-device credit.
This isolates the benefit of set-based encoding within MAAC's attention framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List
from collections import deque
import random
import logging

from algorithms.EA.foea.actor import SetToSetActor, masked_mean
from algorithms.MAAC.fomaac.fomaac import AttentionCriticContinuous, ReplayBuffer

logger = logging.getLogger(__name__)


class MAACSetAgent:
    """
    MAAC with Set-Based Actor.

    Architecture
    ------------
    * Actor : EA's SetToSetActor — one per manager (MAAC uses per-agent actors).
      Each actor processes (g, X, mask, manager_id) -> A[N_max, p].
    * Critic: MAAC's AttentionCriticContinuous — shared attention critic that
      computes per-manager Q-values through multi-head cross-attention among
      manager state-action encodings.

    Training
    --------
    Standard MAAC DDPG-style policy gradient with attention critic:
      critic_loss = Σ_i MSE(Q_i, r_i + γ(1-d)·Q_target_i)
      actor_loss_i = -Q_i(s, π_i(s_i)).mean()   per-manager
    Soft-updated target networks.

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
        attend_heads: int = 4,
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

        # State/action dims for MAAC critic (flattened per manager)
        state_dim = N_max * x_dim + g_dim
        action_dim = N_max * p
        self.state_dim = state_dim
        self.action_dim = action_dim

        # -- Actors (one per manager, MAAC style) --
        self.actors = nn.ModuleList([
            SetToSetActor(
                x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max,
                num_managers=num_managers, emb_dim=emb_dim,
                token_dim=token_dim, hidden_dim=hidden_dim,
            )
            for _ in range(num_managers)
        ]).to(device)

        self.target_actors = nn.ModuleList([
            SetToSetActor(
                x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max,
                num_managers=num_managers, emb_dim=emb_dim,
                token_dim=token_dim, hidden_dim=hidden_dim,
            )
            for _ in range(num_managers)
        ]).to(device)

        # sync target actors
        for a, ta in zip(self.actors, self.target_actors):
            ta.load_state_dict(a.state_dict())

        # -- Critic (shared attention-based, MAAC style) --
        self.critic = AttentionCriticContinuous(
            n_agents=num_managers, state_dim=state_dim,
            action_dim=action_dim, hidden_dim=hidden_dim,
            attend_heads=attend_heads,
        ).to(device)

        self.target_critic = AttentionCriticContinuous(
            n_agents=num_managers, state_dim=state_dim,
            action_dim=action_dim, hidden_dim=hidden_dim,
            attend_heads=attend_heads,
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # -- Optimizers --
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # -- Replay Buffer (MAAC's per-agent list-based buffer) --
        self.replay_buffer = ReplayBuffer(buffer_capacity, num_managers)
        self.total_steps = 0

        total_actor_params = sum(
            sum(p.numel() for p in actor.parameters())
            for actor in self.actors
        )
        logger.info(
            f"MAACSetAgent  M={num_managers}  N_max={N_max}  "
            f"state_dim={state_dim}  action_dim={action_dim}  "
            f"attend_heads={attend_heads}  actor_params={total_actor_params}"
        )

    # ---- helpers ----

    def _flatten_state(self, X: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Flatten per-manager obs for the attention critic.
        X: [N_max, x_dim]  g: [g_dim]  ->  [N_max*x_dim + g_dim]
        """
        return np.concatenate([X.flatten(), g])

    def _flatten_action(self, A: np.ndarray) -> np.ndarray:
        """A: [N_max, p] -> [N_max*p]"""
        return A.flatten()

    def _set_actor_forward_batch(
        self, actor: SetToSetActor, g: torch.Tensor, X: torch.Tensor,
        mask: torch.Tensor, manager_id: int,
    ) -> torch.Tensor:
        """Run set-actor forward on a batch, return flattened actions.

        Args:
            g:    [B, g_dim]
            X:    [B, N_max, x_dim]
            mask: [B, N_max]
        Returns:
            [B, N_max*p]  (flattened for critic)
        """
        B = g.size(0)
        mid = torch.full((B,), manager_id, dtype=torch.long, device=self.device)
        A = actor(g, X, mask, mid)  # [B, N_max, p]
        return A.reshape(B, -1)     # [B, N_max*p]

    # ---- public API ----

    def select_actions(
        self,
        device_states: np.ndarray,
        global_feats: np.ndarray,
        masks: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Select actions for all managers.

        Args:
            device_states: [M, N_max, x_dim]
            global_feats:  [M, g_dim]
            masks:         [M, N_max]

        Returns:
            actions: [M, N_max, p]
        """
        actions = []
        for i in range(self.num_managers):
            self.actors[i].eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(device_states[i:i+1]).to(self.device)
                g_t = torch.FloatTensor(global_feats[i:i+1]).to(self.device)
                m_t = torch.FloatTensor(masks[i:i+1]).to(self.device)
                mid = torch.LongTensor([i]).to(self.device)
                a = self.actors[i](g_t, X_t, m_t, mid).cpu().numpy()[0]  # [N_max, p]
            self.actors[i].train()

            if add_noise:
                noise = np.random.normal(0, self.noise_scale, a.shape)
                noise *= masks[i][:, np.newaxis]
                a = np.clip(a + noise, -1, 1)
                a *= masks[i][:, np.newaxis]
            actions.append(a)

        return np.array(actions)

    def store_experience(
        self,
        device_states: np.ndarray,   # [M, N_max, x_dim]
        global_feats: np.ndarray,    # [M, g_dim]
        masks: np.ndarray,           # [M, N_max]
        actions: np.ndarray,         # [M, N_max, p]
        rewards: List[float],        # [M]  per-manager rewards
        next_device_states: np.ndarray,
        next_global_feats: np.ndarray,
        next_masks: np.ndarray,
        done: bool,
    ):
        """Store experience.  Flattens obs/actions for the attention critic
        buffer while also keeping structured data in a side buffer for actor
        reconstruction during training."""
        states_flat = []
        actions_flat = []
        next_states_flat = []

        for i in range(self.num_managers):
            states_flat.append(self._flatten_state(device_states[i], global_feats[i]))
            actions_flat.append(self._flatten_action(actions[i]))
            next_states_flat.append(self._flatten_state(next_device_states[i], next_global_feats[i]))

        self.replay_buffer.push(states_flat, actions_flat, rewards, next_states_flat, done)

        # Also store structured data for actor forward passes during update
        if not hasattr(self, '_struct_buffer'):
            self._struct_buffer = deque(maxlen=self.replay_buffer.capacity)
            self._struct_index = {}   # maps buffer index -> structured data

        self._struct_buffer.append((
            device_states.copy(), global_feats.copy(), masks.copy(),
            next_device_states.copy(), next_global_feats.copy(), next_masks.copy(),
        ))

    def update(self) -> Dict[str, float]:
        """Full MAAC update with set-based actors.

        1. Sample batch from replay buffer.
        2. Compute target Q for each manager via target attention critic.
        3. Critic loss = Σ_i MSE(Q_i, target_Q_i).
        4. Per-manager actor loss = -Q_i(..., π_i(s_i), ...).mean().
        5. Soft-update all target networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # ---------- 1. sample ----------
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size, self.device)
        # states:      List[M] of [B, state_dim]
        # actions:     List[M] of [B, action_dim]
        # rewards:     List[M] of [B, 1]
        # next_states: List[M] of [B, state_dim]
        # dones:       [B, 1]

        # ---------- 2. target Q ----------
        with torch.no_grad():
            # We need structured data to run SetToSetActor for target actions.
            # Extract (X_next, g_next, mask_next) from flat next_states.
            next_actions_flat = []
            for i in range(self.num_managers):
                # parse flat state back to structured: [X_flat | g]
                ns = next_states[i]  # [B, state_dim]
                g_next = ns[:, self.N_max * self.x_dim:]     # [B, g_dim]
                X_next_flat = ns[:, :self.N_max * self.x_dim] # [B, N_max*x_dim]
                X_next = X_next_flat.reshape(-1, self.N_max, self.x_dim)  # [B, N_max, x_dim]

                # Reconstruct mask from X: active device = any non-zero feature
                mask_next = (X_next.abs().sum(dim=-1) > 1e-8).float()  # [B, N_max]

                na = self._set_actor_forward_batch(
                    self.target_actors[i], g_next, X_next, mask_next, i,
                )
                next_actions_flat.append(na)  # [B, action_dim]

            target_q_values = self.target_critic(next_states, next_actions_flat)
            # List[M] of [B, 1]
            target_q = [
                rewards[i] + self.gamma * (1.0 - dones) * target_q_values[i]
                for i in range(self.num_managers)
            ]

        # ---------- 3. critic loss ----------
        current_q = self.critic(states, actions)  # List[M] of [B, 1]
        critic_loss = sum(
            F.mse_loss(current_q[i], target_q[i])
            for i in range(self.num_managers)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # ---------- 4. per-manager actor loss ----------
        actor_losses = []
        for i in range(self.num_managers):
            # parse flat state to structured for current actor
            s = states[i]  # [B, state_dim]
            g_i = s[:, self.N_max * self.x_dim:]
            X_i = s[:, :self.N_max * self.x_dim].reshape(-1, self.N_max, self.x_dim)
            mask_i = (X_i.abs().sum(dim=-1) > 1e-8).float()

            # Replace manager i's action with current policy output
            curr_actions = []
            for j in range(self.num_managers):
                if j == i:
                    curr_actions.append(
                        self._set_actor_forward_batch(self.actors[i], g_i, X_i, mask_i, i)
                    )
                else:
                    curr_actions.append(actions[j])

            q_values = self.critic(states, curr_actions)
            actor_loss = -q_values[i].mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())

        # ---------- 5. soft update ----------
        self._soft_update()

        self.total_steps += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': np.mean(actor_losses),
            'q_value': np.mean([q.mean().item() for q in current_q]),
            'training_step': self.total_steps,
        }

    def _soft_update(self):
        """Soft update all target networks."""
        for actor, target in zip(self.actors, self.target_actors):
            for p, tp in zip(actor.parameters(), target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ---- episode hook ----

    def set_episode(self, episode: int):
        self.current_episode = episode

    # ---- save / load ----

    def save(self, path: str):
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'target_actors': [a.state_dict() for a in self.target_actors],
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizers': [o.state_dict() for o in self.actor_optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for i, a in enumerate(self.actors):
            a.load_state_dict(ckpt['actors'][i])
        for i, a in enumerate(self.target_actors):
            a.load_state_dict(ckpt['target_actors'][i])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        if 'actor_optimizers' in ckpt:
            for i, o in enumerate(self.actor_optimizers):
                o.load_state_dict(ckpt['actor_optimizers'][i])
        if 'critic_optimizer' in ckpt:
            self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
