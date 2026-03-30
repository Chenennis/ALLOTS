"""
MAPPO-Set: MAPPO with Set-Based Actor from ALLOTS.

Actor:  Stochastic set-based actor built on SetToSetActor backbone.
        Outputs Gaussian distribution (mean via SetToSetActor + learned log_std).
Critic: Centralized MLP value function V(s) (standard MAPPO critic).

Training: PPO clip with GAE advantages, shared across all managers.

Does NOT include: Pair-Set tokens, TD-consistent bootstrapping, per-device credit.
This isolates the benefit of set-based encoding within MAPPO's on-policy framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

from algorithms.EA.foea.actor import SetToSetActor, masked_mean

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stochastic Set Actor (Gaussian policy on top of SetToSetActor)
# ---------------------------------------------------------------------------

class StochasticSetActor(nn.Module):
    """Wraps SetToSetActor to produce a Gaussian policy for PPO.

    mean = SetToSetActor(g, X, mask, manager_id)   # [B, N_max, p]
    std  = softplus(log_std_param) broadcast to same shape
    π(a|s) = N(mean, diag(std²))
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
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.p = p
        self.N_max = N_max

        # Deterministic backbone
        self.backbone = SetToSetActor(
            x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max,
            num_managers=num_managers, emb_dim=emb_dim,
            token_dim=token_dim, hidden_dim=hidden_dim,
        )

        # Learnable log_std (per action dimension, shared across devices)
        self.log_std = nn.Parameter(torch.full((p,), init_log_std))

    def forward(
        self,
        g: torch.Tensor,       # [B, g_dim]
        X: torch.Tensor,       # [B, N_max, x_dim]
        mask: torch.Tensor,    # [B, N_max]
        manager_id: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) both [B, N_max, p]."""
        mean = self.backbone(g, X, mask, manager_id)  # [B, N_max, p]
        std = F.softplus(self.log_std).unsqueeze(0).unsqueeze(0).expand_as(mean)
        return mean, std

    def evaluate_actions(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor,
        actions: torch.Tensor,       # [B, N_max, p]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions under current policy.

        Returns:
            action_log_probs: [B, 1]  (sum over active devices and action dims)
            values:           None     (critic is separate)
            dist_entropy:     [B, 1]
        """
        mean, std = self.forward(g, X, mask, manager_id)

        # Gaussian log-prob
        var = std ** 2
        log_prob = -0.5 * (((actions - mean) ** 2) / var + torch.log(var)
                           + np.log(2 * np.pi))  # [B, N_max, p]

        # Mask: only count active devices
        mask_3d = mask.unsqueeze(-1)  # [B, N_max, 1]
        log_prob = log_prob * mask_3d  # zero out inactive

        # Sum over action dims and devices -> scalar per sample
        action_log_probs = log_prob.sum(dim=(1, 2), keepdim=False).unsqueeze(1)  # [B, 1]

        # Entropy of Gaussian = 0.5 * ln(2πe·σ²) per dim
        entropy = 0.5 * (1.0 + torch.log(var) + np.log(2 * np.pi))  # [B, N_max, p]
        entropy = (entropy * mask_3d).sum(dim=(1, 2), keepdim=False).unsqueeze(1)  # [B, 1]

        return action_log_probs, entropy

    def sample_action(
        self,
        g: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        manager_id: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob).

        Returns:
            action:    [B, N_max, p]  masked, clipped to [-1, 1]
            log_prob:  [B, 1]
        """
        mean, std = self.forward(g, X, mask, manager_id)

        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)

        # Clip and mask
        action = torch.clamp(action, -1.0, 1.0)
        action = action * mask.unsqueeze(-1)

        # Compute log-prob of the (unclipped) action
        log_prob, _ = self.evaluate_actions(g, X, mask, manager_id, action)

        return action, log_prob


# ---------------------------------------------------------------------------
# Centralized Value Critic V(s)
# ---------------------------------------------------------------------------

class SetValueCritic(nn.Module):
    """Centralized value function.

    Input: concatenation of all managers' flattened observations
           [X_0.flat, g_0, mask_0, ..., X_{M-1}.flat, g_{M-1}, mask_{M-1}]
    Output: scalar V(s).
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [B, input_dim] -> [B, 1]"""
        return self.net(state)


# ---------------------------------------------------------------------------
# GAE Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """On-policy rollout buffer with GAE computation.

    Stores per-step structured data for all managers and computes
    GAE advantages at the end of a rollout.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.g_buf: List[np.ndarray] = []          # each [M, g_dim]
        self.X_buf: List[np.ndarray] = []          # each [M, N_max, x_dim]
        self.mask_buf: List[np.ndarray] = []       # each [M, N_max]
        self.action_buf: List[np.ndarray] = []     # each [M, N_max, p]
        self.log_prob_buf: List[float] = []        # each scalar
        self.reward_buf: List[float] = []          # each scalar (global reward)
        self.value_buf: List[float] = []           # each scalar V(s)
        self.done_buf: List[bool] = []
        self.size = 0

    def push(
        self,
        all_g: np.ndarray,
        all_X: np.ndarray,
        all_mask: np.ndarray,
        all_A: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.g_buf.append(all_g.copy())
        self.X_buf.append(all_X.copy())
        self.mask_buf.append(all_mask.copy())
        self.action_buf.append(all_A.copy())
        self.log_prob_buf.append(log_prob)
        self.reward_buf.append(reward)
        self.value_buf.append(value)
        self.done_buf.append(done)
        self.size += 1

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        Returns:
            advantages: [T]
            returns:    [T]
        """
        T = self.size
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.done_buf[t])
            else:
                next_value = self.value_buf[t + 1]
                next_non_terminal = 1.0 - float(self.done_buf[t])

            delta = (self.reward_buf[t]
                     + gamma * next_non_terminal * next_value
                     - self.value_buf[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.value_buf, dtype=np.float32)
        return advantages, returns

    def get_batches(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        mini_batch_size: int,
        device: str = 'cpu',
    ):
        """Yield mini-batches of (g, X, mask, A, old_log_prob, adv, ret).

        All tensors on *device*.
        """
        T = self.size
        indices = np.random.permutation(T)

        for start in range(0, T, mini_batch_size):
            end = min(start + mini_batch_size, T)
            idx = indices[start:end]

            yield (
                torch.FloatTensor(np.array([self.g_buf[i] for i in idx])).to(device),
                torch.FloatTensor(np.array([self.X_buf[i] for i in idx])).to(device),
                torch.FloatTensor(np.array([self.mask_buf[i] for i in idx])).to(device),
                torch.FloatTensor(np.array([self.action_buf[i] for i in idx])).to(device),
                torch.FloatTensor(np.array([self.log_prob_buf[i] for i in idx])).to(device),
                torch.FloatTensor(advantages[idx]).to(device),
                torch.FloatTensor(returns[idx]).to(device),
            )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MAPPOSetAgent:
    """
    MAPPO with Set-Based Actor.

    Architecture
    ------------
    * Actor : StochasticSetActor — shared across managers via manager_id
      embedding. Outputs Gaussian distribution over per-device actions.
    * Critic: Centralized MLP V(s) that takes flattened global state.

    Training
    --------
    On-policy PPO with GAE:
      actor_loss  = -min(ratio·A, clip(ratio)·A)  + entropy bonus
      critic_loss = MSE(V(s), returns)
    No target networks (on-policy).

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
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
    ):
        self.x_dim = x_dim
        self.g_dim = g_dim
        self.p = p
        self.N_max = N_max
        self.num_managers = num_managers
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.current_episode = 0

        # -- Actor (stochastic, shared) --
        self.actor = StochasticSetActor(
            x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max,
            num_managers=num_managers, emb_dim=emb_dim,
            token_dim=token_dim, hidden_dim=hidden_dim,
        ).to(device)

        # -- Critic (centralized value function) --
        per_mgr_dim = N_max * x_dim + g_dim + N_max  # X_flat + g + mask
        critic_input_dim = num_managers * per_mgr_dim
        self.critic = SetValueCritic(critic_input_dim, hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # -- Rollout buffer (reset each episode / rollout) --
        self.rollout = RolloutBuffer()
        self.total_steps = 0

        logger.info(
            f"MAPPOSetAgent  M={num_managers}  N_max={N_max}  "
            f"critic_in={critic_input_dim}  "
            f"actor_params={sum(pp.numel() for pp in self.actor.parameters())}  "
            f"clip={clip_param}  epochs={ppo_epochs}"
        )

    # ---- helpers ----

    def _build_critic_input(
        self,
        all_X: torch.Tensor,      # [B, M, N_max, x_dim]
        all_g: torch.Tensor,      # [B, M, g_dim]
        all_mask: torch.Tensor,   # [B, M, N_max]
    ) -> torch.Tensor:
        """Flatten all managers' obs for the centralized critic. -> [B, D]"""
        B = all_X.size(0)
        parts = []
        for i in range(self.num_managers):
            parts.append(all_X[:, i].reshape(B, -1))
            parts.append(all_g[:, i])
            parts.append(all_mask[:, i])
        return torch.cat(parts, dim=1)

    def _build_critic_input_np(
        self, all_X: np.ndarray, all_g: np.ndarray, all_mask: np.ndarray,
    ) -> np.ndarray:
        """Numpy version for single-sample value prediction."""
        parts = []
        for i in range(self.num_managers):
            parts.append(all_X[i].flatten())
            parts.append(all_g[i])
            parts.append(all_mask[i])
        return np.concatenate(parts)

    # ---- public API ----

    def select_action(
        self,
        g: np.ndarray,          # [g_dim]
        X: np.ndarray,          # [N_max, x_dim]
        mask: np.ndarray,       # [N_max]
        manager_id: int,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Select action for a single manager.

        Returns:
            action:    [N_max, p]
            log_prob:  scalar
        """
        self.actor.eval()
        with torch.no_grad():
            g_t = torch.FloatTensor(g).unsqueeze(0).to(self.device)
            X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            mid = torch.LongTensor([manager_id]).to(self.device)

            action, log_prob = self.actor.sample_action(
                g_t, X_t, m_t, mid, deterministic=deterministic,
            )
        self.actor.train()
        return action.squeeze(0).cpu().numpy(), log_prob.item()

    def get_value(
        self,
        all_g: np.ndarray,      # [M, g_dim]
        all_X: np.ndarray,      # [M, N_max, x_dim]
        all_mask: np.ndarray,   # [M, N_max]
    ) -> float:
        """Compute V(s) for the current global state."""
        s = self._build_critic_input_np(all_X, all_g, all_mask)
        with torch.no_grad():
            s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            v = self.critic(s_t).item()
        return v

    def store_transition(
        self,
        all_g: np.ndarray,
        all_X: np.ndarray,
        all_mask: np.ndarray,
        all_A: np.ndarray,       # [M, N_max, p]
        log_prob: float,         # sum of per-manager log probs
        reward: float,
        value: float,
        done: bool,
    ):
        """Store a step in the rollout buffer."""
        self.rollout.push(all_g, all_X, all_mask, all_A, log_prob, reward, value, done)

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """PPO update on the collected rollout.

        Should be called at the end of an episode or after a fixed number of
        steps.  Pass `last_value = V(s_T)` for bootstrapping (0 if terminal).

        1. Compute GAE advantages and returns.
        2. For each PPO epoch, iterate over mini-batches:
           a. Re-evaluate log-probs and entropy under current policy.
           b. Clipped surrogate actor loss.
           c. MSE value loss.
        3. Clear rollout buffer.
        """
        if self.rollout.size == 0:
            return {}

        # ---------- 1. GAE ----------
        advantages, returns = self.rollout.compute_gae(
            last_value, self.gamma, self.gae_lambda,
        )
        # normalize advantages
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ---------- 2. PPO epochs ----------
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for (g_b, X_b, mask_b, A_b, old_lp_b, adv_b, ret_b) in \
                    self.rollout.get_batches(advantages, returns,
                                            self.mini_batch_size, self.device):
                # g_b: [B, M, g_dim]  X_b: [B, M, N, x]  mask_b: [B, M, N]
                # A_b: [B, M, N, p]   old_lp_b: [B]  adv_b: [B]  ret_b: [B]

                B = g_b.size(0)

                # --- re-evaluate all managers' log-probs ---
                new_log_probs = torch.zeros(B, device=self.device)
                new_entropy = torch.zeros(B, device=self.device)
                for i in range(self.num_managers):
                    mid = torch.full((B,), i, dtype=torch.long, device=self.device)
                    lp_i, ent_i = self.actor.evaluate_actions(
                        g_b[:, i], X_b[:, i], mask_b[:, i], mid, A_b[:, i],
                    )
                    new_log_probs += lp_i.squeeze(1)
                    new_entropy += ent_i.squeeze(1)

                # --- PPO clipped surrogate ---
                ratio = torch.exp(new_log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value loss ---
                critic_in = self._build_critic_input(X_b, g_b, mask_b)
                values = self.critic(critic_in).squeeze(1)  # [B]
                value_loss = F.mse_loss(values, ret_b)

                # --- combined loss ---
                entropy_bonus = new_entropy.mean()
                loss = (policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_bonus)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                n_updates += 1

        # ---------- 3. cleanup ----------
        self.rollout.clear()
        self.total_steps += 1

        if n_updates == 0:
            return {}

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'training_step': self.total_steps,
        }

    # ---- episode hook ----

    def set_episode(self, episode: int):
        self.current_episode = episode

    # ---- save / load ----

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        if 'actor_optimizer' in ckpt:
            self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        if 'critic_optimizer' in ckpt:
            self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
