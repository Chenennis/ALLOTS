"""
Algorithm wrappers for MPE Churn experiments.

Each wrapper adapts a specific MARL algorithm to work with MPEChurnEnv's
observation/action format. All wrappers share a common interface.

Key design:
- EA: direct passthrough (native set-based interface)
- MADDPG/MATD3/SQDDPG: flatten obs+mask into single vector per controller
- MAAC/AGILE: use structured [n_agents, N_max, dim] format
- MAPPO/MAIPPO: flatten obs, on-policy buffer

All wrappers call REAL algorithm code from algorithms/ - no fake learning.
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Modified_MPE.mpe_churn.churn_config import N_CONTROLLERS, N_MAX_PER, X_DIM, G_DIM, P_DIM
from Modified_MPE.mpe_churn.mpe_configs import (
    EA_CONFIG, MADDPG_CONFIG, MATD3_CONFIG, MAAC_CONFIG,
    AGILE_CONFIG, SQDDPG_CONFIG, MAPPO_CONFIG, MAIPPO_CONFIG,
    MADDPG_SET_CONFIG, MAAC_SET_CONFIG, MAPPO_SET_CONFIG,
    PADDING_STATE_DIM, PADDING_ACTION_DIM,
)


# ============================================================
# Base Wrapper
# ============================================================

class MPEAlgoWrapper(ABC):
    """Base class for all algorithm wrappers."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.ctrl_ids = [f"ctrl_{i}" for i in range(N_CONTROLLERS)]
        self._last_metrics = {}

    @abstractmethod
    def select_action(
        self,
        obs_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        explore: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Select actions for all controllers.

        Args:
            obs_dict: {ctrl_id: (X[N_max,6], g[14], mask[N_max])}
            explore: whether to add exploration noise

        Returns:
            {ctrl_id: action[N_max, P_DIM]}
        """
        pass

    @abstractmethod
    def store_and_update(
        self,
        obs: Dict, actions: Dict, rewards: Dict,
        next_obs: Dict, done: bool, info: dict,
    ) -> dict:
        """Store transition and update networks.

        Returns:
            training metrics dict
        """
        pass

    def set_episode(self, episode: int):
        """Hook for episode-level updates (e.g., credit warmup)."""
        pass

    def get_last_metrics(self) -> dict:
        return self._last_metrics

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


# ============================================================
# Helper: Flatten obs for padding-based baselines
# ============================================================

def flatten_obs_dict(obs_dict: Dict, ctrl_ids: List[str]) -> np.ndarray:
    """Flatten obs_dict to [n_agents, state_dim] for padding-based algorithms.

    state = [X.flatten() || g || mask] per controller
    """
    states = np.zeros((len(ctrl_ids), PADDING_STATE_DIM))
    for i, ctrl_id in enumerate(ctrl_ids):
        X, g, mask = obs_dict[ctrl_id]
        states[i] = np.concatenate([X.flatten(), g, mask])
    return states


def flatten_actions_dict(actions_dict: Dict, ctrl_ids: List[str]) -> np.ndarray:
    """Flatten actions_dict to [n_agents, action_dim]."""
    actions = np.zeros((len(ctrl_ids), PADDING_ACTION_DIM))
    for i, ctrl_id in enumerate(ctrl_ids):
        actions[i] = actions_dict[ctrl_id].flatten()
    return actions


def unflatten_actions(flat_actions: np.ndarray, obs_dict: Dict, ctrl_ids: List[str]) -> Dict[str, np.ndarray]:
    """Unflatten [n_agents, action_dim] back to {ctrl_id: [N_max, P_DIM]} with masking."""
    result = {}
    for i, ctrl_id in enumerate(ctrl_ids):
        _, _, mask = obs_dict[ctrl_id]
        a = flat_actions[i].reshape(N_MAX_PER, P_DIM)
        a *= mask[:, np.newaxis]  # zero out inactive slots
        result[ctrl_id] = a
    return result


# ============================================================
# EA (ALLOTS) Wrapper
# ============================================================

class MPE_EA(MPEAlgoWrapper):
    """EA/ALLOTS: direct passthrough to EAAgent (set-based, no flattening)."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.EA.foea.ea_agent import EAAgent

        self.agents = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            self.agents[ctrl_id] = EAAgent(
                x_dim=EA_CONFIG['x_dim'],
                g_dim=EA_CONFIG['g_dim'],
                p=EA_CONFIG['p'],
                N_max=EA_CONFIG['N_max'],
                num_managers=EA_CONFIG['num_managers'],
                emb_dim=EA_CONFIG['emb_dim'],
                token_dim=EA_CONFIG['token_dim'],
                hidden_dim=EA_CONFIG['hidden_dim'],
                gamma=EA_CONFIG['gamma'],
                tau=EA_CONFIG['tau'],
                lr_actor=EA_CONFIG['lr_actor'],
                lr_critic=EA_CONFIG['lr_critic'],
                policy_delay=EA_CONFIG['policy_delay'],
                noise_scale=EA_CONFIG['noise_scale'],
                noise_clip=EA_CONFIG['noise_clip'],
                advantage_tau=EA_CONFIG['advantage_tau'],
                buffer_capacity=EA_CONFIG['buffer_capacity'],
                credit_warmup_start=EA_CONFIG.get('credit_warmup_start', 100),
                credit_warmup_end=EA_CONFIG.get('credit_warmup_end', 300),
                credit_max_weight=EA_CONFIG.get('credit_max_weight', 0.05),
                device=device,
            )

    def select_action(self, obs_dict, explore=True):
        actions = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[ctrl_id]
            actions[ctrl_id] = self.agents[ctrl_id].select_action(
                g=g, X=X, mask=mask, manager_id=i, explore=explore,
            )
        return actions

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        all_metrics = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs[ctrl_id]
            X_next, g_next, mask_next = next_obs[ctrl_id]
            self.agents[ctrl_id].store_transition(
                manager_id=i, g=g, X=X, mask=mask, A=actions[ctrl_id],
                r=rewards[ctrl_id],
                g_next=g_next, X_next=X_next, mask_next=mask_next, done=done,
            )
            m = self.agents[ctrl_id].update(batch_size=EA_CONFIG['batch_size'])
            if m:
                all_metrics[ctrl_id] = m

        self._last_metrics = _aggregate_ctrl_metrics(all_metrics)
        return self._last_metrics

    def set_episode(self, episode):
        for agent in self.agents.values():
            if hasattr(agent, 'set_episode'):
                agent.set_episode(episode)


# ============================================================
# MADDPG Wrapper (padding-based)
# ============================================================

class MPE_MADDPG(MPEAlgoWrapper):
    """MADDPG: flatten obs → [n_agents, state_dim], centralized critic."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.MADDPG.fomaddpg.fomaddpg import FOMADDPG

        cfg = MADDPG_CONFIG
        self.algo = FOMADDPG(
            n_agents=cfg['n_agents'],
            state_dim=cfg['state_dim'],
            action_dim=cfg['action_dim'],
            lr_actor=cfg['lr_actor'],
            lr_critic=cfg['lr_critic'],
            hidden_dim=cfg['hidden_dim'],
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'],
            device=device,
        )

    def select_action(self, obs_dict, explore=True):
        states = flatten_obs_dict(obs_dict, self.ctrl_ids)
        raw = self.algo.select_actions(states, add_noise=explore)
        return unflatten_actions(raw, obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        states = flatten_obs_dict(obs, self.ctrl_ids)
        acts = flatten_actions_dict(actions, self.ctrl_ids)
        rews = np.array([rewards[k] for k in self.ctrl_ids])
        next_states = flatten_obs_dict(next_obs, self.ctrl_ids)
        dones = np.array([done] * N_CONTROLLERS)

        self.algo.store_experience(states, acts, rews, next_states, dones)
        m = self.algo.update()
        self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# MATD3 Wrapper (padding-based)
# ============================================================

class MPE_MATD3(MPEAlgoWrapper):
    """MATD3: same interface as MADDPG but with twin critics + delayed update.

    Note: FOMATD3's replay buffer has a state_dim mismatch (initialized with
    per-agent dim but store_experience flattens to global). We fix this by
    re-initializing the buffer with the correct global dim after construction.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.MATD3.fomatd3.fomatd3 import FOMATD3, FOReplayBuffer

        cfg = MATD3_CONFIG
        self.algo = FOMATD3(
            n_agents=cfg['n_agents'],
            state_dim=cfg['state_dim'],
            action_dim=cfg['action_dim'],
            lr_actor=cfg['lr_actor'],
            lr_critic=cfg['lr_critic'],
            hidden_dim=cfg['hidden_dim'],
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'],
            device=device,
        )
        # Fix buffer: replace with correct global state dim
        global_state_dim = cfg['n_agents'] * cfg['state_dim']
        self.algo.replay_buffer = FOReplayBuffer(
            cfg['buffer_capacity'], global_state_dim, cfg['action_dim'], cfg['n_agents']
        )

    def select_action(self, obs_dict, explore=True):
        states = flatten_obs_dict(obs_dict, self.ctrl_ids)
        raw = self.algo.select_actions(states, add_noise=explore)
        return unflatten_actions(raw, obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        states = flatten_obs_dict(obs, self.ctrl_ids)
        acts = flatten_actions_dict(actions, self.ctrl_ids)
        rews = np.array([rewards[k] for k in self.ctrl_ids])
        next_states = flatten_obs_dict(next_obs, self.ctrl_ids)
        dones = np.array([done] * N_CONTROLLERS)

        self.algo.store_experience(states, acts, rews, next_states, dones)
        m = self.algo.update()
        self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# SQDDPG Wrapper (padding-based)
# ============================================================

class MPE_SQDDPG(MPEAlgoWrapper):
    """SQDDPG: Shapley Q-value credit assignment, same flat interface."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG

        cfg = SQDDPG_CONFIG
        self.algo = FOSQDDPG(
            n_agents=cfg['n_agents'],
            state_dim=cfg['state_dim'],
            action_dim=cfg['action_dim'],
            lr_actor=cfg['lr_actor'],
            lr_critic=cfg['lr_critic'],
            hidden_dim=cfg['hidden_dim'],
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'],
            sample_size=cfg['sample_size'],
            device=device,
        )

    def select_action(self, obs_dict, explore=True):
        states = flatten_obs_dict(obs_dict, self.ctrl_ids)
        raw = self.algo.select_actions(states, add_noise=explore)
        return unflatten_actions(raw, obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        states = flatten_obs_dict(obs, self.ctrl_ids)
        acts = flatten_actions_dict(actions, self.ctrl_ids)
        rews = np.array([rewards[k] for k in self.ctrl_ids])
        next_states = flatten_obs_dict(next_obs, self.ctrl_ids)
        dones = np.array([done] * N_CONTROLLERS)

        self.algo.store_experience(states, acts, rews, next_states, dones)
        m = self.algo.update()
        self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# MAAC Wrapper (structured: [n_agents, N_max, dim])
# ============================================================

class MPE_MAAC(MPEAlgoWrapper):
    """MAAC: multi-head attention critic, native structured obs support."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.MAAC.fomaac.fomaac import FOMAAC

        cfg = MAAC_CONFIG
        self.algo = FOMAAC(
            n_agents=cfg['n_agents'],
            N_max=cfg['N_max'],
            device_dim=cfg['device_dim'],
            global_dim=cfg['global_dim'],
            action_dim=cfg['action_dim'],
            hidden_dim=cfg['hidden_dim'],
            attend_heads=cfg['attend_heads'],
            lr_actor=cfg['lr_actor'],
            lr_critic=cfg['lr_critic'],
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'],
            device=device,
        )

    def _build_structured(self, obs_dict):
        """Build [n_agents, N_max, x_dim], [n_agents, g_dim], [n_agents, N_max]."""
        ds = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
        gf = np.zeros((N_CONTROLLERS, G_DIM))
        ms = np.zeros((N_CONTROLLERS, N_MAX_PER))
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[ctrl_id]
            ds[i] = X
            gf[i] = g
            ms[i] = mask
        return ds, gf, ms

    def select_action(self, obs_dict, explore=True):
        ds, gf, ms = self._build_structured(obs_dict)
        raw = self.algo.select_actions(ds, gf, ms, add_noise=explore)
        # raw: [n_agents, N_max, action_dim]
        result = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            result[ctrl_id] = raw[i]
        return result

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        ds, gf, ms = self._build_structured(obs)
        nds, ngf, nms = self._build_structured(next_obs)

        # Build actions array [n_agents, N_max, action_dim]
        act_arr = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        rew_list = []
        for i, ctrl_id in enumerate(self.ctrl_ids):
            act_arr[i] = actions[ctrl_id]
            rew_list.append(rewards[ctrl_id])

        self.algo.store_experience(ds, gf, ms, act_arr, rew_list, nds, ngf, nms, done)
        m = self.algo.update()
        self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# AGILE Wrapper (structured: [n_agents, N_max, dim])
# ============================================================

class MPE_AGILE(MPEAlgoWrapper):
    """AGILE: GAT-based, native structured obs support."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.AGILE.foagile.foagile import FOAGILE

        cfg = AGILE_CONFIG
        self.algo = FOAGILE(
            n_agents=cfg['n_agents'],
            N_max=cfg['N_max'],
            device_dim=cfg['device_dim'],
            global_dim=cfg['global_dim'],
            action_dim=cfg['action_dim'],
            hidden_dim=cfg['hidden_dim'],
            lr_actor=cfg['lr_actor'],
            lr_critic=cfg['lr_critic'],
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'],
            device=device,
        )

    def _build_structured(self, obs_dict):
        ds = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
        gf = np.zeros((N_CONTROLLERS, G_DIM))
        ms = np.zeros((N_CONTROLLERS, N_MAX_PER))
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[ctrl_id]
            ds[i] = X
            gf[i] = g
            ms[i] = mask
        return ds, gf, ms

    def select_action(self, obs_dict, explore=True):
        ds, gf, ms = self._build_structured(obs_dict)
        raw = self.algo.select_actions(ds, gf, ms, add_noise=explore)
        result = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            result[ctrl_id] = raw[i]
        return result

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        ds, gf, ms = self._build_structured(obs)
        nds, ngf, nms = self._build_structured(next_obs)

        for i, ctrl_id in enumerate(self.ctrl_ids):
            self.algo.store_experience(
                agent_idx=i,
                device_states=ds[i], global_feat=gf[i], mask=ms[i],
                actions=actions[ctrl_id],
                reward=rewards[ctrl_id],
                next_device_states=nds[i], next_global_feat=ngf[i],
                next_mask=nms[i], done=done,
            )

        m = self.algo.update()
        self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# MAPPO Wrapper (on-policy, padding-based)
# ============================================================

class MPE_MAPPO(MPEAlgoWrapper):
    """MAPPO: on-policy shared PPO. Uses flat state, rollout buffer."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.MADDPG.fomaddpg.fomaddpg import FOMADDPG

        # MAPPO uses a simpler on-policy approach.
        # We implement a lightweight PPO loop here rather than importing the
        # complex FOMappoPPOAdapter which has FOgym-specific dependencies.
        self._init_ppo(device)

    def _init_ppo(self, device):
        """Initialize a simple shared-policy PPO for multi-agent."""
        import torch.nn as nn
        import torch.optim as optim

        cfg = MAPPO_CONFIG
        self.state_dim = cfg['state_dim']
        self.action_dim = cfg['action_dim']
        self.gamma = cfg['gamma']
        self.gae_lambda = cfg['gae_lambda']
        self.clip_epsilon = cfg['clip_epsilon']
        self.entropy_coef = cfg['entropy_coef']
        self.value_loss_coef = cfg['value_loss_coef']
        self.max_grad_norm = cfg['max_grad_norm']
        self.n_epochs = cfg['n_epochs']
        self.batch_size = cfg['batch_size']
        self.dev = torch.device(device)

        hidden = cfg['hidden_dim']

        # Shared actor (Gaussian policy)
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        ).to(self.dev)
        self.action_mean = nn.Linear(hidden, self.action_dim).to(self.dev)
        self.action_log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.dev))

        # Shared critic
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        ).to(self.dev)

        all_params = (
            list(self.actor.parameters())
            + list(self.action_mean.parameters())
            + [self.action_log_std]
            + list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=cfg['lr'])

        # Rollout storage
        self.rollout = []

    def _get_action_dist(self, state_t):
        h = self.actor(state_t)
        mean = self.action_mean(h)
        std = self.action_log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def select_action(self, obs_dict, explore=True):
        states = flatten_obs_dict(obs_dict, self.ctrl_ids)
        states_t = torch.FloatTensor(states).to(self.dev)

        with torch.no_grad():
            dist = self._get_action_dist(states_t)
            if explore:
                raw = dist.sample()
            else:
                raw = dist.mean
            raw = torch.clamp(raw, -1.0, 1.0)

        raw_np = raw.cpu().numpy()
        return unflatten_actions(raw_np, obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        states = flatten_obs_dict(obs, self.ctrl_ids)
        acts_flat = flatten_actions_dict(actions, self.ctrl_ids)
        rews = np.clip(np.array([rewards[k] for k in self.ctrl_ids]), -10.0, 10.0)  # clip rewards

        # Store for on-policy update
        self.rollout.append((states, acts_flat, rews, done))

        # Update at end of episode
        if done and len(self.rollout) > 0:
            m = self._ppo_update()
            self._last_metrics = m
            self.rollout = []
            return m
        return {}

    def _ppo_update(self):
        """Standard PPO update on collected rollout."""
        # Compute returns with GAE
        states_all, actions_all, rewards_all, returns_all, advantages_all = [], [], [], [], []

        # Process per agent
        for agent_idx in range(N_CONTROLLERS):
            states_seq = [s[agent_idx] for s, _, _, _ in self.rollout]
            actions_seq = [a[agent_idx] for _, a, _, _ in self.rollout]
            rewards_seq = [r[agent_idx] for _, _, r, _ in self.rollout]

            states_t = torch.FloatTensor(np.array(states_seq)).to(self.dev)
            with torch.no_grad():
                values = self.critic(states_t).squeeze(-1).cpu().numpy()

            # GAE
            T = len(rewards_seq)
            advantages = np.zeros(T)
            last_gae = 0
            for t in reversed(range(T)):
                next_val = values[t + 1] if t + 1 < T else 0.0
                delta = rewards_seq[t] + self.gamma * next_val - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
                advantages[t] = last_gae
            returns = advantages + values

            states_all.extend(states_seq)
            actions_all.extend(actions_seq)
            returns_all.extend(returns.tolist())
            advantages_all.extend(advantages.tolist())

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states_all)).to(self.dev)
        actions_t = torch.FloatTensor(np.array(actions_all)).to(self.dev)
        returns_t = torch.FloatTensor(returns_all).to(self.dev)
        advantages_t = torch.FloatTensor(advantages_all).to(self.dev)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Old log probs
        with torch.no_grad():
            old_dist = self._get_action_dist(states_t)
            old_log_probs = old_dist.log_prob(actions_t).sum(-1)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            dist = self._get_action_dist(states_t)
            log_probs = dist.log_prob(actions_t).sum(-1)
            ratio = (log_probs - old_log_probs).exp()

            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.critic(states_t).squeeze(-1)
            value_loss = F.mse_loss(values, returns_t)

            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.action_mean.parameters())
                + [self.action_log_std] + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
            total_loss += loss.item()

        return {
            'policy_loss': total_loss / self.n_epochs,
        }


# ============================================================
# MAIPPO Wrapper (independent PPO - same as MAPPO but separate policies)
# ============================================================

class MPE_MAIPPO(MPE_MAPPO):
    """MAIPPO: Independent PPO. Each controller has its own policy."""

    def _init_ppo(self, device):
        """Initialize independent policies for each controller."""
        import torch.nn as nn
        import torch.optim as optim

        cfg = MAIPPO_CONFIG
        self.state_dim = cfg['state_dim']
        self.action_dim = cfg['action_dim']
        self.gamma = cfg['gamma']
        self.gae_lambda = cfg['gae_lambda']
        self.clip_epsilon = cfg['clip_epsilon']
        self.entropy_coef = cfg['entropy_coef']
        self.value_loss_coef = cfg['value_loss_coef']
        self.max_grad_norm = cfg['max_grad_norm']
        self.n_epochs = cfg['n_epochs']
        self.batch_size = cfg['batch_size']
        self.dev = torch.device(device)

        hidden = cfg['hidden_dim']

        # Per-controller actors and critics
        self.actors_list = nn.ModuleList()
        self.action_means_list = nn.ModuleList()
        self.critics_list = nn.ModuleList()
        self.action_log_stds_list = nn.ParameterList()

        for _ in range(N_CONTROLLERS):
            actor = nn.Sequential(
                nn.Linear(self.state_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )
            self.actors_list.append(actor)
            self.action_means_list.append(nn.Linear(hidden, self.action_dim))
            self.action_log_stds_list.append(nn.Parameter(torch.zeros(self.action_dim)))
            self.critics_list.append(nn.Sequential(
                nn.Linear(self.state_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1),
            ))

        self.actors_list = self.actors_list.to(self.dev)
        self.action_means_list = self.action_means_list.to(self.dev)
        self.critics_list = self.critics_list.to(self.dev)
        self.action_log_stds_list = nn.ParameterList([p.to(self.dev) for p in self.action_log_stds_list])

        all_params = (
            list(self.actors_list.parameters())
            + list(self.action_means_list.parameters())
            + list(self.action_log_stds_list.parameters())
            + list(self.critics_list.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=cfg['lr'])
        self.rollout = []

        # Override single actor/critic with per-agent versions
        self.actor = None
        self.action_mean = None
        self.action_log_std = None
        self.critic = None

    def _get_action_dist_by_agent(self, state_t, agent_idx):
        h = self.actors_list[agent_idx](state_t)
        mean = self.action_means_list[agent_idx](h)
        std = self.action_log_stds_list[agent_idx].exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def select_action(self, obs_dict, explore=True):
        states = flatten_obs_dict(obs_dict, self.ctrl_ids)
        result_flat = np.zeros((N_CONTROLLERS, PADDING_ACTION_DIM))

        for i in range(N_CONTROLLERS):
            s_t = torch.FloatTensor(states[i:i+1]).to(self.dev)
            with torch.no_grad():
                dist = self._get_action_dist_by_agent(s_t, i)
                if explore:
                    raw = dist.sample()
                else:
                    raw = dist.mean
                raw = torch.clamp(raw, -1.0, 1.0)
            result_flat[i] = raw.cpu().numpy()[0]

        return unflatten_actions(result_flat, obs_dict, self.ctrl_ids)

    def _ppo_update(self):
        """PPO update with independent policies."""
        total_loss = 0.0
        for agent_idx in range(N_CONTROLLERS):
            states_seq = [s[agent_idx] for s, _, _, _ in self.rollout]
            actions_seq = [a[agent_idx] for _, a, _, _ in self.rollout]
            rewards_seq = [r[agent_idx] for _, _, r, _ in self.rollout]

            states_t = torch.FloatTensor(np.array(states_seq)).to(self.dev)
            actions_t = torch.FloatTensor(np.array(actions_seq)).to(self.dev)

            with torch.no_grad():
                values = self.critics_list[agent_idx](states_t).squeeze(-1).cpu().numpy()

            T_len = len(rewards_seq)
            advantages = np.zeros(T_len)
            last_gae = 0
            for t in reversed(range(T_len)):
                next_val = values[t + 1] if t + 1 < T_len else 0.0
                delta = rewards_seq[t] + self.gamma * next_val - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
                advantages[t] = last_gae
            returns = advantages + values

            returns_t = torch.FloatTensor(returns).to(self.dev)
            advantages_t = torch.FloatTensor(advantages).to(self.dev)
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            with torch.no_grad():
                old_dist = self._get_action_dist_by_agent(states_t, agent_idx)
                old_log_probs = old_dist.log_prob(actions_t).sum(-1)

            for _ in range(self.n_epochs):
                dist = self._get_action_dist_by_agent(states_t, agent_idx)
                log_probs = dist.log_prob(actions_t).sum(-1)
                ratio = (log_probs - old_log_probs).exp()

                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                v = self.critics_list[agent_idx](states_t).squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(v, returns_t)

                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return {'policy_loss': total_loss / (N_CONTROLLERS * self.n_epochs)}


# ============================================================
# MADDPG-Set Wrapper (set-based actor, centralized MLP critic)
# ============================================================

class MPE_MADDPG_Set(MPEAlgoWrapper):
    """MADDPG-Set: EA's SetToSetActor + standard MADDPG centralized critic.
    No pair-set, no TD-consistent, no per-device credit."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.MADDPG.fomaddpg.maddpg_set import MADDPGSetAgent

        cfg = MADDPG_SET_CONFIG
        self.algo = MADDPGSetAgent(
            x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
            N_max=cfg['N_max'], num_managers=cfg['num_managers'],
            emb_dim=cfg['emb_dim'], token_dim=cfg['token_dim'],
            hidden_dim=cfg['hidden_dim'],
            gamma=cfg['gamma'], tau=cfg['tau'],
            lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'],
            device=device,
        )

    def _build_arrays(self, obs_dict):
        """Build [M, N_max, x_dim], [M, g_dim], [M, N_max] arrays."""
        all_X = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
        all_g = np.zeros((N_CONTROLLERS, G_DIM))
        all_mask = np.zeros((N_CONTROLLERS, N_MAX_PER))
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[ctrl_id]
            all_X[i] = X
            all_g[i] = g
            all_mask[i] = mask
        return all_g, all_X, all_mask

    def select_action(self, obs_dict, explore=True):
        all_g, all_X, all_mask = self._build_arrays(obs_dict)
        result = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            a = self.algo.select_action(
                g=all_g[i], X=all_X[i], mask=all_mask[i],
                manager_id=i, explore=explore,
            )
            result[ctrl_id] = a
        return result

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        all_g, all_X, all_mask = self._build_arrays(obs)
        all_g_n, all_X_n, all_mask_n = self._build_arrays(next_obs)
        all_A = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        for i, ctrl_id in enumerate(self.ctrl_ids):
            all_A[i] = actions[ctrl_id]
        avg_reward = np.mean([rewards[c] for c in self.ctrl_ids])

        self.algo.store_transition(
            all_g, all_X, all_mask, all_A, avg_reward,
            all_g_n, all_X_n, all_mask_n, float(done),
        )
        m = self.algo.update()
        self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# MAAC-Set Wrapper (set-based actor, attention critic)
# ============================================================

class MPE_MAAC_Set(MPEAlgoWrapper):
    """MAAC-Set: EA's SetToSetActor (per-manager) + MAAC attention critic.
    No pair-set, no TD-consistent, no per-device credit."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.MAAC.fomaac.maac_set import MAACSetAgent

        cfg = MAAC_SET_CONFIG
        self.algo = MAACSetAgent(
            x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
            N_max=cfg['N_max'], num_managers=cfg['num_managers'],
            emb_dim=cfg['emb_dim'], token_dim=cfg['token_dim'],
            hidden_dim=cfg['hidden_dim'],
            attend_heads=cfg['attend_heads'],
            gamma=cfg['gamma'], tau=cfg['tau'],
            lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'],
            device=device,
        )

    def _build_structured(self, obs_dict):
        """Build [n_agents, N_max, x_dim], [n_agents, g_dim], [n_agents, N_max]."""
        ds = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
        gf = np.zeros((N_CONTROLLERS, G_DIM))
        ms = np.zeros((N_CONTROLLERS, N_MAX_PER))
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[ctrl_id]
            ds[i] = X
            gf[i] = g
            ms[i] = mask
        return ds, gf, ms

    def select_action(self, obs_dict, explore=True):
        ds, gf, ms = self._build_structured(obs_dict)
        raw = self.algo.select_actions(ds, gf, ms, add_noise=explore)
        result = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            result[ctrl_id] = raw[i]
        return result

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        ds, gf, ms = self._build_structured(obs)
        nds, ngf, nms = self._build_structured(next_obs)
        act_arr = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        rew_list = []
        for i, ctrl_id in enumerate(self.ctrl_ids):
            act_arr[i] = actions[ctrl_id]
            rew_list.append(rewards[ctrl_id])
        self.algo.store_experience(ds, gf, ms, act_arr, rew_list, nds, ngf, nms, done)
        m = self.algo.update()
        self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# MAPPO-Set Wrapper (set-based actor, centralized value critic)
# ============================================================

class MPE_MAPPO_Set(MPEAlgoWrapper):
    """MAPPO-Set: EA's SetToSetActor (stochastic) + centralized value critic.
    On-policy PPO with GAE. No pair-set, no TD-consistent, no per-device credit."""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from algorithms.MAPPO.fomappo.mappo_set import MAPPOSetAgent

        cfg = MAPPO_SET_CONFIG
        self.algo = MAPPOSetAgent(
            x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
            N_max=cfg['N_max'], num_managers=cfg['num_managers'],
            emb_dim=cfg['emb_dim'], token_dim=cfg['token_dim'],
            hidden_dim=cfg['hidden_dim'],
            gamma=cfg['gamma'], gae_lambda=cfg['gae_lambda'],
            clip_param=cfg['clip_param'], ppo_epochs=cfg['ppo_epochs'],
            mini_batch_size=cfg['mini_batch_size'],
            lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            entropy_coef=cfg['entropy_coef'],
            value_loss_coef=cfg['value_loss_coef'],
            max_grad_norm=cfg['max_grad_norm'],
            device=device,
        )

    def _build_arrays(self, obs_dict):
        """Build [M, N_max, x_dim], [M, g_dim], [M, N_max] arrays."""
        all_X = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
        all_g = np.zeros((N_CONTROLLERS, G_DIM))
        all_mask = np.zeros((N_CONTROLLERS, N_MAX_PER))
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[ctrl_id]
            all_X[i] = X
            all_g[i] = g
            all_mask[i] = mask
        return all_g, all_X, all_mask

    def select_action(self, obs_dict, explore=True):
        result = {}
        self._last_log_probs = {}
        for i, ctrl_id in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[ctrl_id]
            a, log_prob = self.algo.select_action(
                g=g, X=X, mask=mask, manager_id=i, deterministic=(not explore),
            )
            result[ctrl_id] = a
            self._last_log_probs[ctrl_id] = log_prob
        return result

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        all_g, all_X, all_mask = self._build_arrays(obs)
        all_A = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        for i, ctrl_id in enumerate(self.ctrl_ids):
            all_A[i] = actions[ctrl_id]
        avg_reward = np.mean([rewards[c] for c in self.ctrl_ids])
        total_log_prob = sum(self._last_log_probs.get(c, 0.0) for c in self.ctrl_ids)
        value = self.algo.get_value(all_g, all_X, all_mask)

        self.algo.store_transition(
            all_g, all_X, all_mask, all_A,
            log_prob=total_log_prob, reward=avg_reward,
            value=value, done=float(done),
        )

        if done:
            m = self.algo.update()
            self._last_metrics = m if m else {}
        return self._last_metrics


# ============================================================
# Helper: aggregate per-controller metrics
# ============================================================

def _aggregate_ctrl_metrics(ctrl_metrics: dict) -> dict:
    """Average metrics across controllers."""
    if not ctrl_metrics:
        return {}
    all_keys = set()
    for m in ctrl_metrics.values():
        all_keys.update(m.keys())
    result = {}
    for key in all_keys:
        vals = [m[key] for m in ctrl_metrics.values() if key in m]
        result[key] = np.mean(vals) if vals else 0.0
    return result


# ============================================================
# Factory
# ============================================================

# Need F import for MAPPO loss computation
import torch.nn.functional as F
import torch.nn as nn

WRAPPER_REGISTRY = {
    'ea': MPE_EA,
    'maddpg': MPE_MADDPG,
    'matd3': MPE_MATD3,
    'mappo': MPE_MAPPO,
    'maippo': MPE_MAIPPO,
    'maac': MPE_MAAC,
    'sqddpg': MPE_SQDDPG,
    'agile': MPE_AGILE,
    'maddpg_set': MPE_MADDPG_Set,
    'maac_set': MPE_MAAC_Set,
    'mappo_set': MPE_MAPPO_Set,
}

# Ablation wrappers (lazy import to avoid circular deps)
def _get_ablation_wrapper(name):
    if name == 'ea_no_pairset':
        from Modified_MPE.mpe_churn.ablations.ea_mpe_no_pairset import MPE_EA_NoPairSet
        return MPE_EA_NoPairSet
    elif name == 'ea_no_tdconsistent':
        from Modified_MPE.mpe_churn.ablations.ea_mpe_no_tdconsistent import MPE_EA_NoTDConsistent
        return MPE_EA_NoTDConsistent
    elif name == 'ea_no_credit':
        from Modified_MPE.mpe_churn.ablations.ea_mpe_no_credit import MPE_EA_NoCredit
        return MPE_EA_NoCredit
    return None


def create_mpe_agent(algo_name: str, device: str = "cpu") -> MPEAlgoWrapper:
    """Factory function to create algorithm wrapper by name."""
    if algo_name in WRAPPER_REGISTRY:
        return WRAPPER_REGISTRY[algo_name](device=device)
    # Check ablation wrappers
    ablation_cls = _get_ablation_wrapper(algo_name)
    if ablation_cls is not None:
        return ablation_cls(device=device)
    raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(WRAPPER_REGISTRY.keys()) + ['ea_no_pairset', 'ea_no_tdconsistent', 'ea_no_credit']}")

