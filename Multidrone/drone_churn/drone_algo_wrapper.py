"""
Algorithm wrappers for Multidrone experiments.

Structurally identical to MPE wrappers (mpe_algo_wrapper.py), but uses
Drone dimensions: x_dim=9, g_dim=17, p=4 (3D physics).
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Multidrone.drone_churn.churn_config import N_CONTROLLERS, N_MAX_PER, X_DIM, G_DIM, P_DIM
from Multidrone.drone_churn.drone_configs import (
    EA_CONFIG, MADDPG_CONFIG, MATD3_CONFIG, MAAC_CONFIG,
    AGILE_CONFIG, SQDDPG_CONFIG, MAPPO_CONFIG, MAIPPO_CONFIG,
    MADDPG_SET_CONFIG, MAAC_SET_CONFIG, MAPPO_SET_CONFIG,
    PADDING_STATE_DIM, PADDING_ACTION_DIM,
)


# ============================================================
# Base + Helpers (same structure as MPE)
# ============================================================

class DroneAlgoWrapper(ABC):
    def __init__(self, device="cpu"):
        self.device = device
        self.ctrl_ids = [f"ctrl_{i}" for i in range(N_CONTROLLERS)]
        self._last_metrics = {}

    @abstractmethod
    def select_action(self, obs_dict, explore=True): pass
    @abstractmethod
    def store_and_update(self, obs, actions, rewards, next_obs, done, info): pass
    def set_episode(self, episode): pass
    def get_last_metrics(self): return self._last_metrics


def flatten_obs(obs_dict, ctrl_ids):
    states = np.zeros((len(ctrl_ids), PADDING_STATE_DIM))
    for i, cid in enumerate(ctrl_ids):
        X, g, mask = obs_dict[cid]
        states[i] = np.concatenate([X.flatten(), g, mask])
    return states


def flatten_acts(actions_dict, ctrl_ids):
    acts = np.zeros((len(ctrl_ids), PADDING_ACTION_DIM))
    for i, cid in enumerate(ctrl_ids):
        acts[i] = actions_dict[cid].flatten()
    return acts


def unflatten_acts(flat, obs_dict, ctrl_ids):
    result = {}
    for i, cid in enumerate(ctrl_ids):
        _, _, mask = obs_dict[cid]
        a = flat[i].reshape(N_MAX_PER, P_DIM)
        a *= mask[:, np.newaxis]
        result[cid] = a
    return result


def _agg_metrics(ctrl_metrics):
    if not ctrl_metrics: return {}
    keys = set()
    for m in ctrl_metrics.values(): keys.update(m.keys())
    return {k: np.mean([m[k] for m in ctrl_metrics.values() if k in m]) for k in keys}


# ============================================================
# EA
# ============================================================

class Drone_EA(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        from algorithms.EA.foea.ea_agent import EAAgent
        cfg = EA_CONFIG
        self.agents = {}
        for i, cid in enumerate(self.ctrl_ids):
            self.agents[cid] = EAAgent(
                x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
                N_max=cfg['N_max'], num_managers=cfg['num_managers'],
                emb_dim=cfg['emb_dim'], token_dim=cfg['token_dim'], hidden_dim=cfg['hidden_dim'],
                gamma=cfg['gamma'], tau=cfg['tau'],
                lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
                policy_delay=cfg['policy_delay'], noise_scale=cfg['noise_scale'],
                noise_clip=cfg['noise_clip'], advantage_tau=cfg['advantage_tau'],
                buffer_capacity=cfg['buffer_capacity'], device=device,
            )

    def select_action(self, obs_dict, explore=True):
        actions = {}
        for i, cid in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[cid]
            actions[cid] = self.agents[cid].select_action(g=g, X=X, mask=mask, manager_id=i, explore=explore)
        return actions

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        all_m = {}
        for i, cid in enumerate(self.ctrl_ids):
            X, g, mask = obs[cid]; Xn, gn, mn = next_obs[cid]
            self.agents[cid].store_transition(
                manager_id=i, g=g, X=X, mask=mask, A=actions[cid],
                r=rewards[cid], g_next=gn, X_next=Xn, mask_next=mn, done=done)
            m = self.agents[cid].update(batch_size=EA_CONFIG['batch_size'])
            if m: all_m[cid] = m
        self._last_metrics = _agg_metrics(all_m)
        return self._last_metrics

    def set_episode(self, ep):
        for a in self.agents.values(): a.set_episode(ep)


# ============================================================
# Flat baselines: MADDPG, MATD3, SQDDPG
# ============================================================

class Drone_MADDPG(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        from algorithms.MADDPG.fomaddpg.fomaddpg import FOMADDPG
        cfg = MADDPG_CONFIG
        self.algo = FOMADDPG(n_agents=cfg['n_agents'], state_dim=cfg['state_dim'],
            action_dim=cfg['action_dim'], lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            hidden_dim=cfg['hidden_dim'], gamma=cfg['gamma'], tau=cfg['tau'],
            noise_scale=cfg['noise_scale'], buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'], device=device)

    def select_action(self, obs_dict, explore=True):
        return unflatten_acts(self.algo.select_actions(flatten_obs(obs_dict, self.ctrl_ids), explore), obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        self.algo.store_experience(flatten_obs(obs, self.ctrl_ids), flatten_acts(actions, self.ctrl_ids),
            np.array([rewards[k] for k in self.ctrl_ids]), flatten_obs(next_obs, self.ctrl_ids), np.array([done]*N_CONTROLLERS))
        m = self.algo.update()
        self._last_metrics = m if m else {}; return self._last_metrics


class Drone_MATD3(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        from algorithms.MATD3.fomatd3.fomatd3 import FOMATD3
        cfg = MATD3_CONFIG
        self.algo = FOMATD3(n_agents=cfg['n_agents'], state_dim=cfg['state_dim'],
            action_dim=cfg['action_dim'], lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            hidden_dim=cfg['hidden_dim'], gamma=cfg['gamma'], tau=cfg['tau'],
            noise_scale=cfg['noise_scale'], buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'], device=device)

    def select_action(self, obs_dict, explore=True):
        return unflatten_acts(self.algo.select_actions(flatten_obs(obs_dict, self.ctrl_ids), explore), obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        self.algo.store_experience(flatten_obs(obs, self.ctrl_ids), flatten_acts(actions, self.ctrl_ids),
            np.array([rewards[k] for k in self.ctrl_ids]), flatten_obs(next_obs, self.ctrl_ids), np.array([done]*N_CONTROLLERS))
        m = self.algo.update()
        self._last_metrics = m if m else {}; return self._last_metrics


class Drone_SQDDPG(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        from algorithms.SQDDPG.fosqddpg.fosqddpg import FOSQDDPG
        cfg = SQDDPG_CONFIG
        self.algo = FOSQDDPG(n_agents=cfg['n_agents'], state_dim=cfg['state_dim'],
            action_dim=cfg['action_dim'], lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            hidden_dim=cfg['hidden_dim'], gamma=cfg['gamma'], tau=cfg['tau'],
            noise_scale=cfg['noise_scale'], buffer_capacity=cfg['buffer_capacity'],
            batch_size=cfg['batch_size'], sample_size=cfg['sample_size'], device=device)

    def select_action(self, obs_dict, explore=True):
        return unflatten_acts(self.algo.select_actions(flatten_obs(obs_dict, self.ctrl_ids), explore), obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        self.algo.store_experience(flatten_obs(obs, self.ctrl_ids), flatten_acts(actions, self.ctrl_ids),
            np.array([rewards[k] for k in self.ctrl_ids]), flatten_obs(next_obs, self.ctrl_ids), np.array([done]*N_CONTROLLERS))
        m = self.algo.update()
        self._last_metrics = m if m else {}; return self._last_metrics


# ============================================================
# Structured baselines: MAAC, AGILE
# ============================================================

def _build_structured(obs_dict, ctrl_ids):
    ds = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
    gf = np.zeros((N_CONTROLLERS, G_DIM))
    ms = np.zeros((N_CONTROLLERS, N_MAX_PER))
    for i, cid in enumerate(ctrl_ids):
        X, g, mask = obs_dict[cid]
        ds[i] = X; gf[i] = g; ms[i] = mask
    return ds, gf, ms


class Drone_MAAC(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        from algorithms.MAAC.fomaac.fomaac import FOMAAC
        cfg = MAAC_CONFIG
        self.algo = FOMAAC(n_agents=cfg['n_agents'], N_max=cfg['N_max'],
            device_dim=cfg['device_dim'], global_dim=cfg['global_dim'], action_dim=cfg['action_dim'],
            hidden_dim=cfg['hidden_dim'], attend_heads=cfg['attend_heads'],
            lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            gamma=cfg['gamma'], tau=cfg['tau'], noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'], batch_size=cfg['batch_size'], device=device)

    def select_action(self, obs_dict, explore=True):
        ds, gf, ms = _build_structured(obs_dict, self.ctrl_ids)
        raw = self.algo.select_actions(ds, gf, ms, add_noise=explore)
        return {cid: raw[i] for i, cid in enumerate(self.ctrl_ids)}

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        ds, gf, ms = _build_structured(obs, self.ctrl_ids)
        nds, ngf, nms = _build_structured(next_obs, self.ctrl_ids)
        act_arr = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        rew_list = []
        for i, cid in enumerate(self.ctrl_ids):
            act_arr[i] = actions[cid]; rew_list.append(rewards[cid])
        self.algo.store_experience(ds, gf, ms, act_arr, rew_list, nds, ngf, nms, done)
        m = self.algo.update()
        self._last_metrics = m if m else {}; return self._last_metrics


class Drone_AGILE(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        from algorithms.AGILE.foagile.foagile import FOAGILE
        cfg = AGILE_CONFIG
        self.algo = FOAGILE(n_agents=cfg['n_agents'], N_max=cfg['N_max'],
            device_dim=cfg['device_dim'], global_dim=cfg['global_dim'], action_dim=cfg['action_dim'],
            hidden_dim=cfg['hidden_dim'], lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            gamma=cfg['gamma'], tau=cfg['tau'], noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'], batch_size=cfg['batch_size'], device=device)

    def select_action(self, obs_dict, explore=True):
        ds, gf, ms = _build_structured(obs_dict, self.ctrl_ids)
        raw = self.algo.select_actions(ds, gf, ms, add_noise=explore)
        return {cid: raw[i] for i, cid in enumerate(self.ctrl_ids)}

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        ds, gf, ms = _build_structured(obs, self.ctrl_ids)
        nds, ngf, nms = _build_structured(next_obs, self.ctrl_ids)
        for i, cid in enumerate(self.ctrl_ids):
            self.algo.store_experience(agent_idx=i, device_states=ds[i], global_feat=gf[i],
                mask=ms[i], actions=actions[cid], reward=rewards[cid],
                next_device_states=nds[i], next_global_feat=ngf[i], next_mask=nms[i], done=done)
        m = self.algo.update()
        self._last_metrics = m if m else {}; return self._last_metrics


# ============================================================
# MAPPO / MAIPPO (on-policy, same as MPE version but with drone dims)
# ============================================================

class Drone_MAPPO(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        cfg = MAPPO_CONFIG
        self.state_dim = cfg['state_dim']; self.action_dim = cfg['action_dim']
        self.gamma = cfg['gamma']; self.gae_lambda = cfg['gae_lambda']
        self.clip_epsilon = cfg['clip_epsilon']; self.entropy_coef = cfg['entropy_coef']
        self.value_loss_coef = cfg['value_loss_coef']; self.max_grad_norm = cfg['max_grad_norm']
        self.n_epochs = cfg['n_epochs']; self.dev = torch.device(device)
        h = cfg['hidden_dim']
        self.actor = nn.Sequential(nn.Linear(self.state_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh()).to(self.dev)
        self.action_mean = nn.Linear(h, self.action_dim).to(self.dev)
        self.action_log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.dev))
        self.critic = nn.Sequential(nn.Linear(self.state_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)).to(self.dev)
        params = list(self.actor.parameters()) + list(self.action_mean.parameters()) + [self.action_log_std] + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg['lr'])
        self.rollout = []

    def _dist(self, s):
        h = self.actor(s); m = self.action_mean(h)
        return torch.distributions.Normal(m, self.action_log_std.exp().expand_as(m))

    def select_action(self, obs_dict, explore=True):
        s = torch.FloatTensor(flatten_obs(obs_dict, self.ctrl_ids)).to(self.dev)
        with torch.no_grad():
            d = self._dist(s); raw = d.sample() if explore else d.mean
            raw = torch.clamp(raw, -1, 1)
        return unflatten_acts(raw.cpu().numpy(), obs_dict, self.ctrl_ids)

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        # Scale rewards to prevent NaN (Multidrone rewards ~-1000 to -3000)
        scaled_rewards = {k: v / 1000.0 for k, v in rewards.items()}
        self.rollout.append((flatten_obs(obs, self.ctrl_ids), flatten_acts(actions, self.ctrl_ids),
                             np.array([scaled_rewards[k] for k in self.ctrl_ids]), done))
        if done and self.rollout:
            m = self._ppo_update(); self._last_metrics = m; self.rollout = []; return m
        return {}

    def _ppo_update(self):
        all_s, all_a, all_ret, all_adv = [], [], [], []
        for ai in range(N_CONTROLLERS):
            ss = [s[ai] for s,_,_,_ in self.rollout]; aa = [a[ai] for _,a,_,_ in self.rollout]; rr = [r[ai] for _,_,r,_ in self.rollout]
            st = torch.FloatTensor(np.array(ss)).to(self.dev)
            with torch.no_grad(): vals = self.critic(st).squeeze(-1).cpu().numpy()
            T = len(rr); adv = np.zeros(T); lg = 0
            for t in reversed(range(T)):
                nv = vals[t+1] if t+1<T else 0; d = rr[t]+self.gamma*nv-vals[t]; lg = d+self.gamma*self.gae_lambda*lg; adv[t] = lg
            all_s.extend(ss); all_a.extend(aa); all_ret.extend((adv+vals).tolist()); all_adv.extend(adv.tolist())
        st = torch.FloatTensor(np.array(all_s)).to(self.dev); at = torch.FloatTensor(np.array(all_a)).to(self.dev)
        rt = torch.FloatTensor(all_ret).to(self.dev); advt = torch.FloatTensor(all_adv).to(self.dev)
        advt = (advt - advt.mean())/(advt.std()+1e-8)
        with torch.no_grad(): old_lp = self._dist(st).log_prob(at).sum(-1)
        tl = 0
        for _ in range(self.n_epochs):
            d = self._dist(st); lp = d.log_prob(at).sum(-1); ratio = (lp-old_lp).exp()
            s1 = ratio*advt; s2 = torch.clamp(ratio,1-self.clip_epsilon,1+self.clip_epsilon)*advt
            pl = -torch.min(s1,s2).mean(); vl = F.mse_loss(self.critic(st).squeeze(-1), rt)
            ent = d.entropy().sum(-1).mean(); loss = pl+self.value_loss_coef*vl-self.entropy_coef*ent
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.action_mean.parameters())+[self.action_log_std]+list(self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step(); tl += loss.item()
        return {'policy_loss': tl/self.n_epochs}


class Drone_MAIPPO(Drone_MAPPO):
    """Independent PPO: separate policy per controller."""
    def __init__(self, device="cpu"):
        DroneAlgoWrapper.__init__(self, device)
        cfg = MAIPPO_CONFIG
        self.state_dim = cfg['state_dim']; self.action_dim = cfg['action_dim']
        self.gamma = cfg['gamma']; self.gae_lambda = cfg['gae_lambda']
        self.clip_epsilon = cfg['clip_epsilon']; self.entropy_coef = cfg['entropy_coef']
        self.value_loss_coef = cfg['value_loss_coef']; self.max_grad_norm = cfg['max_grad_norm']
        self.n_epochs = cfg['n_epochs']; self.dev = torch.device(device)
        h = cfg['hidden_dim']
        self.actors_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh()) for _ in range(N_CONTROLLERS)]).to(self.dev)
        self.action_means_list = nn.ModuleList([nn.Linear(h, self.action_dim) for _ in range(N_CONTROLLERS)]).to(self.dev)
        self.action_log_stds_list = nn.ParameterList([nn.Parameter(torch.zeros(self.action_dim,device=self.dev)) for _ in range(N_CONTROLLERS)])
        self.critics_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1)) for _ in range(N_CONTROLLERS)]).to(self.dev)
        params = list(self.actors_list.parameters())+list(self.action_means_list.parameters())+list(self.action_log_stds_list.parameters())+list(self.critics_list.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg['lr'])
        self.rollout = []; self.actor = None; self.action_mean = None; self.action_log_std = None; self.critic = None

    def _dist_i(self, s, i):
        h = self.actors_list[i](s); m = self.action_means_list[i](h)
        return torch.distributions.Normal(m, self.action_log_stds_list[i].exp().expand_as(m))

    def select_action(self, obs_dict, explore=True):
        s = flatten_obs(obs_dict, self.ctrl_ids); rf = np.zeros((N_CONTROLLERS, PADDING_ACTION_DIM))
        for i in range(N_CONTROLLERS):
            st = torch.FloatTensor(s[i:i+1]).to(self.dev)
            with torch.no_grad():
                d = self._dist_i(st, i); raw = d.sample() if explore else d.mean; raw = torch.clamp(raw,-1,1)
            rf[i] = raw.cpu().numpy()[0]
        return unflatten_acts(rf, obs_dict, self.ctrl_ids)

    def _ppo_update(self):
        tl = 0
        for ai in range(N_CONTROLLERS):
            ss = [s[ai] for s,_,_,_ in self.rollout]; aa = [a[ai] for _,a,_,_ in self.rollout]; rr = [r[ai] for _,_,r,_ in self.rollout]
            st = torch.FloatTensor(np.array(ss)).to(self.dev); at = torch.FloatTensor(np.array(aa)).to(self.dev)
            with torch.no_grad(): vals = self.critics_list[ai](st).squeeze(-1).cpu().numpy()
            T = len(rr); adv = np.zeros(T); lg = 0
            for t in reversed(range(T)):
                nv = vals[t+1] if t+1<T else 0; d = rr[t]+self.gamma*nv-vals[t]; lg = d+self.gamma*self.gae_lambda*lg; adv[t] = lg
            rt = torch.FloatTensor(adv+vals).to(self.dev); advt = torch.FloatTensor(adv).to(self.dev)
            advt = (advt-advt.mean())/(advt.std()+1e-8)
            with torch.no_grad(): old_lp = self._dist_i(st, ai).log_prob(at).sum(-1)
            for _ in range(self.n_epochs):
                dist = self._dist_i(st, ai); lp = dist.log_prob(at).sum(-1); ratio = (lp-old_lp).exp()
                s1 = ratio*advt; s2 = torch.clamp(ratio,1-self.clip_epsilon,1+self.clip_epsilon)*advt
                pl = -torch.min(s1,s2).mean(); vl = F.mse_loss(self.critics_list[ai](st).squeeze(-1), rt)
                ent = dist.entropy().sum(-1).mean(); loss = pl+self.value_loss_coef*vl-self.entropy_coef*ent
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); tl += loss.item()
        return {'policy_loss': tl/(N_CONTROLLERS*self.n_epochs)}


# ============================================================
# Factory
# ============================================================

class Drone_MAAC_Set(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
        super().__init__(device)
        from algorithms.MAAC.fomaac.maac_set import MAACSetAgent
        cfg = MAAC_SET_CONFIG
        self.algo = MAACSetAgent(
            x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
            N_max=cfg['N_max'], num_managers=cfg['num_managers'],
            emb_dim=cfg['emb_dim'], token_dim=cfg['token_dim'],
            hidden_dim=cfg['hidden_dim'], attend_heads=cfg['attend_heads'],
            gamma=cfg['gamma'], tau=cfg['tau'],
            lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
            noise_scale=cfg['noise_scale'],
            buffer_capacity=cfg['buffer_capacity'], batch_size=cfg['batch_size'],
            device=device)

    def select_action(self, obs_dict, explore=True):
        ds, gf, ms = _build_structured(obs_dict, self.ctrl_ids)
        raw = self.algo.select_actions(ds, gf, ms, add_noise=explore)
        return {cid: raw[i] for i, cid in enumerate(self.ctrl_ids)}

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        ds, gf, ms = _build_structured(obs, self.ctrl_ids)
        nds, ngf, nms = _build_structured(next_obs, self.ctrl_ids)
        act_arr = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        rew_list = []
        for i, cid in enumerate(self.ctrl_ids):
            act_arr[i] = actions[cid]; rew_list.append(rewards[cid])
        self.algo.store_experience(ds, gf, ms, act_arr, rew_list, nds, ngf, nms, done)
        m = self.algo.update()
        self._last_metrics = m if m else {}; return self._last_metrics


class Drone_MADDPG_Set(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
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
            buffer_capacity=cfg['buffer_capacity'], batch_size=cfg['batch_size'],
            device=device)

    def _build_arrays(self, obs_dict):
        all_X = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
        all_g = np.zeros((N_CONTROLLERS, G_DIM))
        all_mask = np.zeros((N_CONTROLLERS, N_MAX_PER))
        for i, cid in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[cid]
            all_X[i] = X; all_g[i] = g; all_mask[i] = mask
        return all_g, all_X, all_mask

    def select_action(self, obs_dict, explore=True):
        all_g, all_X, all_mask = self._build_arrays(obs_dict)
        result = {}
        for i, cid in enumerate(self.ctrl_ids):
            result[cid] = self.algo.select_action(
                g=all_g[i], X=all_X[i], mask=all_mask[i],
                manager_id=i, explore=explore)
        return result

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        all_g, all_X, all_mask = self._build_arrays(obs)
        all_g_n, all_X_n, all_mask_n = self._build_arrays(next_obs)
        all_A = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        for i, cid in enumerate(self.ctrl_ids):
            all_A[i] = actions[cid]
        avg_reward = np.mean([rewards[c] for c in self.ctrl_ids])
        self.algo.store_transition(
            all_g, all_X, all_mask, all_A, avg_reward,
            all_g_n, all_X_n, all_mask_n, float(done))
        m = self.algo.update()
        self._last_metrics = m if m else {}; return self._last_metrics


class Drone_MAPPO_Set(DroneAlgoWrapper):
    def __init__(self, device="cpu"):
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
            device=device)

    def _build_arrays(self, obs_dict):
        all_X = np.zeros((N_CONTROLLERS, N_MAX_PER, X_DIM))
        all_g = np.zeros((N_CONTROLLERS, G_DIM))
        all_mask = np.zeros((N_CONTROLLERS, N_MAX_PER))
        for i, cid in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[cid]
            all_X[i] = X; all_g[i] = g; all_mask[i] = mask
        return all_g, all_X, all_mask

    def select_action(self, obs_dict, explore=True):
        result = {}
        self._last_log_probs = {}
        for i, cid in enumerate(self.ctrl_ids):
            X, g, mask = obs_dict[cid]
            a, log_prob = self.algo.select_action(
                g=g, X=X, mask=mask, manager_id=i, deterministic=(not explore))
            result[cid] = a
            self._last_log_probs[cid] = log_prob
        return result

    def store_and_update(self, obs, actions, rewards, next_obs, done, info):
        all_g, all_X, all_mask = self._build_arrays(obs)
        all_A = np.zeros((N_CONTROLLERS, N_MAX_PER, P_DIM))
        for i, cid in enumerate(self.ctrl_ids):
            all_A[i] = actions[cid]
        avg_reward = np.mean([rewards[c] for c in self.ctrl_ids])
        total_log_prob = sum(self._last_log_probs.get(c, 0.0) for c in self.ctrl_ids)
        value = self.algo.get_value(all_g, all_X, all_mask)
        self.algo.store_transition(
            all_g, all_X, all_mask, all_A,
            log_prob=total_log_prob, reward=avg_reward,
            value=value, done=float(done))
        if done:
            m = self.algo.update()
            self._last_metrics = m if m else {}
        return self._last_metrics


WRAPPER_REGISTRY = {
    'ea': Drone_EA, 'maddpg': Drone_MADDPG, 'matd3': Drone_MATD3,
    'mappo': Drone_MAPPO, 'maippo': Drone_MAIPPO, 'maac': Drone_MAAC,
    'sqddpg': Drone_SQDDPG, 'agile': Drone_AGILE,
    'maac_set': Drone_MAAC_Set, 'maddpg_set': Drone_MADDPG_Set, 'mappo_set': Drone_MAPPO_Set,
}

def _get_ablation(name):
    # Ablation wrappers reuse EA pattern with ablation agents
    if name == 'ea_no_pairset':
        class _W(Drone_EA):
            def __init__(self, device="cpu"):
                DroneAlgoWrapper.__init__(self, device)
                from Test.Ablation.agents.ea_no_pairset import EAAgentNoPairSet
                cfg = EA_CONFIG
                self.agents = {cid: EAAgentNoPairSet(x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
                    N_max=cfg['N_max'], num_managers=cfg['num_managers'], emb_dim=cfg['emb_dim'],
                    token_dim=cfg['token_dim'], hidden_dim=cfg['hidden_dim'], gamma=cfg['gamma'],
                    tau=cfg['tau'], lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
                    policy_delay=cfg['policy_delay'], noise_scale=cfg['noise_scale'],
                    noise_clip=cfg['noise_clip'], advantage_tau=cfg['advantage_tau'],
                    buffer_capacity=cfg['buffer_capacity'], device=device) for cid in self.ctrl_ids}
        return _W
    elif name == 'ea_no_tdconsistent':
        class _W(Drone_EA):
            def __init__(self, device="cpu"):
                DroneAlgoWrapper.__init__(self, device)
                from Test.Ablation.agents.ea_no_tdconsistent import EAAgentNoTDConsistent
                cfg = EA_CONFIG
                self.agents = {cid: EAAgentNoTDConsistent(x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
                    N_max=cfg['N_max'], num_managers=cfg['num_managers'], emb_dim=cfg['emb_dim'],
                    token_dim=cfg['token_dim'], hidden_dim=cfg['hidden_dim'], gamma=cfg['gamma'],
                    tau=cfg['tau'], lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
                    policy_delay=cfg['policy_delay'], noise_scale=cfg['noise_scale'],
                    noise_clip=cfg['noise_clip'], advantage_tau=cfg['advantage_tau'],
                    buffer_capacity=cfg['buffer_capacity'], device=device) for cid in self.ctrl_ids}
        return _W
    elif name == 'ea_no_credit':
        class _W(Drone_EA):
            def __init__(self, device="cpu"):
                DroneAlgoWrapper.__init__(self, device)
                from Test.Ablation.agents.ea_no_credit import EAAgentNoCredit
                cfg = EA_CONFIG
                self.agents = {cid: EAAgentNoCredit(x_dim=cfg['x_dim'], g_dim=cfg['g_dim'], p=cfg['p'],
                    N_max=cfg['N_max'], num_managers=cfg['num_managers'], emb_dim=cfg['emb_dim'],
                    token_dim=cfg['token_dim'], hidden_dim=cfg['hidden_dim'], gamma=cfg['gamma'],
                    tau=cfg['tau'], lr_actor=cfg['lr_actor'], lr_critic=cfg['lr_critic'],
                    policy_delay=cfg['policy_delay'], noise_scale=cfg['noise_scale'],
                    noise_clip=cfg['noise_clip'], advantage_tau=cfg['advantage_tau'],
                    buffer_capacity=cfg['buffer_capacity'], device=device) for cid in self.ctrl_ids}
        return _W
    return None


def create_drone_agent(algo_name, device="cpu"):
    if algo_name in WRAPPER_REGISTRY:
        return WRAPPER_REGISTRY[algo_name](device=device)
    cls = _get_ablation(algo_name)
    if cls: return cls(device=device)
    raise ValueError(f"Unknown: {algo_name}. Available: {list(WRAPPER_REGISTRY.keys()) + ['ea_no_pairset','ea_no_tdconsistent','ea_no_credit']}")
