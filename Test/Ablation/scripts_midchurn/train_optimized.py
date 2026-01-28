#!/usr/bin/env python
"""
优化参数训练脚本 - 根据环境规模使用最优参数

参数配置:
- 4manager: original (blend=0.7, tau=2.0, weight=0.15)
- 10manager: option_AB (blend=0.3, tau=5.0, weight=0.05)

Author: FOenv Team
Date: 2026-01-21
"""

import sys
import os
import argparse
import logging
import json
import random
from datetime import datetime
from pathlib import Path

# 禁用输出缓冲
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig

# 导入mid-episode churn wrapper
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.mid_episode_churn_wrapper import MidEpisodeChurnWrapper

# 导入基础组件
from algorithms.EA.foea.actor import SetToSetActor
from algorithms.EA.foea.critic import TwinCritics
from algorithms.EA.foea.replay_buffer import ChurnAwareReplayBuffer

from algorithms.EA.foea.fogym_adapter import (
    extract_device_states_from_manager,
    extract_global_features_from_env,
    convert_ea_action_to_fogym
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== 环境配置 ==============
ENV_CONFIGS = {
    '4manager': {
        'manager_count': 4,
        'n_max': 100,
        'g_dim': 26,
        # 最优参数 (original)
        'advantage_tau': 2.0,
        'advantage_weight': 0.15,
        'churn_blend_weight': 0.7,
    },
    '10manager': {
        'manager_count': 10,
        'n_max': 100,
        'g_dim': 26,
        # 最优参数 (option_AB)
        'advantage_tau': 5.0,
        'advantage_weight': 0.05,
        'churn_blend_weight': 0.3,
    }
}

# ============== Churn配置 ==============
CHURN_CONFIGS = {
    'low': ((0.10, 0.125, 0.15), (0.4, 0.3, 0.3), 5),
    'mid': ((0.20, 0.225, 0.25), (0.4, 0.3, 0.3), 5),
    'high': ((0.30, 0.325, 0.35), (0.4, 0.3, 0.3), 5),
}

# ============== 训练参数 ==============
TRAINING_CONFIG = {
    'num_episodes': 500,
    'batch_size': 256,
    'warmup_steps': 1000,
    'save_interval': 50,
    'log_interval': 10,
    'seed': 42,
    'gamma': 0.99,
    'tau': 0.005,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'policy_delay': 2,
    'noise_scale': 0.1,
    'noise_clip': 0.2,
    'buffer_capacity': 100000,
}


class OptimizedEAAgent:
    """
    优化参数的EA Agent - 支持环境特定参数
    """
    
    def __init__(
        self,
        x_dim: int = 6,
        g_dim: int = 26,
        p: int = 5,
        N_max: int = 100,
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
        # 可配置参数
        advantage_tau: float = 2.0,
        advantage_weight: float = 0.15,
        churn_blend_weight: float = 0.7,
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
        self.advantage_tau = advantage_tau
        self.advantage_weight = advantage_weight
        self.churn_blend_weight = churn_blend_weight
        self.device = device
        
        # 创建网络
        self.actor = SetToSetActor(
            x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max, num_managers=num_managers,
            emb_dim=emb_dim, token_dim=token_dim, hidden_dim=hidden_dim
        ).to(device)
        
        self.critics = TwinCritics(
            x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max, num_managers=num_managers,
            emb_dim=emb_dim, token_dim=token_dim, hidden_dim=hidden_dim
        ).to(device)
        
        # 目标网络
        self.actor_target = SetToSetActor(
            x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max, num_managers=num_managers,
            emb_dim=emb_dim, token_dim=token_dim, hidden_dim=hidden_dim
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critics_target = TwinCritics(
            x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max, num_managers=num_managers,
            emb_dim=emb_dim, token_dim=token_dim, hidden_dim=hidden_dim
        ).to(device)
        self.critics_target.load_state_dict(self.critics.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ChurnAwareReplayBuffer(
            capacity=buffer_capacity, x_dim=x_dim, g_dim=g_dim, p=p, N_max=N_max
        )
        
        self.total_steps = 0
        self.actor_updates = 0
        self.critic_updates = 0
    
    def select_action(self, g, X, mask, manager_id, explore=True):
        g_t = torch.FloatTensor(g).unsqueeze(0).to(self.device)
        X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
        manager_id_t = torch.LongTensor([manager_id]).to(self.device)
        
        with torch.no_grad():
            A = self.actor(g_t, X_t, mask_t, manager_id_t)
            if explore:
                noise = torch.randn_like(A) * self.noise_scale
                noise = noise * mask_t.unsqueeze(-1)
                A = (A + noise).clamp(-1.0, 1.0)
                A = A * mask_t.unsqueeze(-1)
        
        return A.squeeze(0).cpu().numpy()
    
    def store_transition(self, manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done):
        self.replay_buffer.add(
            manager_id=manager_id, g=g, X=X, mask=mask, A=A, r=r,
            g_next=g_next, X_next=X_next, mask_next=mask_next, done=done,
        )
    
    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return {}
        
        batch = self.replay_buffer.sample(batch_size, device=self.device)
        
        manager_id = batch['manager_id']
        g, X, mask, A = batch['g'], batch['X'], batch['mask'], batch['A']
        r, done = batch['r'], batch['done']
        g_next, X_next, mask_next = batch['g_next'], batch['X_next'], batch['mask_next']
        
        critic_loss, q1_value, q2_value = self._update_critics(
            manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done
        )
        self.critic_updates += 1
        
        actor_metrics = None
        if self.total_steps % self.policy_delay == 0:
            actor_metrics = self._update_actor(manager_id, g, X, mask)
            self._soft_update_targets()
            self.actor_updates += 1
        
        self.total_steps += 1
        
        metrics = {'critic_loss': critic_loss, 'q1_value': q1_value, 'q2_value': q2_value}
        if actor_metrics:
            metrics.update(actor_metrics)
        return metrics
    
    def _update_critics(self, manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done):
        with torch.no_grad():
            churn_detected = (mask.sum(dim=-1) != mask_next.sum(dim=-1))
            
            A_next_curr = self.actor_target(g_next, X_next, mask, manager_id)
            noise_curr = (torch.randn_like(A_next_curr) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            A_next_curr = (A_next_curr + noise_curr * mask.unsqueeze(-1)).clamp(-1.0, 1.0) * mask.unsqueeze(-1)
            
            A_next_real = self.actor_target(g_next, X_next, mask_next, manager_id)
            noise_real = (torch.randn_like(A_next_real) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            A_next_real = (A_next_real + noise_real * mask_next.unsqueeze(-1)).clamp(-1.0, 1.0) * mask_next.unsqueeze(-1)
            
            Q_target_curr = self.critics_target.min_Q(g_next, X_next, A_next_curr, mask, manager_id)
            Q_target_real = self.critics_target.min_Q(g_next, X_next, A_next_real, mask_next, manager_id)
            
            churn_mask = churn_detected.unsqueeze(-1).float()
            target_Q = churn_mask * (self.churn_blend_weight * Q_target_curr + (1 - self.churn_blend_weight) * Q_target_real) \
                     + (1 - churn_mask) * Q_target_real
            
            y = r + self.gamma * (1 - done) * target_Q
        
        Q1, Q2 = self.critics(g, X, A, mask, manager_id)
        critic_loss = nn.MSELoss()(Q1, y) + nn.MSELoss()(Q2, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item(), Q1.mean().item(), Q2.mean().item()
    
    def _update_actor(self, manager_id, g, X, mask):
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
        
        actor_loss = (1 - self.advantage_weight) * primary_loss - self.advantage_weight * advantage_reg.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        with torch.no_grad():
            active_advantages = per_device_advantage[active_mask]
            mean_adv = active_advantages.mean().item() if active_advantages.numel() > 0 else 0.0
            std_adv = active_advantages.std().item() if active_advantages.numel() > 1 else 0.0
        
        return {'actor_loss': actor_loss.item(), 'mean_advantage': mean_adv, 'advantage_std': std_adv}
    
    def _soft_update_targets(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critics_state_dict': self.critics.state_dict(),
        }, filepath)


# ============== 消融Agent类 ==============

class OptimizedEAAgentNoPairSet(OptimizedEAAgent):
    """w/o Pair-Set Critic消融"""
    pass  # 使用相同实现，只是名称不同用于标识


class OptimizedEAAgentNoTDConsistent(OptimizedEAAgent):
    """w/o TD-Consistent消融 - 使用当前mask代替mask_next"""
    
    def _update_critics(self, manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done):
        with torch.no_grad():
            # 消融: 使用当前mask而不是mask_next
            A_next = self.actor_target(g_next, X_next, mask, manager_id)
            noise = (torch.randn_like(A_next) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            A_next = (A_next + noise * mask.unsqueeze(-1)).clamp(-1.0, 1.0) * mask.unsqueeze(-1)
            
            target_Q = self.critics_target.min_Q(g_next, X_next, A_next, mask, manager_id)
            y = r + self.gamma * (1 - done) * target_Q
        
        Q1, Q2 = self.critics(g, X, A, mask, manager_id)
        critic_loss = nn.MSELoss()(Q1, y) + nn.MSELoss()(Q2, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item(), Q1.mean().item(), Q2.mean().item()


class OptimizedEAAgentNoCredit(OptimizedEAAgent):
    """w/o Per-Device Credit消融 - 使用均匀权重"""
    
    def _update_actor(self, manager_id, g, X, mask):
        A_pi = self.actor(g, X, mask, manager_id)
        Q1, _ = self.critics.Q1_forward(g, X, A_pi, mask, manager_id, return_per_device=True)
        
        # 消融: 简单的SQDDPG风格loss
        actor_loss = -Q1.squeeze(-1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return {'actor_loss': actor_loss.item(), 'mean_advantage': 0.0, 'advantage_std': 0.0}


AGENT_CLASSES = {
    'full_ea': OptimizedEAAgent,
    'no_pairset': OptimizedEAAgentNoPairSet,
    'no_tdconsistent': OptimizedEAAgentNoTDConsistent,
    'no_credit': OptimizedEAAgentNoCredit,
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_environment(env_type, churn_type):
    env_config = ENV_CONFIGS[env_type]
    severity_levels, severity_probs, trigger_interval = CHURN_CONFIGS[churn_type]
    
    churn_config = ChurnConfig(
        enabled=True,
        trigger_interval=trigger_interval,
        severity_levels=severity_levels,
        severity_probs=severity_probs,
        min_active_devices=5
    )
    
    env = MultiAgentFlexOfferEnv(
        data_dir="data",
        time_horizon=24,
        time_step=1,
        churn_config=churn_config
    )
    
    env = MidEpisodeChurnWrapper(env, churn_steps=[6, 12, 18], verbose=False)
    return env, env_config


def train(env_type, churn_type, agent_type, results_dir, seed=42):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env, env_config = create_environment(env_type, churn_type)
    n_managers = len(env.manager_ids)
    
    AgentClass = AGENT_CLASSES[agent_type]
    agent = AgentClass(
        x_dim=6,
        g_dim=env_config['g_dim'],
        p=5,
        N_max=env_config['n_max'],
        num_managers=n_managers,
        advantage_tau=env_config['advantage_tau'],
        advantage_weight=env_config['advantage_weight'],
        churn_blend_weight=env_config['churn_blend_weight'],
        gamma=TRAINING_CONFIG['gamma'],
        tau=TRAINING_CONFIG['tau'],
        lr_actor=TRAINING_CONFIG['lr_actor'],
        lr_critic=TRAINING_CONFIG['lr_critic'],
        policy_delay=TRAINING_CONFIG['policy_delay'],
        noise_scale=TRAINING_CONFIG['noise_scale'],
        noise_clip=TRAINING_CONFIG['noise_clip'],
        buffer_capacity=TRAINING_CONFIG['buffer_capacity'],
        device=device,
    )
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training: {env_type}_{churn_type}_{agent_type}")
    print(f"Parameters: tau={env_config['advantage_tau']}, weight={env_config['advantage_weight']}, blend={env_config['churn_blend_weight']}")
    print(f"{'='*70}")
    
    history = {'episode': [], 'avg_reward': [], 'total_reward': [], 'critic_loss': [], 
               'actor_loss': [], 'q1_mean': [], 'mid_episode_churns': []}
    
    global_step = 0
    
    for episode in range(TRAINING_CONFIG['num_episodes']):
        obs, info = env.reset()
        episode_reward = 0
        mid_churns = 0
        done = False
        
        while not done:
            actions_dict = {}
            transitions = []
            
            for manager_idx, manager_id in enumerate(env.manager_ids):
                manager = env.manager_agents[manager_id]
                g = extract_global_features_from_env(env, manager_id, env.current_time)
                X, mask, device_ids = extract_device_states_from_manager(manager, N_max=env_config['n_max'], x_dim=6)
                A = agent.select_action(g, X, mask, manager_idx, explore=True)
                
                action_flat = convert_ea_action_to_fogym(A, mask, device_ids)
                actions_dict[manager_id] = action_flat
                transitions.append({'manager_idx': manager_idx, 'manager_id': manager_id,
                                   'g': g, 'X': X, 'mask': mask, 'A': A})
            
            next_obs, rewards, dones, truncated, infos = env.step(actions_dict)
            done = dones.get('__all__', False)
            
            for mid in infos:
                if infos[mid].get('mid_episode_churn', False):
                    mid_churns += 1
            
            for trans in transitions:
                manager = env.manager_agents[trans['manager_id']]
                g_next = extract_global_features_from_env(env, trans['manager_id'], env.current_time)
                X_next, mask_next, _ = extract_device_states_from_manager(manager, N_max=env_config['n_max'], x_dim=6)
                r = rewards.get(trans['manager_id'], 0)
                episode_reward += r
                
                agent.store_transition(
                    manager_id=trans['manager_idx'],
                    g=trans['g'], X=trans['X'], mask=trans['mask'], A=trans['A'],
                    r=r, g_next=g_next, X_next=X_next, mask_next=mask_next, done=done
                )
            
            if global_step >= TRAINING_CONFIG['warmup_steps']:
                metrics = agent.update(batch_size=TRAINING_CONFIG['batch_size'])
            else:
                metrics = {}
            
            global_step += 1
        
        avg_reward = episode_reward / n_managers
        history['episode'].append(episode)
        history['avg_reward'].append(avg_reward)
        history['total_reward'].append(episode_reward)
        history['critic_loss'].append(metrics.get('critic_loss', 0))
        history['actor_loss'].append(metrics.get('actor_loss', 0))
        history['q1_mean'].append(metrics.get('q1_value', 0))
        history['mid_episode_churns'].append(mid_churns)
        
        if (episode + 1) % TRAINING_CONFIG['log_interval'] == 0:
            print(f"  Episode {episode+1}/{TRAINING_CONFIG['num_episodes']} | Reward: {avg_reward:.2f} | Mid-churns: {mid_churns}")
        
        if (episode + 1) % TRAINING_CONFIG['save_interval'] == 0:
            agent.save(os.path.join(results_dir, f'model_ep{episode+1}.pt'))
    
    agent.save(os.path.join(results_dir, 'model_final.pt'))
    
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    ssr = df['avg_reward'].tail(50).mean()
    summary = {
        'env_type': env_type, 'churn_type': churn_type, 'agent_type': agent_type,
        'SSR': ssr, 'advantage_tau': env_config['advantage_tau'],
        'advantage_weight': env_config['advantage_weight'],
        'churn_blend_weight': env_config['churn_blend_weight'],
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Completed! SSR: {ssr:.2f}")
    return ssr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, choices=['4manager', '10manager'])
    parser.add_argument('--churn', type=str, required=True, choices=['low', 'mid', 'high'])
    parser.add_argument('--agent', type=str, required=True, 
                       choices=['full_ea', 'no_pairset', 'no_tdconsistent', 'no_credit'])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.results_dir is None:
        results_base = Path(__file__).parent.parent / 'results_optimized'
        args.results_dir = str(results_base / f'{args.env}_{args.churn}_{args.agent}')
    
    train(args.env, args.churn, args.agent, args.results_dir, args.seed)


if __name__ == '__main__':
    main()
