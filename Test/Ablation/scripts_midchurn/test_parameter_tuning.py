#!/usr/bin/env python
"""
参数调优测试脚本 - 测试不同的TD-Consistent和Per-Device Credit参数

测试配置:
1. 原始: blend_weight=0.7, advantage_tau=2.0, advantage_weight=0.15
2. Option A: blend_weight=0.3 (更激进的TD目标)
3. Option B: advantage_tau=5.0, advantage_weight=0.05 (更平滑的advantage)
4. Option A+B: 两者结合

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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TunableEAAgent:
    """
    可调参数的EA Agent - 用于参数调优测试
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
        # 可调参数
        advantage_tau: float = 2.0,        # Option B: 2.0 -> 5.0
        advantage_weight: float = 0.15,    # Option B: 0.15 -> 0.05
        churn_blend_weight: float = 0.7,   # Option A: 0.7 -> 0.3
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
        
        # 可调参数
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
        """选择动作"""
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
        """存储transition"""
        self.replay_buffer.add(
            manager_id=manager_id, g=g, X=X, mask=mask, A=A, r=r,
            g_next=g_next, X_next=X_next, mask_next=mask_next, done=done,
        )
    
    def update(self, batch_size=256):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        batch = self.replay_buffer.sample(batch_size, device=self.device)
        
        manager_id = batch['manager_id']
        g = batch['g']
        X = batch['X']
        mask = batch['mask']
        A = batch['A']
        r = batch['r']
        g_next = batch['g_next']
        X_next = batch['X_next']
        mask_next = batch['mask_next']
        done = batch['done']
        
        # 更新Critic (使用可调的blend_weight)
        critic_loss, q1_value, q2_value = self._update_critics(
            manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done
        )
        
        self.critic_updates += 1
        
        # 延迟Actor更新
        actor_metrics = None
        if self.total_steps % self.policy_delay == 0:
            actor_metrics = self._update_actor(manager_id, g, X, mask)
            self._soft_update_targets()
            self.actor_updates += 1
        
        self.total_steps += 1
        
        metrics = {
            'critic_loss': critic_loss,
            'q1_value': q1_value,
            'q2_value': q2_value,
        }
        
        if actor_metrics is not None:
            metrics.update(actor_metrics)
        
        return metrics
    
    def _update_critics(self, manager_id, g, X, mask, A, r, g_next, X_next, mask_next, done):
        """使用可调blend_weight的TD-Consistent更新"""
        with torch.no_grad():
            # 检测churn
            churn_detected = (mask.sum(dim=-1) != mask_next.sum(dim=-1))
            
            # 两种mask的动作
            A_next_curr = self.actor_target(g_next, X_next, mask, manager_id)
            noise_curr = (torch.randn_like(A_next_curr) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            A_next_curr = (A_next_curr + noise_curr * mask.unsqueeze(-1)).clamp(-1.0, 1.0) * mask.unsqueeze(-1)
            
            A_next_real = self.actor_target(g_next, X_next, mask_next, manager_id)
            noise_real = (torch.randn_like(A_next_real) * self.noise_clip).clamp(-self.noise_clip, self.noise_clip)
            A_next_real = (A_next_real + noise_real * mask_next.unsqueeze(-1)).clamp(-1.0, 1.0) * mask_next.unsqueeze(-1)
            
            # Q值
            Q_target_curr = self.critics_target.min_Q(g_next, X_next, A_next_curr, mask, manager_id)
            Q_target_real = self.critics_target.min_Q(g_next, X_next, A_next_real, mask_next, manager_id)
            
            # 使用可调的blend_weight
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
        """使用可调参数的Per-Device Credit更新"""
        A_pi = self.actor(g, X, mask, manager_id)
        Q1, q_per_device = self.critics.Q1_forward(g, X, A_pi, mask, manager_id, return_per_device=True)
        
        # Primary loss
        primary_loss = -Q1.squeeze(-1).mean()
        
        # Per-device advantage (使用可调的advantage_tau)
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
        
        # Softmax权重 (使用可调的advantage_tau)
        weights = F.softmax(masked_advantage / self.advantage_tau, dim=-1)
        weights = weights * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        weighted_q = (weights * q_per_device * mask).sum(dim=-1)
        uniform_q = (q_per_device * mask).sum(dim=-1) / mask_sum.squeeze(-1)
        advantage_reg = weighted_q - uniform_q
        
        # Combined loss (使用可调的advantage_weight)
        actor_loss = (1 - self.advantage_weight) * primary_loss - self.advantage_weight * advantage_reg.mean()
        
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
    
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critics_state_dict': self.critics.state_dict(),
        }, filepath)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_environment(env_type, churn_type):
    """创建环境"""
    churn_configs = {
        'high': ((0.30, 0.325, 0.35), (0.4, 0.3, 0.3), 5),
    }
    severity_levels, severity_probs, trigger_interval = churn_configs[churn_type]
    
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
        aggregation_method="simple",
        trading_method="random",
        disaggregation_method="equal",
        churn_config=churn_config
    )
    
    # 包装mid-episode churn
    env = MidEpisodeChurnWrapper(env, churn_steps=[6, 12, 18], verbose=False)
    
    return env


def train_with_config(env_type, config_name, params, num_episodes=200, seed=42):
    """使用指定参数训练"""
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = create_environment(env_type, 'high')
    n_managers = len(env.manager_ids)
    
    agent = TunableEAAgent(
        x_dim=6, g_dim=26, p=5, N_max=100, num_managers=n_managers,
        advantage_tau=params['advantage_tau'],
        advantage_weight=params['advantage_weight'],
        churn_blend_weight=params['churn_blend_weight'],
        device=device,
    )
    
    rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            actions_dict = {}
            transitions = []
            
            for manager_idx, manager_id in enumerate(env.manager_ids):
                manager = env.manager_agents[manager_id]
                
                g = extract_global_features_from_env(env, manager_id, env.current_time)
                X, mask, device_ids = extract_device_states_from_manager(manager, N_max=100, x_dim=6)
                A = agent.select_action(g, X, mask, manager_idx, explore=True)
                
                action_flat = convert_ea_action_to_fogym(A, mask, device_ids)
                actions_dict[manager_id] = action_flat
                
                transitions.append({
                    'manager_idx': manager_idx, 'manager_id': manager_id,
                    'g': g, 'X': X, 'mask': mask, 'A': A, 'device_ids': device_ids,
                })
            
            next_obs, rewards_dict, dones, truncated, infos = env.step(actions_dict)
            done = dones.get('__all__', False)
            
            for trans in transitions:
                manager = env.manager_agents[trans['manager_id']]
                g_next = extract_global_features_from_env(env, trans['manager_id'], env.current_time)
                X_next, mask_next, _ = extract_device_states_from_manager(manager, N_max=100, x_dim=6)
                r = rewards_dict.get(trans['manager_id'], 0)
                episode_reward += r
                
                agent.store_transition(
                    manager_id=trans['manager_idx'],
                    g=trans['g'], X=trans['X'], mask=trans['mask'], A=trans['A'],
                    r=r, g_next=g_next, X_next=X_next, mask_next=mask_next, done=done
                )
            
            agent.update(batch_size=256)
        
        avg_reward = episode_reward / n_managers
        rewards.append(avg_reward)
        
        if (episode + 1) % 20 == 0:
            recent_avg = np.mean(rewards[-20:])
            print(f"  [{config_name}] Episode {episode+1}/{num_episodes} | Recent Avg: {recent_avg:.2f}")
    
    # 计算SSR
    ssr = np.mean(rewards[-50:])
    return ssr, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='4manager')
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()
    
    # 测试配置
    configs = {
        'original': {
            'churn_blend_weight': 0.7,
            'advantage_tau': 2.0,
            'advantage_weight': 0.15,
        },
        'option_A': {
            'churn_blend_weight': 0.3,  # 更激进
            'advantage_tau': 2.0,
            'advantage_weight': 0.15,
        },
        'option_B': {
            'churn_blend_weight': 0.7,
            'advantage_tau': 5.0,        # 更平滑
            'advantage_weight': 0.05,    # 更小权重
        },
        'option_AB': {
            'churn_blend_weight': 0.3,   # A
            'advantage_tau': 5.0,        # B
            'advantage_weight': 0.05,    # B
        },
    }
    
    print("=" * 80)
    print(f"参数调优测试 - {args.env}_high (Mid-Episode Churn)")
    print("=" * 80)
    
    results = {}
    
    for config_name, params in configs.items():
        print(f"\n测试配置: {config_name}")
        print(f"  blend_weight={params['churn_blend_weight']}, tau={params['advantage_tau']}, weight={params['advantage_weight']}")
        
        ssr, rewards = train_with_config(args.env, config_name, params, args.episodes)
        results[config_name] = ssr
        print(f"  SSR: {ssr:.2f}")
    
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    for name, ssr in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:15s}: {ssr:.2f}")


if __name__ == '__main__':
    main()
