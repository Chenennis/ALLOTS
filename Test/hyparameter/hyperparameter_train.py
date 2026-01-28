#!/usr/bin/env python
"""
超参数敏感性分析 - 完整训练模块

基于主实验EA脚本(4manager_mid_midf_ea.py)改编
支持对 ALLOTS 的两个关键超参进行敏感性分析:
1. Advantage Temperature τw (advantage_tau)
2. Token Dimension H (token_dim)

使用 mid-episode churn wrapper 与消融实验保持一致

Author: FOenv Team
Date: 2026-01-24
"""

import sys
import os
import argparse
import logging
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig
from algorithms.EA.foea.ea_agent import EAAgent
from algorithms.EA.foea.fogym_adapter import (
    extract_device_states_from_manager,
    extract_global_features_from_env,
    convert_ea_action_to_fogym
)

# 导入mid-episode churn wrapper
ablation_path = Path(__file__).parent.parent / 'Ablation'
sys.path.insert(0, str(ablation_path))
from envs.mid_episode_churn_wrapper import MidEpisodeChurnWrapper


# ============================================================================
# 配置参数 - 与主实验保持一致
# ============================================================================

# 环境配置
ENV_CONFIGS = {
    '4manager': {
        'data_dir': 'data',          # 相对于项目根目录
        'n_max': 80,
        'x_dim': 6,
        'g_dim': 26,
        'p': 5,
    },
    '10manager': {
        'data_dir': 'data_10manager',
        'n_max': 100,
        'x_dim': 6,
        'g_dim': 26,
        'p': 5,
    }
}

# Churn配置 (mid severity, midf frequency)
CHURN_CONFIG = {
    'enabled': True,
    'severity_levels': (0.20, 0.225, 0.25),  # 20%, 22.5%, 25%
    'severity_probs': (0.4, 0.3, 0.3),
    'trigger_interval': 5,
    'min_active_devices': 5,
}

# FOgym模块配置 - 与主实验一致
FOGYM_CONFIG = {
    'time_horizon': 24,
    'time_step': 1.0,
    'aggregation_method': 'LP',
    'trading_method': 'bidding',
    'disaggregation_method': 'proportional',
}

# 训练参数
TRAINING_CONFIG = {
    'num_episodes': 500,
    'batch_size': 256,
    'warmup_episodes': 10,
    'save_interval': 50,
    'log_interval': 10,
    # EA算法参数
    'gamma': 0.99,
    'tau': 0.005,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'policy_delay': 2,
    'noise_scale': 0.1,
    'noise_clip': 0.2,
    'buffer_capacity': 100000,
    # 网络架构默认值
    'emb_dim': 16,
    'hidden_dim': 256,
}


def setup_logging(output_dir: str, name: str = 'hyper'):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 清除已有handlers
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    
    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_environment(env_type: str):
    """
    创建FOgym环境 (带mid-episode churn wrapper)
    
    与主实验完全一致的配置
    """
    env_config = ENV_CONFIGS[env_type]
    
    # 创建churn配置
    churn_config = ChurnConfig(
        enabled=CHURN_CONFIG['enabled'],
        trigger_interval=CHURN_CONFIG['trigger_interval'],
        severity_levels=CHURN_CONFIG['severity_levels'],
        severity_probs=CHURN_CONFIG['severity_probs'],
        min_active_devices=CHURN_CONFIG['min_active_devices']
    )
    
    # 创建FOgym环境
    env = MultiAgentFlexOfferEnv(
        data_dir=env_config['data_dir'],
        time_horizon=FOGYM_CONFIG['time_horizon'],
        time_step=FOGYM_CONFIG['time_step'],
        aggregation_method=FOGYM_CONFIG['aggregation_method'],
        trading_method=FOGYM_CONFIG['trading_method'],
        disaggregation_method=FOGYM_CONFIG['disaggregation_method'],
        churn_config=churn_config
    )
    
    # 包装mid-episode churn (与消融实验一致)
    env = MidEpisodeChurnWrapper(env, churn_steps=[6, 12, 18], verbose=False)
    
    return env, env_config


def create_agent(
    env_config: dict,
    num_managers: int,
    device: str,
    advantage_tau: float = 1.0,
    token_dim: int = 128,
) -> EAAgent:
    """
    创建EA Agent，支持自定义超参
    
    Args:
        env_config: 环境配置
        num_managers: manager数量
        device: 计算设备
        advantage_tau: τw超参 (Per-Device Credit温度)
        token_dim: H超参 (Token维度)
    """
    agent = EAAgent(
        x_dim=env_config['x_dim'],
        g_dim=env_config['g_dim'],
        p=env_config['p'],
        N_max=env_config['n_max'],
        num_managers=num_managers,
        # 网络架构
        emb_dim=TRAINING_CONFIG['emb_dim'],
        token_dim=token_dim,           # 可变超参 H
        hidden_dim=TRAINING_CONFIG['hidden_dim'],
        # 优化参数
        gamma=TRAINING_CONFIG['gamma'],
        tau=TRAINING_CONFIG['tau'],
        lr_actor=TRAINING_CONFIG['lr_actor'],
        lr_critic=TRAINING_CONFIG['lr_critic'],
        policy_delay=TRAINING_CONFIG['policy_delay'],
        noise_scale=TRAINING_CONFIG['noise_scale'],
        noise_clip=TRAINING_CONFIG['noise_clip'],
        # 信用分配
        advantage_tau=advantage_tau,   # 可变超参 τw
        buffer_capacity=TRAINING_CONFIG['buffer_capacity'],
        device=device,
    )
    
    return agent


def get_observations(env, env_config: dict):
    """提取所有manager的观测"""
    obs_dict = {}
    n_max = env_config['n_max']
    x_dim = env_config['x_dim']
    
    for manager_id, manager in env.manager_agents.items():
        X, mask, device_ids = extract_device_states_from_manager(
            manager, N_max=n_max, x_dim=x_dim
        )
        g = extract_global_features_from_env(env, manager_id, env.current_time)
        
        obs_dict[manager_id] = {
            'g': g,
            'X': X,
            'mask': mask,
            'device_ids': device_ids,
            'n_devices': int(mask.sum())
        }
    
    return obs_dict


def convert_actions_to_env_format(actions_dict, obs_dict):
    """转换EA动作到FOgym格式"""
    env_actions = {}
    
    for manager_id, A_padded in actions_dict.items():
        obs = obs_dict[manager_id]
        action = convert_ea_action_to_fogym(
            A_padded, obs['mask'], obs['device_ids']
        )
        env_actions[manager_id] = action
    
    return env_actions


def train_single(
    env_type: str,
    seed: int,
    advantage_tau: float,
    token_dim: int,
    results_dir: str,
) -> Dict[str, Any]:
    """
    执行单次训练实验
    
    Args:
        env_type: '4manager' or '10manager'
        seed: 随机种子
        advantage_tau: τw超参
        token_dim: H超参
        results_dir: 结果保存目录
    
    Returns:
        summary: 实验结果摘要
    """
    # 设置日志
    logger = setup_logging(results_dir, name=f'hyper_{env_type}_{seed}')
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 设置随机种子
    set_seed(seed)
    
    logger.info("=" * 70)
    logger.info(f"Hyperparameter Sensitivity Analysis")
    logger.info(f"  Environment: {env_type}-mid (mid-episode churn)")
    logger.info(f"  τw (advantage_tau): {advantage_tau}")
    logger.info(f"  H (token_dim): {token_dim}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 70)
    
    # 创建环境
    logger.info("Creating FOgym environment...")
    env, env_config = create_environment(env_type)
    num_managers = len(env.manager_agents)
    logger.info(f"Environment created with {num_managers} managers")
    logger.info(f"Mid-episode churn enabled at steps [6, 12, 18]")
    
    # 创建agent
    logger.info("Creating EA agent...")
    agent = create_agent(
        env_config=env_config,
        num_managers=num_managers,
        device=device,
        advantage_tau=advantage_tau,
        token_dim=token_dim,
    )
    logger.info(f"EA Agent created: N_max={env_config['n_max']}, token_dim={token_dim}")
    
    # 训练历史
    history = {
        'episode': [], 'avg_reward': [], 'total_reward': [],
        'critic_loss': [], 'actor_loss': [],
        'q1_mean': [], 'q2_mean': [],
        'mean_advantage': [], 'advantage_std': [],
        'churn_triggered': [], 'mid_episode_churns': [],
        'global_step': [], 'episode_steps': [],
    }
    
    # 添加per-manager奖励
    for i in range(num_managers):
        history[f'reward_manager_{i}'] = []
    
    global_step = 0
    time_horizon = FOGYM_CONFIG['time_horizon']
    num_episodes = TRAINING_CONFIG['num_episodes']
    batch_size = TRAINING_CONFIG['batch_size']
    warmup_steps = TRAINING_CONFIG['warmup_episodes'] * time_horizon
    
    logger.info(f"\nStarting training: {num_episodes} episodes")
    
    for episode in range(num_episodes):
        # Reset环境
        env_obs, env_infos = env.reset()
        obs_dict = get_observations(env, env_config)
        
        episode_rewards = {mid: 0.0 for mid in env.manager_agents.keys()}
        episode_steps = 0
        mid_episode_churn_count = 0
        done = False
        
        # 检查reset时的churn
        churn_at_reset = False
        for mid, info in env_infos.items():
            if 'churn_events' in info and len(info['churn_events']) > 0:
                churn_at_reset = True
                break
        
        # 累积loss
        ep_critic_losses = []
        ep_actor_losses = []
        ep_q1_values = []
        ep_q2_values = []
        ep_advantages = []
        ep_advantage_stds = []
        
        # Episode循环
        while not done and episode_steps < time_horizon:
            actions_dict = {}
            
            for manager_id, obs in obs_dict.items():
                manager_idx = list(env.manager_agents.keys()).index(manager_id)
                
                # 选择动作 (带探索噪声)
                A_padded = agent.select_action(
                    obs['g'], obs['X'], obs['mask'], manager_idx, explore=True
                )
                actions_dict[manager_id] = A_padded
            
            # 转换动作格式
            env_actions = convert_actions_to_env_format(actions_dict, obs_dict)
            
            # 执行动作
            next_env_obs, rewards, dones_dict, truncated, infos = env.step(env_actions)
            done = dones_dict.get('__all__', False)
            next_obs_dict = get_observations(env, env_config)
            
            # 检查mid-episode churn
            for mid in infos:
                if infos[mid].get('mid_episode_churn', False):
                    mid_episode_churn_count += 1
            
            # 存储transitions
            for manager_id, obs in obs_dict.items():
                manager_idx = list(env.manager_agents.keys()).index(manager_id)
                reward = rewards.get(manager_id, 0.0)
                episode_rewards[manager_id] += reward
                
                agent.store_transition(
                    manager_id=manager_idx,
                    g=obs['g'],
                    X=obs['X'],
                    mask=obs['mask'],
                    A=actions_dict[manager_id],
                    r=reward,
                    g_next=next_obs_dict[manager_id]['g'],
                    X_next=next_obs_dict[manager_id]['X'],
                    mask_next=next_obs_dict[manager_id]['mask'],
                    done=done
                )
            
            # 更新agent
            if global_step >= warmup_steps:
                losses = agent.update(batch_size)
                if losses:
                    ep_critic_losses.append(losses.get('critic_loss', 0.0))
                    if 'actor_loss' in losses:
                        ep_actor_losses.append(losses['actor_loss'])
                        ep_advantages.append(losses.get('mean_advantage', 0.0))
                        ep_advantage_stds.append(losses.get('advantage_std', 0.0))
                    ep_q1_values.append(losses.get('q1_value', 0.0))
                    ep_q2_values.append(losses.get('q2_value', 0.0))
            
            # 更新状态
            obs_dict = next_obs_dict
            episode_steps += 1
            global_step += 1
        
        # Episode总结
        avg_reward = np.mean(list(episode_rewards.values()))
        
        history['episode'].append(episode)
        history['avg_reward'].append(avg_reward)
        history['total_reward'].append(sum(episode_rewards.values()))
        history['critic_loss'].append(np.mean(ep_critic_losses) if ep_critic_losses else 0)
        history['actor_loss'].append(np.mean(ep_actor_losses) if ep_actor_losses else 0)
        history['q1_mean'].append(np.mean(ep_q1_values) if ep_q1_values else 0)
        history['q2_mean'].append(np.mean(ep_q2_values) if ep_q2_values else 0)
        history['mean_advantage'].append(np.mean(ep_advantages) if ep_advantages else 0)
        history['advantage_std'].append(np.mean(ep_advantage_stds) if ep_advantage_stds else 0)
        history['churn_triggered'].append(churn_at_reset)
        history['mid_episode_churns'].append(mid_episode_churn_count)
        history['global_step'].append(global_step)
        history['episode_steps'].append(episode_steps)
        
        for i, mid in enumerate(env.manager_agents.keys()):
            history[f'reward_manager_{i}'].append(episode_rewards[mid])
        
        # 日志
        if (episode + 1) % TRAINING_CONFIG['log_interval'] == 0:
            logger.info(f"Episode {episode+1:3d}/{num_episodes} | "
                       f"Reward: {avg_reward:8.2f} | "
                       f"Mid-churns: {mid_episode_churn_count} | "
                       f"CriticLoss: {history['critic_loss'][-1]:.4f}")
        
        # 保存checkpoint
        if (episode + 1) % TRAINING_CONFIG['save_interval'] == 0:
            model_path = os.path.join(results_dir, f'model_ep{episode+1}.pt')
            torch.save({
                'actor': agent.actor.state_dict(),
                'critics': agent.critics.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critics_target': agent.critics_target.state_dict(),
            }, model_path)
            logger.info(f"Checkpoint saved: {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(results_dir, 'model_final.pt')
    torch.save({
        'actor': agent.actor.state_dict(),
        'critics': agent.critics.state_dict(),
        'actor_target': agent.actor_target.state_dict(),
        'critics_target': agent.critics_target.state_dict(),
    }, final_model_path)
    
    # 保存训练历史
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    # 计算结果摘要
    final_reward_50ep = df['avg_reward'].tail(50).mean()
    final_reward_std = df['avg_reward'].tail(50).std()
    initial_reward_10ep = df['avg_reward'].head(10).mean()
    
    summary = {
        'env_type': env_type,
        'churn_type': 'mid',
        'mid_episode_churn': True,
        'advantage_tau': advantage_tau,
        'token_dim': token_dim,
        'seed': seed,
        'final_reward_50ep': float(final_reward_50ep),
        'final_reward_std': float(final_reward_std),
        'initial_reward_10ep': float(initial_reward_10ep),
        'total_episodes': num_episodes,
        'total_steps': global_step,
        'device': device,
    }
    
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info(f"  Final (50ep): {final_reward_50ep:.2f} ± {final_reward_std:.2f}")
    logger.info(f"  Initial (10ep): {initial_reward_10ep:.2f}")
    logger.info(f"  Improvement: {((final_reward_50ep - initial_reward_10ep) / abs(initial_reward_10ep) * 100):.1f}%")
    logger.info(f"  Results: {results_dir}")
    logger.info("=" * 70)
    
    return summary


def run_hyperparameter_sweep(
    param_name: str,
    param_value: float,
    envs: List[str] = None,
    seeds: List[int] = None,
    base_results_dir: str = None,
) -> List[Dict[str, Any]]:
    """
    运行单个超参值的完整实验 (多环境 × 多种子)
    
    Args:
        param_name: 'tau_w' 或 'H'
        param_value: 超参数值
        envs: 环境列表
        seeds: 种子列表
        base_results_dir: 结果基础目录
    
    Returns:
        all_summaries: 所有实验结果
    """
    if envs is None:
        envs = ['4manager', '10manager']
    if seeds is None:
        seeds = [1, 2, 3]
    if base_results_dir is None:
        base_results_dir = str(Path(__file__).parent / 'results')
    
    all_summaries = []
    
    for env_type in envs:
        for seed in seeds:
            # 设置超参
            if param_name == 'tau_w':
                advantage_tau = param_value
                token_dim = 128  # 固定H
                config_name = f"tau_{param_value}"
            else:  # param_name == 'H'
                advantage_tau = 1.0  # 固定τw
                token_dim = int(param_value)
                config_name = f"h_{int(param_value)}"
            
            results_dir = os.path.join(
                base_results_dir, config_name, f"{env_type}_seed{seed}"
            )
            
            print("\n" + "#" * 70)
            print(f"# Running: {config_name} | {env_type} | seed={seed}")
            print("#" * 70 + "\n")
            
            try:
                summary = train_single(
                    env_type=env_type,
                    seed=seed,
                    advantage_tau=advantage_tau,
                    token_dim=token_dim,
                    results_dir=results_dir,
                )
                all_summaries.append(summary)
            except Exception as e:
                print(f"ERROR in {config_name}/{env_type}/seed{seed}: {e}")
                import traceback
                traceback.print_exc()
                all_summaries.append({
                    'env_type': env_type,
                    'seed': seed,
                    'error': str(e),
                })
    
    # 保存汇总结果
    if param_name == 'tau_w':
        config_name = f"tau_{param_value}"
    else:
        config_name = f"h_{int(param_value)}"
    
    summary_dir = os.path.join(base_results_dir, config_name)
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(os.path.join(summary_dir, 'all_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    # 打印统计
    print("\n" + "=" * 70)
    print(f"Sweep Complete: {config_name}")
    print("=" * 70)
    
    for env_type in envs:
        env_results = [s for s in all_summaries if s.get('env_type') == env_type and 'error' not in s]
        if env_results:
            rewards = [s['final_reward_50ep'] for s in env_results]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            print(f"  {env_type}: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Sensitivity Analysis for ALLOTS')
    parser.add_argument('--param', type=str, required=True, choices=['tau_w', 'H'],
                       help='Which hyperparameter to vary')
    parser.add_argument('--value', type=float, required=True,
                       help='Value of the hyperparameter')
    parser.add_argument('--envs', type=str, nargs='+', default=['4manager', '10manager'],
                       help='Environments to test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                       help='Random seeds')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Base results directory')
    
    args = parser.parse_args()
    
    run_hyperparameter_sweep(
        param_name=args.param,
        param_value=args.value,
        envs=args.envs,
        seeds=args.seeds,
        base_results_dir=args.results_dir,
    )


if __name__ == '__main__':
    main()
