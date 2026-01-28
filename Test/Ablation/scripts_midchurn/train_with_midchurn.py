#!/usr/bin/env python
"""
通用训练脚本 - 支持 Mid-Episode Churn

用法:
    python train_with_midchurn.py --env 4manager --churn low --agent full_ea
    python train_with_midchurn.py --env 10manager --churn high --agent no_tdconsistent

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

import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig

# 导入mid-episode churn wrapper
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.mid_episode_churn_wrapper import MidEpisodeChurnWrapper

# 导入agents
from algorithms.EA.foea.ea_agent import EAAgent
from Test.Ablation.agents.ea_no_pairset import EAAgentNoPairSet
from Test.Ablation.agents.ea_no_tdconsistent import EAAgentNoTDConsistent
from Test.Ablation.agents.ea_no_credit import EAAgentNoCredit

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
        'user_count': 60,
        'n_max': 100,
        'g_dim': 26,  # From fogym_adapter: time(9)+price(4)+weather(3)+market(4)+manager(6)
    },
    '10manager': {
        'manager_count': 10,
        'user_count': 90,
        'n_max': 100,
        'g_dim': 26,  # From fogym_adapter: time(9)+price(4)+weather(3)+market(4)+manager(6)
    }
}

# ============== Churn配置 ==============
CHURN_CONFIGS = {
    'low': {
        'severity_levels': (0.10, 0.125, 0.15),  # 10%, 12.5%, 15%
        'severity_probs': (0.4, 0.3, 0.3),
        'trigger_interval': 5,  # midf frequency
    },
    'mid': {
        'severity_levels': (0.20, 0.225, 0.25),  # 20%, 22.5%, 25%
        'severity_probs': (0.4, 0.3, 0.3),
        'trigger_interval': 5,
    },
    'high': {
        'severity_levels': (0.30, 0.325, 0.35),  # 30%, 32.5%, 35%
        'severity_probs': (0.4, 0.3, 0.3),
        'trigger_interval': 5,
    },
}

# ============== Agent类型 ==============
AGENT_CLASSES = {
    'full_ea': EAAgent,
    'no_pairset': EAAgentNoPairSet,
    'no_tdconsistent': EAAgentNoTDConsistent,
    'no_credit': EAAgentNoCredit,
}

# ============== 训练参数 ==============
TRAINING_CONFIG = {
    'num_episodes': 500,
    'batch_size': 256,
    'warmup_steps': 1000,
    'save_interval': 50,
    'log_interval': 10,
    'seed': 42,
    # EA算法参数
    'gamma': 0.99,
    'tau': 0.005,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'policy_delay': 2,
    'noise_scale': 0.1,
    'noise_clip': 0.2,
    'buffer_capacity': 100000,
}


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_churn_config(churn_type: str) -> ChurnConfig:
    """创建churn配置"""
    config = CHURN_CONFIGS[churn_type]
    
    return ChurnConfig(
        enabled=True,
        trigger_interval=config['trigger_interval'],
        severity_levels=config['severity_levels'],
        severity_probs=config['severity_probs'],
        min_active_devices=5
    )


def create_environment(env_type: str, churn_type: str, use_midchurn: bool = True):
    """创建环境（带mid-episode churn wrapper）"""
    env_config = ENV_CONFIGS[env_type]
    churn_config = create_churn_config(churn_type)
    
    # 创建基础环境
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
    if use_midchurn:
        # 在每个episode的第6、12、18步触发churn
        env = MidEpisodeChurnWrapper(env, churn_steps=[6, 12, 18], verbose=False)
        logger.info("Mid-episode churn enabled at steps [6, 12, 18]")
    
    return env, env_config


def create_agent(agent_type: str, env_config: dict, device: str):
    """创建agent - 根据环境规模使用优化的参数"""
    AgentClass = AGENT_CLASSES[agent_type]
    
    # 根据参数调优结果设置最优参数
    if env_config['manager_count'] == 10:
        # 10manager: option_AB参数 (SSR: 74.48, +5.78 vs original)
        advantage_tau = 5.0       # 更平滑的softmax权重
        advantage_weight = 0.05   # 更小的advantage权重
        churn_blend_weight = 0.3  # 更激进的TD目标
        logger.info("10manager配置: tau=5.0, weight=0.05, blend=0.3 (option_AB)")
    else:
        # 4manager: original参数 (SSR: 75.33, 最优)
        advantage_tau = 2.0       # 原始tau
        advantage_weight = 0.15   # 原始权重
        churn_blend_weight = 0.7  # 保守的TD目标
        logger.info("4manager配置: tau=2.0, weight=0.15, blend=0.7 (original)")
    
    return AgentClass(
        x_dim=6,
        g_dim=env_config['g_dim'],
        p=5,
        N_max=env_config['n_max'],
        num_managers=env_config['manager_count'],
        gamma=TRAINING_CONFIG['gamma'],
        tau=TRAINING_CONFIG['tau'],
        lr_actor=TRAINING_CONFIG['lr_actor'],
        lr_critic=TRAINING_CONFIG['lr_critic'],
        policy_delay=TRAINING_CONFIG['policy_delay'],
        noise_scale=TRAINING_CONFIG['noise_scale'],
        noise_clip=TRAINING_CONFIG['noise_clip'],
        advantage_tau=advantage_tau,
        buffer_capacity=TRAINING_CONFIG['buffer_capacity'],
        device=device,
    )


def train(env_type: str, churn_type: str, agent_type: str, results_dir: str):
    """执行训练"""
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 设置随机种子
    set_seed(TRAINING_CONFIG['seed'])
    
    # 创建环境和agent
    env, env_config = create_environment(env_type, churn_type, use_midchurn=True)
    agent = create_agent(agent_type, env_config, device)
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 训练历史
    history = {
        'episode': [], 'avg_reward': [], 'total_reward': [],
        'critic_loss': [], 'actor_loss': [], 'q1_mean': [], 'q2_mean': [],
        'mean_advantage': [], 'advantage_std': [],
        'churn_triggered': [], 'mid_episode_churns': [],
        'global_step': [], 'episode_steps': [],
    }
    
    # 添加per-manager奖励
    for i in range(env_config['manager_count']):
        history[f'reward_manager_{i+1}'] = []
    
    global_step = 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {env_type}_{churn_type}_{agent_type} (with mid-episode churn)")
    logger.info(f"{'='*60}")
    
    for episode in range(TRAINING_CONFIG['num_episodes']):
        obs, info = env.reset()
        episode_reward = 0
        episode_rewards_per_manager = {f'manager_{i+1}': 0 for i in range(env_config['manager_count'])}
        episode_steps = 0
        mid_episode_churn_count = 0
        
        # 检查reset时是否触发churn
        churn_at_reset = any(info[mid].get('churn_events', []) for mid in info)
        
        done = False
        
        # Episode循环
        while not done:
            # 获取每个manager的observation
            actions_dict = {}
            transitions = []
            
            for manager_idx, manager_id in enumerate(env.manager_ids):
                manager = env.manager_agents[manager_id]
                
                # 提取状态
                g = extract_global_features_from_env(env, manager_id, env.current_time)
                X, mask, device_ids = extract_device_states_from_manager(
                    manager, N_max=env_config['n_max'], x_dim=6
                )
                
                # 选择动作
                A = agent.select_action(g, X, mask, manager_idx, explore=True)
                
                # 转换动作格式为FOgym格式
                action_flat = convert_ea_action_to_fogym(A, mask, device_ids)
                actions_dict[manager_id] = action_flat
                
                # 保存状态用于后续存储
                transitions.append({
                    'manager_idx': manager_idx,
                    'manager_id': manager_id,
                    'g': g, 'X': X, 'mask': mask, 'A': A,
                    'device_ids': device_ids,
                })
            
            # 执行动作
            next_obs, rewards, dones, truncated, infos = env.step(actions_dict)
            done = dones.get('__all__', False)
            
            # 检查mid-episode churn
            for mid in infos:
                if infos[mid].get('mid_episode_churn', False):
                    mid_episode_churn_count += 1
            
            # 存储transitions
            for trans in transitions:
                manager_idx = trans['manager_idx']
                manager_id = trans['manager_id']
                manager = env.manager_agents[manager_id]
                
                # 提取下一状态
                g_next = extract_global_features_from_env(env, manager_id, env.current_time)
                X_next, mask_next, _ = extract_device_states_from_manager(
                    manager, N_max=env_config['n_max'], x_dim=6
                )
                
                r = rewards.get(manager_id, 0)
                episode_reward += r
                episode_rewards_per_manager[f'manager_{manager_idx+1}'] += r
                
                # 存储到replay buffer
                agent.store_transition(
                    manager_id=manager_idx,
                    g=trans['g'], X=trans['X'], mask=trans['mask'], A=trans['A'],
                    r=r,
                    g_next=g_next, X_next=X_next, mask_next=mask_next,
                    done=done
                )
            
            # 更新agent
            if global_step >= TRAINING_CONFIG['warmup_steps']:
                metrics = agent.update(batch_size=TRAINING_CONFIG['batch_size'])
            else:
                metrics = {}
            
            global_step += 1
            episode_steps += 1
        
        # 记录episode结果
        avg_reward = episode_reward / env_config['manager_count']
        
        history['episode'].append(episode)
        history['avg_reward'].append(avg_reward)
        history['total_reward'].append(episode_reward)
        history['critic_loss'].append(metrics.get('critic_loss', 0))
        history['actor_loss'].append(metrics.get('actor_loss', 0))
        history['q1_mean'].append(metrics.get('q1_value', 0))
        history['q2_mean'].append(metrics.get('q2_value', 0))
        history['mean_advantage'].append(metrics.get('mean_advantage', 0))
        history['advantage_std'].append(metrics.get('advantage_std', 0))
        history['churn_triggered'].append(churn_at_reset)
        history['mid_episode_churns'].append(mid_episode_churn_count)
        history['global_step'].append(global_step)
        history['episode_steps'].append(episode_steps)
        
        for i in range(env_config['manager_count']):
            history[f'reward_manager_{i+1}'].append(
                episode_rewards_per_manager[f'manager_{i+1}']
            )
        
        # 日志
        if (episode + 1) % TRAINING_CONFIG['log_interval'] == 0:
            logger.info(f"Episode {episode+1}/{TRAINING_CONFIG['num_episodes']} | "
                       f"Reward: {avg_reward:.2f} | "
                       f"Mid-churns: {mid_episode_churn_count} | "
                       f"Steps: {global_step}")
        
        # 保存checkpoint
        if (episode + 1) % TRAINING_CONFIG['save_interval'] == 0:
            agent.save(os.path.join(results_dir, f'model_ep{episode+1}.pt'))
    
    # 保存最终模型和历史
    agent.save(os.path.join(results_dir, 'model_final.pt'))
    
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    # 保存结果摘要
    summary = {
        'env_type': env_type,
        'churn_type': churn_type,
        'agent_type': agent_type,
        'mid_episode_churn': True,
        'final_reward_50ep': df['avg_reward'].tail(50).mean(),
        'final_reward_std': df['avg_reward'].tail(50).std(),
        'initial_reward_10ep': df['avg_reward'].head(10).mean(),
        'total_episodes': TRAINING_CONFIG['num_episodes'],
        'total_steps': global_step,
    }
    
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"  Final SSR (50ep): {summary['final_reward_50ep']:.2f} ± {summary['final_reward_std']:.2f}")
    logger.info(f"  Initial (10ep): {summary['initial_reward_10ep']:.2f}")
    logger.info(f"  Results saved to: {results_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Train with Mid-Episode Churn')
    parser.add_argument('--env', type=str, required=True, choices=['4manager', '10manager'])
    parser.add_argument('--churn', type=str, required=True, choices=['low', 'mid', 'high'])
    parser.add_argument('--agent', type=str, required=True, 
                       choices=['full_ea', 'no_pairset', 'no_tdconsistent', 'no_credit'])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 更新种子
    TRAINING_CONFIG['seed'] = args.seed
    
    # 设置结果目录
    if args.results_dir is None:
        results_base = Path(__file__).parent.parent / 'results_midchurn'
        args.results_dir = str(results_base / f'{args.env}_{args.churn}_{args.agent}')
    
    # 运行训练
    train(args.env, args.churn, args.agent, args.results_dir)


if __name__ == '__main__':
    main()
