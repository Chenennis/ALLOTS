#!/usr/bin/env python
"""
Enhanced EA Training Script for Comparative Study

Enhanced Features:
1. Progressive Credit: warmup from episode 0-99 (weight=0) to 300+ (weight=0.05)
2. Enhanced Pair-Set Critic: 3-layer MLP, 512 hidden dim
3. Mid-Episode Churn: churn triggered at steps [6, 12, 18] within episode

Author: FOenv Team
Date: 2026-01-23
"""

import sys
import os
import argparse
import logging
import json
import random
from datetime import datetime
from pathlib import Path

# Disable output buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig
from envs.mid_episode_churn_wrapper import MidEpisodeChurnWrapper
from algorithms.EA.foea.ea_agent import EAAgent
from algorithms.EA.foea.fogym_adapter import (
    extract_device_states_from_manager,
    extract_global_features_from_env,
    convert_ea_action_to_fogym
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== Environment Configs ==============
ENV_CONFIGS = {
    '4manager': {
        'data_dir': 'data',
        'manager_count': 4,
        'n_max': 100,
        'g_dim': 26,
        'advantage_tau': 2.0,
    },
    '10manager': {
        'data_dir': 'data_10manager',
        'manager_count': 10,
        'n_max': 100,
        'g_dim': 26,
        'advantage_tau': 5.0,
    }
}

# ============== Churn Configs ==============
CHURN_CONFIGS = {
    'low': ((0.10, 0.125, 0.15), (0.4, 0.3, 0.3), 5),
    'mid': ((0.20, 0.225, 0.25), (0.4, 0.3, 0.3), 5),
    'high': ((0.30, 0.325, 0.35), (0.4, 0.3, 0.3), 5),
}

# ============== Training Config ==============
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
        data_dir=env_config['data_dir'],
        time_horizon=24,
        time_step=1,
        churn_config=churn_config
    )
    
    # Wrap with mid-episode churn (churn at steps 6, 12, 18)
    env = MidEpisodeChurnWrapper(env, churn_steps=[6, 12, 18], verbose=False)
    return env, env_config


def train(env_type, churn_type, results_dir, seed=42):
    """
    Train Enhanced EA Agent
    
    Key improvements:
    1. Call agent.set_episode(episode) at start of each episode
    2. Enhanced Pair-Set Critic (3-layer MLP, 512 hidden)
    3. Progressive Credit warmup
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env, env_config = create_environment(env_type, churn_type)
    n_managers = len(env.manager_ids)
    
    # Create Enhanced EA Agent
    agent = EAAgent(
        x_dim=6,
        g_dim=env_config['g_dim'],
        p=5,
        N_max=env_config['n_max'],
        num_managers=n_managers,
        advantage_tau=env_config['advantage_tau'],
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
    print(f"Training Enhanced EA: {env_type}_{churn_type}")
    print(f"Features: Progressive Credit + Enhanced Pair-Set Critic")
    print(f"Mid-Episode Churn: steps [6, 12, 18]")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    history = {
        'episode': [], 'avg_reward': [], 'total_reward': [], 
        'critic_loss': [], 'actor_loss': [], 'q1_mean': [], 
        'mid_episode_churns': [], 'credit_weight': []
    }
    
    global_step = 0
    
    for episode in range(TRAINING_CONFIG['num_episodes']):
        # Critical: set episode for progressive Credit
        agent.set_episode(episode)
        
        obs, info = env.reset()
        episode_reward = 0
        mid_churns = 0
        done = False
        last_metrics = {}
        
        while not done:
            actions_dict = {}
            transitions = []
            
            for manager_idx, manager_id in enumerate(env.manager_ids):
                manager = env.manager_agents[manager_id]
                g = extract_global_features_from_env(env, manager_id, env.current_time)
                X, mask, device_ids = extract_device_states_from_manager(
                    manager, N_max=env_config['n_max'], x_dim=6
                )
                A = agent.select_action(g, X, mask, manager_idx, explore=True)
                
                action_flat = convert_ea_action_to_fogym(A, mask, device_ids)
                actions_dict[manager_id] = action_flat
                transitions.append({
                    'manager_idx': manager_idx, 'manager_id': manager_id,
                    'g': g, 'X': X, 'mask': mask, 'A': A
                })
            
            next_obs, rewards, dones, truncated, infos = env.step(actions_dict)
            done = dones.get('__all__', False)
            
            # Count mid-episode churns
            for mid in infos:
                if infos[mid].get('mid_episode_churn', False):
                    mid_churns += 1
            
            for trans in transitions:
                manager = env.manager_agents[trans['manager_id']]
                g_next = extract_global_features_from_env(
                    env, trans['manager_id'], env.current_time
                )
                X_next, mask_next, _ = extract_device_states_from_manager(
                    manager, N_max=env_config['n_max'], x_dim=6
                )
                r = rewards.get(trans['manager_id'], 0)
                episode_reward += r
                
                agent.store_transition(
                    manager_id=trans['manager_idx'],
                    g=trans['g'], X=trans['X'], mask=trans['mask'], A=trans['A'],
                    r=r, g_next=g_next, X_next=X_next, mask_next=mask_next, done=done
                )
            
            if global_step >= TRAINING_CONFIG['warmup_steps']:
                last_metrics = agent.update(batch_size=TRAINING_CONFIG['batch_size'])
            
            global_step += 1
        
        avg_reward = episode_reward / n_managers
        credit_weight = agent._get_credit_weight()
        
        history['episode'].append(episode)
        history['avg_reward'].append(avg_reward)
        history['total_reward'].append(episode_reward)
        history['critic_loss'].append(last_metrics.get('critic_loss', 0))
        history['actor_loss'].append(last_metrics.get('actor_loss', 0))
        history['q1_mean'].append(last_metrics.get('q1_value', 0))
        history['mid_episode_churns'].append(mid_churns)
        history['credit_weight'].append(credit_weight)
        
        if (episode + 1) % TRAINING_CONFIG['log_interval'] == 0:
            print(f"  Episode {episode+1}/{TRAINING_CONFIG['num_episodes']} | "
                  f"Reward: {avg_reward:.2f} | Mid-churns: {mid_churns} | "
                  f"Credit: {credit_weight:.3f}")
        
        if (episode + 1) % TRAINING_CONFIG['save_interval'] == 0:
            agent.save(os.path.join(results_dir, f'model_ep{episode+1}.pt'))
    
    agent.save(os.path.join(results_dir, 'model_final.pt'))
    
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    ssr = df['avg_reward'].tail(50).mean()
    summary = {
        'env_type': env_type, 
        'churn_type': churn_type, 
        'agent_type': 'enhanced_ea',
        'SSR': ssr, 
        'advantage_tau': env_config['advantage_tau'],
        'features': ['progressive_credit', 'enhanced_pairset', 'mid_episode_churn'],
        'churn_steps': [6, 12, 18],
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Completed! SSR: {ssr:.2f}")
    return ssr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, choices=['4manager', '10manager'])
    parser.add_argument('--churn', type=str, required=True, choices=['low', 'mid', 'high'])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.results_dir is None:
        results_base = Path(__file__).parent.parent / 'results'
        args.results_dir = str(results_base / f'ea_{args.env}_{args.churn}')
    
    train(args.env, args.churn, args.results_dir, args.seed)


if __name__ == '__main__':
    main()
