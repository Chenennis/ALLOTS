#!/usr/bin/env python
"""
Baseline Training Script for Comparative Study

Baselines with FULL Compatibility Layer (use_stable_mapping=True):
- MADDPG
- MATD3
- SQDDPG
- MAPPO
- MAIPPO

Mid-Episode Churn: churn triggered at steps [6, 12, 18] within episode

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

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import pandas as pd

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig
from envs.mid_episode_churn_wrapper import MidEpisodeChurnWrapper

# Import baseline compatibility wrappers
from algorithms.MADDPG.compat_wrapper import MADDPGCompatAgent
from algorithms.MATD3.compat_wrapper import MATD3CompatAgent
from algorithms.SQDDPG.compat_wrapper import SQDDPGCompatAgent
from algorithms.MAPPO.compat_wrapper import MAPPOCompatAgent
from algorithms.MAPPO.maippo_compat_wrapper import MAIPPOCompatAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== Environment Configs ==============
ENV_CONFIGS = {
    '4manager': {
        'data_dir': 'data',
        'manager_count': 4,
        'n_max': 100,
        'g_dim': 26,
    },
    '10manager': {
        'data_dir': 'data_10manager',
        'manager_count': 10,
        'n_max': 100,
        'g_dim': 26,
    }
}

# ============== Churn Configs ==============
CHURN_CONFIGS = {
    'low': ((0.10, 0.125, 0.15), (0.4, 0.3, 0.3), 5),
    'mid': ((0.20, 0.225, 0.25), (0.4, 0.3, 0.3), 5),
    'high': ((0.30, 0.325, 0.35), (0.4, 0.3, 0.3), 5),
}

# ============== Baseline Algorithm Configs ==============
BASELINE_CONFIGS = {
    'maddpg': {
        'class': MADDPGCompatAgent,
        'params': {
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'hidden_dim': 64,
            'gamma': 0.95,
            'tau': 0.01,
            'noise_scale': 0.1,
            'buffer_capacity': 1000000,
            'batch_size': 1024,
        }
    },
    'matd3': {
        'class': MATD3CompatAgent,
        'params': {
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'hidden_dim': 256,
            'gamma': 0.99,
            'tau': 0.005,
            'noise_scale': 0.1,
            'noise_clip': 0.2,
            'policy_delay': 2,
            'buffer_capacity': 100000,
            'batch_size': 64,
        }
    },
    'sqddpg': {
        'class': SQDDPGCompatAgent,
        'params': {
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'hidden_dim': 64,
            'gamma': 0.95,
            'tau': 0.01,
            'noise_scale': 0.1,
            'buffer_capacity': 1000000,
            'batch_size': 64,
        }
    },
    'mappo': {
        'class': MAPPOCompatAgent,
        'params': {
            'lr': 3e-4,
            'gamma': 0.99,
        }
    },
    'maippo': {
        'class': MAIPPOCompatAgent,
        'params': {
            'lr': 3e-4,
            'gamma': 0.99,
        }
    },
}

# ============== Training Config ==============
TRAINING_CONFIG = {
    'num_episodes': 500,
    'warmup_episodes': 10,
    'save_interval': 50,
    'log_interval': 10,
    'seed': 42,
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


def get_observations(env, env_config):
    """Extract observations for all managers in compatibility format."""
    obs_dict = {}
    
    for manager_id in env.manager_ids:
        manager = env.manager_agents[manager_id]
        
        # Global features
        g = np.zeros(env_config['g_dim'], dtype=np.float32)
        
        # Handle current_time (may be datetime or int/float)
        current_time = env.current_time
        if hasattr(current_time, 'hour'):
            # datetime object
            hour = current_time.hour + current_time.minute / 60.0
        else:
            # int or float (step count or hour)
            hour = float(current_time) % 24
        
        g[0] = hour / 24.0
        g[1] = np.sin(2 * np.pi * hour / 24)
        g[2] = np.cos(2 * np.pi * hour / 24)
        
        # Get active devices
        active_devices = list(manager.device_mdps.keys())
        
        # Device states
        device_states = {}
        for device_id in active_devices:
            device_mdp = manager.device_mdps.get(device_id)
            if device_mdp:
                # Use get_state_features() method
                state = device_mdp.get_state_features()
                # Normalize to 6 dimensions
                if len(state) < 6:
                    state = np.concatenate([state, np.zeros(6 - len(state))])
                elif len(state) > 6:
                    state = state[:6]
                device_states[device_id] = state.astype(np.float32)
            else:
                device_states[device_id] = np.zeros(6, dtype=np.float32)
        
        obs_dict[manager_id] = {
            'g': g,
            'device_ids': active_devices,
            'device_states': device_states
        }
    
    return obs_dict


def train(algorithm, env_type, churn_type, results_dir, seed=42):
    """Train baseline agent with full compatibility layer."""
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env, env_config = create_environment(env_type, churn_type)
    manager_ids = env.manager_ids
    n_managers = len(manager_ids)
    
    # Get baseline config
    baseline_config = BASELINE_CONFIGS[algorithm]
    AgentClass = baseline_config['class']
    agent_params = baseline_config['params'].copy()
    
    # Create agent with FULL compatibility layer (use_stable_mapping=True)
    agent = AgentClass(
        manager_ids=manager_ids,
        N_max=env_config['n_max'],
        x_dim=6,
        g_dim=env_config['g_dim'],
        p=5,
        device=device,
        **agent_params
    )
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training {algorithm.upper()}: {env_type}_{churn_type}")
    print(f"Compatibility Layer: FULL (use_stable_mapping=True)")
    print(f"Mid-Episode Churn: steps [6, 12, 18]")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    history = {
        'episode': [], 'avg_reward': [], 'total_reward': [], 
        'actor_loss': [], 'critic_loss': [], 'mid_episode_churns': []
    }
    
    for episode in range(TRAINING_CONFIG['num_episodes']):
        obs, info = env.reset()
        episode_reward = 0
        mid_churns = 0
        done = False
        
        # Get initial observations
        raw_obs = get_observations(env, env_config)
        
        actor_losses = []
        critic_losses = []
        
        while not done:
            # Select actions using compatibility layer
            actions_padded = agent.select_actions(raw_obs, explore=True)
            
            # Convert to environment format
            actions_dict = {}
            for manager_id in manager_ids:
                # The compat wrapper handles action extraction
                manager = env.manager_agents[manager_id]
                active_devices = list(manager.device_mdps.keys())
                n_devices = len(active_devices)
                
                if n_devices > 0:
                    # Create action array for active devices
                    action_arr = np.zeros((n_devices, 5), dtype=np.float32)
                    for i, dev_id in enumerate(active_devices):
                        slot = agent.compat_wrapper.slot_mappers[manager_id].slot_of_device.get(dev_id)
                        if slot is not None:
                            action_arr[i] = actions_padded[manager_id][slot]
                    actions_dict[manager_id] = action_arr.flatten()
                else:
                    actions_dict[manager_id] = np.array([])
            
            # Step environment
            next_obs, rewards, dones, truncated, infos = env.step(actions_dict)
            done = dones.get('__all__', False)
            
            # Count mid-episode churns
            for mid in infos:
                if infos[mid].get('mid_episode_churn', False):
                    mid_churns += 1
            
            # Get next observations
            next_raw_obs = get_observations(env, env_config)
            
            # Calculate total reward
            total_step_reward = sum(rewards.get(mid, 0) for mid in manager_ids)
            episode_reward += total_step_reward
            
            # Store transition and update (if past warmup)
            if episode >= TRAINING_CONFIG['warmup_episodes']:
                # Store transition
                agent.store_transition(
                    obs=raw_obs,
                    actions=actions_padded,
                    rewards={mid: rewards.get(mid, 0) for mid in manager_ids},
                    next_obs=next_raw_obs,
                    done=done
                )
                
                # Update agent
                metrics = agent.update()
                if metrics:
                    actor_losses.append(metrics.get('actor_loss', 0))
                    critic_losses.append(metrics.get('critic_loss', 0))
            
            raw_obs = next_raw_obs
        
        avg_reward = episode_reward / n_managers
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        
        history['episode'].append(episode)
        history['avg_reward'].append(avg_reward)
        history['total_reward'].append(episode_reward)
        history['actor_loss'].append(avg_actor_loss)
        history['critic_loss'].append(avg_critic_loss)
        history['mid_episode_churns'].append(mid_churns)
        
        if (episode + 1) % TRAINING_CONFIG['log_interval'] == 0:
            print(f"  Episode {episode+1}/{TRAINING_CONFIG['num_episodes']} | "
                  f"Reward: {avg_reward:.2f} | Mid-churns: {mid_churns}")
        
        if (episode + 1) % TRAINING_CONFIG['save_interval'] == 0:
            agent.save(os.path.join(results_dir, f'model_ep{episode+1}.pt'))
    
    agent.save(os.path.join(results_dir, 'model_final.pt'))
    
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    ssr = df['avg_reward'].tail(50).mean()
    summary = {
        'env_type': env_type, 
        'churn_type': churn_type, 
        'algorithm': algorithm,
        'SSR': ssr,
        'compatibility_layer': 'full_stable_mapping',
        'churn_steps': [6, 12, 18],
        'params': agent_params,
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Completed! SSR: {ssr:.2f}")
    return ssr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, 
                       choices=['maddpg', 'matd3', 'sqddpg', 'mappo', 'maippo'])
    parser.add_argument('--env', type=str, required=True, choices=['4manager', '10manager'])
    parser.add_argument('--churn', type=str, required=True, choices=['low', 'mid', 'high'])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.results_dir is None:
        results_base = Path(__file__).parent.parent / 'results'
        args.results_dir = str(results_base / f'{args.algorithm}_{args.env}_{args.churn}')
    
    train(args.algorithm, args.env, args.churn, args.results_dir, args.seed)


if __name__ == '__main__':
    main()
