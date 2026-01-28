"""
AGILE Baseline Test: 4 Managers Environment with Mid Severity, Mid Frequency Churn

This test validates AGILE's performance under device churn using compatibility layer.

- Algorithm: AGILE (Action-Graph Integrated Learning)
- Environment: 4 Managers
- Churn Severity: Mid (20-25% devices change)
- Churn Frequency: Mid (every 5 episodes)
- Training: 500 episodes

Author: FOenv Team
Date: 2026-01-24
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig
from algorithms.AGILE.compat_wrapper import AGILECompatAgent

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# ============== Environment Parameters ==============
ENV_TYPE = "4manager"
DATA_DIR = "data"
TIME_HORIZON = 24
TIME_STEP = 1.0

# ============== Churn Parameters ==============
CHURN_ENABLED = True
CHURN_SEVERITY = "mid"
CHURN_FREQUENCY = "midf"
CHURN_SEVERITY_LEVELS = [0.20, 0.225, 0.25]
CHURN_SEVERITY_PROBS = [0.4, 0.3, 0.3]
CHURN_TRIGGER_INTERVAL = 5
MIN_ACTIVE_DEVICES = 5

# ============== Compatibility Layer Parameters ==============
N_MAX = 80  # Maximum devices (4manager environment)
X_DIM = 6   # Device state dimension
G_DIM = 26  # Global feature dimension
P = 5       # FlexOffer action parameters

# ============== AGILE Hyperparameters ==============
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
HIDDEN_DIM = 64
NUM_HEADS = 2   # GAT attention heads
GAMMA = 0.99
TAU = 0.005
NOISE_SCALE = 0.1
BUFFER_CAPACITY = 100000
BATCH_SIZE = 256

# ============== FOgym Module Selection ==============
AGGREGATION_METHOD = "LP"
TRADING_METHOD = "bidding"
DISAGGREGATION_METHOD = "proportional"

# ============== Training Parameters ==============
TRAIN_EPISODES = 500
WARMUP_EPISODES = 10
EVAL_INTERVAL = 10
SAVE_INTERVAL = 50
USE_GPU = True
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# ============== Output Parameters ==============
OUTPUT_DIR = f"Test/results/agile_{ENV_TYPE}_{CHURN_SEVERITY}_{CHURN_FREQUENCY}"
SAVE_CSV = True
SAVE_JSON = True

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str, mini_log: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING if mini_log else logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    return logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_environment():
    churn_config = ChurnConfig(
        enabled=CHURN_ENABLED,
        trigger_interval=CHURN_TRIGGER_INTERVAL,
        severity_levels=tuple(CHURN_SEVERITY_LEVELS),
        severity_probs=tuple(CHURN_SEVERITY_PROBS),
        min_active_devices=MIN_ACTIVE_DEVICES
    )
    
    env = MultiAgentFlexOfferEnv(
        data_dir=DATA_DIR,
        time_horizon=TIME_HORIZON,
        time_step=TIME_STEP,
        aggregation_method=AGGREGATION_METHOD,
        trading_method=TRADING_METHOD,
        disaggregation_method=DISAGGREGATION_METHOD,
        churn_config=churn_config
    )
    return env

# =============================================================================
# OBSERVATION EXTRACTION
# =============================================================================

def get_observations(env):
    from algorithms.EA.foea.fogym_adapter import (
        extract_device_states_from_manager,
        extract_global_features_from_env
    )
    
    obs_dict = {}
    
    for manager_id, manager in env.manager_agents.items():
        X, mask, device_ids = extract_device_states_from_manager(manager, N_max=N_MAX, x_dim=X_DIM)
        g = extract_global_features_from_env(env, manager_id, env.current_time)
        
        device_states = {}
        for i, dev_id in enumerate(device_ids):
            slot = np.where(mask == 1)[0][i] if i < mask.sum() else None
            if slot is not None:
                device_states[dev_id] = X[slot]
        
        obs_dict[manager_id] = {
            'g': g,
            'device_ids': device_ids,
            'device_states': device_states
        }
    
    return obs_dict

# =============================================================================
# ACTION CONVERSION
# =============================================================================

def convert_actions_to_env_format(padded_actions, obs_dict):
    from algorithms.EA.foea.fogym_adapter import convert_ea_action_to_fogym
    
    env_actions = {}
    for manager_id, A_pad in padded_actions.items():
        device_ids = obs_dict[manager_id]['device_ids']
        mask = np.array([1.0] * len(device_ids) + [0.0] * (N_MAX - len(device_ids)))
        action = convert_ea_action_to_fogym(A_pad, mask, device_ids)
        env_actions[manager_id] = action
    return env_actions

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(env, agent, num_episodes, logger, output_dir, mini_log=False):
    logger.info("=" * 80)
    logger.info(f"Starting AGILE Training: {num_episodes} episodes")
    logger.info(f"Environment: {ENV_TYPE}, Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
    logger.info(f"Device: {DEVICE}, GAT Heads: {NUM_HEADS}")
    logger.info("=" * 80)
    
    if mini_log:
        print("\n" + "=" * 80)
        print(f">>> AGILE Training: {num_episodes} episodes | Env: {ENV_TYPE} | Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
        print("=" * 80)
    
    training_history = []
    global_step = 0
    
    for episode in range(num_episodes):
        env_obs, env_infos = env.reset()
        obs_dict = get_observations(env)
        
        episode_rewards = {mid: 0.0 for mid in env.manager_agents.keys()}
        episode_steps = 0
        done = False
        
        churn_triggered = False
        for mid, info in env_infos.items():
            if 'churn_events' in info and len(info['churn_events']) > 0:
                churn_triggered = True
                break
        
        if mini_log:
            churn_marker = "[CHURN]" if churn_triggered else ""
            print(f"\nEp {episode:3d} {churn_marker} - Starting...", end='', flush=True)
        
        episode_actor_losses = []
        episode_critic_losses = []
        
        while not done and episode_steps < TIME_HORIZON:
            explore = (global_step >= WARMUP_EPISODES * TIME_HORIZON)
            padded_actions = agent.select_actions(obs_dict, explore=explore)
            env_actions = convert_actions_to_env_format(padded_actions, obs_dict)
            
            next_env_obs, rewards, dones_dict, truncated, info = env.step(env_actions)
            done = dones_dict.get('__all__', False)
            next_obs_dict = get_observations(env)
            
            for manager_id in env.manager_agents.keys():
                episode_rewards[manager_id] += rewards.get(manager_id, 0.0)
            
            agent.store_transition(obs_dict, padded_actions, rewards, next_obs_dict, done)
            
            if global_step >= WARMUP_EPISODES * TIME_HORIZON:
                losses = agent.update()
                if losses:
                    episode_actor_losses.append(losses.get('actor_loss', 0.0))
                    episode_critic_losses.append(losses.get('critic_loss', 0.0))
            
            obs_dict = next_obs_dict
            episode_steps += 1
            global_step += 1
        
        avg_reward = np.mean(list(episode_rewards.values()))
        actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0.0
        critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0.0
        
        episode_metrics = {
            'episode': episode,
            'avg_reward': avg_reward,
            'total_reward': sum(episode_rewards.values()),
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'churn_triggered': churn_triggered,
            'global_step': global_step,
            'episode_steps': episode_steps,
            'buffer_size': agent.get_buffer_size()
        }
        
        for mid, rew in episode_rewards.items():
            episode_metrics[f'reward_{mid}'] = rew
        
        training_history.append(episode_metrics)
        
        if mini_log:
            status = "[CHURN]" if churn_triggered else "[OK]  "
            print(f" {status} Reward: {avg_reward:7.1f} | "
                  f"ActorLoss: {actor_loss:6.3f} | CriticLoss: {critic_loss:6.3f}")
        
        if episode % EVAL_INTERVAL == 0:
            logger.info(f"Episode {episode:3d}: reward={avg_reward:8.2f}, "
                       f"actor_loss={actor_loss:8.4f}, critic_loss={critic_loss:8.4f}, "
                       f"buffer={agent.get_buffer_size()}, churn={'YES' if churn_triggered else 'NO'}")
        
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save(os.path.join(output_dir, f"model_ep{episode+1}.pt"))
            logger.info(f"Checkpoint saved: episode {episode+1}")
    
    if mini_log:
        final_avg_reward = np.mean([h['avg_reward'] for h in training_history[-10:]])
        print("\n" + "=" * 80)
        print(f">>> Training Complete! Final 10-ep avg reward: {final_avg_reward:.2f}")
        print("=" * 80 + "\n")
    
    return training_history

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(training_history, output_dir, logger):
    if SAVE_CSV and training_history:
        train_df = pd.DataFrame(training_history)
        train_csv = os.path.join(output_dir, "training_history.csv")
        train_df.to_csv(train_csv, index=False)
        logger.info(f"Training history saved to {train_csv}")
    
    if SAVE_JSON:
        results = {
            'config': {
                'algorithm': 'AGILE',
                'env_type': ENV_TYPE,
                'churn_severity': CHURN_SEVERITY,
                'churn_frequency': CHURN_FREQUENCY,
                'train_episodes': TRAIN_EPISODES,
                'hidden_dim': HIDDEN_DIM,
                'num_heads': NUM_HEADS,
                'lr_actor': LR_ACTOR,
                'lr_critic': LR_CRITIC,
                'gamma': GAMMA,
                'tau': TAU
            },
            'final_metrics': {
                'avg_reward': np.mean([h['avg_reward'] for h in training_history[-10:]]) if training_history else 0.0
            }
        }
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR, mini_log=True)
    
    logger.info(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    env = create_environment()
    logger.info(f"Environment created with {len(env.manager_agents)} managers")
    
    manager_ids = list(env.manager_agents.keys())
    agent = AGILECompatAgent(
        manager_ids=manager_ids,
        N_max=N_MAX,
        x_dim=X_DIM,
        g_dim=G_DIM,
        p=P,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        gamma=GAMMA,
        tau=TAU,
        noise_scale=NOISE_SCALE,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    training_history = train(env, agent, TRAIN_EPISODES, logger, OUTPUT_DIR, mini_log=True)
    
    final_model_path = os.path.join(OUTPUT_DIR, "model_final.pt")
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    save_results(training_history, OUTPUT_DIR, logger)
    
    env.close()
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
