"""
EA Ablation Test: 10Manager + Low Churn + w/o TD-Consistent
=============================================================

Ablation variant: Uses current mask instead of next mask (NO TD-Consistent)
- Ignores churn events during bootstrapping
- Tests if TD-consistent bootstrapping is necessary

Environment: 10 Managers
Churn Severity: Low (2% devices change)
Churn Frequency: Mid (every 5 episodes)

Author: FOenv Team
Date: 2026-01-20
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
# ============== Random Seed (for reproducibility) ==============
SEED = 42
import random
import numpy as np
import torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig
from Test.Ablation.agents.ea_no_tdconsistent import EAAgentNoTDConsistent
from algorithms.EA.foea.fogym_adapter import (
    extract_device_states_from_manager,
    extract_global_features_from_env,
    convert_ea_action_to_fogym
)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# ============== Ablation Info ==============
ABLATION_TYPE = "no_tdconsistent"
ABLATION_NAME = "w/o TD-Consistent"

# ============== Environment Parameters ==============
ENV_TYPE = "10manager"
DATA_DIR = "data"
TIME_HORIZON = 24
TIME_STEP = 1.0

# ============== Churn Parameters ==============
CHURN_ENABLED = True
CHURN_SEVERITY = "low"
CHURN_FREQUENCY = "midf"
CHURN_SEVERITY_LEVELS = [0.02, 0.02, 0.02]
CHURN_SEVERITY_PROBS = [0.4, 0.3, 0.3]
CHURN_TRIGGER_INTERVAL = 5
MIN_ACTIVE_DEVICES = 5

# ============== EA Algorithm Parameters ==============
N_MAX = 100
X_DIM = 6
G_DIM = 26
P = 5
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
BATCH_SIZE = 256
POLICY_DELAY = 2
BUFFER_CAPACITY = 100000
NOISE_SCALE = 0.1
NOISE_CLIP = 0.2
ADVANTAGE_TAU = 2.0

# ============== Network Architecture ==============
EMB_DIM = 16
TOKEN_DIM = 128
HIDDEN_DIM = 256

# ============== FOgym Module Selection ==============
AGGREGATION_METHOD = "LP"
TRADING_METHOD = "bidding"
DISAGGREGATION_METHOD = "proportional"

# ============== Training Parameters ==============
TRAIN_EPISODES = 500
TEST_EPISODES = 100
WARMUP_EPISODES = 10
EVAL_INTERVAL = 10
SAVE_INTERVAL = 50
USE_GPU = True
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# ============== Output Parameters ==============
OUTPUT_DIR = f"Test/Ablation/results/ea_{ENV_TYPE}_{CHURN_SEVERITY}_{CHURN_FREQUENCY}_{ABLATION_TYPE}"
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
    if mini_log:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO)
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
# AGENT CREATION (ABLATION VARIANT)
# =============================================================================

def create_agent(num_managers: int):
    """Create ABLATION Agent: w/o TD-Consistent"""
    agent = EAAgentNoTDConsistent(
        x_dim=X_DIM,
        g_dim=G_DIM,
        p=P,
        N_max=N_MAX,
        num_managers=num_managers,
        emb_dim=EMB_DIM,
        token_dim=TOKEN_DIM,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        tau=TAU,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        policy_delay=POLICY_DELAY,
        noise_scale=NOISE_SCALE,
        noise_clip=NOISE_CLIP,
        advantage_tau=ADVANTAGE_TAU,
        buffer_capacity=BUFFER_CAPACITY,
        device=DEVICE
    )
    return agent

# =============================================================================
# OBSERVATION EXTRACTION
# =============================================================================

def get_observations(env, episode_step=0):
    obs_dict = {}
    for manager_id, manager in env.manager_agents.items():
        X, mask, device_ids = extract_device_states_from_manager(manager, N_max=N_MAX, x_dim=X_DIM)
        g = extract_global_features_from_env(env, manager_id, env.current_time)
        obs_dict[manager_id] = {'g': g, 'X': X, 'mask': mask, 'device_ids': device_ids, 'n_devices': int(mask.sum())}
    return obs_dict

# =============================================================================
# ACTION CONVERSION
# =============================================================================

def convert_actions_to_env_format(actions_dict, obs_dict):
    env_actions = {}
    for manager_id, A_padded in actions_dict.items():
        obs = obs_dict[manager_id]
        action = convert_ea_action_to_fogym(A_padded, obs['mask'], obs['device_ids'])
        env_actions[manager_id] = action
    return env_actions

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(env, agent, num_episodes, logger, output_dir, mini_log=False):
    logger.info("=" * 80)
    logger.info(f"Starting ABLATION Training: {ABLATION_NAME}")
    logger.info(f"Environment: {ENV_TYPE}, Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
    logger.info(f"Device: {DEVICE}, Episodes: {num_episodes}")
    logger.info("=" * 80)
    
    if mini_log:
        print("\n" + "=" * 80)
        print(f">>> ABLATION: {ABLATION_NAME} | Env: {ENV_TYPE} | Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
        print("=" * 80)
    
    training_history = []
    global_step = 0
    
    for episode in range(num_episodes):
        env_obs, env_infos = env.reset()
        obs_dict = get_observations(env, episode_step=0)
        
        episode_rewards = {mid: 0.0 for mid in env.manager_agents.keys()}
        episode_steps = 0
        done = False
        
        churn_triggered = False
        for mid, info in env_infos.items():
            if 'churn_events' in info and len(info['churn_events']) > 0:
                churn_triggered = True
                break
        
        episode_critic_losses = []
        episode_actor_losses = []
        episode_q1_values = []
        episode_q2_values = []
        episode_advantages = []
        episode_advantage_stds = []
        
        if mini_log:
            churn_marker = "[CHURN]" if churn_triggered else ""
            print(f"\nEp {episode:3d} {churn_marker}", end='', flush=True)
        
        while not done and episode_steps < TIME_HORIZON:
            actions_dict = {}
            for manager_id, obs in obs_dict.items():
                manager_idx = list(env.manager_agents.keys()).index(manager_id)
                A_padded = agent.select_action(obs['g'], obs['X'], obs['mask'], manager_idx, explore=True)
                actions_dict[manager_id] = A_padded
            
            env_actions = convert_actions_to_env_format(actions_dict, obs_dict)
            next_env_obs, rewards, dones_dict, truncated, info = env.step(env_actions)
            done = dones_dict.get('__all__', False)
            next_obs_dict = get_observations(env, episode_step=episode_steps+1)
            
            for manager_id, obs in obs_dict.items():
                manager_idx = list(env.manager_agents.keys()).index(manager_id)
                reward = rewards.get(manager_id, 0.0)
                episode_rewards[manager_id] += reward
                
                agent.store_transition(
                    manager_id=manager_idx, g=obs['g'], X=obs['X'], mask=obs['mask'],
                    A=actions_dict[manager_id], r=reward,
                    g_next=next_obs_dict[manager_id]['g'],
                    X_next=next_obs_dict[manager_id]['X'],
                    mask_next=next_obs_dict[manager_id]['mask'],
                    done=done
                )
            
            if global_step >= WARMUP_EPISODES * TIME_HORIZON:
                losses = agent.update(BATCH_SIZE)
                if losses:
                    episode_critic_losses.append(losses.get('critic_loss', 0.0))
                    if 'actor_loss' in losses:
                        episode_actor_losses.append(losses['actor_loss'])
                        episode_advantages.append(losses.get('mean_advantage', 0.0))
                        episode_advantage_stds.append(losses.get('advantage_std', 0.0))
                    episode_q1_values.append(losses.get('q1_value', 0.0))
                    episode_q2_values.append(losses.get('q2_value', 0.0))
            
            obs_dict = next_obs_dict
            episode_steps += 1
            global_step += 1
        
        avg_reward = np.mean(list(episode_rewards.values()))
        critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0.0
        actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0.0
        q1_mean = np.mean(episode_q1_values) if episode_q1_values else 0.0
        q2_mean = np.mean(episode_q2_values) if episode_q2_values else 0.0
        mean_advantage = np.mean(episode_advantages) if episode_advantages else 0.0
        advantage_std = np.mean(episode_advantage_stds) if episode_advantage_stds else 0.0
        
        episode_metrics = {
            'episode': episode, 'avg_reward': avg_reward, 'total_reward': sum(episode_rewards.values()),
            'critic_loss': critic_loss, 'actor_loss': actor_loss,
            'q1_mean': q1_mean, 'q2_mean': q2_mean,
            'mean_advantage': mean_advantage, 'advantage_std': advantage_std,
            'churn_triggered': churn_triggered, 'global_step': global_step, 'episode_steps': episode_steps,
            'ablation_type': ABLATION_TYPE
        }
        for mid, rew in episode_rewards.items():
            episode_metrics[f'reward_{mid}'] = rew
        training_history.append(episode_metrics)
        
        if mini_log:
            status = "[CHURN]" if churn_triggered else "[OK]  "
            print(f" {status} R: {avg_reward:7.1f} | CL: {critic_loss:6.2f} | AL: {actor_loss:6.3f} | Q: {q1_mean:5.1f}/{q2_mean:5.1f}")
        
        if episode % EVAL_INTERVAL == 0:
            logger.info(f"Ep {episode:3d}: R={avg_reward:8.2f}, CL={critic_loss:6.2f}, AL={actor_loss:6.4f}, churn={churn_triggered}")
        
        if (episode + 1) % SAVE_INTERVAL == 0:
            save_model(agent, os.path.join(output_dir, f"model_ep{episode+1}.pt"))
    
    if mini_log:
        final_avg = np.mean([h['avg_reward'] for h in training_history[-10:]])
        print(f"\n>>> Training Complete! Final 10-ep avg: {final_avg:.2f}\n")
    
    return training_history

# =============================================================================
# SAVE/LOAD FUNCTIONS
# =============================================================================

def save_model(agent, path):
    torch.save({
        'actor': agent.actor.state_dict(),
        'critics': agent.critics.state_dict(),
        'actor_target': agent.actor_target.state_dict(),
        'critics_target': agent.critics_target.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
    }, path)

def load_model(agent, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critics.load_state_dict(checkpoint['critics'])
    agent.actor_target.load_state_dict(checkpoint['actor_target'])
    agent.critics_target.load_state_dict(checkpoint['critics_target'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

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
                'ablation_type': ABLATION_TYPE,
                'ablation_name': ABLATION_NAME,
                'env_type': ENV_TYPE,
                'churn_severity': CHURN_SEVERITY,
                'churn_frequency': CHURN_FREQUENCY,
                'train_episodes': TRAIN_EPISODES,
                'device': DEVICE
            },
            'training_history': training_history
        }
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_path}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f'EA Ablation: {ABLATION_NAME}')
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--mini_log', action='store_true')
    args = parser.parse_args()
    
    train_episodes = args.episodes if args.episodes else TRAIN_EPISODES
    logger = setup_logging(OUTPUT_DIR, mini_log=args.mini_log)
    
    logger.info(f"=== ABLATION EXPERIMENT: {ABLATION_NAME} ===")
    logger.info(f"Environment: {ENV_TYPE}, Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
    logger.info(f"Device: {DEVICE}, Episodes: {train_episodes}")
    
    env = create_environment()
    num_managers = len(env.manager_agents)
    logger.info(f"Environment created with {num_managers} managers")
    
    agent = create_agent(num_managers)
    logger.info(f"Ablation Agent ({ABLATION_NAME}) created")
    
    training_history = train(env, agent, train_episodes, logger, OUTPUT_DIR, mini_log=args.mini_log)
    
    final_model_path = os.path.join(OUTPUT_DIR, "model_final.pt")
    save_model(agent, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    save_results(training_history, OUTPUT_DIR, logger)
    
    logger.info("=" * 80)
    logger.info("ABLATION TRAINING COMPLETE")
    if training_history:
        final_reward = training_history[-1]['avg_reward']
        initial_reward = training_history[0]['avg_reward']
        logger.info(f"Initial: {initial_reward:.2f}, Final: {final_reward:.2f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
