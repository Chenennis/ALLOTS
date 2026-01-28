"""
MADDPG Baseline Test: 10 Managers Environment with High Severity, Mid Frequency Churn

This test validates MADDPG's performance under device churn using compatibility layer.

- Algorithm: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- Environment: 10 Managers
- Churn Severity: High (10-15% devices change)
- Churn Frequency: Mid (every 10 episodes)
- Training: 100 episodes (for baseline comparison with EA)

Author: FOenv Team
Date: 2026-01-13
"""
import torch

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig
from algorithms.MADDPG.compat_wrapper import MADDPGCompatAgent

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# ============== Environment Parameters ==============
ENV_TYPE = "10manager"
DATA_DIR = "data_10manager"
TIME_HORIZON = 24
TIME_STEP = 1.0

# ============== Churn Parameters ==============
CHURN_ENABLED = True
CHURN_SEVERITY = "high"
CHURN_FREQUENCY = "midf"
CHURN_SEVERITY_LEVELS = [0.30, 0.325, 0.35]  # 30%, 32.5%, 35%
CHURN_SEVERITY_PROBS = [0.4, 0.3, 0.3]
CHURN_TRIGGER_INTERVAL = 5
MIN_ACTIVE_DEVICES = 5

# ============== Compatibility Layer Parameters ==============
N_MAX = 100  # Maximum devices in 10manager environment (with churn margin)
X_DIM = 6   # Device state dimension (from fogym_adapter)
G_DIM = 26  # Global feature dimension
P = 5       # FlexOffer action parameters

# ============== MADDPG Hyperparameters (from paper) ==============
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
HIDDEN_DIM = 64
GAMMA = 0.95
TAU = 0.01
NOISE_SCALE = 0.1
BUFFER_CAPACITY = 1000000
BATCH_SIZE = 1024

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
DEVICE = "cuda" if USE_GPU else "cpu"

# ============== Output Parameters ==============
OUTPUT_DIR = f"Test/results/maddpg_{ENV_TYPE}_{CHURN_SEVERITY}_{CHURN_FREQUENCY}"
SAVE_CSV = True
SAVE_JSON = True

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str, mini_log: bool = False):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # File handler - always detailed
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console handler - minimal if mini_log=True
    console_handler = logging.StreamHandler()
    if mini_log:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_environment():
    """Create MultiAgentFlexOfferEnv with churn configuration"""
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
# OBSERVATION EXTRACTION (Compatible with adapter layer)
# =============================================================================

def get_observations(env):
    """
    Extract observations compatible with compatibility layer.
    
    Returns:
        obs_dict: {manager_id: {'g': [...], 'device_ids': [...], 'device_states': {...}}}
    """
    from algorithms.EA.foea.fogym_adapter import (
        extract_device_states_from_manager,
        extract_global_features_from_env
    )
    
    obs_dict = {}
    
    for manager_id, manager in env.manager_agents.items():
        # Extract device states using EA adapter (reuse)
        X, mask, device_ids = extract_device_states_from_manager(
            manager, N_max=N_MAX, x_dim=X_DIM
        )
        
        # Extract global features
        g = extract_global_features_from_env(
            env, manager_id, env.current_time
        )
        
        # Create device_states dict
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
    """Convert padded actions to environment format"""
    from algorithms.EA.foea.fogym_adapter import convert_ea_action_to_fogym
    
    env_actions = {}
    
    for manager_id, A_pad in padded_actions.items():
        device_ids = obs_dict[manager_id]['device_ids']
        # Create mask
        mask = np.array([1.0] * len(device_ids) + [0.0] * (N_MAX - len(device_ids)))
        
        action = convert_ea_action_to_fogym(A_pad, mask, device_ids)
        env_actions[manager_id] = action
    
    return env_actions

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(env, agent, num_episodes, logger, output_dir, mini_log=False):
    """Train MADDPG agent"""
    logger.info("=" * 80)
    logger.info(f"Starting MADDPG Training: {num_episodes} episodes")
    logger.info(f"Environment: {ENV_TYPE}, Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 80)
    
    if mini_log:
        print("\n" + "=" * 80)
        print(f">>> MADDPG Training: {num_episodes} episodes | Env: {ENV_TYPE} | Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
        print("=" * 80)
    
    training_history = []
    global_step = 0
    
    for episode in range(num_episodes):
        env_obs, env_infos = env.reset()
        obs_dict = get_observations(env)
        
        episode_rewards = {mid: 0.0 for mid in env.manager_agents.keys()}
        episode_steps = 0
        done = False
        
        # Check for churn
        churn_triggered = False
        for mid, info in env_infos.items():
            if 'churn_events' in info and len(info['churn_events']) > 0:
                churn_triggered = True
                break
        
        if mini_log:
            churn_marker = "[CHURN]" if churn_triggered else ""
            print(f"\nEp {episode:3d} {churn_marker} - Starting...", end='', flush=True)
        
        # Accumulate losses
        episode_actor_losses = []
        episode_critic_losses = []
        
        while not done and episode_steps < TIME_HORIZON:
            # Select actions
            explore = (global_step >= WARMUP_EPISODES * TIME_HORIZON)
            padded_actions = agent.select_actions(obs_dict, explore=explore)
            
            # Convert to environment format
            env_actions = convert_actions_to_env_format(padded_actions, obs_dict)
            
            # Step environment
            next_env_obs, rewards, dones_dict, truncated, info = env.step(env_actions)
            done = dones_dict.get('__all__', False)
            next_obs_dict = get_observations(env)
            
            # Store rewards
            for manager_id in env.manager_agents.keys():
                episode_rewards[manager_id] += rewards.get(manager_id, 0.0)
            
            # Store transition
            agent.store_transition(obs_dict, padded_actions, rewards, next_obs_dict, done)
            
            # Update agent
            if global_step >= WARMUP_EPISODES * TIME_HORIZON:
                losses = agent.update()
                if losses:
                    episode_actor_losses.append(losses.get('actor_loss', 0.0))
                    episode_critic_losses.append(losses.get('critic_loss', 0.0))
            
            obs_dict = next_obs_dict
            episode_steps += 1
            global_step += 1
        
        # Episode summary
        avg_reward = np.mean(list(episode_rewards.values()))
        
        actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0.0
        critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0.0
        
        # Record metrics
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
    """Save training results"""
    if SAVE_CSV and training_history:
        train_df = pd.DataFrame(training_history)
        train_csv = os.path.join(output_dir, "training_history.csv")
        train_df.to_csv(train_csv, index=False)
        logger.info(f"Training history saved to {train_csv}")
    
    if SAVE_JSON:
        results = {
            'config': {
                'algorithm': 'MADDPG',
                'env_type': ENV_TYPE,
                'churn_severity': CHURN_SEVERITY,
                'churn_frequency': CHURN_FREQUENCY,
                'train_episodes': TRAIN_EPISODES,
                'lr_actor': LR_ACTOR,
                'lr_critic': LR_CRITIC,
                'gamma': GAMMA,
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
    parser = argparse.ArgumentParser(description='MADDPG Baseline Test - 4Manager Mid/MidF')
    parser.add_argument('--mode', type=str, default='train', choices=['train'],
                       help='Run mode (only train for now)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes')
    parser.add_argument('--mini_log', action='store_true',
                       help='Use minimal logging')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    train_episodes = args.episodes if args.episodes is not None else TRAIN_EPISODES
    
    # GPU configuration
    use_gpu = args.gpu if args.gpu else USE_GPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    logger = setup_logging(OUTPUT_DIR, mini_log=args.mini_log)
    
    logger.info("Configuration:")
    logger.info(f"  Algorithm: MADDPG")
    logger.info(f"  Environment: {ENV_TYPE}")
    logger.info(f"  Churn: {CHURN_SEVERITY} severity, {CHURN_FREQUENCY} frequency")
    logger.info(f"  N_max: {N_MAX}, x_dim: {X_DIM}, g_dim: {G_DIM}, p: {P}")
    logger.info(f"  Hyperparameters: lr_actor={LR_ACTOR}, lr_critic={LR_CRITIC}, gamma={GAMMA}")
    logger.info(f"  Training episodes: {train_episodes}")
    
    logger.info("\nCreating environment...")
    env = create_environment()
    manager_ids = list(env.manager_agents.keys())
    logger.info(f"Environment created with {len(manager_ids)} managers")
    
    logger.info("\nCreating MADDPG agent with compatibility layer...")
    agent = MADDPGCompatAgent(
        manager_ids=manager_ids,
        N_max=N_MAX,
        x_dim=X_DIM,
        g_dim=G_DIM,
        p=P,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        tau=TAU,
        noise_scale=NOISE_SCALE,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        device=device
    )
    logger.info("MADDPG agent created successfully")
    
    training_history = train(env, agent, train_episodes, logger, OUTPUT_DIR, mini_log=args.mini_log)
    
    final_model_path = os.path.join(OUTPUT_DIR, "model_final.pt")
    agent.save(final_model_path)
    logger.info(f"\nFinal model saved to {final_model_path}")
    
    save_results(training_history, OUTPUT_DIR, logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    if training_history:
        final_reward = training_history[-1]['avg_reward']
        initial_reward = training_history[0]['avg_reward']
        improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100
        logger.info(f"Initial reward: {initial_reward:.2f}")
        logger.info(f"Final reward: {final_reward:.2f}")
        logger.info(f"Improvement: {improvement:.2f}%")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
