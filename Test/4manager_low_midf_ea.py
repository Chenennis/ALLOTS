"""
EA Algorithm Test: 4Manager Environment with Low Severity, MIDF Churn

This test validates EA's learning capability under light device churn:
- Environment: 4 Managers
- Churn Severity: Low (1.0%-2.0% devices change)
- Churn Frequency: Mid (every 10 episodes)
- Training: 100 episodes (initial validation)

Author: FOenv Team
Date: 2026-01-12
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import ChurnConfig
from algorithms.EA.foea.ea_agent import EAAgent
from algorithms.EA.foea.fogym_adapter import (
    extract_device_states_from_manager,
    extract_global_features_from_env,
    convert_ea_action_to_fogym
)

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
CHURN_SEVERITY = "low"  # low/mid/high
CHURN_FREQUENCY = "midf"  # lowf/midf/highf
CHURN_SEVERITY_LEVELS = [0.10, 0.125, 0.15]  # 10%, 12.5%, 15%
CHURN_SEVERITY_PROBS = [0.4, 0.3, 0.3]
CHURN_TRIGGER_INTERVAL = 5  # episodes
MIN_ACTIVE_DEVICES = 5

# ============== EA Algorithm Parameters ==============
N_MAX = 80
X_DIM = 6
G_DIM = 26  # From fogym_adapter
P = 5  # FlexOffer parameters
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
BATCH_SIZE = 256
POLICY_DELAY = 2
BUFFER_CAPACITY = 100000
NOISE_SCALE = 0.1
NOISE_CLIP = 0.2
ADVANTAGE_TAU = 1.0  # Temperature for softmax advantage weighting (lower = sharper focus)

# ============== Network Architecture ==============
EMB_DIM = 16
TOKEN_DIM = 128
HIDDEN_DIM = 256

# ============== FOgym Module Selection ==============
AGGREGATION_METHOD = "LP"  # LP or DP
TRADING_METHOD = "bidding"  # bidding or market_clearing
DISAGGREGATION_METHOD = "proportional"  # proportional or average

# ============== Training Parameters ==============
TRAIN_EPISODES = 500  # Initial validation (increase to 1000 for full training)
TEST_EPISODES = 100
WARMUP_EPISODES = 10
EVAL_INTERVAL = 10
SAVE_INTERVAL = 50
USE_GPU = True
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# ============== Output Parameters ==============
OUTPUT_DIR = f"Test/results/ea_{ENV_TYPE}_{CHURN_SEVERITY}_{CHURN_FREQUENCY}"
SAVE_CSV = True
SAVE_JSON = True

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str, mini_log: bool = False):
    """Setup logging configuration
    
    Args:
        output_dir: Directory to save log files
        mini_log: If True, only show key info in terminal (full log still saved to file)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # File handler - always detailed
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console handler - minimal if mini_log=True
    console_handler = logging.StreamHandler()
    if mini_log:
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors in terminal
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Configure root logger
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
    
    # Create churn configuration
    churn_config = ChurnConfig(
        enabled=CHURN_ENABLED,
        trigger_interval=CHURN_TRIGGER_INTERVAL,
        severity_levels=tuple(CHURN_SEVERITY_LEVELS),
        severity_probs=tuple(CHURN_SEVERITY_PROBS),
        min_active_devices=MIN_ACTIVE_DEVICES
    )
    
    # Create environment
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
# AGENT CREATION
# =============================================================================

def create_agent(num_managers: int):
    """Create EA Agent"""
    
    agent = EAAgent(
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
    """
    Extract observations for all managers
    
    Returns:
        obs_dict: {manager_id: (g, X, mask, device_ids)}
    """
    obs_dict = {}
    
    for manager_id, manager in env.manager_agents.items():
        # Extract device states
        X, mask, device_ids = extract_device_states_from_manager(
            manager, N_max=N_MAX, x_dim=X_DIM
        )
        
        # Extract global features
        g = extract_global_features_from_env(
            env, manager_id, env.current_time
        )
        
        obs_dict[manager_id] = {
            'g': g,
            'X': X,
            'mask': mask,
            'device_ids': device_ids,
            'n_devices': int(mask.sum())
        }
    
    return obs_dict

# =============================================================================
# ACTION CONVERSION
# =============================================================================

def convert_actions_to_env_format(actions_dict, obs_dict):
    """
    Convert EA actions to environment format
    
    Args:
        actions_dict: {manager_id: A_padded [N_max, p]}
        obs_dict: {manager_id: obs_info}
    
    Returns:
        env_actions: {manager_id: flattened_action}
    """
    env_actions = {}
    
    for manager_id, A_padded in actions_dict.items():
        obs = obs_dict[manager_id]
        action = convert_ea_action_to_fogym(
            A_padded, obs['mask'], obs['device_ids']
        )
        env_actions[manager_id] = action
    
    return env_actions

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(env, agent, num_episodes, logger, output_dir, mini_log=False):
    """
    Train EA agent
    
    Args:
        mini_log: If True, only print key episode info to terminal
    
    Args:
        env: Environment instance
        agent: EAAgent instance
        num_episodes: Number of training episodes
        logger: Logger instance
        output_dir: Output directory
    
    Returns:
        training_history: List of episode metrics
    """
    logger.info("=" * 80)
    logger.info(f"Starting EA Training: {num_episodes} episodes")
    logger.info(f"Environment: {ENV_TYPE}, Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 80)
    
    # Mini log header
    if mini_log:
        print("\n" + "=" * 80)
        print(f">>> EA Training: {num_episodes} episodes | Env: {ENV_TYPE} | Churn: {CHURN_SEVERITY}/{CHURN_FREQUENCY}")
        print("=" * 80)
    
    training_history = []
    global_step = 0
    num_managers = len(env.manager_agents)
    
    for episode in range(num_episodes):
        # Reset environment (returns observations and infos)
        env_obs, env_infos = env.reset()
        obs_dict = get_observations(env, episode_step=0)
        
        episode_rewards = {mid: 0.0 for mid in env.manager_agents.keys()}
        episode_steps = 0
        done = False
        
        # Check for churn event from infos
        churn_triggered = False
        churn_info = {}
        for mid, info in env_infos.items():
            if 'churn_events' in info and len(info['churn_events']) > 0:
                churn_triggered = True
                churn_info[mid] = info['churn_events'][0]
                break
        
        # Accumulate losses within episode
        episode_critic_losses = []
        episode_actor_losses = []
        episode_q1_values = []
        episode_q2_values = []
        episode_advantages = []
        episode_advantage_stds = []
        
        if churn_triggered:
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode}: CHURN EVENT TRIGGERED")
            for mid, event_dict in churn_info.items():
                logger.info(f"  {mid}: severity={event_dict.get('severity', 0):.2%}, "
                          f"{event_dict.get('devices_left', 0)} left, "
                          f"{event_dict.get('devices_joined', 0)} joined")
            logger.info(f"{'='*60}\n")
        
        # Mini log: print episode start
        if mini_log:
            churn_marker = "[CHURN]" if churn_triggered else ""
            print(f"\nEp {episode:3d} {churn_marker} - Starting...", end='', flush=True)
        
        # Episode loop
        while not done and episode_steps < TIME_HORIZON:
            # Select actions for all managers
            actions_dict = {}
            
            for manager_id, obs in obs_dict.items():
                # Get manager index
                manager_idx = list(env.manager_agents.keys()).index(manager_id)
                
                # select_action expects numpy arrays, not tensors
                # Select action (with exploration noise during training)
                A_padded = agent.select_action(
                    obs['g'], obs['X'], obs['mask'], manager_idx, explore=True
                )
                
                actions_dict[manager_id] = A_padded  # Already numpy array [N_max, p]
            
            # Convert to environment format
            env_actions = convert_actions_to_env_format(actions_dict, obs_dict)
            
            # Step environment (Gym format: obs, rewards, dones_dict, truncated, info)
            next_env_obs, rewards, dones_dict, truncated, info = env.step(env_actions)
            done = dones_dict.get('__all__', False)  # Extract boolean from dict
            next_obs_dict = get_observations(env, episode_step=episode_steps+1)
            
            # Store transitions for each manager
            for manager_id, obs in obs_dict.items():
                manager_idx = list(env.manager_agents.keys()).index(manager_id)
                
                # Get reward
                reward = rewards.get(manager_id, 0.0)
                episode_rewards[manager_id] += reward
                
                # Store transition
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
            
            # Update agent
            if global_step >= WARMUP_EPISODES * TIME_HORIZON:
                losses = agent.update(BATCH_SIZE)
                # Accumulate losses
                if losses is not None and losses:
                    episode_critic_losses.append(losses.get('critic_loss', 0.0))
                    if 'actor_loss' in losses:  # Only present on policy_delay steps
                        episode_actor_losses.append(losses['actor_loss'])
                        episode_advantages.append(losses.get('mean_advantage', 0.0))
                        episode_advantage_stds.append(losses.get('advantage_std', 0.0))
                    episode_q1_values.append(losses.get('q1_value', 0.0))
                    episode_q2_values.append(losses.get('q2_value', 0.0))
            else:
                losses = None
            
            # Update state
            obs_dict = next_obs_dict
            episode_steps += 1
            global_step += 1
        
        # Episode summary
        avg_reward = np.mean(list(episode_rewards.values()))
        
        # Calculate average training metrics from accumulated losses
        if episode_critic_losses:
            critic_loss = np.mean(episode_critic_losses)
            q1_mean = np.mean(episode_q1_values)
            q2_mean = np.mean(episode_q2_values)
        else:
            critic_loss = q1_mean = q2_mean = 0.0
        
        if episode_actor_losses:
            actor_loss = np.mean(episode_actor_losses)
            mean_advantage = np.mean(episode_advantages)
            advantage_std = np.mean(episode_advantage_stds)
        else:
            actor_loss = mean_advantage = advantage_std = 0.0
        
        # Record metrics
        episode_metrics = {
            'episode': episode,
            'avg_reward': avg_reward,
            'total_reward': sum(episode_rewards.values()),
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'q1_mean': q1_mean,
            'q2_mean': q2_mean,
            'mean_advantage': mean_advantage,
            'advantage_std': advantage_std,
            'churn_triggered': churn_triggered,
            'global_step': global_step,
            'episode_steps': episode_steps
        }
        
        # Add per-manager rewards
        for mid, rew in episode_rewards.items():
            episode_metrics[f'reward_{mid}'] = rew
        
        training_history.append(episode_metrics)
        
        # Mini log: print episode summary (always, in mini_log mode)
        if mini_log:
            actor_updates_count = len(episode_actor_losses)
            status = "[CHURN]" if churn_triggered else "[OK]  "
            print(f" {status} Reward: {avg_reward:7.1f} | "
                  f"CriticLoss: {critic_loss:6.2f} | "
                  f"ActorLoss: {actor_loss:6.3f} ({actor_updates_count}x) | "
                  f"Q: {q1_mean:5.1f}/{q2_mean:5.1f}")
        
        # Logging (detailed, goes to file and console if not mini_log)
        if episode % EVAL_INTERVAL == 0:
            actor_updates_count = len(episode_actor_losses)
            logger.info(f"Episode {episode:3d}: "
                       f"reward={avg_reward:8.2f}, "
                       f"critic_loss={critic_loss:8.2f}, "
                       f"actor_loss={actor_loss:8.4f} ({actor_updates_count}×), "
                       f"Q1={q1_mean:6.2f}, Q2={q2_mean:6.2f}, "
                       f"steps={episode_steps}, "
                       f"churn={'YES' if churn_triggered else 'NO'}")
        
        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            save_model(agent, os.path.join(output_dir, f"model_ep{episode+1}.pt"))
            logger.info(f"Checkpoint saved: episode {episode+1}")
    
    # Mini log: training complete summary
    if mini_log:
        final_avg_reward = np.mean([h['avg_reward'] for h in training_history[-10:]])
        print("\n" + "=" * 80)
        print(f">>> Training Complete! Final 10-ep avg reward: {final_avg_reward:.2f}")
        print("=" * 80 + "\n")
    
    return training_history

# =============================================================================
# SAVE/LOAD FUNCTIONS
# =============================================================================

def save_model(agent, path):
    """Save trained model"""
    torch.save({
        'actor': agent.actor.state_dict(),
        'critics': agent.critics.state_dict(),
        'actor_target': agent.actor_target.state_dict(),
        'critics_target': agent.critics_target.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
    }, path)

def load_model(agent, path):
    """Load trained model"""
    checkpoint = torch.load(path, map_location=DEVICE)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critics.load_state_dict(checkpoint['critics'])
    agent.actor_target.load_state_dict(checkpoint['actor_target'])
    agent.critics_target.load_state_dict(checkpoint['critics_target'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

# =============================================================================
# TEST FUNCTION
# =============================================================================

def test(env, agent, num_episodes, logger, output_dir):
    """
    Test agent performance with periodic churn
    
    Args:
        env: Environment instance
        agent: Trained EAAgent instance
        num_episodes: Number of test episodes
        logger: Logger instance
        output_dir: Output directory
    
    Returns:
        test_history: List of episode metrics
    """
    logger.info("=" * 80)
    logger.info(f"Starting EA Testing: {num_episodes} episodes")
    logger.info(f"Churn every {CHURN_TRIGGER_INTERVAL} episodes")
    logger.info("=" * 80)
    
    test_history = []
    
    for episode in range(num_episodes):
        # Reset environment (returns observations and infos)
        env_obs, env_infos = env.reset()
        obs_dict = get_observations(env, episode_step=0)
        
        episode_rewards = {mid: 0.0 for mid in env.manager_agents.keys()}
        episode_steps = 0
        done = False
        
        # Check for churn event from infos
        churn_triggered = False
        churn_info = {}
        for mid, info in env_infos.items():
            if 'churn_events' in info and len(info['churn_events']) > 0:
                churn_triggered = True
                churn_info[mid] = info['churn_events'][0]
        
        if churn_triggered:
            logger.info(f"\nEpisode {episode}: CHURN EVENT")
            for mid, event_dict in churn_info.items():
                logger.info(f"  {mid}: severity={event_dict.get('severity', 0):.2%}, "
                          f"{event_dict.get('devices_left', 0)} left, "
                          f"{event_dict.get('devices_joined', 0)} joined")
        
        # Episode loop (deterministic actions)
        while not done and episode_steps < TIME_HORIZON:
            actions_dict = {}
            
            for manager_id, obs in obs_dict.items():
                manager_idx = list(env.manager_agents.keys()).index(manager_id)
                
                # Select action (no noise during testing)
                A_padded = agent.select_action(
                    obs['g'], obs['X'], obs['mask'], manager_idx, explore=False
                )
                
                actions_dict[manager_id] = A_padded.cpu().numpy()[0]
            
            # Convert and step
            env_actions = convert_actions_to_env_format(actions_dict, obs_dict)
            next_env_obs, rewards, dones_dict, truncated, info = env.step(env_actions)
            done = dones_dict.get('__all__', False)  # Extract boolean from dict
            next_obs_dict = get_observations(env, episode_step=episode_steps+1)
            
            # Accumulate rewards
            for manager_id in env.manager_agents.keys():
                episode_rewards[manager_id] += rewards.get(manager_id, 0.0)
            
            obs_dict = next_obs_dict
            episode_steps += 1
        
        # Record test metrics
        avg_reward = np.mean(list(episode_rewards.values()))
        test_history.append({
            'episode': episode,
            'avg_reward': avg_reward,
            'total_reward': sum(episode_rewards.values()),
            'churn_triggered': churn_triggered,
            'episode_steps': episode_steps
        })
        
        logger.info(f"Test Episode {episode:3d}: reward={avg_reward:8.2f}, "
                   f"churn={'YES' if churn_triggered else 'NO'}")
    
    return test_history

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(training_history, test_history, output_dir, logger):
    """Save training and test results"""
    
    # Save training history to CSV
    if SAVE_CSV and training_history:
        train_df = pd.DataFrame(training_history)
        train_csv = os.path.join(output_dir, "training_history.csv")
        train_df.to_csv(train_csv, index=False)
        logger.info(f"Training history saved to {train_csv}")
    
    # Save test history to CSV
    if SAVE_CSV and test_history:
        test_df = pd.DataFrame(test_history)
        test_csv = os.path.join(output_dir, "test_history.csv")
        test_df.to_csv(test_csv, index=False)
        logger.info(f"Test history saved to {test_csv}")
    
    # Save to JSON
    if SAVE_JSON:
        results = {
            'config': {
                'env_type': ENV_TYPE,
                'churn_severity': CHURN_SEVERITY,
                'churn_frequency': CHURN_FREQUENCY,
                'train_episodes': TRAIN_EPISODES,
                'test_episodes': TEST_EPISODES,
                'device': DEVICE
            },
            'training_history': training_history,
            'test_history': test_history
        }
        
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_path}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='EA Algorithm Test - 4Manager Mid/MidF')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'both'],
                       help='Run mode: train, test, or both')
    parser.add_argument('--load_path', type=str, default=None,
                       help='Path to load trained model')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--mini_log', action='store_true',
                       help='Use minimal logging (only show key episode info)')
    
    args = parser.parse_args()
    
    # Use command line arguments or defaults
    use_gpu = args.gpu if args.gpu else USE_GPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    train_episodes = args.episodes if args.episodes is not None else TRAIN_EPISODES
    
    # Setup logging
    logger = setup_logging(OUTPUT_DIR, mini_log=args.mini_log)
    
    logger.info("Configuration:")
    logger.info(f"  Environment: {ENV_TYPE}")
    logger.info(f"  Churn: {CHURN_SEVERITY} severity, {CHURN_FREQUENCY} frequency")
    logger.info(f"  Severity levels: {CHURN_SEVERITY_LEVELS}")
    logger.info(f"  Trigger interval: {CHURN_TRIGGER_INTERVAL} episodes")
    logger.info(f"  Device: {device}")
    logger.info(f"  N_max: {N_MAX}, x_dim: {X_DIM}, g_dim: {G_DIM}")
    logger.info(f"  Batch size: {BATCH_SIZE}, LR: actor={LR_ACTOR}, critic={LR_CRITIC}")
    logger.info(f"  Training episodes: {train_episodes}")
    
    # Create environment
    logger.info("\nCreating environment...")
    env = create_environment()
    num_managers = len(env.manager_agents)
    logger.info(f"Environment created with {num_managers} managers")
    
    # Create agent
    logger.info("\nCreating EA agent...")
    agent = create_agent(num_managers)
    logger.info(f"EA Agent created successfully")
    
    training_history = []
    test_history = []
    
    # Training mode
    if args.mode in ['train', 'both']:
        training_history = train(env, agent, train_episodes, logger, OUTPUT_DIR, mini_log=args.mini_log)
        
        # Save final model
        final_model_path = os.path.join(OUTPUT_DIR, "model_final.pt")
        save_model(agent, final_model_path)
        logger.info(f"\nFinal model saved to {final_model_path}")
    
    # Testing mode
    if args.mode in ['test', 'both']:
        # Load model if path provided
        if args.load_path:
            logger.info(f"\nLoading model from {args.load_path}")
            load_model(agent, args.load_path)
        
        test_history = test(env, agent, TEST_EPISODES, logger, OUTPUT_DIR)
    
    # Save results
    save_results(training_history, test_history, OUTPUT_DIR, logger)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    if training_history:
        final_reward = training_history[-1]['avg_reward']
        initial_reward = training_history[0]['avg_reward']
        improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100
        logger.info(f"Initial reward: {initial_reward:.2f}")
        logger.info(f"Final reward: {final_reward:.2f}")
        logger.info(f"Improvement: {improvement:.2f}%")
    
    if test_history:
        avg_test_reward = np.mean([h['avg_reward'] for h in test_history])
        logger.info(f"Average test reward: {avg_test_reward:.2f}")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
