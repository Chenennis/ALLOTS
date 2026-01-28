"""
MAAC Baseline Test: 10 Managers with Mid Severity, Mid-Episode Churn

This test validates MAAC's performance under device churn using compatibility layer.

- Algorithm: MAAC (Multi-Actor-Attention-Critic)
- Environment: 10 Managers
- Churn Severity: Mid (20-30% devices change)
- Churn Type: Mid-Episode (triggered at steps 6, 12, 18)
- Training: 500 episodes

Author: FOenv Team
Date: 2026-01-25
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
from Test.enhance.envs.mid_episode_churn_wrapper import MidEpisodeChurnWrapper
from algorithms.MAAC.compat_wrapper import MAACCompatAgent

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
CHURN_SEVERITY = "mid"
CHURN_SEVERITY_LEVELS = [0.20, 0.25, 0.30]
CHURN_SEVERITY_PROBS = [0.4, 0.3, 0.3]
CHURN_TRIGGER_INTERVAL = 1
MIN_ACTIVE_DEVICES = 5
MID_EPISODE_CHURN_STEPS = [6, 12, 18]

# ============== Compatibility Layer Parameters ==============
N_MAX = 80
X_DIM = 6
G_DIM = 26
P = 5

# ============== MAAC Hyperparameters ==============
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
HIDDEN_DIM = 128
ATTEND_HEADS = 4
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
OUTPUT_DIR = f"Test/Test/results/maac_{ENV_TYPE}_{CHURN_SEVERITY}_midf"
SAVE_CSV = True
SAVE_JSON = True

# =============================================================================
# OBSERVATION EXTRACTION (Convert env obs to dict format)
# =============================================================================

def get_observations(env):
    """Extract observations from environment in dict format."""
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
    """Convert padded actions to environment format."""
    from algorithms.EA.foea.fogym_adapter import convert_ea_action_to_fogym
    
    env_actions = {}
    for manager_id, A_pad in padded_actions.items():
        device_ids = obs_dict[manager_id]['device_ids']
        mask = np.array([1.0] * len(device_ids) + [0.0] * (N_MAX - len(device_ids)))
        action = convert_ea_action_to_fogym(A_pad, mask, device_ids)
        env_actions[manager_id] = action
    return env_actions

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
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train():
    """Main training loop with mid-episode churn."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR, mini_log=True)
    
    print("=" * 80)
    print(f"MAAC Test: {ENV_TYPE} | {CHURN_SEVERITY} churn | Mid-Episode | Device: {DEVICE}")
    print(f"Churn Steps: {MID_EPISODE_CHURN_STEPS}")
    print("=" * 80)
    
    # Create churn config
    churn_config = ChurnConfig(
        enabled=CHURN_ENABLED,
        trigger_interval=CHURN_TRIGGER_INTERVAL,
        severity_levels=tuple(CHURN_SEVERITY_LEVELS),
        severity_probs=tuple(CHURN_SEVERITY_PROBS),
        min_active_devices=MIN_ACTIVE_DEVICES
    )
    
    # Create base environment
    base_env = MultiAgentFlexOfferEnv(
        data_dir=DATA_DIR,
        time_horizon=TIME_HORIZON,
        time_step=TIME_STEP,
        churn_config=churn_config,
        aggregation_method=AGGREGATION_METHOD,
        trading_method=TRADING_METHOD,
        disaggregation_method=DISAGGREGATION_METHOD
    )
    
    # Wrap with MidEpisodeChurnWrapper
    env = MidEpisodeChurnWrapper(
        base_env,
        churn_steps=MID_EPISODE_CHURN_STEPS,
        verbose=False
    )
    
    manager_ids = list(env.manager_ids)
    
    # Create MAAC agent
    agent = MAACCompatAgent(
        manager_ids=manager_ids,
        N_max=N_MAX,
        x_dim=X_DIM,
        g_dim=G_DIM,
        p=P,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        hidden_dim=HIDDEN_DIM,
        attend_heads=ATTEND_HEADS,
        gamma=GAMMA,
        tau=TAU,
        noise_scale=NOISE_SCALE,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    # Training history
    history = []
    global_step = 0
    
    # Training loop
    for episode in range(TRAIN_EPISODES):
        env_obs, env_infos = env.reset()
        obs_dict = get_observations(env)  # Convert to dict format
        
        episode_rewards = {mid: 0.0 for mid in manager_ids}
        episode_steps = 0
        done = False
        churn_count = 0
        
        # Check reset-time churn
        for mid, info in env_infos.items():
            if 'churn_events' in info and len(info['churn_events']) > 0:
                churn_count += 1
                break
        
        episode_actor_losses = []
        episode_critic_losses = []
        
        while not done and episode_steps < TIME_HORIZON:
            # Select actions
            explore = (global_step >= WARMUP_EPISODES * TIME_HORIZON)
            padded_actions = agent.select_actions(obs_dict, explore=explore)
            env_actions = convert_actions_to_env_format(padded_actions, obs_dict)
            
            # Step environment
            next_env_obs, rewards, dones_dict, truncated, infos = env.step(env_actions)
            done = dones_dict.get('__all__', False)
            next_obs_dict = get_observations(env)
            
            # Check for mid-episode churn
            for mid in manager_ids:
                if infos.get(mid, {}).get('mid_episode_churn', False):
                    churn_count += 1
                    break
            
            # Accumulate rewards
            for mid in manager_ids:
                episode_rewards[mid] += rewards.get(mid, 0.0)
            
            # Store transition
            agent.store_transition(obs_dict, padded_actions, rewards, next_obs_dict, done)
            
            # Update
            if agent.get_buffer_size() >= BATCH_SIZE:
                metrics = agent.update()
                if metrics:
                    episode_actor_losses.append(metrics.get('actor_loss', 0))
                    episode_critic_losses.append(metrics.get('critic_loss', 0))
            
            obs_dict = next_obs_dict
            episode_steps += 1
            global_step += 1
        
        # Episode stats
        episode_reward = sum(episode_rewards.values()) / len(manager_ids)
        avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
        avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
        
        # Log
        churn_tag = f"[CHURN x{churn_count}]" if churn_count > 0 else ""
        print(f"Ep {episode+1:3d} {churn_tag:12s} Reward: {episode_reward:7.1f} | "
              f"ActorLoss: {avg_actor_loss:7.3f} | CriticLoss: {avg_critic_loss:7.3f}")
        
        # Save history
        history.append({
            'episode': episode + 1,
            'avg_reward': episode_reward,
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'churn_count': churn_count
        })
        
        # Save model periodically
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save(os.path.join(OUTPUT_DIR, f"model_ep{episode+1}.pt"))
    
    # Save final model and history
    agent.save(os.path.join(OUTPUT_DIR, "model_final.pt"))
    
    if SAVE_CSV:
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    
    if SAVE_JSON:
        config = {
            "algorithm": "MAAC",
            "env_type": ENV_TYPE,
            "churn_severity": CHURN_SEVERITY,
            "churn_type": "mid_episode",
            "churn_steps": MID_EPISODE_CHURN_STEPS,
            "n_managers": len(manager_ids),
            "n_max": N_MAX,
            "episodes": TRAIN_EPISODES,
            "hyperparameters": {
                "lr_actor": LR_ACTOR,
                "lr_critic": LR_CRITIC,
                "hidden_dim": HIDDEN_DIM,
                "attend_heads": ATTEND_HEADS,
                "gamma": GAMMA,
                "tau": TAU
            }
        }
        with open(os.path.join(OUTPUT_DIR, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    print()
    print("=" * 80)
    print(f">>> Training Complete! Final 10-ep avg reward: {np.mean([h['avg_reward'] for h in history[-10:]]):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    train()
