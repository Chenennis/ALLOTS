"""
Training and evaluation script for Modified MPE experiments.

Usage:
    python Modified_MPE/mpe_churn/train_mpe.py --algo ea --churn mid --seed 42
    python Modified_MPE/mpe_churn/train_mpe.py --algo maddpg --churn high --seed 123 --episodes 5000
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
import time
import torch
import random
from pathlib import Path

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Modified_MPE.mpe_churn.mpe_churn_env import MPEChurnEnv
from Modified_MPE.mpe_churn.mpe_algo_wrapper import create_mpe_agent
from Modified_MPE.mpe_churn.churn_config import (
    MPE_CHURN_CONFIGS, TRAIN_EPISODES, EVAL_EPISODES, T, ALL_METHODS,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """Main training loop."""
    set_seed(args.seed)

    # Setup output directory
    results_dir = Path(project_root) / "Modified_MPE" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    # Create environment
    churn_cfg = MPE_CHURN_CONFIGS[args.churn]
    env = MPEChurnEnv(
        churn_rho_range=(churn_cfg['rho_min'], churn_cfg['rho_max']),
        seed=args.seed,
    )
    print(f"Environment: MPE Churn ({churn_cfg['label']})")

    # Create algorithm wrapper
    agent = create_mpe_agent(args.algo, device=device)
    print(f"Algorithm: {args.algo.upper()}")

    # Training loop
    history = []
    start_time = time.time()

    for ep in range(args.episodes):
        agent.set_episode(ep)
        obs = env.reset()
        ep_rewards = {k: 0.0 for k in obs}
        ep_metrics = {}
        n_churns = 0

        for t in range(T):
            actions = agent.select_action(obs, explore=True)
            next_obs, rewards, done, info = env.step(actions)
            metrics = agent.store_and_update(obs, actions, rewards, next_obs, done, info)

            for k in rewards:
                ep_rewards[k] += rewards[k]
            if info.get('churn_triggered'):
                n_churns += 1
            if metrics:
                ep_metrics = metrics

            obs = next_obs
            if done:
                break

        avg_reward = np.mean(list(ep_rewards.values()))
        row = {
            'episode': ep,
            'avg_reward': avg_reward,
            'n_churns': n_churns,
        }
        for k, v in ep_rewards.items():
            row[f'reward_{k}'] = v
        for k, v in ep_metrics.items():
            if isinstance(v, (int, float, np.floating)):
                row[k] = float(v)
        history.append(row)

        if ep % 100 == 0 or ep == args.episodes - 1:
            elapsed = time.time() - start_time
            print(f"  Ep {ep:5d}/{args.episodes}: avg_reward={avg_reward:.4f}, "
                  f"churns={n_churns}, elapsed={elapsed:.0f}s")

    # Evaluation
    print(f"\nEvaluation ({args.eval_episodes} episodes)...")
    eval_history = []
    for eval_ep in range(args.eval_episodes):
        obs = env.reset()
        ep_reward = 0.0
        for t in range(T):
            actions = agent.select_action(obs, explore=False)
            next_obs, rewards, done, info = env.step(actions)
            ep_reward += np.mean(list(rewards.values()))
            obs = next_obs
            if done:
                break
        eval_history.append({'eval_episode': eval_ep, 'avg_reward': ep_reward})
        print(f"  Eval {eval_ep}: reward={ep_reward:.4f}")

    eval_rewards = [e['avg_reward'] for e in eval_history]
    print(f"  Eval mean: {np.mean(eval_rewards):.4f} ± {np.std(eval_rewards):.4f}")

    # Save results
    prefix = f"{args.algo}_{args.churn}_{args.seed}"

    train_csv = results_dir / f"{prefix}_train.csv"
    pd.DataFrame(history).to_csv(train_csv, index=False)

    eval_csv = results_dir / f"{prefix}_eval.csv"
    pd.DataFrame(eval_history).to_csv(eval_csv, index=False)

    summary = {
        'algo': args.algo,
        'churn': args.churn,
        'seed': args.seed,
        'episodes': args.episodes,
        'eval_episodes': args.eval_episodes,
        'device': device,
        'eval_mean': float(np.mean(eval_rewards)),
        'eval_std': float(np.std(eval_rewards)),
        'train_final_50_mean': float(np.mean([h['avg_reward'] for h in history[-50:]])),
        'total_time_seconds': time.time() - start_time,
    }
    summary_json = results_dir / f"{prefix}_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print(f"  {train_csv.name}")
    print(f"  {eval_csv.name}")
    print(f"  {summary_json.name}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train MARL algorithms on Modified MPE")
    parser.add_argument('--algo', type=str, required=True,
                        choices=ALL_METHODS,
                        help='Algorithm name')
    parser.add_argument('--churn', type=str, default='mid',
                        choices=['low', 'mid', 'high'],
                        help='Churn intensity level')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--episodes', type=int, default=TRAIN_EPISODES,
                        help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=EVAL_EPISODES,
                        help='Number of evaluation episodes')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA available')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
