"""
Training script for Multidrone experiments. Same structure as train_mpe.py.

Usage:
    python Multidrone/drone_churn/train_drone.py --algo ea --churn mid --seed 42
"""

import sys, os, argparse, numpy as np, pandas as pd, json, time, torch, random
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Multidrone.drone_churn.drone_coverage_env import DroneChurnEnv
from Multidrone.drone_churn.drone_algo_wrapper import create_drone_agent
from Multidrone.drone_churn.churn_config import (
    CHURN_CONFIGS, TRAIN_EPISODES, EVAL_EPISODES, ALL_METHODS, AGENT_HZ, MAX_DURATION_SECONDS,
)


def set_seed(seed):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    results_dir = Path(project_root) / "Multidrone" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    churn_cfg = CHURN_CONFIGS[args.churn]
    env = DroneChurnEnv(churn_rho_range=(churn_cfg['rho_min'], churn_cfg['rho_max']), seed=args.seed)
    max_steps = env.max_steps
    print(f"Environment: Drone Coverage ({args.churn} churn), max_steps={max_steps}")

    agent = create_drone_agent(args.algo, device=device)
    print(f"Algorithm: {args.algo.upper()}")

    history = []
    start_time = time.time()

    for ep in range(args.episodes):
        agent.set_episode(ep)
        obs = env.reset()
        ep_rewards = {k: 0.0 for k in obs}
        ep_metrics = {}
        n_churns = 0

        for t in range(max_steps):
            actions = agent.select_action(obs, explore=True)
            next_obs, rewards, done, info = env.step(actions)
            metrics = agent.store_and_update(obs, actions, rewards, next_obs, done, info)
            for k in rewards: ep_rewards[k] += rewards[k]
            if info.get('churn_triggered'): n_churns += 1
            if metrics: ep_metrics = metrics
            obs = next_obs
            if done: break

        avg_reward = np.mean(list(ep_rewards.values()))
        row = {'episode': ep, 'avg_reward': avg_reward, 'n_churns': n_churns}
        for k, v in ep_rewards.items(): row[f'reward_{k}'] = v
        for k, v in ep_metrics.items():
            if isinstance(v, (int, float, np.floating)): row[k] = float(v)
        history.append(row)

        if ep % 100 == 0 or ep == args.episodes - 1:
            print(f"  Ep {ep:5d}/{args.episodes}: avg_reward={avg_reward:.4f}, churns={n_churns}, elapsed={time.time()-start_time:.0f}s")

    # Evaluation
    print(f"\nEvaluation ({args.eval_episodes} episodes)...")
    eval_history = []
    for eval_ep in range(args.eval_episodes):
        obs = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            actions = agent.select_action(obs, explore=False)
            obs, rewards, done, info = env.step(actions)
            ep_reward += np.mean(list(rewards.values()))
            if done: break
        eval_history.append({'eval_episode': eval_ep, 'avg_reward': ep_reward})
        print(f"  Eval {eval_ep}: reward={ep_reward:.4f}")

    eval_rewards = [e['avg_reward'] for e in eval_history]
    print(f"  Eval mean: {np.mean(eval_rewards):.4f} ± {np.std(eval_rewards):.4f}")

    # Save
    prefix = f"{args.algo}_{args.churn}_{args.seed}"
    pd.DataFrame(history).to_csv(results_dir / f"{prefix}_train.csv", index=False)
    pd.DataFrame(eval_history).to_csv(results_dir / f"{prefix}_eval.csv", index=False)
    with open(results_dir / f"{prefix}_summary.json", 'w') as f:
        json.dump({
            'algo': args.algo, 'churn': args.churn, 'seed': args.seed,
            'episodes': args.episodes, 'eval_episodes': args.eval_episodes, 'device': device,
            'eval_mean': float(np.mean(eval_rewards)), 'eval_std': float(np.std(eval_rewards)),
            'train_final_50_mean': float(np.mean([h['avg_reward'] for h in history[-50:]])),
            'total_time_seconds': time.time() - start_time,
        }, f, indent=2)

    env.close()
    print(f"Results saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=ALL_METHODS)
    parser.add_argument('--churn', type=str, default='mid', choices=['low', 'mid', 'high'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=TRAIN_EPISODES)
    parser.add_argument('--eval_episodes', type=int, default=EVAL_EPISODES)
    parser.add_argument('--cpu', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
