#!/usr/bin/env python
"""
Parallel Runner for Enhanced Comparative Study

Runs all 36 experiments with configurable parallelism.

Author: FOenv Team
Date: 2026-01-23
"""

import subprocess
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Configuration
ENVS = ['4manager', '10manager']
CHURNS = ['low', 'mid', 'high']
ALGORITHMS = ['ea', 'maddpg', 'matd3', 'sqddpg', 'mappo', 'maippo']

SCRIPT_DIR = Path(__file__).parent / 'scripts'
RESULTS_DIR = Path(__file__).parent / 'results'


def run_experiment(exp_config):
    """Run a single experiment."""
    algo, env, churn = exp_config
    
    results_path = RESULTS_DIR / f'{algo}_{env}_{churn}'
    
    if algo == 'ea':
        cmd = [
            sys.executable, str(SCRIPT_DIR / 'train_ea.py'),
            '--env', env,
            '--churn', churn,
            '--results_dir', str(results_path)
        ]
    else:
        cmd = [
            sys.executable, str(SCRIPT_DIR / 'train_baseline.py'),
            '--algorithm', algo,
            '--env', env,
            '--churn', churn,
            '--results_dir', str(results_path)
        ]
    
    start_time = time.time()
    print(f"  [START] {algo}_{env}_{churn}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Get SSR from summary
            summary_path = results_path / 'summary.json'
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                    ssr = summary.get('SSR', 'N/A')
            else:
                ssr = 'N/A'
            
            print(f"  [DONE] {algo}_{env}_{churn} | SSR: {ssr:.2f} | Time: {elapsed:.0f}s")
            return (algo, env, churn, 'success', ssr, elapsed)
        else:
            print(f"  [FAIL] {algo}_{env}_{churn} | Error: {result.stderr[:200]}")
            return (algo, env, churn, 'failed', None, elapsed)
            
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {algo}_{env}_{churn}")
        return (algo, env, churn, 'timeout', None, 7200)
    except Exception as e:
        print(f"  [ERROR] {algo}_{env}_{churn}: {str(e)}")
        return (algo, env, churn, 'error', None, 0)


def main():
    parser = argparse.ArgumentParser(description='Run enhanced comparative study')
    parser.add_argument('--parallel', type=int, default=6,
                       help='Number of parallel experiments (default: 6)')
    parser.add_argument('--algorithms', nargs='+', default=ALGORITHMS,
                       help='Algorithms to run')
    parser.add_argument('--envs', nargs='+', default=ENVS,
                       help='Environments to run')
    parser.add_argument('--churns', nargs='+', default=CHURNS,
                       help='Churn levels to run')
    parser.add_argument('--ea_only', action='store_true',
                       help='Run only EA experiments')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Run only baseline experiments')
    
    args = parser.parse_args()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build experiment list
    experiments = []
    
    algorithms = args.algorithms
    if args.ea_only:
        algorithms = ['ea']
    elif args.baseline_only:
        algorithms = [a for a in algorithms if a != 'ea']
    
    for algo in algorithms:
        for env in args.envs:
            for churn in args.churns:
                experiments.append((algo, env, churn))
    
    print("=" * 70)
    print("Enhanced Comparative Study - Parallel Runner")
    print("=" * 70)
    print(f"Total experiments: {len(experiments)}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Algorithms: {algorithms}")
    print(f"Environments: {args.envs}")
    print(f"Churn levels: {args.churns}")
    print("=" * 70)
    print()
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in experiments}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    successful = [r for r in results if r[3] == 'success']
    failed = [r for r in results if r[3] != 'success']
    
    print(f"\nSuccessful: {len(successful)} / {len(experiments)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    
    print("\n--- SSR Results ---")
    for algo, env, churn, status, ssr, time_elapsed in sorted(results):
        if status == 'success':
            print(f"  {algo}_{env}_{churn}: {ssr:.2f}")
        else:
            print(f"  {algo}_{env}_{churn}: {status}")
    
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(results, columns=['algorithm', 'env', 'churn', 'status', 'ssr', 'time'])
    df.to_csv(RESULTS_DIR / 'experiment_summary.csv', index=False)
    print(f"\nResults saved to: {RESULTS_DIR / 'experiment_summary.csv'}")


if __name__ == '__main__':
    main()
