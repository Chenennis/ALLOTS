#!/usr/bin/env python
"""
Run w/o Per-Device Credit experiments with Mid-Episode Churn (6 tests)

Environments: 4manager, 10manager
Churn: low, mid, high

Usage:
    python Test/Ablation/run_midchurn_no_credit.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

AGENT_TYPE = 'no_credit'
EXPERIMENTS = [
    ('4manager', 'low'),
    ('4manager', 'mid'),
    ('4manager', 'high'),
    ('10manager', 'low'),
    ('10manager', 'mid'),
    ('10manager', 'high'),
]


def run_experiment(env, churn, seed=42):
    """Run single experiment"""
    script_path = Path(__file__).parent / 'scripts_midchurn' / 'train_with_midchurn.py'
    results_dir = Path(__file__).parent / 'results_midchurn' / f'{env}_{churn}_{AGENT_TYPE}'
    
    cmd = [
        sys.executable, str(script_path),
        '--env', env,
        '--churn', churn,
        '--agent', AGENT_TYPE,
        '--results_dir', str(results_dir),
        '--seed', str(seed),
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    return result.returncode == 0, elapsed


def main():
    print("=" * 70)
    print(f"Mid-Episode Churn Ablation: {AGENT_TYPE.upper()}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print("=" * 70)
    
    results = []
    total_start = time.time()
    
    for i, (env, churn) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {env}_{churn}_{AGENT_TYPE}")
        success, elapsed = run_experiment(env, churn)
        results.append((env, churn, success, elapsed))
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"     {status} ({elapsed/60:.1f}min)")
    
    total_elapsed = time.time() - total_start
    success_count = sum(1 for r in results if r[2])
    
    print("\n" + "=" * 70)
    print(f"COMPLETED: {success_count}/{len(EXPERIMENTS)} succeeded")
    print(f"Total: {total_elapsed/60:.1f} minutes")
    print("=" * 70)
    
    return 0 if success_count == len(EXPERIMENTS) else 1


if __name__ == "__main__":
    sys.exit(main())
