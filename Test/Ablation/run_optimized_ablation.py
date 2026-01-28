#!/usr/bin/env python
"""
运行优化参数的消融实验 - 4个算法并行

参数配置:
- 4manager: original (blend=0.7, tau=2.0, weight=0.15)
- 10manager: option_AB (blend=0.3, tau=5.0, weight=0.05)

每个算法运行6个实验 (2 envs × 3 churns)
总共24个实验，4个脚本并行运行

Usage:
    python Test/Ablation/run_optimized_ablation.py

Author: FOenv Team
Date: 2026-01-21
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import argparse

# 禁用缓冲
sys.stdout.reconfigure(line_buffering=True)

EXPERIMENTS = [
    ('4manager', 'low'),
    ('4manager', 'mid'),
    ('4manager', 'high'),
    ('10manager', 'low'),
    ('10manager', 'mid'),
    ('10manager', 'high'),
]


def run_agent_experiments(agent_type, seed=42):
    """运行单个agent的所有实验"""
    script_path = Path(__file__).parent / 'scripts_midchurn' / 'train_optimized.py'
    results_base = Path(__file__).parent / 'results_optimized'
    
    print(f"\n{'='*70}")
    print(f"Agent: {agent_type.upper()}")
    print(f"{'='*70}")
    
    results = []
    
    for i, (env, churn) in enumerate(EXPERIMENTS, 1):
        results_dir = results_base / f'{env}_{churn}_{agent_type}'
        
        print(f"\n[{i}/6] {env}_{churn}_{agent_type}")
        
        cmd = [
            sys.executable, '-u', str(script_path),
            '--env', env,
            '--churn', churn,
            '--agent', agent_type,
            '--results_dir', str(results_dir),
            '--seed', str(seed),
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - start_time
        
        success = result.returncode == 0
        results.append((env, churn, success, elapsed))
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"     {status} ({elapsed/60:.1f}min)")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True,
                       choices=['full_ea', 'no_pairset', 'no_tdconsistent', 'no_credit'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"优化参数消融实验 - {args.agent.upper()}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n参数配置:")
    print("  4manager:  tau=2.0, weight=0.15, blend=0.7 (original)")
    print("  10manager: tau=5.0, weight=0.05, blend=0.3 (option_AB)")
    
    results = run_agent_experiments(args.agent, args.seed)
    
    success_count = sum(1 for r in results if r[2])
    total_time = sum(r[3] for r in results)
    
    print("\n" + "=" * 70)
    print(f"COMPLETED: {success_count}/6 succeeded")
    print(f"Total: {total_time/60:.1f} minutes")
    print("=" * 70)
    
    return 0 if success_count == 6 else 1


if __name__ == "__main__":
    sys.exit(main())
