#!/usr/bin/env python
"""
运行增强版EA消融实验

测试渐进式Credit + 增强Pair-Set Critic的效果

Author: FOenv Team
Date: 2026-01-22
"""

import subprocess
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

# 环境和churn配置
ENV_TYPES = ['4manager', '10manager']
CHURN_TYPES = ['low', 'mid', 'high']

def run_experiment(env_type, churn_type):
    """运行单个实验"""
    results_dir = Path(__file__).parent / 'results_enhanced' / f'{env_type}_{churn_type}_enhanced_ea'
    
    cmd = [
        sys.executable, '-u',
        str(Path(__file__).parent / 'scripts_midchurn' / 'train_enhanced.py'),
        '--env', env_type,
        '--churn', churn_type,
        '--results_dir', str(results_dir),
        '--seed', '42'
    ]
    
    print(f"\n{'='*70}")
    print(f"Starting: {env_type}_{churn_type}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        print(f"ERROR: {env_type}_{churn_type} failed with code {result.returncode}")
    else:
        print(f"SUCCESS: {env_type}_{churn_type} completed")
    
    return result.returncode == 0


def main():
    # 创建结果目录
    results_dir = Path(__file__).parent / 'results_enhanced'
    results_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Enhanced EA Ablation Experiments")
    print("Features: Progressive Credit + Enhanced Pair-Set Critic")
    print("="*70)
    
    successes = 0
    failures = 0
    
    for env_type in ENV_TYPES:
        for churn_type in CHURN_TYPES:
            if run_experiment(env_type, churn_type):
                successes += 1
            else:
                failures += 1
    
    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Success: {successes}, Failed: {failures}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
