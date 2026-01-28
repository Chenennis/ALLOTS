#!/usr/bin/env python
"""
超参数敏感性分析: τw = 2.0 (较保守)
运行环境: 4manager-mid, 10manager-mid
种子: 1, 2, 3

Usage:
    python Test/hyparameter/run_tau_2.0.py
"""

import sys
from pathlib import Path

# 确保从项目根目录运行
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Test.hyparameter.hyperparameter_train import run_hyperparameter_sweep

if __name__ == '__main__':
    run_hyperparameter_sweep(
        param_name='tau_w',
        param_value=2.0,
        envs=['4manager', '10manager'],
        seeds=[1, 2, 3],
    )
