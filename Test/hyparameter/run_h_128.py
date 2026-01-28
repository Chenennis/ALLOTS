#!/usr/bin/env python
"""
超参数敏感性分析: H = 128 (默认值)
运行环境: 4manager-mid, 10manager-mid
种子: 1, 2, 3

注意: 此配置与 τw=1.0 相同，可复用 run_tau_1.0.py 的结果

Usage:
    python Test/hyparameter/run_h_128.py
"""

import sys
from pathlib import Path

# 确保从项目根目录运行
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Test.hyparameter.hyperparameter_train import run_hyperparameter_sweep

if __name__ == '__main__':
    run_hyperparameter_sweep(
        param_name='H',
        param_value=128,
        envs=['4manager', '10manager'],
        seeds=[1, 2, 3],
    )
