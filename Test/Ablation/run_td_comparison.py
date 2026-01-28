#!/usr/bin/env python
"""
TD-Consistent模块消融对比测试

比较:
1. Full EA (方案D: 智能TD目标)
2. w/o TD-Consistent (使用当前mask)

测试环境: 4manager和10manager，3种churn强度
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# 测试配置
EXPERIMENTS = [
    # (环境, churn强度, 脚本类型)
    ('4manager', 'low_midf', 'full_ea'),
    ('4manager', 'low_midf', 'no_tdconsistent'),
    ('4manager', 'mid_midf', 'full_ea'),
    ('4manager', 'mid_midf', 'no_tdconsistent'),
    ('4manager', 'high_midf', 'full_ea'),
    ('4manager', 'high_midf', 'no_tdconsistent'),
    ('10manager', 'low_midf', 'full_ea'),
    ('10manager', 'low_midf', 'no_tdconsistent'),
    ('10manager', 'mid_midf', 'full_ea'),
    ('10manager', 'mid_midf', 'no_tdconsistent'),
    ('10manager', 'high_midf', 'full_ea'),
    ('10manager', 'high_midf', 'no_tdconsistent'),
]

def get_script_path(env, churn, script_type):
    """获取测试脚本路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if script_type == 'full_ea':
        # Full EA脚本在Test目录
        return os.path.join(base_dir, '..', f'{env}_{churn}_ea.py')
    else:
        # 消融脚本在Ablation/scripts目录
        return os.path.join(base_dir, 'scripts', f'{env}_{churn}_{script_type}.py')

def run_experiment(env, churn, script_type, seed=42):
    """运行单个实验"""
    script_path = get_script_path(env, churn, script_type)
    
    if not os.path.exists(script_path):
        print(f"  ⚠️ 脚本不存在: {script_path}")
        return False, 0
    
    start_time = time.time()
    
    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, script_path, '--seed', str(seed)],
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        elapsed = (time.time() - start_time) / 60
        
        if result.returncode == 0:
            print(f"  ✓ SUCCESS ({elapsed:.1f}min)")
            return True, elapsed
        else:
            print(f"  ✗ FAILED ({elapsed:.1f}min)")
            print(f"    Error: {result.stderr[-500:] if result.stderr else 'No error message'}")
            return False, elapsed
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT (>120min)")
        return False, 120
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False, 0

def main():
    print("=" * 80)
    print("TD-Consistent模块消融对比测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    print("测试配置:")
    print("  - Full EA: 使用方案D (智能TD目标)")
    print("  - w/o TD-Consistent: 使用当前mask")
    print("  - 环境: 4manager, 10manager")
    print("  - Churn: low, mid, high")
    print("  - 固定种子: 42")
    print()
    
    # 解析参数
    seed = 42
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        if idx + 1 < len(sys.argv):
            seed = int(sys.argv[idx + 1])
    
    # 并行运行选项
    parallel = '--parallel' in sys.argv
    
    results = []
    total_start = time.time()
    
    if parallel:
        print("并行模式: 同时运行Full EA和w/o TD-Consistent")
        print()
        
        # 按环境和churn分组，每组同时运行full_ea和no_tdconsistent
        import multiprocessing
        
        def run_pair(args):
            env, churn = args
            results = []
            for script_type in ['full_ea', 'no_tdconsistent']:
                print(f"\n[{env}_{churn}] {script_type}")
                success, elapsed = run_experiment(env, churn, script_type, seed)
                results.append((env, churn, script_type, success, elapsed))
            return results
        
        pairs = [
            ('4manager', 'low_midf'),
            ('4manager', 'mid_midf'),
            ('4manager', 'high_midf'),
            ('10manager', 'low_midf'),
            ('10manager', 'mid_midf'),
            ('10manager', 'high_midf'),
        ]
        
        # 使用进程池并行运行
        with multiprocessing.Pool(processes=2) as pool:
            all_results = pool.map(run_pair, pairs)
        
        for pair_results in all_results:
            results.extend(pair_results)
    else:
        # 顺序运行
        for i, (env, churn, script_type) in enumerate(EXPERIMENTS):
            print(f"\n[{i+1}/{len(EXPERIMENTS)}] {env}_{churn}_{script_type}")
            success, elapsed = run_experiment(env, churn, script_type, seed)
            results.append((env, churn, script_type, success, elapsed))
    
    # 总结
    total_elapsed = (time.time() - total_start) / 60
    success_count = sum(1 for r in results if r[3])
    
    print()
    print("=" * 80)
    print("测试完成")
    print("=" * 80)
    print(f"成功: {success_count}/{len(results)}")
    print(f"总时间: {total_elapsed:.1f}分钟")
    print()
    
    # 打印结果表格
    print("结果汇总:")
    print(f"{'环境':12s} {'Churn':10s} {'类型':20s} {'状态':8s} {'时间(min)':>10s}")
    print("-" * 65)
    for env, churn, script_type, success, elapsed in results:
        status = "✓ OK" if success else "✗ FAIL"
        print(f"{env:12s} {churn:10s} {script_type:20s} {status:8s} {elapsed:10.1f}")
    
    print()
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
