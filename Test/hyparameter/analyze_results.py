#!/usr/bin/env python
"""
超参数敏感性分析 - 结果分析脚本

分析τw和H超参数对ALLOTS性能的影响，生成论文用的表格和图表。

Usage:
    cd <project_root>
    python Test/hyparameter/analyze_results.py

Author: FOenv Team
Date: 2026-01-24
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


# 配置
RESULTS_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR = Path(__file__).parent / 'analysis'

# 超参数配置
TAU_W_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
H_VALUES = [64, 128, 256]
ENVS = ['4manager', '10manager']
SEEDS = [1, 2, 3]


def load_all_results() -> Dict[str, Dict[str, List[Dict]]]:
    """加载所有实验结果"""
    results = {
        'tau_w': {},  # {value: [{env, seed, reward, ...}, ...]}
        'H': {},
    }
    
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return results
    
    # 加载τw结果
    for tau in TAU_W_VALUES:
        config_dir = RESULTS_DIR / f"tau_{tau}"
        if config_dir.exists():
            results['tau_w'][tau] = []
            for env in ENVS:
                for seed in SEEDS:
                    summary_path = config_dir / f"{env}_seed{seed}" / "summary.json"
                    if summary_path.exists():
                        with open(summary_path) as f:
                            data = json.load(f)
                            results['tau_w'][tau].append(data)
    
    # 加载H结果
    for h in H_VALUES:
        config_dir = RESULTS_DIR / f"h_{h}"
        if config_dir.exists():
            results['H'][h] = []
            for env in ENVS:
                for seed in SEEDS:
                    summary_path = config_dir / f"{env}_seed{seed}" / "summary.json"
                    if summary_path.exists():
                        with open(summary_path) as f:
                            data = json.load(f)
                            results['H'][h].append(data)
    
    # 特殊处理: τw=1.0和H=128是同一组实验
    # 如果H=128目录不存在但τw=1.0存在，复用τw=1.0的结果
    if 128 not in results['H'] or not results['H'][128]:
        if 1.0 in results['tau_w'] and results['tau_w'][1.0]:
            results['H'][128] = results['tau_w'][1.0]
            print("Note: Using τw=1.0 results for H=128 (same configuration)")
    
    return results


def compute_statistics(results: List[Dict], env: str = None) -> Dict[str, float]:
    """计算统计量"""
    if env:
        filtered = [r for r in results if r.get('env_type') == env and 'error' not in r]
    else:
        filtered = [r for r in results if 'error' not in r]
    
    if not filtered:
        return {'mean': np.nan, 'std': np.nan, 'n': 0}
    
    rewards = [r['final_reward_50ep'] for r in filtered]
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'n': len(rewards),
    }


def generate_tables(results: Dict) -> str:
    """生成论文用的表格 (Markdown格式)"""
    output = []
    
    # Table 1: τw sensitivity
    output.append("## Table 1: Sensitivity to Advantage Temperature τw (H=128)")
    output.append("")
    output.append("| τw | 4Manager-Mid | 10Manager-Mid |")
    output.append("|:--:|:------------:|:-------------:|")
    
    for tau in TAU_W_VALUES:
        if tau in results['tau_w']:
            stat_4m = compute_statistics(results['tau_w'][tau], '4manager')
            stat_10m = compute_statistics(results['tau_w'][tau], '10manager')
            
            cell_4m = f"{stat_4m['mean']:.2f} ± {stat_4m['std']:.2f}" if not np.isnan(stat_4m['mean']) else "N/A"
            cell_10m = f"{stat_10m['mean']:.2f} ± {stat_10m['std']:.2f}" if not np.isnan(stat_10m['mean']) else "N/A"
            
            # 标记默认值
            tau_str = f"**{tau}**" if tau == 1.0 else str(tau)
            output.append(f"| {tau_str} | {cell_4m} | {cell_10m} |")
        else:
            output.append(f"| {tau} | N/A | N/A |")
    
    output.append("")
    output.append("*Default value (τw=1.0) shown in bold*")
    output.append("")
    
    # Table 2: H sensitivity
    output.append("## Table 2: Sensitivity to Token Dimension H (τw=1.0)")
    output.append("")
    output.append("| H | 4Manager-Mid | 10Manager-Mid |")
    output.append("|:-:|:------------:|:-------------:|")
    
    for h in H_VALUES:
        if h in results['H']:
            stat_4m = compute_statistics(results['H'][h], '4manager')
            stat_10m = compute_statistics(results['H'][h], '10manager')
            
            cell_4m = f"{stat_4m['mean']:.2f} ± {stat_4m['std']:.2f}" if not np.isnan(stat_4m['mean']) else "N/A"
            cell_10m = f"{stat_10m['mean']:.2f} ± {stat_10m['std']:.2f}" if not np.isnan(stat_10m['mean']) else "N/A"
            
            # 标记默认值
            h_str = f"**{h}**" if h == 128 else str(h)
            output.append(f"| {h_str} | {cell_4m} | {cell_10m} |")
        else:
            output.append(f"| {h} | N/A | N/A |")
    
    output.append("")
    output.append("*Default value (H=128) shown in bold*")
    output.append("")
    
    return "\n".join(output)


def generate_plots(results: Dict, output_dir: Path):
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: τw sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        
        means = []
        stds = []
        valid_taus = []
        
        for tau in TAU_W_VALUES:
            if tau in results['tau_w']:
                stat = compute_statistics(results['tau_w'][tau], env)
                if not np.isnan(stat['mean']):
                    means.append(stat['mean'])
                    stds.append(stat['std'])
                    valid_taus.append(tau)
        
        if valid_taus:
            ax.errorbar(valid_taus, means, yerr=stds, 
                       marker='o', markersize=8, capsize=5, linewidth=2)
            ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Default (τw=1.0)')
            ax.set_xlabel('Advantage Temperature τw', fontsize=12)
            ax.set_ylabel('Average Reward (50ep)', fontsize=12)
            ax.set_title(f'{env.replace("manager", "-Manager")} Environment', fontsize=14)
            ax.set_xscale('log')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tau_w_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: H sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        
        means = []
        stds = []
        valid_hs = []
        
        for h in H_VALUES:
            if h in results['H']:
                stat = compute_statistics(results['H'][h], env)
                if not np.isnan(stat['mean']):
                    means.append(stat['mean'])
                    stds.append(stat['std'])
                    valid_hs.append(h)
        
        if valid_hs:
            ax.bar(range(len(valid_hs)), means, yerr=stds, 
                  capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(valid_hs)])
            ax.set_xticks(range(len(valid_hs)))
            ax.set_xticklabels([str(h) for h in valid_hs])
            ax.axhline(y=means[valid_hs.index(128)] if 128 in valid_hs else 0, 
                      color='red', linestyle='--', alpha=0.5, label='Default (H=128)')
            ax.set_xlabel('Token Dimension H', fontsize=12)
            ax.set_ylabel('Average Reward (50ep)', fontsize=12)
            ax.set_title(f'{env.replace("manager", "-Manager")} Environment', fontsize=14)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'h_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_csv(results: Dict, output_dir: Path):
    """生成CSV格式的结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # τw results
    tau_rows = []
    for tau in TAU_W_VALUES:
        if tau in results['tau_w']:
            for r in results['tau_w'][tau]:
                if 'error' not in r:
                    tau_rows.append({
                        'tau_w': tau,
                        'env': r['env_type'],
                        'seed': r['seed'],
                        'final_reward': r['final_reward_50ep'],
                        'final_std': r['final_reward_std'],
                        'initial_reward': r['initial_reward_10ep'],
                    })
    
    if tau_rows:
        pd.DataFrame(tau_rows).to_csv(output_dir / 'tau_w_results.csv', index=False)
    
    # H results
    h_rows = []
    for h in H_VALUES:
        if h in results['H']:
            for r in results['H'][h]:
                if 'error' not in r:
                    h_rows.append({
                        'H': h,
                        'env': r['env_type'],
                        'seed': r['seed'],
                        'final_reward': r['final_reward_50ep'],
                        'final_std': r['final_reward_std'],
                        'initial_reward': r['initial_reward_10ep'],
                    })
    
    if h_rows:
        pd.DataFrame(h_rows).to_csv(output_dir / 'h_results.csv', index=False)
    
    print(f"CSV files saved to {output_dir}")


def main():
    print("=" * 60)
    print("ALLOTS Hyperparameter Sensitivity Analysis")
    print("=" * 60)
    print()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载结果
    print("Loading results...")
    results = load_all_results()
    
    # 统计加载的实验数量
    tau_count = sum(len(v) for v in results['tau_w'].values())
    h_count = sum(len(v) for v in results['H'].values())
    print(f"  Loaded: {tau_count} τw experiments, {h_count} H experiments")
    print()
    
    if tau_count == 0 and h_count == 0:
        print("No results found! Please run experiments first:")
        print("  bash Test/hyparameter/run_all_parallel.sh")
        return 1
    
    # 生成表格
    print("Generating tables...")
    tables = generate_tables(results)
    print(tables)
    
    # 保存Markdown报告
    report_path = OUTPUT_DIR / 'sensitivity_report.md'
    with open(report_path, 'w') as f:
        f.write("# ALLOTS Hyperparameter Sensitivity Analysis\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Experiment Configuration\n\n")
        f.write("- Environments: 4Manager-Mid, 10Manager-Mid\n")
        f.write("- Mid-episode churn at steps [6, 12, 18]\n")
        f.write("- 3 random seeds per configuration\n")
        f.write("- 500 training episodes\n\n")
        f.write(tables)
    print(f"\nReport saved to: {report_path}")
    
    # 生成CSV
    print("\nGenerating CSV files...")
    generate_csv(results, OUTPUT_DIR)
    
    # 生成图表
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        generate_plots(results, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
