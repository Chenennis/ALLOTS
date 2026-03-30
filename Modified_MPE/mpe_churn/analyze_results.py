"""
Analyze MPE experiment results and generate summary tables.

Usage:
    python Modified_MPE/mpe_churn/analyze_results.py
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Modified_MPE.mpe_churn.churn_config import ALL_ALGORITHMS, ABLATION_VARIANTS, SEEDS


def analyze(results_dir: str = None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "Modified_MPE", "results")

    # Collect all summary JSONs
    summaries = []
    for json_path in glob.glob(os.path.join(results_dir, "*_summary.json")):
        with open(json_path) as f:
            summaries.append(json.load(f))

    if not summaries:
        print(f"No results found in {results_dir}")
        return

    df = pd.DataFrame(summaries)
    print(f"Found {len(df)} experiment results\n")

    # Build summary table: method × churn → mean±std of eval_mean
    methods = ALL_ALGORITHMS + ABLATION_VARIANTS
    churns = ['low', 'mid', 'high']

    rows = []
    for method in methods:
        row = {'Method': method.upper()}
        for churn in churns:
            subset = df[(df['algo'] == method) & (df['churn'] == churn)]
            if len(subset) > 0:
                mean = subset['eval_mean'].mean()
                std = subset['eval_mean'].std()
                row[f'{churn.capitalize()} Churn'] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[f'{churn.capitalize()} Churn'] = "—"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    print("=" * 70)
    print("Modified MPE — Multi-Controller Cooperative Coverage with Entity Churn")
    print("Average reward (mean ± std over seeds, eval episodes)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()

    # Save summary CSV
    summary_path = os.path.join(results_dir, "mpe_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    analyze()
