"""Analyze Multidrone results. Same structure as MPE analyze_results.py."""

import os, sys, glob, json, numpy as np, pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from Multidrone.drone_churn.churn_config import ALL_ALGORITHMS, ABLATION_VARIANTS


def analyze(results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "Multidrone", "results")
    summaries = []
    for p in glob.glob(os.path.join(results_dir, "*_summary.json")):
        with open(p) as f: summaries.append(json.load(f))
    if not summaries: print(f"No results in {results_dir}"); return
    df = pd.DataFrame(summaries)
    print(f"Found {len(df)} results\n")

    methods = ALL_ALGORITHMS + ABLATION_VARIANTS
    rows = []
    for m in methods:
        row = {'Method': m.upper()}
        for ch in ['low', 'mid', 'high']:
            sub = df[(df['algo']==m) & (df['churn']==ch)]
            if len(sub)>0:
                row[f'{ch.capitalize()} Churn'] = f"{sub['eval_mean'].mean():.3f} ± {sub['eval_mean'].std():.3f}"
            else: row[f'{ch.capitalize()} Churn'] = "—"
        rows.append(row)

    sdf = pd.DataFrame(rows)
    print("="*70)
    print("Multidrone — Multi-Controller Cooperative Coverage with Fleet Dynamics")
    print("="*70)
    print(sdf.to_string(index=False))
    sdf.to_csv(os.path.join(results_dir, "drone_summary.csv"), index=False)
    print(f"\nSaved to {results_dir}/drone_summary.csv")


if __name__ == '__main__':
    analyze()
