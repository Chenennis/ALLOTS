#!/usr/bin/env python
"""
Run All w/o Pair-Set Ablation Experiments (6 tests)

Environments: 4manager, 10manager
Churn: low, mid, high (all midf frequency)

Usage:
    python Test/Ablation/run_ablation_no_pairset.py --mini_log

Author: FOenv Team
Date: 2026-01-20
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time

ABLATION_TYPE = "no_pairset"
ABLATION_NAME = "w/o Pair-Set Critic"

SCRIPTS = [
    "4manager_low_midf_no_pairset.py",
    "4manager_mid_midf_no_pairset.py",
    "4manager_high_midf_no_pairset.py",
    "10manager_low_midf_no_pairset.py",
    "10manager_mid_midf_no_pairset.py",
    "10manager_high_midf_no_pairset.py",
]


def run_experiment(script_name: str, scripts_dir: str, mini_log: bool = False) -> dict:
    script_path = os.path.join(scripts_dir, script_name)
    
    if not os.path.exists(script_path):
        return {'script': script_name, 'status': 'NOT_FOUND', 'duration': 0}
    
    cmd = [sys.executable, script_path]
    if mini_log:
        cmd.append("--mini_log")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=str(Path(scripts_dir).parent.parent.parent))
        duration = time.time() - start_time
        return {
            'script': script_name,
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
            'duration': duration,
            'returncode': result.returncode,
        }
    except Exception as e:
        return {'script': script_name, 'status': 'ERROR', 'duration': time.time() - start_time, 'error': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f'Run {ABLATION_NAME} Ablation')
    parser.add_argument('--mini_log', action='store_true')
    args = parser.parse_args()
    
    scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    
    print("=" * 80)
    print(f"ABLATION: {ABLATION_NAME}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts: {len(SCRIPTS)}")
    print("=" * 80)
    
    results = []
    total_start = time.time()
    
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] {script}")
        result = run_experiment(script, scripts_dir, args.mini_log)
        results.append(result)
        
        status = "✓" if result['status'] == 'SUCCESS' else "✗"
        duration = f"{result['duration']/60:.1f}min"
        print(f"     {status} {result['status']} ({duration})")
    
    total_duration = time.time() - total_start
    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    print("\n" + "=" * 80)
    print(f"COMPLETED: {success}/{len(SCRIPTS)} succeeded")
    print(f"Total: {total_duration/60:.1f} minutes")
    print("=" * 80)
    
    return 0 if success == len(SCRIPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
