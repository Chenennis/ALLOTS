#!/usr/bin/env python
"""
Run All SALSA/EA Ablation Experiments

This script runs all 18 ablation experiments sequentially or in parallel.

Usage:
    # Run all sequentially
    python Test/Ablation/run_all_ablations.py
    
    # Run with mini logging
    python Test/Ablation/run_all_ablations.py --mini_log
    
    # Run specific experiment
    python Test/Ablation/run_all_ablations.py --filter 4manager_mid

Author: FOenv Team
Date: 2026-01-20
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import time


# All ablation experiment scripts
ABLATION_SCRIPTS = [
    # 4manager environment
    "4manager_low_midf_no_pairset.py",
    "4manager_mid_midf_no_pairset.py",
    "4manager_high_midf_no_pairset.py",
    "4manager_low_midf_no_tdconsistent.py",
    "4manager_mid_midf_no_tdconsistent.py",
    "4manager_high_midf_no_tdconsistent.py",
    "4manager_low_midf_no_credit.py",
    "4manager_mid_midf_no_credit.py",
    "4manager_high_midf_no_credit.py",
    # 10manager environment
    "10manager_low_midf_no_pairset.py",
    "10manager_mid_midf_no_pairset.py",
    "10manager_high_midf_no_pairset.py",
    "10manager_low_midf_no_tdconsistent.py",
    "10manager_mid_midf_no_tdconsistent.py",
    "10manager_high_midf_no_tdconsistent.py",
    "10manager_low_midf_no_credit.py",
    "10manager_mid_midf_no_credit.py",
    "10manager_high_midf_no_credit.py",
]


def run_experiment(script_name: str, scripts_dir: str, mini_log: bool = False) -> dict:
    """
    Run a single ablation experiment.
    
    Args:
        script_name: Name of the script to run
        scripts_dir: Directory containing scripts
        mini_log: Use minimal logging
    
    Returns:
        Dict with status and timing info
    """
    script_path = os.path.join(scripts_dir, script_name)
    
    if not os.path.exists(script_path):
        return {
            'script': script_name,
            'status': 'NOT_FOUND',
            'duration': 0,
            'error': f"Script not found: {script_path}"
        }
    
    cmd = [sys.executable, script_path]
    if mini_log:
        cmd.append("--mini_log")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not mini_log,
            text=True,
            cwd=str(Path(scripts_dir).parent.parent.parent)  # Project root
        )
        
        duration = time.time() - start_time
        
        return {
            'script': script_name,
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
            'duration': duration,
            'returncode': result.returncode,
            'error': result.stderr if result.returncode != 0 else None
        }
        
    except Exception as e:
        return {
            'script': script_name,
            'status': 'ERROR',
            'duration': time.time() - start_time,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run All Ablation Experiments')
    parser.add_argument('--mini_log', action='store_true',
                       help='Use minimal logging output')
    parser.add_argument('--filter', type=str, default=None,
                       help='Filter scripts by name (e.g., "4manager_mid")')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print scripts that would be run without executing')
    
    args = parser.parse_args()
    
    # Get scripts directory
    scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    
    # Filter scripts if needed
    scripts_to_run = ABLATION_SCRIPTS
    if args.filter:
        scripts_to_run = [s for s in scripts_to_run if args.filter in s]
    
    print("=" * 80)
    print("SALSA/EA ABLATION EXPERIMENT RUNNER")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\nTotal experiments: {len(scripts_to_run)}")
    print("\nScripts to run:")
    for i, script in enumerate(scripts_to_run, 1):
        print(f"  {i:2d}. {script}")
    
    if args.dry_run:
        print("\n[DRY RUN] No experiments executed.")
        return
    
    print("\n" + "-" * 80)
    print("Starting experiments...")
    print("-" * 80)
    
    results = []
    total_start = time.time()
    
    for i, script in enumerate(scripts_to_run, 1):
        print(f"\n[{i}/{len(scripts_to_run)}] Running: {script}")
        
        result = run_experiment(script, scripts_dir, args.mini_log)
        results.append(result)
        
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        duration_str = f"{result['duration']:.1f}s" if result['duration'] < 60 else f"{result['duration']/60:.1f}min"
        print(f"     {status_symbol} {result['status']} ({duration_str})")
        
        if result.get('error'):
            print(f"     Error: {result['error'][:200]}...")
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR'])
    not_found_count = sum(1 for r in results if r['status'] == 'NOT_FOUND')
    
    print(f"\nTotal: {len(results)} experiments")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ? Not Found: {not_found_count}")
    print(f"\nTotal Duration: {total_duration/60:.1f} minutes")
    
    if failed_count > 0:
        print("\nFailed experiments:")
        for r in results:
            if r['status'] in ['FAILED', 'ERROR']:
                print(f"  - {r['script']}: {r.get('error', 'Unknown error')[:100]}")
    
    print("\n" + "=" * 80)
    
    # Return exit code based on results
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
