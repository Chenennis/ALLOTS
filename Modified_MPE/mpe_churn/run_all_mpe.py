"""
Batch runner for all MPE experiments.

Usage:
    python Modified_MPE/mpe_churn/run_all_mpe.py
    python Modified_MPE/mpe_churn/run_all_mpe.py --parallel 4
    python Modified_MPE/mpe_churn/run_all_mpe.py --algos ea maddpg --churns mid high
"""

import subprocess
import sys
import os
import time
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Modified_MPE.mpe_churn.churn_config import ALL_METHODS, SEEDS, TRAIN_EPISODES


def run_single(algo, churn, seed, episodes, cpu_flag):
    """Run a single experiment."""
    cmd = [
        sys.executable,
        os.path.join(project_root, "Modified_MPE", "mpe_churn", "train_mpe.py"),
        "--algo", algo,
        "--churn", churn,
        "--seed", str(seed),
        "--episodes", str(episodes),
    ]
    if cpu_flag:
        cmd.append("--cpu")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    return {
        'algo': algo, 'churn': churn, 'seed': seed,
        'status': status, 'time': elapsed,
        'stderr': result.stderr[-200:] if result.returncode != 0 else '',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', nargs='+', default=ALL_METHODS)
    parser.add_argument('--churns', nargs='+', default=['low', 'mid', 'high'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--episodes', type=int, default=TRAIN_EPISODES)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    combos = list(product(args.algos, args.churns, args.seeds))
    print(f"Total experiments: {len(combos)}")
    print(f"Algorithms: {args.algos}")
    print(f"Churns: {args.churns}")
    print(f"Seeds: {args.seeds}")
    print(f"Parallel: {args.parallel}")
    print()

    results = []
    start_all = time.time()

    if args.parallel <= 1:
        for algo, churn, seed in combos:
            print(f"Running {algo} / {churn} / seed={seed} ...")
            r = run_single(algo, churn, seed, args.episodes, args.cpu)
            results.append(r)
            print(f"  {r['status']} ({r['time']:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for algo, churn, seed in combos:
                f = executor.submit(run_single, algo, churn, seed, args.episodes, args.cpu)
                futures[f] = (algo, churn, seed)

            for f in as_completed(futures):
                algo, churn, seed = futures[f]
                r = f.result()
                results.append(r)
                print(f"  {algo}/{churn}/s{seed}: {r['status']} ({r['time']:.0f}s)")

    total_time = time.time() - start_all
    ok = sum(1 for r in results if r['status'] == 'OK')
    print(f"\n{'='*50}")
    print(f"Completed: {ok}/{len(results)} succeeded")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    failed = [r for r in results if r['status'] == 'FAILED']
    if failed:
        print(f"\nFailed experiments:")
        for r in failed:
            print(f"  {r['algo']}/{r['churn']}/s{r['seed']}: {r['stderr']}")


if __name__ == '__main__':
    main()
