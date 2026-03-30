"""Batch runner for Multidrone experiments. Same structure as run_all_mpe.py."""

import subprocess, sys, os, time, argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from Multidrone.drone_churn.churn_config import ALL_METHODS, SEEDS, TRAIN_EPISODES


def run_single(algo, churn, seed, episodes, cpu_flag):
    train_script = os.path.join(project_root, "Multidrone", "drone_churn", "train_drone.py")
    args = f"--algo {algo} --churn {churn} --seed {seed} --episodes {episodes}"
    if cpu_flag: args += " --cpu"
    # Activate conda env FO so pybullet/PyFlyt are available
    shell_cmd = f"source activate FO 2>/dev/null && python {train_script} {args}"
    start = time.time()
    result = subprocess.run(shell_cmd, shell=True, executable="/bin/bash",
                            capture_output=True, text=True, cwd=project_root)
    return {'algo': algo, 'churn': churn, 'seed': seed,
            'status': "OK" if result.returncode == 0 else "FAILED",
            'time': time.time()-start,
            'stderr': result.stderr[-200:] if result.returncode != 0 else ''}


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
    print(f"Total: {len(combos)} experiments, parallel={args.parallel}")

    results = []; start_all = time.time()
    if args.parallel <= 1:
        for algo, churn, seed in combos:
            print(f"Running {algo}/{churn}/s{seed}...")
            r = run_single(algo, churn, seed, args.episodes, args.cpu)
            results.append(r); print(f"  {r['status']} ({r['time']:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            fs = {ex.submit(run_single, a, c, s, args.episodes, args.cpu): (a,c,s) for a,c,s in combos}
            for f in as_completed(fs):
                a,c,s = fs[f]; r = f.result(); results.append(r)
                print(f"  {a}/{c}/s{s}: {r['status']} ({r['time']:.0f}s)")

    ok = sum(1 for r in results if r['status']=='OK')
    print(f"\n{'='*50}\nCompleted: {ok}/{len(results)}, total: {time.time()-start_all:.0f}s")
    for r in results:
        if r['status']=='FAILED': print(f"  FAILED: {r['algo']}/{r['churn']}/s{r['seed']}: {r['stderr']}")


if __name__ == '__main__':
    main()
