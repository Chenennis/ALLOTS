#!/bin/bash
#
# Run Enhanced EA Tests Only (6 total)
# 
# Mid-Episode Churn: churn at steps [6, 12, 18]
#

cd "$(dirname "$0")"

echo "=========================================="
echo "Enhanced EA Tests - Mid-Episode Churn"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

mkdir -p results

# Run all 6 EA tests in parallel
for env in 4manager 10manager; do
    for churn in low mid high; do
        echo "Starting: EA ${env} ${churn}"
        nohup python scripts/train_ea.py --env "$env" --churn "$churn" \
            --results_dir "results/ea_${env}_${churn}" > "results/ea_${env}_${churn}.log" 2>&1 &
    done
done

echo ""
echo "All 6 EA tests started in background!"
echo ""
echo "Monitor with: tail -f results/ea_*.log"
echo "Check progress: ps aux | grep train_ea"
