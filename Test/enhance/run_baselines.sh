#!/bin/bash
#
# Run All Baseline Tests (30 total)
# 
# Mid-Episode Churn: churn at steps [6, 12, 18]
# Full Compatibility Layer (use_stable_mapping=True)
#

cd "$(dirname "$0")"

echo "=========================================="
echo "Baseline Tests - Mid-Episode Churn"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

mkdir -p results

# Run baseline tests - 6 at a time to avoid overloading GPU
for algo in maddpg matd3 sqddpg mappo maippo; do
    echo "=== Starting ${algo^^} tests (6 total) ==="
    
    for env in 4manager 10manager; do
        for churn in low mid high; do
            echo "  Starting: ${algo^^} ${env} ${churn}"
            nohup python scripts/train_baseline.py --algorithm "$algo" \
                --env "$env" --churn "$churn" \
                --results_dir "results/${algo}_${env}_${churn}" > "results/${algo}_${env}_${churn}.log" 2>&1 &
        done
    done
    
    echo "  Waiting for ${algo^^} to complete..."
    wait
    echo "  ${algo^^} completed!"
    echo ""
done

echo "=========================================="
echo "All Baseline Tests Completed!"
echo "End Time: $(date)"
echo "=========================================="
