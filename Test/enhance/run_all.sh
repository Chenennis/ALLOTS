#!/bin/bash
#
# Run All Enhanced Comparative Study Experiments
# 
# Total: 36 experiments
# - 6 EA (2 envs × 3 churns)
# - 30 Baselines (5 algorithms × 2 envs × 3 churns)
#
# Mid-Episode Churn: churn at steps [6, 12, 18]
# Baselines use FULL compatibility layer (stable mapping)
#

cd "$(dirname "$0")/.."

SCRIPT_DIR="scripts"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Enhanced Comparative Study - All Tests"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

# ============== EA Tests ==============
echo "=== Running EA Tests (6 total) ==="
for env in 4manager 10manager; do
    for churn in low mid high; do
        echo "  Starting: EA ${env} ${churn}"
        python "$SCRIPT_DIR/train_ea.py" --env "$env" --churn "$churn" \
            --results_dir "$RESULTS_DIR/ea_${env}_${churn}" &
    done
done

echo "  Waiting for EA tests to complete..."
wait
echo "  EA tests completed!"
echo ""

# ============== Baseline Tests ==============
for algo in maddpg matd3 sqddpg mappo maippo; do
    echo "=== Running ${algo^^} Tests (6 total) ==="
    for env in 4manager 10manager; do
        for churn in low mid high; do
            echo "  Starting: ${algo^^} ${env} ${churn}"
            python "$SCRIPT_DIR/train_baseline.py" --algorithm "$algo" \
                --env "$env" --churn "$churn" \
                --results_dir "$RESULTS_DIR/${algo}_${env}_${churn}" &
        done
    done
    
    echo "  Waiting for ${algo^^} tests to complete..."
    wait
    echo "  ${algo^^} tests completed!"
    echo ""
done

echo "=========================================="
echo "All Tests Completed!"
echo "End Time: $(date)"
echo "=========================================="

# Generate summary
echo ""
echo "=== Results Summary ==="
for dir in "$RESULTS_DIR"/*/; do
    if [ -f "${dir}summary.json" ]; then
        name=$(basename "$dir")
        ssr=$(python -c "import json; print(json.load(open('${dir}summary.json'))['SSR'])" 2>/dev/null || echo "N/A")
        echo "  $name: SSR = $ssr"
    fi
done
