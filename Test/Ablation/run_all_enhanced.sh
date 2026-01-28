#!/bin/bash
#
# 并行运行增强版EA实验
# 
# 增强特性:
# - 渐进式Credit: Episode 0-99=0, 100-299=ramp to 0.05, 300+=0.05
# - 增强Pair-Set: 3层MLP, 512 hidden dim
#
# Author: FOenv Team
# Date: 2026-01-22

# Navigate to project root (adjust if needed)
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Enhanced EA Ablation Experiments"
echo "=========================================="
echo ""
echo "Features:"
echo "  - Progressive Credit warmup"
echo "  - Enhanced Pair-Set Critic (3-layer, 512 hidden)"
echo ""

# 创建日志目录
mkdir -p Test/Ablation/results_enhanced

# 并行运行6个实验 (4manager和10manager各3个churn级别)
echo "Starting parallel experiments..."

# 4manager实验
python -u Test/Ablation/scripts_midchurn/train_enhanced.py --env 4manager --churn low > Test/Ablation/results_enhanced/log_4manager_low.txt 2>&1 &
python -u Test/Ablation/scripts_midchurn/train_enhanced.py --env 4manager --churn mid > Test/Ablation/results_enhanced/log_4manager_mid.txt 2>&1 &
python -u Test/Ablation/scripts_midchurn/train_enhanced.py --env 4manager --churn high > Test/Ablation/results_enhanced/log_4manager_high.txt 2>&1 &

# 10manager实验
python -u Test/Ablation/scripts_midchurn/train_enhanced.py --env 10manager --churn low > Test/Ablation/results_enhanced/log_10manager_low.txt 2>&1 &
python -u Test/Ablation/scripts_midchurn/train_enhanced.py --env 10manager --churn mid > Test/Ablation/results_enhanced/log_10manager_mid.txt 2>&1 &
python -u Test/Ablation/scripts_midchurn/train_enhanced.py --env 10manager --churn high > Test/Ablation/results_enhanced/log_10manager_high.txt 2>&1 &

echo ""
echo "All 6 experiments started in background!"
echo ""
echo "Monitor progress:"
echo "  tail -f Test/Ablation/results_enhanced/log_*.txt"
echo ""
echo "Check episode progress:"
echo "  for f in Test/Ablation/results_enhanced/log_*.txt; do echo \"=== \$(basename \$f) ===\"; grep 'Episode' \"\$f\" | tail -1; done"

wait
echo ""
echo "All experiments completed!"
