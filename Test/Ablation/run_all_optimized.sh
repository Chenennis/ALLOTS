#!/bin/bash

# 运行优化参数的全部消融实验 - 4个算法并行
# 
# 参数配置:
# - 4manager: original (blend=0.7, tau=2.0, weight=0.15)
# - 10manager: option_AB (blend=0.3, tau=5.0, weight=0.05)

# Navigate to project root (adjust if needed)
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "启动优化参数消融实验 - 4个算法并行"
echo "============================================================"
echo ""
echo "参数配置:"
echo "  4manager:  tau=2.0, weight=0.15, blend=0.7 (original)"
echo "  10manager: tau=5.0, weight=0.05, blend=0.3 (option_AB)"
echo ""
echo "实验数量: 4 agents × 6 configs = 24 experiments"
echo ""

# 创建结果目录
mkdir -p Test/Ablation/results_optimized

# 启动4个并行进程
echo "启动 Full EA..."
python -u Test/Ablation/run_optimized_ablation.py --agent full_ea --seed 42 \
    > Test/Ablation/results_optimized/log_full_ea.txt 2>&1 &
echo "  PID: $!"

echo "启动 w/o Pair-Set..."
python -u Test/Ablation/run_optimized_ablation.py --agent no_pairset --seed 42 \
    > Test/Ablation/results_optimized/log_no_pairset.txt 2>&1 &
echo "  PID: $!"

echo "启动 w/o TD-Consistent..."
python -u Test/Ablation/run_optimized_ablation.py --agent no_tdconsistent --seed 42 \
    > Test/Ablation/results_optimized/log_no_tdconsistent.txt 2>&1 &
echo "  PID: $!"

echo "启动 w/o Credit..."
python -u Test/Ablation/run_optimized_ablation.py --agent no_credit --seed 42 \
    > Test/Ablation/results_optimized/log_no_credit.txt 2>&1 &
echo "  PID: $!"

echo ""
echo "============================================================"
echo "所有进程已启动！"
echo "============================================================"
echo ""
echo "监控命令:"
echo "  查看进度: ps aux | grep run_optimized"
echo "  查看日志: tail -f Test/Ablation/results_optimized/log_*.txt"
echo "  GPU状态: nvidia-smi"
