#!/bin/bash
# TD-Consistent模块对比测试 - 并行运行
# Full EA (方案D) vs w/o TD-Consistent

echo "=========================================="
echo "TD-Consistent模块消融对比测试"
echo "开始时间: $(date)"
echo "=========================================="
echo ""
echo "并行运行:"
echo "  1. Full EA (方案D: 智能TD目标)"
echo "  2. w/o TD-Consistent (使用当前mask)"
echo ""

# Navigate to project root (adjust if needed)
cd "$(dirname "$0")/../.."

# 创建日志目录
mkdir -p Test/Ablation/results/td_comparison

# 备份旧结果
if [ -f "Test/Ablation/results/log_full_ea.txt" ]; then
    mv Test/Ablation/results/log_full_ea.txt Test/Ablation/results/td_comparison/log_full_ea_backup_$(date +%Y%m%d_%H%M%S).txt
fi
if [ -f "Test/Ablation/results/log_no_tdconsistent.txt" ]; then
    mv Test/Ablation/results/log_no_tdconsistent.txt Test/Ablation/results/td_comparison/log_no_tdconsistent_backup_$(date +%Y%m%d_%H%M%S).txt
fi

# 并行运行两个测试
echo "启动 Full EA 测试..."
python Test/run_all_full_ea.py --gpu --seed 42 > Test/Ablation/results/log_full_ea.txt 2>&1 &
PID_FULL=$!

echo "启动 w/o TD-Consistent 测试..."
python Test/Ablation/run_ablation_no_tdconsistent.py --gpu --seed 42 > Test/Ablation/results/log_no_tdconsistent.txt 2>&1 &
PID_ABLATION=$!

echo ""
echo "进程ID:"
echo "  Full EA: $PID_FULL"
echo "  w/o TD-Consistent: $PID_ABLATION"
echo ""
echo "等待完成... (预计2-4小时)"
echo ""

# 等待两个进程完成
wait $PID_FULL
STATUS_FULL=$?

wait $PID_ABLATION
STATUS_ABLATION=$?

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo "Full EA 状态: $([ $STATUS_FULL -eq 0 ] && echo '✓ 成功' || echo '✗ 失败')"
echo "w/o TD-Consistent 状态: $([ $STATUS_ABLATION -eq 0 ] && echo '✓ 成功' || echo '✗ 失败')"
echo ""
echo "日志文件:"
echo "  Full EA: Test/Ablation/results/log_full_ea.txt"
echo "  w/o TD-Consistent: Test/Ablation/results/log_no_tdconsistent.txt"
echo ""
echo "完成时间: $(date)"
