#!/usr/bin/env python3
"""
快速测试脚本 - 验证测试运行器功能
Quick Test - Verify test runner functionality
"""
import subprocess
import sys

def run_command(cmd):
    """运行命令并显示输出"""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print('='*80)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

def main():
    """测试主函数"""
    print("\n" + "="*80)
    print("FlexOffer测试运行器 - 快速验证")
    print("="*80)
    
    # 测试1: 显示帮助
    print("\n[测试1] 显示帮助信息...")
    if not run_command([sys.executable, "Test/run_test.py", "--help"]):
        print("ERROR: 帮助信息显示失败")
        return False
    
    print("\n[OK] 帮助信息显示成功")
    
    # 告知用户如何运行实际测试
    print("\n" + "="*80)
    print("快速验证完成！")
    print("="*80)
    print("\n要运行实际测试，使用以下命令：")
    print("\n1. 4Manager + MAPPO (快速测试, 2轮):")
    print("   python Test/run_test.py --env 4manager --algo mappo --episodes 2")
    print("\n2. 10Manager + MADDPG + DP:")
    print("   python Test/run_test.py --env 10manager --algo maddpg --aggregation DP --episodes 2")
    print("\n3. 测试所有算法:")
    print("   # MAPPO")
    print("   python Test/run_test.py --env 4manager --algo mappo --episodes 2")
    print("   # MAIPPO")
    print("   python Test/run_test.py --env 4manager --algo maippo --episodes 2")
    print("   # MADDPG")
    print("   python Test/run_test.py --env 4manager --algo maddpg --episodes 2")
    print("   # MATD3")
    print("   python Test/run_test.py --env 4manager --algo matd3 --episodes 2")
    print("   # SQDDPG")
    print("   python Test/run_test.py --env 4manager --algo sqddpg --episodes 2")
    print("\n" + "="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
