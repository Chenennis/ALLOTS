#!/usr/bin/env python3
"""
FlexOffer测试运行器 - Test Runner for FlexOffer System

支持选择不同的测试环境和算法组合进行测试
Supports selecting different test environments and algorithm combinations

使用方法 Usage:
    python Test/run_test.py --env 4manager --algo mappo
    python Test/run_test.py --env 10manager --algo maddpg --aggregation DP
    python Test/run_test.py --config Test/configs/test_config.yaml
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # YAML模块未安装，配置文件功能不可用

import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入FOPipeline
from run_fo_pipeline import FOPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestRunConfig:
    """测试运行配置"""
    # 环境配置
    env_type: str = "4manager"  # "4manager" or "10manager"
    env_data_dir: str = "Test/data"
    
    # 算法配置
    algorithm: str = "mappo"  # mappo, maippo, maddpg, matd3, sqddpg
    
    # 模块配置
    aggregation_method: str = "LP"  # LP or DP
    trading_strategy: str = "market_clearing"
    disaggregation_method: str = "proportional"  # proportional or average
    
    # 训练配置
    num_episodes: int = 10
    max_steps_per_episode: int = 24
    
    # 输出配置
    output_dir: str = "Test/results"
    save_interval: int = 5
    verbose: bool = True
    use_gpu: bool = False
    
    def to_pipeline_config(self) -> Dict[str, Any]:
        """转换为FOPipeline所需的配置格式"""
        # 算法名称映射（Test文件夹使用简化名称，实际调用使用完整名称）
        algo_mapping = {
            "mappo": "fomappo",
            "maippo": "fomaippo",
            "maddpg": "fomaddpg",
            "matd3": "fomatd3",
            "sqddpg": "fosqddpg"
        }
        
        config = {
            # 基础配置
            "rl_algorithm": algo_mapping.get(self.algorithm, self.algorithm),
            "num_episodes": self.num_episodes,
            "use_gpu": self.use_gpu,
            
            # 环境配置
            "num_managers": 4 if self.env_type == "4manager" else 10,
            "num_users": 36 if self.env_type == "4manager" else 90,
            
            # 模块配置
            "aggregation_method": self.aggregation_method,
            "trading_strategy": self.trading_strategy,
            "disaggregation_method": self.disaggregation_method,
            
            # 数据路径（如果使用自定义CSV）
            "custom_data_dir": self.env_data_dir if os.path.exists(self.env_data_dir) else None,
            
            # 输出配置
            "output_dir": self.output_dir,
            "save_interval": self.save_interval,
            
            # 其他默认配置
            "time_horizon": 24,
            "time_step": 1.0,
            "enable_visualization": False,
            "save_training_data": True,
        }
        
        return config


class TestRunner:
    """测试运行器 - 调用现有算法接口"""
    
    def __init__(self, config: TestRunConfig):
        self.config = config
        self.pipeline = None
        
        logger.info("="*80)
        logger.info("FlexOffer测试运行器初始化")
        logger.info(f"环境: {config.env_type}")
        logger.info(f"算法: {config.algorithm}")
        logger.info(f"Aggregation: {config.aggregation_method}")
        logger.info(f"Trading: {config.trading_strategy}")
        logger.info(f"Disaggregation: {config.disaggregation_method}")
        logger.info("="*80)
        
    def setup(self):
        """初始化FOPipeline"""
        logger.info("\n🏗️ 初始化FOPipeline...")
        
        # 转换为Pipeline配置
        pipeline_config = self.config.to_pipeline_config()
        
        # 创建Pipeline
        try:
            self.pipeline = FOPipeline(pipeline_config)
            logger.info("✅ FOPipeline初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ FOPipeline初始化失败: {e}")
            return False
    
    def run_training(self):
        """执行训练 - 调用现有算法接口"""
        if self.pipeline is None:
            logger.error("❌ Pipeline未初始化，请先调用setup()")
            return False
        
        logger.info("\n🚀 开始训练...")
        logger.info(f"算法: {self.config.algorithm.upper()}")
        logger.info(f"训练轮数: {self.config.num_episodes}")
        
        try:
            # 调用FOPipeline的训练方法
            # 这会自动根据配置中的rl_algorithm调用对应的算法
            self.pipeline.train_rl_agents()
            
            logger.info("✅ 训练完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """完整运行流程"""
        print("\n" + "="*80)
        print(f"🧪 FlexOffer测试运行")
        print(f"📊 环境: {self.config.env_type}")
        print(f"🤖 算法: {self.config.algorithm.upper()}")
        print(f"📈 模块配置:")
        print(f"   - Aggregation: {self.config.aggregation_method}")
        print(f"   - Trading: {self.config.trading_strategy}")
        print(f"   - Disaggregation: {self.config.disaggregation_method}")
        print("="*80 + "\n")
        
        # 初始化
        if not self.setup():
            print("\n❌ 初始化失败")
            return False
        
        # 训练
        if not self.run_training():
            print("\n❌ 训练失败")
            return False
        
        print("\n" + "="*80)
        print("✅ 测试运行完成")
        print(f"📁 结果保存在: {self.config.output_dir}")
        print("="*80 + "\n")
        
        return True


def load_config_from_yaml(filepath: str) -> TestRunConfig:
    """从YAML文件加载配置"""
    if not YAML_AVAILABLE:
        raise ImportError("YAML模块未安装，无法加载配置文件。请安装: pip install pyyaml")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    config = TestRunConfig(
        env_type=data.get('environment', {}).get('type', '4manager'),
        algorithm=data.get('algorithm', {}).get('name', 'mappo'),
        aggregation_method=data.get('modules', {}).get('aggregation', 'LP'),
        trading_strategy=data.get('modules', {}).get('trading', 'market_clearing'),
        disaggregation_method=data.get('modules', {}).get('disaggregation', 'proportional'),
        num_episodes=data.get('training', {}).get('num_episodes', 10),
        output_dir=data.get('training', {}).get('output_dir', 'Test/results'),
        use_gpu=data.get('training', {}).get('use_gpu', False),
    )
    
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='FlexOffer测试运行器 - 调用现有算法进行测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 Examples:
  # 4Manager环境 + MAPPO算法
  python Test/run_test.py --env 4manager --algo mappo
  
  # 10Manager环境 + MADDPG算法 + DP聚合
  python Test/run_test.py --env 10manager --algo maddpg --aggregation DP
  
  # 使用配置文件
  python Test/run_test.py --config Test/configs/test_config.yaml
  
  # 完整配置
  python Test/run_test.py --env 4manager --algo matd3 \\
    --aggregation LP --trading market_clearing \\
    --disaggregation proportional --episodes 20 --gpu
        """
    )
    
    # 环境选择
    parser.add_argument('--env', '--environment', type=str, default='4manager',
                       choices=['4manager', '10manager', '4m', '10m'],
                       help='测试环境选择 (4manager或10manager)')
    
    # 算法选择
    parser.add_argument('--algo', '--algorithm', type=str, default='mappo',
                       choices=['mappo', 'maippo', 'maddpg', 'matd3', 'sqddpg'],
                       help='MARL算法选择 (mappo/maippo/maddpg/matd3/sqddpg)')
    
    # 模块配置
    parser.add_argument('--aggregation', type=str, default='LP',
                       choices=['LP', 'DP'],
                       help='Aggregation方法 (LP或DP)')
    
    parser.add_argument('--trading', type=str, default='market_clearing',
                       help='Trading策略')
    
    parser.add_argument('--disaggregation', type=str, default='proportional',
                       choices=['proportional', 'average'],
                       help='Disaggregation方法')
    
    # 训练配置
    parser.add_argument('--episodes', type=int, default=10,
                       help='训练轮数')
    
    parser.add_argument('--gpu', action='store_true',
                       help='使用GPU加速')
    
    # 输出配置
    parser.add_argument('--output', type=str, default='Test/results',
                       help='结果输出目录')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='从YAML配置文件加载配置')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        logger.info(f"从配置文件加载: {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        # 标准化环境名称
        env_type = args.env
        if env_type == '4m':
            env_type = '4manager'
        elif env_type == '10m':
            env_type = '10manager'
        
        config = TestRunConfig(
            env_type=env_type,
            algorithm=args.algo,
            aggregation_method=args.aggregation,
            trading_strategy=args.trading,
            disaggregation_method=args.disaggregation,
            num_episodes=args.episodes,
            output_dir=args.output,
            use_gpu=args.gpu,
        )
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 创建并运行测试
    runner = TestRunner(config)
    success = runner.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
