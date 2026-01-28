"""
Churn Environment Builder for Test Suite

This module builds churn-enabled environments from YAML configurations.

Author: FOenv Team
Date: 2026-01-12
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
from fo_generate.churn_config import (
    ChurnConfig, MILD_CHURN_CONFIG, MODERATE_CHURN_CONFIG, SEVERE_CHURN_CONFIG
)
from fo_common.dec_pomdp_config import DecPOMDPConfig
import logging

logger = logging.getLogger(__name__)


class ChurnEnvBuilder:
    """
    Builder for creating churn-enabled FlexOffer environments
    
    Supports creating environments from YAML config files with churn parameters.
    """
    
    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> MultiAgentFlexOfferEnv:
        """
        Build environment from configuration dictionary
        
        Args:
            config: Configuration dictionary with keys:
                - environment: Environment settings (type, data_dir)
                - modules: Algorithm modules (aggregation, trading, disaggregation)
                - churn: Churn configuration (optional)
                - dec_pomdp: Dec-POMDP configuration (optional)
        
        Returns:
            Configured MultiAgentFlexOfferEnv
        """
        # Extract environment settings
        env_config = config.get('environment', {})
        env_type = env_config.get('type', '4manager')
        data_dir = env_config.get('data_dir', 'data')
        time_horizon = env_config.get('time_horizon', 24)
        time_step = env_config.get('time_step', 1.0)
        
        # Extract module settings
        modules_config = config.get('modules', {})
        aggregation = modules_config.get('aggregation', 'LP')
        trading = modules_config.get('trading', 'bidding')
        disaggregation = modules_config.get('disaggregation', 'proportional')
        
        # Build churn config
        churn_config = ChurnEnvBuilder._build_churn_config(config.get('churn', {}))
        
        # Build Dec-POMDP config
        dec_pomdp_config = ChurnEnvBuilder._build_dec_pomdp_config(
            config.get('dec_pomdp', {})
        )
        
        logger.info(f"Building {env_type} environment with churn={'enabled' if churn_config.enabled else 'disabled'}")
        
        # Create environment
        env = MultiAgentFlexOfferEnv(
            data_dir=data_dir,
            time_horizon=time_horizon,
            time_step=time_step,
            aggregation_method=aggregation,
            trading_method=trading,
            disaggregation_method=disaggregation,
            churn_config=churn_config,
            dec_pomdp_config=dec_pomdp_config,
        )
        
        return env
    
    @staticmethod
    def _build_churn_config(churn_dict: Dict[str, Any]) -> ChurnConfig:
        """
        Build ChurnConfig from dictionary
        
        Args:
            churn_dict: Churn configuration dictionary
        
        Returns:
            ChurnConfig object
        """
        if not churn_dict or not churn_dict.get('enabled', False):
            return ChurnConfig(enabled=False)
        
        # Check for preset
        preset = churn_dict.get('preset')
        if preset:
            preset_configs = {
                'mild': MILD_CHURN_CONFIG,
                'moderate': MODERATE_CHURN_CONFIG,
                'severe': SEVERE_CHURN_CONFIG,
            }
            base_config = preset_configs.get(preset.lower())
            if base_config:
                logger.info(f"Using churn preset: {preset}")
                # Allow overriding preset values
                config_dict = base_config.to_dict()
                config_dict.update(churn_dict)
                config_dict.pop('preset', None)  # Remove preset key
                return ChurnConfig.from_dict(config_dict)
        
        # Build from scratch
        return ChurnConfig(
            enabled=churn_dict.get('enabled', True),
            trigger_interval=churn_dict.get('trigger_interval', 10),
            severity_levels=tuple(churn_dict.get('severity_levels', [0.02, 0.05, 0.10])),
            severity_probs=tuple(churn_dict.get('severity_probs', [0.6, 0.3, 0.1])),
            min_active_devices=churn_dict.get('min_active_devices', 5),
            max_universe_size_multiplier=churn_dict.get('max_universe_size_multiplier', 2.0),
            seed=churn_dict.get('seed'),
            create_new_on_insufficient=churn_dict.get('create_new_on_insufficient', True),
        )
    
    @staticmethod
    def _build_dec_pomdp_config(dec_pomdp_dict: Dict[str, Any]) -> DecPOMDPConfig:
        """
        Build DecPOMDPConfig from dictionary
        
        Args:
            dec_pomdp_dict: Dec-POMDP configuration dictionary
        
        Returns:
            DecPOMDPConfig object
        """
        if not dec_pomdp_dict:
            return DecPOMDPConfig()
        
        return DecPOMDPConfig(
            enable_observation_noise=dec_pomdp_dict.get('enable_observation_noise', False),
            observation_noise_std=dec_pomdp_dict.get('observation_noise_std', 0.05),
            enable_information_delay=dec_pomdp_dict.get('enable_information_delay', False),
            delay_steps=dec_pomdp_dict.get('delay_steps', 1),
            enable_limited_collaboration=dec_pomdp_dict.get('enable_limited_collaboration', True),
            collaboration_info_limit=dec_pomdp_dict.get('collaboration_info_limit', 0.3),
        )
    
    @staticmethod
    def build_4manager_churn(
        churn_preset: str = "moderate",
        data_dir: str = "data",
        time_horizon: int = 24
    ) -> MultiAgentFlexOfferEnv:
        """
        Quick builder for 4-manager churn environment
        
        Args:
            churn_preset: Churn preset ('mild', 'moderate', 'severe')
            data_dir: Data directory
            time_horizon: Time horizon in hours
        
        Returns:
            Configured environment
        """
        config = {
            'environment': {
                'type': '4manager',
                'data_dir': data_dir,
                'time_horizon': time_horizon,
            },
            'modules': {
                'aggregation': 'LP',
                'trading': 'bidding',
                'disaggregation': 'proportional',
            },
            'churn': {
                'enabled': True,
                'preset': churn_preset,
            },
        }
        return ChurnEnvBuilder.build_from_config(config)
    
    @staticmethod
    def build_10manager_churn(
        churn_preset: str = "moderate",
        data_dir: str = "data",
        time_horizon: int = 24
    ) -> MultiAgentFlexOfferEnv:
        """
        Quick builder for 10-manager churn environment
        
        Args:
            churn_preset: Churn preset ('mild', 'moderate', 'severe')
            data_dir: Data directory
            time_horizon: Time horizon in hours
        
        Returns:
            Configured environment
        """
        config = {
            'environment': {
                'type': '10manager',
                'data_dir': data_dir,
                'time_horizon': time_horizon,
            },
            'modules': {
                'aggregation': 'LP',
                'trading': 'bidding',
                'disaggregation': 'proportional',
            },
            'churn': {
                'enabled': True,
                'preset': churn_preset,
            },
        }
        return ChurnEnvBuilder.build_from_config(config)


def test_builder():
    """Test the builder"""
    print("=== Testing ChurnEnvBuilder ===\n")
    
    # Test 4manager moderate
    print("Building 4-manager moderate churn environment...")
    env = ChurnEnvBuilder.build_4manager_churn(churn_preset="moderate", time_horizon=4)
    print(f"  Created with {len(env.manager_agents)} managers")
    print(f"  Churn enabled: {env.churn_config.enabled}")
    print(f"  Trigger interval: {env.churn_config.trigger_interval}")
    print()
    
    # Test from config dict
    print("Building from custom config dict...")
    config = {
        'environment': {
            'type': '4manager',
            'data_dir': 'data',
            'time_horizon': 4,
        },
        'modules': {
            'aggregation': 'LP',
            'trading': 'market_clearing',
            'disaggregation': 'proportional',
        },
        'churn': {
            'enabled': True,
            'trigger_interval': 5,
            'severity_levels': [0.05, 0.10],
            'severity_probs': [0.7, 0.3],
            'min_active_devices': 3,
        },
    }
    env2 = ChurnEnvBuilder.build_from_config(config)
    print(f"  Created with {len(env2.manager_agents)} managers")
    print(f"  Churn config: {env2.churn_config}")
    print()
    
    print("✅ Builder tests passed\n")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_builder()
