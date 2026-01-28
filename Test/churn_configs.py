"""
Churn Configuration Presets for Testing EA vs Baselines

These configurations are designed to test the robustness of algorithms
under different levels of environment instability.

Author: FOenv Team
Date: 2026-01-18
"""

# ============== Current (Moderate) Churn ==============
CHURN_MODERATE = {
    'name': 'moderate',
    'description': '当前设置：温和的churn，每10ep触发，20-25%变化',
    'severity_levels': [0.20, 0.225, 0.25],
    'severity_probs': [0.4, 0.3, 0.3],
    'trigger_interval': 10,
    'min_active_devices': 5,
}

# ============== High Churn (Recommended for EA testing) ==============
CHURN_HIGH = {
    'name': 'high',
    'description': '高强度churn：每5ep触发，30-50%变化',
    'severity_levels': [0.30, 0.40, 0.50],
    'severity_probs': [0.3, 0.4, 0.3],
    'trigger_interval': 5,  # 更频繁
    'min_active_devices': 5,
}

# ============== Extreme Churn (Stress Test) ==============
CHURN_EXTREME = {
    'name': 'extreme',
    'description': '极端churn：每2ep触发，40-60%变化',
    'severity_levels': [0.40, 0.50, 0.60],
    'severity_probs': [0.3, 0.4, 0.3],
    'trigger_interval': 2,  # 非常频繁
    'min_active_devices': 3,
}

# ============== Continuous Churn (Every Episode) ==============
CHURN_CONTINUOUS = {
    'name': 'continuous',
    'description': '连续churn：每个ep都触发，15-30%变化',
    'severity_levels': [0.15, 0.20, 0.30],
    'severity_probs': [0.4, 0.4, 0.2],
    'trigger_interval': 1,  # 每个episode
    'min_active_devices': 5,
}

# ============== Burst Churn (Occasional but Severe) ==============
CHURN_BURST = {
    'name': 'burst',
    'description': '突发churn：每20ep触发，但变化剧烈50-70%',
    'severity_levels': [0.50, 0.60, 0.70],
    'severity_probs': [0.3, 0.4, 0.3],
    'trigger_interval': 20,
    'min_active_devices': 3,
}


def get_churn_config(preset_name: str) -> dict:
    """Get churn configuration by preset name"""
    presets = {
        'moderate': CHURN_MODERATE,
        'high': CHURN_HIGH,
        'extreme': CHURN_EXTREME,
        'continuous': CHURN_CONTINUOUS,
        'burst': CHURN_BURST,
    }
    return presets.get(preset_name, CHURN_MODERATE)


def print_all_configs():
    """Print all available churn configurations"""
    print("=" * 70)
    print("Available Churn Configurations")
    print("=" * 70)
    
    for name, config in [
        ('moderate', CHURN_MODERATE),
        ('high', CHURN_HIGH),
        ('extreme', CHURN_EXTREME),
        ('continuous', CHURN_CONTINUOUS),
        ('burst', CHURN_BURST),
    ]:
        print(f"\n📊 {name.upper()}")
        print(f"   Description: {config['description']}")
        print(f"   Severity: {config['severity_levels']}")
        print(f"   Interval: every {config['trigger_interval']} episodes")


if __name__ == "__main__":
    print_all_configs()
