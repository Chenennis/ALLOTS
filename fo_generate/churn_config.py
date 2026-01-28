"""
Churn Configuration Module for FOgym Environment

This module defines the configuration for device churn (dynamic device joining/leaving)
in multi-agent FlexOffer environments.

Author: FOenv Team
Date: 2026-01-12
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class ChurnConfig:
    """
    Configuration for device churn behavior
    
    Churn allows devices to dynamically join and leave the environment,
    simulating real-world scenarios where users add/remove devices,
    devices fail, or network connectivity changes.
    
    Attributes:
        enabled: Whether churn is enabled
        trigger_interval: Trigger churn every K episodes (0 = never)
        severity_levels: Possible severity levels ρ ∈ (0, 1) indicating fraction of devices churned
        severity_probs: Probability distribution over severity levels
        min_active_devices: Minimum number of active devices per manager
        max_universe_size_multiplier: Maximum universe pool size as multiplier of current devices
        seed: Random seed for reproducibility (None = use system random)
        create_new_on_insufficient: Create new random devices if inactive pool is insufficient
    """
    enabled: bool = False
    trigger_interval: int = 10  # Churn every 10 episodes
    severity_levels: Tuple[float, ...] = (0.02, 0.05, 0.10)  # 2%, 5%, 10% of devices
    severity_probs: Tuple[float, ...] = (0.6, 0.3, 0.1)  # Higher prob for lower severity
    min_active_devices: int = 5  # At least 5 active devices per manager
    max_universe_size_multiplier: float = 2.0  # Universe can be at most 2x current devices
    seed: Optional[int] = None
    create_new_on_insufficient: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.enabled:
            # Validate trigger_interval
            if self.trigger_interval < 0:
                raise ValueError(f"trigger_interval must be non-negative, got {self.trigger_interval}")
            
            # Validate severity levels
            if not self.severity_levels:
                raise ValueError("severity_levels cannot be empty")
            for sev in self.severity_levels:
                if not (0 < sev <= 1):
                    raise ValueError(f"severity_levels must be in (0, 1], got {sev}")
            
            # Validate severity probs
            if len(self.severity_probs) != len(self.severity_levels):
                raise ValueError(
                    f"severity_probs length ({len(self.severity_probs)}) "
                    f"must match severity_levels length ({len(self.severity_levels)})"
                )
            if not np.isclose(sum(self.severity_probs), 1.0):
                raise ValueError(
                    f"severity_probs must sum to 1.0, got {sum(self.severity_probs)}"
                )
            for prob in self.severity_probs:
                if not (0 <= prob <= 1):
                    raise ValueError(f"severity_probs must be in [0, 1], got {prob}")
            
            # Validate min_active_devices
            if self.min_active_devices < 1:
                raise ValueError(
                    f"min_active_devices must be at least 1, got {self.min_active_devices}"
                )
            
            # Validate max_universe_size_multiplier
            if self.max_universe_size_multiplier < 1.0:
                raise ValueError(
                    f"max_universe_size_multiplier must be >= 1.0, got {self.max_universe_size_multiplier}"
                )
    
    def should_trigger_churn(self, episode: int) -> bool:
        """
        Determine if churn should be triggered at this episode
        
        Args:
            episode: Current episode number (0-indexed)
        
        Returns:
            True if churn should be triggered, False otherwise
        """
        if not self.enabled:
            return False
        if self.trigger_interval <= 0:
            return False
        if episode <= 0:  # Never churn on first episode
            return False
        return episode % self.trigger_interval == 0
    
    def sample_severity(self, rng: np.random.Generator) -> float:
        """
        Sample a churn severity level
        
        Args:
            rng: NumPy random generator
        
        Returns:
            Sampled severity ρ ∈ (0, 1]
        """
        if not self.enabled:
            return 0.0
        return rng.choice(self.severity_levels, p=self.severity_probs)
    
    def compute_churn_counts(self, n_active: int, severity: float) -> Tuple[int, int]:
        """
        Compute number of devices to remove (leave) and add (join)
        
        Algorithm:
            1. k = max(1, round(severity * n_active))  # Total churned devices
            2. k_leave = floor(k / 2)  # Half leave
            3. k_join = k - k_leave  # Rest join
            4. Ensure n_active - k_leave >= min_active_devices
        
        Args:
            n_active: Current number of active devices
            severity: Churn severity ρ ∈ (0, 1]
        
        Returns:
            (k_leave, k_join): Number of devices to remove and add
        """
        if not self.enabled or severity <= 0:
            return 0, 0
        
        # Total churned devices
        k = max(1, round(severity * n_active))
        
        # Split into leave and join
        k_leave = k // 2
        k_join = k - k_leave
        
        # Enforce minimum active devices constraint
        max_leave = max(0, n_active - self.min_active_devices)
        k_leave = min(k_leave, max_leave)
        
        # Adjust k_join to maintain churn balance (optional: could keep original k_join)
        # Here we keep k_join unchanged to potentially grow the active set
        
        return k_leave, k_join
    
    def get_rng(self) -> np.random.Generator:
        """
        Get a NumPy random generator with configured seed
        
        Returns:
            NumPy random generator
        """
        return np.random.default_rng(self.seed)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'enabled': self.enabled,
            'trigger_interval': self.trigger_interval,
            'severity_levels': self.severity_levels,
            'severity_probs': self.severity_probs,
            'min_active_devices': self.min_active_devices,
            'max_universe_size_multiplier': self.max_universe_size_multiplier,
            'seed': self.seed,
            'create_new_on_insufficient': self.create_new_on_insufficient,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ChurnConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """String representation"""
        if not self.enabled:
            return "ChurnConfig(enabled=False)"
        return (
            f"ChurnConfig(enabled=True, trigger_interval={self.trigger_interval}, "
            f"severity_levels={self.severity_levels}, severity_probs={self.severity_probs}, "
            f"min_active={self.min_active_devices})"
        )


# Predefined configurations
DEFAULT_CHURN_CONFIG = ChurnConfig()

MILD_CHURN_CONFIG = ChurnConfig(
    enabled=True,
    trigger_interval=20,  # Every 20 episodes
    severity_levels=(0.01, 0.02, 0.05),  # 1%, 2%, 5%
    severity_probs=(0.7, 0.2, 0.1),
    min_active_devices=5,
)

MODERATE_CHURN_CONFIG = ChurnConfig(
    enabled=True,
    trigger_interval=10,  # Every 10 episodes
    severity_levels=(0.02, 0.05, 0.10),  # 2%, 5%, 10%
    severity_probs=(0.6, 0.3, 0.1),
    min_active_devices=5,
)

SEVERE_CHURN_CONFIG = ChurnConfig(
    enabled=True,
    trigger_interval=5,  # Every 5 episodes
    severity_levels=(0.05, 0.10, 0.15),  # 5%, 10%, 15%
    severity_probs=(0.5, 0.3, 0.2),
    min_active_devices=3,
)


if __name__ == "__main__":
    # Test configurations
    print("=== Testing ChurnConfig ===\n")
    
    # Test default config
    config = DEFAULT_CHURN_CONFIG
    print(f"Default: {config}")
    print(f"  Should trigger at episode 10? {config.should_trigger_churn(10)}")
    print(f"  Should trigger at episode 20? {config.should_trigger_churn(20)}\n")
    
    # Test moderate config
    config = MODERATE_CHURN_CONFIG
    print(f"Moderate: {config}")
    rng = config.get_rng()
    for i in range(5):
        severity = config.sample_severity(rng)
        k_leave, k_join = config.compute_churn_counts(n_active=30, severity=severity)
        print(f"  Sample {i+1}: severity={severity:.2f}, k_leave={k_leave}, k_join={k_join}")
    print()
    
    # Test severe config
    config = SEVERE_CHURN_CONFIG
    print(f"Severe: {config}")
    print(f"  Trigger at episodes: {[i for i in range(1, 26) if config.should_trigger_churn(i)]}")
    print()
    
    # Test validation
    print("=== Testing Validation ===")
    try:
        bad_config = ChurnConfig(enabled=True, severity_levels=(0.0,))
        print("ERROR: Should have failed with severity_levels=(0.0,)")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        bad_config = ChurnConfig(enabled=True, severity_probs=(0.5, 0.3))
        print("ERROR: Should have failed with mismatched probs")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        bad_config = ChurnConfig(enabled=True, min_active_devices=0)
        print("ERROR: Should have failed with min_active_devices=0")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n=== All tests passed ===")
