"""
Churn configuration constants for Modified MPE experiments.

Mirrors FOgym's churn configuration structure:
- 4 controllers (= 4 managers in FOgym)
- 3 churn intensity levels (low/mid/high)
- Mid-episode churn at fixed timesteps
"""

# ============================================================
# Environment Constants
# ============================================================

N_CONTROLLERS = 4       # Number of controllers (= managers in FOgym)
N_INIT_PER = 15         # Initial active particles per controller (60 total)
N_MAX_PER = 20          # Max slots per controller (80 total)
N_LANDMARKS = 60        # Shared landmark count
T = 25                  # Episode length (timesteps)
MIN_ACTIVE_PER = 5      # Minimum active particles per controller

# Physical space
WORLD_SIZE = 6.0        # World bounds: [-WORLD_SIZE, WORLD_SIZE]^2 (large enough that random coverage is poor)
AGENT_SIZE = 0.04       # Agent collision radius (slightly smaller)
LANDMARK_SIZE = 0.04    # Landmark visual size
SENSITIVITY = 8.0       # Action force multiplier (scaled up for larger world)

# Churn timing (mid-episode, mirrors FOgym's t=6,12,18)
CHURN_STEPS = [6, 12, 18]

# ============================================================
# Churn Intensity Configurations
# ============================================================

MPE_CHURN_CONFIGS = {
    'low': {
        'rho_min': 0.10,
        'rho_max': 0.15,
        'label': 'Low Churn',
        'description': '~0-1 particle change per controller per event',
    },
    'mid': {
        'rho_min': 0.20,
        'rho_max': 0.25,
        'label': 'Mid Churn',
        'description': '~1 particle change per controller per event',
    },
    'high': {
        'rho_min': 0.30,
        'rho_max': 0.35,
        'label': 'High Churn',
        'description': '~1-2 particles change per controller per event',
    },
}

# ============================================================
# Observation Dimensions
# ============================================================

X_DIM = 6              # Per-entity obs: [pos_x, pos_y, vel_x, vel_y, lm_rel_x, lm_rel_y]
G_DIM = 14             # Global context: [public(2) + private(3) + cross-ctrl(9)]
P_DIM = 5              # Per-entity action: [no_action, left, right, down, up]

# ============================================================
# Reward Weights
# ============================================================

REWARD_WEIGHTS = {
    'global_coverage': 0.5,      # Shared coverage quality
    'local_efficiency': 0.3,     # Per-controller fleet efficiency
    'collision_penalty': 0.2,    # Collision avoidance
}

# ============================================================
# Experiment Configuration
# ============================================================

SEEDS = [42]           # Start with 1 seed, add more later
EVAL_EPISODES = 10
TRAIN_EPISODES = 1000  # MPE episodes are short (25 steps each)

# ============================================================
# All Algorithms
# ============================================================

ALL_ALGORITHMS = [
    'ea', 'maddpg', 'matd3', 'mappo', 'maippo', 'maac', 'sqddpg', 'agile',
]

SET_ENHANCED_VARIANTS = [
    'maddpg_set', 'maac_set', 'mappo_set',
]

ABLATION_VARIANTS = [
    'ea_no_pairset', 'ea_no_tdconsistent', 'ea_no_credit',
]

ALL_METHODS = ALL_ALGORITHMS + SET_ENHANCED_VARIANTS + ABLATION_VARIANTS
