"""
Churn configuration constants for Multidrone experiments.
"""

N_CONTROLLERS = 4
N_INIT_PER = 7
N_MAX_PER = 10
N_WAYPOINTS = 50
MAX_DURATION_SECONDS = 30.0
AGENT_HZ = 40
MIN_ACTIVE_PER = 3

# Churn timing: interval-based (every CHURN_INTERVAL_SEC seconds)
CHURN_INTERVAL_SEC = 1.25  # 50 steps at 40Hz → ~23 churn events per 30s episode

# Coverage area
COVERAGE_X = (0.0, 30.0)
COVERAGE_Y = (0.0, 30.0)
CRUISE_ALTITUDE = 1.5
FLIGHT_DOME_SIZE = 45.0

# Observation dims (3D)
X_DIM = 9   # [pos3D, vel3D, target_rel3D]
G_DIM = 17  # [public(3) + private(4) + cross_ctrl(3*(M-1))=10]
P_DIM = 4   # [vx, vy, vr, vz] flight mode 6

CHURN_CONFIGS = {
    'low':  {'rho_min': 0.10, 'rho_max': 0.15},
    'mid':  {'rho_min': 0.20, 'rho_max': 0.25},
    'high': {'rho_min': 0.30, 'rho_max': 0.35},
}

REWARD_WEIGHTS = {
    'global_coverage': 0.4,
    'local_efficiency': 0.3,
    'collision_penalty': 0.2,
    'energy_penalty': 0.1,
}

SEEDS = [42]
EVAL_EPISODES = 10
TRAIN_EPISODES = 500  # Drone episodes are long (1200 steps each)

ALL_ALGORITHMS = ['ea', 'maddpg', 'matd3', 'mappo', 'maac', 'sqddpg', 'agile']
SET_VARIANTS = ['maddpg_set', 'mappo_set', 'maac_set']
ABLATION_VARIANTS = ['ea_no_pairset', 'ea_no_tdconsistent', 'ea_no_credit']
ALL_METHODS = ALL_ALGORITHMS + SET_VARIANTS + ABLATION_VARIANTS
