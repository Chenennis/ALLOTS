"""
Per-algorithm hyperparameter configurations for MPE experiments.

All algorithms use the same environment dimensions but different training parameters.
Dimensions are smaller than FOgym since the MPE problem is simpler.
"""

from Modified_MPE.mpe_churn.churn_config import (
    N_CONTROLLERS, N_MAX_PER, X_DIM, G_DIM, P_DIM, TRAIN_EPISODES,
)

# ============================================================
# Shared Environment Dimensions
# ============================================================

MPE_ENV_DIMS = {
    'x_dim': X_DIM,           # 6: per-entity obs dim
    'g_dim': G_DIM,           # 14: global context dim
    'p': P_DIM,               # 5: action dim per entity
    'N_max': N_MAX_PER,       # 20: max entities per controller
    'n_controllers': N_CONTROLLERS,  # 4
}

# Derived dimensions for padding-based baselines
# state_dim = N_max * x_dim + g_dim + N_max (obs + global + mask flattened)
PADDING_STATE_DIM = N_MAX_PER * X_DIM + G_DIM + N_MAX_PER  # 20*6+14+20 = 154
# action_dim = N_max * p
PADDING_ACTION_DIM = N_MAX_PER * P_DIM  # 20*5 = 100

# ============================================================
# EA (ALLOTS) Configuration
# ============================================================

EA_CONFIG = {
    **MPE_ENV_DIMS,
    'num_managers': N_CONTROLLERS,
    'emb_dim': 16,
    'token_dim': 64,
    'hidden_dim': 64,
    'gamma': 0.95,
    'tau': 0.005,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'policy_delay': 2,
    'noise_scale': 0.1,
    'noise_clip': 0.2,
    'advantage_tau': 1.0,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
    # MPE-specific credit warmup (faster and stronger than FOgym defaults)
    'credit_warmup_start': 30,
    'credit_warmup_end': 150,
    'credit_max_weight': 0.20,
}

# ============================================================
# MADDPG Configuration
# ============================================================

MADDPG_CONFIG = {
    'n_agents': N_CONTROLLERS,
    'state_dim': PADDING_STATE_DIM,
    'action_dim': PADDING_ACTION_DIM,
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'hidden_dim': 128,
    'gamma': 0.95,
    'tau': 0.01,
    'noise_scale': 0.1,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
}

# ============================================================
# MATD3 Configuration
# ============================================================

MATD3_CONFIG = {
    'n_agents': N_CONTROLLERS,
    'state_dim': PADDING_STATE_DIM,
    'action_dim': PADDING_ACTION_DIM,
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'hidden_dim': 128,
    'gamma': 0.95,
    'tau': 0.005,
    'noise_scale': 0.1,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
}

# ============================================================
# MAPPO Configuration
# ============================================================

MAPPO_CONFIG = {
    'n_agents': N_CONTROLLERS,
    'state_dim': PADDING_STATE_DIM,
    'action_dim': PADDING_ACTION_DIM,
    'lr': 3e-4,
    'hidden_dim': 128,
    'gamma': 0.95,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    'n_epochs': 10,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
}

# ============================================================
# MAIPPO Configuration (Independent PPO)
# ============================================================

MAIPPO_CONFIG = {
    **MAPPO_CONFIG,
    'shared_policy': False,
}

# ============================================================
# MAAC Configuration
# ============================================================

MAAC_CONFIG = {
    'n_agents': N_CONTROLLERS,
    'N_max': N_MAX_PER,
    'device_dim': X_DIM,
    'global_dim': G_DIM,
    'action_dim': P_DIM,
    'hidden_dim': 128,
    'attend_heads': 4,
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'gamma': 0.95,
    'tau': 0.005,
    'noise_scale': 0.1,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
}

# ============================================================
# SQDDPG Configuration
# ============================================================

SQDDPG_CONFIG = {
    'n_agents': N_CONTROLLERS,
    'state_dim': PADDING_STATE_DIM,
    'action_dim': PADDING_ACTION_DIM,
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'hidden_dim': 128,
    'gamma': 0.95,
    'tau': 0.005,
    'noise_scale': 0.1,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'sample_size': 5,
    'train_episodes': TRAIN_EPISODES,
}

# ============================================================
# AGILE Configuration
# ============================================================

AGILE_CONFIG = {
    'n_agents': N_CONTROLLERS,
    'N_max': N_MAX_PER,
    'device_dim': X_DIM,
    'global_dim': G_DIM,
    'action_dim': P_DIM,
    'hidden_dim': 128,
    'lr_actor': 3e-4,
    'lr_critic': 3e-4,
    'gamma': 0.95,
    'tau': 0.005,
    'noise_scale': 0.1,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
}

# ============================================================
# Set-Enhanced Baselines (use EA's SetToSetActor, but no ALLOTS innovations)
# ============================================================

# Shared set-based dimensions (same architecture as EA actor, same network size)
_SET_ACTOR_DIMS = {
    **MPE_ENV_DIMS,
    'num_managers': N_CONTROLLERS,
    'emb_dim': 16,
    'token_dim': 64,
    'hidden_dim': 64,
}

MADDPG_SET_CONFIG = {
    **_SET_ACTOR_DIMS,
    'gamma': 0.95,
    'tau': 0.005,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'noise_scale': 0.1,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
}

MAAC_SET_CONFIG = {
    **_SET_ACTOR_DIMS,
    'attend_heads': 4,
    'gamma': 0.95,
    'tau': 0.005,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'noise_scale': 0.1,
    'buffer_capacity': 100000,
    'batch_size': 256,
    'train_episodes': TRAIN_EPISODES,
}

MAPPO_SET_CONFIG = {
    **_SET_ACTOR_DIMS,
    'gamma': 0.95,
    'gae_lambda': 0.95,
    'clip_param': 0.2,
    'ppo_epochs': 10,
    'mini_batch_size': 64,
    'lr_actor': 3e-4,
    'lr_critic': 1e-3,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    'train_episodes': TRAIN_EPISODES,
}

# ============================================================
# Config Registry
# ============================================================

ALGO_CONFIGS = {
    'ea': EA_CONFIG,
    'maddpg': MADDPG_CONFIG,
    'matd3': MATD3_CONFIG,
    'mappo': MAPPO_CONFIG,
    'maippo': MAIPPO_CONFIG,
    'maac': MAAC_CONFIG,
    'sqddpg': SQDDPG_CONFIG,
    'agile': AGILE_CONFIG,
    'maddpg_set': MADDPG_SET_CONFIG,
    'maac_set': MAAC_SET_CONFIG,
    'mappo_set': MAPPO_SET_CONFIG,
}
