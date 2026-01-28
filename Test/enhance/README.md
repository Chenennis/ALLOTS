# Enhanced Comparative Study

## Overview

This folder contains the enhanced comparative study between the EA algorithm and 7 baseline methods.

## Key Features

### Enhanced EA Algorithm
- **Progressive Credit**: Warmup from episode 0-99 (weight=0) to 300+ (weight=0.05)
- **Enhanced Pair-Set Critic**: 3-layer MLP, 512 hidden dim
- **`set_episode()` method**: Enables progressive Credit warmup

### Baselines with Full Compatibility Layer
- **AGILE, MAAC, MADDPG, MATD3, SQDDPG, MAPPO, MAIPPO**
- **Stable Slot Mapping**: `use_stable_mapping=True` for positive rewards
- Uses full compatibility layer benefits

### Mid-Episode Churn
- **Churn triggered at steps [6, 12, 18]** within each episode (every 6 hours)
- Tests algorithm robustness during episode execution
- More realistic than only reset-time churn

## Experiment Matrix

| Algorithm | Environment | Churn Levels | Total Tests |
|-----------|-------------|--------------|-------------|
| EA | 4manager, 10manager | low, mid, high | 6 |
| AGILE | 4manager, 10manager | low, mid, high | 6 |
| MAAC | 4manager, 10manager | low, mid, high | 6 |
| MADDPG | 4manager, 10manager | low, mid, high | 6 |
| MATD3 | 4manager, 10manager | low, mid, high | 6 |
| SQDDPG | 4manager, 10manager | low, mid, high | 6 |
| MAPPO | 4manager, 10manager | low, mid, high | 6 |
| MAIPPO | 4manager, 10manager | low, mid, high | 6 |
| **Total** | | | **48** |

## Churn Configuration

| Level | Severity Range | Probability |
|-------|----------------|-------------|
| Low | 10-15% | [0.4, 0.3, 0.3] |
| Mid | 20-25% | [0.4, 0.3, 0.3] |
| High | 30-35% | [0.4, 0.3, 0.3] |

- **Trigger Interval**: Every 5 episodes
- **Mid-Episode Steps**: [6, 12, 18] (within 24-step episode)

## Usage

### Run EA Tests Only
```bash
bash run_ea_only.sh
```

### Run All Baseline Tests
```bash
bash run_baselines.sh
```

### Run All Tests (Sequential by Algorithm)
```bash
bash run_all.sh
```

### Run with Parallel Control
```bash
python run_parallel.py --parallel 6          # 6 parallel jobs
python run_parallel.py --ea_only             # EA only
python run_parallel.py --baseline_only       # Baselines only
python run_parallel.py --algorithms maddpg sqddpg  # Specific algorithms
```

## Directory Structure

```
enhance/
├── scripts/
│   ├── train_ea.py         # Enhanced EA training
│   └── train_baseline.py   # Baseline training with full compat layer
├── envs/
│   └── mid_episode_churn_wrapper.py
├── results/
│   ├── ea_4manager_low/
│   ├── ea_4manager_mid/
│   ├── ...
│   └── experiment_summary.csv
├── run_ea_only.sh
├── run_baselines.sh
├── run_all.sh
├── run_parallel.py
└── README.md
```

## Output Files

Each experiment creates:
- `training_history.csv`: Episode-by-episode metrics
- `summary.json`: Final SSR and configuration
- `model_ep*.pt`: Saved model checkpoints
- `model_final.pt`: Final model
