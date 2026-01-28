# Test - Experimental Framework

This folder contains the complete experimental framework for evaluating multi-agent reinforcement learning (MARL) algorithms in dynamic FlexOffer environments with device churn.

## 📁 Directory Structure

```
Test/
├── README.md                    # This file
├── EXPERIMENT_GUIDE.md          # Detailed experiment setup guide
│
├── Comparison Study Scripts     # 48 scripts (8 algorithms × 2 envs × 3 churns)
│   ├── 4manager_*_midf_*.py    # 4Manager environment tests
│   └── 10manager_*_midf_*.py   # 10Manager environment tests
│
├── Ablation/                    # Ablation study
│   ├── agents/                  # EA variants (no_credit, no_pairset, no_tdconsistent)
│   ├── scripts/                 # Ablation test scripts
│   ├── scripts_midchurn/        # Mid-episode churn scripts
│   ├── envs/                    # Mid-episode churn wrapper
│
├── hyparameter/                 # Hyperparameter sensitivity analysis
│   ├── run_*.py                 # Parameter sweep scripts
│
├── enhance/                     # Enhanced comparison with mid-episode churn
│   ├── scripts/                 # Training scripts
│   └── envs/                    # Environment wrappers
│
├── configs/                     # YAML configuration files
├── data/                        # CSV data (4manager, 10manager)
└── examples/                    # Example configurations
```

## 🚀 Quick Start

### 1. Comparison Study
```bash
# Run EA algorithm (4Manager, mid churn)
python Test/4manager_mid_midf_ea.py --episodes 100 --mini_log

# Run baseline algorithms
python Test/4manager_mid_midf_maddpg.py --episodes 100 --mini_log
python Test/4manager_mid_midf_mappo.py --episodes 100 --mini_log
```

### 2. Ablation Study
```bash
# Run ablation variants
python Test/Ablation/run_ablation_no_credit.py --episodes 100
python Test/Ablation/run_ablation_no_pairset.py --episodes 100
python Test/Ablation/run_ablation_no_tdconsistent.py --episodes 100
```

### 3. Hyperparameter Analysis
```bash
# Run tau sensitivity
python Test/hyparameter/run_tau_0.1.py
python Test/hyparameter/run_tau_1.0.py

# Analyze results
python Test/hyparameter/analyze_results.py
```

## 📊 Experiment Overview

### Comparison Study (48 experiments)

Compares **EA (Environment-Adaptive)** algorithm with 7 baseline MARL methods:

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **EA** (Proposed) | Actor-Critic | Per-device advantage weighting, native churn handling |
| AGILE | Actor-Critic | Action set learning for large discrete spaces |
| MAAC | Actor-Critic | Multi-agent attention critic |
| MADDPG | Actor-Critic | Centralized critic, decentralized actors |
| MATD3 | Actor-Critic | Twin critics, delayed policy updates |
| SQDDPG | Actor-Critic | Shapley value credit assignment |
| MAPPO | Policy Gradient | PPO with centralized value function |
| MAIPPO | Policy Gradient | Independent PPO agents |

**Test Configurations:**
- **Environments**: 4Manager (36 users, 118 devices) / 10Manager (90 users, 328 devices)
- **Churn Levels**: Low (10-15%) / Mid (20-25%) / High (30-35%)
- **Churn Mechanism**: 
  - Episode-level: Every 5 episodes
  - **Mid-episode**: At steps 6, 12, 18 (every 6 hours within 24-hour episode)

### Ablation Study

Evaluates the contribution of each EA component:

| Variant | Removed Component | Tests |
|---------|-------------------|-------|
| EA-full | None (baseline) | Per-device advantage + Pair-set critic + TD-consistent |
| No Credit | Per-device advantage weighting | Uniform weighting |
| No Pairset | Pair-set critic | Standard critic |
| No TDconsistent | TD-consistent updates | Standard TD updates |

### Hyperparameter Analysis

Sensitivity analysis on key hyperparameters:

| Parameter | Values Tested | Purpose |
|-----------|--------------|---------|
| τ (advantage temperature) | 0.1, 0.5, 1.0, 2.0, 5.0 | Controls advantage sharpness |
| h (hidden dimension) | 64, 128, 256 | Network capacity |

## 🔧 Environment Configuration

### 4Manager Environment
- **Managers**: 4
- **Users**: 36
- **Devices**: 118 (Battery: 24, EV: 14, Heat Pump: 36, Dishwasher: 36, PV: 8)
- **N_max**: 44 (max devices per manager)

### 10Manager Environment
- **Managers**: 10
- **Users**: 90
- **Devices**: 328 (Battery: 76, EV: 45, Heat Pump: 90, Dishwasher: 90, PV: 27)
- **N_max**: 64 (max devices per manager)


## 📝 Output Files

Each experiment generates:
```
Test/results/{algorithm}_{env}_{churn}/
├── training_history.csv    # Episode-by-episode metrics
├── results.json            # Final results with config
├── train_*.log             # Detailed training log
├── model_ep50.pt           # Checkpoint
└── model_final.pt          # Final model
```

## ⚙️ Command Line Arguments

```bash
python Test/{script}.py [OPTIONS]

OPTIONS:
  --mode {train,test,both}  # Run mode (default: train)
  --episodes INT            # Training episodes (default: 100)
  --mini_log                # Minimal console logging
  --gpu                     # Enable GPU acceleration
```

## 📚 For More Details

See **EXPERIMENT_GUIDE.md** for:
- Complete algorithm configurations
- Detailed churn mechanism
- Reward function design
- Compatibility layer architecture
- Full test matrix

---

**Last Updated**: January 2026
