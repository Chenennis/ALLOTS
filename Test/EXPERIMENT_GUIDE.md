# Experiment Guide

Detailed documentation for the comparative study, ablation study, and hyperparameter analysis.

---

## 1. Comparison Study

### 1.1 Research Objective

This comparative study evaluates the performance of eight multi-agent reinforcement learning (MARL) algorithms in FlexOffer aggregation environments with dynamic device participation (churn). **ALLOTS（EA (Environment-Adaptive)）** algorithm outperforms traditional MARL baselines when devices join and leave the system dynamically.

### 1.2 Algorithm Configurations

#### EA (Environment-Adaptive) - Proposed Method

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_max | 44 (4M) / 64 (10M) | Maximum device slots |
| x_dim | 6 | Device state dimension |
| g_dim | 26 | Global feature dimension |
| p | 5 | Action parameters per device |
| hidden_dim | 256 | Hidden layer dimension |
| lr_actor | 1e-4 | Actor learning rate |
| lr_critic | 1e-3 | Critic learning rate |
| gamma | 0.99 | Discount factor |
| tau | 0.005 | Soft update coefficient |
| batch_size | 256 | Batch size |
| policy_delay | 2 | Actor update frequency |
| advantage_tau | 1.0 | Softmax temperature for per-device weighting |

#### Baseline Algorithms

| Algorithm | lr_actor | lr_critic | hidden_dim | batch_size | Special |
|-----------|----------|-----------|------------|------------|---------|
| AGILE | 1e-4 | 1e-3 | 256 | 256 | Action set learning |
| MAAC | 1e-4 | 1e-3 | 128 | 1024 | Attention mechanism |
| MADDPG | 1e-3 | 1e-3 | 64 | 1024 | - |
| MATD3 | 1e-4 | 1e-3 | 256 | 64 | policy_delay=2 |
| SQDDPG | 1e-3 | 1e-3 | 64 | 64 | Shapley sampling |
| MAPPO | 3e-4 | 3e-4 | 64 | 1024 | PPO clip=0.2 |
| MAIPPO | 3e-4 | 3e-4 | 64 | 1024 | Independent |

### 1.3 Device Churn Configuration

#### Churn Severity Levels

| Parameter | Low Churn | Mid Churn | High Churn |
|-----------|-----------|-----------|------------|
| Severity Range | 10-15% | 20-25% | 30-35% |
| Severity Levels | [0.10, 0.125, 0.15] | [0.20, 0.225, 0.25] | [0.30, 0.325, 0.35] |
| Probability | [0.4, 0.3, 0.3] | [0.4, 0.3, 0.3] | [0.4, 0.3, 0.3] |

#### Churn Trigger Mechanism

| Trigger Type | When | Description |
|--------------|------|-------------|
| **Episode-level** | Every 5 episodes | Churn at episode reset |
| **Mid-episode** | Steps 6, 12, 18 | Churn during episode (every 6 hours) |

The mid-episode churn is more realistic as devices can join/leave at any time during operation, not just at episode boundaries.

**Churn Impact (Approximate devices changed per trigger):**

| Environment | Low | Mid | High |
|-------------|-----|-----|------|
| 4Manager | 12-18 | 24-30 | 35-42 |
| 10Manager | 33-50 | 66-82 | 98-115 |

### 1.4 Complete Test Matrix (48 experiments)

```
4Manager Environment:
├── Low Churn:  ea, agile, maac, maddpg, matd3, sqddpg, mappo, maippo
├── Mid Churn:  ea, agile, maac, maddpg, matd3, sqddpg, mappo, maippo
└── High Churn: ea, agile, maac, maddpg, matd3, sqddpg, mappo, maippo

10Manager Environment:
├── Low Churn:  ea, agile, maac, maddpg, matd3, sqddpg, mappo, maippo
├── Mid Churn:  ea, agile, maac, maddpg, matd3, sqddpg, mappo, maippo
└── High Churn: ea, agile, maac, maddpg, matd3, sqddpg, mappo, maippo
```

**Total: 8 algorithms × 2 environments × 3 churn levels = 48 experiments**

### 1.5 Running Comparison Study

```bash
# Run all algorithms for 4Manager mid churn
for algo in ea agile maac maddpg matd3 sqddpg mappo maippo; do
    python Test/4manager_mid_midf_${algo}.py --episodes 500 --mini_log
done

# Run specific configuration
python Test/10manager_high_midf_ea.py --episodes 500 --mini_log --gpu
```

---

## 2. Ablation Study

### 2.1 Design

The ablation study evaluates the contribution of each key component of the EA algorithm:

| Component | Purpose |
|-----------|---------|
| **Per-device advantage weighting** | Focus learning on high-advantage devices |
| **Pair-set critic** | Handle set-to-set input/output structure |
| **TD-consistent updates** | Stable learning under churn |

### 2.2 Variants

| Variant | Description | What's Changed |
|---------|-------------|----------------|
| EA-full | Complete EA algorithm | Baseline |
| No Credit | Remove per-device advantage | Uniform weighting (all devices equal) |
| No Pairset | Remove pair-set critic | Standard MLP critic |
| No TDconsistent | Remove TD-consistent | Standard TD updates |

### 2.3 Test Configurations

Each variant is tested on:
- **Environments**: 4Manager, 10Manager
- **Churn levels**: Low, Mid, High
- **Total**: 4 variants × 2 environments × 3 churn levels = 24 experiments

### 2.4 Running Ablation Study

```bash
cd Test/Ablation

# Run individual ablation
python run_ablation_no_credit.py --episodes 500
python run_ablation_no_pairset.py --episodes 500
python run_ablation_no_tdconsistent.py --episodes 500

# Run all ablations
python run_all_ablations.py
```

### 2.5 Ablation Scripts Location

```
Test/Ablation/
├── agents/
│   ├── ea_no_credit.py          # EA without per-device weighting
│   ├── ea_no_pairset.py         # EA without pair-set critic
│   └── ea_no_tdconsistent.py    # EA without TD-consistent
├── scripts/
│   └── {env}_{churn}_no_{component}.py
└── analysis/
    ├── analyze_results.py
    └── visualization.py
```

---

## 3. Hyperparameter Analysis

### 3.1 Parameters Analyzed

| Parameter | Symbol | Values Tested | Purpose |
|-----------|--------|---------------|---------|
| Advantage Temperature | τ | 0.1, 0.5, 1.0, 2.0, 5.0 | Controls softmax sharpness in advantage weighting |
| Hidden Dimension | h | 64, 128, 256 | Network capacity |

### 3.2 Expected Effects

**Advantage Temperature (τ):**
- τ → 0: Sharp distribution, focuses on highest-advantage device only
- τ → ∞: Uniform distribution, treats all devices equally
- Optimal: Balance between focus and exploration (typically τ ≈ 1.0)

**Hidden Dimension (h):**
- Larger h: More expressive but risk overfitting
- Smaller h: Faster training but limited capacity

### 3.3 Running Hyperparameter Analysis

```bash
cd Test/hyparameter

# Run tau sweep
python run_tau_0.1.py
python run_tau_0.5.py
python run_tau_1.0.py
python run_tau_2.0.py
python run_tau_5.0.py

# Run hidden dimension sweep
python run_h_64.py
python run_h_128.py
python run_h_256.py

# Analyze results
python analyze_results.py
```

### 3.4 Output

Results saved to:
```
Test/hyparameter/analysis/
├── tau_w_results.csv       # τ sensitivity results
├── tau_w_sensitivity.png   # τ sensitivity plot
├── h_results.csv           # h sensitivity results
└── h_sensitivity.png       # h sensitivity plot
```

---

## 4. Compatibility Layer

### 4.1 Overview

The compatibility layer enables fixed-dimension MARL algorithms to operate in dynamic environments:

```
┌─────────────────────────────────────────────────────────┐
│                 Compatibility Layer                      │
├─────────────────────────────────────────────────────────┤
│  ┌───────────┐   ┌───────────┐   ┌───────────┐         │
│  │SlotMapper │   │ObsAdapter │   │ActAdapter │         │
│  │device↔slot│   │raw→padded │   │padded→dict│         │
│  └───────────┘   └───────────┘   └───────────┘         │
│         │               │               │               │
│         └───────────────┼───────────────┘               │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │         MultiManagerCompatWrapper                │   │
│  │  • Manages adapters for all managers            │   │
│  │  • use_stable_mapping=True for baselines        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Key Components

| Component | File | Purpose |
|-----------|------|---------|
| SlotMapper | `adapters/slot_mapper.py` | Stable device-to-slot mapping |
| ObsAdapter | `adapters/obs_adapter.py` | Convert to fixed-length observations |
| ActAdapter | `adapters/act_adapter.py` | Convert fixed-length actions back |
| Wrapper | `adapters/multi_manager_wrapper.py` | Orchestrate all adapters |

### 4.3 EA vs Baseline Handling

| Feature | EA (Native) | Baseline + Compat Layer |
|---------|-------------|-------------------------|
| Observation | Native set-to-set | Padded + Mask |
| Slot Mapping | Implicit (slot-invariant) | Stable (maintains binding) |
| Churn Handling | Built-in architecture | Adapter handles |

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| Average Reward | Mean episodic reward | Higher |
| Final Reward | Reward at last episode | Higher |
| Convergence Speed | Episodes to threshold | Lower |
| Reward Stability | Std dev of rewards | Lower |

### 5.2 Churn-Specific Metrics

| Metric | Description |
|--------|-------------|
| SSR (Slot Stability Rate) | Stability of device-to-slot mappings |
| PCV (Post-Churn Value) | Performance immediately after churn |
| Recovery Rate | Speed of returning to pre-churn performance |
| Adaptation Speed | Episodes to recover after churn |

---

## 6. Reward Function

### 6.1 Design Philosophy

- Center around zero for random policies
- Provide meaningful gradients for learning
- Range: -40 (poor) to +100 (excellent)

### 6.2 Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Economic Reward | ~40% | Net profit from trading |
| Satisfaction Reward | ~30% | User satisfaction level |
| Coordination Reward | ~20% | Trade success rate |
| Strategy Quality | ~10% | Action consistency |

### 6.3 Expected Ranges

| Policy Type | Reward Range |
|-------------|--------------|
| Random | -10 to +10 |
| Poor | -40 to -10 |
| Average | 0 to +20 |
| Good | +20 to +60 |
| Excellent | +60 to +100 |

---

## 7. Hardware Requirements

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA with CUDA support |
| VRAM | ~3GB per training |
| CPU | Multi-core for parallel tests |
| RAM | 16GB+ recommended |

---

## 8. Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Episodes | 500 |
| Test Episodes | 100 |
| Warmup Episodes | 10 |
| Evaluation Interval | 10 episodes |
| Model Save Interval | 50 episodes |

---

**Document Version**: 1.0  
**Last Updated**: January 2026
