# EA (Environment-Adaptive MARL) Technical Design Document

## 1. Overview

EA is a churn-consistent, set-to-set MATD3-style multi-agent actor-critic algorithm designed for FOgym environments with dynamic device sets.

**Key Features**:
- Support device churn without changing network dimensions
- Fixed N_max + mask mechanism
- Twin critics with delayed policy update
- Churn-consistent TD targets
- True learning with complete gradient flow

---

## 2. Dimension Parameters

### 2.1 Environment Dimensions (from analysis)

**4Manager Environment**:
- Total devices: 131
- Max devices per Manager: 44
- Managers: 4

**10Manager Environment**:
- Total devices: 328
- Max devices per Manager: 48
- Managers: 10

**Unified Parameters**:
- **N_max = 60**: Fixed maximum devices per Manager (provides churn buffer)
- **M = 4 or 10**: Number of Managers
- **p = 5**: Action dimension per device (FlexOffer parameters)
- **x_dim = 6**: Unified device state dimension (max across types, will pad smaller ones)
- **g_dim ≈ 50**: Manager global features (time + env + market + history)

### 2.2 Device State Dimensions (x_dim)

| Device Type | Original Dim | Features |
|-------------|--------------|----------|
| Battery     | 4D | SOC, max_charge_power, max_discharge_power, health |
| Dishwasher  | 6D | is_deployed, is_running, is_completed, progress, urgency, remaining_energy |
| EV          | 3D | SOC, is_connected, urgency |
| HeatPump    | 3D | current_temp, target_temp, comfort_score |
| PV          | 2D | current_power, forecast_power |

**Unified x_dim = 6D** (pad smaller dimensions with zeros)

### 2.3 Action Dimensions (p)

Per-device FlexOffer parameters (5D):
1. start_flex: Start time flexibility [-1, 1]
2. end_flex: End time flexibility [-1, 1]
3. energy_min_factor: Minimum energy factor [0.1, 1.0]
4. energy_max_factor: Maximum energy factor [1.0, 2.0]
5. priority_weight: Priority weight [0.1, 2.0]

---

## 3. Network Architecture

### 3.1 Set-to-Set Actor πθ

**Input**:
- g: [B, g_dim] - Manager global features
- X: [B, N_max, x_dim] - Device states (padded)
- mask: [B, N_max] - Active device mask {0,1}
- manager_id: int - Manager identifier

**Architecture**:
```
1. Token Encoder φ_k:
   Input: [x_j, g, emb(manager_id)] → [x_dim + g_dim + emb_dim]
   MLP: [x_dim + g_dim + emb_dim] → hidden_dim → token_dim
   Output: u_j [B, N_max, token_dim]

2. Pooling:
   u_bar = masked_mean(u, mask) → [B, token_dim]

3. Refiner φ_r:
   Input: [u_j, u_bar] → [token_dim * 2]
   MLP: [token_dim * 2] → hidden_dim → hidden_dim
   Output: h_j [B, N_max, hidden_dim]

4. Action Head:
   Input: h_j → hidden_dim
   MLP: hidden_dim → hidden_dim → p
   Output: a_j [B, N_max, p]
   
5. Mask Application:
   a_j = a_j * mask[:, :, None]  # Zero out inactive slots
```

**Hyperparameters**:
- emb_dim: 16 (manager ID embedding)
- token_dim: 128
- hidden_dim: 256
- activation: ReLU
- output_activation: Tanh (for bounded actions)

### 3.2 Pair-Set Critics Q1ϕ, Q2ϕ

**Input**:
- g: [B, g_dim]
- X: [B, N_max, x_dim]
- A: [B, N_max, p]
- mask: [B, N_max]
- manager_id: int

**Architecture**:
```
1. Pair Token Encoder ψ_Q:
   p_j = [x_j, a_j] → [x_dim + p]
   Input: [p_j, g, emb(manager_id)] → [x_dim + p + g_dim + emb_dim]
   MLP: → hidden_dim → token_dim
   Output: v_j [B, N_max, token_dim]

2. Pooling:
   v_bar = masked_mean(v, mask) → [B, token_dim]

3. Per-device Q Head ρ_dev:
   Input: v_j → token_dim
   MLP: token_dim → hidden_dim → 1
   Output: q_j [B, N_max, 1]

4. Global Q Head ρ_glob:
   Input: [v_bar, g] → [token_dim + g_dim]
   MLP: → hidden_dim → hidden_dim → 1
   Output: q_glob [B, 1]

5. Final Q value:
   Q = masked_mean(q_j, mask) + q_glob
```

**Twin Critics**: Two independent networks Q1 and Q2 with same architecture

---

## 4. Training Algorithm

### 4.1 Replay Buffer

**Transition Structure**:
```python
{
    'manager_id': int,
    'g': [g_dim],           # Current global features
    'X': [N_max, x_dim],    # Current device states (padded)
    'mask': [N_max],        # Current active mask
    'A': [N_max, p],        # Actions (padded)
    'r': float,             # Reward
    'g_next': [g_dim],      # Next global features
    'X_next': [N_max, x_dim], # Next device states
    'mask_next': [N_max],   # Next active mask (churn-consistent!)
    'done': bool
}
```

**Capacity**: 100,000 transitions

### 4.2 Training Loop

**Hyperparameters**:
- gamma: 0.99 (discount factor)
- tau: 0.005 (soft update rate)
- lr_actor: 1e-4
- lr_critic: 1e-3
- batch_size: 256
- policy_delay: 2
- warmup_steps: 1000
- noise_scale: 0.1 (exploration noise)
- noise_clip: 0.2 (target policy smoothing)

**Algorithm**:
```
For each environment step:
    1. Select actions for all Managers using actor + exploration noise
    2. Convert padded actions to FOgym action dict
    3. Step environment
    4. Store transitions (per Manager) with mask and mask_next
    5. If buffer size >= warmup_steps:
        a. Sample batch
        b. Compute target actions: A_next = actor_target(g_next, X_next, mask_next)
           - Apply mask_next: A_next *= mask_next  (churn-consistent!)
           - Add target policy smoothing noise (only on active slots)
        c. Compute target Q values: y = r + γ * min(Q1_target, Q2_target)
        d. Update critics: loss = MSE(Q1, y) + MSE(Q2, y)
        e. If step % policy_delay == 0:
            - Compute actor actions: A_pi = actor(g, X, mask)
            - Update actor: loss = -Q1(g, X, A_pi * mask, mask)
            - Soft update targets: θ' ← τθ + (1-τ)θ'
```

### 4.3 Churn-Consistent TD Target

**Key Point**: Use `mask_next` (realized next active set) instead of `mask`

```python
# WRONG (not churn-consistent):
A_next = actor_target(g_next, X_next, mask)  # Uses current mask

# CORRECT (churn-consistent):
A_next = actor_target(g_next, X_next, mask_next)  # Uses next mask
A_next = A_next * mask_next[:, :, None]  # Enforce mask
```

This avoids "semantic drift" when device sets change due to churn.

---

## 5. Churn Mechanism

### 5.1 Churn Configuration

```python
@dataclass
class ChurnConfig:
    enabled: bool = True
    trigger_interval: int = 10  # Churn every K episodes
    severity_levels: Tuple[float, float, float] = (0.02, 0.05, 0.10)
    severity_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1)
    min_active_devices: int = 5  # Minimum active devices
    device_types: List[str] = ['battery', 'dishwasher', 'ev', 'heat_pump', 'pv']
```

### 5.2 Universe Pool Design

**Initial State**:
- Universe Pool U^(i) = all devices in CSV file for Manager i
- Active Set D^(i) = U^(i) initially
- Inactive Set I^(i) = ∅ initially

**Churn Execution** (at episode reset, if episode % K == 0):
```
For each Manager i:
    1. Sample severity ρ ~ {0.02, 0.05, 0.10} with probs
    2. n = |D^(i)|  (current active count)
    3. k = max(1, round(ρ * n))  (devices to toggle)
    4. k_leave = floor(k/2)
    5. k_join = k - k_leave
    6. Sample k_leave devices from D^(i) → leaving_set
    7. Sample k_join devices from I^(i) → joining_set
       - If |I^(i)| < k_join: k_join = |I^(i)|
       - If I^(i) is empty: create k_join new random devices
    8. Update:
       D^(i) = (D^(i) \ leaving_set) ∪ joining_set
       I^(i) = U^(i) \ D^(i)
    9. Ensure |D^(i)| >= min_active_devices
```

### 5.3 Churn Logging

**Log Information**:
- Episode index
- Per-Manager: (ρ, k_leave, k_join, |D^(i)| before, |D^(i)| after)
- Device IDs that left/joined
- Device types distribution after churn

**Exposure in info dict**:
```python
info['churn_event'] = {
    'triggered': bool,
    'episode': int,
    'managers': {
        'manager_1': {
            'severity': float,
            'left': int,
            'joined': int,
            'active_before': int,
            'active_after': int,
            'left_devices': List[str],
            'joined_devices': List[str]
        },
        ...
    }
}
```

---

## 6. Implementation Plan

### Phase 1: Core Components
1. Utility functions: masked_mean, pad_to_Nmax, ManagerEmbedding
2. Set-to-Set Actor network
3. Pair-Set Critics networks (Q1, Q2)
4. Replay Buffer with mask support

### Phase 2: Training System
1. EAAgent class (manages actor + critics + targets)
2. select_action() with exploration noise
3. compute_td_target() with churn-consistent logic
4. update() method with delayed policy update

### Phase 3: Churn Environment
1. ChurnConfig dataclass
2. DevicePool class (manages U, D, I)
3. ChurnManager (handles churn execution)
4. Extend MultiAgentFlexOfferEnv with churn support

### Phase 4: Integration
1. EA Adapter for FOPipeline
2. Test scripts
3. Integration with Test/run_test.py
4. Documentation

---

## 7. Testing Strategy

### 7.1 Unit Tests
- [ ] Test masked_mean correctness
- [ ] Test pad_to_Nmax with various input sizes
- [ ] Test Actor forward pass with different masks
- [ ] Test Critics forward pass
- [ ] Test Replay Buffer add/sample

### 7.2 Integration Tests
- [ ] Test EA training without churn (sanity check)
- [ ] Test EA training with churn (adaptive learning)
- [ ] Test churn mechanism (devices leave/join correctly)
- [ ] Test TD target uses mask_next (churn-consistent)

### 7.3 Performance Tests
- [ ] Verify critics loss decreases
- [ ] Verify actor improves policy (increasing Q values)
- [ ] Verify learning continues after churn events
- [ ] Compare EA vs other algorithms (MAPPO, MADDPG, etc.)

---

## 8. Key Design Decisions

### 8.1 Why N_max = 60?
- 4Manager max: 44 devices
- 10Manager max: 48 devices
- Buffer for churn: devices can join, increasing active count beyond initial max
- Provides ~25% headroom for device growth during churn events
- Balance between memory efficiency and flexibility

### 8.2 Why unified x_dim = 6?
- Simplifies network design (single input dimension)
- Pad smaller device states with zeros
- Network learns to ignore padding through mask

### 8.3 Why p = 5 (FlexOffer parameters)?
- Matches FOgym's FlexOffer generation paradigm
- Provides sufficient flexibility for device scheduling
- Compatible with existing FOPipeline

### 8.4 Why Twin Critics?
- Reduces overestimation bias (MATD3 feature)
- Improves training stability
- Standard practice in modern off-policy methods

### 8.5 Why Delayed Policy Update?
- Reduces variance in policy gradient
- Allows critics to stabilize before actor update
- Improves sample efficiency

---

## 9. Expected Behavior

### 9.1 Without Churn
- Stable learning, critics loss decreases
- Actor improves policy over episodes
- Performance converges after ~500-1000 episodes

### 9.2 With Churn
- Temporary performance drop after churn event
- Quick recovery (within 50-100 episodes)
- Demonstrates adaptive learning capability
- Long-term performance comparable to no-churn case

---

## 10. References

- EA.md: Algorithm specification
- churnFOgym.md: Churn mechanism specification
- MATD3: algorithms/MATD3/fomatd3/ (inspiration for twin critics)
- FOgym: fo_generate/multi_agent_env.py (environment interface)

---

**Document Version**: 1.0
**Date**: 2026-01-12
**Author**: EA Implementation Team
