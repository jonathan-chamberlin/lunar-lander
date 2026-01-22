# Experiment Ideas

Ideas for future experiments, organized by category. Parameter values chosen based on first-principles reasoning against current defaults.

**Current Optimal Defaults (validated by experiments):**
- batch_size: 128 (EXP_004: 32 fastest start, 256 best final)
- buffer_size: 16384 (EXP_007: optimal; larger catastrophically fails)
- actor_lr: 0.001 (EXP_009: 1:1 ratio optimal)
- critic_lr: 0.001 (EXP_009: equal LRs beat 2:1 ratio)
- tau: 0.005 (default, EXP_010 testing)
- gamma: 0.99 (default, untested)
- hidden_sizes: [64, 32] (EXP_006: 2x faster, same performance)
- training_updates_per_episode: 25 (EXP_005: 10 optimal but 25 is default)
- noise_decay_episodes: 300
- time_penalty: False (EXP_012: hurts learning)
- altitude_bonus: True (EXP_012: part of optimal F/T/T/T)
- leg_contact: True (EXP_012: part of optimal F/T/T/T)
- stability: True (EXP_012: part of optimal F/T/T/T)

---

## Completed Experiments

| ID | Parameter | Key Finding |
|----|-----------|-------------|
| EXP_004 | batch_size | 128 fails; 32 fastest, 256 best final |
| EXP_005 | training_updates_per_episode | 10 optimal; 50 highly unstable |
| EXP_006 | hidden_sizes | [64,32] matches [256,128] while 2x faster |
| EXP_007 | buffer_size | 16384 optimal; 65536 catastrophic |
| EXP_009 | actor_lr, critic_lr | 1:1 ratio (0.001/0.001) optimal; actor>critic catastrophic |
| EXP_012 | reward shaping | F/T/T/T (no time penalty) best at 51% |

---

## Planned Experiments

### EXP_010: Tau Sweep

**Status:** PLANNED

**Hypothesis:** The default tau=0.005 may not be optimal; different soft update coefficients could improve stability or speed.

**Values:** [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]

See `EXP_010_tau_sweep/EXPERIMENT.md` for full details.

---

## Future Experiment Ideas

### EXP_013: Gamma (Discount Factor) Sweep

**Hypothesis:** The default gamma=0.99 may be suboptimal for LunarLander's episodic structure; different values could change the exploration/exploitation balance.

**Reasoning:**
- Gamma controls how much future rewards are valued vs immediate rewards
- gamma=0.99: Values rewards ~100 steps out significantly (0.99^100 ≈ 0.37)
- gamma=0.95: Horizon shrinks to ~20 steps (0.95^20 ≈ 0.36)
- gamma=0.999: Very long horizon, almost no discounting
- LunarLander episodes typically 200-400 steps for successful landings
- Lower gamma might encourage faster, more decisive actions
- Higher gamma might improve long-term planning but slow learning

**First Principles Analysis:**
- Effective horizon = 1 / (1 - gamma)
  - gamma=0.9 → 10 step horizon
  - gamma=0.95 → 20 step horizon
  - gamma=0.99 → 100 step horizon
  - gamma=0.999 → 1000 step horizon
- For LunarLander, the +100 landing reward is the key signal
- Too low gamma: Agent may not "see" the landing reward from starting altitude
- Too high gamma: TD targets become very high variance (bootstrapping compounds errors)

**Values to test:** [0.9, 0.95, 0.99, 0.995, 0.999]

```json
{
  "name": "gamma_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "gamma": [0.9, 0.95, 0.99, 0.995, 0.999]
  }
}
```

**Predictions:**
- gamma=0.9 will fail (horizon too short to see landing reward)
- gamma=0.99 (default) will be near-optimal
- gamma=0.999 may be unstable due to high variance TD targets

---

### EXP_014: Gamma + Tau Interaction

**Hypothesis:** Gamma and tau interact - higher gamma may require lower tau for stability.

**Reasoning:**
- High gamma = longer credit assignment chains = more potential for error accumulation
- High tau = faster target updates = more potential instability
- The combination of high gamma AND high tau may be catastrophic
- Conversely, low gamma may tolerate higher tau

**Values to test (2x2 factorial):**

```json
{
  "name": "gamma_tau_interaction",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "gamma": [0.95, 0.99],
    "tau": [0.005, 0.02]
  }
}
```

---

### EXP_015: PER Alpha/Beta Sweep

**Hypothesis:** The PER hyperparameters (alpha, beta) affect learning stability; current defaults may not be optimal.

**Reasoning:**
- alpha controls prioritization strength (0=uniform, 1=full priority)
- beta controls importance sampling correction (0=none, 1=full)
- Current: alpha=0.6, beta_start=0.4, beta_end=1.0
- Higher alpha = more focus on high-TD experiences, but more bias
- Higher beta = more correction for sampling bias, but higher variance

**Values to test:**

```json
{
  "name": "per_alpha_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "per_alpha": [0.4, 0.6, 0.8]
  }
}
```

---

### EXP_016: Gradient Clipping Sweep

**Hypothesis:** The gradient clip value (currently 10.0) may be too loose or too tight.

**Reasoning:**
- Gradient clipping prevents exploding gradients
- Too tight: Limits learning speed, may cause slow convergence
- Too loose: May not prevent instability
- Current value (10.0) is conservative; many implementations use 1.0 or 0.5

**Values to test:** [0.5, 1.0, 5.0, 10.0, None]

```json
{
  "name": "gradient_clip_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "gradient_clip_value": [0.5, 1.0, 5.0, 10.0]
  }
}
```

---

### EXP_017: Noise Schedule Sweep

**Hypothesis:** The noise decay schedule affects exploration quality; different schedules may improve sample efficiency.

**Reasoning:**
- Current: Linear decay from 1.0 to 0.2 over 300 episodes
- Fast decay: Quick exploitation, risk of local minima
- Slow decay: More exploration, slower convergence
- For 500-episode runs, 300 episodes means noise is minimal for final 200

**Values to test:**

```json
{
  "name": "noise_decay_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "noise_decay_episodes": [100, 200, 300, 400]
  }
}
```

---

### EXP_018: Noise Sigma Sweep

**Hypothesis:** The initial noise magnitude (sigma=0.3) affects exploration quality.

**Reasoning:**
- sigma controls the magnitude of Ornstein-Uhlenbeck noise
- Too low: Insufficient exploration, may miss good policies
- Too high: Actions too random, slow to converge
- LunarLander actions are bounded [-1, 1]; sigma=0.3 is moderate

**Values to test:** [0.1, 0.2, 0.3, 0.5]

```json
{
  "name": "noise_sigma_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "sigma": [0.1, 0.2, 0.3, 0.5]
  }
}
```

---

### EXP_019: TD3 Policy Noise Sweep

**Hypothesis:** The target policy noise parameters affect learning stability and final performance.

**Reasoning:**
- TD3 adds noise to target actions for smoothing (reduces overestimation)
- target_policy_noise: Current 0.1, controls noise magnitude
- target_noise_clip: Current 0.3, clips noise to prevent extreme values
- Higher noise = more regularization but potentially slower learning

**Values to test:**

```json
{
  "name": "td3_noise_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "target_policy_noise": [0.05, 0.1, 0.2],
    "target_noise_clip": [0.2, 0.3, 0.5]
  }
}
```

---

### EXP_020: Policy Update Frequency Sweep

**Hypothesis:** The delayed policy update frequency (currently 3) may not be optimal.

**Reasoning:**
- TD3 delays policy updates to let critic stabilize
- policy_update_frequency=1: Update every critic update (DDPG-style)
- policy_update_frequency=2-3: TD3 default range
- Higher delay = more stable but slower policy improvement

**Values to test:** [1, 2, 3, 5]

```json
{
  "name": "policy_update_freq_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "policy_update_frequency": [1, 2, 3, 5]
  }
}
```

---

### EXP_021: Long Training Run

**Hypothesis:** Current 500-episode runs may not reach full potential; longer training could show continued improvement or plateau.

**Reasoning:**
- Best configs achieve ~50-70% final-100 success
- Unknown if this is the ceiling or just insufficient training
- Longer runs (1000-2000 episodes) reveal true learning dynamics

**Config:**

```json
{
  "name": "long_training",
  "type": "grid",
  "episodes_per_run": 2000,
  "num_runs_per_config": 3,
  "parameters": {
    "placeholder": [true]
  }
}
```

---

### EXP_022: Min Experiences Before Training

**Hypothesis:** The warmup period (min_experiences_before_training=2000) affects early learning quality.

**Reasoning:**
- Current: Wait for 2000 experiences before first training update
- Too few: Training on bad initial data may harm learning
- Too many: Delays start of learning, wastes early episodes

**Values to test:** [500, 1000, 2000, 4000]

```json
{
  "name": "warmup_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "min_experiences_before_training": [500, 1000, 2000, 4000]
  }
}
```

---

## Priority Queue

Based on expected impact and ease of implementation:

### High Priority (likely impactful, well-defined)
1. **EXP_013: Gamma Sweep** - Untested core hyperparameter
2. **EXP_017: Noise Schedule** - May improve exploration efficiency
3. **EXP_021: Long Training** - Understand true performance ceiling

### Medium Priority (useful but secondary)
4. **EXP_014: Gamma+Tau Interaction** - Test interaction effects
5. **EXP_015: PER Alpha/Beta** - Tune prioritized replay
6. **EXP_020: Policy Update Frequency** - TD3-specific tuning

### Lower Priority (incremental improvements)
7. **EXP_016: Gradient Clipping** - Minor tuning
8. **EXP_018: Noise Sigma** - Minor tuning
9. **EXP_019: TD3 Noise** - Minor tuning
10. **EXP_022: Warmup** - Minor tuning
