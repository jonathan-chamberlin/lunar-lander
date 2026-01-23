---
id: EXP_013
title: Gamma Sweep
status: CONCLUDED
created: 2026-01-22
completed: 2026-01-22
concluded: 2026-01-22
---

# Experiment: Gamma (Discount Factor) Sweep

## Hypothesis

The default gamma=0.99 may not be optimal for LunarLander; the interaction between discount factor and reward shaping could favor different values. Lower gamma with good shaping might work; higher gamma might improve long-term planning but increase variance.

**Reasoning:**
- Gamma controls how much future rewards are valued: Q(s,a) = r + γ * max Q(s',a')
- Effective horizon ≈ 1 / (1 - γ)
- For reward n steps away, agent sees γ^n of its value
- LunarLander episodes: 200-400 steps for successful landings
- The +100 landing reward is the key signal, but reward shaping provides intermediate gradients

**Value Propagation Analysis:**

| gamma | γ^100 | γ^200 | γ^300 | Effective Horizon |
|-------|-------|-------|-------|-------------------|
| 0.90 | 0.00003 | ~0 | ~0 | 10 steps |
| 0.99 | 0.37 | 0.13 | 0.05 | 100 steps |
| 0.995 | 0.61 | 0.37 | 0.22 | 200 steps |
| 0.999 | 0.90 | 0.82 | 0.74 | 1000 steps |

**Key insight:** At γ=0.9, a reward 100 steps away is worth 0.003% - effectively invisible. The agent must rely entirely on reward shaping. At γ=0.999, the agent can "see" 74% of the landing reward from 300 steps away.

## Variables

### Independent (what you're changing)
- `gamma`: [0.9, 0.99, 0.995, 0.999]

### Dependent (what you're measuring)

**Standard metrics (ALWAYS report these):**
- Success rate (%)
- Max consecutive successes
- First success episode
- Final 100 success rate (%)
- Run time (seconds)
- Total successes
- Average env reward
- Final 100 average env reward

### Controlled (what stays fixed)
- actor_lr: 0.001 (from EXP_009)
- critic_lr: 0.001 (from EXP_009)
- tau: 0.005
- batch_size: 128
- buffer_size: 16384 (from EXP_007)
- hidden_sizes: [64, 32] (from EXP_006)
- training_updates_per_episode: 25
- time_penalty: False (from EXP_012)
- altitude_bonus: True (from EXP_012)
- leg_contact: True (from EXP_012)
- stability: True (from EXP_012)
- noise_decay_episodes: 300
- episodes_per_run: 1000
- render_mode: none

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1: gamma=0.9 will fail (<10% final-100) because 10-step horizon cannot see any meaningful future reward, even with shaping
- [ ] Prediction 2: gamma=0.99 (default) will be near-optimal, as it balances horizon length with TD variance
- [ ] Prediction 3: gamma=0.995 will perform similarly to 0.99, possibly slightly better due to longer horizon
- [ ] Prediction 4: gamma=0.999 will show higher variance between runs due to longer bootstrapping chains amplifying errors

## Configuration

```json
{
  "name": "gamma_sweep",
  "type": "grid",
  "episodes_per_run": 1000,
  "num_runs_per_config": 2,
  "parameters": {
    "gamma": [0.9, 0.99, 0.995, 0.999]
  }
}
```

## Execution

- **Started:** 2026-01-22 16:30
- **Completed:** 2026-01-22 22:30
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 1000
- **Number of configurations:** 4
- **Runs per configuration:** 2
- **Total runs:** 8

## Results

### Summary Table

| Run | gamma | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|-------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | 0.9 | 0.1 | 1 | 462 | 0.0 | 2741 | 1 | -109.60 | -36.53 |
| 2 | 0.9 | 0.1 | 1 | 111 | 0.0 | 2992 | 1 | -151.97 | -69.86 |
| 3 | 0.99 | 32.5 | 15 | 119 | 61.0 | 3936 | 325 | 9.06 | 181.35 |
| 4 | 0.99 | 31.7 | 12 | 111 | 68.0 | 1888 | 317 | 45.56 | 180.37 |
| 5 | 0.995 | 6.1 | 4 | 187 | 12.0 | 4587 | 61 | -101.35 | 109.22 |
| 6 | 0.995 | 38.6 | 16 | 175 | 66.0 | 2001 | 386 | 61.29 | 200.45 |
| 7 | 0.999 | 10.4 | 4 | 222 | 15.0 | 1660 | 104 | -34.08 | 57.33 |
| 8 | 0.999 | 10.3 | 4 | 99 | 0.0 | 1672 | 103 | -25.02 | -42.49 |

### Aggregated by Config (sorted by Final 100 %)

| gamma | Effective Horizon | Avg Success % | Avg Final 100 % | Avg Time | Avg First Success |
|-------|-------------------|---------------|-----------------|----------|-------------------|
| **0.99** | 100 steps | **32.1** | **64.5** | 2912 | **115** |
| 0.995 | 200 steps | 22.4 | 39.0 | 3294 | 181 |
| 0.999 | 1000 steps | 10.4 | 7.5 | 1666 | 161 |
| 0.9 | 10 steps | 0.1 | 0.0 | 2867 | 287 |

### Best Configuration

```json
{
  "gamma": 0.99
}
```

### Key Observations

1. **gamma=0.99 (default) is clearly optimal**: 64.5% avg final-100, consistent across both runs (61%, 68%)
2. **gamma=0.9 catastrophically fails**: 0% final-100 in both runs; 10-step horizon cannot see landing reward
3. **gamma=0.995 shows extreme variance**: 12% vs 66% final-100 between runs; longer horizon amplifies learning instability
4. **gamma=0.999 underperforms with high variance**: 15% vs 0% final-100; 1000-step horizon causes TD error accumulation

### Charts

Charts for each run are saved in the `charts/` folder.

## Analysis

The results strongly validate the default gamma=0.99. The effective horizon analysis in the hypothesis proved accurate:

**Why gamma=0.9 fails completely:**
- γ^100 = 0.00003 means the +100 landing reward is invisible from 100 steps away
- Agent must rely entirely on shaped intermediate rewards, which proved insufficient
- Even with 1000 episodes, only achieved 1 success per run (likely luck)

**Why gamma=0.99 works best:**
- γ^200 = 0.13 means agent sees 13% of landing reward from typical landing distance
- Balance between seeing future rewards and not accumulating too much TD error
- Consistent performance across both runs (61%, 68%)

**Why gamma=0.995 and 0.999 show high variance:**
- Longer bootstrapping chains amplify small errors
- γ^300 = 0.22 (0.995) and 0.74 (0.999) see more of the landing reward
- But longer chains mean Q-values take longer to stabilize
- Small differences in early learning compound into large outcome differences

### Prediction Outcomes

- [x] Prediction 1: **SUPPORTED** - gamma=0.9 achieved 0% final-100 in both runs; 10-step horizon is too short
- [x] Prediction 2: **SUPPORTED** - gamma=0.99 achieved 64.5% avg final-100, clearly the best performer
- [x] Prediction 3: **REFUTED** - gamma=0.995 did NOT perform similarly to 0.99; showed 39% avg vs 64.5% with massive variance (12-66%)
- [x] Prediction 4: **SUPPORTED** - gamma=0.999 showed high variance (0-15% final-100) due to long bootstrapping chains

## Conclusion

### Hypothesis Status: **PARTIALLY SUPPORTED**

The hypothesis that gamma=0.99 might not be optimal was refuted - it IS optimal. However, the hypothesis correctly identified that gamma values interact with reward shaping and affect learning stability.

### Key Finding
gamma=0.99 (TD3 default) is optimal for LunarLander; lower values (0.9) catastrophically fail, higher values (0.995, 0.999) increase variance without improving performance.

### Implications
- Keep gamma=0.99 (the default)
- The 100-step effective horizon matches well with typical LunarLander episode lengths
- Do not increase gamma hoping for "more long-term thinking" - it backfires

### Next Steps
- [x] Confirm gamma=0.99 is already the default in config
- [ ] Consider testing gamma values between 0.95-0.99 if finer tuning desired
