---
id: EXP_013
title: Gamma Sweep
status: PLANNED
created: 2026-01-22
completed:
concluded:
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
- episodes_per_run: 500
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
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "gamma": [0.9, 0.99, 0.995, 0.999]
  }
}
```

## Execution

- **Started:**
- **Completed:**
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 500
- **Number of configurations:** 4
- **Runs per configuration:** 2
- **Total runs:** 8

## Results

<!-- Fill after experiment completes -->

### Summary Table

| Run | gamma | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|-------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |

### Aggregated by Config (sorted by Final 100 %)

| gamma | Effective Horizon | Avg Success % | Avg Final 100 % | Avg Time | Avg First Success |
|-------|-------------------|---------------|-----------------|----------|-------------------|
| | | | | | |

### Best Configuration

```json
{
}
```

### Key Observations

1.
2.
3.

### Charts

Charts for each run are saved in the `charts/` folder.

## Analysis

[What patterns emerged? Any surprises? How do results compare to predictions?]

### Prediction Outcomes

- [ ] Prediction 1: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 2: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 3: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 4: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]

## Conclusion

### Hypothesis Status: **SUPPORTED / REFUTED / INCONCLUSIVE**

### Key Finding
[One sentence summarizing the main takeaway]

### Implications
[What does this mean for future training?]

### Next Steps
- [ ]
- [ ]
