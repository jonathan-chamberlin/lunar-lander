---
id: EXP_010
title: Tau Sweep
status: PLANNED
created: 2026-01-22
completed:
concluded:
---

# Experiment: Tau Sweep

## Hypothesis

The default tau=0.005 (TD3 paper) may not be optimal for our specific setup; a different soft update coefficient could improve learning stability or speed.

**Reasoning:**
- Tau controls how quickly target networks track online networks: θ_target = τ * θ_online + (1 - τ) * θ_target
- Small tau = stable but potentially stale targets; Large tau = up-to-date but potentially unstable
- Our setup uses PER (higher gradient variance) and small networks [64,32] (faster adaptation)
- With 25 updates/episode, tau=0.005 gives ~5.5 episode half-life for target absorption
- LunarLander is relatively simple; may tolerate faster target tracking than complex environments

## Variables

### Independent (what you're changing)
- `tau`: [0.001, 0.005, 0.01, 0.05]

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

- [ ] Prediction 1: tau=0.05 will be unstable with significantly worse performance (<20% final-100) due to moving target problem
- [ ] Prediction 2: tau=0.001 will learn more slowly (later first success) but may achieve comparable final performance
- [ ] Prediction 3: An intermediate value (0.003-0.01) will outperform the default tau=0.005
- [ ] Prediction 4: The optimal tau will show lower variance between runs compared to extreme values

## Configuration

```json
{
  "name": "tau_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "tau": [0.001, 0.005, 0.01, 0.05]
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

| Run | tau | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|-----|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |

### Aggregated by Config (sorted by Final 100 %)

| tau | Avg Success % | Avg Final 100 % | Avg Time | Avg First Success |
|-----|---------------|-----------------|----------|-------------------|
| | | | | |

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
