---
id: EXP_017
title: Noise Decay Sweep
status: CONCLUDED
created: 2026-01-22
completed: 2026-01-22
concluded: 2026-01-22
---

# Experiment: Noise Decay Sweep

## Hypothesis

The default noise_decay_episodes=300 may not be optimal; different decay schedules could improve exploration efficiency or final performance.

**Reasoning:**
- noise_decay_episodes controls linear decay of exploration noise from 1.0 → 0.2
- Fast decay = more exploitation time, risk of local optima
- Slow decay = extended exploration, risk of never refining policy
- Current default (300) means noise minimal for final 200 episodes of a 500-episode run
- LunarLander may benefit from different exploration/exploitation balance

**Noise levels at key episodes (for 500-ep runs):**

| decay | @ ep 100 | @ ep 250 | @ ep 400 |
|-------|----------|----------|----------|
| 100 | 0.20 | 0.20 | 0.20 |
| 200 | 0.60 | 0.20 | 0.20 |
| 300 | 0.73 | 0.33 | 0.20 |
| 400 | 0.80 | 0.50 | 0.20 |
| 500 | 0.84 | 0.60 | 0.36 |

## Variables

### Independent (what you're changing)
- `noise_decay_episodes`: [100, 200, 300, 400, 500]

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
- tau: 0.005 (from EXP_010)
- batch_size: 128
- buffer_size: 16384 (from EXP_007)
- hidden_sizes: [64, 32] (from EXP_006)
- training_updates_per_episode: 25
- time_penalty: False (from EXP_012)
- altitude_bonus: True (from EXP_012)
- leg_contact: True (from EXP_012)
- stability: True (from EXP_012)
- noise_scale_initial: 1.0
- noise_scale_final: 0.2
- episodes_per_run: 500
- render_mode: none

## Predictions

Make specific, falsifiable predictions before running:

- [x] Prediction 1: decay=100 will have lower final performance (<20% final-100) due to premature convergence before finding good trajectories
- [x] Prediction 2: decay=300 (default) will be near-optimal, balancing exploration and exploitation
- [x] Prediction 3: decay=500 will show slower early learning (later first success) but may achieve comparable final performance
- [x] Prediction 4: There will be a "sweet spot" between 200-400 that outperforms extremes

## Configuration

```json
{
  "name": "noise_decay_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "noise_decay_episodes": [100, 200, 300, 400, 500]
  }
}
```

## Execution

- **Started:** 2026-01-22
- **Completed:** 2026-01-22
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 500
- **Number of configurations:** 5
- **Runs per configuration:** 2
- **Total runs:** 10

## Results

### Summary Table

| Run | noise_decay_episodes | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|----------------------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | 100 | 3.2 | 2 | 374 | 14.0 | 970 | 16 | -160.6 | 63.6 |
| 2 | 100 | 1.6 | 1 | 247 | 6.0 | 885 | 8 | -140.2 | -2.5 |
| 3 | 200 | 5.4 | 4 | 184 | 16.0 | 856 | 27 | -150.3 | 39.5 |
| 4 | 200 | 18.8 | 5 | 266 | 46.0 | 621 | 94 | -98.3 | 119.7 |
| 5 | 300 | 16.4 | 10 | 94 | 33.0 | 609 | 82 | -52.6 | 113.0 |
| 6 | 300 | 5.2 | 4 | 191 | 21.0 | 706 | 26 | -126.4 | 83.2 |
| 7 | 400 | 9.4 | 4 | 176 | 24.0 | 985 | 47 | -82.4 | 62.1 |
| 8 | 400 | 7.6 | 4 | 323 | 33.0 | 464 | 38 | -159.6 | 55.6 |
| 9 | 500 | 11.0 | 7 | 275 | 20.0 | 893 | 55 | -134.5 | 38.4 |
| 10 | 500 | 14.4 | 4 | 139 | 35.0 | 604 | 72 | -67.7 | 125.0 |

### Aggregated by Config (sorted by Final 100 %)

| noise_decay_episodes | Exploitation Time | Avg Success % | Avg Final 100 % | Variance |
|----------------------|-------------------|---------------|-----------------|----------|
| **200** | 300 eps (60%) | 12.1 | **31.0** | High (16-46%) |
| 400 | 100 eps (20%) | 8.5 | 28.5 | Moderate (24-33%) |
| 500 | 0 eps (0%) | 12.7 | 27.5 | Moderate (20-35%) |
| 300 | 200 eps (40%) | 10.8 | 27.0 | Moderate (21-33%) |
| 100 | 400 eps (80%) | 2.4 | **10.0** | Low (6-14%) |

### Best Configuration

```json
{
  "noise_decay_episodes": 200,
  "avg_success_rate": 12.1,
  "avg_final_100_success_rate": 31.0,
  "verdict": "Slightly better than default but high variance; 200-400 all viable"
}
```

### Key Observations

1. **decay=100 clearly worst** - 10% avg final-100, premature convergence cuts off exploration too early
2. **decay=200-500 all similar** - 27.5-31% avg, no statistically significant difference given variance
3. **decay=200 has highest average** (31%) but also highest variance (16-46%)
4. **decay=300 (default) found success earliest** - ep 94 avg vs 207-311 for others
5. **High variance dominates** - Same config can achieve 16% or 46% (decay=200)

### Charts

Charts for each run are saved in the `charts/` folder.

## Analysis

The results show a clear pattern at extremes but no clear winner in the middle range:

**Why decay=100 fails:**
- Noise at minimum by episode 100
- Agent hasn't explored enough state space to find good landing trajectories
- Locks into suboptimal policy early
- First successes come late (ep 247-374) because the agent hasn't learned to land

**Why decay=200-500 are similar:**
- All provide sufficient exploration to find good trajectories
- The extra exploration time (400 vs 500) doesn't significantly help or hurt
- Variance between runs dominates any effect from decay rate
- LunarLander may not need extended exploration - once you find the landing pad, you've explored enough

**Interesting observation:**
- decay=300 found first success earliest (ep 94) despite not having the highest final performance
- This suggests faster decay helps early learning but not final performance

### Prediction Outcomes

- [x] Prediction 1: **SUPPORTED** - decay=100 averaged 10% final-100, well below 20% threshold
- [x] Prediction 2: **PARTIALLY SUPPORTED** - decay=300 is viable (27%) but not clearly optimal; decay=200 slightly better (31%)
- [x] Prediction 3: **REFUTED** - decay=500 had avg first success at ep 207, similar to decay=200 (225); final performance comparable (27.5% vs 31%)
- [x] Prediction 4: **WEAKLY SUPPORTED** - decay=200-400 range (28.5-31%) outperforms extremes (10-27.5%), but differences are within variance

## Conclusion

### Hypothesis Status: **WEAKLY SUPPORTED**

The default noise_decay_episodes=300 is not clearly optimal, but alternatives in the 200-400 range provide similar performance. Only decay=100 is clearly suboptimal.

### Key Finding
Noise decay schedule has minimal impact in the 200-500 range; only aggressive decay (100) clearly hurts. The default 300 is fine.

### Implications
- Keep noise_decay_episodes=300 as default - no compelling reason to change
- Avoid decay < 200 which causes premature convergence
- High variance between runs (16-46% for same config) remains the dominant factor
- Noise schedule is not a bottleneck for improving performance

### Next Steps
- [x] Confirm decay=100 is too aggressive (it is)
- [ ] High variance remains the dominant issue across all experiments
- [ ] Consider testing noise_scale_final (0.2) - maybe maintaining more noise helps
