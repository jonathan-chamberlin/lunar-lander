---
id: EXP_021
title: Long Training Run
status: RUNNING
created: 2026-01-22
completed:
concluded:
---

# Experiment: Long Training Run

## Hypothesis

Current 500-1000 episode runs may not reach full potential; longer training (3000 episodes) could show continued improvement beyond the ~65% final-100 ceiling observed in previous experiments, or reveal a true performance plateau.

**Reasoning:**
- Best configs in previous experiments achieve ~50-70% final-100 success
- Unknown if this is the true ceiling or just insufficient training time
- Longer runs reveal: (a) continued improvement, (b) plateau, or (c) catastrophic forgetting
- 3000 episodes = 6x longer than standard 500-episode runs
- With optimal hyperparameters from EXP_004-013, agent has best chance of reaching peak performance

## Variables

### Independent (what you're changing)
- episodes_per_run: 3000 (vs standard 500)

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

**Additional metrics for long runs:**
- Success rate at episode 500, 1000, 1500, 2000, 2500, 3000 (learning curve)
- Whether performance plateaus, continues improving, or degrades

### Controlled (what stays fixed)
- batch_size: 32 (from EXP_004 - fastest learning)
- buffer_size: 16384 (from EXP_007)
- actor_lr: 0.001 (from EXP_009)
- critic_lr: 0.001 (from EXP_009)
- tau: 0.005 (from EXP_010)
- gamma: 0.99 (from EXP_013)
- hidden_sizes: [256, 128]
- training_updates_per_episode: 25
- time_penalty: False (from EXP_012)
- altitude_bonus: True
- leg_contact: True
- stability: True
- noise_decay_episodes: 300
- render_mode: none

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1: Final-100 success rate will exceed 70% (surpassing best short-run results)
- [ ] Prediction 2: Performance will plateau by episode 2000 (diminishing returns after that)
- [ ] Prediction 3: No catastrophic forgetting will occur (performance won't drop below 50% after reaching peak)
- [ ] Prediction 4: Max consecutive successes will exceed 20 (longer runs allow more chances for streaks)

## Configuration

```json
{
  "name": "long_training",
  "type": "grid",
  "episodes_per_run": 3000,
  "num_runs_per_config": 3,
  "parameters": {
    "batch_size": [32]
  }
}
```

## Execution

- **Started:** 2026-01-22
- **Completed:**
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 3000
- **Number of configurations:** 1
- **Runs per configuration:** 3
- **Total runs:** 3

## Results

<!-- Fill after experiment completes -->

### Summary Table

| Run | Episodes | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|----------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | 3000 | | | | | | | | |
| 2 | 3000 | | | | | | | | |
| 3 | 3000 | | | | | | | | |

### Learning Curve (Success Rate by Episode Milestone)

| Run | ep 500 | ep 1000 | ep 1500 | ep 2000 | ep 2500 | ep 3000 |
|-----|--------|---------|---------|---------|---------|---------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

### Aggregated

| Metric | Run 1 | Run 2 | Run 3 | Average |
|--------|-------|-------|-------|---------|
| Success % | | | | |
| Final 100 % | | | | |
| Max Consec | | | | |
| Total Successes | | | | |

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
