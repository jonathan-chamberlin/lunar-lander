---
id: EXP_005
title: Update Frequency
status: RUNNING
created: 2026-01-20
completed:
concluded:
---

# Experiment: Update Frequency

## Hypothesis

Updating every 2-4 steps instead of every step will improve wall-clock speed with minimal impact on sample efficiency.

**Reasoning:**
- update_every=1 means gradient computation after every env step (expensive)
- Collecting multiple transitions before updating reduces overhead
- Too infrequent (16+) wastes collected data without learning
- Sweet spot likely 2-4 based on literature

## Variables

### Independent (what you're changing)
- `update_every_n_steps`: [1, 2, 4, 8]

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
- batch_size: 128
- buffer_size: 16384
- actor_lr: 0.001
- critic_lr: 0.002
- tau: 0.005
- hidden_sizes: [256, 128]
- noise_decay_episodes: 300
- max_episode_steps: 1000
- episodes_per_run: 500
- render_mode: none

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1: update_every_n_steps=2 or 4 will have the fastest wall-clock time (lowest run time in seconds)
- [ ] Prediction 2: update_every_n_steps=1 will have the highest success rate (more frequent updates = better learning)
- [ ] Prediction 3: update_every_n_steps=8 will have the worst success rate (too infrequent updates)
- [ ] Prediction 4: Wall-clock speedup from 1 to 4 will be >30% with <10% success rate degradation

## Configuration

```json
{
  "name": "update_frequency",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "update_every_n_steps": [1, 2, 4, 8]
  }
}
```

## Execution

- **Started:** 2026-01-20
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

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|------|-----------------|------------|------------------|
| 1 | update_every_n_steps=1 | | | | | | | | |
| 2 | update_every_n_steps=1 | | | | | | | | |
| 3 | update_every_n_steps=2 | | | | | | | | |
| 4 | update_every_n_steps=2 | | | | | | | | |
| 5 | update_every_n_steps=4 | | | | | | | | |
| 6 | update_every_n_steps=4 | | | | | | | | |
| 7 | update_every_n_steps=8 | | | | | | | | |
| 8 | update_every_n_steps=8 | | | | | | | | |

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

Charts for each run are saved in the `charts/` folder:
- `run_001_chart.png`
- `run_002_chart.png`
- ...

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
