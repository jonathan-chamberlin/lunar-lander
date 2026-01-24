---
id: EXP_023
title: Long Training with Video Recording
status: PLANNED
created: 2026-01-23
completed:
concluded:
---

# Experiment: Long Training with Video Recording

## Hypothesis

A 10,000 episode training run with noise decay to 0 over 5000 episodes will:
1. Achieve 100+ consecutive successes (replication of EXP_022)
2. Successfully record videos at all milestone episodes (unlike EXP_022 where recording failed)
3. Show consistent high performance in the pure exploitation phase (episodes 5000+)

## Variables

### Independent (what you're changing)
- Total episodes: 10,000 (vs 15,000 in EXP_022)
- Diagnostics batch size: 200 (vs 100 default)
- Recording milestones: 1-5, 40-50, 100-103, 300-303, 500-505, 1000-1005, 2000-2005, 3000-3004, 4000-4004, 5000-5004, 6000-6004, 7000-7004, 8000-8004, 9000-9004, 9996-10000

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

**Additional metrics:**
- Number of videos successfully recorded
- Memory usage patterns
- Model checkpoint sizes

### Controlled (what stays fixed)
- Noise: 1.0 -> 0.0 over 5000 episodes (same as EXP_022)
- Network: [256, 128] hidden sizes
- LR: 0.001 for both actor and critic
- Batch size: 128 (training batch)
- Buffer size: 16384
- All other hyperparameters same as EXP_022

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1: Will achieve 100+ consecutive successes in episodes 5000-10000
- [ ] Prediction 2: Videos will be recorded at all milestone episodes (fix verification)
- [ ] Prediction 3: Final 100 success rate will exceed 80%
- [ ] Prediction 4: Max consecutive will occur shortly after noise reaches 0 (around ep 5000-6000)

## Configuration

```python
# Key settings
num_episodes = 10000
noise_scale_initial = 1.0
noise_scale_final = 0.0
noise_decay_episodes = 5000
diagnostics_batch_size = 200
checkpoint_interval = 1000  # Model saves every 1000 episodes

# Recording episodes (total: 76 episodes)
record_episodes = (
    1-5, 40-50, 100-103, 300-303, 500-505,
    1000-1005, 2000-2005, 3000-3004, 4000-4004,
    5000-5004, 6000-6004, 7000-7004, 8000-8004,
    9000-9004, 9996-10000
)
```

## Execution

- **Started:**
- **Completed:**
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 10,000
- **Number of configurations:** 1
- **Runs per configuration:** 1
- **Total runs:** 1

## Results

<!-- Fill after experiment completes -->

### Summary Table

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|------|-----------------|------------|------------------|
| 1 | baseline | | | | | | | | |

### Model Checkpoints

| Checkpoint | Episode | Path |
|------------|---------|------|
| checkpoint_ep1000 | 1000 | results/models/checkpoint_ep1000 |
| checkpoint_ep2000 | 2000 | results/models/checkpoint_ep2000 |
| ... | ... | ... |
| final_model | 10000 | results/models/exp023_model |

### Videos Recorded

| Milestone | Episodes | Videos Created | Notes |
|-----------|----------|----------------|-------|
| Early learning | 1-5 | | |
| Early progress | 40-50 | | |
| 100 mark | 100-103 | | |
| 300 mark | 300-303 | | |
| 500 mark | 500-505 | | |
| 1000 mark | 1000-1005 | | |
| 2000 mark | 2000-2005 | | |
| 3000 mark | 3000-3004 | | |
| 4000 mark | 4000-4004 | | |
| Exploitation start | 5000-5004 | | |
| 6000 mark | 6000-6004 | | |
| 7000 mark | 7000-7004 | | |
| 8000 mark | 8000-8004 | | |
| 9000 mark | 9000-9004 | | |
| Final | 9996-10000 | | |

### Key Observations

1.
2.
3.

### Charts

Charts for each run are saved in the `charts/` folder:
- `chart_episode_*.png` - periodic progress charts
- `chart_final_10000.png` - final training chart

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
- [ ] Analyze video quality at different training stages
- [ ] Compare learning curves to EXP_022
- [ ] Test reproducibility with different random seeds
