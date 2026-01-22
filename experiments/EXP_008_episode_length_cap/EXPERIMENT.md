---
id: EXP_008
title: Episode Length Cap
status: RUNNING
created: 2026-01-21
completed:
concluded:
---

# Experiment: Episode Length Cap

## Hypothesis

Capping episodes at 400-600 steps will save compute without hurting learning, since successful landings typically complete in 200-400 steps.

**Reasoning:**
- Default max is 1000 steps (gymnasium default)
- Skilled landings complete in 200-400 steps
- Failed episodes (floating, crashing slowly) waste compute going to 1000
- Cap too aggressive (<400) may terminate potentially successful attempts
- 400 is minimum viable; 600 provides safety margin

## Variables

### Independent (what you're changing)
- `max_episode_steps`: [400, 600, 800, 1000]

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
- batch_size: 32 (from EXP_004)
- buffer_size: 16384 (from EXP_007)
- actor_lr: 0.001
- critic_lr: 0.002
- tau: 0.005
- hidden_sizes: [256, 128]
- noise_decay_episodes: 300
- training_updates_per_episode: 10 (from EXP_005)
- render_mode: none

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1: max_episode_steps=400 will have the fastest wall-clock time per run
- [ ] Prediction 2: max_episode_steps=400 will have slightly lower success rate than 600+ due to premature terminations
- [ ] Prediction 3: max_episode_steps=600 will be the best tradeoff (similar success to 1000 but faster)
- [ ] Prediction 4: max_episode_steps=800 and 1000 will have similar success rates (extra steps rarely help)

## Configuration

```json
{
  "name": "episode_length_cap",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "max_episode_steps": [400, 600, 800, 1000]
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

| Run | max_episode_steps | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|-------------------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | 400 | | | | | | | | |
| 2 | 400 | | | | | | | | |
| 3 | 600 | | | | | | | | |
| 4 | 600 | | | | | | | | |
| 5 | 800 | | | | | | | | |
| 6 | 800 | | | | | | | | |
| 7 | 1000 | | | | | | | | |
| 8 | 1000 | | | | | | | | |

### Aggregated by Config

| max_episode_steps | Avg Success % | Avg Final 100 % | Avg Time (s) | Avg First Success |
|-------------------|---------------|-----------------|--------------|-------------------|
| 400 | | | | |
| 600 | | | | |
| 800 | | | | |
| 1000 | | | | |

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
- `run_001_chart.png` through `run_008_chart.png`

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
