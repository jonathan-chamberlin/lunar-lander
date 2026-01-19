---
id: EXP_001
title: Render Mode Comparison
status: RUNNING
created: 2026-01-19
completed:
concluded:
---

# Experiment: Render Mode Comparison

## Hypothesis

Training with `render_mode='all'` will produce significantly better success metrics than training with `render_mode='none'`.

**Reasoning:** When running simulations with rendering enabled (viewing the agent), the agent eventually achieves >=80% success rate after ~3000 episodes. However, when running without rendering (`render_mode='none'`), the agent consistently achieves <=0.2% success rate.

Key difference: When `render_mode='all'`, training uses a **single environment**. When `render_mode='none'`, training uses **8 async (AsyncVectorEnv) environments** for parallel data collection. The assumption was that async envs would speed up training with no effect on learning, but the dramatic performance difference suggests the async vectorized environment setup may be disrupting training dynamics (e.g., experience correlation, timing, state synchronization, or how experiences are batched into the replay buffer).

## Variables

### Independent (what you're changing)
- `render_mode`: 'all' vs 'none'
- `num_envs`: 1 (when rendering) vs 8 (when not rendering) - **Note: Directly dependenet on render_mode**

### Dependent (what you're measuring)
- Total number of successes
- Success rate (percentage of episodes that were successes)
- Max consecutive successes
- Env reward
- First success episode
- Final 100 success rate

### Controlled (what stays fixed)
- All hyperparameters (learning rate, batch size, buffer size, etc.)
- Number of episodes per run (3000 to match observed successful training)
- Number of environments (num_envs)
- Random seed
- Network architecture

## Predictions

Make specific, falsifiable predictions before running:

- [x] Prediction 1: `render_mode='all'` will achieve success rate >= 50% while `render_mode='none'` will achieve <= 5%
- [x] Prediction 2: `render_mode='all'` will achieve first success before episode 500, while `render_mode='none'` may never achieve first success
- [x] Prediction 3: `render_mode='all'` will achieve max consecutive successes >= 10, while `render_mode='none'` will achieve <= 2

## Configuration

```json
{
  "name": "render_mode_comparison",
  "type": "grid",
  "episodes_per_run": 3000,
  "parameters": {
    "render_mode": ["all", "none"]
  }
}
```

## Execution

- **Started:** 2026-01-19 13:30
- **Completed:**
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 3000
- **Number of configurations:** 2
- **Total runs:** 2

## Results

<!-- Fill after experiment completes -->

### Summary Table

| Config | Total Successes | Success Rate | Max Consecutive | Env Reward | First Success | Final 100 |
|--------|-----------------|--------------|-----------------|------------|---------------|-----------|
| render_mode='all' | | | | | | |
| render_mode='none' | | | | | | |

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
- `run_001_chart.png` (render_mode='all')
- `run_002_chart.png` (render_mode='none')

## Analysis

[What patterns emerged? Any surprises? How do results compare to predictions?]

### Prediction Outcomes

- [ ] Prediction 1: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 2: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 3: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]

## Conclusion

### Hypothesis Status: **SUPPORTED / REFUTED / INCONCLUSIVE**

### Key Finding
[One sentence summarizing the main takeaway]

### Implications
[What does this mean for future training? If render_mode affects results, investigate root cause - is it the rendering itself, the number of environments, or the async execution?]

### Next Steps
- [ ] If hypothesis supported: Run follow-up experiment to disentangle confounded variables (test num_envs=1 with render_mode='none')
- [ ] Investigate AsyncVectorEnv behavior - are experiences being collected/stored correctly?
- [ ] Check if replay buffer is handling multi-env experiences properly
- [ ] Consider whether this indicates a bug in the vectorized environment code path
