---
id: EXP_004
title: Batch Size vs Training Speed
status: PLANNED
created: 2026-01-20
completed:
concluded:
---

# Experiment: Batch Size vs Training Speed

## Hypothesis

Smaller batch sizes may improve learning in RL by providing more frequent, noisier gradient updates that help escape local minima, while larger batches may improve wall-clock speed through better GPU utilization.

**Reasoning:**
- Unlike supervised learning, larger batches aren't always better in RL
- Smaller batches = more frequent updates = faster adaptation to new experiences
- Current default (128) is mid-range; testing both smaller and larger
- 512+ is likely overkill for this simple 8-state environment

## Variables

### Independent (what you're changing)
- batch_size: [32, 64, 128, 256]

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
- All other hyperparameters (learning rate, buffer size, tau, etc.)
- Network architecture ([256, 128])
- Number of episodes per run (500)
- Noise decay schedule (300 episodes)

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1: Batch size 64 will achieve the highest final 100 success rate (sweet spot between update frequency and gradient quality)
- [ ] Prediction 2: Batch size 32 will achieve first success earliest (most frequent updates)
- [ ] Prediction 3: Batch size 256 will have the fastest wall-clock time per episode but lower final success rate

## Configuration

```json
{
  "name": "batch_size_performance",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "batch_size": [32, 64, 128, 256]
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

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|------|-----------------|------------|------------------|
| 1 | batch_size=32 | | | | | | | | |
| 2 | batch_size=32 | | | | | | | | |
| 3 | batch_size=64 | | | | | | | | |
| 4 | batch_size=64 | | | | | | | | |
| 5 | batch_size=128 | | | | | | | | |
| 6 | batch_size=128 | | | | | | | | |
| 7 | batch_size=256 | | | | | | | | |
| 8 | batch_size=256 | | | | | | | | |

### Aggregated by Config

| Config | Avg Success % | Avg Final 100 % | Avg Time | Avg First Success |
|--------|---------------|-----------------|----------|-------------------|
| batch_size=32 | | | | |
| batch_size=64 | | | | |
| batch_size=128 | | | | |
| batch_size=256 | | | | |

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
- `run_001_chart.png` (batch_size=32 #1)
- `run_002_chart.png` (batch_size=32 #2)
- `run_003_chart.png` (batch_size=64 #1)
- `run_004_chart.png` (batch_size=64 #2)
- `run_005_chart.png` (batch_size=128 #1)
- `run_006_chart.png` (batch_size=128 #2)
- `run_007_chart.png` (batch_size=256 #1)
- `run_008_chart.png` (batch_size=256 #2)

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
[What does this mean for future training?]

### Next Steps
- [ ]
- [ ]
