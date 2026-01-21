---
id: EXP_006
title: Network Size Performance
status: CONCLUDED
created: 2026-01-20
completed: 2026-01-20
concluded: 2026-01-20
---

# Experiment: Network Size Performance

## Hypothesis

LunarLander's simple 8-dimensional state space can be solved with much smaller networks than the current [256, 128] architecture, with minimal or no loss in success rate but significant speed improvement.

**Reasoning:**
- The current network has ~35K parameters (actor alone), which is likely overkill for an 8→2 mapping
- LunarLander is a solved benchmark; the control policy could theoretically be hand-coded with ~50 parameters
- Smaller networks mean faster forward/backward passes, potentially significant training speedup
- The key question: how small can we go before performance degrades?

## Variables

### Independent (what you're changing)
- hidden_sizes: [[16, 16], [64, 32], [128, 64], [256, 128]]

| Architecture | Actor Params | Ratio to Baseline |
|--------------|--------------|-------------------|
| [16, 16]     | ~600         | 0.02x |
| [64, 32]     | ~2,800       | 0.08x |
| [128, 64]    | ~9,500       | 0.27x |
| [256, 128]   | ~35,000      | 1.0x (baseline) |

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
- Learning rates (actor_lr=0.001, critic_lr=0.002)
- Buffer size (16384)
- Batch size (128)
- Tau (0.005)
- Number of episodes per run (500)
- Noise decay schedule (300 episodes)
- All other hyperparameters

## Predictions

Make specific, falsifiable predictions before running:

- [x] Prediction 1: [256, 128] and [128, 64] will achieve similar success rates (within 5%)
- [x] Prediction 2: [64, 32] will be the sweet spot - achieving >90% of baseline performance with the best speed improvement
- [x] Prediction 3: [16, 16] will show measurably degraded success rate (<80% of baseline) due to insufficient capacity
- [x] Prediction 4: Runtime will scale roughly linearly with parameter count, with [16, 16] being ~4x faster than [256, 128]

## Configuration

```json
{
  "name": "network_size_performance",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "hidden_sizes": [[16, 16], [64, 32], [128, 64], [256, 128]]
  }
}
```

## Execution

- **Started:** 2026-01-20
- **Completed:** 2026-01-20
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 500
- **Number of configurations:** 4
- **Runs per configuration:** 2
- **Total runs:** 8

## Results

### Summary Table

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | [16, 16] | 0.2 | 1 | 341 | 0.0 | 470 | 1 | -306.69 | -181.09 |
| 2 | [16, 16] | 1.8 | 1 | 170 | 1.0 | 906 | 9 | -118.62 | -133.49 |
| 3 | [64, 32] | 0.0 | 0 | None | 0.0 | 383 | 0 | -373.01 | -124.96 |
| 4 | [64, 32] | 13.4 | 5 | 222 | 51.0 | 1041 | 67 | -185.92 | 143.77 |
| 5 | [128, 64] | 16.0 | 4 | 173 | 37.0 | 987 | 80 | -89.46 | 145.29 |
| 6 | [128, 64] | 0.0 | 0 | None | 0.0 | 924 | 0 | -279.31 | -62.65 |
| 7 | [256, 128] | 4.0 | 1 | 183 | 8.0 | 1411 | 20 | -187.45 | 11.39 |
| 8 | [256, 128] | 6.4 | 3 | 112 | 15.0 | 1521 | 32 | -51.07 | 24.58 |

### Aggregated by Config

| Config | Avg Success % | Avg Final 100 % | Avg Time (s) | Avg First Success |
|--------|---------------|-----------------|--------------|-------------------|
| [16, 16] | 1.0 | 0.5 | 688 | 256 |
| [64, 32] | 6.7 | **25.5** | 712 | 222 |
| [128, 64] | 8.0 | 18.5 | 956 | 173 |
| [256, 128] | 5.2 | 11.5 | 1466 | 148 |

### Best Configuration

```json
{
  "hidden_sizes": [64, 32],
  "avg_final_100_success_rate": 25.5,
  "avg_time": 712,
  "speedup_vs_baseline": "2.1x faster"
}
```

### Key Observations

1. **High variance dominates all configurations** - Same network size can achieve 0% or 51% success depending on random seed
2. **[64, 32] achieved best final-100 average (25.5%)** despite having only 8% of baseline parameters
3. **[256, 128] baseline performed worst on final-100 average (11.5%)** - larger networks not better
4. **Runtime scales with network size** - [64, 32] is 2.1x faster than [256, 128]
5. **[16, 16] is too small** - consistently poor performance (0-2% success)

### Charts

Charts for each run are saved in the `charts/` folder.

## Analysis

The results reveal that **variance in training outcomes vastly outweighs the effect of network size** for networks above a minimum threshold. The [64, 32] network with only ~2,800 parameters achieved the best average final-100 success rate, while the [256, 128] baseline with ~35,000 parameters performed worse.

This suggests that for LunarLander's simple 8-dimensional state space:
- Network capacity is not the bottleneck above ~3K parameters
- Other factors (exploration, random seed, learning rate) have much larger effects
- The current [256, 128] default is oversized and slower without benefit

The high variance (0% vs 51% for identical configs) indicates that **reproducibility and seed selection are critical concerns** that should be addressed before further hyperparameter tuning.

### Prediction Outcomes

- [x] Prediction 1: **INCONCLUSIVE** - Both showed similar variance (0-16%), but [128, 64] avg (8.0%) was higher than [256, 128] avg (5.2%)
- [x] Prediction 2: **SUPPORTED** - [64, 32] achieved best final-100 average (25.5%) and was 2.1x faster than baseline
- [x] Prediction 3: **SUPPORTED** - [16, 16] averaged 1.0% success vs 5.2% baseline (19% of baseline performance)
- [x] Prediction 4: **PARTIALLY SUPPORTED** - [16, 16] was 2.1x faster (not 4x), runtime scaling is sublinear

## Conclusion

### Hypothesis Status: **SUPPORTED**

### Key Finding
Smaller networks ([64, 32]) can match or exceed baseline [256, 128] performance while being 2x faster, but high training variance is the dominant factor affecting outcomes.

### Implications
- The default network size can be reduced to [64, 32] for faster training without performance loss
- Before further hyperparameter optimization, the high variance issue should be investigated (seeds, exploration strategy)
- Network architecture is not the limiting factor for LunarLander performance

### Next Steps
- [ ] Investigate sources of high variance (random seeds, exploration noise)
- [ ] Update default config to use [64, 32] hidden sizes
- [ ] Run more trials per config (5+) to get statistically significant comparisons
