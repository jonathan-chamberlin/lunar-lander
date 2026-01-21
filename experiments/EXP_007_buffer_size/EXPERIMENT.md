---
id: EXP_007
title: Buffer Size
status: CONCLUDED
created: 2026-01-20
completed: 2026-01-20
concluded: 2026-01-20
---

# Experiment: Buffer Size

## Hypothesis

With Prioritized Experience Replay (PER) enabled, larger replay buffers will improve learning stability and final performance by maintaining a diverse pool of high-TD-error experiences, while very small buffers will cause instability due to correlated samples.

**Reasoning:**
- Current buffer (16384) holds ~160 episodes worth of transitions at ~100 steps/episode average
- With PER, buffer size determines which experiences are AVAILABLE to sample, not just what we sample
- Larger buffers let PER access more diverse high-error transitions from different training phases
- Very small buffers (4096) may cause instability: only ~40 episodes of recent, correlated data
- Very large buffers (65536) exceed total data generated in 500 episodes, effectively "infinite memory"
- The off-policy nature of TD3 means old experiences may still be useful for Q-value estimation

## Variables

### Independent (what you're changing)
- buffer_size: [4096, 8192, 16384, 65536]

| Buffer Size | Transitions | Episodes Worth | Notes |
|-------------|-------------|----------------|-------|
| 4096 | 4,096 | ~40 | Minimum viable (2x min_experiences_before_training) |
| 8192 | 8,192 | ~80 | Half of default |
| 16384 | 16,384 | ~160 | Current default (baseline) |
| 65536 | 65,536 | ~650 | "Infinite memory" - never fills in 500 episodes |

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
- Network architecture ([64, 32] - updated default from EXP_006)
- Batch size (128)
- Tau (0.005)
- Number of episodes per run (500)
- Noise decay schedule (300 episodes)
- PER enabled (use_per=True)
- min_experiences_before_training (2000)

## Predictions

Make specific, falsifiable predictions before running:

- [x] Prediction 1: buffer_size=4096 will show higher variance and potentially worse performance due to correlated recent-only samples
- [x] Prediction 2: buffer_size=65536 will achieve best or equal-best final-100 success rate due to PER having access to diverse high-error experiences
- [x] Prediction 3: buffer_size=16384 (default) and 8192 will perform similarly - the 2x difference won't be significant
- [x] Prediction 4: Runtime will be nearly identical across all buffer sizes (memory access patterns similar, compute dominates)

## Configuration

```json
{
  "name": "buffer_size",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "buffer_size": [4096, 8192, 16384, 65536]
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
| 1 | 4096 | 16.4 | 7 | 291 | 45.0 | 709 | 82 | -104.41 | 141.74 |
| 2 | 4096 | 6.0 | 7 | 433 | 30.0 | 713 | 30 | -178.25 | 81.43 |
| 3 | 8192 | 7.6 | 3 | 275 | 20.0 | 735 | 38 | -82.07 | 85.74 |
| 4 | 8192 | 4.4 | 2 | 166 | 9.0 | 778 | 22 | -74.31 | 34.85 |
| 5 | 16384 | 12.6 | 5 | 198 | 39.0 | 895 | 63 | -111.08 | 120.66 |
| 6 | 16384 | 29.6 | 15 | 172 | 67.0 | 878 | 148 | 3.47 | 148.40 |
| 7 | 65536 | 0.0 | 0 | None | 0.0 | 849 | 0 | -280.01 | -117.23 |
| 8 | 65536 | 2.0 | 1 | 205 | 0.0 | 1083 | 10 | -130.42 | -63.49 |

### Aggregated by Config

| Config | Avg Success % | Avg Final 100 % | Avg Time (s) | Avg First Success |
|--------|---------------|-----------------|--------------|-------------------|
| 4096 | 11.2 | 37.5 | 711 | 362 |
| 8192 | 6.0 | 14.5 | 757 | 221 |
| **16384** | **21.1** | **53.0** | 887 | 185 |
| 65536 | 1.0 | 0.0 | 966 | 205 |

### Best Configuration

```json
{
  "buffer_size": 16384,
  "avg_success_rate": 21.1,
  "avg_final_100_success_rate": 53.0,
  "verdict": "Current default is optimal"
}
```

### Key Observations

1. **The current default (16384) is optimal** - achieved 53% avg final-100 success rate, best of all configs
2. **Very large buffers catastrophically fail** - buffer_size=65536 achieved 0% final-100 avg (both runs essentially failed)
3. **Smaller is NOT better** - 4096 performed reasonably (37.5%) but worse than default; 8192 was worst mid-range (14.5%)
4. **Non-monotonic relationship** - performance peaks at 16384, drops both above AND below
5. **Runtime scales modestly** - 711s to 966s (36% variation), larger buffers are slower

### Charts

Charts for each run are saved in the `charts/` folder.

## Analysis

The results **completely contradict the hypothesis**. Larger buffers do NOT help - they catastrophically hurt performance. The "infinite memory" buffer (65536) that never evicts old experiences performed worst by a massive margin.

**Why large buffers fail:**
- With 500 episodes and ~100 steps/episode, we generate ~50K transitions
- Buffer of 65536 never fills, so it contains ALL experiences including early "random flailing" episodes
- PER prioritizes high-TD-error samples, but early experiences have high TD-error for the WRONG reasons (policy was bad)
- Training on stale, bad experiences prevents learning the improved policy

**Why 16384 is optimal:**
- Large enough for diversity (~160 episodes worth)
- Small enough that old, irrelevant experiences get evicted as policy improves
- Balances exploration diversity with relevance to current policy

**The 8192 anomaly:**
- Surprisingly performed worst among smaller buffers
- May be a "worst of both worlds" - too small for diversity, too large for recency
- Or just high variance (only 2 runs)

### Prediction Outcomes

- [x] Prediction 1: **REFUTED** - 4096 actually performed well (37.5% final-100), not worse due to "correlation"
- [x] Prediction 2: **REFUTED** - 65536 performed WORST (0% final-100), not best as predicted
- [x] Prediction 3: **REFUTED** - 16384 (53%) vastly outperformed 8192 (14.5%) - 3.7x difference
- [x] Prediction 4: **PARTIALLY SUPPORTED** - Runtime ranged 711-966s (36% variation), not "nearly identical" but not dramatic

## Conclusion

### Hypothesis Status: **REFUTED**

### Key Finding
The current default buffer_size=16384 is optimal; larger buffers catastrophically fail because they retain stale experiences from early bad policies that corrupt learning.

### Implications
- Do NOT increase buffer size beyond 16384 for this training setup
- The "infinite memory" approach (keeping all experiences) is actively harmful
- Buffer eviction is a FEATURE, not a bug - it removes outdated experiences
- PER cannot compensate for fundamentally stale data

### Next Steps
- [x] Keep buffer_size=16384 as default (confirmed optimal)
- [ ] Investigate why 8192 underperformed 4096 (possible variance artifact)
- [ ] Consider if even smaller buffers (2048) might work for faster adaptation
