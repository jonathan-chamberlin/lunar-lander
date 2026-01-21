---
id: EXP_005
title: Training Updates Per Episode
status: CONCLUDED
created: 2026-01-20
completed: 2026-01-20
concluded: 2026-01-20
---

# Experiment: Training Updates Per Episode

## Hypothesis

Fewer training updates per episode will improve wall-clock speed with minimal impact on learning quality, while more updates may improve sample efficiency at the cost of speed.

**Reasoning:**
- Current default is 25 updates per episode
- Fewer updates (5, 10) = faster episodes but potentially worse learning
- More updates (50) = slower episodes but potentially better sample efficiency
- Need to find optimal speed/quality trade-off

## Variables

### Independent (what you're changing)
- `training_updates_per_episode`: [5, 10, 25, 50]

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

- [ ] Prediction 1: training_updates_per_episode=5 will have the fastest wall-clock time
- [ ] Prediction 2: training_updates_per_episode=50 will have the highest success rate (more updates = better learning)
- [ ] Prediction 3: training_updates_per_episode=25 (default) will be a good balance of speed and performance
- [ ] Prediction 4: Wall-clock speedup from 50 to 5 will be >50% (10x fewer updates)

## Configuration

```json
{
  "name": "update_frequency",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "training_updates_per_episode": [5, 10, 25, 50]
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
- **Note:** Experiment run with subprocess isolation due to memory leak in training loop

## Results

### Summary Table

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | updates=5 | 2.2 | 1 | 320 | 3.0 | 1480 | 11 | -121.6 | 60.7 |
| 2 | updates=5 | 22.6 | 16 | 124 | 70.0 | 1596 | 113 | -58.3 | 187.6 |
| 3 | updates=10 | 22.0 | 8 | 240 | 60.0 | 1486 | 110 | -70.2 | 170.3 |
| 4 | updates=10 | 23.4 | 22 | 299 | **80.0** | 1011 | 117 | -97.4 | 207.1 |
| 5 | updates=25 | 19.6 | 6 | 264 | 30.0 | 1388 | 98 | -35.7 | 99.6 |
| 6 | updates=25 | 12.4 | 2 | 148 | 18.0 | 1275 | 62 | -96.7 | 51.0 |
| 7 | updates=50 | **0.0** | 0 | — | 0.0 | 142 | 0 | -567.5 | -576.7 |
| 8 | updates=50 | **26.0** | 6 | 165 | 44.0 | 1199 | 130 | -13.8 | 132.9 |

### Aggregated by Config

| Config | Avg Success % | Avg Final 100 % | Avg Time (s) | Avg First Success |
|--------|---------------|-----------------|--------------|-------------------|
| updates=5 | 12.4 | 36.5 | 1538 | 222 |
| updates=10 | **22.7** | **70.0** | 1249 | 269.5 |
| updates=25 | 16.0 | 24.0 | 1331 | 206 |
| updates=50 | 13.0 | 22.0 | 670 | 165* |

\* Only run 8 achieved success for updates=50

### Best Configuration

```json
{
  "training_updates_per_episode": 10,
  "rationale": "Highest average success rate (22.7%) and best final-100 performance (70%), with reasonable wall-clock time"
}
```

**Alternative:** training_updates_per_episode=50 achieved the highest single-run success rate (26% in run 8) but is highly unstable (run 7 achieved 0%).

### Key Observations

1. **training_updates_per_episode=10 is optimal** - Best average performance across both runs (22.7% success, 70% final-100)
2. **training_updates_per_episode=50 is highly unstable** - Run 7 completely failed (0%), while run 8 achieved best single-run performance (26%)
3. **training_updates_per_episode=5 has high variance** - Run 1 achieved only 2.2% success while run 2 achieved 22.6%
4. **Wall-clock time is NOT simply correlated with updates** - Run 4 (updates=10) was fastest among successful runs at 1011s
5. **Fewer updates does NOT always mean faster** - updates=5 averaged 1538s, slower than updates=10 (1249s)

### Charts

Charts for each run are saved in the `charts/` folder:
- `run_001_chart.png` (updates=5 #1)
- `run_002_chart.png` (updates=5 #2)
- `run_003_chart.png` (updates=10 #1)
- `run_004_chart.png` (updates=10 #2)
- `run_005_chart.png` (updates=25 #1)
- `run_006_chart.png` (updates=25 #2)
- `run_007_chart.png` (updates=50 #1 - FAILED)
- `run_008_chart.png` (updates=50 #2)

## Analysis

The results reveal a non-monotonic relationship between training updates per episode and learning performance. The optimal value was 10 updates, not the default 25 or the extremes (5 or 50).

**Surprising findings:**
- training_updates_per_episode=50 showed extreme instability (0% vs 26% success between runs)
- The default value of 25 was NOT optimal; 10 performed better
- Wall-clock time did not follow expected patterns - fewer updates didn't always mean faster runs
- training_updates_per_episode=10 achieved 80% final-100 success rate in one run - the highest observed

**Possible explanations for updates=50 instability:**
- Too many updates may cause overfitting to recent experiences early in training
- May require longer training to stabilize (500 episodes insufficient)
- May interact poorly with the noise decay schedule (300 episodes)

### Prediction Outcomes

- [x] Prediction 1: **REFUTED** - training_updates_per_episode=5 did NOT have the fastest wall-clock time (1538s avg vs 1249s for updates=10)
- [x] Prediction 2: **REFUTED** - training_updates_per_episode=50 did NOT have the highest success rate; it had extreme variance (0% to 26%) with 13% average
- [x] Prediction 3: **REFUTED** - training_updates_per_episode=25 was NOT a good balance; updates=10 performed better (22.7% vs 16.0%)
- [x] Prediction 4: **REFUTED** - Wall-clock time showed no clear pattern; updates=10 was actually faster than updates=5 on average

## Conclusion

### Hypothesis Status: **PARTIALLY REFUTED**

The hypothesis that fewer updates = faster with worse learning, and more updates = slower with better learning was wrong. The relationship is non-monotonic, with an optimal middle value (10) outperforming both extremes.

### Key Finding
**training_updates_per_episode=10 is optimal, providing the best balance of learning stability (22.7% avg success) and final performance (70% final-100 average), outperforming both the default (25) and extreme values (5, 50).**

### Implications
1. **Change default training_updates_per_episode** from 25 to 10
2. **Avoid extreme values** - both 5 and 50 showed high variance or instability
3. **Wall-clock time is not predictable** from update frequency alone; other factors dominate

### Next Steps
- [ ] Investigate why updates=50 run 7 completely failed (possible bad random seed)
- [ ] Test updates=10 with longer training (1000+ episodes) to see if it maintains advantage
- [ ] Combine best settings from EXP_004 and EXP_005: batch_size=32 + updates=10
