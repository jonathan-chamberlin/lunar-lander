---
id: EXP_004
title: Batch Size vs Training Speed
status: COMPLETED
created: 2026-01-20
completed: 2026-01-20
concluded: 2026-01-20
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

- **Started:** 2026-01-20 12:40
- **Completed:** 2026-01-20 16:00
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 500
- **Number of configurations:** 4
- **Runs per configuration:** 2
- **Total runs:** 8
- **Note:** Experiment required subprocess isolation due to memory leak in training loop

## Results

### Summary Table

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time (s) | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|----------|-----------------|------------|------------------|
| 1 | batch_size=32 | 6.4 | 4 | 151 | 18.0 | 1061 | 32 | -126.6 | 118.7 |
| 2 | batch_size=32 | 9.0 | 5 | 68 | 28.0 | 771 | 45 | -94.6 | 95.3 |
| 3 | batch_size=64 | 3.4 | 2 | 221 | 3.0 | 901 | 17 | -77.8 | 1.2 |
| 4 | batch_size=64 | 7.0 | 2 | 257 | 5.0 | 1068 | 35 | -72.9 | 68.3 |
| 5 | batch_size=128 | 0.0 | 0 | — | 0.0 | 135 | 0 | -194.0 | -125.0 |
| 6 | batch_size=128 | 0.0 | 0 | — | 0.0 | 109 | 0 | -443.5 | -124.1 |
| 7 | batch_size=256 | 10.0 | 4 | 334 | **41.0** | 1063 | 50 | -215.7 | 122.4 |
| 8 | batch_size=256 | 6.6 | 6 | 335 | 23.0 | 1122 | 33 | -254.1 | 112.5 |

### Aggregated by Config

| Config | Avg Success % | Avg Final 100 % | Avg Time (s) | Avg First Success |
|--------|---------------|-----------------|--------------|-------------------|
| batch_size=32 | 7.7 | **23.0** | 916 | **109.5** |
| batch_size=64 | 5.2 | 4.0 | 985 | 239.0 |
| batch_size=128 | 0.0 | 0.0 | 122 | — |
| batch_size=256 | **8.3** | **32.0** | 1093 | 334.5 |

### Best Configuration

```json
{
  "batch_size": 32,
  "rationale": "Best balance of early learning (avg first success at ep 109) and consistent final performance (23% final-100 rate)"
}
```

**Alternative:** batch_size=256 achieved highest final-100 success rate (32%) but required much longer to start learning (first success ~335 episodes).

### Key Observations

1. **batch_size=128 completely failed** - Zero successes in both runs, suggesting this specific value creates pathological learning dynamics with current hyperparameters
2. **batch_size=32 learned fastest** - First success at episode 68 (run 2) and 151 (run 1), averaging 109.5 episodes
3. **batch_size=256 showed strong late-game learning** - Despite slow start (first success ~335), achieved 41% and 23% final-100 success rates
4. **High variance across runs** - Same batch_size showed significant variation (e.g., batch_size=32: 6.4% vs 9.0% success rate)
5. **Smaller batches = faster wall-clock time** - batch_size=128 was fastest but failed; batch_size=32 was ~200s faster than batch_size=256

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

The results reveal a non-monotonic relationship between batch size and performance, with batch_size=128 being a pathological value that completely failed to learn. This was unexpected and warrants further investigation.

**Surprising findings:**
- batch_size=128 (the default) achieved 0% success in both runs
- batch_size=256 outperformed mid-range values despite slower convergence
- The relationship is not linear: 32 > 256 > 64 >> 128

**Possible explanations for batch_size=128 failure:**
- May hit a resonance with update frequency or buffer sampling patterns
- Could interact poorly with current learning rate settings
- May need more episodes to show learning (500 may be insufficient)

### Prediction Outcomes

- [x] Prediction 1: **REFUTED** - batch_size=64 did NOT achieve highest final-100 success rate; batch_size=256 did (32% avg vs 4% for batch_size=64)
- [x] Prediction 2: **SUPPORTED** - batch_size=32 DID achieve first success earliest (episode 68 in run 2, avg 109.5)
- [x] Prediction 3: **REFUTED** - batch_size=256 did NOT have fastest wall-clock time (1093s avg vs 916s for batch_size=32) and had HIGHER final success rate

## Conclusion

### Hypothesis Status: **PARTIALLY SUPPORTED**

The hypothesis that smaller batch sizes provide more frequent updates helping escape local minima was supported (batch_size=32 learned fastest). However, the prediction that larger batches would sacrifice final performance for speed was refuted—batch_size=256 achieved the best final-100 performance while being slower.

### Key Finding
**Batch size has a non-monotonic effect on learning: both small (32) and large (256) batch sizes outperform mid-range values (64, 128), with the current default of 128 being pathologically bad.**

### Implications
1. **Change default batch_size** from 128 to either 32 (for faster learning) or 256 (for higher final performance)
2. **Investigate batch_size=128 failure** - may reveal important hyperparameter interactions
3. **Consider adaptive batch sizing** - start small for exploration, increase for exploitation

### Next Steps
- [ ] Investigate why batch_size=128 fails (run with more episodes, different seeds)
- [ ] Test batch_size=32 vs 256 with longer training (1000+ episodes)
- [ ] Explore batch size scheduling (start small, increase over training)
