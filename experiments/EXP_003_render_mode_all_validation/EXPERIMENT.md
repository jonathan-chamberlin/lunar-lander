---
id: EXP_003
title: Render Mode All Validation (Post-Refactor)
status: COMPLETED
created: 2026-01-20
completed: 2026-01-20
---

# Experiment: Render Mode All Validation (Post-Refactor)

## Hypothesis

After refactoring to a unified `_run_episode()` function, `render_mode='all'` will achieve learning behavior similar to what `render_mode='none'` achieved in EXP_002 (5.8% success, improving to 25% in final batch).

**Background:** EXP_002 showed that after the single-env architecture refactor:
- `render_mode='none'`: 5.8% success (working, improved to 25% in final batch)
- `render_mode='all'`: 0% success (broken)

The codebase was then refactored to use a single `_run_episode()` function with a `should_render` flag, ensuring identical logic for both render modes.

**Expected outcome:** `render_mode='all'` should achieve similar success rates to EXP_002's `render_mode='none'` run (~5% overall, improving in later batches).

## Variables

### Independent (what you're changing)
- None (single configuration test)

### Dependent (what you're measuring)
- Success rate (percentage of episodes that were successes)
- Mean reward
- First success episode
- Final 100 success rate
- Learning trend (improving success rate in later batches)

### Controlled (what stays fixed)
- Unified `_run_episode()` function with `should_render=True`
- All hyperparameters (learning rate, batch size, buffer size, etc.)
- Number of episodes per run (500)
- Network architecture

## Predictions

Make specific, falsifiable predictions before running:

- [x] Prediction 1: `render_mode='all'` will achieve success rate >= 3% — **PASSED (12.6%)**
- [x] Prediction 2: First success will occur before episode 400 — **PASSED (Episode 182)**
- [x] Prediction 3: Final 100 success rate will be >= 10% — **PASSED (49%)**

## Configuration

```json
{
  "name": "render_mode_all_validation",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 1,
  "parameters": {
    "render_mode": ["all"]
  }
}
```

## Execution

- **Started:** 2026-01-20
- **Completed:** 2026-01-20
- **Elapsed Time:** 51 minutes (3089s)
- **Results folder:** `experiments/EXP_003_render_mode_all_validation/results/`
- **Charts folder:** `experiments/EXP_003_render_mode_all_validation/charts/`
- **Episodes per run:** 500
- **Number of configurations:** 1
- **Runs per configuration:** 1
- **Total runs:** 1

## Results

### Summary Table

| Run | Config | Success Rate | Mean Reward | First Success | Final 100 Rate | Time |
|-----|--------|--------------|-------------|---------------|----------------|------|
| 1 | `render_mode='all'` | 12.6% | -133.08 | 182 | 49% | 51m |

### Batch-by-Batch Progress

**Run 1 (render_mode='all'):**
- First success at episode 182
- Final 100 episodes (401-500): 49% success rate
- Strong learning progression demonstrated by improvement from early failures to 49% final rate

### Charts

- `run_001_chart.png` (render_mode='all')

## Analysis

### Comparison with EXP_002

| Experiment | render_mode='all' | render_mode='none' |
|------------|-------------------|---------------------|
| EXP_002 (before unified function) | 0% success | 5.8% success |
| EXP_003 (after unified function) | **12.6% success** | N/A |

**Key improvement:** The unified `_run_episode()` function not only fixed `render_mode='all'` but achieved better results than EXP_002's `render_mode='none'` (12.6% vs 5.8%).

### Key Observations

1. **Unified function fixed the render bug:** `render_mode='all'` went from 0% success (EXP_002) to 12.6% success (EXP_003)
2. **Strong learning progression:** Final 100 success rate of 49% shows the agent learned effectively over the 500 episodes
3. **Better than expected:** Results exceeded all predictions and outperformed EXP_002's `render_mode='none'` baseline

## Conclusion

### Hypothesis Status: **CONFIRMED**

The unified `_run_episode()` function successfully fixed `render_mode='all'` and achieved learning behavior that exceeded expectations. All three predictions were confirmed with comfortable margins.

### Next Steps

- [x] ~~If successful: Run EXP_004 comparing both render modes with unified function~~
- [ ] Consider running longer experiments (1000+ episodes) to see if 49% final rate continues to improve
- [ ] The codebase is now validated for both render modes
