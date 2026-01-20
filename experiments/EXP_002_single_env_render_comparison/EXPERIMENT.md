---
id: EXP_002
title: Single Environment Render Mode Comparison
status: COMPLETED
created: 2026-01-20
completed: 2026-01-20
---

# Experiment: Single Environment Render Mode Comparison

## Hypothesis

After refactoring to single-environment architecture, both `render_mode='all'` and `render_mode='none'` will achieve similar success rates, validating that the architectural simplification works correctly.

**Background:** EXP_001 demonstrated that the vectorized training path (8 parallel AsyncVectorEnv) was completely broken (0% success), while the single-env rendered path worked (~49% success). The codebase has been refactored to use a single environment for all training, regardless of render mode.

**Expected outcome:** Both render modes should achieve similar success rates (within ~10% of each other), proving the single-env architecture works for headless training.

## Variables

### Independent (what you're changing)
- `render_mode`: 'all' vs 'none'

### Dependent (what you're measuring)
- Success rate (percentage of episodes that were successes)
- Mean reward
- First success episode
- Final 100 success rate (if applicable)

### Controlled (what stays fixed)
- Single environment architecture (no vectorized envs)
- All hyperparameters (learning rate, batch size, buffer size, etc.)
- Number of episodes per run (500)
- Network architecture

## Predictions

Make specific, falsifiable predictions before running:

- [x] Prediction 1: Both `render_mode='all'` and `render_mode='none'` will achieve success rate >= 20%
  - **FAILED**: render_mode='all' achieved 0%, render_mode='none' achieved 5.8%
- [x] Prediction 2: The difference in success rate between modes will be <= 15 percentage points
  - **FAILED**: Difference was 5.8 percentage points, but both were very low
- [x] Prediction 3: Both modes will achieve first success before episode 300
  - **PARTIAL**: render_mode='none' achieved first success at episode 322, render_mode='all' never succeeded

## Configuration

```json
{
  "name": "single_env_render_comparison",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 1,
  "parameters": {
    "render_mode": ["all", "none"]
  }
}
```

## Execution

- **Started:** 2026-01-20 02:52
- **Completed:** 2026-01-20 03:10
- **Results folder:** `experiments/EXP_002_single_env_render_comparison/results/`
- **Charts folder:** `experiments/EXP_002_single_env_render_comparison/charts/`
- **Episodes per run:** 500
- **Number of configurations:** 2
- **Runs per configuration:** 1
- **Total runs:** 2

## Results

### Summary Table

| Run | Config | Success Rate | Mean Reward | First Success | Final 100 Rate | Time |
|-----|--------|--------------|-------------|---------------|----------------|------|
| 1 | `render_mode='all'` | **0.0%** | -760.2 | None | 0% | 504s |
| 2 | `render_mode='none'` | **5.8%** | -515.6 | Episode 322 | 25% | 458s |

### Batch-by-Batch Progress

**Run 1 (render_mode='all'):**
- Batch 1 (1-100): 0%
- Batch 2 (101-200): 0%
- Batch 3 (201-300): 0%
- Batch 4 (301-400): 0%
- Batch 5 (401-500): 0%

**Run 2 (render_mode='none'):**
- Batch 1 (1-100): 0%
- Batch 2 (101-200): 0%
- Batch 3 (201-300): 0%
- Batch 4 (301-400): 4%
- Batch 5 (401-500): 25% (improving!)

### Charts

- `run_001_chart.png` (render_mode='all') - Shows flat failure across all episodes
- `run_002_chart.png` (render_mode='none') - Shows learning curve improving in final batches

## Analysis

### Unexpected Results

The results are the **opposite** of what was expected based on EXP_001:

| Experiment | render_mode='all' | render_mode='none' |
|------------|-------------------|---------------------|
| EXP_001 (before refactor) | ~49% success | 0% success |
| EXP_002 (after refactor) | 0% success | 5.8% success |

### Possible Causes

1. **Bug in `_run_rendered_episode()`**: The rendered episode function may have a bug introduced during refactoring. The unrendered path works (shows learning), but the rendered path doesn't.

2. **Timing/State Issues**: The rendered path includes pygame event handling and frame rendering which may be interfering with the training loop timing or state management.

3. **Noise Reset**: In the refactored code, `noise.reset()` is called after each episode in the main loop. This should be correct for both paths, but there may be subtle differences in how noise state evolves.

### Key Observations

1. **Unrendered path is now functional**: The single-env unrendered path (`_run_unrendered_episode`) IS learning - it improved from 0% to 25% success in the final batch, suggesting the core training logic works.

2. **Rendered path is broken**: The rendered path (`_run_rendered_episode`) never achieved any success across 500 episodes, indicating a bug specific to that code path.

3. **The refactoring fixed one bug but introduced another**: By removing the broken vectorized path, we fixed `render_mode='none'`. But in the process, something was broken in `render_mode='all'`.

## Conclusion

### Hypothesis Status: **PARTIALLY SUPPORTED**

The single-env architecture **does** work for `render_mode='none'` (5.8% success with clear learning trend), but there's a bug in the `render_mode='all'` path that needs to be fixed.

### Key Finding

The refactoring successfully fixed the headless training path (`render_mode='none'`), which now shows learning behavior (0% → 25% in final batch). However, a regression was introduced in the rendered training path (`render_mode='all'`), which now shows 0% success.

### Next Steps

- [ ] **Debug `_run_rendered_episode()`**: Compare with `_run_unrendered_episode()` to find the bug
- [ ] **Check experience storage**: Verify experiences are being stored correctly in rendered mode
- [ ] **Check action generation**: Verify noise and action clamping work correctly in rendered mode
- [ ] **Re-run experiment after fix**: Verify both modes achieve similar success rates
