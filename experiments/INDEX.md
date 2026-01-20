# Experiments Index

Quick reference for all experiments. Update this file when creating or concluding experiments.

## Status Legend

| Status | Meaning |
|--------|---------|
| PLANNED | Hypothesis and config defined, not yet run |
| RUNNING | Currently executing |
| COMPLETED | Finished running, awaiting analysis |
| ANALYZED | Results reviewed, predictions evaluated |
| CONCLUDED | Final conclusions written |

## Experiments

| ID | Title | Status | Key Finding | Date |
|----|-------|--------|-------------|------|
| EXP_001 | Render Mode Comparison | CONCLUDED | Vectorized 8-env path broken (0% success), single-env rendered path works (~49%) | 2026-01-19 |
| EXP_002 | Single Env Render Comparison | COMPLETED | After single-env refactor: render_mode='none' works (5.8%), render_mode='all' broken (0%) | 2026-01-20 |
| EXP_003 | Render Mode All Validation | COMPLETED | Unified _run_episode() fixed render_mode='all': 12.6% success, 49% final 100 rate | 2026-01-20 |
| EXP_004 | Batch Size vs Training Speed | PLANNED | Testing batch_size [32, 64, 128, 256] for performance optimization | 2026-01-20 |
| EXP_005 | Update Frequency | PLANNED | Testing update_every_n_steps [1, 2, 4, 8] for wall-clock speed vs sample efficiency | 2026-01-20 |

## Quick Stats

- **Total experiments:** 5
- **Concluded:** 1
- **Completed:** 2
- **Planned:** 2

## Recent Insights

<!-- Add key learnings that span multiple experiments -->

1. **Vectorized environments were the problem**: EXP_001 proved the 8-env AsyncVectorEnv path was completely broken
2. **Single-env architecture works**: EXP_002 showed single-env training can learn, but revealed a bug in rendered path
3. **Unified function is the solution**: EXP_003 confirmed that a single `_run_episode()` function with a `should_render` flag fixes all issues and achieves strong learning (49% final success rate)

## Next Experiments Queue

<!-- Ideas for future experiments -->

- [ ] Longer training runs (1000+ episodes) to see continued improvement beyond 49%
- [ ] Hyperparameter tuning now that the architecture is validated
- [ ] Compare learning curves between render modes with unified function
