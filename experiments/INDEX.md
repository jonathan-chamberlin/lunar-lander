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
| EXP_004 | Batch Size vs Training Speed | CONCLUDED | batch_size=128 fails completely; 32 learns fastest, 256 achieves best final performance | 2026-01-20 |
| EXP_005 | Update Frequency | CONCLUDED | updates=10 optimal (22.7% success, 70% final-100); updates=50 highly unstable (0% to 26%) | 2026-01-20 |
| EXP_006 | Network Size Performance | CONCLUDED | [64,32] matches/exceeds [256,128] while 2x faster; high variance dominates all configs | 2026-01-20 |
| EXP_007 | Buffer Size | CONCLUDED | buffer_size=16384 optimal; larger buffers (65536) catastrophically fail due to stale experiences | 2026-01-20 |
| EXP_008 | Episode Length Cap | CONCLUDED | Default 1000 steps is optimal; shorter caps hurt learning without saving compute | 2026-01-22 |
| EXP_009 | Learning Rate Sweep | CONCLUDED | Equal LRs optimal (actor=critic=0.001 → 68%); actor>critic catastrophic (7.5%) | 2026-01-21 |
| EXP_010 | Tau Sweep | PLANNED | Testing tau [0.001, 0.005, 0.01, 0.05] | 2026-01-22 |
| EXP_012 | Reward Shaping Ablation | CONCLUDED | time_penalty hurts learning; F/T/T/T (no time penalty) achieves 51% vs 16.5% for full shaping | 2026-01-21 |
| EXP_013 | Gamma Sweep | PLANNED | Testing gamma [0.9, 0.99, 0.995, 0.999] | 2026-01-22 |

## Quick Stats

- **Total experiments:** 12
- **Concluded:** 8
- **Completed:** 1
- **Planned:** 2
- **Running:** 0

## Best Known Parameters

Parameters that have been experimentally validated. Update when experiments conclude.

| Parameter | Best Value | Metric | Source | Notes |
|-----------|------------|--------|--------|-------|
| `batch_size` | 32 | Fastest learning | EXP_004 | First success at ep 109 avg; 23% final-100 rate |
| `batch_size` | 256 | Best final performance | EXP_004 | 32% final-100 rate; but slow start (ep 335) |
| `render_mode` | 'none' or 'all' | Both work | EXP_003 | Unified _run_episode() makes both equivalent |
| `hidden_sizes` | [64, 32] | Best speed/performance | EXP_006 | 25.5% final-100 avg, 2.1x faster than [256,128] |
| `training_updates_per_episode` | 10 | Best overall | EXP_005 | 22.7% success avg, 70% final-100 avg; outperforms default (25) |
| `buffer_size` | 16384 | Best performance | EXP_007 | 53% final-100 avg; smaller and larger buffers both worse |
| `actor_lr` | 0.001 | Best performance | EXP_009 | 68% final-100 with equal critic_lr; 1:1 ratio optimal |
| `critic_lr` | 0.001 | Best performance | EXP_009 | 68% final-100 with equal actor_lr; TD3-style 1:1 ratio |
| `time_penalty` | False | Best performance | EXP_012 | 51% final-100 (F/T/T/T) vs 16.5% with time penalty enabled |
| `max_episode_steps` | 1000 | Best performance | EXP_008 | 42% final-100 avg; shorter caps hurt learning and don't save compute |

### Parameters to Avoid

| Parameter | Value | Why | Source |
|-----------|-------|-----|--------|
| `batch_size` | 128 | 0% success in both runs | EXP_004 |
| vectorized envs | AsyncVectorEnv | Completely broken (0% success) | EXP_001 |
| `training_updates_per_episode` | 50 | Highly unstable (0% to 26% variance between runs) | EXP_005 |
| `buffer_size` | 65536+ | Catastrophic failure (0% final-100); stale experiences corrupt learning | EXP_007 |
| `actor_lr > critic_lr` | any | Catastrophic (7.5-15% final-100); policy outruns critic's ability to evaluate | EXP_009 |
| `time_penalty` | True | Full shaping (T/T/T/T) worse than no shaping (16.5% vs 24%); creates conflicting gradients | EXP_012 |
| `altitude_bonus` alone | F/T/F/F | Worst config (5.5% final-100); encourages diving without landing skills | EXP_012 |
| `max_episode_steps` | 400-800 | Shorter caps hurt learning (13-21.5% final-100) vs 42% at 1000; no compute savings | EXP_008 |

### Untested Parameters

<!-- Parameters that would benefit from experimentation -->

- `noise_decay_episodes` - exploration schedule
- `per_alpha` - PER prioritization strength
- `policy_update_frequency` - TD3 delayed policy updates

## Recent Insights

<!-- Add key learnings that span multiple experiments -->

1. **Vectorized environments were the problem**: EXP_001 proved the 8-env AsyncVectorEnv path was completely broken
2. **Single-env architecture works**: EXP_002 showed single-env training can learn, but revealed a bug in rendered path
3. **Unified function is the solution**: EXP_003 confirmed that a single `_run_episode()` function with a `should_render` flag fixes all issues and achieves strong learning (49% final success rate)
4. **Batch size has non-monotonic effect**: EXP_004 found batch_size=128 (default) completely fails; both smaller (32) and larger (256) values work well
5. **Network size is not the bottleneck**: EXP_006 showed [64,32] (2,800 params) matches [256,128] (35K params) performance while being 2x faster
6. **High variance is the dominant factor**: EXP_006 revealed same config can achieve 0% or 51% success - variance vastly outweighs hyperparameter effects
7. **Updates per episode has a non-monotonic optimum**: EXP_005 found training_updates_per_episode=10 beats both smaller (5) and larger (25, 50) values; 50 is pathologically unstable
8. **Buffer eviction is a feature, not a bug**: EXP_007 showed that "infinite memory" buffers (65536) catastrophically fail because PER prioritizes stale, irrelevant experiences from early bad policies; the default 16384 is optimal
9. **Equal learning rates are optimal**: EXP_009 showed TD3-style 1:1 ratio (actor_lr=critic_lr=0.001) achieves 68% final-100, beating both DDPG-style high ratios and the previous 2:1 default; actor > critic causes catastrophic failure
10. **Episode caps don't save compute**: EXP_008 showed shorter max_episode_steps (400-800) hurt learning without saving time because successful episodes terminate early anyway; keep at 1000

## Next Experiments Queue

<!-- Ideas for future experiments -->

- [ ] Longer training runs (1000+ episodes) to see continued improvement beyond 49%
- [ ] Hyperparameter tuning now that the architecture is validated
- [ ] Compare learning curves between render modes with unified function
