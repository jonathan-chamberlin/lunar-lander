# Experiment Log

Records hypotheses, results, and conclusions from hyperparameter experiments.

---

## Experiment Template

```markdown
### [DATE] - [EXPERIMENT NAME]

**Sweep directory:** `sweep_results/[timestamp]/`

───────────────────────────────────────────────────────────────────────
HYPOTHESIS
───────────────────────────────────────────────────────────────────────
[What do we expect to find? What question are we answering?]

───────────────────────────────────────────────────────────────────────
CONFIGURATION
───────────────────────────────────────────────────────────────────────
- Type: grid / random
- Episodes per run:
- Runs per config:
- Parameters tested:

───────────────────────────────────────────────────────────────────────
RESULTS
───────────────────────────────────────────────────────────────────────
| Config | Success Rate | Std Dev | Mean Reward | Notes |
|--------|--------------|---------|-------------|-------|
| ... | | | | |

───────────────────────────────────────────────────────────────────────
CONCLUSION
───────────────────────────────────────────────────────────────────────
**Hypothesis supported?** Yes / No / Partially

**Key findings:**
1.
2.

**Recommended configuration:**
- actor_lr:
- critic_lr:

**Limitations / caveats:**
-

───────────────────────────────────────────────────────────────────────
NEXT STEPS
───────────────────────────────────────────────────────────────────────
- [ ] Action items based on findings
```

---

## Planned Experiments

### Experiment 1: 3×3 Learning Rate Grid Search

**Sweep directories:**
- Replication 1: `sweep_results/2026-01-16_18h44m57s_lr_grid_3x3/`
- Replication 2: `sweep_results/2026-01-16_21h20m53s_lr_grid_3x3/`

───────────────────────────────────────────────────────────────────────
HYPOTHESIS
───────────────────────────────────────────────────────────────────────
The commonly recommended 2:1 critic:actor learning rate ratio may not be optimal for TD3 on LunarLander. By testing actor and critic learning rates independently across a range from conservative (0.0003/0.0006) to aggressive (0.003/0.006), we will identify:

1. Whether the 2:1 ratio is actually optimal, or if other ratios perform better
2. The optimal absolute magnitude for each learning rate
3. Which parameter (actor_lr or critic_lr) has greater impact on performance

**Null hypothesis:** The current defaults (actor_lr=0.001, critic_lr=0.002) are near-optimal.

───────────────────────────────────────────────────────────────────────
CONFIGURATION
───────────────────────────────────────────────────────────────────────
- Type: grid
- Episodes per run: 500
- Runs per config: 2 replications
- Total runs: 18 (9 configs × 2)

**Parameters:**
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| actor_lr | 0.0003, 0.001, 0.003 | Conservative → Aggressive (10× range) |
| critic_lr | 0.0006, 0.002, 0.006 | Conservative → Aggressive (10× range) |

**Ratio coverage:**
|  | actor=0.0003 | actor=0.001 | actor=0.003 |
|--|--------------|-------------|-------------|
| **critic=0.0006** | 2:1 | 0.6:1 | 0.2:1 |
| **critic=0.002** | 6.7:1 | **2:1 (baseline)** | 0.67:1 |
| **critic=0.006** | 20:1 | 6:1 | 2:1 |

───────────────────────────────────────────────────────────────────────
RESULTS (Replication 1)
───────────────────────────────────────────────────────────────────────

| actor_lr | critic_lr | Ratio | Success% | Final100% | Mean Reward | First Success | Time(s) |
|----------|-----------|-------|----------|-----------|-------------|---------------|---------|
| 0.0003 | 0.0006 | 2:1 | 0.2% | 0% | -141.2 | ep 49 | 1639 |
| 0.0003 | 0.002 | 6.7:1 | 0.0% | 0% | -147.5 | - | 1402 |
| 0.0003 | 0.006 | 20:1 | 0.0% | 0% | -142.9 | - | 1245 |
| 0.001 | 0.0006 | 0.6:1 | 0.2% | 1% | -178.5 | ep 437 | 424 |
| **0.001** | **0.002** | **2:1** | **0.0%** | **0%** | **-192.8** | **-** | **524** |
| 0.001 | 0.006 | 6:1 | 0.0% | 0% | -474.9 | - | 592 |
| 0.003 | 0.0006 | 0.2:1 | 0.0% | 0% | -174.1 | - | 1548 |
| 0.003 | 0.002 | 0.67:1 | 0.0% | 0% | -209.2 | - | 1182 |
| 0.003 | 0.006 | 2:1 | 0.0% | 0% | -340.0 | - | 768 |

**Baseline config (bold) performed poorly - no successes in 500 episodes.**

**Heat map by mean reward (lower is worse):**
```
           critic=0.0006  critic=0.002  critic=0.006
actor=0.0003   -141.2        -147.5        -142.9
actor=0.001    -178.5        -192.8        -474.9  ← worst
actor=0.003    -174.1        -209.2        -340.0
```

───────────────────────────────────────────────────────────────────────
RESULTS (Replication 2)
───────────────────────────────────────────────────────────────────────

| actor_lr | critic_lr | Ratio | Success% | Final100% | Mean Reward | First Success | Time(s) |
|----------|-----------|-------|----------|-----------|-------------|---------------|---------|
| 0.0003 | 0.0006 | 2:1 | 0.0% | 0% | -197.3 | - | 298 |
| 0.0003 | 0.002 | 6.7:1 | 0.0% | 0% | -186.8 | - | 1244 |
| 0.0003 | 0.006 | 20:1 | 0.0% | 0% | -131.0 | - | 1446 |
| 0.001 | 0.0006 | 0.6:1 | 0.0% | 0% | -178.4 | - | 1416 |
| **0.001** | **0.002** | **2:1** | **0.0%** | **0%** | **-171.8** | **-** | **916** |
| 0.001 | 0.006 | 6:1 | 0.0% | 0% | -141.2 | - | 1538 |
| 0.003 | 0.0006 | 0.2:1 | 0.0% | 0% | -145.2 | - | 1468 |
| 0.003 | 0.002 | 0.67:1 | 0.0% | 0% | -168.0 | - | 1485 |
| 0.003 | 0.006 | 2:1 | 0.0% | 0% | -318.8 | - | 1043 |

**Replication 2 showed 0% success across all configurations.**

───────────────────────────────────────────────────────────────────────
COMBINED RESULTS (Averaged across replications)
───────────────────────────────────────────────────────────────────────

| actor_lr | critic_lr | Ratio | Avg Success% | Avg Mean Reward | Variance Notes |
|----------|-----------|-------|--------------|-----------------|----------------|
| 0.0003 | 0.0006 | 2:1 | 0.1% | -169.2 | High variance (-141 vs -197) |
| 0.0003 | 0.002 | 6.7:1 | 0.0% | -167.2 | Moderate variance |
| 0.0003 | 0.006 | 20:1 | 0.0% | -137.0 | High variance (-143 vs -131) |
| 0.001 | 0.0006 | 0.6:1 | 0.1% | -178.4 | Consistent |
| **0.001** | **0.002** | **2:1** | **0.0%** | **-182.3** | **Baseline** |
| 0.001 | 0.006 | 6:1 | 0.0% | -308.1 | Extreme variance (-475 vs -141) |
| 0.003 | 0.0006 | 0.2:1 | 0.0% | -159.7 | High variance (-174 vs -145) |
| 0.003 | 0.002 | 0.67:1 | 0.0% | -188.6 | Moderate variance |
| 0.003 | 0.006 | 2:1 | 0.0% | -329.4 | Consistent (both poor) |

**Combined heat map by average mean reward:**
```
           critic=0.0006  critic=0.002  critic=0.006
actor=0.0003   -169.2        -167.2        -137.0  ← best avg reward
actor=0.001    -178.4        -182.3        -308.1  ← high variance at 0.006
actor=0.003    -159.7        -188.6        -329.4  ← worst at 0.006
```

───────────────────────────────────────────────────────────────────────
CONCLUSION
───────────────────────────────────────────────────────────────────────
**Hypothesis supported?** Partially

The 2:1 ratio is NOT universally optimal - performance depends heavily on absolute magnitude of learning rates. However, we could not identify an optimal configuration because **all configs failed to learn reliably in 500 episodes**.

**Key findings:**
1. **All configurations achieved ≤0.2% success rate** - the learning horizon is insufficient
2. **High critic_lr (0.006) is detrimental** - especially combined with high actor_lr, producing worst results (-308 to -329 mean reward)
3. **Lower critic_lr (0.0006) showed the only successes** but only in replication 1
4. **Extreme variance between replications** - same config can vary by 50+ in mean reward
5. **Baseline (0.001/0.002) performed average** - not optimal but not worst either
6. **The ratio matters less than absolute magnitude** - 2:1 at (0.003/0.006) was catastrophic while 2:1 at (0.0003/0.0006) was among the best

**Recommended configuration (tentative):**
- actor_lr: 0.0003 (conservative)
- critic_lr: 0.0006 (conservative)
- **Caveat:** Needs validation with longer training runs (1000+ episodes)

**Limitations / caveats:**
- 500 episodes is insufficient for reliable convergence
- Only 2 replications - high uncertainty in estimates
- No statistical significance testing performed
- Other hyperparameters (batch_size, tau, noise) held constant

───────────────────────────────────────────────────────────────────────
NEXT STEPS
───────────────────────────────────────────────────────────────────────
- [x] Execute sweep 1
- [x] Execute sweep 2
- [x] Analyze combined results
- [x] Update conclusion
- [x] Run 1000+ episode experiments with conservative LRs (see Experiment 2)
- [ ] Investigate other hyperparameters (exploration noise, batch size)
- [ ] Consider whether the TD3 implementation has bugs affecting learning

---

### Experiment 2: Ultra-Conservative Learning Rates (1500 episodes)

**Sweep directory:** `sweep_results/2026-01-17_01h08m53s_lr_conservative_1500ep/`

───────────────────────────────────────────────────────────────────────
HYPOTHESIS
───────────────────────────────────────────────────────────────────────
Based on Experiment 1's finding that conservative learning rates (0.0003/0.0006) showed most promise, we hypothesize that:

1. Even more conservative learning rates (0.0001/0.0002) may enable stable learning
2. With 1500 episodes (3× longer training), the agent will achieve meaningful success rates
3. The learning rate configuration is the primary bottleneck preventing learning

**Null hypothesis:** Learning rate tuning alone cannot achieve reliable learning on LunarLander.

───────────────────────────────────────────────────────────────────────
CONFIGURATION
───────────────────────────────────────────────────────────────────────
- Type: grid
- Episodes per run: 1500
- Runs per config: 1
- Total runs: 4

**Parameters:**
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| actor_lr | 0.0001, 0.0003 | Ultra-conservative range based on Experiment 1 |
| critic_lr | 0.0002, 0.0006 | Ultra-conservative range based on Experiment 1 |

**Ratio coverage:**
|  | actor=0.0001 | actor=0.0003 |
|--|--------------|--------------|
| **critic=0.0002** | 2:1 | 0.67:1 |
| **critic=0.0006** | 6:1 | 2:1 |

───────────────────────────────────────────────────────────────────────
RESULTS
───────────────────────────────────────────────────────────────────────

| actor_lr | critic_lr | Ratio | Success% | Final100% | Mean Reward | Max Reward | Time(s) |
|----------|-----------|-------|----------|-----------|-------------|------------|---------|
| 0.0001 | 0.0002 | 2:1 | 0.0% | 0% | -109.6 | 47.0 | 4126 |
| 0.0001 | 0.0006 | 6:1 | 0.0% | 0% | -107.3 | 47.0 | 4230 |
| 0.0003 | 0.0002 | 0.67:1 | 0.0% | 0% | -115.8 | -16.2 | 4548 |
| 0.0003 | 0.0006 | 2:1 | 0.0% | 0% | -130.0 | 14.6 | 3406 |

**All configurations achieved 0% success rate despite 1500 episodes.**

**Heat map by mean reward:**
```
           critic=0.0002  critic=0.0006
actor=0.0001   -109.6        -107.3  ← best
actor=0.0003   -115.8        -130.0  ← worst
```

**Observations:**
- Lower actor_lr (0.0001) consistently outperformed 0.0003
- Max rewards of ~47 for actor_lr=0.0001 configs indicate near-landing behavior but never crossing the 200 threshold
- Mean rewards improved vs 500-episode runs (~-110 vs ~-170) but no successful landings

───────────────────────────────────────────────────────────────────────
CONCLUSION
───────────────────────────────────────────────────────────────────────
**Hypothesis supported?** No

The null hypothesis is supported: **Learning rate tuning alone cannot achieve reliable learning.** Even with:
- Ultra-conservative learning rates (10× lower than literature defaults)
- Extended training (1500 episodes, 3× longer than Experiment 1)
- Multiple learning rate ratios tested

...the agent achieved 0% success rate across all configurations.

**Key findings:**
1. **Learning rate is NOT the primary bottleneck** - extensive tuning produced no successes
2. **Lower actor_lr (0.0001) is marginally better** than 0.0003 based on mean reward
3. **The agent gets close but never succeeds** - max rewards of 47 vs success threshold of 200
4. **Training length alone is insufficient** - 3× more episodes did not enable learning
5. **The problem lies elsewhere** - likely TD3 implementation, exploration, or reward shaping

**Recommended next steps:**
- Stop tuning learning rates - diminishing returns
- Investigate TD3 implementation for bugs
- Review exploration noise parameters
- Consider other hyperparameters (tau, batch_size, gamma)
- Compare against known-working TD3 implementations

**Limitations / caveats:**
- Only 1 run per config (no variance estimates)
- Did not test actor_lr < 0.0001 (likely too slow to learn)

───────────────────────────────────────────────────────────────────────
NEXT STEPS
───────────────────────────────────────────────────────────────────────
- [ ] Audit TD3 implementation against reference implementations
- [ ] Test exploration noise parameters (noise_std, noise_clip)
- [ ] Review tau (soft update coefficient) - currently may be too aggressive
- [ ] Check gamma (discount factor) - ensure proper value function estimation
- [ ] Consider testing PPO or SAC as baseline comparison

---

## Completed Experiments

### 2026-01-16 - LR Sweep Configs (NOT EXECUTED)

**Status:** Three sweep configurations were created but never run. Superseded by Experiment 1 above.

| Directory | Status |
|-----------|--------|
| `2026-01-16_18h25m02s_lr_sweep_1000/` | Config only, no results |
| `2026-01-16_18h25m15s_lr_sweep_1000/` | Config only, no results |
| `2026-01-16_18h29m18s_lr_sweep_1000/` | Config only, no results |

---

## Running Summary of Best Known Configurations

| Date | actor_lr | critic_lr | Ratio | Success Rate | Mean Reward | Episodes | Source |
|------|----------|-----------|-------|--------------|-------------|----------|--------|
| baseline | 0.001 | 0.002 | 2:1 | 0.0% | -182.3 | 500 | config.py defaults |
| 2026-01-16 | 0.0003 | 0.0006 | 2:1 | 0.1% | -169.2 | 500 | Experiment 1 |
| 2026-01-17 | 0.0001 | 0.0006 | 6:1 | 0.0% | -107.3 | 1500 | Experiment 2 |

**Critical Finding:** After 22 runs across 13 configurations with 500-1500 episodes, no configuration achieved meaningful success. The best mean reward (-107.3) is still far from the success threshold (200). **Learning rate tuning is not the solution.**

---

## Key Insights

### Learning Rates
- **Absolute magnitude matters more than ratio** - 2:1 at (0.003/0.006) is catastrophic, 2:1 at (0.0003/0.0006) is best
- **High critic_lr (0.006) is consistently harmful** - especially when paired with high actor_lr
- **Ultra-conservative learning rates (0.0001) show best mean rewards** but still 0% success
- Literature's "critic LR >= actor LR" guidance holds but magnitude is key
- **Learning rate tuning has reached diminishing returns** - further tuning unlikely to help

### Interactions
- **actor_lr × critic_lr interaction is significant** - cannot optimize independently
- High actor_lr (0.003) + high critic_lr (0.006) = worst performance
- Low actor_lr (0.0001-0.0003) is relatively robust across critic_lr values

### Failed Approaches
- **500 episodes is insufficient** for any configuration to learn reliably
- **1500 episodes is also insufficient** - extended training did not enable learning
- **Aggressive learning rates (0.003/0.006) consistently fail** - avoid these ranges
- **Default baseline (0.001/0.002) underperforms** compared to conservative values
- **Learning rate tuning alone cannot solve this problem** - other factors are limiting

### Root Cause Investigation Needed
The agent achieves max rewards of ~47 but never crosses the 200 threshold. Possible causes:
1. **TD3 implementation bugs** - compare against reference implementations
2. **Exploration strategy** - noise parameters may be suboptimal
3. **Soft update coefficient (tau)** - may be too aggressive or too slow
4. **Reward shaping** - environment reward structure may need modification
5. **Network architecture** - current architecture may be insufficient

---

## Methodology Notes

### Metrics Priority
1. **final_100_success_rate** - Most important; shows converged performance
2. **first_success_episode** - Shows learning speed
3. **success_rate** - Overall performance (can be misleading if early episodes dominate)
4. **mean_reward** - Secondary metric; success rate more interpretable

### Minimum Episodes for Valid Conclusions
- 300 episodes: Can detect gross differences
- 500 episodes: Reasonable for screening
- 1000 episodes: High confidence in convergence behavior
- 1500 episodes: Tested in Experiment 2 - still insufficient for TD3 to learn LunarLander

**Note:** The issue is not training length - it's fundamental learning capability.

### Variance Considerations
- Single runs have seed-dependent variance
- Top 3 configs should be re-run with different seeds before final selection
- <5% difference in success rate may not be statistically significant
