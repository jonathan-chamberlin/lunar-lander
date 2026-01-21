---
id: EXP_012
title: Reward Shaping Ablation
status: PLANNED
created: 2026-01-21
completed:
concluded:
---

# Experiment: Reward Shaping Ablation

## Hypothesis

Some reward shaping components may be harmful to learning by creating local optima or conflicting gradients; a minimal shaping approach may outperform the full shaping function.

**Reasoning:**
- Current shaping has 4 per-step bonuses: time penalty, altitude bonus, leg contact bonus, stability bonus
- Each bonus adds learning signal but also risk of reward hacking or local optima
- Time penalty: Could rush agent into crashes, or helpfully discourage hovering
- Altitude bonus: Could cause diving behavior, or help guide descent
- Leg contact bonus: Could encourage premature landing attempts, or provide crucial signal
- Stability bonus: Could distract from position/velocity control, or improve final landings
- Testing all 2^4 = 16 combinations reveals which components actually help

## Variables

### Independent (what you're changing)
- `reward_time_penalty`: Enable -0.05/step penalty (discourages hovering)
- `reward_altitude_bonus`: Enable +0.5 close-to-ground bonus when y_pos < 0.25 AND descending
- `reward_leg_contact`: Enable +2/+5 for one/both leg contacts AND descending
- `reward_stability`: Enable +0.3/+0.1 for |angle| < 0.1/0.2 AND descending

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
- Terminal landing bonus: +100 (always enabled - this is the goal signal)
- batch_size: 32 (from EXP_004)
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

- [ ] Prediction 1: Full shaping [T,T,T,T] will NOT be the best performer - some components hurt learning
- [ ] Prediction 2: Leg contact bonus alone [F,F,T,F] will be a top performer (most direct landing signal)
- [ ] Prediction 3: No shaping [F,F,F,F] will perform worse than at least one shaped variant
- [ ] Prediction 4: Time penalty will have mixed effects - helping some configs, hurting others

## Configuration

```json
{
  "name": "reward_shaping_ablation",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "reward_time_penalty": [false, true],
    "reward_altitude_bonus": [false, true],
    "reward_leg_contact": [false, true],
    "reward_stability": [false, true]
  }
}
```

## Execution

- **Started:**
- **Completed:**
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 500
- **Number of configurations:** 16
- **Runs per configuration:** 2
- **Total runs:** 32

## Results

<!-- Fill after experiment completes -->

### Summary Table

| Run | Config (T/A/L/S) | Success % | Max Consec | First Success | Final 100 % | Time | Total Successes | Avg Reward | Final 100 Reward |
|-----|------------------|-----------|------------|---------------|-------------|------|-----------------|------------|------------------|
| 1 | F/F/F/F | | | | | | | | |
| 2 | F/F/F/F | | | | | | | | |
| 3 | T/F/F/F | | | | | | | | |
| 4 | T/F/F/F | | | | | | | | |
| 5 | F/T/F/F | | | | | | | | |
| 6 | F/T/F/F | | | | | | | | |
| 7 | T/T/F/F | | | | | | | | |
| 8 | T/T/F/F | | | | | | | | |
| 9 | F/F/T/F | | | | | | | | |
| 10 | F/F/T/F | | | | | | | | |
| 11 | T/F/T/F | | | | | | | | |
| 12 | T/F/T/F | | | | | | | | |
| 13 | F/T/T/F | | | | | | | | |
| 14 | F/T/T/F | | | | | | | | |
| 15 | T/T/T/F | | | | | | | | |
| 16 | T/T/T/F | | | | | | | | |
| 17 | F/F/F/T | | | | | | | | |
| 18 | F/F/F/T | | | | | | | | |
| 19 | T/F/F/T | | | | | | | | |
| 20 | T/F/F/T | | | | | | | | |
| 21 | F/T/F/T | | | | | | | | |
| 22 | F/T/F/T | | | | | | | | |
| 23 | T/T/F/T | | | | | | | | |
| 24 | T/T/F/T | | | | | | | | |
| 25 | F/F/T/T | | | | | | | | |
| 26 | F/F/T/T | | | | | | | | |
| 27 | T/F/T/T | | | | | | | | |
| 28 | T/F/T/T | | | | | | | | |
| 29 | F/T/T/T | | | | | | | | |
| 30 | F/T/T/T | | | | | | | | |
| 31 | T/T/T/T | | | | | | | | |
| 32 | T/T/T/T | | | | | | | | |

**Legend:** T=time_penalty, A=altitude_bonus, L=leg_contact, S=stability

### Aggregated by Config

| Config (T/A/L/S) | Avg Success % | Avg Final 100 % | Avg Time (s) |
|------------------|---------------|-----------------|--------------|
| F/F/F/F (none) | | | |
| T/T/T/T (full) | | | |
| F/F/T/F (leg only) | | | |
| T/F/F/F (time only) | | | |

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
- `run_001_chart.png` through `run_032_chart.png`

## Analysis

[What patterns emerged? Any surprises? How do results compare to predictions?]

### Prediction Outcomes

- [ ] Prediction 1: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 2: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 3: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 4: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]

## Conclusion

### Hypothesis Status: **SUPPORTED / REFUTED / INCONCLUSIVE**

### Key Finding
[One sentence summarizing the main takeaway]

### Implications
[What does this mean for future training?]

### Next Steps
- [ ]
- [ ]
