---
id: EXP_XXX
title: [Experiment Title]
status: PLANNED
created: YYYY-MM-DD
completed:
concluded:
---

# Experiment: [Title]

## Hypothesis

[What do you expect to happen and why? Be specific and falsifiable.]

## Variables

### Independent (what you're changing)
-

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
-

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1: [e.g., "Higher learning rate (1e-3) will achieve first success earlier than lower (1e-4)"]
- [ ] Prediction 2: [e.g., "Success rate will peak at learning_rate=5e-4"]
- [ ] Prediction 3:

## Configuration

```json
{
  "name": "experiment_name",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 1,
  "parameters": {
  }
}
```

## Execution

- **Started:**
- **Completed:**
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:**
- **Number of configurations:**
- **Runs per configuration:**
- **Total runs:**

## Results

<!-- Fill after experiment completes -->

### Summary Table

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|------|-----------------|------------|------------------|
| 1 | | | | | | | | | |

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
- `run_001_chart.png`
- `run_002_chart.png`
- ...

## Analysis

[What patterns emerged? Any surprises? How do results compare to predictions?]

### Prediction Outcomes

- [ ] Prediction 1: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 2: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 3: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]

## Conclusion

### Hypothesis Status: **SUPPORTED / REFUTED / INCONCLUSIVE**

### Key Finding
[One sentence summarizing the main takeaway]

### Implications
[What does this mean for future training?]

### Next Steps
- [ ]
- [ ]
