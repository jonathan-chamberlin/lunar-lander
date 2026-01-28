# Hyperparameter Sweep Strategies

## Sweep Types Comparison

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Grid Search** | Few parameters, discrete values | Exhaustive, reproducible | Exponential scaling |
| **Random Search** | Many parameters, continuous ranges | Better coverage, scalable | May miss optima |
| **Bayesian** | Expensive evaluations | Sample efficient | Complex setup |

## Grid Search

### Best For
- 2-3 parameters
- Known good value ranges
- When you need all combinations tested

### Example
```json
{
  "type": "grid",
  "parameters": {
    "actor_lr": [0.0005, 0.001, 0.002],
    "critic_lr": [0.001, 0.002, 0.004]
  }
}
```
Total: 3 × 3 = 9 runs

### Scaling Problem
| Parameters | Values Each | Total Runs |
|------------|-------------|------------|
| 2 | 3 | 9 |
| 3 | 3 | 27 |
| 4 | 3 | 81 |
| 5 | 3 | 243 |

## Random Search

### Best For
- 4+ parameters
- Continuous value ranges
- Limited compute budget

### Why It Works
Random search often outperforms grid search because:
1. Not all parameters are equally important
2. Random samples explore more unique values
3. Better coverage of the search space

### Example
```json
{
  "type": "random",
  "n_samples": 20,
  "parameters": {
    "actor_lr": {"min": 0.0001, "max": 0.01, "log": true},
    "critic_lr": {"min": 0.0002, "max": 0.01, "log": true},
    "batch_size": [64, 128, 256],
    "tau": {"min": 0.001, "max": 0.02, "log": true}
  }
}
```

### Log Scale
Use `"log": true` for parameters where:
- The range spans multiple orders of magnitude
- Relative changes matter more than absolute (e.g., 0.001 → 0.002 is as significant as 0.01 → 0.02)

Good for: learning rates, tau, buffer sizes
Not for: gamma, batch size, noise parameters

## Recommended Sweep Workflows

### Workflow 1: Quick Exploration (1-2 hours)
1. Random search with 10-20 samples
2. Reduced episodes (200-300)
3. Identify promising regions

```json
{
  "name": "quick_explore",
  "type": "random",
  "n_samples": 15,
  "episodes_per_run": 250,
  "parameters": {
    "actor_lr": {"min": 0.0001, "max": 0.005, "log": true},
    "critic_lr": {"min": 0.0002, "max": 0.01, "log": true}
  }
}
```

### Workflow 2: Focused Grid (2-4 hours)
1. Use insights from quick exploration
2. Grid search around best values
3. Full episode count

```json
{
  "name": "focused_grid",
  "type": "grid",
  "episodes_per_run": 500,
  "parameters": {
    "actor_lr": [0.0008, 0.001, 0.0012],
    "critic_lr": [0.0015, 0.002, 0.0025]
  }
}
```

### Workflow 3: Full Sweep (overnight)
1. Comprehensive random search
2. Many parameters
3. Full episodes

```json
{
  "name": "comprehensive",
  "type": "random",
  "n_samples": 50,
  "episodes_per_run": 500,
  "parameters": {
    "actor_lr": {"min": 0.0001, "max": 0.01, "log": true},
    "critic_lr": {"min": 0.0002, "max": 0.01, "log": true},
    "batch_size": [64, 128, 256],
    "buffer_size": [8192, 16384, 32768],
    "tau": {"min": 0.001, "max": 0.02, "log": true},
    "sigma": [0.2, 0.3, 0.4]
  }
}
```

## Analysis Tips

### Identifying Important Parameters

After a sweep, check parameter impact:
1. **High impact**: Large variance in success rate across values
2. **Low impact**: Similar performance regardless of value

Focus future sweeps on high-impact parameters.

### Detecting Interactions

Some parameters interact:
- `actor_lr` and `critic_lr` (ratio matters)
- `batch_size` and `buffer_size`
- `sigma` and `noise_decay_episodes`

If best configs have inconsistent patterns, parameters may be interacting.

### Statistical Significance

With random seeds, single runs have variance. For reliable comparisons:
- Run best configs 3-5 times with different seeds
- Compare mean ± std of success rate
- Difference of <5% may not be significant

## Common Mistakes

1. **Too few episodes**: Need 300+ episodes to assess convergence
2. **Grid search with too many parameters**: Exponential blowup
3. **Ignoring variance**: Single runs can be misleading
4. **Not saving intermediate results**: Long sweeps can crash
5. **Searching too wide**: Narrow range around known good values first
