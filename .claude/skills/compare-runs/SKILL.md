---
name: compare-runs
description: Compare metrics across multiple training runs to identify best configurations. Use when user wants to compare different training sessions, find the best hyperparameters, or understand which configuration performed better.
allowed-tools: Read, Bash, Glob, Grep
---

# Compare Runs: Training Run Comparison

Compare multiple training runs to identify which configuration performed best. Key metrics include the maximum number of consecutive successes, and percentage of successes in each batch.

## When to Use This Skill

- User asks "which run was better" or "compare these runs"
- User wants to find the best configuration
- User has multiple log files to compare
- After hyperparameter sweeps to rank results

## Workflow

1. **Find logs**: Locate training logs in `docs/*.txt`, `sweep_results/`
2. **Parse all**: Extract metrics from each log
3. **Compare**: Run `python scripts/compare_runs.py` to generate comparison
4. **Visualize**: Run `python scripts/plot_comparison.py` for charts
5. **Recommend**: Identify best configuration

## Reference Documentation

- For metric definitions, see [metrics_definitions.md](references/metrics_definitions.md)
- For statistical guidance, see [statistical_tests.md](references/statistical_tests.md)

## Quick Start

```bash
# Compare multiple log files
python scripts/compare_runs.py docs/run1.txt docs/run2.txt docs/run3.txt

# Compare all logs in a directory
python scripts/compare_runs.py docs/*.txt

# Export comparison to CSV
python scripts/compare_runs.py docs/*.txt --output comparison.csv

# Generate comparison plots
python scripts/plot_comparison.py docs/*.txt --output charts/
```

## Metrics Compared

| Metric | Description | Higher is Better |
|--------|-------------|-----------------|
| Success Rate | % episodes with reward >= 200 | Yes |
| Mean Reward | Average episode reward | Yes |
| Max Reward | Best single episode | Yes |
| First Success | Episode of first success | No (lower = faster) |
| Final 100 Success | Last 100 episodes success rate | Yes |
| Reward Std Dev | Consistency | No (lower = stable) |

## Output Format

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        TRAINING RUN COMPARISON                           ║
╠═══════════════════════╦══════════╦══════════╦══════════╦════════════════╣
║ Run                   ║ Success% ║ Mean Rwd ║ Max Rwd  ║ First Success  ║
╠═══════════════════════╬══════════╬══════════╬══════════╬════════════════╣
║ run_lr0.002 ⭐        ║  82.5%   ║  178.9   ║  305.2   ║    32          ║
║ run_lr0.001           ║  78.2%   ║  156.3   ║  298.5   ║    45          ║
║ run_baseline          ║  71.4%   ║  142.1   ║  287.6   ║    89          ║
╚═══════════════════════╩══════════╩══════════╩══════════╩════════════════╝

Best: run_lr0.002 (82.5% success rate)
```

## Comparison Considerations

- Ensure runs have similar episode counts for fair comparison
- Consider statistical significance for close results
- Check final performance, not just averages
- Review learning curves, not just final metrics
