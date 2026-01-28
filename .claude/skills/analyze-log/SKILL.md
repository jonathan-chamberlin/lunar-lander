---
name: analyze-log
description: Parse and analyze training output logs for insights. Use when user wants to understand training results, extract metrics from logs, see learning curves, or analyze episode outcomes and behavior distributions.
allowed-tools: Read, Bash, Glob, Grep
---

# Analyze Log: Training Log Parser and Analyzer

Parse training log files to extract metrics and generate insights about training performance.

## When to Use This Skill

- User asks about training results or performance
- User wants to analyze a log file
- User mentions "success rate", "reward", or "learning curve"
- User wants to understand training outcomes

## Workflow

1. **Locate logs**: Find training logs in `docs/*.txt` or user-specified path
2. **Parse**: Run `python scripts/log_analyzer.py <logfile>` to extract metrics
3. **Visualize**: Run `python scripts/log_plotter.py <logfile>` for charts
4. **Report**: Generate summary with key insights

## Reference Documentation

- For log format specification, see [log_format_spec.md](references/log_format_spec.md)
- For regex patterns, see [regex_patterns.md](references/regex_patterns.md)

## Quick Start

```bash
# Analyze a specific log file
python scripts/log_analyzer.py docs/training_output.txt

# Generate plots
python scripts/log_plotter.py docs/training_output.txt --output charts/

# Export to CSV
python scripts/log_analyzer.py docs/training_output.txt --csv results.csv
```

## Metrics Extracted

| Metric | Description |
|--------|-------------|
| Episode count | Total training episodes |
| Success rate | % with reward >= 200 |
| Mean reward | Average episode reward |
| Max/Min reward | Best and worst episodes |
| Behavior distribution | Outcome frequencies |
| Learning curve | Reward over time |
| First success episode | When agent first succeeded |

## Log Format Expected

```
Run 123 ✓ LANDED_SOFTLY ✅ Landed Safely 🥕 Reward: 245.3 (env: 250.1 / shaped: -4.8)
Run 124 ✗ CRASHED_HIGH_VELOCITY ❌ Didn't land safely 🥕 Reward: -89.2 (env: -100.0 / shaped: +10.8)
```

## Output Example

```
=== Training Log Analysis ===
File: docs/training_output.txt
Episodes: 500

Success Rate: 78.2% (391/500)
Mean Reward: 156.3 (std: 89.2)
Max Reward: 298.5
Min Reward: -234.1

Outcome Distribution:
  LANDED_PERFECTLY: 45 (9.0%)
  LANDED_SOFTLY: 198 (39.6%)
  LANDED_HARD: 89 (17.8%)
  CRASHED_HIGH_VELOCITY: 78 (15.6%)
  ...

Learning Curve (50-episode windows):
  Episodes 1-50: 12% success
  Episodes 51-100: 34% success
  ...
```
