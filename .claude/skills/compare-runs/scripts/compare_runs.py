#!/usr/bin/env python3
"""
Training Run Comparison Tool

Compare metrics across multiple training runs.

Usage:
    python compare_runs.py run1.txt run2.txt run3.txt
    python compare_runs.py logs/*.txt --output comparison.csv
"""

import argparse
import re
import statistics
from pathlib import Path
from dataclasses import dataclass
import csv


@dataclass
class RunMetrics:
    """Metrics for a single training run."""
    name: str
    file: str
    total_episodes: int = 0
    successes: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_reward: float = float('-inf')
    min_reward: float = float('inf')
    first_success: int = -1
    final_100_success_rate: float = 0.0


def parse_log_file(filepath: Path) -> RunMetrics:
    """Parse a log file and extract metrics."""
    name = filepath.stem
    episodes = []

    pattern = r"Run (\d+)\s*([✓✗])\s*\w+.*Reward:\s*([\d.-]+)"

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                episodes.append({
                    'number': int(match.group(1)),
                    'success': match.group(2) == '✓',
                    'reward': float(match.group(3))
                })

    if not episodes:
        return RunMetrics(name=name, file=str(filepath))

    rewards = [e['reward'] for e in episodes]
    successes = [e for e in episodes if e['success']]

    # Find first success
    first_success = -1
    for e in episodes:
        if e['success']:
            first_success = e['number']
            break

    # Final 100 episodes
    final_100 = episodes[-100:] if len(episodes) >= 100 else episodes
    final_100_successes = sum(1 for e in final_100 if e['success'])

    return RunMetrics(
        name=name,
        file=str(filepath),
        total_episodes=len(episodes),
        successes=len(successes),
        success_rate=len(successes) / len(episodes),
        mean_reward=statistics.mean(rewards),
        std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0,
        max_reward=max(rewards),
        min_reward=min(rewards),
        first_success=first_success,
        final_100_success_rate=final_100_successes / len(final_100)
    )


def compare_runs(log_files: list[Path]) -> list[RunMetrics]:
    """Parse and compare multiple runs."""
    metrics = []

    for filepath in log_files:
        if filepath.exists():
            run_metrics = parse_log_file(filepath)
            metrics.append(run_metrics)
        else:
            print(f"Warning: File not found: {filepath}")

    # Sort by success rate (descending)
    metrics.sort(key=lambda x: x.success_rate, reverse=True)

    return metrics


def format_comparison_table(metrics: list[RunMetrics]) -> str:
    """Format metrics as a comparison table."""
    if not metrics:
        return "No runs to compare."

    output = []

    # Header
    output.append("=" * 90)
    output.append("TRAINING RUN COMPARISON")
    output.append("=" * 90)

    # Table header
    output.append(f"\n{'Run':<30} {'Episodes':>8} {'Success%':>10} {'Mean Rwd':>10} {'Max Rwd':>10} {'1st Success':>12}")
    output.append("-" * 90)

    # Rows
    for i, m in enumerate(metrics):
        star = " *" if i == 0 else "  "
        first_success_str = str(m.first_success) if m.first_success > 0 else "N/A"

        output.append(
            f"{m.name:<28}{star} {m.total_episodes:>8} {m.success_rate:>9.1%} "
            f"{m.mean_reward:>10.1f} {m.max_reward:>10.1f} {first_success_str:>12}"
        )

    output.append("-" * 90)

    # Summary
    if metrics:
        best = metrics[0]
        output.append(f"\nBest run: {best.name} ({best.success_rate:.1%} success rate)")

        # Performance spread
        if len(metrics) > 1:
            worst = metrics[-1]
            spread = best.success_rate - worst.success_rate
            output.append(f"Performance spread: {spread:.1%} (best vs worst)")

    return '\n'.join(output)


def format_detailed_comparison(metrics: list[RunMetrics]) -> str:
    """Format detailed comparison for each run."""
    output = []

    for i, m in enumerate(metrics, 1):
        output.append(f"\n{'='*60}")
        output.append(f"#{i} - {m.name}")
        output.append(f"{'='*60}")
        output.append(f"  File: {m.file}")
        output.append(f"  Episodes: {m.total_episodes}")
        output.append(f"  Success Rate: {m.success_rate:.1%} ({m.successes}/{m.total_episodes})")
        output.append(f"  Mean Reward: {m.mean_reward:.1f} (std: {m.std_reward:.1f})")
        output.append(f"  Max Reward: {m.max_reward:.1f}")
        output.append(f"  Min Reward: {m.min_reward:.1f}")
        output.append(f"  First Success: Episode {m.first_success if m.first_success > 0 else 'N/A'}")
        output.append(f"  Final 100 Success Rate: {m.final_100_success_rate:.1%}")

    return '\n'.join(output)


def export_csv(metrics: list[RunMetrics], filepath: Path):
    """Export comparison to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'name', 'file', 'episodes', 'successes', 'success_rate',
            'mean_reward', 'std_reward', 'max_reward', 'min_reward',
            'first_success', 'final_100_success_rate'
        ])

        for i, m in enumerate(metrics, 1):
            writer.writerow([
                i, m.name, m.file, m.total_episodes, m.successes,
                f"{m.success_rate:.4f}", f"{m.mean_reward:.2f}", f"{m.std_reward:.2f}",
                f"{m.max_reward:.2f}", f"{m.min_reward:.2f}",
                m.first_success, f"{m.final_100_success_rate:.4f}"
            ])


def main():
    parser = argparse.ArgumentParser(description="Compare training runs")
    parser.add_argument("logs", nargs="+", help="Log files to compare")
    parser.add_argument("--output", "-o", help="Export to CSV file")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed comparison")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    log_files = [Path(f) for f in args.logs]

    print(f"Comparing {len(log_files)} runs...")

    metrics = compare_runs(log_files)

    if not metrics:
        print("No valid runs found.")
        return 1

    if args.json:
        import json
        data = [vars(m) for m in metrics]
        print(json.dumps(data, indent=2))
    else:
        print(format_comparison_table(metrics))

        if args.detailed:
            print(format_detailed_comparison(metrics))

    if args.output:
        export_csv(metrics, Path(args.output))
        print(f"\nExported to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
