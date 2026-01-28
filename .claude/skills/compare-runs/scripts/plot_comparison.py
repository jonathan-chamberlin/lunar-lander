#!/usr/bin/env python3
"""
Training Run Comparison Plotter

Generate visual comparisons of multiple training runs.

Usage:
    python plot_comparison.py run1.txt run2.txt run3.txt
    python plot_comparison.py logs/*.txt --output charts/
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class Episode:
    number: int
    success: bool
    reward: float


def parse_log_file(filepath: Path) -> tuple[str, list[Episode]]:
    """Parse a log file and return name and episodes."""
    name = filepath.stem
    episodes = []

    pattern = r"Run (\d+)\s*([✓✗])\s*\w+.*Reward:\s*([\d.-]+)"

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                episodes.append(Episode(
                    number=int(match.group(1)),
                    success=match.group(2) == '✓',
                    reward=float(match.group(3))
                ))

    return name, episodes


def plot_reward_comparison(runs: dict[str, list[Episode]], output_path: Path):
    """Plot reward curves for all runs on same chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(range(len(runs)))

    for (name, episodes), color in zip(runs.items(), colors):
        if not episodes:
            continue

        x = [e.number for e in episodes]
        y = [e.reward for e in episodes]

        # Plot rolling average
        window = 50
        if len(y) >= window:
            rolling_avg = []
            for i in range(len(y) - window + 1):
                rolling_avg.append(sum(y[i:i+window]) / window)
            ax.plot(x[window-1:], rolling_avg, label=name, color=color, linewidth=2)

    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Success threshold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (50-episode rolling avg)')
    ax.set_title('Reward Comparison Across Runs')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'reward_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'reward_comparison.png'}")


def plot_success_rate_comparison(runs: dict[str, list[Episode]], output_path: Path):
    """Plot success rate curves for all runs."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(range(len(runs)))

    for (name, episodes), color in zip(runs.items(), colors):
        if len(episodes) < 50:
            continue

        window = 50
        success_rates = []
        x_values = []

        for i in range(window, len(episodes) + 1):
            window_episodes = episodes[i-window:i]
            rate = sum(1 for e in window_episodes if e.success) / window * 100
            success_rates.append(rate)
            x_values.append(i)

        ax.plot(x_values, success_rates, label=name, color=color, linewidth=2)

    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% target')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% baseline')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%, 50-episode window)')
    ax.set_title('Success Rate Comparison Across Runs')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'success_rate_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'success_rate_comparison.png'}")


def plot_metrics_bar_chart(runs: dict[str, list[Episode]], output_path: Path):
    """Plot bar chart comparing key metrics."""
    if not runs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(runs.keys())
    success_rates = []
    mean_rewards = []
    max_rewards = []

    for name, episodes in runs.items():
        if episodes:
            rewards = [e.reward for e in episodes]
            successes = sum(1 for e in episodes if e.success)
            success_rates.append(successes / len(episodes) * 100)
            mean_rewards.append(sum(rewards) / len(rewards))
            max_rewards.append(max(rewards))
        else:
            success_rates.append(0)
            mean_rewards.append(0)
            max_rewards.append(0)

    colors = plt.cm.tab10(range(len(names)))

    # Success rate
    axes[0].bar(names, success_rates, color=colors)
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate')
    axes[0].axhline(y=80, color='green', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='x', rotation=45)

    # Mean reward
    axes[1].bar(names, mean_rewards, color=colors)
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Mean Reward')
    axes[1].axhline(y=200, color='green', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)

    # Max reward
    axes[2].bar(names, max_rewards, color=colors)
    axes[2].set_ylabel('Max Reward')
    axes[2].set_title('Max Reward')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'metrics_comparison.png'}")


def generate_ascii_comparison(runs: dict[str, list[Episode]]) -> str:
    """Generate ASCII comparison for environments without matplotlib."""
    output = []
    output.append("\nASCII Comparison Chart")
    output.append("=" * 60)

    max_name_len = max(len(name) for name in runs.keys()) if runs else 10

    for name, episodes in runs.items():
        if not episodes:
            continue

        successes = sum(1 for e in episodes if e.success)
        success_rate = successes / len(episodes) * 100

        bar_len = int(success_rate / 2)
        bar = "█" * bar_len

        output.append(f"{name:<{max_name_len}} | {bar:<50} {success_rate:>5.1f}%")

    output.append("=" * 60)
    output.append(f"{'':>{max_name_len}} | {'0%':<25}{'50%':<25}100%")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description="Plot training run comparisons")
    parser.add_argument("logs", nargs="+", help="Log files to compare")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    parser.add_argument("--ascii", action="store_true", help="ASCII output only")
    args = parser.parse_args()

    # Parse all log files
    runs = {}
    for filepath in args.logs:
        path = Path(filepath)
        if path.exists():
            name, episodes = parse_log_file(path)
            runs[name] = episodes
            print(f"Loaded: {name} ({len(episodes)} episodes)")

    if not runs:
        print("No valid runs found.")
        return 1

    if args.ascii or not HAS_MATPLOTLIB:
        if not HAS_MATPLOTLIB:
            print("\nmatplotlib not installed - generating ASCII output")
        print(generate_ascii_comparison(runs))
        return 0

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in: {output_path}")

    plot_reward_comparison(runs, output_path)
    plot_success_rate_comparison(runs, output_path)
    plot_metrics_bar_chart(runs, output_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
