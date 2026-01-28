#!/usr/bin/env python3
"""
Training Log Plotter

Generate visualizations from training logs.

Usage:
    python log_plotter.py training_output.txt
    python log_plotter.py training_output.txt --output charts/
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class Episode:
    number: int
    success: bool
    outcome: str
    reward: float


def parse_log_file(filepath: Path) -> list[Episode]:
    """Parse episodes from log file."""
    episodes = []
    pattern = r"Run (\d+)\s*([✓✗])\s*(\w+).*Reward:\s*([\d.-]+)"

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                episodes.append(Episode(
                    number=int(match.group(1)),
                    success=match.group(2) == "✓",
                    outcome=match.group(3),
                    reward=float(match.group(4))
                ))

    return episodes


def plot_rewards(episodes: list[Episode], output_path: Path):
    """Plot reward over episodes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = [e.number for e in episodes]
    y = [e.reward for e in episodes]

    # Plot individual rewards
    colors = ['green' if e.success else 'red' for e in episodes]
    ax.scatter(x, y, c=colors, alpha=0.5, s=10)

    # Plot rolling average
    window = 50
    if len(y) >= window:
        rolling_avg = []
        for i in range(len(y) - window + 1):
            rolling_avg.append(sum(y[i:i+window]) / window)
        ax.plot(x[window-1:], rolling_avg, 'b-', linewidth=2, label=f'{window}-episode rolling avg')

    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Success threshold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'rewards.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'rewards.png'}")


def plot_success_rate(episodes: list[Episode], output_path: Path):
    """Plot success rate over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    window = 50
    success_rates = []
    x_values = []

    for i in range(window, len(episodes) + 1):
        window_episodes = episodes[i-window:i]
        rate = sum(1 for e in window_episodes if e.success) / window
        success_rates.append(rate * 100)
        x_values.append(i)

    ax.plot(x_values, success_rates, 'b-', linewidth=2)
    ax.fill_between(x_values, success_rates, alpha=0.3)

    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% target')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% baseline')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Success Rate ({window}-episode rolling window)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'success_rate.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'success_rate.png'}")


def plot_outcome_distribution(episodes: list[Episode], output_path: Path):
    """Plot outcome distribution as pie chart."""
    from collections import Counter

    outcomes = Counter(e.outcome for e in episodes)

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = list(outcomes.keys())
    sizes = list(outcomes.values())

    # Color mapping
    colors = []
    for label in labels:
        if 'LANDED' in label:
            if 'PERFECTLY' in label:
                colors.append('#2ecc71')  # Bright green
            elif 'SOFTLY' in label:
                colors.append('#27ae60')  # Green
            else:
                colors.append('#f39c12')  # Orange (hard landing)
        elif 'CRASHED' in label:
            colors.append('#e74c3c')  # Red
        elif 'FLEW' in label:
            colors.append('#9b59b6')  # Purple
        else:
            colors.append('#95a5a6')  # Gray

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90
    )

    ax.set_title('Outcome Distribution')

    plt.tight_layout()
    plt.savefig(output_path / 'outcomes.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'outcomes.png'}")


def plot_reward_histogram(episodes: list[Episode], output_path: Path):
    """Plot reward distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))

    rewards = [e.reward for e in episodes]

    ax.hist(rewards, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=200, color='green', linestyle='--', label='Success threshold')
    ax.axvline(x=sum(rewards)/len(rewards), color='red', linestyle='--', label='Mean reward')

    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'reward_histogram.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'reward_histogram.png'}")


def generate_ascii_chart(episodes: list[Episode], width: int = 60) -> str:
    """Generate ASCII chart for environments without matplotlib."""
    if not episodes:
        return "No episodes to plot."

    rewards = [e.reward for e in episodes]
    min_r, max_r = min(rewards), max(rewards)
    range_r = max_r - min_r if max_r != min_r else 1

    output = []
    output.append("\nASCII Reward Chart (sampled)")
    output.append("=" * width)

    # Sample points
    step = max(1, len(episodes) // 20)
    sampled = episodes[::step]

    for e in sampled:
        normalized = (e.reward - min_r) / range_r
        bar_len = int(normalized * (width - 25))
        marker = "█" if e.success else "░"
        bar = marker * bar_len
        output.append(f"Ep {e.number:>4}: {bar:<{width-25}} {e.reward:>7.1f}")

    output.append("=" * width)
    output.append(f"Min: {min_r:.1f}  Max: {max_r:.1f}  "
                 f"Mean: {sum(rewards)/len(rewards):.1f}")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description="Generate training log plots")
    parser.add_argument("logfile", help="Path to training log file")
    parser.add_argument("--output", "-o", default=".", help="Output directory for plots")
    parser.add_argument("--ascii", action="store_true", help="Generate ASCII chart only")
    args = parser.parse_args()

    filepath = Path(args.logfile)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1

    episodes = parse_log_file(filepath)
    if not episodes:
        print("No episodes found in log file.")
        return 1

    print(f"Found {len(episodes)} episodes")

    if args.ascii or not HAS_MATPLOTLIB:
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed - generating ASCII chart")
        print(generate_ascii_chart(episodes))
        return 0

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in: {output_path}")

    plot_rewards(episodes, output_path)
    plot_success_rate(episodes, output_path)
    plot_outcome_distribution(episodes, output_path)
    plot_reward_histogram(episodes, output_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
