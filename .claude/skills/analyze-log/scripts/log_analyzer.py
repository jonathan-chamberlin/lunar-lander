#!/usr/bin/env python3
"""
Training Log Analyzer

Parse training logs and extract metrics.

Usage:
    python log_analyzer.py training_output.txt
    python log_analyzer.py training_output.txt --csv results.csv
    python log_analyzer.py training_output.txt --json
"""

import argparse
import re
import json
import csv
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Episode:
    """Single episode result."""
    number: int
    success: bool
    outcome: str
    reward: float
    env_reward: float = 0.0
    shaped_reward: float = 0.0


@dataclass
class LogAnalysis:
    """Complete log analysis results."""
    file: str
    episodes: list[Episode] = field(default_factory=list)
    total_episodes: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    first_success_episode: int = -1
    outcome_distribution: dict = field(default_factory=dict)
    learning_curve: list[dict] = field(default_factory=list)


def parse_episode_line(line: str) -> Episode | None:
    """Parse a single episode result line."""
    # Pattern: Run 123 ✓ OUTCOME ... Reward: 245.3 (env: 250.1 / shaped: -4.8)
    pattern = r"Run (\d+)\s*([✓✗])\s*(\w+).*Reward:\s*([\d.-]+)"
    match = re.search(pattern, line)

    if not match:
        return None

    episode_num = int(match.group(1))
    success = match.group(2) == "✓"
    outcome = match.group(3)
    reward = float(match.group(4))

    # Try to extract env and shaped rewards
    env_match = re.search(r"env:\s*([\d.-]+)", line)
    shaped_match = re.search(r"shaped:\s*([\d.-]+)", line)

    env_reward = float(env_match.group(1)) if env_match else reward
    shaped_reward = float(shaped_match.group(1)) if shaped_match else 0.0

    return Episode(
        number=episode_num,
        success=success,
        outcome=outcome,
        reward=reward,
        env_reward=env_reward,
        shaped_reward=shaped_reward
    )


def parse_log_file(filepath: Path) -> list[Episode]:
    """Parse all episodes from a log file."""
    episodes = []

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            episode = parse_episode_line(line)
            if episode:
                episodes.append(episode)

    return episodes


def compute_learning_curve(episodes: list[Episode], window_size: int = 50) -> list[dict]:
    """Compute learning curve with rolling windows."""
    curve = []

    for i in range(0, len(episodes), window_size):
        window = episodes[i:i + window_size]
        if not window:
            continue

        successes = sum(1 for e in window if e.success)
        rewards = [e.reward for e in window]

        curve.append({
            "start_episode": window[0].number,
            "end_episode": window[-1].number,
            "success_rate": successes / len(window),
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0
        })

    return curve


def analyze_log(filepath: Path) -> LogAnalysis:
    """Perform complete analysis of a log file."""
    episodes = parse_log_file(filepath)

    if not episodes:
        return LogAnalysis(file=str(filepath))

    rewards = [e.reward for e in episodes]
    successes = [e for e in episodes if e.success]
    outcomes = Counter(e.outcome for e in episodes)

    # Find first success
    first_success = -1
    for e in episodes:
        if e.success:
            first_success = e.number
            break

    analysis = LogAnalysis(
        file=str(filepath),
        episodes=episodes,
        total_episodes=len(episodes),
        successes=len(successes),
        failures=len(episodes) - len(successes),
        success_rate=len(successes) / len(episodes) if episodes else 0,
        mean_reward=statistics.mean(rewards),
        std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0,
        max_reward=max(rewards),
        min_reward=min(rewards),
        first_success_episode=first_success,
        outcome_distribution=dict(outcomes.most_common()),
        learning_curve=compute_learning_curve(episodes)
    )

    return analysis


def format_analysis(analysis: LogAnalysis) -> str:
    """Format analysis results for display."""
    output = []

    output.append("=" * 60)
    output.append("TRAINING LOG ANALYSIS")
    output.append("=" * 60)
    output.append(f"\nFile: {analysis.file}")
    output.append(f"Episodes: {analysis.total_episodes}")

    if analysis.total_episodes == 0:
        output.append("\nNo episodes found in log file.")
        return '\n'.join(output)

    # Success metrics
    output.append(f"\n{'='*60}")
    output.append("SUCCESS METRICS")
    output.append(f"{'='*60}")
    output.append(f"Success Rate: {analysis.success_rate:.1%} ({analysis.successes}/{analysis.total_episodes})")
    output.append(f"First Success: Episode {analysis.first_success_episode}" if analysis.first_success_episode > 0 else "First Success: None")

    # Reward statistics
    output.append(f"\n{'='*60}")
    output.append("REWARD STATISTICS")
    output.append(f"{'='*60}")
    output.append(f"Mean Reward: {analysis.mean_reward:.1f} (std: {analysis.std_reward:.1f})")
    output.append(f"Max Reward: {analysis.max_reward:.1f}")
    output.append(f"Min Reward: {analysis.min_reward:.1f}")

    # Outcome distribution
    output.append(f"\n{'='*60}")
    output.append("OUTCOME DISTRIBUTION")
    output.append(f"{'='*60}")
    for outcome, count in analysis.outcome_distribution.items():
        pct = count / analysis.total_episodes * 100
        bar = "#" * int(pct / 2)
        output.append(f"  {outcome:<25} {count:>4} ({pct:>5.1f}%) {bar}")

    # Learning curve
    if analysis.learning_curve:
        output.append(f"\n{'='*60}")
        output.append("LEARNING CURVE")
        output.append(f"{'='*60}")
        for window in analysis.learning_curve:
            success_bar = "#" * int(window['success_rate'] * 40)
            output.append(
                f"  Episodes {window['start_episode']:>4}-{window['end_episode']:<4}: "
                f"{window['success_rate']:>5.1%} success, "
                f"mean: {window['mean_reward']:>7.1f}  {success_bar}"
            )

    # Insights
    output.append(f"\n{'='*60}")
    output.append("INSIGHTS")
    output.append(f"{'='*60}")

    # Check for learning
    if len(analysis.learning_curve) >= 2:
        first_window = analysis.learning_curve[0]
        last_window = analysis.learning_curve[-1]
        improvement = last_window['success_rate'] - first_window['success_rate']

        if improvement > 0.2:
            output.append(f"  Strong learning: {improvement:.1%} improvement in success rate")
        elif improvement > 0.05:
            output.append(f"  Moderate learning: {improvement:.1%} improvement in success rate")
        elif improvement < -0.05:
            output.append(f"  Warning: Performance degraded by {-improvement:.1%}")
        else:
            output.append("  Flat learning curve - consider hyperparameter tuning")

    # Check for instability
    if analysis.std_reward > 100:
        output.append("  High variance in rewards - training may be unstable")

    # Success patterns
    if analysis.success_rate > 0.8:
        output.append("  Excellent performance - agent is well-trained")
    elif analysis.success_rate > 0.5:
        output.append("  Good performance - consider more training or tuning")
    elif analysis.success_rate > 0.2:
        output.append("  Moderate performance - needs more training")
    else:
        output.append("  Poor performance - review algorithm and hyperparameters")

    return '\n'.join(output)


def export_csv(analysis: LogAnalysis, filepath: Path):
    """Export episode data to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'success', 'outcome', 'reward', 'env_reward', 'shaped_reward'])
        for e in analysis.episodes:
            writer.writerow([e.number, e.success, e.outcome, e.reward, e.env_reward, e.shaped_reward])


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("logfile", help="Path to training log file")
    parser.add_argument("--csv", help="Export to CSV file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--window", type=int, default=50, help="Learning curve window size")
    args = parser.parse_args()

    filepath = Path(args.logfile)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1

    analysis = analyze_log(filepath)

    if args.json:
        # Convert to JSON-serializable format
        data = {
            "file": analysis.file,
            "total_episodes": analysis.total_episodes,
            "successes": analysis.successes,
            "success_rate": analysis.success_rate,
            "mean_reward": analysis.mean_reward,
            "std_reward": analysis.std_reward,
            "max_reward": analysis.max_reward,
            "min_reward": analysis.min_reward,
            "first_success_episode": analysis.first_success_episode,
            "outcome_distribution": analysis.outcome_distribution,
            "learning_curve": analysis.learning_curve
        }
        print(json.dumps(data, indent=2))
    else:
        print(format_analysis(analysis))

    if args.csv:
        export_csv(analysis, Path(args.csv))
        print(f"\nExported to: {args.csv}")

    return 0


if __name__ == "__main__":
    exit(main())
