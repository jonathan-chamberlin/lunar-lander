"""Compare metrics across training runs.

Parses training logs and generates comparison tables.

Usage:
    python tools/compare_runs.py
    python tools/compare_runs.py docs/log1.txt docs/log2.txt
    python tools/compare_runs.py --dir docs/
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RunMetrics:
    """Metrics extracted from a training run."""
    name: str
    total_episodes: int
    success_count: int
    success_rate: float
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    first_success_episode: Optional[int]
    final_100_success_rate: Optional[float]
    outcome_counts: Dict[str, int]


def parse_log_file(filepath: str) -> Optional[RunMetrics]:
    """Parse a training log file and extract metrics.

    Args:
        filepath: Path to the log file

    Returns:
        RunMetrics object or None if parsing failed
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Extract episode results
    # Pattern: Run 123 ✓ LANDED_SOFTLY ... Reward: 245.3 (env: 250.1 / shaped: -4.8)
    episode_pattern = r"Run (\d+) ([✓✗]) (\w+).*?Reward: ([\d.-]+) \(env: ([\d.-]+)"
    matches = re.findall(episode_pattern, content)

    if not matches:
        # Try alternate patterns
        alt_pattern = r"Run (\d+) ([✓✗]) (\w+)"
        matches = re.findall(alt_pattern, content)
        if matches:
            # Use partial data
            matches = [(m[0], m[1], m[2], '0', '0') for m in matches]

    if not matches:
        print(f"No episode data found in {filepath}")
        return None

    # Extract data
    episodes = []
    outcomes = []
    env_rewards = []
    first_success = None

    for match in matches:
        episode_num = int(match[0])
        success = match[1] == '✓'
        outcome = match[2]
        env_reward = float(match[4]) if len(match) > 4 else 0.0

        episodes.append(episode_num)
        outcomes.append(outcome)
        env_rewards.append(env_reward)

        if success and first_success is None:
            first_success = episode_num

    # Compute metrics
    import numpy as np
    env_rewards = np.array(env_rewards)
    successes = env_rewards >= 200  # Success threshold

    # Count outcomes
    outcome_counts = {}
    for outcome in outcomes:
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    # Final 100 success rate
    final_100_rate = None
    if len(successes) >= 100:
        final_100_rate = float(np.mean(successes[-100:])) * 100

    return RunMetrics(
        name=Path(filepath).stem,
        total_episodes=len(episodes),
        success_count=int(np.sum(successes)),
        success_rate=float(np.mean(successes)) * 100,
        mean_reward=float(np.mean(env_rewards)),
        std_reward=float(np.std(env_rewards)),
        max_reward=float(np.max(env_rewards)),
        min_reward=float(np.min(env_rewards)),
        first_success_episode=first_success,
        final_100_success_rate=final_100_rate,
        outcome_counts=outcome_counts
    )


def find_log_files(search_paths: List[str]) -> List[str]:
    """Find log files in the given paths.

    Args:
        search_paths: List of file paths or directories to search

    Returns:
        List of log file paths
    """
    log_files = []

    for path in search_paths:
        p = Path(path)
        if p.is_file():
            log_files.append(str(p))
        elif p.is_dir():
            # Search for .txt files in directory
            log_files.extend([str(f) for f in p.glob('*.txt')])
            # Also check subdirectories
            log_files.extend([str(f) for f in p.glob('**/*.txt')])

    return sorted(set(log_files))


def print_comparison_table(metrics_list: List[RunMetrics]) -> None:
    """Print a comparison table of run metrics.

    Args:
        metrics_list: List of RunMetrics objects to compare
    """
    if not metrics_list:
        print("No runs to compare")
        return

    # Sort by success rate descending
    metrics_list.sort(key=lambda m: m.success_rate, reverse=True)

    # Print header
    print("\n" + "=" * 100)
    print("TRAINING RUN COMPARISON")
    print("=" * 100)

    # Table header
    header = f"{'Run Name':<30} {'Episodes':<10} {'Success%':<10} {'Mean Rew':<10} {'Max Rew':<10} {'1st Success':<12}"
    print(header)
    print("-" * 100)

    # Table rows
    for m in metrics_list:
        first_success_str = str(m.first_success_episode) if m.first_success_episode else "N/A"
        name = m.name[:28] if len(m.name) > 28 else m.name
        row = f"{name:<30} {m.total_episodes:<10} {m.success_rate:<10.1f} {m.mean_reward:<10.1f} {m.max_reward:<10.1f} {first_success_str:<12}"
        print(row)

    print("-" * 100)

    # Best configuration
    best = metrics_list[0]
    print(f"\nBest configuration: {best.name}")
    print(f"  Success rate: {best.success_rate:.1f}%")
    print(f"  Mean reward: {best.mean_reward:.1f}")
    if best.first_success_episode:
        print(f"  First success: Episode {best.first_success_episode}")
    if best.final_100_success_rate:
        print(f"  Final 100 episode success rate: {best.final_100_success_rate:.1f}%")

    # Outcome distribution for best run
    if best.outcome_counts:
        print("\n  Outcome distribution:")
        sorted_outcomes = sorted(best.outcome_counts.items(), key=lambda x: x[1], reverse=True)
        for outcome, count in sorted_outcomes[:10]:
            pct = count / best.total_episodes * 100
            print(f"    {outcome}: {count} ({pct:.1f}%)")


def print_detailed_comparison(metrics_list: List[RunMetrics]) -> None:
    """Print detailed statistics for each run.

    Args:
        metrics_list: List of RunMetrics objects
    """
    print("\n" + "=" * 100)
    print("DETAILED STATISTICS")
    print("=" * 100)

    for m in metrics_list:
        print(f"\n{m.name}")
        print("-" * 50)
        print(f"  Episodes: {m.total_episodes}")
        print(f"  Successes: {m.success_count} ({m.success_rate:.1f}%)")
        print(f"  Rewards: mean={m.mean_reward:.1f}, std={m.std_reward:.1f}")
        print(f"           min={m.min_reward:.1f}, max={m.max_reward:.1f}")
        if m.first_success_episode:
            print(f"  First success: Episode {m.first_success_episode}")
        if m.final_100_success_rate:
            print(f"  Final 100 success rate: {m.final_100_success_rate:.1f}%")

        # Categorize outcomes
        if m.outcome_counts:
            landed = sum(v for k, v in m.outcome_counts.items() if 'LANDED' in k)
            crashed = sum(v for k, v in m.outcome_counts.items() if 'CRASHED' in k)
            flew_off = sum(v for k, v in m.outcome_counts.items() if 'FLEW_OFF' in k)
            timed_out = sum(v for k, v in m.outcome_counts.items() if 'TIMED_OUT' in k)

            print(f"  Outcome categories:")
            print(f"    Landed: {landed} ({landed/m.total_episodes*100:.1f}%)")
            print(f"    Crashed: {crashed} ({crashed/m.total_episodes*100:.1f}%)")
            print(f"    Flew off: {flew_off} ({flew_off/m.total_episodes*100:.1f}%)")
            print(f"    Timed out: {timed_out} ({timed_out/m.total_episodes*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Compare training runs')
    parser.add_argument('files', nargs='*', help='Log files or directories to compare')
    parser.add_argument('--dir', '-d', type=str, help='Directory to search for logs')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')

    args = parser.parse_args()

    # Determine search paths
    search_paths = args.files if args.files else []
    if args.dir:
        search_paths.append(args.dir)

    # Default search locations if nothing specified
    if not search_paths:
        default_paths = ['docs', 'sweep_results']
        search_paths = [p for p in default_paths if os.path.exists(p)]
        if not search_paths:
            print("No log files specified and no default directories found.")
            print("Usage: python tools/compare_runs.py docs/log1.txt docs/log2.txt")
            print("       python tools/compare_runs.py --dir docs/")
            sys.exit(1)

    # Find log files
    log_files = find_log_files(search_paths)

    if not log_files:
        print(f"No log files found in: {search_paths}")
        sys.exit(1)

    print(f"Found {len(log_files)} log files")

    # Parse each file
    metrics_list = []
    for filepath in log_files:
        metrics = parse_log_file(filepath)
        if metrics:
            metrics_list.append(metrics)

    if not metrics_list:
        print("No valid log data found")
        sys.exit(1)

    # Print comparison
    print_comparison_table(metrics_list)

    if args.detailed:
        print_detailed_comparison(metrics_list)


if __name__ == '__main__':
    main()
