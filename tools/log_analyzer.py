"""Training log analyzer for lunar lander TD3 training.

Parses training output logs and extracts metrics and insights.

Usage:
    python tools/log_analyzer.py docs/training_output.txt
    python tools/log_analyzer.py --recent
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EpisodeData:
    """Data extracted from a single episode."""
    episode_num: int
    success: bool
    outcome: str
    total_reward: float
    env_reward: float
    shaped_bonus: float
    behaviors: List[str] = field(default_factory=list)


@dataclass
class LogAnalysis:
    """Complete analysis of a training log."""
    filepath: str
    episodes: List[EpisodeData]
    total_episodes: int
    success_count: int
    success_rate: float
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    first_success_episode: Optional[int]
    outcome_distribution: Dict[str, int]
    behavior_distribution: Dict[str, int]
    learning_curve: List[Tuple[int, float, float]]  # (episode_batch, success_rate, mean_reward)


def parse_episode_line(line: str) -> Optional[EpisodeData]:
    """Parse a single episode result line.

    Args:
        line: Line from training log

    Returns:
        EpisodeData or None if line doesn't match
    """
    # Primary pattern with full reward info
    # Run 123 ✓ LANDED_SOFTLY ✅ Landed Safely 🥕 Reward: 245.3 (env: 250.1 / shaped: -4.8)
    full_pattern = r"Run (\d+) ([✓✗]) (\w+).*?Reward: ([\d.-]+) \(env: ([\d.-]+) / shaped: ([+\d.-]+)\)"
    match = re.search(full_pattern, line)

    if match:
        return EpisodeData(
            episode_num=int(match.group(1)),
            success=match.group(2) == '✓',
            outcome=match.group(3),
            total_reward=float(match.group(4)),
            env_reward=float(match.group(5)),
            shaped_bonus=float(match.group(6))
        )

    # Simplified pattern without shaped bonus
    simple_pattern = r"Run (\d+) ([✓✗]) (\w+).*?Reward: ([\d.-]+)"
    match = re.search(simple_pattern, line)

    if match:
        return EpisodeData(
            episode_num=int(match.group(1)),
            success=match.group(2) == '✓',
            outcome=match.group(3),
            total_reward=float(match.group(4)),
            env_reward=float(match.group(4)),
            shaped_bonus=0.0
        )

    return None


def parse_log_file(filepath: str) -> Optional[LogAnalysis]:
    """Parse a complete training log file.

    Args:
        filepath: Path to the log file

    Returns:
        LogAnalysis object or None if parsing failed
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    episodes = []

    for line in lines:
        episode_data = parse_episode_line(line)
        if episode_data:
            episodes.append(episode_data)

    if not episodes:
        print(f"No episode data found in {filepath}")
        return None

    # Compute statistics
    env_rewards = np.array([e.env_reward for e in episodes])
    successes = np.array([e.success for e in episodes])

    # First success
    first_success = None
    for e in episodes:
        if e.success:
            first_success = e.episode_num
            break

    # Outcome distribution
    outcome_dist = defaultdict(int)
    for e in episodes:
        outcome_dist[e.outcome] += 1

    # Behavior distribution (if we had behavior data)
    behavior_dist = defaultdict(int)
    for e in episodes:
        for b in e.behaviors:
            behavior_dist[b] += 1

    # Learning curve (batches of 50 episodes)
    batch_size = 50
    learning_curve = []
    for i in range(0, len(episodes), batch_size):
        batch_episodes = episodes[i:i+batch_size]
        if batch_episodes:
            batch_success = np.mean([e.success for e in batch_episodes]) * 100
            batch_reward = np.mean([e.env_reward for e in batch_episodes])
            learning_curve.append((i + batch_size, batch_success, batch_reward))

    return LogAnalysis(
        filepath=filepath,
        episodes=episodes,
        total_episodes=len(episodes),
        success_count=int(np.sum(successes)),
        success_rate=float(np.mean(successes)) * 100,
        mean_reward=float(np.mean(env_rewards)),
        std_reward=float(np.std(env_rewards)),
        max_reward=float(np.max(env_rewards)),
        min_reward=float(np.min(env_rewards)),
        first_success_episode=first_success,
        outcome_distribution=dict(outcome_dist),
        behavior_distribution=dict(behavior_dist),
        learning_curve=learning_curve
    )


def print_analysis(analysis: LogAnalysis) -> None:
    """Print formatted analysis report.

    Args:
        analysis: LogAnalysis object to print
    """
    print("\n" + "=" * 70)
    print("TRAINING LOG ANALYSIS")
    print("=" * 70)
    print(f"File: {analysis.filepath}")
    print(f"Episodes: {analysis.total_episodes}")

    print("\n--- Performance Metrics ---")
    print(f"Success Rate: {analysis.success_rate:.1f}% ({analysis.success_count}/{analysis.total_episodes})")
    print(f"Mean Reward: {analysis.mean_reward:.1f} (std: {analysis.std_reward:.1f})")
    print(f"Max Reward: {analysis.max_reward:.1f}")
    print(f"Min Reward: {analysis.min_reward:.1f}")
    if analysis.first_success_episode is not None:
        print(f"First Success: Episode {analysis.first_success_episode}")

    print("\n--- Outcome Distribution ---")
    sorted_outcomes = sorted(analysis.outcome_distribution.items(),
                            key=lambda x: x[1], reverse=True)
    for outcome, count in sorted_outcomes:
        pct = count / analysis.total_episodes * 100
        bar = '█' * int(pct / 5)
        print(f"  {outcome:<25} {count:>5} ({pct:>5.1f}%) {bar}")

    # Categorized outcomes
    landed = sum(v for k, v in analysis.outcome_distribution.items()
                 if 'LANDED' in k)
    crashed = sum(v for k, v in analysis.outcome_distribution.items()
                  if 'CRASHED' in k)
    flew_off = sum(v for k, v in analysis.outcome_distribution.items()
                   if 'FLEW_OFF' in k)
    timed_out = sum(v for k, v in analysis.outcome_distribution.items()
                    if 'TIMED_OUT' in k)

    print("\n--- Outcome Categories ---")
    total = analysis.total_episodes
    print(f"  Landed:    {landed:>5} ({landed/total*100:>5.1f}%)")
    print(f"  Crashed:   {crashed:>5} ({crashed/total*100:>5.1f}%)")
    print(f"  Flew off:  {flew_off:>5} ({flew_off/total*100:>5.1f}%)")
    print(f"  Timed out: {timed_out:>5} ({timed_out/total*100:>5.1f}%)")

    print("\n--- Learning Curve ---")
    print(f"{'Episodes':<12} {'Success%':<10} {'Mean Reward':<12}")
    print("-" * 34)
    for batch_end, success_pct, mean_rew in analysis.learning_curve:
        print(f"{batch_end:<12} {success_pct:<10.1f} {mean_rew:<12.1f}")

    # Calculate improvement
    if len(analysis.learning_curve) >= 2:
        first_batch = analysis.learning_curve[0]
        last_batch = analysis.learning_curve[-1]
        success_improvement = last_batch[1] - first_batch[1]
        reward_improvement = last_batch[2] - first_batch[2]
        print(f"\nImprovement over training:")
        print(f"  Success rate: {first_batch[1]:.1f}% -> {last_batch[1]:.1f}% ({success_improvement:+.1f}%)")
        print(f"  Mean reward: {first_batch[2]:.1f} -> {last_batch[2]:.1f} ({reward_improvement:+.1f})")


def find_recent_log(search_dir: str = 'docs') -> Optional[str]:
    """Find the most recently modified log file.

    Args:
        search_dir: Directory to search

    Returns:
        Path to most recent log file or None
    """
    search_path = Path(search_dir)
    if not search_path.exists():
        return None

    txt_files = list(search_path.glob('*.txt'))
    if not txt_files:
        return None

    # Sort by modification time
    txt_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(txt_files[0])


def main():
    parser = argparse.ArgumentParser(description='Analyze training log')
    parser.add_argument('file', nargs='?', help='Path to log file')
    parser.add_argument('--recent', '-r', action='store_true',
                        help='Analyze most recent log in docs/')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output in JSON format')

    args = parser.parse_args()

    # Determine file to analyze
    if args.file:
        filepath = args.file
    elif args.recent:
        filepath = find_recent_log()
        if not filepath:
            print("No log files found in docs/")
            sys.exit(1)
        print(f"Analyzing most recent log: {filepath}")
    else:
        # Try to find any log
        filepath = find_recent_log()
        if not filepath:
            print("Usage: python tools/log_analyzer.py <logfile.txt>")
            print("       python tools/log_analyzer.py --recent")
            sys.exit(1)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    # Parse and analyze
    analysis = parse_log_file(filepath)

    if not analysis:
        sys.exit(1)

    if args.json:
        import json
        output = {
            'filepath': analysis.filepath,
            'total_episodes': analysis.total_episodes,
            'success_rate': analysis.success_rate,
            'mean_reward': analysis.mean_reward,
            'std_reward': analysis.std_reward,
            'max_reward': analysis.max_reward,
            'min_reward': analysis.min_reward,
            'first_success_episode': analysis.first_success_episode,
            'outcome_distribution': analysis.outcome_distribution,
            'learning_curve': [
                {'episodes': e, 'success_rate': s, 'mean_reward': r}
                for e, s, r in analysis.learning_curve
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print_analysis(analysis)


if __name__ == '__main__':
    main()
