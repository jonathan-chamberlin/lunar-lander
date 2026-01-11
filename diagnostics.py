"""Diagnostics tracking and reporting for TD3 training."""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from data_types import (
    EpisodeResult,
    ActionStatistics,
    TrainingMetrics,
    AggregatedTrainingMetrics
)
from behavior_analysis import BehaviorReport

logger = logging.getLogger(__name__)


# Behavior categories for grouping
OUTCOME_CATEGORIES = {
    'landed': ['LANDED_PERFECTLY', 'LANDED_SOFTLY', 'LANDED_HARD', 'LANDED_TILTED', 'LANDED_ONE_LEG'],
    'crashed': ['CRASHED_FAST_VERTICAL', 'CRASHED_FAST_TILTED', 'CRASHED_SIDEWAYS', 'CRASHED_SPINNING'],
    'timed_out': ['TIMED_OUT_HOVERING', 'TIMED_OUT_DESCENDING', 'TIMED_OUT_ASCENDING'],
    'flew_off': ['FLEW_OFF_TOP', 'FLEW_OFF_LEFT', 'FLEW_OFF_RIGHT', 'FLEW_OFF_LEFT_TILTED', 'FLEW_OFF_RIGHT_TILTED'],
}

QUALITY_BEHAVIORS = {
    'good': ['STAYED_UPRIGHT', 'STAYED_CENTERED', 'CONTROLLED_DESCENT', 'CONTROLLED_THROUGHOUT',
             'SMOOTH_THRUST', 'TOUCHED_DOWN_CLEAN', 'DIRECT_APPROACH', 'GRADUAL_SLOWDOWN',
             'REACHED_LOW_ALTITUDE', 'RECOVERED_FROM_TILT', 'RETURNED_TO_CENTER'],
    'bad': ['NEVER_STABILIZED', 'LOST_CONTROL_LATE', 'OVERCORRECTED_TO_CRASH', 'SPINNING_UNCONTROLLED',
            'FLIPPED_OVER', 'ERRATIC_THRUST', 'FREEFALL', 'NO_CONTACT_MADE', 'STAYED_HIGH'],
}


@dataclass
class BehaviorStatistics:
    """Comprehensive behavior statistics computed from tracked behavior reports."""

    # Outcome distribution
    total_episodes: int
    outcome_counts: Dict[str, int]
    outcome_category_counts: Dict[str, int]  # landed/crashed/timed_out/flew_off

    # Behavior frequencies
    behavior_counts: Dict[str, int]
    top_behaviors: List[Tuple[str, int, float]]  # (behavior, count, percentage)

    # Quality metrics
    good_behavior_rate: float  # % of runs with good behaviors
    bad_behavior_rate: float   # % of runs with bad behaviors

    # Progress indicators
    low_altitude_rate: float   # % reaching low altitude
    contact_rate: float        # % making any leg contact
    clean_touchdown_rate: float  # % with clean touchdown

    # Flight quality
    stayed_upright_rate: float
    stayed_centered_rate: float
    controlled_descent_rate: float
    controlled_throughout_rate: float
    never_stabilized_rate: float

    # Batch trends (per 50 episodes)
    batch_success_rates: List[float]
    batch_outcome_distributions: List[Dict[str, float]]
    batch_low_altitude_rates: List[float]
    batch_contact_rates: List[float]

    # Failure analysis
    crash_type_distribution: Dict[str, float]

    # Success correlation
    success_behavior_rates: Dict[str, float]  # behavior rates in successful runs
    failure_behavior_rates: Dict[str, float]  # behavior rates in failed runs


@dataclass
class DiagnosticsSummary:
    """Summary statistics computed from tracked metrics."""

    # Episode statistics
    total_episodes: int
    num_successes: int
    success_rate: float
    mean_reward: float
    max_reward: float
    min_reward: float
    final_50_mean_reward: Optional[float]

    # Action statistics
    mean_main_thruster: Optional[float]
    mean_side_thruster: Optional[float]
    mean_action_magnitude: Optional[float]
    high_thruster_episode_ratio: Optional[float]

    # Training statistics
    mean_q_value: Optional[float]
    q_value_trend: Optional[tuple]  # (first_10_avg, last_10_avg)
    mean_actor_loss: Optional[float]
    mean_critic_loss: Optional[float]
    mean_actor_grad_norm: Optional[float]
    mean_critic_grad_norm: Optional[float]


@dataclass
class BatchSpeedMetrics:
    """Speed metrics for a batch of episodes."""
    batch_num: int
    elapsed_time: float
    total_steps: int
    total_training_updates: int
    sps: float  # Steps per second
    ups: float  # Updates per second


class DiagnosticsTracker:
    """Collects and stores training metrics without performing I/O.

    This class is responsible for accumulating metrics during training.
    Use DiagnosticsReporter for outputting the collected data.
    """

    def __init__(self) -> None:
        # Episode results
        self.episode_results: List[EpisodeResult] = []
        self.successes: List[int] = []

        # Action statistics per episode
        self.action_stats: List[ActionStatistics] = []

        # Behavior reports per episode
        self.behavior_reports: List[BehaviorReport] = []

        # Training metrics (recorded periodically, not every episode)
        self.q_values: List[float] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.actor_grad_norms: List[float] = []
        self.critic_grad_norms: List[float] = []

        # Speed metrics per batch (recorded every 50 episodes)
        self.batch_speed_metrics: List[BatchSpeedMetrics] = []

    def record_episode(self, result: EpisodeResult) -> None:
        """Record a completed episode result.

        Args:
            result: The episode result to record
        """
        self.episode_results.append(result)
        if result.success:
            self.successes.append(result.episode_num)

    def record_action_stats(self, stats: ActionStatistics) -> None:
        """Record action statistics for an episode.

        Args:
            stats: Action statistics computed from episode actions
        """
        self.action_stats.append(stats)

    def record_behavior(self, report: BehaviorReport, success: bool) -> None:
        """Record behavior report for an episode.

        Args:
            report: Behavior report from behavior analysis
            success: Whether the episode was successful
        """
        self.behavior_reports.append(report)

    def record_training_metrics(self, metrics: AggregatedTrainingMetrics) -> None:
        """Record aggregated training metrics.

        Args:
            metrics: Aggregated metrics from training updates
        """
        self.q_values.append(metrics.avg_q_value)
        self.actor_losses.append(metrics.actor_loss)
        self.critic_losses.append(metrics.critic_loss)
        self.actor_grad_norms.append(metrics.actor_grad_norm)
        self.critic_grad_norms.append(metrics.critic_grad_norm)

    def record_batch_speed(
        self,
        batch_num: int,
        elapsed_time: float,
        total_steps: int,
        total_training_updates: int
    ) -> None:
        """Record speed metrics for a batch of episodes.

        Args:
            batch_num: Batch number (e.g., 1 for episodes 1-50, 2 for 51-100)
            elapsed_time: Total elapsed time since training started
            total_steps: Total environment steps taken so far
            total_training_updates: Total training updates performed so far
        """
        sps = total_steps / elapsed_time if elapsed_time > 0 else 0.0
        ups = total_training_updates / elapsed_time if elapsed_time > 0 else 0.0

        self.batch_speed_metrics.append(BatchSpeedMetrics(
            batch_num=batch_num,
            elapsed_time=elapsed_time,
            total_steps=total_steps,
            total_training_updates=total_training_updates,
            sps=sps,
            ups=ups
        ))

    def get_rewards(self) -> List[float]:
        """Get list of total rewards for all episodes."""
        return [r.total_reward for r in self.episode_results]

    def get_summary(self) -> DiagnosticsSummary:
        """Compute summary statistics from all tracked metrics.

        Returns:
            DiagnosticsSummary with computed statistics
        """
        rewards = self.get_rewards()
        num_episodes = len(rewards)

        if num_episodes == 0:
            return DiagnosticsSummary(
                total_episodes=0, num_successes=0, success_rate=0.0,
                mean_reward=0.0, max_reward=0.0, min_reward=0.0,
                final_50_mean_reward=None, mean_main_thruster=None,
                mean_side_thruster=None, mean_action_magnitude=None,
                high_thruster_episode_ratio=None, mean_q_value=None,
                q_value_trend=None, mean_actor_loss=None,
                mean_critic_loss=None, mean_actor_grad_norm=None,
                mean_critic_grad_norm=None
            )

        # Episode statistics
        final_50_mean = (
            float(np.mean(rewards[-50:]))
            if num_episodes >= 50 else None
        )

        # Action statistics
        mean_main = None
        mean_side = None
        mean_magnitude = None
        high_thruster_ratio = None

        if self.action_stats:
            mean_main = float(np.mean([s.mean_main_thruster for s in self.action_stats]))
            mean_side = float(np.mean([s.mean_side_thruster for s in self.action_stats]))
            mean_magnitude = float(np.mean([s.mean_magnitude for s in self.action_stats]))
            high_thruster_count = sum(
                1 for s in self.action_stats if s.mean_main_thruster > 0.5
            )
            high_thruster_ratio = high_thruster_count / len(self.action_stats)

        # Training statistics
        mean_q = None
        q_trend = None
        mean_actor_loss = None
        mean_critic_loss = None
        mean_actor_grad = None
        mean_critic_grad = None

        if self.q_values:
            mean_q = float(np.mean(self.q_values))
            if len(self.q_values) >= 10:
                q_trend = (
                    float(np.mean(self.q_values[:10])),
                    float(np.mean(self.q_values[-10:]))
                )
            mean_actor_loss = float(np.mean(self.actor_losses))
            mean_critic_loss = float(np.mean(self.critic_losses))
            mean_actor_grad = float(np.mean(self.actor_grad_norms))
            mean_critic_grad = float(np.mean(self.critic_grad_norms))

        return DiagnosticsSummary(
            total_episodes=num_episodes,
            num_successes=len(self.successes),
            success_rate=len(self.successes) / num_episodes,
            mean_reward=float(np.mean(rewards)),
            max_reward=float(np.max(rewards)),
            min_reward=float(np.min(rewards)),
            final_50_mean_reward=final_50_mean,
            mean_main_thruster=mean_main,
            mean_side_thruster=mean_side,
            mean_action_magnitude=mean_magnitude,
            high_thruster_episode_ratio=high_thruster_ratio,
            mean_q_value=mean_q,
            q_value_trend=q_trend,
            mean_actor_loss=mean_actor_loss,
            mean_critic_loss=mean_critic_loss,
            mean_actor_grad_norm=mean_actor_grad,
            mean_critic_grad_norm=mean_critic_grad
        )

    def get_behavior_statistics(self, batch_size: int = 50) -> Optional[BehaviorStatistics]:
        """Compute comprehensive behavior statistics.

        Args:
            batch_size: Size of batches for trend analysis

        Returns:
            BehaviorStatistics if behavior data exists, None otherwise
        """
        if not self.behavior_reports:
            return None

        num_episodes = len(self.behavior_reports)

        # Count outcomes
        outcome_counts: Counter = Counter()
        for report in self.behavior_reports:
            outcome_counts[report.outcome] += 1

        # Count outcome categories
        outcome_category_counts: Dict[str, int] = {cat: 0 for cat in OUTCOME_CATEGORIES}
        for outcome, count in outcome_counts.items():
            for category, outcomes in OUTCOME_CATEGORIES.items():
                if outcome in outcomes:
                    outcome_category_counts[category] += count
                    break

        # Count all behaviors
        behavior_counts: Counter = Counter()
        for report in self.behavior_reports:
            for behavior in report.behaviors:
                behavior_counts[behavior] += 1

        # Top behaviors
        top_behaviors = [
            (behavior, count, count / num_episodes * 100)
            for behavior, count in behavior_counts.most_common(15)
        ]

        # Quality metrics
        good_count = sum(
            1 for report in self.behavior_reports
            if any(b in QUALITY_BEHAVIORS['good'] for b in report.behaviors)
        )
        bad_count = sum(
            1 for report in self.behavior_reports
            if any(b in QUALITY_BEHAVIORS['bad'] for b in report.behaviors)
        )

        # Progress indicators
        low_altitude_count = behavior_counts.get('REACHED_LOW_ALTITUDE', 0)
        contact_behaviors = ['TOUCHED_DOWN_CLEAN', 'SCRAPED_LEFT_LEG', 'SCRAPED_RIGHT_LEG',
                            'BOUNCED', 'PROLONGED_ONE_LEG', 'MULTIPLE_TOUCHDOWNS']
        contact_count = sum(
            1 for report in self.behavior_reports
            if any(b in contact_behaviors for b in report.behaviors)
        )
        clean_touchdown_count = behavior_counts.get('TOUCHED_DOWN_CLEAN', 0)

        # Flight quality rates
        stayed_upright_count = behavior_counts.get('STAYED_UPRIGHT', 0)
        stayed_centered_count = behavior_counts.get('STAYED_CENTERED', 0)
        controlled_descent_count = behavior_counts.get('CONTROLLED_DESCENT', 0)
        controlled_throughout_count = behavior_counts.get('CONTROLLED_THROUGHOUT', 0)
        never_stabilized_count = behavior_counts.get('NEVER_STABILIZED', 0)

        # Batch trends
        batch_success_rates: List[float] = []
        batch_outcome_distributions: List[Dict[str, float]] = []
        batch_low_altitude_rates: List[float] = []
        batch_contact_rates: List[float] = []

        num_batches = (num_episodes + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_episodes)
            batch_reports = self.behavior_reports[start:end]
            batch_results = self.episode_results[start:end] if start < len(self.episode_results) else []
            batch_len = len(batch_reports)

            if batch_len == 0:
                continue

            # Success rate for batch
            if batch_results:
                batch_successes = sum(1 for r in batch_results if r.success)
                batch_success_rates.append(batch_successes / len(batch_results) * 100)
            else:
                batch_success_rates.append(0.0)

            # Outcome distribution for batch
            batch_outcomes: Counter = Counter()
            for report in batch_reports:
                for category, outcomes in OUTCOME_CATEGORIES.items():
                    if report.outcome in outcomes:
                        batch_outcomes[category] += 1
                        break
            batch_outcome_distributions.append({
                cat: batch_outcomes.get(cat, 0) / batch_len * 100
                for cat in OUTCOME_CATEGORIES
            })

            # Low altitude rate for batch
            batch_low_alt = sum(
                1 for r in batch_reports
                if 'REACHED_LOW_ALTITUDE' in r.behaviors
            )
            batch_low_altitude_rates.append(batch_low_alt / batch_len * 100)

            # Contact rate for batch
            batch_contact = sum(
                1 for r in batch_reports
                if any(b in contact_behaviors for b in r.behaviors)
            )
            batch_contact_rates.append(batch_contact / batch_len * 100)

        # Crash type distribution
        crash_outcomes = OUTCOME_CATEGORIES['crashed']
        total_crashes = sum(outcome_counts.get(o, 0) for o in crash_outcomes)
        crash_type_distribution = {}
        if total_crashes > 0:
            for crash_type in crash_outcomes:
                crash_type_distribution[crash_type] = outcome_counts.get(crash_type, 0) / total_crashes * 100

        # Success correlation - behavior rates in successful vs failed runs
        success_behaviors: Counter = Counter()
        failure_behaviors: Counter = Counter()
        success_count = 0
        failure_count = 0

        for i, report in enumerate(self.behavior_reports):
            if i < len(self.episode_results):
                if self.episode_results[i].success:
                    success_count += 1
                    for behavior in report.behaviors:
                        success_behaviors[behavior] += 1
                else:
                    failure_count += 1
                    for behavior in report.behaviors:
                        failure_behaviors[behavior] += 1

        success_behavior_rates = {}
        failure_behavior_rates = {}
        key_behaviors = ['STAYED_UPRIGHT', 'STAYED_CENTERED', 'CONTROLLED_DESCENT',
                        'REACHED_LOW_ALTITUDE', 'TOUCHED_DOWN_CLEAN', 'CONTROLLED_THROUGHOUT',
                        'NEVER_STABILIZED', 'RAPID_DESCENT', 'HOVER_MAINTAINED',
                        'ERRATIC_THRUST', 'SMOOTH_THRUST', 'NO_CONTACT_MADE']

        for behavior in key_behaviors:
            if success_count > 0:
                success_behavior_rates[behavior] = success_behaviors.get(behavior, 0) / success_count * 100
            else:
                success_behavior_rates[behavior] = 0.0
            if failure_count > 0:
                failure_behavior_rates[behavior] = failure_behaviors.get(behavior, 0) / failure_count * 100
            else:
                failure_behavior_rates[behavior] = 0.0

        return BehaviorStatistics(
            total_episodes=num_episodes,
            outcome_counts=dict(outcome_counts),
            outcome_category_counts=outcome_category_counts,
            behavior_counts=dict(behavior_counts),
            top_behaviors=top_behaviors,
            good_behavior_rate=good_count / num_episodes * 100,
            bad_behavior_rate=bad_count / num_episodes * 100,
            low_altitude_rate=low_altitude_count / num_episodes * 100,
            contact_rate=contact_count / num_episodes * 100,
            clean_touchdown_rate=clean_touchdown_count / num_episodes * 100,
            stayed_upright_rate=stayed_upright_count / num_episodes * 100,
            stayed_centered_rate=stayed_centered_count / num_episodes * 100,
            controlled_descent_rate=controlled_descent_count / num_episodes * 100,
            controlled_throughout_rate=controlled_throughout_count / num_episodes * 100,
            never_stabilized_rate=never_stabilized_count / num_episodes * 100,
            batch_success_rates=batch_success_rates,
            batch_outcome_distributions=batch_outcome_distributions,
            batch_low_altitude_rates=batch_low_altitude_rates,
            batch_contact_rates=batch_contact_rates,
            crash_type_distribution=crash_type_distribution,
            success_behavior_rates=success_behavior_rates,
            failure_behavior_rates=failure_behavior_rates,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert all tracked data to a dictionary for serialization."""
        return {
            'episode_results': [asdict(r) for r in self.episode_results],
            'successes': self.successes,
            'action_stats': [asdict(s) for s in self.action_stats],
            'q_values': self.q_values,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'actor_grad_norms': self.actor_grad_norms,
            'critic_grad_norms': self.critic_grad_norms
        }


class DiagnosticsReporter:
    """Handles output of diagnostics to various destinations.

    Separates I/O concerns from data collection.
    """

    def __init__(self, tracker: DiagnosticsTracker) -> None:
        self.tracker = tracker

    def print_summary(self) -> None:
        """Print comprehensive diagnostic summary to console."""
        summary = self.tracker.get_summary()

        print("\n" + "=" * 80)
        print("TRAINING DIAGNOSTICS SUMMARY")
        print("=" * 80)

        # Reward statistics
        print(f"\n--- REWARD STATISTICS ---")
        print(f"Total episodes: {summary.total_episodes}")
        print(f"Successes: {summary.num_successes} (episodes: {self.tracker.successes})")
        print(f"Success rate: {summary.success_rate * 100:.1f}%")
        print(f"Mean reward: {summary.mean_reward:.2f}")
        print(f"Max reward: {summary.max_reward:.2f}")
        print(f"Min reward: {summary.min_reward:.2f}")

        if summary.final_50_mean_reward is not None:
            print(f"Final 50 episodes mean reward: {summary.final_50_mean_reward:.2f}")

        # Action statistics
        print(f"\n--- ACTION STATISTICS ---")
        if summary.mean_main_thruster is not None:
            print(f"Episodes with action tracking: {len(self.tracker.action_stats)}")
            print(f"Mean main thruster: {summary.mean_main_thruster:.3f}")
            print(f"Mean side thruster: {summary.mean_side_thruster:.3f}")
            print(f"Mean action magnitude: {summary.mean_action_magnitude:.3f}")

            if len(self.tracker.action_stats) >= 50:
                recent_main = np.mean([
                    s.mean_main_thruster for s in self.tracker.action_stats[-50:]
                ])
                recent_side = np.mean([
                    s.mean_side_thruster for s in self.tracker.action_stats[-50:]
                ])
                recent_mag = np.mean([
                    s.mean_magnitude for s in self.tracker.action_stats[-50:]
                ])
                print(f"\nLast 50 episodes:")
                print(f"  Main thruster: {recent_main:.3f}")
                print(f"  Side thruster: {recent_side:.3f}")
                print(f"  Action magnitude: {recent_mag:.3f}")

            high_count = int(summary.high_thruster_episode_ratio * len(self.tracker.action_stats))
            print(f"\nEpisodes with high main thruster (>0.5): "
                  f"{high_count}/{len(self.tracker.action_stats)}")
        else:
            print("No action data collected (training hasn't started yet)")

        # Training metrics
        print(f"\n--- TRAINING METRICS ---")
        if summary.mean_q_value is not None:
            print(f"Mean Q-value: {summary.mean_q_value:.3f}")
            if summary.q_value_trend:
                print(f"Q-value trend (first 10 vs last 10): "
                      f"{summary.q_value_trend[0]:.3f} -> {summary.q_value_trend[1]:.3f}")
            print(f"Mean actor loss: {summary.mean_actor_loss:.4f}")
            print(f"Mean critic loss: {summary.mean_critic_loss:.4f}")
            print(f"Mean actor gradient norm: {summary.mean_actor_grad_norm:.4f}")
            print(f"Mean critic gradient norm: {summary.mean_critic_grad_norm:.4f}")

            high_loss_count = sum(1 for x in self.tracker.actor_losses if x > 1.0)
            print(f"\nEpisodes with high actor loss (>1.0): "
                  f"{high_loss_count}/{len(self.tracker.actor_losses)}")
        else:
            print("No training metrics collected yet")

        # Behavior statistics
        self._print_behavior_statistics()

        # Recent episodes
        self._print_recent_episodes()

        # Key data
        self._print_key_data()

        print("\n" + "=" * 80)
        print("END OF DIAGNOSTICS")
        print("=" * 80)

    def _print_recent_episodes(self, n: int = 5) -> None:
        """Print details of recent episodes."""
        print(f"\n--- LAST {n} EPISODES DETAIL ---")

        results = self.tracker.episode_results
        if not results:
            print("No episodes recorded")
            return

        start_idx = max(0, len(results) - n)
        for result in results[start_idx:]:
            info_str = str(result)

            # Try to find corresponding action stats
            stats_offset = len(results) - len(self.tracker.action_stats)
            stats_idx = result.episode_num - stats_offset

            if 0 <= stats_idx < len(self.tracker.action_stats):
                stats = self.tracker.action_stats[stats_idx]
                q_idx = stats_idx if stats_idx < len(self.tracker.q_values) else -1
                q_val = self.tracker.q_values[q_idx] if q_idx >= 0 else 0

                info_str += (f", Main: {stats.mean_main_thruster:.2f}, "
                            f"Side: {stats.mean_side_thruster:.2f}, Q: {q_val:.2f}")

            print(info_str)

    def _print_key_data(self) -> None:
        """Print key data counts for analysis."""
        rewards = self.tracker.get_rewards()
        print(f"\nData points collected: {len(rewards)} episodes, "
              f"{len(self.tracker.action_stats)} action stats, "
              f"{len(self.tracker.q_values)} training logs")

    def _print_behavior_statistics(self) -> None:
        """Print comprehensive behavior analysis."""
        stats = self.tracker.get_behavior_statistics()

        print(f"\n--- BEHAVIOR ANALYSIS ---")
        
        if stats is None:
            print("No behavior data collected")
            return


        print(f"Episodes analyzed: {stats.total_episodes}")

        # Outcome distribution
        print(f"\n  OUTCOME DISTRIBUTION:")
        total = stats.total_episodes
        for category in ['landed', 'crashed', 'timed_out', 'flew_off']:
            count = stats.outcome_category_counts.get(category, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"    {category.upper():12} {count:4} ({pct:5.1f}%)")

        # Detailed outcome breakdown
        print(f"\n  OUTCOME DETAILS:")
        sorted_outcomes = sorted(stats.outcome_counts.items(), key=lambda x: -x[1])
        for outcome, count in sorted_outcomes[:8]:
            pct = count / total * 100 if total > 0 else 0
            print(f"    {outcome:24} {count:4} ({pct:5.1f}%)")

        # Crash type breakdown (if any crashes)
        if stats.crash_type_distribution:
            print(f"\n  CRASH TYPE BREAKDOWN:")
            for crash_type, pct in sorted(stats.crash_type_distribution.items(), key=lambda x: -x[1]):
                print(f"    {crash_type:24} {pct:5.1f}%")

        # Top behaviors
        print(f"\n  TOP 15 BEHAVIORS:")
        for behavior, count, pct in stats.top_behaviors:
            print(f"    {behavior:28} {count:4} ({pct:5.1f}%)")

        # Flight quality metrics
        print(f"\n  FLIGHT QUALITY METRICS:")
        print(f"    Stayed upright:          {stats.stayed_upright_rate:5.1f}%")
        print(f"    Stayed centered:         {stats.stayed_centered_rate:5.1f}%")
        print(f"    Controlled descent:      {stats.controlled_descent_rate:5.1f}%")
        print(f"    Controlled throughout:   {stats.controlled_throughout_rate:5.1f}%")
        print(f"    Never stabilized:        {stats.never_stabilized_rate:5.1f}%")

        # Progress indicators
        print(f"\n  PROGRESS INDICATORS:")
        print(f"    Reached low altitude:    {stats.low_altitude_rate:5.1f}%")
        print(f"    Made leg contact:        {stats.contact_rate:5.1f}%")
        print(f"    Clean touchdown:         {stats.clean_touchdown_rate:5.1f}%")

        # Quality summary
        print(f"\n  QUALITY SUMMARY:")
        print(f"    Runs with good behaviors: {stats.good_behavior_rate:5.1f}%")
        print(f"    Runs with bad behaviors:  {stats.bad_behavior_rate:5.1f}%")

        # Batch trends
        if len(stats.batch_success_rates) > 1:
            print(f"\n  BATCH TRENDS (per 50 episodes):")
            # Check if we have speed metrics
            has_speed = len(self.tracker.batch_speed_metrics) > 0
            if has_speed:
                print(f"    {'Batch':<8} {'Success%':>9} {'Landed%':>9} {'Crashed%':>9} {'SPS':>8} {'UPS':>8}")
                print(f"    {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
            else:
                print(f"    {'Batch':<8} {'Success%':>9} {'Landed%':>9} {'Crashed%':>9} {'LowAlt%':>9} {'Contact%':>9}")
                print(f"    {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

            for i, (success_rate, outcome_dist, low_alt, contact) in enumerate(zip(
                stats.batch_success_rates,
                stats.batch_outcome_distributions,
                stats.batch_low_altitude_rates,
                stats.batch_contact_rates
            )):
                batch_label = f"{i*50+1}-{min((i+1)*50, stats.total_episodes)}"
                landed_pct = outcome_dist.get('landed', 0)
                crashed_pct = outcome_dist.get('crashed', 0)

                if has_speed and i < len(self.tracker.batch_speed_metrics):
                    speed = self.tracker.batch_speed_metrics[i]
                    print(f"    {batch_label:<8} {success_rate:>8.1f}% {landed_pct:>8.1f}% {crashed_pct:>8.1f}% {speed.sps:>7.0f} {speed.ups:>7.0f}")
                else:
                    print(f"    {batch_label:<8} {success_rate:>8.1f}% {landed_pct:>8.1f}% {crashed_pct:>8.1f}% {low_alt:>8.1f}% {contact:>8.1f}%")

        # Success correlation
        print(f"\n  BEHAVIOR CORRELATION WITH SUCCESS:")
        print(f"    {'Behavior':<28} {'In Success':>12} {'In Failure':>12} {'Delta':>8}")
        print(f"    {'-'*28} {'-'*12} {'-'*12} {'-'*8}")

        # Sort by delta (difference between success and failure rates)
        correlations = []
        for behavior in stats.success_behavior_rates:
            success_rate = stats.success_behavior_rates[behavior]
            failure_rate = stats.failure_behavior_rates[behavior]
            delta = success_rate - failure_rate
            correlations.append((behavior, success_rate, failure_rate, delta))

        # Sort by absolute delta to show most discriminating behaviors
        correlations.sort(key=lambda x: -abs(x[3]))

        for behavior, success_rate, failure_rate, delta in correlations:
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            print(f"    {behavior:<28} {success_rate:>11.1f}% {failure_rate:>11.1f}% {delta_str:>8}")

    def save_to_json(self, path: Path) -> None:
        """Save all tracked data to a JSON file.

        Args:
            path: Path to save the JSON file
        """
        data = self.tracker.to_dict()
        data['summary'] = asdict(self.tracker.get_summary())

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Diagnostics saved to {path}")

    def log_training_update(
        self,
        metrics: AggregatedTrainingMetrics,
        noise_scale: float
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Aggregated training metrics
            noise_scale: Current noise scale
        """
        logger.info(
            f"Training update - Critic Loss: {metrics.critic_loss:.4f}, "
            f"Actor Loss: {metrics.actor_loss:.4f}, "
            f"Avg Q: {metrics.avg_q_value:.3f}, "
            f"Noise: {noise_scale:.3f}"
        )
