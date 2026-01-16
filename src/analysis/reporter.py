"""Diagnostics reporting for TD3 training.

Handles output of diagnostics to console, files, and logs.
Separates I/O concerns from data collection.
"""

import io
import json
import logging
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from data_types import AggregatedTrainingMetrics

if TYPE_CHECKING:
    from analysis.diagnostics import DiagnosticsTracker

logger = logging.getLogger(__name__)


class DiagnosticsReporter:
    """Handles output of diagnostics to various destinations.

    Separates I/O concerns from data collection.
    """

    def __init__(self, tracker: 'DiagnosticsTracker') -> None:
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
        print(f"Successes: {summary.num_successes}")
        print(f"Success rate: {summary.success_rate * 100:.1f}%")
        print(f"Mean reward: {summary.mean_reward:.2f}")
        print(f"Max reward: {summary.max_reward:.2f}")
        print(f"Min reward: {summary.min_reward:.2f}")

        if summary.final_50_mean_reward is not None:
            print(f"Final 50 episodes mean reward: {summary.final_50_mean_reward:.2f}")

        # Advanced statistics
        self._print_advanced_statistics()

        # Action statistics (removed for memory efficiency)
        print(f"\n--- ACTION STATISTICS ---")
        print("Action tracking disabled for memory efficiency")

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

        # Streak statistics
        self._print_streak_statistics()

        # Env reward distribution for landings
        self._print_env_reward_distribution()

        # Recent episodes
        self._print_recent_episodes()

        # Key data
        self._print_key_data()

        print("\n" + "=" * 80)
        print("END OF DIAGNOSTICS")
        print("=" * 80)

    def get_summary_text(self) -> str:
        """Get comprehensive diagnostic summary as a string.

        Returns:
            The same output as print_summary(), but as a string
        """
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.print_summary()
        return buffer.getvalue()

    def _print_advanced_statistics(self) -> None:
        """Print advanced training statistics."""
        stats = self.tracker.get_advanced_statistics()

        if not stats:
            return

        print(f"\n--- ADVANCED STATISTICS ---")

        # Env reward stats
        if 'env_reward' in stats:
            env = stats['env_reward']
            print(f"Env reward: mean={env['mean']:.1f}, std={env['std']:.1f}, "
                  f"range=[{env['min']:.1f}, {env['max']:.1f}]")

        # First success
        if stats.get('first_success_episode') is not None:
            print(f"First success (env >= 200): episode {stats['first_success_episode']}")
        else:
            print(f"First success (env >= 200): not yet achieved")

        # Near misses
        if 'near_misses' in stats:
            nm = stats['near_misses']
            print(f"Near misses (180-199): {nm['count']} episodes")

        # Rolling success rates
        if 'rolling_success_rate_50' in stats:
            print(f"Rolling success rate (last 50): {stats['rolling_success_rate_50']:.1f}%")
        if 'rolling_success_rate_100' in stats:
            print(f"Rolling success rate (last 100): {stats['rolling_success_rate_100']:.1f}%")

        # Recent vs overall
        if 'recent_100' in stats:
            recent = stats['recent_100']
            overall_mean = stats['env_reward']['mean']
            diff = recent['env_reward_mean'] - overall_mean
            trend = "↑" if diff > 5 else "↓" if diff < -5 else "→"
            print(f"Last 100 vs overall: env_reward {recent['env_reward_mean']:.1f} vs {overall_mean:.1f} ({trend})")

        # Max streak in recent episodes
        if 'max_streak_last_200' in stats:
            print(f"Max streak (last 200 episodes): {stats['max_streak_last_200']}")

        # Successful landing quality
        if 'successful_landings' in stats:
            sl = stats['successful_landings']
            print(f"\n  SUCCESSFUL LANDINGS (env >= 200):")
            print(f"    Count: {sl['count']}")
            print(f"    Env reward: mean={sl['env_reward_mean']:.1f}, std={sl['env_reward_std']:.1f}")
            print(f"    Range: [{sl['env_reward_min']:.1f}, {sl['env_reward_max']:.1f}]")

            if 'success_quality' in stats:
                sq = stats['success_quality']
                print(f"\n  SUCCESS QUALITY BEHAVIORS:")
                print(f"    Stayed upright:      {sq['stayed_upright']:5.1f}%")
                print(f"    Stayed centered:     {sq['stayed_centered']:5.1f}%")
                print(f"    Controlled descent:  {sq['controlled_descent']:5.1f}%")
                print(f"    Clean touchdown:     {sq['clean_touchdown']:5.1f}%")
                print(f"    Landed perfectly:    {sq['landed_perfectly']:5.1f}%")
                print(f"    Landed softly:       {sq['landed_softly']:5.1f}%")

    def _print_streak_statistics(self) -> None:
        """Print consecutive success streak statistics."""
        streak_stats = self.tracker.get_streak_statistics()

        print(f"\n--- CONSECUTIVE SUCCESS STREAK ---")
        print(f"Max streak: {streak_stats['max_streak']} (achieved at episode {streak_stats['max_streak_episode']})")
        print(f"Current streak: {streak_stats['current_streak']}")

        # Streak breaks analysis
        streak_breaks = streak_stats['streak_breaks']
        if streak_breaks:
            print(f"\n  STREAK BREAKS (streaks of 5+ that ended):")
            print(f"    {'Episode':<10} {'Streak':<8} {'Outcome':<24} {'Env Reward':<12}")
            print(f"    {'-'*10} {'-'*8} {'-'*24} {'-'*12}")

            # Show last 10 streak breaks
            for break_info in streak_breaks[-10:]:
                print(f"    {break_info['episode']:<10} {break_info['streak_length']:<8} "
                      f"{break_info['outcome']:<24} {break_info['env_reward']:<12.1f}")

            # Summarize what breaks streaks
            outcome_counts: Dict[str, int] = {}
            for break_info in streak_breaks:
                outcome = break_info['outcome']
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

            if len(streak_breaks) >= 3:
                print(f"\n  STREAK BREAK CAUSES (total {len(streak_breaks)} breaks):")
                sorted_outcomes = sorted(outcome_counts.items(), key=lambda x: -x[1])
                for outcome, count in sorted_outcomes[:5]:
                    pct = count / len(streak_breaks) * 100
                    print(f"    {outcome:<24} {count:3} ({pct:5.1f}%)")
        else:
            print("  No significant streaks (5+) have been broken yet")

    def _print_env_reward_distribution(self) -> None:
        """Print env_reward distribution for landed episodes."""
        dist = self.tracker.get_env_reward_distribution()

        print(f"\n--- ENV REWARD DISTRIBUTION (LANDINGS) ---")

        if 'landed' not in dist:
            print("  No landing data available")
            return

        landed = dist['landed']
        print(f"  Total landings: {landed['count']}")
        print(f"  Mean env_reward: {landed['mean']:.1f} (std: {landed['std']:.1f})")
        print(f"  Range: {landed['min']:.1f} to {landed['max']:.1f}")

        print(f"\n  ENV REWARD BREAKDOWN:")
        print(f"    >= 200 (SUCCESS):    {landed['above_200']:4} ({landed['above_200_pct']:5.1f}%)")
        print(f"    180-199 (near miss): {landed['in_range_180_200']:4} ({landed['in_range_180_200'] / landed['count'] * 100:5.1f}%)")
        print(f"    150-179:             {landed['in_range_150_180']:4} ({landed['in_range_150_180'] / landed['count'] * 100:5.1f}%)")
        print(f"    < 150:               {landed['below_150']:4} ({landed['below_150'] / landed['count'] * 100:5.1f}%)")

        if 'crashed' in dist:
            crashed = dist['crashed']
            print(f"\n  Crashed episodes: {crashed['count']} (mean: {crashed['mean']:.1f})")

    def _print_recent_episodes(self, n: int = 5) -> None:
        """Print details of recent episodes from primitive lists."""
        print(f"\n--- LAST {n} EPISODES DETAIL ---")

        total = self.tracker.total_episodes
        if total == 0:
            print("No episodes recorded")
            return

        start_idx = max(0, total - n)
        for i in range(start_idx, total):
            env_reward = self.tracker.env_rewards[i]
            shaped_bonus = self.tracker.shaped_bonuses[i]
            total_reward = env_reward + shaped_bonus
            success = self.tracker.successes_bool[i]
            duration = self.tracker.durations[i]
            outcome = self.tracker.outcomes[i] if i < len(self.tracker.outcomes) else 'UNKNOWN'

            status = "SUCCESS" if success else "FAILURE"
            print(f"  Episode {i+1}: {outcome} - {status} - "
                  f"reward={total_reward:.1f} (env={env_reward:.1f}, shaped={shaped_bonus:+.1f}), "
                  f"duration={duration:.1f}s")

    def _print_key_data(self) -> None:
        """Print key data counts for analysis."""
        print(f"\nData points collected: {self.tracker.total_episodes} episodes, "
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
