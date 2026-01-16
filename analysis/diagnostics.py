"""Diagnostics tracking and reporting for TD3 training.

Uses incremental statistics to avoid O(n) recomputation and unbounded memory growth.
All per-episode data is stored as primitive types (floats/ints/strings) rather than objects.
"""

import json
import logging
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np

from data_types import (
    AggregatedTrainingMetrics
)
from constants import (
    OUTCOME_CATEGORIES, QUALITY_BEHAVIORS, OUTCOME_TO_CATEGORY,
    OUTCOME_CATEGORY_ORDER, REPORT_CARD_BEHAVIORS
)

logger = logging.getLogger(__name__)


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
    """Collects training metrics using incremental statistics.

    Uses lightweight primitive lists instead of object storage to prevent
    unbounded memory growth. Statistics are computed incrementally as
    episodes complete, avoiding O(n) recomputation.

    Memory efficient: ~78 bytes/episode vs ~368 bytes/episode with objects.
    """

    def __init__(self, batch_size: int = 50) -> None:
        self.batch_size = batch_size

        # =====================================================================
        # Per-episode primitive lists (for charting)
        # =====================================================================
        self.env_rewards: List[float] = []
        self.shaped_bonuses: List[float] = []
        self.durations: List[float] = []
        self.successes_bool: List[bool] = []
        self.outcomes: List[str] = []
        self.streak_history: List[int] = []
        self.landed_env_rewards: List[float] = []  # Only for landed episodes

        # =====================================================================
        # Running statistics (incremental, no recomputation)
        # =====================================================================
        self.total_episodes: int = 0
        self.success_count: int = 0
        self.current_streak: int = 0
        self.max_streak: int = 0
        self.max_streak_episode: int = 0
        self.first_success_episode: Optional[int] = None

        # Global outcome and behavior counters
        self.outcome_counts: Counter = Counter()
        self.behavior_counts: Counter = Counter()

        # Running sums for mean calculations
        self._reward_sum: float = 0.0
        self._env_reward_sum: float = 0.0
        self._duration_sum: float = 0.0
        self._min_reward: float = float('inf')
        self._max_reward: float = float('-inf')
        self._min_env_reward: float = float('inf')
        self._max_env_reward: float = float('-inf')

        # Quality behavior tracking
        self._good_behavior_count: int = 0
        self._bad_behavior_count: int = 0
        self._contact_count: int = 0
        self._low_altitude_count: int = 0
        self._clean_touchdown_count: int = 0

        # =====================================================================
        # Per-batch counters (reset every batch_size episodes)
        # =====================================================================
        self._current_batch_episode_count: int = 0
        self._current_batch_success_count: int = 0
        self._current_batch_outcome_counts: Counter = Counter()
        self._current_batch_behavior_counts: Counter = Counter()

        # =====================================================================
        # Completed batch statistics (primitive lists)
        # =====================================================================
        self.batch_success_rates: List[float] = []
        self.batch_outcome_distributions: List[Dict[str, float]] = []
        self.batch_behavior_frequencies: List[Dict[str, float]] = []

        # =====================================================================
        # Report card: first 100 + rolling last 100
        # =====================================================================
        self._first_100_behavior_counts: Counter = Counter()
        self._first_100_outcome_counts: Counter = Counter()
        self._first_100_frozen: bool = False
        self._last_100_behaviors: deque = deque(maxlen=100)  # Each entry is Set[str]
        self._last_100_successes: deque = deque(maxlen=100)  # Each entry is bool

        # =====================================================================
        # Success correlation tracking
        # =====================================================================
        self._success_behavior_counts: Counter = Counter()
        self._failure_behavior_counts: Counter = Counter()

        # =====================================================================
        # Training metrics (recorded periodically, not every episode)
        # =====================================================================
        self.q_values: List[float] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.actor_grad_norms: List[float] = []
        self.critic_grad_norms: List[float] = []

        # =====================================================================
        # Speed metrics per batch
        # =====================================================================
        self.batch_speed_metrics: List[BatchSpeedMetrics] = []

        # Pre-compute behavior sets for fast lookup
        self._good_behaviors_set: Set[str] = set(QUALITY_BEHAVIORS['good'])
        self._bad_behaviors_set: Set[str] = set(QUALITY_BEHAVIORS['bad'])
        self._contact_behaviors_set: Set[str] = {
            'TOUCHED_DOWN_CLEAN', 'SCRAPED_LEFT_LEG', 'SCRAPED_RIGHT_LEG',
            'BOUNCED', 'PROLONGED_ONE_LEG', 'MULTIPLE_TOUCHDOWNS'
        }
        self._landed_outcomes_set: Set[str] = {
            'LANDED_PERFECTLY', 'LANDED_SOFTLY', 'LANDED_HARD',
            'LANDED_TILTED', 'LANDED_ONE_LEG', 'LANDED_SLIDING'
        }

    def record_episode(
        self,
        episode_num: int,
        env_reward: float,
        shaped_bonus: float,
        duration_seconds: float,
        success: bool
    ) -> None:
        """Record episode metrics incrementally.

        Args:
            episode_num: Episode number
            env_reward: Raw environment reward
            shaped_bonus: Reward shaping bonus
            duration_seconds: Episode duration
            success: Whether episode was successful (env_reward >= 200)
        """
        total_reward = env_reward + shaped_bonus

        # Append primitives for charts
        self.env_rewards.append(env_reward)
        self.shaped_bonuses.append(shaped_bonus)
        self.durations.append(duration_seconds)
        self.successes_bool.append(success)

        # Update running statistics
        self.total_episodes += 1
        self._reward_sum += total_reward
        self._env_reward_sum += env_reward
        self._duration_sum += duration_seconds
        self._min_reward = min(self._min_reward, total_reward)
        self._max_reward = max(self._max_reward, total_reward)
        self._min_env_reward = min(self._min_env_reward, env_reward)
        self._max_env_reward = max(self._max_env_reward, env_reward)

        # Update success tracking
        if success:
            self.success_count += 1
            self.current_streak += 1
            if self.current_streak > self.max_streak:
                self.max_streak = self.current_streak
                self.max_streak_episode = episode_num
            if self.first_success_episode is None:
                self.first_success_episode = episode_num
        else:
            self.current_streak = 0

        self.streak_history.append(self.current_streak)

        # Update batch success counter
        self._current_batch_episode_count += 1
        if success:
            self._current_batch_success_count += 1

        # Update rolling last 100 successes
        self._last_100_successes.append(success)

    def record_behavior(
        self,
        outcome: str,
        behaviors: List[str],
        env_reward: float,
        success: bool
    ) -> None:
        """Record behavior data incrementally.

        Args:
            outcome: Episode outcome string (e.g., 'LANDED_PERFECTLY')
            behaviors: List of behavior strings detected in the episode
            env_reward: Environment reward (for landed histogram)
            success: Whether episode was successful
        """
        self.outcomes.append(outcome)

        # Update outcome counters
        self.outcome_counts[outcome] += 1
        self._current_batch_outcome_counts[outcome] += 1

        # Update behavior counters and check quality
        behavior_set = set(behaviors)
        has_good = False
        has_bad = False
        has_contact = False
        has_low_alt = 'REACHED_LOW_ALTITUDE' in behavior_set

        for behavior in behaviors:
            self.behavior_counts[behavior] += 1
            self._current_batch_behavior_counts[behavior] += 1

            if behavior in self._good_behaviors_set:
                has_good = True
            if behavior in self._bad_behaviors_set:
                has_bad = True
            if behavior in self._contact_behaviors_set:
                has_contact = True
            if behavior == 'TOUCHED_DOWN_CLEAN':
                self._clean_touchdown_count += 1

        if has_good:
            self._good_behavior_count += 1
        if has_bad:
            self._bad_behavior_count += 1
        if has_contact:
            self._contact_count += 1
        if has_low_alt:
            self._low_altitude_count += 1

        # First 100 episodes tracking
        if not self._first_100_frozen:
            for behavior in behavior_set:
                self._first_100_behavior_counts[behavior] += 1
            self._first_100_outcome_counts[outcome] += 1
            if self.total_episodes >= 100:
                self._first_100_frozen = True

        # Rolling last 100 behaviors (for report card)
        self._last_100_behaviors.append(behavior_set)

        # Success correlation
        if success:
            for behavior in behavior_set:
                self._success_behavior_counts[behavior] += 1
        else:
            for behavior in behavior_set:
                self._failure_behavior_counts[behavior] += 1

        # Store landed rewards for histogram
        if outcome in self._landed_outcomes_set:
            self.landed_env_rewards.append(env_reward)

        # Check if batch complete
        if self._current_batch_episode_count >= self.batch_size:
            self._finalize_batch()

    def _finalize_batch(self) -> None:
        """Finalize current batch statistics and reset counters."""
        batch_len = self._current_batch_episode_count
        if batch_len == 0:
            return

        # Compute and store batch success rate
        success_rate = (self._current_batch_success_count / batch_len) * 100
        self.batch_success_rates.append(success_rate)

        # Compute and store outcome distribution
        outcome_dist = {}
        for category in OUTCOME_CATEGORY_ORDER:
            category_outcomes = OUTCOME_CATEGORIES.get(category, [])
            count = sum(self._current_batch_outcome_counts.get(o, 0) for o in category_outcomes)
            # Also check OUTCOME_TO_CATEGORY for outcomes not in OUTCOME_CATEGORIES
            for outcome, cat in OUTCOME_TO_CATEGORY.items():
                if cat == category and outcome not in category_outcomes:
                    count += self._current_batch_outcome_counts.get(outcome, 0)
            outcome_dist[category] = (count / batch_len) * 100
        self.batch_outcome_distributions.append(outcome_dist)

        # Compute and store behavior frequencies for heatmap
        behavior_freqs = {}
        for behavior in REPORT_CARD_BEHAVIORS:
            count = self._current_batch_behavior_counts.get(behavior, 0)
            behavior_freqs[behavior] = (count / batch_len) * 100
        self.batch_behavior_frequencies.append(behavior_freqs)

        # Reset batch counters
        self._current_batch_episode_count = 0
        self._current_batch_success_count = 0
        self._current_batch_outcome_counts.clear()
        self._current_batch_behavior_counts.clear()

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

    # =========================================================================
    # Getter Methods for Charts (return primitive data)
    # =========================================================================

    def get_rewards(self) -> List[float]:
        """Get list of total rewards for all episodes."""
        return [e + s for e, s in zip(self.env_rewards, self.shaped_bonuses)]

    def get_reward_data(self) -> Tuple[List[float], List[float]]:
        """Get reward data for charts.

        Returns:
            Tuple of (env_rewards, shaped_bonuses) lists
        """
        return self.env_rewards, self.shaped_bonuses

    def get_success_data(self) -> Tuple[List[float], List[bool]]:
        """Get success rate data for charts.

        Returns:
            Tuple of (batch_success_rates, successes_bool) lists
        """
        return self.batch_success_rates, self.successes_bool

    def get_outcome_distribution_per_batch(self) -> List[Dict[str, float]]:
        """Get pre-computed outcome distributions per batch.

        Returns:
            List of dicts mapping category -> percentage for each batch
        """
        return self.batch_outcome_distributions

    def get_behavior_frequencies_per_batch(self) -> List[Dict[str, float]]:
        """Get pre-computed behavior frequencies per batch for heatmap.

        Returns:
            List of dicts mapping behavior -> frequency% for each batch
        """
        return self.batch_behavior_frequencies

    def get_streak_data(self) -> Tuple[List[int], int, int]:
        """Get streak data for charts.

        Returns:
            Tuple of (streak_history, max_streak, max_streak_episode)
        """
        return self.streak_history, self.max_streak, self.max_streak_episode

    def get_duration_data(self) -> List[float]:
        """Get episode durations for charts.

        Returns:
            List of episode durations in seconds
        """
        return self.durations

    def get_report_card_data(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get first 100 vs last 100 behavior frequencies for report card.

        Returns:
            Tuple of (first_100_freqs, last_100_freqs) dicts
        """
        # First 100 frequencies
        first_100_count = min(100, self.total_episodes)
        first_100_freqs = {}
        for behavior in REPORT_CARD_BEHAVIORS:
            count = self._first_100_behavior_counts.get(behavior, 0)
            first_100_freqs[behavior] = (count / first_100_count * 100) if first_100_count > 0 else 0.0

        # Last 100 frequencies from rolling buffer
        last_100_count = len(self._last_100_behaviors)
        last_100_freqs = {}
        if last_100_count > 0:
            behavior_counts: Counter = Counter()
            for behavior_set in self._last_100_behaviors:
                behavior_counts.update(behavior_set)
            for behavior in REPORT_CARD_BEHAVIORS:
                last_100_freqs[behavior] = (behavior_counts.get(behavior, 0) / last_100_count) * 100
        else:
            for behavior in REPORT_CARD_BEHAVIORS:
                last_100_freqs[behavior] = 0.0

        return first_100_freqs, last_100_freqs

    def get_landing_histogram_data(self) -> List[float]:
        """Get env_rewards for landed episodes for histogram.

        Returns:
            List of env_rewards for episodes that landed
        """
        return self.landed_env_rewards

    def get_summary(self) -> DiagnosticsSummary:
        """Compute summary statistics from incremental metrics.

        Returns:
            DiagnosticsSummary with computed statistics
        """
        num_episodes = self.total_episodes

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

        # Episode statistics from running sums
        mean_reward = self._reward_sum / num_episodes
        final_50_mean = None
        if num_episodes >= 50:
            # Compute from primitive lists (last 50 only)
            recent_rewards = [e + s for e, s in zip(self.env_rewards[-50:], self.shaped_bonuses[-50:])]
            final_50_mean = float(np.mean(recent_rewards))

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
            num_successes=self.success_count,
            success_rate=self.success_count / num_episodes,
            mean_reward=mean_reward,
            max_reward=self._max_reward if self._max_reward != float('-inf') else 0.0,
            min_reward=self._min_reward if self._min_reward != float('inf') else 0.0,
            final_50_mean_reward=final_50_mean,
            mean_main_thruster=None,  # Action stats removed for memory efficiency
            mean_side_thruster=None,
            mean_action_magnitude=None,
            high_thruster_episode_ratio=None,
            mean_q_value=mean_q,
            q_value_trend=q_trend,
            mean_actor_loss=mean_actor_loss,
            mean_critic_loss=mean_critic_loss,
            mean_actor_grad_norm=mean_actor_grad,
            mean_critic_grad_norm=mean_critic_grad
        )

    def get_behavior_statistics(self, batch_size: int = 50) -> Optional[BehaviorStatistics]:
        """Get behavior statistics from incremental counters.

        Args:
            batch_size: Size of batches (ignored, uses instance batch_size)

        Returns:
            BehaviorStatistics from incremental data, None if no data
        """
        if self.total_episodes == 0:
            return None

        num_episodes = self.total_episodes

        # Outcome category counts from OUTCOME_TO_CATEGORY
        outcome_category_counts: Dict[str, int] = {cat: 0 for cat in OUTCOME_CATEGORY_ORDER}
        for outcome, count in self.outcome_counts.items():
            category = OUTCOME_TO_CATEGORY.get(outcome, 'crashed')
            outcome_category_counts[category] += count

        # Top behaviors from incremental counter
        top_behaviors = [
            (behavior, count, count / num_episodes * 100)
            for behavior, count in self.behavior_counts.most_common(15)
        ]

        # Crash type distribution
        crash_outcomes = OUTCOME_CATEGORIES.get('crashed', [])
        total_crashes = sum(self.outcome_counts.get(o, 0) for o in crash_outcomes)
        # Also include outcomes mapped to 'crashed' via OUTCOME_TO_CATEGORY
        for outcome, cat in OUTCOME_TO_CATEGORY.items():
            if cat == 'crashed' and outcome not in crash_outcomes:
                total_crashes += self.outcome_counts.get(outcome, 0)

        crash_type_distribution = {}
        if total_crashes > 0:
            for crash_type in crash_outcomes:
                crash_type_distribution[crash_type] = self.outcome_counts.get(crash_type, 0) / total_crashes * 100

        # Success correlation rates
        key_behaviors = ['STAYED_UPRIGHT', 'STAYED_CENTERED', 'CONTROLLED_DESCENT',
                        'REACHED_LOW_ALTITUDE', 'TOUCHED_DOWN_CLEAN', 'CONTROLLED_THROUGHOUT',
                        'NEVER_STABILIZED', 'RAPID_DESCENT', 'HOVER_MAINTAINED',
                        'ERRATIC_THRUST', 'SMOOTH_THRUST', 'NO_CONTACT_MADE']

        success_behavior_rates = {}
        failure_behavior_rates = {}
        failure_count = num_episodes - self.success_count
        for behavior in key_behaviors:
            success_behavior_rates[behavior] = (
                self._success_behavior_counts.get(behavior, 0) / self.success_count * 100
                if self.success_count > 0 else 0.0
            )
            failure_behavior_rates[behavior] = (
                self._failure_behavior_counts.get(behavior, 0) / failure_count * 100
                if failure_count > 0 else 0.0
            )

        return BehaviorStatistics(
            total_episodes=num_episodes,
            outcome_counts=dict(self.outcome_counts),
            outcome_category_counts=outcome_category_counts,
            behavior_counts=dict(self.behavior_counts),
            top_behaviors=top_behaviors,
            good_behavior_rate=self._good_behavior_count / num_episodes * 100,
            bad_behavior_rate=self._bad_behavior_count / num_episodes * 100,
            low_altitude_rate=self._low_altitude_count / num_episodes * 100,
            contact_rate=self._contact_count / num_episodes * 100,
            clean_touchdown_rate=self._clean_touchdown_count / num_episodes * 100,
            stayed_upright_rate=self.behavior_counts.get('STAYED_UPRIGHT', 0) / num_episodes * 100,
            stayed_centered_rate=self.behavior_counts.get('STAYED_CENTERED', 0) / num_episodes * 100,
            controlled_descent_rate=self.behavior_counts.get('CONTROLLED_DESCENT', 0) / num_episodes * 100,
            controlled_throughout_rate=self.behavior_counts.get('CONTROLLED_THROUGHOUT', 0) / num_episodes * 100,
            never_stabilized_rate=self.behavior_counts.get('NEVER_STABILIZED', 0) / num_episodes * 100,
            batch_success_rates=self.batch_success_rates,
            batch_outcome_distributions=self.batch_outcome_distributions,
            batch_low_altitude_rates=[],  # Not tracked incrementally
            batch_contact_rates=[],  # Not tracked incrementally
            crash_type_distribution=crash_type_distribution,
            success_behavior_rates=success_behavior_rates,
            failure_behavior_rates=failure_behavior_rates,
        )

    def get_streak_statistics(self) -> Dict[str, Any]:
        """Get streak statistics from incremental tracking.

        Returns:
            Dictionary with streak data (no streak_breaks - would require object storage)
        """
        return {
            'max_streak': self.max_streak,
            'max_streak_episode': self.max_streak_episode,
            'current_streak': self.current_streak,
            'streaks': self.streak_history,
            'streak_breaks': []  # Not tracked to avoid object storage
        }

    def get_env_reward_distribution(self) -> Dict[str, Any]:
        """Get env_reward distribution from incremental data.

        Returns:
            Dictionary with env_reward statistics for landed episodes.
        """
        if not self.landed_env_rewards:
            return {}

        landed_rewards = self.landed_env_rewards
        landed_arr = np.array(landed_rewards)

        stats = {
            'landed': {
                'count': len(landed_rewards),
                'mean': float(np.mean(landed_arr)),
                'std': float(np.std(landed_arr)),
                'min': float(np.min(landed_arr)),
                'max': float(np.max(landed_arr)),
                'above_200': sum(1 for r in landed_rewards if r >= 200),
                'above_200_pct': sum(1 for r in landed_rewards if r >= 200) / len(landed_rewards) * 100,
                'in_range_180_200': sum(1 for r in landed_rewards if 180 <= r < 200),
                'in_range_150_180': sum(1 for r in landed_rewards if 150 <= r < 180),
                'below_150': sum(1 for r in landed_rewards if r < 150),
                'rewards': landed_rewards  # For histogram
            }
        }

        return stats

    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get advanced statistics from incremental data.

        Returns:
            Dictionary with env_reward stats, rolling rates, etc.
        """
        n = self.total_episodes
        if n == 0:
            return {}

        stats = {}

        # 1. Env reward statistics from running stats
        if self.env_rewards:
            env_arr = np.array(self.env_rewards)
            stats['env_reward'] = {
                'mean': self._env_reward_sum / n,
                'std': float(np.std(env_arr)),
                'max': self._max_env_reward if self._max_env_reward != float('-inf') else 0.0,
                'min': self._min_env_reward if self._min_env_reward != float('inf') else 0.0,
            }

        # 2. Recent vs overall comparison (last 100 vs all)
        if n >= 100:
            recent_env_rewards = self.env_rewards[-100:]
            recent_successes = sum(self._last_100_successes)
            stats['recent_100'] = {
                'env_reward_mean': float(np.mean(recent_env_rewards)),
                'success_count': recent_successes,
                'success_rate': recent_successes / min(100, len(self._last_100_successes)) * 100,
            }

        # 3. Rolling success rates (last 50 and last 100)
        if n >= 50:
            last_50 = list(self._last_100_successes)[-50:]
            stats['rolling_success_rate_50'] = sum(last_50) / len(last_50) * 100 if last_50 else 0.0
        if n >= 100:
            stats['rolling_success_rate_100'] = sum(self._last_100_successes) / len(self._last_100_successes) * 100

        # 4. Near-miss count (env_reward 180-199) - compute from primitive list
        near_miss_count = sum(1 for r in self.env_rewards if 180 <= r < 200)
        stats['near_misses'] = {
            'count': near_miss_count,
            'episodes': [],  # Episode numbers not tracked to save memory
        }

        # 5. Time to first success
        stats['first_success_episode'] = self.first_success_episode

        # 6. Best recent streak (from streak_history if long enough)
        if n >= 50:
            lookback = min(200, n)
            recent_streaks = self.streak_history[-lookback:]
            stats['max_streak_last_200'] = max(recent_streaks) if recent_streaks else 0

        # 7. Landing quality for successful episodes
        if self.success_count > 0:
            # Get env_rewards for successful episodes
            success_rewards = [r for r, s in zip(self.env_rewards, self.successes_bool) if s]
            if success_rewards:
                stats['successful_landings'] = {
                    'count': len(success_rewards),
                    'env_reward_mean': float(np.mean(success_rewards)),
                    'env_reward_std': float(np.std(success_rewards)),
                    'env_reward_min': float(np.min(success_rewards)),
                    'env_reward_max': float(np.max(success_rewards)),
                }

            # Success quality from incremental counters
            num_successes = self.success_count
            stats['success_quality'] = {
                'stayed_upright': self._success_behavior_counts.get('STAYED_UPRIGHT', 0) / num_successes * 100,
                'stayed_centered': self._success_behavior_counts.get('STAYED_CENTERED', 0) / num_successes * 100,
                'controlled_descent': self._success_behavior_counts.get('CONTROLLED_DESCENT', 0) / num_successes * 100,
                'clean_touchdown': self._success_behavior_counts.get('TOUCHED_DOWN_CLEAN', 0) / num_successes * 100,
                'landed_perfectly': self._success_behavior_counts.get('LANDED_PERFECTLY', 0) / num_successes * 100,
                'landed_softly': self._success_behavior_counts.get('LANDED_SOFTLY', 0) / num_successes * 100,
            }

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked data to a dictionary for serialization."""
        return {
            'env_rewards': self.env_rewards,
            'shaped_bonuses': self.shaped_bonuses,
            'durations': self.durations,
            'successes_bool': self.successes_bool,
            'outcomes': self.outcomes,
            'streak_history': self.streak_history,
            'total_episodes': self.total_episodes,
            'success_count': self.success_count,
            'max_streak': self.max_streak,
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
