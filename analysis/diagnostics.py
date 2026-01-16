"""Diagnostics tracking and reporting for TD3 training.

Uses incremental statistics to avoid O(n) recomputation and unbounded memory growth.
All per-episode data is stored as primitive types (floats/ints/strings) rather than objects.
"""

import json
import logging
from collections import Counter, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np

from data_types import (
    AggregatedTrainingMetrics
)
from constants import (
    QUALITY_BEHAVIORS, OUTCOME_TO_CATEGORY,
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

        # Compute and store outcome distribution using OUTCOME_TO_CATEGORY as single source of truth
        outcome_dist = {}
        for category in OUTCOME_CATEGORY_ORDER:
            count = sum(
                self._current_batch_outcome_counts.get(outcome, 0)
                for outcome, cat in OUTCOME_TO_CATEGORY.items()
                if cat == category
            )
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

        # Crash type distribution using OUTCOME_TO_CATEGORY as single source of truth
        crash_outcomes = [outcome for outcome, cat in OUTCOME_TO_CATEGORY.items() if cat == 'crashed']
        total_crashes = sum(self.outcome_counts.get(o, 0) for o in crash_outcomes)

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


# Re-export DiagnosticsReporter for backwards compatibility
from analysis.reporter import DiagnosticsReporter
