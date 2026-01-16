"""Batch aggregation for per-batch statistics.

Computes per-batch statistics incrementally as episodes complete,
avoiding O(n) recomputation.
"""

from collections import Counter
from typing import Dict, List, Set

from constants import (
    OUTCOME_TO_CATEGORY,
    OUTCOME_CATEGORY_ORDER, REPORT_CARD_BEHAVIORS, QUALITY_BEHAVIORS
)


class BatchAggregator:
    """Computes per-batch statistics incrementally.

    Tracks per-batch counters and finalizes them when each batch completes,
    avoiding O(n) recomputation over all episodes.
    """

    def __init__(self, batch_size: int = 50) -> None:
        self.batch_size = batch_size

        # Per-batch counters (reset every batch_size episodes)
        self._current_batch_episode_count: int = 0
        self._current_batch_success_count: int = 0
        self._current_batch_outcome_counts: Counter = Counter()
        self._current_batch_behavior_counts: Counter = Counter()

        # Completed batch statistics (primitive lists)
        self.batch_success_rates: List[float] = []
        self.batch_outcome_distributions: List[Dict[str, float]] = []
        self.batch_behavior_frequencies: List[Dict[str, float]] = []

        # Global outcome and behavior counters
        self.outcome_counts: Counter = Counter()
        self.behavior_counts: Counter = Counter()

        # Quality behavior tracking
        self._good_behavior_count: int = 0
        self._bad_behavior_count: int = 0
        self._contact_count: int = 0
        self._low_altitude_count: int = 0
        self._clean_touchdown_count: int = 0

        # Success correlation tracking
        self._success_behavior_counts: Counter = Counter()
        self._failure_behavior_counts: Counter = Counter()

        # Pre-compute behavior sets for fast lookup
        self._good_behaviors_set: Set[str] = set(QUALITY_BEHAVIORS['good'])
        self._bad_behaviors_set: Set[str] = set(QUALITY_BEHAVIORS['bad'])
        self._contact_behaviors_set: Set[str] = {
            'TOUCHED_DOWN_CLEAN', 'SCRAPED_LEFT_LEG', 'SCRAPED_RIGHT_LEG',
            'BOUNCED', 'PROLONGED_ONE_LEG', 'MULTIPLE_TOUCHDOWNS'
        }

    def record_episode(self, success: bool) -> None:
        """Record episode for batch tracking.

        Args:
            success: Whether episode was successful
        """
        self._current_batch_episode_count += 1
        if success:
            self._current_batch_success_count += 1

    def record_behavior(
        self,
        outcome: str,
        behaviors: List[str],
        success: bool
    ) -> bool:
        """Record behavior data incrementally.

        Args:
            outcome: Episode outcome string
            behaviors: List of behavior strings
            success: Whether episode was successful

        Returns:
            True if a batch was finalized, False otherwise
        """
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

        # Success correlation
        if success:
            for behavior in behavior_set:
                self._success_behavior_counts[behavior] += 1
        else:
            for behavior in behavior_set:
                self._failure_behavior_counts[behavior] += 1

        # Check if batch complete
        if self._current_batch_episode_count >= self.batch_size:
            self._finalize_batch()
            return True
        return False

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

    def get_success_data(self) -> List[float]:
        """Get batch success rates for charts."""
        return self.batch_success_rates

    def get_outcome_distribution_per_batch(self) -> List[Dict[str, float]]:
        """Get pre-computed outcome distributions per batch."""
        return self.batch_outcome_distributions

    def get_behavior_frequencies_per_batch(self) -> List[Dict[str, float]]:
        """Get pre-computed behavior frequencies per batch for heatmap."""
        return self.batch_behavior_frequencies

    def get_quality_rates(self, total_episodes: int) -> Dict[str, float]:
        """Get quality behavior rates.

        Args:
            total_episodes: Total number of episodes for percentage calculation

        Returns:
            Dictionary with good/bad/contact/etc rates
        """
        if total_episodes == 0:
            return {
                'good_behavior_rate': 0.0,
                'bad_behavior_rate': 0.0,
                'contact_rate': 0.0,
                'low_altitude_rate': 0.0,
                'clean_touchdown_rate': 0.0,
            }
        return {
            'good_behavior_rate': self._good_behavior_count / total_episodes * 100,
            'bad_behavior_rate': self._bad_behavior_count / total_episodes * 100,
            'contact_rate': self._contact_count / total_episodes * 100,
            'low_altitude_rate': self._low_altitude_count / total_episodes * 100,
            'clean_touchdown_rate': self._clean_touchdown_count / total_episodes * 100,
        }

    def get_success_correlation(
        self,
        success_count: int,
        failure_count: int
    ) -> Dict[str, Dict[str, float]]:
        """Get behavior correlation with success/failure.

        Args:
            success_count: Number of successful episodes
            failure_count: Number of failed episodes

        Returns:
            Dict with 'success_rates' and 'failure_rates' sub-dicts
        """
        key_behaviors = [
            'STAYED_UPRIGHT', 'STAYED_CENTERED', 'CONTROLLED_DESCENT',
            'REACHED_LOW_ALTITUDE', 'TOUCHED_DOWN_CLEAN', 'CONTROLLED_THROUGHOUT',
            'NEVER_STABILIZED', 'RAPID_DESCENT', 'HOVER_MAINTAINED',
            'ERRATIC_THRUST', 'SMOOTH_THRUST', 'NO_CONTACT_MADE'
        ]

        success_rates = {}
        failure_rates = {}
        for behavior in key_behaviors:
            success_rates[behavior] = (
                self._success_behavior_counts.get(behavior, 0) / success_count * 100
                if success_count > 0 else 0.0
            )
            failure_rates[behavior] = (
                self._failure_behavior_counts.get(behavior, 0) / failure_count * 100
                if failure_count > 0 else 0.0
            )

        return {
            'success_rates': success_rates,
            'failure_rates': failure_rates,
        }
