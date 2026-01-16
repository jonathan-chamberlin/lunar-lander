"""Streak tracking and report card data.

Tracks success streaks and maintains first 100 / last 100 episode data
for report card comparison charts.
"""

from collections import Counter, deque
from typing import Dict, List, Set, Tuple

from constants import REPORT_CARD_BEHAVIORS


class StreakTracker:
    """Tracks success streaks and report card data.

    Maintains:
    - Current and max consecutive success streaks
    - First 100 episodes behavior/outcome counts (frozen after 100)
    - Rolling last 100 episodes for comparison
    """

    def __init__(self) -> None:
        # Streak tracking
        self.current_streak: int = 0
        self.max_streak: int = 0
        self.max_streak_episode: int = 0
        self.streak_history: List[int] = []

        # Report card: first 100 episodes (frozen after reaching 100)
        self._first_100_behavior_counts: Counter = Counter()
        self._first_100_outcome_counts: Counter = Counter()
        self._first_100_frozen: bool = False
        self._first_100_episode_count: int = 0

        # Rolling last 100 episodes
        self._last_100_behaviors: deque = deque(maxlen=100)  # Each entry is Set[str]
        self._last_100_successes: deque = deque(maxlen=100)  # Each entry is bool

    def record_episode(self, episode_num: int, success: bool) -> None:
        """Record episode for streak tracking.

        Args:
            episode_num: Episode number
            success: Whether episode was successful
        """
        # Update streak
        if success:
            self.current_streak += 1
            if self.current_streak > self.max_streak:
                self.max_streak = self.current_streak
                self.max_streak_episode = episode_num
        else:
            self.current_streak = 0

        self.streak_history.append(self.current_streak)

        # Update rolling last 100 successes
        self._last_100_successes.append(success)

    def record_behavior(
        self,
        outcome: str,
        behaviors: List[str],
        total_episodes: int
    ) -> None:
        """Record behavior for report card tracking.

        Args:
            outcome: Episode outcome string
            behaviors: List of behavior strings
            total_episodes: Current total episode count
        """
        behavior_set = set(behaviors)

        # First 100 episodes tracking
        if not self._first_100_frozen:
            self._first_100_episode_count += 1
            for behavior in behavior_set:
                self._first_100_behavior_counts[behavior] += 1
            self._first_100_outcome_counts[outcome] += 1
            if self._first_100_episode_count >= 100:
                self._first_100_frozen = True

        # Rolling last 100 behaviors (for report card)
        self._last_100_behaviors.append(behavior_set)

    def get_streak_data(self) -> Tuple[List[int], int, int]:
        """Get streak data for charts.

        Returns:
            Tuple of (streak_history, max_streak, max_streak_episode)
        """
        return self.streak_history, self.max_streak, self.max_streak_episode

    def get_streak_statistics(self) -> Dict:
        """Get streak statistics.

        Returns:
            Dictionary with streak data
        """
        return {
            'max_streak': self.max_streak,
            'max_streak_episode': self.max_streak_episode,
            'current_streak': self.current_streak,
            'streaks': self.streak_history,
            'streak_breaks': []  # Not tracked to avoid object storage
        }

    def get_report_card_data(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get first 100 vs last 100 behavior frequencies for report card.

        Returns:
            Tuple of (first_100_freqs, last_100_freqs) dicts
        """
        # First 100 frequencies
        first_100_count = min(100, self._first_100_episode_count)
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

    def get_rolling_success_rate(self, n: int = 100) -> float:
        """Get rolling success rate for last n episodes.

        Args:
            n: Number of recent episodes (default 100)

        Returns:
            Success rate as percentage
        """
        if not self._last_100_successes:
            return 0.0

        # Get last n successes from deque
        successes = list(self._last_100_successes)
        if n < len(successes):
            successes = successes[-n:]

        return sum(successes) / len(successes) * 100 if successes else 0.0

    def get_recent_successes_count(self) -> int:
        """Get count of successes in last 100 episodes."""
        return sum(self._last_100_successes)
