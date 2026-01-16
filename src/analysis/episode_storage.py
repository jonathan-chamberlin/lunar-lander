"""Episode data storage using primitive types.

Stores per-episode data as lightweight primitives (floats, ints, strings, bools)
to prevent unbounded memory growth from object storage.
"""

from typing import List, Optional, Tuple

import numpy as np


class EpisodeStorage:
    """Stores primitive episode data for charting and analysis.

    Uses lightweight primitive lists instead of object storage to prevent
    unbounded memory growth. Memory efficient: ~78 bytes/episode vs ~368 bytes/episode.
    """

    def __init__(self) -> None:
        # Per-episode primitive lists (for charting)
        self.env_rewards: List[float] = []
        self.shaped_bonuses: List[float] = []
        self.durations: List[float] = []
        self.successes_bool: List[bool] = []
        self.outcomes: List[str] = []
        self.landed_env_rewards: List[float] = []  # Only for landed episodes

        # Running statistics (incremental, no recomputation)
        self.total_episodes: int = 0
        self.success_count: int = 0
        self.first_success_episode: Optional[int] = None

        # Running sums for mean calculations
        self._reward_sum: float = 0.0
        self._env_reward_sum: float = 0.0
        self._duration_sum: float = 0.0
        self._min_reward: float = float('inf')
        self._max_reward: float = float('-inf')
        self._min_env_reward: float = float('inf')
        self._max_env_reward: float = float('-inf')

        # Pre-compute landed outcomes set
        self._landed_outcomes_set = {
            'LANDED_PERFECTLY', 'LANDED_SOFTLY', 'LANDED_HARD',
            'LANDED_TILTED', 'TOUCHED_DOWN_ONE_LEG', 'LANDED_WITH_DRIFT'
        }

    def record_episode(
        self,
        episode_num: int,
        env_reward: float,
        shaped_bonus: float,
        duration_seconds: float,
        success: bool
    ) -> None:
        """Record episode metrics.

        Args:
            episode_num: Episode number
            env_reward: Raw environment reward
            shaped_bonus: Reward shaping bonus
            duration_seconds: Episode duration
            success: Whether episode was successful
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

        # Track success
        if success:
            self.success_count += 1
            if self.first_success_episode is None:
                self.first_success_episode = episode_num

    def record_outcome(self, outcome: str, env_reward: float) -> None:
        """Record episode outcome.

        Args:
            outcome: Episode outcome string (e.g., 'LANDED_PERFECTLY')
            env_reward: Environment reward (for landed histogram)
        """
        self.outcomes.append(outcome)

        # Store landed rewards for histogram
        if outcome in self._landed_outcomes_set:
            self.landed_env_rewards.append(env_reward)

    def get_rewards(self) -> List[float]:
        """Get list of total rewards for all episodes."""
        return [e + s for e, s in zip(self.env_rewards, self.shaped_bonuses)]

    def get_reward_data(self) -> Tuple[List[float], List[float]]:
        """Get reward data for charts.

        Returns:
            Tuple of (env_rewards, shaped_bonuses) lists
        """
        return self.env_rewards, self.shaped_bonuses

    def get_duration_data(self) -> List[float]:
        """Get episode durations for charts."""
        return self.durations

    def get_landing_histogram_data(self) -> List[float]:
        """Get env_rewards for landed episodes for histogram."""
        return self.landed_env_rewards

    def get_success_rate(self) -> float:
        """Get overall success rate."""
        if self.total_episodes == 0:
            return 0.0
        return self.success_count / self.total_episodes

    def get_mean_reward(self) -> float:
        """Get mean total reward."""
        if self.total_episodes == 0:
            return 0.0
        return self._reward_sum / self.total_episodes

    def get_mean_env_reward(self) -> float:
        """Get mean environment reward."""
        if self.total_episodes == 0:
            return 0.0
        return self._env_reward_sum / self.total_episodes

    def get_reward_range(self) -> Tuple[float, float]:
        """Get (min, max) total reward."""
        min_r = self._min_reward if self._min_reward != float('inf') else 0.0
        max_r = self._max_reward if self._max_reward != float('-inf') else 0.0
        return min_r, max_r

    def get_env_reward_range(self) -> Tuple[float, float]:
        """Get (min, max) environment reward."""
        min_r = self._min_env_reward if self._min_env_reward != float('inf') else 0.0
        max_r = self._max_env_reward if self._max_env_reward != float('-inf') else 0.0
        return min_r, max_r

    def get_final_n_mean_reward(self, n: int = 50) -> Optional[float]:
        """Get mean reward for last n episodes.

        Args:
            n: Number of recent episodes to average

        Returns:
            Mean reward or None if not enough episodes
        """
        if self.total_episodes < n:
            return None
        recent_rewards = [
            e + s for e, s in zip(self.env_rewards[-n:], self.shaped_bonuses[-n:])
        ]
        return float(np.mean(recent_rewards))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'env_rewards': self.env_rewards,
            'shaped_bonuses': self.shaped_bonuses,
            'durations': self.durations,
            'successes_bool': self.successes_bool,
            'outcomes': self.outcomes,
            'total_episodes': self.total_episodes,
            'success_count': self.success_count,
        }
