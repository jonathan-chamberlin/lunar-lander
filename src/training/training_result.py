"""TrainingResult dataclass for unified training output."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.diagnostics import DiagnosticsTracker


@dataclass
class TrainingResult:
    """Result of a training run.

    Contains all metrics needed by both main.py CLI and sweep_runner.py.
    """

    success_rate: float  # Percentage (0-100)
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    first_success_episode: Optional[int]
    final_100_success_rate: Optional[float]  # Percentage (0-100), None if < 100 episodes
    total_episodes: int
    elapsed_time: float
    user_quit: bool
    error: Optional[str]
    diagnostics: "DiagnosticsTracker"

    def to_sweep_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by sweep_runner.

        Returns:
            Dict with all sweep result fields.
        """
        return {
            'total_episodes': self.total_episodes,
            'success_rate': self.success_rate,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'max_reward': self.max_reward,
            'min_reward': self.min_reward,
            'first_success_episode': self.first_success_episode,
            'final_100_success_rate': self.final_100_success_rate,
            'elapsed_time': self.elapsed_time,
            'user_quit': self.user_quit,
            'error': self.error,
        }
