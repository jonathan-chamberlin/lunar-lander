"""Parameter dataclasses for reducing function parameter counts.

These dataclasses group related parameters to simplify function signatures
and improve code readability.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np

from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from analysis.diagnostics import DiagnosticsTracker

if TYPE_CHECKING:
    from data.run_logger import RunLogger
    import pygame as pg


@dataclass
class EpisodeData:
    """Computed episode results passed to finalize_episode().

    Groups all the data collected during an episode that needs to be
    recorded and analyzed.

    Note: actions_array and observations_array are used only for behavior
    analysis and should be cleared immediately after use to prevent memory
    accumulation during long training runs.
    """
    episode_num: int
    total_reward: float
    env_reward: float
    shaped_bonus: float
    steps: int
    duration_seconds: float
    terminated: bool
    truncated: bool
    # Arrays for behavior analysis - made Optional to allow clearing after use
    actions_array: Optional[np.ndarray] = None
    observations_array: Optional[np.ndarray] = None
    rendered: bool = False

    def clear_arrays(self) -> None:
        """Clear trajectory arrays to free memory. Call after behavior analysis."""
        self.actions_array = None
        self.observations_array = None


@dataclass
class TimingState:
    """Timing and progress tracking across episodes.

    Groups the running totals used for speed calculations and
    batch metrics recording.
    """
    start_time: Optional[float] = None
    total_steps: int = 0
    total_training_updates: int = 0


@dataclass
class TrainingContext:
    """Context objects needed for training operations.

    Groups the service objects and configuration values used
    during episode finalization.
    """
    diagnostics: DiagnosticsTracker
    replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer]
    success_threshold: float
    min_experiences: int
    run_logger: Optional["RunLogger"] = None


@dataclass
class PyGameContext:
    """Pygame objects for rendering.

    Groups the pygame-related objects needed for rendered episodes.
    Type hints use Any to avoid requiring pygame import at module load time,
    enabling headless operation in sweep_runner.
    """
    font: Any  # pg.font.Font
    screen: Any  # pg.Surface
    clock: Any  # pg.time.Clock
