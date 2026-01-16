"""Parameter dataclasses for reducing function parameter counts.

These dataclasses group related parameters to simplify function signatures
and improve code readability.
"""

from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import pygame as pg

from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from analysis.diagnostics import DiagnosticsTracker

if TYPE_CHECKING:
    from data.run_logger import RunLogger


@dataclass
class EpisodeData:
    """Computed episode results passed to finalize_episode().

    Groups all the data collected during an episode that needs to be
    recorded and analyzed.
    """
    episode_num: int
    total_reward: float
    env_reward: float
    shaped_bonus: float
    steps: int
    duration_seconds: float
    actions_array: np.ndarray
    observations_array: np.ndarray
    terminated: bool
    truncated: bool
    rendered: bool = False


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
    """
    font: pg.font.Font
    screen: pg.Surface
    clock: pg.time.Clock
