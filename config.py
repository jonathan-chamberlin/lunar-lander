"""Configuration dataclasses for the Lunar Lander TD3 training."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for TD3 training."""

    actor_lr: float = 0.0001
    critic_lr: float = 0.0003
    gamma: float = 0.99
    tau: float = 0.001
    batch_size: int = 256
    buffer_size: int = 1 << 16  # 65536
    min_experiences_before_training: int = 500
    training_updates_per_episode: int = 50
    gradient_clip_value: float = 1.0

    # TD3-specific parameters
    policy_update_frequency: int = 2
    target_policy_noise: float = 0.1
    target_noise_clip: float = 0.3


@dataclass(frozen=True)
class NoiseConfig:
    """Ornstein-Uhlenbeck noise parameters."""

    mu: Tuple[float, float] = (0.0, 0.0)
    sigma: float = 0.1
    theta: float = 0.2
    dt: float = 0.01
    x0: float = 0.0
    action_dimensions: int = 2

    # Noise decay parameters
    noise_scale_initial: float = 1.0
    noise_scale_final: float = 0.2
    noise_decay_episodes: int = 300


@dataclass(frozen=True)
class RunConfig:
    """Configuration for training runs."""

    num_episodes: int = 400
    num_envs: int = 8
    random_warmup_episodes: int = 5
    framerate: int = 600
    timing: bool = True
    render_episodes: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Handle mutable default - render all episodes if not specified
        if not self.render_episodes:
            # Use object.__setattr__ because frozen=True
            object.__setattr__(
                self,
                'render_episodes',
                tuple(range(self.num_episodes))
            )

    @classmethod
    def with_render_episodes(
        cls,
        render_episodes: Tuple[int, ...],
        **kwargs
    ) -> 'RunConfig':
        """Factory method to create config with specific render episodes."""
        return cls(render_episodes=render_episodes, **kwargs)

    @classmethod
    def render_none(cls, **kwargs) -> 'RunConfig':
        """Factory method to create config that renders no episodes."""
        return cls(render_episodes=(-1,), **kwargs)  # -1 will never match


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for the Gymnasium environment."""

    env_name: str = "LunarLanderContinuous-v3"
    state_dim: int = 8
    action_dim: int = 2
    success_threshold: float = 200.0


@dataclass
class Config:
    """Master configuration combining all config sections."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    run: RunConfig = field(default_factory=RunConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration."""
        return cls()

    @classmethod
    def fast_training(cls) -> 'Config':
        """Create configuration optimized for fast training (no rendering)."""
        return cls(
            run=RunConfig.render_none(num_envs=16),
            training=TrainingConfig(min_experiences_before_training=256)
        )
