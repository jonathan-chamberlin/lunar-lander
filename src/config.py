"""Configuration dataclasses for the Lunar Lander TD3 training."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
@dataclass(frozen=True)
class RunConfig:
    """Configuration for training runs."""

    num_episodes: int = 5000
    num_envs: int = 8
    random_warmup_episodes: int = 5  # Reduced from 15 (good init bias helps)
    framerate: Optional[int] = None  # None = unlimited speed, or set FPS limit
    timing: bool = True
    training_enabled: bool = True  # Set False to skip all network training (for testing)

    # Print mode options:
    #   'human'      - Verbose, narrative, detailed output with emojis
    #   'minimal'    - Minimal, low-entropy, structured output for LLM context efficiency
    #   'silent'     - No console output
    #   'background' - Only batch completion messages (e.g., "Batch 1 (Runs 1-100) completed")
    print_mode: str = 'background'

    # Rendering options:
    #   'none' - No episodes rendered (fastest training)
    #   'all'  - All episodes rendered (slowest, but visual feedback)
    #   'custom' - Only render episodes specified in render_episodes_custom
    render_mode: str = 'all'
    render_episodes_custom: Tuple[int, ...] = field(default_factory=tuple)

    # Internal field set by __post_init__ based on render_mode
    render_episodes: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Validate print_mode
        valid_print_modes = ('human', 'minimal', 'silent', 'background')
        if self.print_mode not in valid_print_modes:
            raise ValueError(f"Invalid print_mode: {self.print_mode}. Use one of {valid_print_modes}")

        # Set render_episodes based on render_mode
        if self.render_mode == 'none':
            episodes = tuple()
        elif self.render_mode == 'all':
            episodes = tuple(range(self.num_episodes))
        elif self.render_mode == 'custom':
            episodes = self.render_episodes_custom
        else:
            raise ValueError(f"Invalid render_mode: {self.render_mode}. Use 'none', 'all', or 'custom'")

        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, 'render_episodes', episodes)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for the Gymnasium environment."""

    env_name: str = "LunarLanderContinuous-v3"
    state_dim: int = 8
    action_dim: int = 2
    success_threshold: float = 200.0


@dataclass(frozen=True)
class DisplayConfig:
    """Configuration for pygame display overlays."""

    # Set to True to show run number overlay on screen during rendering.
    # Set to False to disable overlay and improve rendering performance (~50% faster).
    show_run_overlay: bool = True

    font_size: int = 30
    font_color: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    text_x: int = 5
    text_y: int = 5

@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for TD3 training."""

    actor_lr: float = 0.001  # Increased from 0.0005 for faster learning
    critic_lr: float = 0.002  # Increased from 0.001 (2x actor)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128  # Reduced from 256 for faster iterations
    buffer_size: int = 1 << 14  # 16384 (reduced from 65536 for cache locality)
    min_experiences_before_training: int = 2000  # Reduced from 5000
    training_updates_per_episode: int = 25  # Reduced from 50
    gradient_clip_value: float = 10.0

    # TD3-specific parameters
    policy_update_frequency: int = 3
    target_policy_noise: float = 0.1
    target_noise_clip: float = 0.3

    # Prioritized Experience Replay parameters
    use_per: bool = True  # Enable PER for faster learning
    per_alpha: float = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
    per_beta_start: float = 0.4  # Initial importance sampling weight
    per_beta_end: float = 1.0  # Final importance sampling weight (annealed)
    per_epsilon: float = 1e-6  # Small constant to prevent zero priorities


@dataclass(frozen=True)
class NoiseConfig:
    """Ornstein-Uhlenbeck noise parameters."""

    mu: Tuple[float, float] = (0.0, 0.0)
    sigma: float = 0.3
    theta: float = 0.2
    dt: float = 0.01
    x0: float = 0.0
    action_dimensions: int = 2

    # Noise decay parameters
    noise_scale_initial: float = 1.0
    noise_scale_final: float = 0.2
    noise_decay_episodes: int = 300


@dataclass
class Config:
    """Master configuration combining all config sections."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    run: RunConfig = field(default_factory=RunConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
