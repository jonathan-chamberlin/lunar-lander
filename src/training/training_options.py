"""TrainingOptions dataclass for configuring training execution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingOptions:
    """Options controlling training execution behavior.

    These options are separate from the training hyperparameters (Config).
    They control how training is executed (output, saving, rendering, etc.)
    rather than what the training parameters are.
    """

    # Output mode: 'human', 'minimal', 'silent', 'background'
    output_mode: str = 'background'

    # Directories for saving artifacts (None = use default simulation directory)
    results_dir: Optional[Path] = None
    charts_dir: Optional[Path] = None
    frames_dir: Optional[Path] = None  # Directory for video frame capture
    video_dir: Optional[Path] = None   # Directory for compiled videos

    # Run identification
    run_name: Optional[str] = None

    # What to enable
    enable_logging: bool = True  # JSONL run logging
    save_model: bool = True  # Save final model checkpoint
    show_final_charts: bool = False  # Display charts on screen at end

    # Pygame control
    require_pygame: bool = True  # False = skip pygame init for headless

    # Experiment mode (affects chart generation)
    is_experiment: bool = False  # True = only generate final chart, skip periodic

    # Model loading for multi-phase training
    load_model_path: Optional[Path] = None  # Path to pre-trained model to continue from

    # Chart reporting batch size (episodes per batch in charts/diagnostics)
    diagnostics_batch_size: int = 50  # Default: 50 episodes per batch

    # Memory limit in MB (prevents system-wide memory exhaustion)
    # Default 4000 MB (4GB) leaves room for browser and other apps
    # Set to None to use auto-detected limit based on system memory
    memory_limit_mb: Optional[int] = 4000
