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

    # Run identification
    run_name: Optional[str] = None

    # What to enable
    enable_logging: bool = True  # JSONL run logging
    save_model: bool = True  # Save final model checkpoint
    show_final_charts: bool = False  # Display charts on screen at end

    # Pygame control
    require_pygame: bool = True  # False = skip pygame init for headless
