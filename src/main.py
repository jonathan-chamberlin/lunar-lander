"""Main entry point for Lunar Lander TD3 training.

This module provides the CLI interface for training. The core training logic
is in training/runner.py which is shared with sweep_runner.py.
"""

import io
import logging
import os
import sys
import warnings

# Suppress warnings BEFORE importing libraries that generate them
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress pygame "Hello from the pygame community" message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Force unbuffered stdout for real-time output display with UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from config import Config
from training.runner import run_training
from training.training_options import TrainingOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for CLI training."""
    config = Config()

    options = TrainingOptions(
        output_mode=config.run.print_mode,
        require_pygame=(config.run.render_mode != 'none'),
        show_final_charts=True,
        save_model=True,
        enable_logging=True,
    )

    result = run_training(config, options)

    if result.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
