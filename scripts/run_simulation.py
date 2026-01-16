#!/usr/bin/env python
"""Minimal-output simulation wrapper.

This wrapper enforces a minimal-output execution environment by:
- Automatically enabling minimal, low-entropy output (print_mode='minimal')
- Preventing verbose per-episode diagnostics
- Preserving context for interpreting trends, diagnostics, and charts

Usage:
    python scripts/run_simulation.py [--episodes N] [--human]

Options:
    --episodes N    Override number of episodes (default: from config)
    --human         Use human mode (verbose output) instead of minimal mode
"""

import sys
import os

# Get the project root (lunar-lander folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')

# Add src to path so we can import from there
sys.path.insert(0, src_dir)

# Change to src directory so relative imports work
os.chdir(src_dir)

# Parse simple command line args before importing heavy modules
use_human_mode = '--human' in sys.argv
num_episodes = None
for i, arg in enumerate(sys.argv):
    if arg == '--episodes' and i + 1 < len(sys.argv):
        try:
            num_episodes = int(sys.argv[i + 1])
        except ValueError:
            print(f"Invalid episodes value: {sys.argv[i + 1]}")
            sys.exit(1)

# Now import config and patch it for minimal mode
from config import Config, RunConfig

# Create a modified RunConfig with minimal mode
original_run_config = RunConfig

class MinimalRunConfig(RunConfig):
    """RunConfig with minimal mode defaults."""

    def __new__(cls, *args, **kwargs):
        # Force minimal mode unless --human was specified
        if 'print_mode' not in kwargs:
            kwargs['print_mode'] = 'human' if use_human_mode else 'minimal'

        # Override episodes if specified
        if num_episodes is not None and 'num_episodes' not in kwargs:
            kwargs['num_episodes'] = num_episodes

        return original_run_config(*args, **kwargs)

# Monkey-patch the config module
import config
config.RunConfig = MinimalRunConfig

# Now run main
if __name__ == '__main__':
    from main import main

    mode = 'HUMAN' if use_human_mode else 'MINIMAL'
    print(f"=== Simulation Wrapper: {mode} mode ===")
    if num_episodes:
        print(f"=== Episodes override: {num_episodes} ===")
    print()

    main()
