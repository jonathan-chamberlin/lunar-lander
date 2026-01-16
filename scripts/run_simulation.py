#!/usr/bin/env python
"""Agent-safe simulation wrapper.

This wrapper enforces an agent-safe execution environment by:
- Automatically enabling minimal, low-entropy output (print_mode='agent')
- Preventing verbose per-episode diagnostics
- Preserving agent context for interpreting trends, diagnostics, and charts

Usage:
    python scripts/run_simulation.py [--episodes N] [--human]

Options:
    --episodes N    Override number of episodes (default: from config)
    --human         Use human mode (verbose output) instead of agent mode
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

# Now import config and patch it for agent mode
from config import Config, RunConfig

# Create a modified RunConfig with agent mode
original_run_config = RunConfig

class AgentRunConfig(RunConfig):
    """RunConfig with agent mode defaults."""

    def __new__(cls, *args, **kwargs):
        # Force agent mode unless --human was specified
        if 'print_mode' not in kwargs:
            kwargs['print_mode'] = 'human' if use_human_mode else 'agent'

        # Override episodes if specified
        if num_episodes is not None and 'num_episodes' not in kwargs:
            kwargs['num_episodes'] = num_episodes

        return original_run_config(*args, **kwargs)

# Monkey-patch the config module
import config
config.RunConfig = AgentRunConfig

# Now run main
if __name__ == '__main__':
    from main import main

    mode = 'HUMAN' if use_human_mode else 'AGENT'
    print(f"=== Simulation Wrapper: {mode} mode ===")
    if num_episodes:
        print(f"=== Episodes override: {num_episodes} ===")
    print()

    main()
