#!/usr/bin/env python3
"""Run a single training session with JSON config.

This script is designed to be called from sweep_runner.py in subprocess mode.
It accepts a JSON config file path and runs training with those parameters.
Memory is completely freed when this process exits.

Usage:
    python run_single_training.py <config.json>

The config.json should contain:
{
    "params": {"batch_size": 128, ...},
    "run_name": "run_001",
    "results_dir": "/path/to/results",
    "episodes": 500
}
"""

import io
import json
import os
import sys
import warnings

# Suppress warnings BEFORE importing libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Force unbuffered stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Add src directory to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataclasses import replace
from config import Config, TrainingConfig, NoiseConfig, RunConfig
from training.runner import run_training
from training.training_options import TrainingOptions


def apply_params_to_config(config: Config, params: dict) -> Config:
    """Apply parameter overrides to a config object."""
    training_params = {}
    noise_params = {}
    run_params = {}

    training_fields = set(TrainingConfig.__dataclass_fields__.keys())
    noise_fields = set(NoiseConfig.__dataclass_fields__.keys())
    run_fields = set(RunConfig.__dataclass_fields__.keys())

    for name, value in params.items():
        if name.startswith('_'):  # Skip internal params
            continue
        if name in training_fields:
            training_params[name] = value
        elif name in noise_fields:
            noise_params[name] = value
        elif name in run_fields:
            run_params[name] = value

    new_training = replace(config.training, **training_params) if training_params else config.training
    new_noise = replace(config.noise, **noise_params) if noise_params else config.noise
    new_run = replace(config.run, **run_params) if run_params else config.run

    return Config(
        training=new_training,
        noise=new_noise,
        run=new_run,
        environment=config.environment,
        display=config.display
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_single_training.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        run_config = json.load(f)

    params = run_config.get('params', {})
    run_name = run_config['run_name']
    results_dir = Path(run_config['results_dir'])
    episodes = run_config.get('episodes', 500)

    # Create base config
    base_config = Config()
    base_config = Config(
        training=base_config.training,
        noise=base_config.noise,
        run=replace(base_config.run,
                    num_episodes=episodes,
                    render_mode='none'),
        environment=base_config.environment,
        display=base_config.display
    )

    # Apply parameter overrides
    config = apply_params_to_config(base_config, params)

    # Set up training options
    options = TrainingOptions(
        output_mode='background',
        results_dir=results_dir,
        charts_dir=None,
        run_name=run_name,
        require_pygame=False,
        enable_logging=False,
        save_model=True,
        show_final_charts=False,
        is_experiment=True,
    )

    # Run training
    result = run_training(config, options)

    # Save results
    results = result.to_sweep_dict()
    results['run_name'] = run_name

    results_file = results_dir / f"{run_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")

    if result.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
