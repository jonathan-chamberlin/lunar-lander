#!/usr/bin/env python3
"""
Sweep Configuration Generator

Interactive tool to create hyperparameter sweep configurations.

Usage:
    python generate_config.py
    python generate_config.py --output my_sweep.json
"""

import argparse
import json
from pathlib import Path


# Default hyperparameter specifications
HYPERPARAMETERS = {
    "actor_lr": {
        "description": "Actor learning rate",
        "default": 0.001,
        "suggested_values": [0.0001, 0.0005, 0.001, 0.002, 0.005],
        "range": {"min": 0.0001, "max": 0.01, "log": True}
    },
    "critic_lr": {
        "description": "Critic learning rate",
        "default": 0.002,
        "suggested_values": [0.0005, 0.001, 0.002, 0.004, 0.01],
        "range": {"min": 0.0001, "max": 0.01, "log": True}
    },
    "batch_size": {
        "description": "Training batch size",
        "default": 128,
        "suggested_values": [32, 64, 128, 256, 512],
        "range": {"min": 32, "max": 512, "log": False}
    },
    "buffer_size": {
        "description": "Replay buffer capacity",
        "default": 16384,
        "suggested_values": [8192, 16384, 32768, 65536],
        "range": {"min": 4096, "max": 131072, "log": True}
    },
    "gamma": {
        "description": "Discount factor",
        "default": 0.99,
        "suggested_values": [0.95, 0.97, 0.99, 0.995],
        "range": {"min": 0.9, "max": 0.999, "log": False}
    },
    "tau": {
        "description": "Target network soft update rate",
        "default": 0.005,
        "suggested_values": [0.001, 0.005, 0.01, 0.02],
        "range": {"min": 0.001, "max": 0.05, "log": True}
    },
    "policy_update_frequency": {
        "description": "TD3 delayed policy updates",
        "default": 3,
        "suggested_values": [1, 2, 3, 4, 5],
        "range": {"min": 1, "max": 10, "log": False}
    },
    "sigma": {
        "description": "OU noise sigma",
        "default": 0.3,
        "suggested_values": [0.1, 0.2, 0.3, 0.4, 0.5],
        "range": {"min": 0.05, "max": 0.5, "log": False}
    },
    "theta": {
        "description": "OU noise theta",
        "default": 0.2,
        "suggested_values": [0.1, 0.15, 0.2, 0.25, 0.3],
        "range": {"min": 0.05, "max": 0.5, "log": False}
    }
}


def create_quick_config(name: str, params: dict, sweep_type: str = "grid",
                        episodes: int = 500, n_samples: int = 10) -> dict:
    """Create a sweep configuration programmatically."""
    config = {
        "name": name,
        "type": sweep_type,
        "episodes_per_run": episodes,
        "parameters": params
    }

    if sweep_type == "random":
        config["n_samples"] = n_samples

    return config


def interactive_create():
    """Interactive configuration creator."""
    print("=" * 60)
    print("Hyperparameter Sweep Configuration Generator")
    print("=" * 60)

    # Get sweep name
    name = input("\nSweep name (e.g., 'lr_sweep'): ").strip() or "sweep"

    # Get sweep type
    print("\nSweep types:")
    print("  1. grid - Test all combinations")
    print("  2. random - Sample random configurations")
    sweep_type = input("Select type [1/2] (default: 1): ").strip()
    sweep_type = "random" if sweep_type == "2" else "grid"

    # Get episodes
    episodes = input("\nEpisodes per run (default: 500): ").strip()
    episodes = int(episodes) if episodes else 500

    # Select parameters
    print("\nAvailable hyperparameters:")
    param_list = list(HYPERPARAMETERS.keys())
    for i, param in enumerate(param_list, 1):
        info = HYPERPARAMETERS[param]
        print(f"  {i}. {param}: {info['description']} (default: {info['default']})")

    print("\nEnter parameter numbers to include (comma-separated, e.g., '1,2,3'):")
    selection = input("> ").strip()

    selected_indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip()]
    selected_params = [param_list[i] for i in selected_indices if 0 <= i < len(param_list)]

    if not selected_params:
        print("No parameters selected. Using actor_lr and critic_lr as defaults.")
        selected_params = ["actor_lr", "critic_lr"]

    # Configure each parameter
    parameters = {}
    for param in selected_params:
        info = HYPERPARAMETERS[param]
        print(f"\n{param}: {info['description']}")
        print(f"  Suggested values: {info['suggested_values']}")
        print(f"  Default: {info['default']}")

        if sweep_type == "grid":
            values = input(f"  Enter values (comma-separated) or press Enter for suggested: ").strip()
            if values:
                # Parse values
                parsed = []
                for v in values.split(","):
                    v = v.strip()
                    try:
                        parsed.append(float(v) if "." in v else int(v))
                    except ValueError:
                        pass
                parameters[param] = parsed if parsed else info["suggested_values"]
            else:
                parameters[param] = info["suggested_values"]
        else:
            # Random search - use range
            use_range = input(f"  Use range {info['range']} [y/N]: ").strip().lower()
            if use_range == "y":
                parameters[param] = info["range"]
            else:
                parameters[param] = info["suggested_values"]

    # Build config
    config = {
        "name": name,
        "type": sweep_type,
        "episodes_per_run": episodes,
        "parameters": parameters
    }

    if sweep_type == "random":
        n_samples = input("\nNumber of random samples (default: 10): ").strip()
        config["n_samples"] = int(n_samples) if n_samples else 10

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate sweep configuration")
    parser.add_argument("--output", "-o", default="sweep_config.json", help="Output file")
    parser.add_argument("--non-interactive", action="store_true", help="Use defaults")
    args = parser.parse_args()

    if args.non_interactive:
        # Create a sensible default config
        config = create_quick_config(
            name="default_sweep",
            params={
                "actor_lr": [0.0005, 0.001, 0.002],
                "critic_lr": [0.001, 0.002, 0.004]
            },
            sweep_type="grid",
            episodes=500
        )
    else:
        config = interactive_create()

    # Calculate total runs
    if config["type"] == "grid":
        import itertools
        total = 1
        for values in config["parameters"].values():
            if isinstance(values, list):
                total *= len(values)
        print(f"\nTotal configurations: {total}")
    else:
        print(f"\nRandom samples: {config.get('n_samples', 10)}")

    # Save config
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfiguration saved to: {output_path}")
    print(f"\nTo run: python sweep_runner.py {output_path}")


if __name__ == "__main__":
    main()
