#!/usr/bin/env python3
"""
Hyperparameter Sweep Runner

Execute hyperparameter sweeps from a JSON configuration file.

Usage:
    python sweep_runner.py sweep_config.json
    python sweep_runner.py sweep_config.json --dry-run
"""

import argparse
import json
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import csv


def load_config(config_path: str) -> dict:
    """Load sweep configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def generate_grid_combinations(parameters: dict) -> list[dict]:
    """Generate all combinations for grid search."""
    keys = list(parameters.keys())
    values = list(parameters.values())
    combinations = []

    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def generate_random_combinations(parameters: dict, n_samples: int) -> list[dict]:
    """Generate random combinations for random search."""
    import random
    combinations = []

    for _ in range(n_samples):
        combo = {}
        for key, value_spec in parameters.items():
            if isinstance(value_spec, list):
                combo[key] = random.choice(value_spec)
            elif isinstance(value_spec, dict):
                # Range specification: {"min": 0.001, "max": 0.01, "log": true}
                min_val = value_spec["min"]
                max_val = value_spec["max"]
                if value_spec.get("log", False):
                    import math
                    combo[key] = math.exp(random.uniform(math.log(min_val), math.log(max_val)))
                else:
                    combo[key] = random.uniform(min_val, max_val)
        combinations.append(combo)

    return combinations


def run_training(params: dict, episodes: int, run_id: int, output_dir: Path) -> dict:
    """Run a single training with given parameters."""
    run_dir = output_dir / f"run_{run_id:03d}"
    run_dir.mkdir(exist_ok=True)

    # Build command with parameter overrides
    cmd = [
        sys.executable, "-m", "lunar-lander.main",
        "--episodes", str(episodes),
        "--output-dir", str(run_dir),
    ]

    for key, value in params.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(params, f, indent=2)

    # Run training
    print(f"\n{'='*60}")
    print(f"Run {run_id}: {params}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        # Save output
        with open(run_dir / "stdout.txt", "w") as f:
            f.write(result.stdout)
        with open(run_dir / "stderr.txt", "w") as f:
            f.write(result.stderr)

        # Parse results (extract final metrics from output)
        metrics = parse_training_output(result.stdout)
        metrics["params"] = params
        metrics["run_id"] = run_id
        metrics["success"] = result.returncode == 0

        return metrics

    except subprocess.TimeoutExpired:
        return {"params": params, "run_id": run_id, "success": False, "error": "timeout"}
    except Exception as e:
        return {"params": params, "run_id": run_id, "success": False, "error": str(e)}


def parse_training_output(output: str) -> dict:
    """Parse training output to extract metrics."""
    import re

    metrics = {
        "success_rate": 0.0,
        "mean_reward": 0.0,
        "max_reward": -float("inf"),
        "episodes_to_first_success": -1,
    }

    successes = 0
    total = 0
    rewards = []
    first_success = -1

    for line in output.split("\n"):
        # Match episode result lines
        match = re.search(r"Run (\d+).*Reward: ([\d.-]+)", line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            rewards.append(reward)
            total += 1

            if reward >= 200:
                successes += 1
                if first_success == -1:
                    first_success = episode

    if total > 0:
        metrics["success_rate"] = successes / total
        metrics["mean_reward"] = sum(rewards) / len(rewards)
        metrics["max_reward"] = max(rewards) if rewards else 0
        metrics["episodes_to_first_success"] = first_success

    return metrics


def save_summary(results: list[dict], output_dir: Path):
    """Save sweep summary to CSV."""
    if not results:
        return

    # Flatten results for CSV
    rows = []
    for r in results:
        row = {"run_id": r["run_id"], "success": r.get("success", False)}
        row.update(r.get("params", {}))
        row["success_rate"] = r.get("success_rate", 0)
        row["mean_reward"] = r.get("mean_reward", 0)
        row["max_reward"] = r.get("max_reward", 0)
        row["episodes_to_first_success"] = r.get("episodes_to_first_success", -1)
        rows.append(row)

    # Sort by success rate
    rows.sort(key=lambda x: x["success_rate"], reverse=True)

    # Write CSV
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary saved to {csv_path}")

    # Print best config
    if rows:
        best = rows[0]
        print(f"\nBest configuration (success rate: {best['success_rate']:.1%}):")
        for k, v in best.items():
            if k not in ["run_id", "success", "success_rate", "mean_reward", "max_reward", "episodes_to_first_success"]:
                print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("config", help="Path to sweep config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Show combinations without running")
    parser.add_argument("--output-dir", default="sweep_results", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)

    # Generate combinations
    sweep_type = config.get("type", "grid")
    if sweep_type == "grid":
        combinations = generate_grid_combinations(config["parameters"])
    else:
        n_samples = config.get("n_samples", 10)
        combinations = generate_random_combinations(config["parameters"], n_samples)

    print(f"Sweep: {config.get('name', 'unnamed')}")
    print(f"Type: {sweep_type}")
    print(f"Total configurations: {len(combinations)}")

    if args.dry_run:
        print("\nConfigurations:")
        for i, combo in enumerate(combinations):
            print(f"  {i+1}: {combo}")
        return

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{config.get('name', 'sweep')}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep config
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run sweep
    episodes = config.get("episodes_per_run", 500)
    results = []

    for i, combo in enumerate(combinations):
        result = run_training(combo, episodes, i + 1, output_dir)
        results.append(result)

        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Save summary
    save_summary(results, output_dir)


if __name__ == "__main__":
    main()
