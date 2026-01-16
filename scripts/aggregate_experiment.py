#!/usr/bin/env python3
"""Aggregate multiple simulations into an experiment for comparison.

Usage:
    python aggregate_experiment.py --name "lr_sweep" --sims sim1 sim2 sim3
    python aggregate_experiment.py --name "lr_sweep" --pattern "*"
    python aggregate_experiment.py --name "lr_sweep" --pattern "2024-01-*"

The script:
1. Creates or updates an experiment with the given name
2. Adds specified simulations to the experiment
3. Loads final.json from each simulation
4. Computes cross-simulation statistics
5. Writes experiment summary to experiments/{name}/summary.json
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.abstractions import SimulationIdentifier, ExperimentDefinition
from data.experiment_io import ExperimentDirectory


def find_simulations(
    simulations_path: Path,
    patterns: list[str],
) -> list[str]:
    """Find simulation directories matching patterns.

    Args:
        simulations_path: Path to simulations/ directory
        patterns: List of patterns (simulation IDs or glob patterns)

    Returns:
        List of matching simulation IDs
    """
    if not simulations_path.exists():
        print(f"Error: Simulations directory not found: {simulations_path}")
        return []

    all_sims = [
        d.name for d in simulations_path.iterdir()
        if d.is_dir() and (d / "aggregates" / "final.json").exists()
    ]

    matched = set()
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            # Glob pattern
            for sim in all_sims:
                if fnmatch.fnmatch(sim, pattern):
                    matched.add(sim)
        else:
            # Exact match
            if pattern in all_sims:
                matched.add(pattern)
            else:
                print(f"Warning: Simulation not found: {pattern}")

    return sorted(matched)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate simulations into an experiment"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name (used as directory name)",
    )
    parser.add_argument(
        "--sims",
        nargs="+",
        help="Simulation IDs to include",
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        help="Glob patterns to match simulation IDs (e.g., '*', '2024-01-*')",
    )
    parser.add_argument(
        "--description",
        help="Experiment description",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Tags for the experiment",
    )
    parser.add_argument(
        "--base-path",
        default=str(PROJECT_ROOT),
        help="Base path for simulations and experiments",
    )
    parser.add_argument(
        "--list-simulations",
        action="store_true",
        help="List available simulations and exit",
    )

    args = parser.parse_args()

    base_path = Path(args.base_path)
    simulations_path = base_path / "simulations"

    # List mode
    if args.list_simulations:
        if not simulations_path.exists():
            print("No simulations directory found")
            return 1

        print("Available simulations:")
        for sim_dir in sorted(simulations_path.iterdir()):
            if sim_dir.is_dir():
                final_path = sim_dir / "aggregates" / "final.json"
                status = "ready" if final_path.exists() else "incomplete"
                print(f"  {sim_dir.name} ({status})")
        return 0

    # Validate inputs
    if not args.sims and not args.pattern:
        parser.error("Either --sims or --pattern is required")

    # Find simulations
    patterns = []
    if args.sims:
        patterns.extend(args.sims)
    if args.pattern:
        patterns.extend(args.pattern)

    simulation_ids = find_simulations(simulations_path, patterns)

    if not simulation_ids:
        print("Error: No matching simulations found")
        return 1

    print(f"Found {len(simulation_ids)} simulations:")
    for sim_id in simulation_ids:
        print(f"  - {sim_id}")

    # Create/update experiment
    exp_dir = ExperimentDirectory(str(base_path), args.name)

    if exp_dir.exists():
        print(f"\nUpdating existing experiment: {args.name}")
        exp_dir.load()
    else:
        print(f"\nCreating new experiment: {args.name}")
        exp_dir.initialize(
            description=args.description,
            tags=args.tags,
        )

    # Add simulations
    for sim_id_str in simulation_ids:
        sim_id = SimulationIdentifier.from_string(sim_id_str)
        exp_dir.add_simulation(sim_id)

    # Compute summary
    print("\nComputing experiment summary...")
    summary = exp_dir.compute_summary(simulations_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Experiment: {args.name}")
    print("=" * 60)

    meta = summary.get("_metadata", {})
    print(f"Simulations: {meta.get('simulation_count', 0)}")

    agg = summary.get("aggregate_metrics", {})
    if agg:
        sr = agg.get("success_rate", {})
        mr = agg.get("mean_reward", {})
        print(f"\nSuccess Rate:")
        print(f"  Mean: {sr.get('mean', 0):.2%}")
        print(f"  Std:  {sr.get('std', 0):.2%}")
        print(f"  Range: {sr.get('min', 0):.2%} - {sr.get('max', 0):.2%}")

        print(f"\nMean Reward:")
        print(f"  Mean: {mr.get('mean', 0):.2f}")
        print(f"  Std:  {mr.get('std', 0):.2f}")
        print(f"  Range: {mr.get('min', 0):.2f} - {mr.get('max', 0):.2f}")

    best = summary.get("best_simulation", {})
    if best:
        print(f"\nBest Simulation:")
        if "by_success_rate" in best:
            print(f"  By Success Rate: {best['by_success_rate']} ({best.get('success_rate', 0):.2%})")
        if "by_mean_reward" in best:
            print(f"  By Mean Reward: {best.get('by_mean_reward')} ({best.get('mean_reward', 0):.2f})")

    print(f"\nSummary written to: {exp_dir.summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
