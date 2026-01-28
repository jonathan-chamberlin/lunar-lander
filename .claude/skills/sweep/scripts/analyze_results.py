#!/usr/bin/env python3
"""
Sweep Results Analyzer

Analyze and summarize hyperparameter sweep results.

Usage:
    python analyze_results.py sweep_results/my_sweep_20240115/
    python analyze_results.py sweep_results/my_sweep_20240115/ --top 5
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Optional


def load_results(results_dir: Path) -> tuple[dict, list[dict]]:
    """Load sweep config and results from directory."""
    config_path = results_dir / "sweep_config.json"
    results_path = results_dir / "results.json"
    summary_path = results_dir / "summary.csv"

    config = {}
    results = []

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    elif summary_path.exists():
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            results = list(reader)

    return config, results


def analyze_parameter_impact(results: list[dict], param: str) -> dict:
    """Analyze how a parameter affects performance."""
    value_metrics = {}

    for r in results:
        params = r.get("params", r)
        if param not in params:
            continue

        value = params[param]
        if value not in value_metrics:
            value_metrics[value] = {"success_rates": [], "mean_rewards": []}

        success_rate = float(r.get("success_rate", 0))
        mean_reward = float(r.get("mean_reward", 0))

        value_metrics[value]["success_rates"].append(success_rate)
        value_metrics[value]["mean_rewards"].append(mean_reward)

    # Compute averages
    analysis = {}
    for value, metrics in value_metrics.items():
        analysis[value] = {
            "avg_success_rate": sum(metrics["success_rates"]) / len(metrics["success_rates"]),
            "avg_mean_reward": sum(metrics["mean_rewards"]) / len(metrics["mean_rewards"]),
            "n_runs": len(metrics["success_rates"])
        }

    return analysis


def format_analysis(config: dict, results: list[dict], top_n: int = 5):
    """Format analysis output."""
    print("=" * 70)
    print(f"SWEEP ANALYSIS: {config.get('name', 'Unknown')}")
    print("=" * 70)

    print(f"\nSweep type: {config.get('type', 'unknown')}")
    print(f"Episodes per run: {config.get('episodes_per_run', 'unknown')}")
    print(f"Total runs: {len(results)}")

    if not results:
        print("\nNo results found.")
        return

    # Sort by success rate
    sorted_results = sorted(
        results,
        key=lambda x: float(x.get("success_rate", 0)),
        reverse=True
    )

    # Top configurations
    print(f"\n{'='*70}")
    print(f"TOP {min(top_n, len(sorted_results))} CONFIGURATIONS")
    print(f"{'='*70}")

    for i, r in enumerate(sorted_results[:top_n], 1):
        params = r.get("params", {k: v for k, v in r.items()
                                   if k not in ["run_id", "success", "success_rate",
                                               "mean_reward", "max_reward",
                                               "episodes_to_first_success"]})
        print(f"\n#{i} (Run {r.get('run_id', '?')})")
        print(f"  Success rate: {float(r.get('success_rate', 0)):.1%}")
        print(f"  Mean reward: {float(r.get('mean_reward', 0)):.1f}")
        print(f"  Max reward: {float(r.get('max_reward', 0)):.1f}")
        print(f"  First success: Episode {r.get('episodes_to_first_success', 'N/A')}")
        print(f"  Parameters:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    # Parameter impact analysis
    parameters = config.get("parameters", {})
    if parameters:
        print(f"\n{'='*70}")
        print("PARAMETER IMPACT ANALYSIS")
        print(f"{'='*70}")

        for param in parameters:
            analysis = analyze_parameter_impact(results, param)
            if not analysis:
                continue

            print(f"\n{param}:")
            sorted_values = sorted(analysis.items(),
                                   key=lambda x: x[1]["avg_success_rate"],
                                   reverse=True)
            for value, metrics in sorted_values:
                print(f"  {value}: {metrics['avg_success_rate']:.1%} success "
                      f"(avg reward: {metrics['avg_mean_reward']:.1f}, "
                      f"n={metrics['n_runs']})")

    # Overall statistics
    success_rates = [float(r.get("success_rate", 0)) for r in results]
    mean_rewards = [float(r.get("mean_reward", 0)) for r in results]

    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"  Success rate range: {min(success_rates):.1%} - {max(success_rates):.1%}")
    print(f"  Mean reward range: {min(mean_rewards):.1f} - {max(mean_rewards):.1f}")
    print(f"  Avg success rate: {sum(success_rates)/len(success_rates):.1%}")
    print(f"  Avg mean reward: {sum(mean_rewards)/len(mean_rewards):.1f}")

    # Recommendations
    best = sorted_results[0] if sorted_results else None
    if best:
        print(f"\n{'='*70}")
        print("RECOMMENDATION")
        print(f"{'='*70}")
        print(f"\nBest configuration achieved {float(best.get('success_rate', 0)):.1%} success rate.")
        print("\nRecommended settings:")
        params = best.get("params", {})
        for k, v in params.items():
            print(f"  {k} = {v}")


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("results_dir", help="Path to sweep results directory")
    parser.add_argument("--top", type=int, default=5, help="Show top N configurations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return 1

    config, results = load_results(results_dir)

    if args.json:
        output = {
            "config": config,
            "results": results,
            "best": sorted(results, key=lambda x: float(x.get("success_rate", 0)),
                          reverse=True)[:args.top]
        }
        print(json.dumps(output, indent=2))
    else:
        format_analysis(config, results, args.top)

    return 0


if __name__ == "__main__":
    exit(main())
