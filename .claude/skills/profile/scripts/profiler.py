#!/usr/bin/env python3
"""
Python Performance Profiler

Profile Python code using cProfile and generate reports.

Usage:
    python profiler.py --target "python script.py"
    python profiler.py --target "python -m module" --top 30
    python profiler.py --target "python script.py" --output profile.txt
"""

import argparse
import cProfile
import pstats
import subprocess
import sys
import io
import tempfile
from pathlib import Path
from datetime import datetime


def run_with_cprofile(command: str, output_file: str = None) -> pstats.Stats:
    """Run a command with cProfile and return stats."""
    # Create temp file for profile data
    with tempfile.NamedTemporaryFile(suffix='.prof', delete=False) as f:
        prof_file = f.name

    # Build profiling command
    profile_cmd = [
        sys.executable, '-m', 'cProfile',
        '-o', prof_file,
        *command.split()[1:]  # Skip 'python' and use rest
    ]

    print(f"Running: {' '.join(profile_cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(
            profile_cmd,
            capture_output=False,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"Warning: Command exited with code {result.returncode}")

        # Load stats
        stats = pstats.Stats(prof_file)
        return stats

    finally:
        # Cleanup
        Path(prof_file).unlink(missing_ok=True)


def profile_inline(code: str) -> pstats.Stats:
    """Profile inline Python code."""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        exec(code)
    finally:
        profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    return stats


def format_stats(stats: pstats.Stats, top_n: int = 20, sort_by: str = 'cumulative') -> str:
    """Format profile stats as readable text."""
    stream = io.StringIO()
    stats_copy = pstats.Stats(stats.stats, stream=stream)

    # Sort and print
    stats_copy.sort_stats(sort_by)
    stats_copy.print_stats(top_n)

    return stream.getvalue()


def analyze_bottlenecks(stats: pstats.Stats, threshold_pct: float = 5.0) -> list[dict]:
    """Analyze stats to identify bottlenecks."""
    bottlenecks = []

    # Get total time
    total_time = 0
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        total_time = max(total_time, ct)

    # Find significant functions
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        pct = (ct / total_time * 100) if total_time > 0 else 0

        if pct >= threshold_pct:
            filename, line, name = func
            bottlenecks.append({
                'function': name,
                'file': filename,
                'line': line,
                'cumulative_time': ct,
                'total_time': tt,
                'calls': nc,
                'percentage': pct
            })

    # Sort by percentage
    bottlenecks.sort(key=lambda x: x['percentage'], reverse=True)
    return bottlenecks


def suggest_optimizations(bottlenecks: list[dict]) -> list[str]:
    """Generate optimization suggestions based on bottlenecks."""
    suggestions = []

    for b in bottlenecks:
        func = b['function']
        pct = b['percentage']
        calls = b['calls']
        time_per_call = b['cumulative_time'] / calls if calls > 0 else 0

        # Generic suggestions based on patterns
        if 'sample' in func.lower():
            suggestions.append(
                f"- {func} ({pct:.1f}%): Consider batch sampling or more efficient data structures"
            )
        elif 'forward' in func.lower() or 'backward' in func.lower():
            suggestions.append(
                f"- {func} ({pct:.1f}%): Check batch sizes, consider mixed precision training"
            )
        elif 'append' in func.lower() or 'extend' in func.lower():
            suggestions.append(
                f"- {func} ({pct:.1f}%): Pre-allocate lists or use numpy arrays"
            )
        elif calls > 10000 and time_per_call > 0.0001:
            suggestions.append(
                f"- {func} ({pct:.1f}%): Called {calls} times - consider caching or batching"
            )
        elif pct > 20:
            suggestions.append(
                f"- {func} ({pct:.1f}%): Major bottleneck - prioritize optimization"
            )
        else:
            suggestions.append(
                f"- {func} ({pct:.1f}%): Review for optimization opportunities"
            )

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Profile Python code")
    parser.add_argument("--target", "-t", required=True, help="Command to profile")
    parser.add_argument("--top", type=int, default=20, help="Show top N functions")
    parser.add_argument("--sort", default="cumulative",
                       choices=['cumulative', 'time', 'calls'],
                       help="Sort by metric")
    parser.add_argument("--output", "-o", help="Save report to file")
    parser.add_argument("--threshold", type=float, default=2.0,
                       help="Bottleneck threshold percentage")
    args = parser.parse_args()

    print("=" * 60)
    print("PYTHON PROFILER")
    print("=" * 60)
    print(f"Target: {args.target}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Run profiler
    stats = run_with_cprofile(args.target)

    # Format output
    output = []
    output.append("\n" + "=" * 60)
    output.append(f"TOP {args.top} FUNCTIONS BY {args.sort.upper()} TIME")
    output.append("=" * 60)
    output.append(format_stats(stats, args.top, args.sort))

    # Analyze bottlenecks
    bottlenecks = analyze_bottlenecks(stats, args.threshold)

    if bottlenecks:
        output.append("\n" + "=" * 60)
        output.append("BOTTLENECK ANALYSIS")
        output.append("=" * 60)

        for i, b in enumerate(bottlenecks[:10], 1):
            output.append(f"\n{i}. {b['function']}")
            output.append(f"   File: {b['file']}:{b['line']}")
            output.append(f"   Time: {b['cumulative_time']:.3f}s ({b['percentage']:.1f}% of total)")
            output.append(f"   Calls: {b['calls']}")

        # Suggestions
        output.append("\n" + "=" * 60)
        output.append("OPTIMIZATION SUGGESTIONS")
        output.append("=" * 60)
        suggestions = suggest_optimizations(bottlenecks)
        output.extend(suggestions)

    report = '\n'.join(output)
    print(report)

    # Save to file if requested
    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
