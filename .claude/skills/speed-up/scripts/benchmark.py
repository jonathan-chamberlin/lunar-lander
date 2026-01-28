#!/usr/bin/env python3
"""
Benchmark Utility

Compare performance of code snippets before and after optimization.

Usage:
    python benchmark.py  # Interactive mode
    python benchmark.py --file snippets.json
"""

import argparse
import timeit
import statistics
from typing import Callable
import json


def benchmark(func: Callable, n_runs: int = 100, n_iterations: int = 1000) -> dict:
    """Benchmark a function and return statistics."""
    times = []

    for _ in range(n_runs):
        start = timeit.default_timer()
        for _ in range(n_iterations):
            func()
        end = timeit.default_timer()
        times.append((end - start) / n_iterations)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "n_runs": n_runs,
        "n_iterations": n_iterations
    }


def compare(before: Callable, after: Callable, name: str = "Comparison",
            n_runs: int = 100, n_iterations: int = 1000) -> dict:
    """Compare two implementations."""
    print(f"\nBenchmarking: {name}")
    print("-" * 50)

    print("  Running 'before'...", end=" ", flush=True)
    before_stats = benchmark(before, n_runs, n_iterations)
    print(f"done ({before_stats['mean']*1e6:.2f} µs/call)")

    print("  Running 'after'...", end=" ", flush=True)
    after_stats = benchmark(after, n_runs, n_iterations)
    print(f"done ({after_stats['mean']*1e6:.2f} µs/call)")

    speedup = before_stats['mean'] / after_stats['mean'] if after_stats['mean'] > 0 else float('inf')

    result = {
        "name": name,
        "before": before_stats,
        "after": after_stats,
        "speedup": speedup,
        "improvement_percent": (1 - after_stats['mean'] / before_stats['mean']) * 100 if before_stats['mean'] > 0 else 0
    }

    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {result['improvement_percent']:.1f}%")

    return result


def run_common_comparisons():
    """Run benchmarks for common optimization patterns."""
    import random

    results = []

    # 1. List append vs comprehension
    data = list(range(1000))

    def append_loop():
        result = []
        for x in data:
            result.append(x * 2)
        return result

    def list_comp():
        return [x * 2 for x in data]

    results.append(compare(append_loop, list_comp, "Append loop vs list comprehension"))

    # 2. String concatenation
    words = ["word"] * 100

    def string_concat():
        result = ""
        for w in words:
            result += w
        return result

    def join_method():
        return "".join(words)

    results.append(compare(string_concat, join_method, "String += vs join()"))

    # 3. Membership test: list vs set
    items = list(range(1000))
    items_set = set(items)
    test_values = [random.randint(0, 1500) for _ in range(100)]

    def list_membership():
        return [x in items for x in test_values]

    def set_membership():
        return [x in items_set for x in test_values]

    results.append(compare(list_membership, set_membership, "List membership vs set membership"))

    # 4. Dict key access
    d = {f"key_{i}": i for i in range(100)}

    def repeated_access():
        total = 0
        for _ in range(100):
            total += d["key_50"]
        return total

    def cached_access():
        val = d["key_50"]
        total = 0
        for _ in range(100):
            total += val
        return total

    results.append(compare(repeated_access, cached_access, "Repeated dict access vs cached"))

    # 5. range(len()) vs enumerate
    items = list(range(1000))

    def range_len():
        result = []
        for i in range(len(items)):
            result.append((i, items[i]))
        return result

    def enumerate_method():
        result = []
        for i, item in enumerate(items):
            result.append((i, item))
        return result

    results.append(compare(range_len, enumerate_method, "range(len()) vs enumerate"))

    return results


def format_results(results: list[dict]) -> str:
    """Format results as a table."""
    output = []
    output.append("\n" + "=" * 70)
    output.append("BENCHMARK SUMMARY")
    output.append("=" * 70)
    output.append(f"\n{'Pattern':<45} {'Speedup':>10} {'Improvement':>12}")
    output.append("-" * 70)

    for r in sorted(results, key=lambda x: x['speedup'], reverse=True):
        output.append(f"{r['name']:<45} {r['speedup']:>9.2f}x {r['improvement_percent']:>10.1f}%")

    output.append("-" * 70)

    # Highlight biggest wins
    if results:
        best = max(results, key=lambda x: x['speedup'])
        output.append(f"\nBiggest win: {best['name']} ({best['speedup']:.2f}x speedup)")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description="Benchmark code patterns")
    parser.add_argument("--file", help="JSON file with custom benchmarks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    print("=" * 70)
    print("PYTHON PERFORMANCE BENCHMARK")
    print("=" * 70)

    results = run_common_comparisons()

    if args.json:
        # Convert to JSON-serializable format
        for r in results:
            r['before'] = dict(r['before'])
            r['after'] = dict(r['after'])
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
