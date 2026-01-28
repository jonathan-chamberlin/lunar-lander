#!/usr/bin/env python3
"""
Memory Profiler

Track memory usage over time for Python programs.

Usage:
    python memory_profiler.py --target "python script.py"
    python memory_profiler.py --target "python script.py" --interval 0.5
"""

import argparse
import subprocess
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_memory_usage(pid: int) -> dict:
    """Get memory usage for a process."""
    if not HAS_PSUTIL:
        return {"rss": 0, "vms": 0}

    try:
        process = psutil.Process(pid)
        mem = process.memory_info()
        return {
            "rss": mem.rss / 1024 / 1024,  # MB
            "vms": mem.vms / 1024 / 1024,  # MB
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"rss": 0, "vms": 0}


def monitor_memory(pid: int, interval: float, results: list, stop_event: threading.Event):
    """Monitor memory usage in a background thread."""
    start_time = time.time()

    while not stop_event.is_set():
        elapsed = time.time() - start_time
        mem = get_memory_usage(pid)

        if mem["rss"] > 0:
            results.append({
                "time": elapsed,
                "rss_mb": mem["rss"],
                "vms_mb": mem["vms"]
            })

        time.sleep(interval)


def format_memory_report(samples: list[dict], peak_rss: float, peak_vms: float) -> str:
    """Format memory profiling results."""
    output = []

    output.append("\n" + "=" * 60)
    output.append("MEMORY PROFILE SUMMARY")
    output.append("=" * 60)

    if not samples:
        output.append("No memory samples collected.")
        return '\n'.join(output)

    # Statistics
    rss_values = [s["rss_mb"] for s in samples]
    duration = samples[-1]["time"] if samples else 0

    output.append(f"\nDuration: {duration:.1f} seconds")
    output.append(f"Samples: {len(samples)}")
    output.append(f"\nMemory (RSS):")
    output.append(f"  Initial: {rss_values[0]:.1f} MB")
    output.append(f"  Peak: {max(rss_values):.1f} MB")
    output.append(f"  Final: {rss_values[-1]:.1f} MB")
    output.append(f"  Growth: {rss_values[-1] - rss_values[0]:.1f} MB")

    # Memory timeline (simple ASCII chart)
    output.append("\n" + "=" * 60)
    output.append("MEMORY TIMELINE")
    output.append("=" * 60)

    max_mem = max(rss_values)
    min_mem = min(rss_values)
    range_mem = max_mem - min_mem if max_mem > min_mem else 1

    # Sample at most 20 points for the chart
    step = max(1, len(samples) // 20)
    chart_samples = samples[::step]

    for s in chart_samples:
        normalized = (s["rss_mb"] - min_mem) / range_mem
        bar_len = int(normalized * 40)
        bar = "#" * bar_len
        output.append(f"{s['time']:6.1f}s | {bar:<40} {s['rss_mb']:.1f} MB")

    # Warnings
    if rss_values[-1] - rss_values[0] > 100:
        output.append("\n" + "=" * 60)
        output.append("WARNING: Significant memory growth detected!")
        output.append("This may indicate a memory leak.")
        output.append("=" * 60)

    return '\n'.join(output)


def run_with_memory_monitoring(command: str, interval: float = 1.0) -> tuple[list, int]:
    """Run command while monitoring memory."""
    print(f"Running: {command}")
    print(f"Memory sampling interval: {interval}s")
    print("-" * 60)

    # Start the process
    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Start monitoring thread
    samples = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(process.pid, interval, samples, stop_event)
    )
    monitor_thread.start()

    # Wait for process
    try:
        stdout, stderr = process.communicate(timeout=3600)
        return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = -1
    finally:
        stop_event.set()
        monitor_thread.join()

    return samples, return_code


def simple_memory_check():
    """Simple memory check without psutil (fallback)."""
    print("=" * 60)
    print("SIMPLE MEMORY CHECK")
    print("=" * 60)
    print("\npsutil not installed. Install with: pip install psutil")
    print("\nAlternative: Use tracemalloc in your code:")
    print("""
import tracemalloc

tracemalloc.start()

# Your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
""")


def main():
    parser = argparse.ArgumentParser(description="Profile memory usage")
    parser.add_argument("--target", "-t", help="Command to profile")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Sampling interval in seconds")
    parser.add_argument("--output", "-o", help="Save report to file")
    args = parser.parse_args()

    if not HAS_PSUTIL:
        simple_memory_check()
        return 1

    if not args.target:
        parser.print_help()
        return 1

    print("=" * 60)
    print("MEMORY PROFILER")
    print("=" * 60)
    print(f"Target: {args.target}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    samples, return_code = run_with_memory_monitoring(args.target, args.interval)

    peak_rss = max(s["rss_mb"] for s in samples) if samples else 0
    peak_vms = max(s["vms_mb"] for s in samples) if samples else 0

    report = format_memory_report(samples, peak_rss, peak_vms)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
