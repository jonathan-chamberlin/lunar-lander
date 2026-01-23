#!/usr/bin/env python3
"""Test script for video recording functionality.

This is a tracer bullet test that runs 10 episodes with video recording
to verify the frame saving and video compilation pipeline works.

Usage:
    cd lunar-lander
    .venv-3.12.5/Scripts/python.exe tools/test_video_recording.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import Config, RunConfig, VideoConfig
from training.runner import run_training
from training.training_options import TrainingOptions
from training.video import compile_all_videos, create_compilation_video


def main():
    """Run video recording test."""
    print("=" * 60)
    print("VIDEO RECORDING TEST")
    print("=" * 60)

    # Experiment paths
    exp_dir = Path(__file__).parent.parent / "experiments" / "EXP_TEST_video_test"
    frames_dir = exp_dir / "frames"
    video_dir = exp_dir / "video"
    results_dir = exp_dir / "results"
    charts_dir = exp_dir / "charts"

    print(f"Experiment directory: {exp_dir}")
    print(f"Frames will be saved to: {frames_dir}")
    print(f"Videos will be saved to: {video_dir}")

    # Create directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Configure for 10 episodes, all rendered and recorded
    num_episodes = 10
    record_episodes = tuple(range(num_episodes))  # Record all 10

    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Recording episodes: {record_episodes}")
    print(f"  Video framerate: 75 FPS")

    # Create config with video recording
    run_config = RunConfig(
        num_episodes=num_episodes,
        render_mode='all',  # Must render to record
        print_mode='background',  # Use background mode to avoid unicode issues
        training_enabled=False,  # Skip training for speed
    )

    video_config = VideoConfig(
        record_episodes=record_episodes,
        video_framerate=75,
        frame_format="png"
    )

    config = Config(
        run=run_config,
        video=video_config
    )

    # Create training options
    options = TrainingOptions(
        output_mode='background',
        results_dir=results_dir,
        charts_dir=charts_dir,
        frames_dir=frames_dir,
        video_dir=video_dir,
        run_name="video_test",
        enable_logging=False,  # Skip logging for test
        save_model=False,  # Skip model save
        require_pygame=True,
        is_experiment=True,
    )

    print("\n" + "=" * 60)
    print("PHASE 1: Running episodes and saving frames...")
    print("=" * 60)

    try:
        result = run_training(config, options)
        print(f"\nTraining completed: {result.total_episodes} episodes")
        print(f"Success rate: {result.success_rate:.1f}%")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check if frames were saved
    print("\n" + "=" * 60)
    print("PHASE 2: Checking saved frames...")
    print("=" * 60)

    episode_dirs = list(frames_dir.glob("episode_*"))
    print(f"Found {len(episode_dirs)} episode directories")

    total_frames = 0
    for ep_dir in sorted(episode_dirs)[:5]:  # Show first 5
        frames = list(ep_dir.glob("frame_*.png"))
        total_frames += len(frames)
        print(f"  {ep_dir.name}: {len(frames)} frames")

    if len(episode_dirs) > 5:
        print(f"  ... and {len(episode_dirs) - 5} more episodes")

    print(f"\nTotal frames saved: {total_frames}")

    if total_frames == 0:
        print("ERROR: No frames were saved!")
        return 1

    # Compile videos
    print("\n" + "=" * 60)
    print("PHASE 3: Compiling videos...")
    print("=" * 60)

    video_paths = compile_all_videos(
        frames_dir=frames_dir,
        video_dir=video_dir,
        framerate=75,
        frame_format="png"
    )

    print(f"\nCompiled {len(video_paths)} videos:")
    for vp in video_paths[:5]:
        print(f"  {vp.name}")
    if len(video_paths) > 5:
        print(f"  ... and {len(video_paths) - 5} more")

    # Create compilation
    print("\n" + "=" * 60)
    print("PHASE 4: Creating compilation video...")
    print("=" * 60)

    compilation_path = create_compilation_video(
        video_dir=video_dir,
        output_name="all_episodes.mp4",
        framerate=75
    )

    if compilation_path:
        print(f"\nCompilation created: {compilation_path}")
        print(f"File size: {compilation_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("ERROR: Failed to create compilation video!")
        return 1

    # Check run_data.jsonl
    print("\n" + "=" * 60)
    print("PHASE 5: Checking chart data export...")
    print("=" * 60)

    run_data_path = results_dir / "run_data.jsonl"
    if run_data_path.exists():
        import json
        with open(run_data_path, 'r', encoding='utf-8') as f:
            chart_data = json.loads(f.readline())

        print(f"\nChart data file: {run_data_path}")
        print(f"File size: {run_data_path.stat().st_size / 1024:.2f} KB")
        print(f"\nData contents:")
        print(f"  Total episodes: {chart_data.get('total_episodes', 'N/A')}")
        print(f"  Batch size: {chart_data.get('batch_size', 'N/A')}")
        print(f"  Success count: {chart_data.get('success_count', 'N/A')}")
        print(f"  Max streak: {chart_data.get('max_streak', 'N/A')}")
        print(f"  Env rewards count: {len(chart_data.get('env_rewards', []))}")
        print(f"  Durations count: {len(chart_data.get('durations', []))}")
        print(f"  Outcomes count: {len(chart_data.get('outcomes', []))}")
        print(f"  Batch success rates: {chart_data.get('batch_success_rates', [])}")
    else:
        print(f"ERROR: run_data.jsonl not found at {run_data_path}")
        return 1

    # Check chart creation
    print("\n" + "=" * 60)
    print("PHASE 6: Checking chart creation...")
    print("=" * 60)

    chart_files = list(charts_dir.glob("chart_*.png"))
    if chart_files:
        print(f"\nFound {len(chart_files)} chart(s):")
        for chart_file in chart_files:
            print(f"  {chart_file.name} ({chart_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"WARNING: No charts found in {charts_dir}")
        print("(Charts may not be generated for only 10 episodes)")

    print("\n" + "=" * 60)
    print("SUCCESS! Video recording + data export + charts test completed.")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Frames: {frames_dir}")
    print(f"  Videos: {video_dir}")
    print(f"  Compilation: {compilation_path}")
    print(f"  Chart data: {run_data_path}")
    if chart_files:
        print(f"  Charts: {charts_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
