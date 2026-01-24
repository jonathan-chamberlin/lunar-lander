#!/usr/bin/env python3
"""Run EXP_023: Long training with proper video recording.

This script runs a 10,000 episode training where:
- Noise decays from 1.0 to 0.0 over 5000 episodes (exploration -> exploitation)
- Learning rate automatically decays to 20% via built-in scheduler
- Video is recorded at key milestones (fixed: now works with render_mode='none')
- Model checkpoints saved every 1000 episodes
- Batch size for chart reporting: 200 episodes

Usage:
    cd lunar-lander
    .venv-3.12.5/Scripts/python.exe tools/run_exp023.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import Config, RunConfig, VideoConfig, TrainingConfig, NoiseConfig, RewardShapingConfig
from training.runner import run_training
from training.training_options import TrainingOptions
from training.video import compile_all_videos, create_compilation_video


def main():
    """Run long training experiment with parameter decay and video recording."""
    print("=" * 70)
    print("EXP_023: LONG TRAINING WITH VIDEO RECORDING")
    print("=" * 70)

    # Experiment paths
    exp_dir = Path(__file__).parent.parent / "experiments" / "EXP_023_long_training_with_recording"
    frames_dir = exp_dir / "frames"
    video_dir = exp_dir / "video"
    results_dir = exp_dir / "results"
    charts_dir = exp_dir / "charts"

    # Create directories
    for d in [exp_dir, frames_dir, video_dir, results_dir, charts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment Directory: {exp_dir}")

    # Configure episodes and recording
    num_episodes = 10000

    # Record at milestones - more comprehensive coverage
    record_episodes = tuple(
        list(range(1, 6)) +             # Episodes 1-5
        list(range(40, 51)) +           # Episodes 40-50
        list(range(100, 104)) +         # Episodes 100-103
        list(range(300, 304)) +         # Episodes 300-303
        list(range(500, 506)) +         # Episodes 500-505
        list(range(1000, 1006)) +       # Episodes 1000-1005
        list(range(2000, 2006)) +       # Episodes 2000-2005
        list(range(3000, 3005)) +       # Episodes 3000-3004
        list(range(4000, 4005)) +       # Episodes 4000-4004
        list(range(5000, 5005)) +       # Episodes 5000-5004
        list(range(6000, 6005)) +       # Episodes 6000-6004
        list(range(7000, 7005)) +       # Episodes 7000-7004
        list(range(8000, 8005)) +       # Episodes 8000-8004
        list(range(9000, 9005)) +       # Episodes 9000-9004
        list(range(9996, 10001))        # Episodes 9996-10000
    )

    print(f"\nConfiguration:")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Noise: 1.0 -> 0.0 over 5000 episodes")
    print(f"  LR: Decays to 20% of initial over full run (built-in scheduler)")
    print(f"  Recording {len(record_episodes)} episodes at milestones")
    print(f"  Video framerate: 75 FPS")
    print(f"  Chart batch size: 200 episodes")
    print(f"  Model checkpoints: every 1000 episodes")

    # Training config with best settings
    training_config = TrainingConfig(
        actor_lr=0.001,
        critic_lr=0.001,
        hidden_sizes=(256, 128),
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        buffer_size=1 << 14,  # 16384
        min_experiences_before_training=2000,
        training_updates_per_episode=25,
        gradient_clip_value=10.0,
        policy_update_frequency=3,
        target_policy_noise=0.1,
        target_noise_clip=0.3,
        use_per=True,
    )

    # Noise decays from 1.0 to 0.0 over 5000 episodes
    # After episode 5000, noise stays at 0 = pure exploitation
    noise_config = NoiseConfig(
        sigma=0.3,
        theta=0.2,
        dt=0.01,
        noise_scale_initial=1.0,
        noise_scale_final=0.0,  # Zero noise for exploitation
        noise_decay_episodes=5000,  # Decay over first 5000 episodes
    )

    run_config = RunConfig(
        num_episodes=num_episodes,
        render_mode='none',  # Headless - video recording now works without display!
        print_mode='background',
        training_enabled=True,
    )

    video_config = VideoConfig(
        record_episodes=record_episodes,
        video_framerate=75,
        frame_format="png"
    )

    reward_shaping = RewardShapingConfig(
        time_penalty=False,
        altitude_bonus=True,
        leg_contact=True,
        stability=True,
    )

    config = Config(
        run=run_config,
        training=training_config,
        noise=noise_config,
        video=video_config,
        reward_shaping=reward_shaping,
    )

    # Training options
    options = TrainingOptions(
        output_mode='background',
        results_dir=results_dir,
        charts_dir=charts_dir,
        frames_dir=frames_dir,
        video_dir=video_dir,
        run_name="exp023",
        enable_logging=True,
        save_model=True,
        require_pygame=False,  # Headless mode
        is_experiment=True,
        diagnostics_batch_size=200,  # Chart reporting batch size (episodes per batch)
        memory_limit_mb=4000,  # 4GB limit - leaves room for browser and other apps
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING...")
    print("(This will take several hours)")
    print("=" * 70)

    try:
        result = run_training(config, options)
        print(f"\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total Episodes: {result.total_episodes}")
        print(f"Success Rate: {result.success_rate:.1f}%")
        print(f"Total Successes: {result.total_successes}")
        print(f"Max Consecutive: {result.max_consecutive_successes}")
        print(f"First Success: Episode {result.first_success_episode}")
        if result.final_100_success_rate is not None:
            print(f"Final 100 Success Rate: {result.final_100_success_rate:.1f}%")
        if result.final_100_mean_reward is not None:
            print(f"Final 100 Mean Reward: {result.final_100_mean_reward:.1f}")
        print(f"Elapsed Time: {result.elapsed_time:.1f}s ({result.elapsed_time/3600:.2f} hours)")
        print(f"\n100 CONSECUTIVE ACHIEVED: {'YES!' if result.max_consecutive_successes >= 100 else 'NO'}")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compile videos
    print("\n" + "=" * 70)
    print("COMPILING VIDEOS...")
    print("=" * 70)

    video_paths = compile_all_videos(
        frames_dir=frames_dir,
        video_dir=video_dir,
        framerate=75,
        frame_format="png"
    )

    print(f"Compiled {len(video_paths)} videos")

    # Create compilation
    compilation_path = create_compilation_video(
        video_dir=video_dir,
        output_name="all_episodes.mp4",
        framerate=75
    )

    if compilation_path:
        print(f"Compilation: {compilation_path}")
        print(f"Size: {compilation_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Results: {results_dir}")
    print(f"  Charts: {charts_dir}")
    print(f"  Videos: {video_dir}")
    if compilation_path:
        print(f"  Compilation: {compilation_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
