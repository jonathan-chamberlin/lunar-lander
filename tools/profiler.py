"""Profiling wrapper for lunar lander TD3 training.

Profiles training code to identify performance bottlenecks.

Usage:
    python tools/profiler.py --episodes 50
    python tools/profiler.py --episodes 20 --top 30
    python tools/profiler.py --memory --episodes 100
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_training(num_episodes: int = 50, top_n: int = 20) -> str:
    """Profile training for a number of episodes.

    Args:
        num_episodes: Number of episodes to run
        top_n: Number of top functions to display

    Returns:
        Formatted profiling report string
    """
    from dataclasses import replace
    from config import Config

    # Create config with reduced episodes and no rendering
    base_config = Config()
    config = Config(
        training=base_config.training,
        noise=base_config.noise,
        run=replace(base_config.run,
                    num_episodes=num_episodes,
                    render_mode='none'),
        environment=base_config.environment,
        display=base_config.display
    )

    # Profile the training function
    profiler = cProfile.Profile()

    print(f"Profiling training for {num_episodes} episodes...")
    start_time = time.time()

    profiler.enable()
    _run_training(config)
    profiler.disable()

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s")

    # Format results
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')

    # Generate report
    report_lines = [
        "=" * 80,
        "PROFILING REPORT",
        "=" * 80,
        f"Episodes: {num_episodes}",
        f"Total time: {elapsed:.1f}s",
        f"Average per episode: {elapsed/num_episodes:.3f}s",
        "",
        f"Top {top_n} functions by cumulative time:",
        "-" * 80,
    ]

    # Get stats as string
    stats.print_stats(top_n)
    stats_output = stream.getvalue()
    report_lines.append(stats_output)

    # Add optimization suggestions based on common bottlenecks
    suggestions = analyze_bottlenecks(stats)
    if suggestions:
        report_lines.extend([
            "",
            "=" * 80,
            "OPTIMIZATION SUGGESTIONS",
            "=" * 80,
        ])
        for i, suggestion in enumerate(suggestions, 1):
            report_lines.append(f"{i}. {suggestion}")

    return "\n".join(report_lines)


def _run_training(config) -> None:
    """Run training with the given config (extracted for profiling)."""
    import torch as T
    import numpy as np

    from training.trainer import TD3Trainer
    from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    from training.noise import OUActionNoise
    from training.environment import create_environments, shape_reward, compute_noise_scale, EpisodeManager
    from data_types import Experience

    trainer = TD3Trainer(config.training, config.environment, config.run)

    if config.training.use_per:
        replay_buffer = PrioritizedReplayBuffer(
            capacity=config.training.buffer_size,
            alpha=config.training.per_alpha,
            beta_start=config.training.per_beta_start,
            beta_end=config.training.per_beta_end,
            epsilon=config.training.per_epsilon
        )
    else:
        replay_buffer = ReplayBuffer(config.training.buffer_size)

    noise = OUActionNoise(config.noise, config.run.num_envs)

    with create_environments(config.run, config.environment) as env_bundle:
        observations, _ = env_bundle.vec_env.reset()
        states = T.from_numpy(observations).float()
        episode_manager = EpisodeManager(config.run.num_envs)

        completed_episodes = 0
        steps_since_training = 0
        training_started = False

        while completed_episodes < config.run.num_episodes:
            noise_scale = compute_noise_scale(
                completed_episodes,
                config.noise.noise_scale_initial,
                config.noise.noise_scale_final,
                config.noise.noise_decay_episodes
            )

            if completed_episodes < config.run.random_warmup_episodes:
                actions = T.from_numpy(env_bundle.vec_env.action_space.sample()).float()
            else:
                with T.no_grad():
                    exploration_noise = noise.generate() * noise_scale
                    actions = trainer.actor(states) + exploration_noise
                    actions[:, 0] = T.clamp(actions[:, 0], -1.0, 1.0)
                    actions[:, 1] = T.clamp(actions[:, 1], -1.0, 1.0)

            actions_np = actions.detach().cpu().numpy()
            next_observations, rewards, terminateds, truncateds, infos = env_bundle.vec_env.step(actions_np)
            next_states = T.from_numpy(next_observations).float()

            states_detached = states.detach()
            actions_detached = actions.detach()
            next_states_detached = next_states.detach()

            for i in range(config.run.num_envs):
                current_step = episode_manager.step_counts[i]
                shaped_reward = shape_reward(observations[i], rewards[i], terminateds[i], step=current_step)
                episode_manager.add_step(i, float(rewards[i]), shaped_reward, actions_np[i], observations[i].copy())

                experience = Experience(
                    state=states_detached[i].clone(),
                    action=actions_detached[i].clone(),
                    reward=T.tensor(shaped_reward, dtype=T.float32),
                    next_state=next_states_detached[i].clone(),
                    done=T.tensor(terminateds[i])
                )
                replay_buffer.push(experience)

                if terminateds[i] or truncateds[i]:
                    episode_manager.reset_env(i)
                    noise.reset(i)
                    completed_episodes += 1

                    if completed_episodes >= config.run.num_episodes:
                        break

            states = next_states
            observations = next_observations
            steps_since_training += config.run.num_envs

            if config.run.training_enabled and replay_buffer.is_ready(config.training.min_experiences_before_training):
                if not training_started:
                    training_started = True

                if config.training.use_per:
                    progress = completed_episodes / config.run.num_episodes
                    replay_buffer.anneal_beta(progress)

                updates_to_do = max(1, steps_since_training // 4)
                trainer.train_on_buffer(replay_buffer, updates_to_do)
                steps_since_training = 0
                trainer.step_schedulers()


def analyze_bottlenecks(stats: pstats.Stats) -> list:
    """Analyze stats to suggest optimizations.

    Args:
        stats: pstats.Stats object with profiling data

    Returns:
        List of optimization suggestion strings
    """
    suggestions = []

    # Get the stats data
    stats_list = stats.stats

    # Common bottleneck patterns and suggestions
    bottleneck_patterns = {
        'sample': "Consider using numpy vectorized operations for batch sampling",
        'stack': "torch.stack in loops is slow - consider pre-allocating tensors",
        'clone': "Minimize tensor cloning - use views where possible",
        'to_numpy': "Reduce tensor<->numpy conversions by batching operations",
        'backward': "Gradient computation is expected to be expensive - check batch size",
        'step': "Environment stepping time - consider using vectorized environments",
        'analyze': "Behavior analysis overhead - consider reducing analysis frequency",
    }

    for func_key, func_stats in stats_list.items():
        filename, line, func_name = func_key
        cumtime = func_stats[3]  # Cumulative time

        for pattern, suggestion in bottleneck_patterns.items():
            if pattern in func_name.lower() and cumtime > 0.5:
                suggestions.append(f"{func_name} ({cumtime:.2f}s): {suggestion}")
                break

    return suggestions[:5]  # Return top 5 suggestions


def profile_memory(num_episodes: int = 100) -> str:
    """Profile memory usage during training.

    Args:
        num_episodes: Number of episodes to run

    Returns:
        Memory profiling report string
    """
    try:
        import tracemalloc
    except ImportError:
        return "tracemalloc not available for memory profiling"

    from dataclasses import replace
    from config import Config

    base_config = Config()
    config = Config(
        training=base_config.training,
        noise=base_config.noise,
        run=replace(base_config.run,
                    num_episodes=num_episodes,
                    render_mode='none'),
        environment=base_config.environment,
        display=base_config.display
    )

    print(f"Memory profiling training for {num_episodes} episodes...")

    tracemalloc.start()
    _run_training(config)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report_lines = [
        "=" * 80,
        "MEMORY PROFILING REPORT",
        "=" * 80,
        f"Episodes: {num_episodes}",
        f"Current memory: {current / 1024 / 1024:.1f} MB",
        f"Peak memory: {peak / 1024 / 1024:.1f} MB",
    ]

    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description='Profile lunar lander training')
    parser.add_argument('--episodes', '-e', type=int, default=50,
                        help='Number of episodes to profile (default: 50)')
    parser.add_argument('--top', '-t', type=int, default=20,
                        help='Number of top functions to display (default: 20)')
    parser.add_argument('--memory', '-m', action='store_true',
                        help='Profile memory usage instead of CPU')
    parser.add_argument('--output', '-o', type=str,
                        help='Save report to file')

    args = parser.parse_args()

    # Create results directory
    results_dir = Path('profile_results')
    results_dir.mkdir(exist_ok=True)

    if args.memory:
        report = profile_memory(args.episodes)
    else:
        report = profile_training(args.episodes, args.top)

    print(report)

    # Save to file if requested or by default
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        profile_type = "memory" if args.memory else "cpu"
        output_path = results_dir / f"profile_{profile_type}_{timestamp}.txt"

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    main()
