"""Hyperparameter sweep runner for lunar lander TD3 training.

Runs multiple training sessions with different hyperparameter configurations
and collects results for comparison.

Usage:
    python tools/sweep_runner.py sweep_configs/example_lr_sweep.json
    python tools/sweep_runner.py --config sweep_configs/example_lr_sweep.json --dry-run
    python tools/sweep_runner.py --config config.json --output-dir experiments/EXP_001/results
"""

import argparse
import json
import itertools
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config, TrainingConfig, NoiseConfig, RunConfig


def load_sweep_config(config_path: str) -> Dict[str, Any]:
    """Load sweep configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_grid_configs(
    base_config: Config,
    parameters: Dict[str, List[Any]]
) -> Iterator[Tuple[Dict[str, Any], Config]]:
    """Generate all combinations for grid search.

    Args:
        base_config: Base configuration to modify
        parameters: Dict mapping param names to lists of values

    Yields:
        Tuples of (param_dict, modified_config)
    """
    param_names = list(parameters.keys())
    param_values = list(parameters.values())

    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        modified_config = apply_params_to_config(base_config, param_dict)
        yield param_dict, modified_config


def generate_random_configs(
    base_config: Config,
    parameters: Dict[str, Dict[str, Any]],
    num_samples: int,
    seed: int = 42
) -> Iterator[Tuple[Dict[str, Any], Config]]:
    """Generate random configurations from parameter ranges.

    Args:
        base_config: Base configuration to modify
        parameters: Dict mapping param names to {min, max, type} dicts
        num_samples: Number of random configurations to generate
        seed: Random seed for reproducibility

    Yields:
        Tuples of (param_dict, modified_config)
    """
    import random
    random.seed(seed)

    for _ in range(num_samples):
        param_dict = {}
        for name, spec in parameters.items():
            if spec.get('type') == 'int':
                value = random.randint(spec['min'], spec['max'])
            elif spec.get('type') == 'log':
                # Log-uniform sampling for learning rates
                import math
                log_min = math.log10(spec['min'])
                log_max = math.log10(spec['max'])
                value = 10 ** random.uniform(log_min, log_max)
            else:
                value = random.uniform(spec['min'], spec['max'])
            param_dict[name] = value

        modified_config = apply_params_to_config(base_config, param_dict)
        yield param_dict, modified_config


def apply_params_to_config(config: Config, params: Dict[str, Any]) -> Config:
    """Apply parameter overrides to a config object.

    Args:
        config: Base configuration
        params: Dict of parameter names to values

    Returns:
        New Config with parameters applied
    """
    training_params = {}
    noise_params = {}
    run_params = {}

    # Map parameters to their config sections
    training_fields = set(TrainingConfig.__dataclass_fields__.keys())
    noise_fields = set(NoiseConfig.__dataclass_fields__.keys())
    run_fields = set(RunConfig.__dataclass_fields__.keys())

    for name, value in params.items():
        if name in training_fields:
            training_params[name] = value
        elif name in noise_fields:
            noise_params[name] = value
        elif name in run_fields:
            run_params[name] = value
        else:
            print(f"Warning: Unknown parameter '{name}', ignoring")

    # Create new config with modifications
    new_training = replace(config.training, **training_params) if training_params else config.training
    new_noise = replace(config.noise, **noise_params) if noise_params else config.noise
    new_run = replace(config.run, **run_params) if run_params else config.run

    return Config(
        training=new_training,
        noise=new_noise,
        run=new_run,
        environment=config.environment,
        display=config.display
    )


def run_training_with_config(
    config: Config,
    run_name: str,
    results_dir: Path,
    charts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run training with a specific configuration and collect results.

    Args:
        config: Training configuration
        run_name: Name for this run
        results_dir: Directory to save results
        charts_dir: Directory to save charts (if None, no charts generated)

    Returns:
        Dict with training results
    """
    # Import training components
    from training.trainer import TD3Trainer
    from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    from training.noise import OUActionNoise
    from training.environment import create_environments, shape_reward, compute_noise_scale, EpisodeManager
    from analysis.diagnostics import DiagnosticsTracker
    from analysis.behavior_analysis import BehaviorAnalyzer
    from analysis.charts import ChartGenerator

    import torch as T
    import numpy as np

    start_time = time.time()

    # Initialize components
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
    diagnostics = DiagnosticsTracker()
    behavior_analyzer = BehaviorAnalyzer()

    # Track results
    episode_rewards = []
    successes = []
    first_success_episode = None

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

            # Generate actions
            if completed_episodes < config.run.random_warmup_episodes:
                actions = T.from_numpy(env_bundle.vec_env.action_space.sample()).float()
            else:
                with T.no_grad():
                    exploration_noise = noise.generate() * noise_scale
                    actions = trainer.actor(states) + exploration_noise
                    actions[:, 0] = T.clamp(actions[:, 0], -1.0, 1.0)
                    actions[:, 1] = T.clamp(actions[:, 1], -1.0, 1.0)

            # Step environments
            actions_np = actions.detach().cpu().numpy()
            next_observations, rewards, terminateds, truncateds, infos = env_bundle.vec_env.step(actions_np)
            next_states = T.from_numpy(next_observations).float()

            states_detached = states.detach()
            actions_detached = actions.detach()
            next_states_detached = next_states.detach()

            from data_types import Experience

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
                    total_reward, env_reward, shaped_bonus, actions_array, observations_array, duration = \
                        episode_manager.get_episode_stats(i)

                    success = env_reward >= config.environment.success_threshold
                    episode_rewards.append(env_reward)
                    successes.append(success)

                    if success and first_success_episode is None:
                        first_success_episode = completed_episodes

                    # Record to diagnostics tracker
                    diagnostics.record_episode(
                        episode_num=completed_episodes,
                        env_reward=env_reward,
                        shaped_bonus=shaped_bonus,
                        duration_seconds=duration,
                        success=success
                    )

                    # Analyze behavior and record
                    if observations_array is not None and actions_array is not None:
                        behavior_report = behavior_analyzer.analyze(
                            observations_array,
                            actions_array,
                            terminateds[i],
                            truncateds[i]
                        )
                        diagnostics.record_behavior(
                            outcome=behavior_report.outcome,
                            behaviors=behavior_report.behaviors,
                            env_reward=env_reward,
                            success=success
                        )

                    episode_manager.reset_env(i)
                    noise.reset(i)
                    completed_episodes += 1

                    # Print batch completion every 100 episodes
                    if completed_episodes % 100 == 0:
                        batch_num = completed_episodes // 100
                        batch_successes = successes[-100:] if len(successes) >= 100 else successes
                        batch_success_rate = sum(batch_successes) / len(batch_successes) * 100 if batch_successes else 0
                        print(f"  Batch {batch_num} (Runs {(batch_num-1)*100+1}-{completed_episodes}) completed. Success: {batch_success_rate:.0f}%")

                    if completed_episodes >= config.run.num_episodes:
                        break

            states = next_states
            observations = next_observations
            steps_since_training += config.run.num_envs

            # Training
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

    elapsed_time = time.time() - start_time

    # Compute results
    episode_rewards = np.array(episode_rewards)
    successes = np.array(successes)

    results = {
        'run_name': run_name,
        'total_episodes': len(episode_rewards),
        'success_rate': float(np.mean(successes)) * 100,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'first_success_episode': first_success_episode,
        'final_100_success_rate': float(np.mean(successes[-100:])) * 100 if len(successes) >= 100 else None,
        'elapsed_time': elapsed_time,
    }

    # Save individual run results
    run_results_path = results_dir / f"{run_name}_results.json"
    with open(run_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate chart if charts_dir provided
    if charts_dir is not None:
        charts_dir.mkdir(parents=True, exist_ok=True)
        chart_path = charts_dir / f"{run_name}_chart.png"
        chart_generator = ChartGenerator(diagnostics)
        if chart_generator.generate_to_file(str(chart_path)):
            print(f"  Chart saved to: {chart_path}")

    return results


def run_sweep(
    sweep_config: Dict[str, Any],
    dry_run: bool = False,
    output_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Execute a hyperparameter sweep.

    Args:
        sweep_config: Sweep configuration dict
        dry_run: If True, only print configurations without running
        output_dir: Custom output directory. If None, uses sweep_results/<name>_<timestamp>

    Returns:
        List of result dicts from each run
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    sweep_name = sweep_config.get('name', 'sweep')

    if output_dir is not None:
        results_dir = Path(output_dir)
    else:
        results_dir = Path('sweep_results') / f"{sweep_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create charts directory
    charts_dir = results_dir.parent / 'charts' if output_dir else results_dir / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep config
    with open(results_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    # Create base config with reduced episodes for sweep
    base_config = Config()
    episodes_per_run = sweep_config.get('episodes_per_run', 500)
    base_config = Config(
        training=base_config.training,
        noise=base_config.noise,
        run=replace(base_config.run,
                    num_episodes=episodes_per_run,
                    render_mode='none'),  # No rendering for sweeps
        environment=base_config.environment,
        display=base_config.display
    )

    # Generate configurations
    sweep_type = sweep_config.get('type', 'grid')
    parameters = sweep_config.get('parameters', {})

    if sweep_type == 'grid':
        config_generator = generate_grid_configs(base_config, parameters)
    elif sweep_type == 'random':
        num_samples = sweep_config.get('num_samples', 10)
        seed = sweep_config.get('seed', 42)
        config_generator = generate_random_configs(base_config, parameters, num_samples, seed)
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")

    configs = list(config_generator)
    print(f"Generated {len(configs)} configurations for sweep '{sweep_name}'")
    print(f"Results will be saved to: {results_dir}")

    if dry_run:
        print("\n=== DRY RUN - Configurations to test ===")
        for i, (params, _) in enumerate(configs):
            print(f"  Run {i+1}: {params}")
        return []

    # Run each configuration
    all_results = []
    for i, (params, config) in enumerate(configs):
        run_name = f"run_{i+1:03d}"
        param_str = "_".join(f"{k}={v}" for k, v in params.items())

        print(f"\n{'='*60}")
        print(f"Running {run_name}: {param_str}")
        print(f"Progress: {i+1}/{len(configs)}")
        print('='*60)

        try:
            results = run_training_with_config(config, run_name, results_dir, charts_dir)
            results['parameters'] = params
            all_results.append(results)

            print(f"  Success rate: {results['success_rate']:.1f}%")
            print(f"  Mean reward: {results['mean_reward']:.1f}")
            print(f"  Time: {results['elapsed_time']:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                'run_name': run_name,
                'parameters': params,
                'error': str(e)
            })

    # Generate summary
    generate_sweep_summary(all_results, results_dir)

    return all_results


def generate_sweep_summary(results: List[Dict[str, Any]], results_dir: Path) -> None:
    """Generate a summary of sweep results.

    Args:
        results: List of result dicts from each run
        results_dir: Directory to save summary
    """
    import csv

    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("\nNo successful runs to summarize")
        return

    # Sort by success rate
    valid_results.sort(key=lambda x: x.get('success_rate', 0), reverse=True)

    # Save CSV summary
    csv_path = results_dir / 'summary.csv'
    fieldnames = ['run_name', 'success_rate', 'mean_reward', 'max_reward',
                  'first_success_episode', 'final_100_success_rate', 'elapsed_time', 'parameters']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in valid_results:
            row = {**r, 'parameters': json.dumps(r.get('parameters', {}))}
            writer.writerow(row)

    # Print summary
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    print(f"Total runs: {len(results)} ({len(valid_results)} successful)")
    print(f"Results saved to: {results_dir}")
    print("\nTop 5 configurations by success rate:")
    print("-"*80)
    print(f"{'Rank':<5} {'Success%':<10} {'Mean Reward':<12} {'Parameters':<50}")
    print("-"*80)

    for i, r in enumerate(valid_results[:5]):
        params_str = str(r.get('parameters', {}))
        if len(params_str) > 47:
            params_str = params_str[:44] + '...'
        print(f"{i+1:<5} {r['success_rate']:<10.1f} {r['mean_reward']:<12.1f} {params_str:<50}")

    # Save full results JSON
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest configuration: {valid_results[0]['parameters']}")
    print(f"  Success rate: {valid_results[0]['success_rate']:.1f}%")
    print(f"  Mean reward: {valid_results[0]['mean_reward']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep')
    parser.add_argument('config', nargs='?', help='Path to sweep config JSON file')
    parser.add_argument('--config', '-c', dest='config_flag', help='Path to sweep config JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Print configurations without running')
    parser.add_argument('--output-dir', '-o', dest='output_dir',
                        help='Custom output directory for results (e.g., experiments/EXP_001/results)')

    args = parser.parse_args()

    config_path = args.config or args.config_flag
    if not config_path:
        print("Usage: python tools/sweep_runner.py <config.json>")
        print("       python tools/sweep_runner.py --config <config.json> --dry-run")
        print("       python tools/sweep_runner.py --config <config.json> --output-dir experiments/EXP_001/results")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    sweep_config = load_sweep_config(config_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    run_sweep(sweep_config, dry_run=args.dry_run, output_dir=output_dir)


if __name__ == '__main__':
    main()
