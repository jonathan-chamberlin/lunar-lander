"""Compare vectorized vs rendered environment behavior.

This script tests whether the simulation behaves identically between:
1. AsyncVectorEnv (render_mode='none') - used by sweeps
2. Single rendered env (render_mode='all') - used in human mode

If results differ significantly, there's a bug in one of the paths.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch as T

from config import Config, RunConfig
from training.trainer import TD3Trainer
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from training.noise import OUActionNoise
from training.environment import create_environments, shape_reward, compute_noise_scale, EpisodeManager
from data_types import Experience


def run_vectorized_mode(config: Config, num_episodes: int) -> dict:
    """Run training using AsyncVectorEnv (render_mode='none').

    This is the code path used by sweep_runner.py.
    """
    print(f"\n{'='*60}")
    print("VECTORIZED MODE (render_mode='none', AsyncVectorEnv)")
    print(f"Running {num_episodes} episodes with {config.run.num_envs} parallel envs")
    print(f"{'='*60}")

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

    episode_rewards = []
    successes = []
    first_success_episode = None
    start_time = time.time()

    with create_environments(config.run, config.environment) as env_bundle:
        observations, _ = env_bundle.vec_env.reset()
        states = T.from_numpy(observations).float()
        episode_manager = EpisodeManager(config.run.num_envs)

        completed_episodes = 0
        steps_since_training = 0
        training_started = False

        while completed_episodes < num_episodes:
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

                    episode_manager.reset_env(i)
                    noise.reset(i)
                    completed_episodes += 1

                    if completed_episodes % 100 == 0:
                        recent_success = sum(successes[-100:]) / min(100, len(successes)) * 100
                        print(f"  Episode {completed_episodes}: Success rate (last 100) = {recent_success:.1f}%")

                    if completed_episodes >= num_episodes:
                        break

            states = next_states
            observations = next_observations
            steps_since_training += config.run.num_envs

            # Training
            if config.run.training_enabled and replay_buffer.is_ready(config.training.min_experiences_before_training):
                if not training_started:
                    print(f"  Training started at episode {completed_episodes}")
                    training_started = True

                if config.training.use_per:
                    progress = completed_episodes / num_episodes
                    replay_buffer.anneal_beta(progress)

                updates_to_do = max(1, steps_since_training // 4)
                trainer.train_on_buffer(replay_buffer, updates_to_do)
                steps_since_training = 0
                trainer.step_schedulers()

    elapsed = time.time() - start_time

    return {
        'mode': 'vectorized',
        'num_episodes': len(episode_rewards),
        'success_rate': sum(successes) / len(successes) * 100,
        'mean_reward': float(np.mean(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'first_success': first_success_episode,
        'total_successes': sum(successes),
        'elapsed_time': elapsed,
    }


def run_rendered_mode(config: Config, num_episodes: int) -> dict:
    """Run training using single rendered environment (render_mode='all').

    This is the code path used in human mode / main.py with render_mode='all'.
    Note: This runs headless (no display) but uses the same code path.
    """
    import sys
    print(f"\n{'='*60}")
    print("RENDERED MODE (render_mode='all', single env)")
    print(f"Running {num_episodes} episodes sequentially")
    print(f"{'='*60}")
    sys.stdout.flush()

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

    # Single env noise generator
    noise = OUActionNoise(config.noise, num_envs=1)

    episode_rewards = []
    successes = []
    first_success_episode = None
    start_time = time.time()

    # Use render_mode='none' config to get env_bundle, but we'll only use render_env
    with create_environments(config.run, config.environment) as env_bundle:
        completed_episodes = 0
        steps_since_training = 0
        training_started = False

        while completed_episodes < num_episodes:
            # Reset single environment
            obs, _ = env_bundle.render_env.reset()
            state = T.from_numpy(obs).float()

            noise_scale = compute_noise_scale(
                completed_episodes,
                config.noise.noise_scale_initial,
                config.noise.noise_scale_final,
                config.noise.noise_decay_episodes
            )

            episode_rewards_list = []
            episode_done = False

            while not episode_done:
                # Generate action
                if completed_episodes < config.run.random_warmup_episodes:
                    action = T.from_numpy(env_bundle.render_env.action_space.sample()).float()
                else:
                    with T.no_grad():
                        action_noise = noise.generate_single() * noise_scale
                        action = (trainer.actor(state) + action_noise).float()
                        action[0] = T.clamp(action[0], -1.0, 1.0)
                        action[1] = T.clamp(action[1], -1.0, 1.0)

                action_np = action.detach().cpu().numpy()

                # Step environment
                next_obs, reward, terminated, truncated, info = env_bundle.render_env.step(action_np)
                next_state = T.from_numpy(next_obs).float()

                # Apply reward shaping
                current_step = len(episode_rewards_list)
                shaped_reward = shape_reward(obs, reward, terminated, step=current_step)

                # Store experience
                experience = Experience(
                    state=state.detach().clone(),
                    action=action.detach().clone(),
                    reward=T.tensor(shaped_reward, dtype=T.float32),
                    next_state=next_state.detach().clone(),
                    done=T.tensor(terminated)
                )
                replay_buffer.push(experience)

                state = next_state
                obs = next_obs
                episode_rewards_list.append(float(reward))
                steps_since_training += 1

                if terminated or truncated:
                    episode_done = True

            # Episode complete
            env_reward = sum(episode_rewards_list)
            success = env_reward >= config.environment.success_threshold
            episode_rewards.append(env_reward)
            successes.append(success)

            if success and first_success_episode is None:
                first_success_episode = completed_episodes

            noise.reset(0)
            completed_episodes += 1

            if completed_episodes % 100 == 0:
                recent_success = sum(successes[-100:]) / min(100, len(successes)) * 100
                print(f"  Episode {completed_episodes}: Success rate (last 100) = {recent_success:.1f}%")
                sys.stdout.flush()

            # Training after episode
            if config.run.training_enabled and replay_buffer.is_ready(config.training.min_experiences_before_training):
                if not training_started:
                    print(f"  Training started at episode {completed_episodes}")
                    training_started = True

                if config.training.use_per:
                    progress = completed_episodes / num_episodes
                    replay_buffer.anneal_beta(progress)

                updates_to_do = max(1, len(episode_rewards_list) // 4)
                trainer.train_on_buffer(replay_buffer, updates_to_do)
                steps_since_training = 0
                trainer.step_schedulers()

    elapsed = time.time() - start_time

    return {
        'mode': 'rendered',
        'num_episodes': len(episode_rewards),
        'success_rate': sum(successes) / len(successes) * 100,
        'mean_reward': float(np.mean(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'first_success': first_success_episode,
        'total_successes': sum(successes),
        'elapsed_time': elapsed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare vectorized vs rendered environment modes')
    parser.add_argument('--episodes', '-n', type=int, default=500,
                       help='Number of episodes per mode (default: 500)')
    parser.add_argument('--mode', '-m', choices=['both', 'vectorized', 'rendered'], default='both',
                       help='Which mode(s) to run (default: both)')
    args = parser.parse_args()

    # Create config with render_mode='none' (we handle the mode difference in the functions)
    base_config = Config()
    config = Config(
        training=base_config.training,
        noise=base_config.noise,
        run=replace(base_config.run,
                    num_episodes=args.episodes,
                    render_mode='none',  # We handle this differently in each function
                    print_mode='silent'),
        environment=base_config.environment,
        display=base_config.display
    )

    print(f"\n{'#'*60}")
    print("# ENVIRONMENT MODE COMPARISON EXPERIMENT")
    print(f"# Episodes per mode: {args.episodes}")
    print(f"# Training config: actor_lr={config.training.actor_lr}, critic_lr={config.training.critic_lr}")
    print(f"# Buffer size: {config.training.buffer_size}, min_experiences: {config.training.min_experiences_before_training}")
    print(f"# Noise sigma: {config.noise.sigma}")
    print(f"{'#'*60}")

    results = {}

    if args.mode in ('both', 'vectorized'):
        results['vectorized'] = run_vectorized_mode(config, args.episodes)

    if args.mode in ('both', 'rendered'):
        results['rendered'] = run_rendered_mode(config, args.episodes)

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    for mode, r in results.items():
        print(f"\n{mode.upper()} MODE:")
        print(f"  Success rate: {r['success_rate']:.1f}% ({r['total_successes']}/{r['num_episodes']})")
        print(f"  Mean reward: {r['mean_reward']:.1f} +/- {r['std_reward']:.1f}")
        print(f"  Max reward: {r['max_reward']:.1f}")
        print(f"  First success: episode {r['first_success']}")
        print(f"  Time: {r['elapsed_time']:.1f}s")

    if len(results) == 2:
        v = results['vectorized']
        r = results['rendered']

        print(f"\n{'='*60}")
        print("DELTA (rendered - vectorized):")
        print(f"{'='*60}")
        print(f"  Success rate: {r['success_rate'] - v['success_rate']:+.1f}%")
        print(f"  Mean reward: {r['mean_reward'] - v['mean_reward']:+.1f}")
        print(f"  Max reward: {r['max_reward'] - v['max_reward']:+.1f}")

        if abs(r['success_rate'] - v['success_rate']) > 5:
            print(f"\n*** SIGNIFICANT DIFFERENCE DETECTED ***")
            print(f"The two modes behave differently!")
            if r['success_rate'] > v['success_rate']:
                print(f"Rendered mode outperforms vectorized mode by {r['success_rate'] - v['success_rate']:.1f}%")
                print("This suggests a BUG in the vectorized training path!")
            else:
                print(f"Vectorized mode outperforms rendered mode by {v['success_rate'] - r['success_rate']:.1f}%")
        else:
            print(f"\nModes appear to behave similarly (difference < 5%)")


if __name__ == '__main__':
    main()
