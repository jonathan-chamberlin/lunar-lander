"""Main entry point for Lunar Lander TD3 training.

This module orchestrates the training loop using components from:
- config.py: Configuration dataclasses
- network.py: Neural network architectures
- trainer.py: TD3 training logic
- replay_buffer.py: Experience replay
- diagnostics.py: Metrics tracking and reporting
- environment.py: Gymnasium environment management
"""

import logging
import time
import warnings
from typing import Optional

import numpy as np
import pygame as pg
import torch as T

from config import Config, TrainingConfig, NoiseConfig, RunConfig, EnvironmentConfig
from diagnostics import DiagnosticsTracker, DiagnosticsReporter
from environment import (
    create_environments,
    shape_reward,
    compute_noise_scale,
    EpisodeManager
)
from network import OUActionNoise
from replay_buffer import ReplayBuffer
from trainer import TD3Trainer
from data_types import Experience, EpisodeResult, ActionStatistics
from behavior_analysis import BehaviorAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


# Module-level behavior analyzer instance
behavior_analyzer = BehaviorAnalyzer()


def finalize_episode(
    episode_num: int,
    total_reward: float,
    env_reward: float,
    shaped_bonus: float,
    steps: int,
    actions_array: np.ndarray,
    observations_array: np.ndarray,
    terminated: bool,
    truncated: bool,
    success_threshold: float,
    diagnostics: DiagnosticsTracker,
    replay_buffer: ReplayBuffer,
    min_experiences: int,
    rendered: bool = False
) -> EpisodeResult:
    """Create episode result, record it, print status, and track action stats."""
    success = total_reward >= success_threshold
    result = EpisodeResult(
        episode_num=episode_num,
        total_reward=total_reward,
        env_reward=env_reward,
        shaped_bonus=shaped_bonus,
        steps=steps,
        success=success
    )

    diagnostics.record_episode(result)

    status = "SUCCESS" if success else "FAILURE"
    rendered_tag = " | RENDERED" if rendered else ""
    print(f"Run {episode_num} | {status}{rendered_tag} | Reward: {total_reward:.1f} (env: {env_reward:.1f}, shaped: {shaped_bonus:.1f})")

    # Analyze and print behaviors
    if len(observations_array) > 0 and len(actions_array) > 0:
        behavior_report = behavior_analyzer.analyze(
            observations_array, actions_array, terminated, truncated
        )
        print(f"  Behaviors: {behavior_report}")

    if len(actions_array) > 0 and replay_buffer.is_ready(min_experiences):
        diagnostics.record_action_stats(ActionStatistics.from_actions(actions_array))

    return result


def run_rendered_episode(
    render_env,
    trainer: TD3Trainer,
    noise: OUActionNoise,
    replay_buffer: ReplayBuffer,
    episode_num: int,
    config: Config,
    diagnostics: DiagnosticsTracker
) -> Optional[EpisodeResult]:
    """Run a single rendered episode.

    Args:
        render_env: Gymnasium environment with rendering
        trainer: TD3 trainer instance
        noise: Noise generator for exploration
        replay_buffer: Experience replay buffer
        episode_num: Current episode number
        config: Full configuration
        diagnostics: Diagnostics tracker

    Returns:
        EpisodeResult if episode completed, None if user quit
    """
    obs, _ = render_env.reset()
    state = T.from_numpy(obs).float()

    noise_scale = compute_noise_scale(
        episode_num,
        config.noise.noise_scale_initial,
        config.noise.noise_scale_final,
        config.noise.noise_decay_episodes
    )

    rewards = []
    actions = []
    observations_list = []
    shaped_bonus = 0.0
    running = True
    user_quit = False
    episode_terminated = False
    episode_truncated = False

    while running:
        # Handle pygame events
        for event in pg.event.get():
            if event.type == 256:  # Window close
                render_env.close()
                running = False
                user_quit = True
                break

        if not running:
            break

        # Generate action
        if episode_num < config.run.random_warmup_episodes:
            action = T.from_numpy(render_env.action_space.sample()).float()
        else:
            action_noise = noise.generate_single() * noise_scale
            action = (trainer.actor(state) + action_noise).float()
            action[0] = T.clamp(action[0], -1.0, 1.0)
            action[1] = T.clamp(action[1], -1.0, 1.0)

        actions.append(action.detach().cpu().numpy())
        observations_list.append(obs.copy())

        # Step environment
        next_obs, reward, terminated, truncated, info = render_env.step(
            action.detach().numpy()
        )
        next_state = T.from_numpy(next_obs).float()

        # Apply reward shaping
        shaped_reward = shape_reward(obs, reward, terminated)
        shaped_bonus += (shaped_reward - reward)

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
        rewards.append(float(reward))

        if terminated or truncated:
            episode_terminated = terminated
            episode_truncated = truncated
            running = False

    if user_quit:
        return None

    # Compute and finalize episode
    env_reward = float(np.sum(rewards))
    total_reward = env_reward + shaped_bonus

    return finalize_episode(
        episode_num=episode_num,
        total_reward=total_reward,
        env_reward=env_reward,
        shaped_bonus=shaped_bonus,
        steps=len(rewards),
        actions_array=np.array(actions),
        observations_array=np.array(observations_list),
        terminated=episode_terminated,
        truncated=episode_truncated,
        success_threshold=config.environment.success_threshold,
        diagnostics=diagnostics,
        replay_buffer=replay_buffer,
        min_experiences=config.training.min_experiences_before_training,
        rendered=True
    )


def main() -> None:
    """Main training loop."""
    # Initialize configuration
    config = Config()

    # Start timer if timing is enabled
    start_time = time.time() if config.run.timing else None

    # Initialize pygame
    pg.init()

    # Initialize components
    trainer = TD3Trainer(config.training, config.environment)
    replay_buffer = ReplayBuffer(config.training.buffer_size)
    noise = OUActionNoise(config.noise, config.run.num_envs)
    diagnostics = DiagnosticsTracker()
    reporter = DiagnosticsReporter(diagnostics)

    logger.info(f"Starting TD3 training for {config.run.num_episodes} episodes")
    logger.info(f"Using {config.run.num_envs} parallel environments")

    # Training state
    completed_episodes = 0
    steps_since_training = 0
    training_started = False
    user_quit = False

    with create_environments(config.run, config.environment) as env_bundle:
        # Initialize vectorized environment states
        observations, _ = env_bundle.vec_env.reset()
        states = T.from_numpy(observations).float()

        # Episode manager for tracking per-environment state
        episode_manager = EpisodeManager(config.run.num_envs)

        while completed_episodes < config.run.num_episodes and not user_quit:
            # Check if current episode should be rendered
            if completed_episodes in env_bundle.render_episodes:
                result = run_rendered_episode(
                    env_bundle.render_env,
                    trainer,
                    noise,
                    replay_buffer,
                    completed_episodes,
                    config,
                    diagnostics
                )

                if result is None:
                    user_quit = True
                    break

                completed_episodes += 1
                continue

            # Calculate noise scale
            noise_scale = compute_noise_scale(
                completed_episodes,
                config.noise.noise_scale_initial,
                config.noise.noise_scale_final,
                config.noise.noise_decay_episodes
            )

            # Generate actions for all environments
            if completed_episodes < config.run.random_warmup_episodes:
                actions = T.from_numpy(env_bundle.vec_env.action_space.sample()).float()
            else:
                with T.no_grad():
                    exploration_noise = noise.generate() * noise_scale
                    actions = trainer.actor(states) + exploration_noise
                    actions[:, 0] = T.clamp(actions[:, 0], -1.0, 1.0)
                    actions[:, 1] = T.clamp(actions[:, 1], -1.0, 1.0)

            # Step all environments
            next_observations, rewards, terminateds, truncateds, infos = env_bundle.vec_env.step(
                actions.detach().numpy()
            )
            next_states = T.from_numpy(next_observations).float()

            # Process each environment
            for i in range(config.run.num_envs):
                # Compute shaped reward
                shaped_reward = shape_reward(observations[i], rewards[i], terminateds[i])

                # Record step
                episode_manager.add_step(
                    i,
                    float(rewards[i]),
                    shaped_reward,
                    actions[i].detach().cpu().numpy(),
                    observations[i].copy()
                )

                # Store experience
                experience = Experience(
                    state=states[i].detach().clone(),
                    action=actions[i].detach().clone(),
                    reward=T.tensor(shaped_reward, dtype=T.float32),
                    next_state=next_states[i].detach().clone(),
                    done=T.tensor(terminateds[i])
                )
                replay_buffer.push(experience)

                # Check if episode completed
                if terminateds[i] or truncateds[i]:
                    total_reward, env_reward, shaped_bonus, actions_array, observations_array = \
                        episode_manager.get_episode_stats(i)

                    finalize_episode(
                        episode_num=completed_episodes,
                        total_reward=total_reward,
                        env_reward=env_reward,
                        shaped_bonus=shaped_bonus,
                        steps=len(episode_manager.rewards[i]),
                        actions_array=actions_array,
                        observations_array=observations_array,
                        terminated=terminateds[i],
                        truncated=truncateds[i],
                        success_threshold=config.environment.success_threshold,
                        diagnostics=diagnostics,
                        replay_buffer=replay_buffer,
                        min_experiences=config.training.min_experiences_before_training
                    )

                    episode_manager.reset_env(i)
                    noise.reset(i)
                    completed_episodes += 1

                    if completed_episodes >= config.run.num_episodes:
                        break

            # Update states
            states = next_states
            observations = next_observations
            steps_since_training += config.run.num_envs

            # Training updates
            if replay_buffer.is_ready(config.training.min_experiences_before_training):
                if not training_started:
                    logger.info(
                        f">>> TRAINING STARTED at episode {completed_episodes} "
                        f"with {len(replay_buffer)} experiences <<<"
                    )
                    training_started = True

                # Train proportionally to steps taken
                updates_to_do = max(1, steps_since_training // 4)
                metrics = trainer.train_on_buffer(replay_buffer, updates_to_do)
                steps_since_training = 0

                # Log training metrics periodically
                if completed_episodes % 10 == 0:
                    diagnostics.record_training_metrics(metrics)
                    reporter.log_training_update(metrics, noise_scale)

    # Print diagnostics summary
    reporter.print_summary()

    # Print elapsed time if timing is enabled
    if start_time is not None:
        elapsed_time = time.time() - start_time
        print(f"\nTotal simulation time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
