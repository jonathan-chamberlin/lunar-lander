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
import sys
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


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
    shaped_bonus = 0.0
    running = True
    user_quit = False

    logger.info(f"Episode {episode_num} (RENDERED)")

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
            running = False

    if user_quit:
        return None

    # Compute episode results
    env_reward = float(np.sum(rewards))
    total_reward = env_reward + shaped_bonus
    success = total_reward >= config.environment.success_threshold

    result = EpisodeResult(
        episode_num=episode_num,
        total_reward=total_reward,
        env_reward=env_reward,
        shaped_bonus=shaped_bonus,
        steps=len(rewards),
        success=success
    )

    # Record action statistics
    if (len(actions) > 0 and
            replay_buffer.is_ready(config.training.min_experiences_before_training)):
        actions_array = np.array(actions)
        diagnostics.record_action_stats(ActionStatistics.from_actions(actions_array))

    return result


def main() -> None:
    """Main training loop."""
    # Initialize configuration
    config = Config.default()

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

                diagnostics.record_episode(result)
                reporter.log_episode(result, compute_noise_scale(
                    completed_episodes,
                    config.noise.noise_scale_initial,
                    config.noise.noise_scale_final,
                    config.noise.noise_decay_episodes
                ))

                status = "SUCCESS" if result.success else "FAILURE"
                print(f"Run {completed_episodes}")
                print(status)
                print(f"total_reward: {result.total_reward:.1f} "
                      f"(env: {result.env_reward:.1f}, shaped: {result.shaped_bonus:.1f})")

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
    for i in range(num_envs):
        # Track actions
        env_actions[i].append(actions[i].detach().cpu().numpy())

        # Compute shaped reward
        shaped_reward = shape_reward(observations[i], rewards[i], terminateds[i])
        env_shaped_bonus[i] += (shaped_reward - rewards[i])

        # Store experience
        experience = (states[i].detach().clone(), actions[i].detach().clone(),
                     T.tensor(shaped_reward, dtype=T.float32), next_states[i].detach().clone(),
                     T.tensor(terminateds[i]))
        add_experience(experience)

        # Track original reward
        env_rewards[i].append(float(rewards[i]))

        # Check if episode completed (terminated OR truncated)
        if terminateds[i] or truncateds[i]:
            env_reward = float(np.sum(env_rewards[i]))
            shaped_bonus = env_shaped_bonus[i]
            total_reward = env_reward + shaped_bonus

            print(f"Run {completed_episodes}")
            if total_reward >= 200:
                successes_list.append(completed_episodes)
                print("SUCCESS")
            else:
                print("FAILURE")

            # Track action statistics
            if len(env_actions[i]) > 0 and len(experiences) >= min_experiences_before_training:
                actions_array = np.array(env_actions[i])
                episode_action_means.append(np.mean(np.abs(actions_array)))
                episode_action_stds.append(np.std(actions_array))
                episode_main_thruster.append(np.mean(actions_array[:, 0]))
                episode_side_thruster.append(np.mean(actions_array[:, 1]))

            print(f"total reward: {total_reward:.1f} (env: {env_reward:.1f}, shaped: {shaped_bonus:.1f})")
            total_reward_for_alls_runs.append(total_reward)

            # Reset tracking for this env
            env_rewards[i] = []
            env_shaped_bonus[i] = 0.0
            env_actions[i] = []
            lunar_noise.reset(i)

            completed_episodes += 1

            # Stop if we've reached target runs
            if completed_episodes >= runs:
                break

    # Update states
    states = next_states
    observations = next_observations
    steps_since_training += num_envs

    # Training: run updates based on step count
    if len(experiences) >= min_experiences_before_training:
        if not training_started:
            print(f">>> TRAINING STARTED at episode {completed_episodes} with {len(experiences)} experiences <<<")
            training_started = True

        # Train proportionally to steps taken (roughly training_updates_per_episode per ~200 steps)
        updates_to_do = max(1, steps_since_training // 4)  # ~50 updates per 200 steps

        total_critic_loss = 0
        total_actor_loss = 0
        total_critic_grad = 0
        total_actor_grad = 0
        total_q = 0
        actor_update_count = 0

        for _ in range(updates_to_do):
            c_loss, a_loss, c_grad, a_grad, avg_q = do_training_step()
            total_critic_loss += c_loss
            total_actor_loss += a_loss
            total_critic_grad += c_grad
            total_actor_grad += a_grad
            total_q += avg_q
            if a_loss != 0:
                actor_update_count += 1

        steps_since_training = 0

        # Log training metrics periodically
        if completed_episodes % 10 == 0 and updates_to_do > 0:
            avg_c_loss = total_critic_loss / updates_to_do
            avg_a_loss = total_actor_loss / max(actor_update_count, 1)
            avg_q_val = total_q / updates_to_do

            episode_q_values.append(avg_q_val)
            episode_actor_losses.append(avg_a_loss)
            episode_critic_losses.append(avg_c_loss)
            episode_actor_grad_norms.append(total_actor_grad / max(actor_update_count, 1))
            episode_critic_grad_norms.append(total_critic_grad / updates_to_do)

            print(f"Training update - Critic Loss: {avg_c_loss:.4f}, Actor Loss: {avg_a_loss:.4f}, Avg Q: {avg_q_val:.3f}, Noise: {noise_scale:.3f}")

# Close environments
vec_env.close()
render_env.close()


# ===== COMPREHENSIVE DIAGNOSTIC OUTPUT =====
print("\n--- DIAGNOSTIC CODE REACHED - PROCESSING RESULTS ---")
import sys
sys.stdout.flush()

print("\n" + "="*80)
print("TRAINING DIAGNOSTICS SUMMARY")
print("="*80)

# Reward statistics
print(f"\n--- REWARD STATISTICS ---")
print(f"Total episodes: {len(total_reward_for_alls_runs)}")
print(f"Successes: {len(successes_list)} (episodes: {successes_list})")

print(f"Success rate: {len(successes_list)/len(total_reward_for_alls_runs)*100:.1f}%")
print(f"Mean reward: {np.mean(total_reward_for_alls_runs):.2f}")
print(f"Max reward: {np.max(total_reward_for_alls_runs):.2f}")
print(f"Min reward: {np.min(total_reward_for_alls_runs):.2f}")
if len(total_reward_for_alls_runs) >= 50:
    print(f"Final 50 episodes mean reward: {np.mean(total_reward_for_alls_runs[-50:]):.2f}")
# Action statistics


print(f"\n--- ACTION STATISTICS ---")
if len(episode_main_thruster) > 0:
    print(f"Episodes with training: {len(episode_main_thruster)}")
    print(f"Mean main thruster (all episodes): {np.mean(episode_main_thruster):.3f}")
    print(f"Mean side thruster (all episodes): {np.mean(episode_side_thruster):.3f}")
    print(f"Mean action magnitude: {np.mean(episode_action_means):.3f}")
    print(f"Mean action std: {np.mean(episode_action_stds):.3f}")

    print(f"\nLast 50 episodes:")
    print(f"  Main thruster: {np.mean(episode_main_thruster[-50:]):.3f}")
    print(f"  Side thruster: {np.mean(episode_side_thruster[-50:]):.3f}")
    print(f"  Action magnitude: {np.mean(episode_action_means[-50:]):.3f}")

    # Check for blasting upward pattern
    high_thruster_episodes = sum(1 for x in episode_main_thruster if x > 0.5)
    print(f"\nEpisodes with high main thruster (>0.5): {high_thruster_episodes}/{len(episode_main_thruster)}")
else:
    print("No action data collected (training hasn't started yet)")


# Q-value and loss statistics

print(f"\n--- TRAINING METRICS ---")
if len(episode_q_values) > 0:
    print(f"Mean Q-value: {np.mean(episode_q_values):.3f}")
    if len(episode_q_values) >= 10:
        print(f"Q-value trend (first 10 vs last 10): {np.mean(episode_q_values[:10]):.3f} -> {np.mean(episode_q_values[-10:]):.3f}")
    print(f"Mean actor loss: {np.mean(episode_actor_losses):.4f}")
    print(f"Mean critic loss: {np.mean(episode_critic_losses):.4f}")
    print(f"Mean actor gradient norm: {np.mean(episode_actor_grad_norms):.4f}")
    print(f"Mean critic gradient norm: {np.mean(episode_critic_grad_norms):.4f}")

    # Check for divergence patterns
    high_actor_loss_episodes = sum(1 for x in episode_actor_losses if x > 1.0)
    print(f"\nEpisodes with high actor loss (>1.0): {high_actor_loss_episodes}/{len(episode_actor_losses)}")
else:
    print("No training metrics collected yet")


# Sample recent episode details
print(f"\n--- LAST 10 EPISODES DETAIL ---")
length = len(total_reward_for_alls_runs)
if length > 5:
    start_idx = len(total_reward_for_alls_runs) - 5
else:
    start_idx = length-1

for i in range(start_idx, length-1):
    reward = total_reward_for_alls_runs[i]
    episode_num = i
    status = "SUCCESS" if i in successes_list else "FAILURE"

    info_str = f"Ep {episode_num}: {status}, Reward: {reward:.1f}"

    # Calculate the correct index in the tracking lists
    # (tracking starts later than episode 0 since training starts later)
    tracking_idx = i - (len(total_reward_for_alls_runs) - len(episode_main_thruster))
    if 0 <= tracking_idx < len(episode_main_thruster):
        main = episode_main_thruster[tracking_idx]
        side = episode_side_thruster[tracking_idx]
        q_val = episode_q_values[tracking_idx] if tracking_idx < len(episode_q_values) else 0
        info_str += f", Main: {main:.2f}, Side: {side:.2f}, Q: {q_val:.2f}"

    print(info_str)

# Key data section - printed once, outside the loop
print("\n" + "="*80)
print("KEY DATA FOR ANALYSIS")
print("="*80)
print(f"\nReward list (last 50): {total_reward_for_alls_runs[-50:]}")
print(f"\nMain thruster list (last 50): {episode_main_thruster[-50:] if len(episode_main_thruster) >= 50 else episode_main_thruster}")
print(f"\nQ-values list (last 50): {episode_q_values[-50:] if len(episode_q_values) >= 50 else episode_q_values}")
print(f"\nActor losses list (last 50): {episode_actor_losses[-50:] if len(episode_actor_losses) >= 50 else episode_actor_losses}")

print("\n" + "="*80)
print("END OF DIAGNOSTICS")
print("="*80)

# Print elapsed time if timing is enabled
if timing:
    elapsed_time = time.time() - start_time
    print(f"\nTotal simulation time: {elapsed_time:.2f} seconds")
