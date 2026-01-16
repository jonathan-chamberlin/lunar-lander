"""Main entry point for Lunar Lander TD3 training.

This module orchestrates the training loop using components from:
- config.py: Configuration dataclasses
- training/: Neural networks, trainer, replay buffer, environment
- analysis/: Diagnostics, behavior analysis, charts
"""


import gc
import io
import logging
import os
import sys
import time
import warnings

# Suppress warnings BEFORE importing libraries that generate them
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress pygame "Hello from the pygame community" message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Force unbuffered stdout for real-time output display with UTF-8 encoding
# (Windows console defaults to cp1252 which can't display Unicode symbols)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import traceback
from datetime import datetime
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import torch as T

from config import Config, TrainingConfig, NoiseConfig, RunConfig, EnvironmentConfig, DisplayConfig
from data_types import Experience, EpisodeResult
from data.abstractions import SimulationIdentifier
from data.simulation_io import SimulationDirectory
from data.config_serializer import create_config_snapshot
from data.run_logger import RunLogger, RunRecord
from data.aggregate_writer import AggregateWriter
from training.environment import (
    create_environments,
    shape_reward,
    compute_noise_scale,
    EpisodeManager
)
from training.noise import OUActionNoise
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from training.trainer import TD3Trainer
from analysis.diagnostics import DiagnosticsTracker, DiagnosticsReporter
from analysis.behavior_analysis import BehaviorAnalyzer
from analysis.charts import ChartGenerator
from constants import SAFE_LANDING_OUTCOMES, OUTCOME_TO_CATEGORY
from analysis.output_formatter import format_behavior_output
from models import EpisodeData, TimingState, TrainingContext, PyGameContext
from output_mode import OutputController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Module-level behavior analyzer instance
behavior_analyzer = BehaviorAnalyzer()


def run_periodic_diagnostics(
    reporter: DiagnosticsReporter,
    diagnostics: DiagnosticsTracker,
    charts_folder: str,
    text_folder: str,
    completed_episodes: int,
    output: OutputController
) -> None:
    """Run periodic diagnostics and chart generation.

    Args:
        reporter: DiagnosticsReporter for printing summary
        diagnostics: DiagnosticsTracker with collected data
        charts_folder: Path to charts subfolder for this run
        text_folder: Path to text subfolder for this run
        completed_episodes: Number of episodes completed
        output: OutputController for mode-aware printing
    """
    # Print diagnostics summary based on mode
    if output.is_verbose():
        # HUMAN mode: full verbose diagnostics
        reporter.print_summary()
    elif output.is_agent_mode():
        # AGENT mode: detailed structured summary with deltas
        output.print_periodic_summary(completed_episodes, tracker=diagnostics)
    # SILENT mode: no output

    # Save diagnostics text to file (always, regardless of mode)
    text_path = os.path.join(text_folder, f"diagnostics_episode_{completed_episodes}.txt")
    try:
        summary_text = reporter.get_summary_text()
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
    except Exception as e:
        logger.warning(f"Failed to save diagnostics text: {e}")

    # Generate chart to file (always, regardless of mode)
    chart_path = os.path.join(charts_folder, f"chart_episode_{completed_episodes}.png")
    chart_gen = ChartGenerator(diagnostics, batch_size=50)
    if chart_gen.generate_to_file(chart_path):
        # Only auto-open chart viewer in HUMAN mode
        if output.is_verbose():
            os.startfile(chart_path)


def finalize_episode(
    data: EpisodeData,
    context: TrainingContext,
    timing: TimingState,
    output: OutputController
) -> EpisodeResult:
    """Create episode result, record it, print status, and track action stats.

    Args:
        data: Episode data including rewards, actions, observations
        context: Training context with diagnostics, buffer, thresholds
        timing: Timing state for speed tracking
        output: OutputController for mode-aware printing

    Returns:
        EpisodeResult for compatibility
    """
    success = data.env_reward >= context.success_threshold

    # Record episode with incremental statistics (primitives only)
    context.diagnostics.record_episode(
        episode_num=data.episode_num,
        env_reward=data.env_reward,
        shaped_bonus=data.shaped_bonus,
        duration_seconds=data.duration_seconds,
        success=success
    )

    # Analyze behaviors first so we can include outcome in the main line
    behavior_report = None
    outcome = 'UNKNOWN'
    outcome_category = 'crashed'  # Default category
    if len(data.observations_array) > 0 and len(data.actions_array) > 0:
        behavior_report = behavior_analyzer.analyze(
            data.observations_array, data.actions_array, data.terminated, data.truncated
        )
        outcome = behavior_report.outcome
        outcome_category = OUTCOME_TO_CATEGORY.get(outcome, 'crashed')
        # Record behavior with incremental statistics (primitives only)
        context.diagnostics.record_behavior(
            outcome=outcome,
            behaviors=behavior_report.behaviors,
            env_reward=data.env_reward,
            success=success
        )

    # Print episode status via OutputController (handles all mode logic)
    output.print_episode(
        episode_num=data.episode_num,
        success=success,
        outcome=outcome,
        outcome_category=outcome_category,
        env_reward=data.env_reward,
        shaped_bonus=data.shaped_bonus,
        total_reward=data.total_reward,
        rendered=data.rendered,
        behavior_report=behavior_report
    )

    # Record batch speed metrics every 50 episodes for diagnostics
    if timing.start_time is not None and data.episode_num > 0 and data.episode_num % 50 == 0:
        elapsed = time.time() - timing.start_time
        batch_num = data.episode_num // 50
        context.diagnostics.record_batch_speed(batch_num, elapsed, timing.total_steps, timing.total_training_updates)

    # Log run to JSONL if run_logger is available
    if context.run_logger is not None:
        behaviors_list = behavior_report.behaviors if behavior_report else []
        run_record = RunRecord.create(
            run_number=data.episode_num,
            env_reward=data.env_reward,
            shaped_bonus=data.shaped_bonus,
            steps=data.steps,
            duration_seconds=data.duration_seconds,
            success=success,
            outcome=outcome,
            behaviors=behaviors_list,
            terminated=data.terminated,
            truncated=data.truncated,
            rendered=data.rendered,
        )
        context.run_logger.log_run(run_record)

    # Return result for compatibility (but it's no longer stored in diagnostics)
    result = EpisodeResult(
        episode_num=data.episode_num,
        total_reward=data.total_reward,
        env_reward=data.env_reward,
        shaped_bonus=data.shaped_bonus,
        steps=data.steps,
        success=success,
        duration_seconds=data.duration_seconds
    )
    return result

def run_rendered_episode(
    render_env,
    trainer: TD3Trainer,
    noise: OUActionNoise,
    episode_num: int,
    config: Config,
    training_context: TrainingContext,
    timing_state: TimingState,
    pygame_ctx: PyGameContext,
    output: OutputController
) -> tuple[Optional[EpisodeResult], int]:
    """Run a single rendered episode.

    Args:
        render_env: Gymnasium environment with rgb_array rendering
        trainer: TD3 trainer instance
        noise: Noise generator for exploration
        episode_num: Current episode number
        config: Full configuration
        training_context: Training context with diagnostics, buffer, thresholds
        timing_state: Timing state for speed tracking
        pygame_ctx: Pygame context with font, screen, clock
        output: OutputController for mode-aware printing

    Returns:
        Tuple of (EpisodeResult or None if user quit, steps taken this episode)
    """
    obs, _ = render_env.reset()
    state = T.from_numpy(obs).float()
    episode_start_time = time.time()

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

    # Pre-allocate surface for frame rendering (avoids allocation every frame)
    frame_surface = pg.Surface((600, 400))

    # Pre-allocate buffer for transposed frame data (600 width x 400 height x 3 RGB)
    frame_buffer = np.empty((600, 400, 3), dtype=np.uint8)

    # Pre-render text if overlay is enabled (same text every frame, no need to re-render)
    if config.display.show_run_overlay:
        text_surface = pygame_ctx.font.render(f"Run: {episode_num}", True, config.display.font_color)
        text_pos = (config.display.text_x, config.display.text_y)

    while running:
        # Handle pygame events
        for event in pg.event.get():
            if event.type == pg.QUIT:
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

        # Convert action once for both recording and stepping
        action_np = action.detach().cpu().numpy()
        actions.append(action_np)
        observations_list.append(obs.copy())

        # Step environment
        next_obs, reward, terminated, truncated, info = render_env.step(action_np)
        next_state = T.from_numpy(next_obs).float()

        # Render frame with overlay (no flicker since we control all rendering)
        frame = render_env.render()  # Returns rgb_array (height, width, 3)
        # Transpose into pre-allocated buffer and blit (pygame needs width-first format)
        frame_buffer[:] = frame.transpose(1, 0, 2)
        pg.surfarray.blit_array(frame_surface, frame_buffer)
        pygame_ctx.screen.blit(frame_surface, (0, 0))

        # Draw text overlay if enabled (pre-rendered)
        if config.display.show_run_overlay:
            pygame_ctx.screen.blit(text_surface, text_pos)

        # Update display and control frame rate
        pg.display.flip()
        if config.run.framerate is not None:
            pygame_ctx.clock.tick(config.run.framerate)

        # Apply reward shaping (pass step count for progressive penalty)
        current_step = len(rewards)  # 0-indexed step count
        shaped_reward = shape_reward(obs, reward, terminated, step=current_step)
        shaped_bonus += (shaped_reward - reward)

        # Store experience
        experience = Experience(
            state=state.detach().clone(),
            action=action.detach().clone(),
            reward=T.tensor(shaped_reward, dtype=T.float32),
            next_state=next_state.detach().clone(),
            done=T.tensor(terminated)
        )
        training_context.replay_buffer.push(experience)

        state = next_state
        obs = next_obs
        rewards.append(float(reward))

        if terminated or truncated:
            episode_terminated = terminated
            episode_truncated = truncated
            running = False

    episode_steps = len(rewards)

    if user_quit:
        return None, episode_steps

    # Compute and finalize episode
    env_reward = float(np.sum(rewards))
    total_reward = env_reward + shaped_bonus
    duration_seconds = time.time() - episode_start_time

    # Update timing state with steps from this episode
    timing_state.total_steps += episode_steps

    # Create episode data for finalize_episode
    episode_data = EpisodeData(
        episode_num=episode_num,
        total_reward=total_reward,
        env_reward=env_reward,
        shaped_bonus=shaped_bonus,
        steps=episode_steps,
        duration_seconds=duration_seconds,
        actions_array=np.array(actions),
        observations_array=np.array(observations_list),
        terminated=episode_terminated,
        truncated=episode_truncated,
        rendered=True
    )
    result = finalize_episode(episode_data, training_context, timing_state, output)
    return result, episode_steps


def main() -> None:
    """Main training loop with robust error handling.

    Ensures diagnostics are printed even if an error occurs during training,
    as long as at least one episode was completed.
    """
    # Initialize configuration
    config = Config()

    # Start timer if timing is enabled
    start_time = time.time() if config.run.timing else None

    # Initialize pygame
    pg.init()
    pg.font.init()
    font = pg.font.Font(None, config.display.font_size)

    # Create pygame display for rendering (LunarLander is 600x400)
    screen = pg.display.set_mode((600, 400))
    pg.display.set_caption("Lunar Lander Training")
    clock = pg.time.Clock()

    # Create pygame context for rendered episodes
    pygame_ctx = PyGameContext(font=font, screen=screen, clock=clock)

    # Initialize components
    trainer = TD3Trainer(config.training, config.environment, config.run)

    # Create replay buffer (prioritized or uniform based on config)
    if config.training.use_per:
        replay_buffer = PrioritizedReplayBuffer(
            capacity=config.training.buffer_size,
            alpha=config.training.per_alpha,
            beta_start=config.training.per_beta_start,
            beta_end=config.training.per_beta_end,
            epsilon=config.training.per_epsilon
        )
        logger.info("Using Prioritized Experience Replay")
    else:
        replay_buffer = ReplayBuffer(config.training.buffer_size)
        logger.info("Using uniform Experience Replay")

    noise = OUActionNoise(config.noise, config.run.num_envs)
    diagnostics = DiagnosticsTracker()
    reporter = DiagnosticsReporter(diagnostics)

    # Create output controller for mode-aware printing
    output = OutputController.from_string(config.run.print_mode)

    logger.info(f"Starting TD3 training for {config.run.num_episodes} episodes")
    logger.info(f"Using {config.run.num_envs} parallel environments")
    if not config.run.training_enabled:
        logger.warning("TRAINING DISABLED - running simulation only (for chart testing)")

    # Training state
    completed_episodes = 0
    steps_since_training = 0
    total_steps = 0  # Total environment steps for SPS calculation
    training_started = False
    user_quit = False
    error_occurred = None

    # Create shared dataclasses for finalize_episode calls
    training_context = TrainingContext(
        diagnostics=diagnostics,
        replay_buffer=replay_buffer,
        success_threshold=config.environment.success_threshold,
        min_experiences=config.training.min_experiences_before_training
    )
    timing_state = TimingState(
        start_time=start_time,
        total_steps=0,
        total_training_updates=0
    )

    # Create simulation directory using new data architecture
    # Use script directory as base to ensure simulations are always in lunar-lander/simulations/
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # lunar-lander/
    sim_id = SimulationIdentifier.create()
    sim_dir = SimulationDirectory(script_dir, sim_id)
    sim_dir.initialize()

    # Write immutable config snapshot
    config_snapshot = create_config_snapshot(config, sim_id)
    sim_dir.write_config(config_snapshot)

    # Create run logger for per-episode JSONL logging
    run_logger = RunLogger(sim_dir.runs_path)

    # Create aggregate writer for periodic snapshots
    aggregate_writer = AggregateWriter(sim_dir.aggregates_path, write_interval=100)

    # Add run_logger to training context
    training_context.run_logger = run_logger

    # Set paths for charts and text (using new directory structure)
    charts_folder = str(sim_dir.charts_path)
    text_folder = str(sim_dir.text_path)
    run_folder = str(sim_dir.root_path)
    logger.info(f"Created simulation directory: {run_folder}")

    # Maintenance interval (every N episodes, perform cleanup)
    MAINTENANCE_INTERVAL = 100

    try:
        with create_environments(config.run, config.environment) as env_bundle:
            # Initialize vectorized environment states
            observations, _ = env_bundle.vec_env.reset()
            states = T.from_numpy(observations).float()

            # Episode manager for tracking per-environment state
            episode_manager = EpisodeManager(config.run.num_envs)

            # Stop signal file for external termination
            stop_file = os.path.join(script_dir, ".stop_simulation")

            while completed_episodes < config.run.num_episodes and not user_quit:
                # Check for external stop signal
                if os.path.exists(stop_file):
                    logger.info("Stop signal received, shutting down gracefully...")
                    os.remove(stop_file)  # Acknowledge receipt
                    user_quit = True
                    break

                # Periodic maintenance to prevent memory issues
                if completed_episodes > 0 and completed_episodes % MAINTENANCE_INTERVAL == 0:
                    # Pump pygame events to prevent "Out of memory" errors
                    pg.event.pump()
                    # Force garbage collection and log stats
                    gc_counts_before = gc.get_count()
                    gc.collect()
                    gc_counts_after = gc.get_count()
                    logger.debug(f"GC at episode {completed_episodes}: before={gc_counts_before}, after={gc_counts_after}")
                    # Clear PyTorch cache if using CUDA
                    if T.cuda.is_available():
                        T.cuda.empty_cache()

                # Check if current episode should be rendered
                if completed_episodes in env_bundle.render_episodes:
                    result, episode_steps = run_rendered_episode(
                        env_bundle.render_env,
                        trainer,
                        noise,
                        completed_episodes,
                        config,
                        training_context,
                        timing_state,
                        pygame_ctx,
                        output
                    )
                    # Sync local total_steps with timing_state (updated inside run_rendered_episode)
                    total_steps = timing_state.total_steps

                    if result is None:
                        user_quit = True
                        break

                    # Training after rendered episode (same as non-rendered path)
                    if config.run.training_enabled and replay_buffer.is_ready(config.training.min_experiences_before_training):
                        if not training_started:
                            logger.info(
                                f">>> TRAINING STARTED at episode {completed_episodes} "
                                f"with {len(replay_buffer)} experiences <<<"
                            )
                            training_started = True

                        # Anneal PER beta towards 1.0
                        if config.training.use_per:
                            progress = completed_episodes / config.run.num_episodes
                            replay_buffer.anneal_beta(progress)

                        # Train based on episode length (rendered episodes are ~200-1000 steps)
                        updates_to_do = max(1, result.steps // 4)
                        metrics = trainer.train_on_buffer(replay_buffer, updates_to_do)

                        # Step learning rate schedulers
                        trainer.step_schedulers()

                        # Log training metrics periodically
                        if completed_episodes % 10 == 0:
                            noise_scale = compute_noise_scale(
                                completed_episodes,
                                config.noise.noise_scale_initial,
                                config.noise.noise_scale_final,
                                config.noise.noise_decay_episodes
                            )
                            diagnostics.record_training_metrics(metrics)
                            reporter.log_training_update(metrics, noise_scale)

                    completed_episodes += 1

                    # Periodic diagnostics and chart generation every 100 episodes
                    if completed_episodes % 100 == 0:
                        run_periodic_diagnostics(
                            reporter, diagnostics, charts_folder, text_folder, completed_episodes, output
                        )
                        # Write periodic aggregate snapshot
                        aggregate_writer.maybe_write(completed_episodes, diagnostics)

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

                # Step all environments - batch convert actions once
                actions_np = actions.detach().cpu().numpy()
                next_observations, rewards, terminateds, truncateds, infos = env_bundle.vec_env.step(actions_np)
                next_states = T.from_numpy(next_observations).float()

                # Detach states once for all environments (avoids per-env detach)
                states_detached = states.detach()
                actions_detached = actions.detach()
                next_states_detached = next_states.detach()

                # Process each environment
                for i in range(config.run.num_envs):
                    # Compute shaped reward (pass step count for progressive penalty)
                    current_step = episode_manager.step_counts[i]
                    shaped_reward = shape_reward(observations[i], rewards[i], terminateds[i], step=current_step)

                    # Record step (use pre-converted numpy array)
                    episode_manager.add_step(
                        i,
                        float(rewards[i]),
                        shaped_reward,
                        actions_np[i],
                        observations[i].copy()
                    )

                    # Store experience (use pre-detached tensors)
                    experience = Experience(
                        state=states_detached[i].clone(),
                        action=actions_detached[i].clone(),
                        reward=T.tensor(shaped_reward, dtype=T.float32),
                        next_state=next_states_detached[i].clone(),
                        done=T.tensor(terminateds[i])
                    )
                    replay_buffer.push(experience)

                    # Check if episode completed
                    if terminateds[i] or truncateds[i]:
                        total_reward, env_reward, shaped_bonus, actions_array, observations_array, duration_seconds = \
                            episode_manager.get_episode_stats(i)

                        # Create dataclasses for finalize_episode
                        episode_data = EpisodeData(
                            episode_num=completed_episodes,
                            total_reward=total_reward,
                            env_reward=env_reward,
                            shaped_bonus=shaped_bonus,
                            steps=episode_manager.step_counts[i],
                            duration_seconds=duration_seconds,
                            actions_array=actions_array,
                            observations_array=observations_array,
                            terminated=terminateds[i],
                            truncated=truncateds[i],
                            rendered=False
                        )
                        finalize_episode(episode_data, training_context, timing_state, output)

                        episode_manager.reset_env(i)
                        noise.reset(i)
                        completed_episodes += 1

                        # Periodic diagnostics and chart generation every 100 episodes
                        if completed_episodes % 100 == 0:
                            run_periodic_diagnostics(
                                reporter, diagnostics, charts_folder, text_folder, completed_episodes, output
                            )
                            # Write periodic aggregate snapshot
                            aggregate_writer.maybe_write(completed_episodes, diagnostics)

                        if completed_episodes >= config.run.num_episodes:
                            break

                # Update states
                states = next_states
                observations = next_observations
                steps_since_training += config.run.num_envs
                total_steps += config.run.num_envs  # Track total steps for SPS
                timing_state.total_steps = total_steps
                timing_state.total_training_updates = trainer.training_steps

                # Training updates
                if config.run.training_enabled and replay_buffer.is_ready(config.training.min_experiences_before_training):
                    if not training_started:
                        logger.info(
                            f">>> TRAINING STARTED at episode {completed_episodes} "
                            f"with {len(replay_buffer)} experiences <<<"
                        )
                        training_started = True

                    # Anneal PER beta towards 1.0
                    if config.training.use_per:
                        progress = completed_episodes / config.run.num_episodes
                        replay_buffer.anneal_beta(progress)

                    # Train proportionally to steps taken
                    updates_to_do = max(1, steps_since_training // 4)
                    metrics = trainer.train_on_buffer(replay_buffer, updates_to_do)
                    steps_since_training = 0

                    # Step learning rate schedulers (decay happens gradually over episodes)
                    trainer.step_schedulers()

                    # Log training metrics periodically
                    if completed_episodes % 10 == 0:
                        diagnostics.record_training_metrics(metrics)
                        reporter.log_training_update(metrics, noise_scale)

    except KeyboardInterrupt:
        error_occurred = "KeyboardInterrupt"
        logger.warning("\n\nTraining interrupted by user (Ctrl+C)")

    except pg.error as e:
        error_occurred = f"pygame.error: {e}"
        logger.error(f"\n\nPygame error occurred: {e}")
        logger.error("This often happens due to memory issues in long runs.")

    except MemoryError as e:
        error_occurred = f"MemoryError: {e}"
        logger.error(f"\n\nMemory error occurred: {e}")
        logger.error("System ran out of memory.")

    except T.cuda.OutOfMemoryError as e:
        error_occurred = f"CUDA OutOfMemoryError: {e}"
        logger.error(f"\n\nCUDA out of memory: {e}")

    except RuntimeError as e:
        error_occurred = f"RuntimeError: {e}"
        logger.error(f"\n\nRuntime error occurred: {e}")
        traceback.print_exc()

    except Exception as e:
        error_occurred = f"{type(e).__name__}: {e}"
        logger.error(f"\n\nUnexpected error occurred: {e}")
        traceback.print_exc()

    finally:
        # Always print diagnostics if we completed at least 1 episode
        if completed_episodes > 0:
            # Get elapsed time for final summary
            elapsed_time = time.time() - start_time if start_time is not None else None

            # Print final summary via OutputController (mode-aware)
            output.print_final_summary(
                completed_episodes=completed_episodes,
                error_occurred=error_occurred,
                elapsed_time=elapsed_time,
                total_steps=total_steps,
                training_steps=trainer.training_steps,
                tracker=diagnostics
            )

            # Print full diagnostics summary for HUMAN mode
            try:
                if output.is_verbose():
                    reporter.print_summary()
                # AGENT mode: detailed summary already printed in print_final_summary
            except Exception as e:
                logger.error(f"Failed to print full diagnostics: {e}")
                # Try minimal diagnostics (only in non-silent mode)
                if not output.is_silent():
                    try:
                        print("\n--- MINIMAL DIAGNOSTICS (full report failed) ---")
                        print(f"Total episodes: {diagnostics.total_episodes}")
                        print(f"Successes: {diagnostics.success_count}")
                        if diagnostics.env_rewards:
                            rewards = diagnostics.get_rewards()
                            print(f"Mean reward: {np.mean(rewards):.2f}")
                            print(f"Max reward: {np.max(rewards):.2f}")
                    except Exception:
                        print("Could not print even minimal diagnostics")

            # Write final aggregate and save model
            try:
                aggregate_writer.write_final(diagnostics)
                logger.info(f"Final aggregate saved")
            except Exception as e:
                logger.error(f"Failed to write final aggregate: {e}")

            try:
                model_path = str(sim_dir.models_path / "final_model")
                trainer.save(model_path)
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

            # Generate final training visualization charts
            try:
                # Save final chart to folder if one was created
                if charts_folder is not None:
                    chart_path = os.path.join(charts_folder, f"chart_final_{completed_episodes}.png")
                    chart_gen = ChartGenerator(diagnostics, batch_size=50)
                    chart_gen.generate_to_file(chart_path)
                    logger.info(f"Final chart saved to {chart_path}")

                # Generate and show final chart (blocking) - only in HUMAN mode
                if output.is_verbose():
                    chart_generator = ChartGenerator(diagnostics, batch_size=50)
                    chart_generator.generate_all(show=True, block=True)
            except Exception as e:
                logger.error(f"Failed to generate charts: {e}")
        else:
            if not output.is_silent():
                print("\nNo episodes completed - no diagnostics to show")

        # Cleanup pygame
        try:
            pg.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
