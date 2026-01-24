"""Core training runner extracted from main.py.

This module provides a reusable run_training() function that can be called
by both main.py (CLI) and sweep_runner.py (hyperparameter sweeps).
"""

import gc
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import psutil
import torch as T

from config import Config
from data_types import Experience
from training.environment import (
    create_environments,
    shape_reward,
    compute_noise_scale,
)
from training.noise import OUActionNoise
from training.video import EpisodeRecorder
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from training.trainer import TD3Trainer
from training.training_options import TrainingOptions
from training.training_result import TrainingResult
from training.memory_limiter import MemoryLimiter, get_recommended_limit_mb
from analysis.diagnostics import DiagnosticsTracker, DiagnosticsReporter
from analysis.behavior_analysis import BehaviorAnalyzer
from analysis.charts import ChartGenerator
from constants import OUTCOME_TO_CATEGORY
from output_mode import OutputController

if TYPE_CHECKING:
    from models import EpisodeData, TimingState, TrainingContext, PyGameContext
    from data.simulation_io import SimulationDirectory

logger = logging.getLogger(__name__)

# Memory monitoring constants
MEMORY_LOG_INTERVAL = 50  # Log memory usage every N episodes
MEMORY_CHECK_INTERVAL = 10  # Check memory threshold every N episodes (not every episode)
# Default memory limit - leaves headroom for other apps (browser, etc.)
# Can be overridden via TrainingOptions.memory_limit_mb
DEFAULT_MEMORY_LIMIT_MB = 4000

# Checkpoint constants
CHECKPOINT_INTERVAL = 1000  # Save model checkpoint every N episodes


def _save_model_checkpoint(
    trainer: TD3Trainer,
    episode_num: int,
    options: TrainingOptions,
    sim_dir: Optional["SimulationDirectory"] = None,
    is_final: bool = False
) -> Optional[str]:
    """Save a model checkpoint.

    Args:
        trainer: The TD3Trainer instance with model weights
        episode_num: Current episode number (used in checkpoint filename)
        options: Training options with save paths
        sim_dir: Simulation directory (for non-experiment runs)
        is_final: If True, save as 'final_model' instead of checkpoint

    Returns:
        Path to saved model, or None if save failed
    """
    try:
        # Determine save path
        if options.is_experiment and options.results_dir is not None:
            models_dir = options.results_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            if is_final:
                model_path = str(models_dir / f"{options.run_name or 'final'}_model")
            else:
                model_path = str(models_dir / f"checkpoint_ep{episode_num}")
        elif sim_dir is not None:
            sim_dir.models_path.mkdir(parents=True, exist_ok=True)
            if is_final:
                model_path = str(sim_dir.models_path / "final_model")
            else:
                model_path = str(sim_dir.models_path / f"checkpoint_ep{episode_num}")
        else:
            return None

        trainer.save(model_path)
        checkpoint_type = "Final model" if is_final else f"Checkpoint (ep {episode_num})"
        logger.info(f"{checkpoint_type} saved to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to save model checkpoint: {e}")
        return None


def get_memory_mb() -> float:
    """Get current process memory usage in megabytes."""
    return psutil.Process().memory_info().rss / 1024 / 1024


# Module-level behavior analyzer instance
behavior_analyzer = BehaviorAnalyzer()


def _run_periodic_diagnostics(
    reporter: DiagnosticsReporter,
    diagnostics: DiagnosticsTracker,
    charts_folder: Optional[str],
    text_folder: Optional[str],
    completed_episodes: int,
    output: OutputController,
    open_charts: bool = False,
    batch_size: int = 50
) -> None:
    """Run periodic diagnostics and chart generation.

    Args:
        reporter: DiagnosticsReporter for printing summary
        diagnostics: DiagnosticsTracker with collected data
        charts_folder: Path to charts subfolder (None to skip chart generation)
        text_folder: Path to text subfolder (None to skip text saving)
        completed_episodes: Number of episodes completed
        output: OutputController for mode-aware printing
        open_charts: Whether to auto-open generated charts (HUMAN mode only)
        batch_size: Batch size for chart generation
    """
    # Print diagnostics summary based on mode
    if output.is_verbose():
        reporter.print_summary()
    elif output.is_minimal_mode():
        output.print_periodic_summary(completed_episodes, tracker=diagnostics)

    # Save diagnostics text to file (if text_folder provided)
    if text_folder is not None:
        text_path = os.path.join(text_folder, f"diagnostics_episode_{completed_episodes}.txt")
        try:
            summary_text = reporter.get_summary_text()
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
        except Exception as e:
            logger.warning(f"Failed to save diagnostics text: {e}")

    # Generate chart to file (if charts_folder provided)
    if charts_folder is not None:
        chart_path = os.path.join(charts_folder, f"chart_episode_{completed_episodes}.png")
        chart_gen = ChartGenerator(diagnostics, batch_size=batch_size)
        if chart_gen.generate_to_file(chart_path):
            if open_charts and output.is_verbose():
                os.startfile(chart_path)


def _finalize_episode(
    data: "EpisodeData",
    context: "TrainingContext",
    timing: "TimingState",
    output: OutputController
) -> "EpisodeResult":
    """Create episode result, record it, print status, and track action stats.

    Args:
        data: Episode data including rewards, actions, observations
        context: Training context with diagnostics, buffer, thresholds
        timing: Timing state for speed tracking
        output: OutputController for mode-aware printing

    Returns:
        EpisodeResult for compatibility
    """
    from data_types import EpisodeResult

    success = data.env_reward >= context.success_threshold

    # Record episode with incremental statistics
    context.diagnostics.record_episode(
        episode_num=data.episode_num,
        env_reward=data.env_reward,
        shaped_bonus=data.shaped_bonus,
        duration_seconds=data.duration_seconds,
        success=success
    )

    # Analyze behaviors (only if arrays are available)
    behavior_report = None
    outcome = 'UNKNOWN'
    outcome_category = 'crashed'
    if data.observations_array is not None and data.actions_array is not None:
        if len(data.observations_array) > 0 and len(data.actions_array) > 0:
            behavior_report = behavior_analyzer.analyze(
                data.observations_array, data.actions_array, data.terminated, data.truncated
            )
            outcome = behavior_report.outcome
            outcome_category = OUTCOME_TO_CATEGORY.get(outcome, 'crashed')
            context.diagnostics.record_behavior(
                outcome=outcome,
                behaviors=behavior_report.behaviors,
                env_reward=data.env_reward,
                success=success
            )

    # Print episode status via OutputController
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

    # Record batch speed metrics every 50 episodes
    if timing.start_time is not None and data.episode_num > 0 and data.episode_num % 50 == 0:
        elapsed = time.time() - timing.start_time
        batch_num = data.episode_num // 50
        context.diagnostics.record_batch_speed(batch_num, elapsed, timing.total_steps, timing.total_training_updates)

    # Log run to JSONL if run_logger is available
    if context.run_logger is not None:
        from data.run_logger import RunRecord
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


def _run_episode(
    env,
    trainer: TD3Trainer,
    noise: OUActionNoise,
    episode_num: int,
    config: Config,
    training_context: "TrainingContext",
    timing_state: "TimingState",
    output: OutputController,
    should_render: bool = False,
    pygame_ctx: Optional["PyGameContext"] = None,
    record_episode: bool = False,
    frames_dir: Optional[Path] = None
) -> tuple[Optional["EpisodeResult"], int]:
    """Run a single episode with identical logic for rendered and unrendered modes.

    The ONLY difference between rendered and unrendered is:
    - Rendered: displays frames via pygame, handles quit events
    - Unrendered: no display, no pygame

    All training logic (action generation, experience storage, rewards) is identical.

    Args:
        env: Gymnasium environment
        trainer: TD3 trainer instance
        noise: Noise generator for exploration
        episode_num: Current episode number
        config: Full configuration
        training_context: Training context with diagnostics, buffer, thresholds
        timing_state: Timing state for speed tracking
        output: OutputController for mode-aware printing
        should_render: Whether to render frames via pygame
        pygame_ctx: Pygame context (required if should_render=True)
        record_episode: Whether to save frames to disk for video compilation
        frames_dir: Directory to save frames (required if record_episode=True)

    Returns:
        Tuple of (EpisodeResult or None if user quit, steps taken this episode)
    """
    from models import EpisodeData

    # Import pygame only if rendering
    pg = None
    frame_surface = None
    frame_buffer = None
    text_surface = None
    text_pos = None

    # Initialize frame recorder if recording this episode
    recorder = None
    if record_episode and frames_dir is not None:
        recorder = EpisodeRecorder(
            frames_dir=frames_dir,
            episode_num=episode_num,
            frame_format=config.video.frame_format
        )

    if should_render:
        import pygame as pg
        frame_surface = pg.Surface((600, 400))
        frame_buffer = np.empty((600, 400, 3), dtype=np.uint8)
        if config.display.show_run_overlay:
            text_surface = pygame_ctx.font.render(f"Run: {episode_num}", True, config.display.font_color)
            text_pos = (config.display.text_x, config.display.text_y)

    # Initialize episode
    obs, _ = env.reset()
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

    while running:
        # Handle pygame events (only if rendering)
        if should_render and pg is not None:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    logger.warning(f"pygame QUIT event received at episode {episode_num}")
                    running = False
                    user_quit = True
                    break
            if not running:
                break

        # Generate action (IDENTICAL for both modes)
        if episode_num < config.run.random_warmup_episodes:
            action = T.from_numpy(env.action_space.sample()).float()
        else:
            action_noise = noise.generate() * noise_scale
            action = (trainer.actor(state) + action_noise).float()
            action[0] = T.clamp(action[0], -1.0, 1.0)
            action[1] = T.clamp(action[1], -1.0, 1.0)

        action_np = action.detach().cpu().numpy()
        actions.append(action_np)
        observations_list.append(obs.copy())

        # Step environment (IDENTICAL for both modes)
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        next_state = T.from_numpy(next_obs).float()

        # Render frame for display (requires pygame)
        if should_render and pg is not None:
            frame = env.render()
            frame_buffer[:] = frame.transpose(1, 0, 2)
            pg.surfarray.blit_array(frame_surface, frame_buffer)
            pygame_ctx.screen.blit(frame_surface, (0, 0))
            if text_surface is not None:
                pygame_ctx.screen.blit(text_surface, text_pos)
            pg.display.flip()
            if config.run.framerate is not None:
                pygame_ctx.clock.tick(config.run.framerate)

            # Save frame to disk if recording (with display)
            if recorder is not None:
                recorder.save_frame(frame)

        # Render frame for recording only (no display) - headless video capture
        elif recorder is not None:
            frame = env.render()
            recorder.save_frame(frame)

        # Apply reward shaping (IDENTICAL for both modes)
        current_step = len(rewards)
        shaped_reward = shape_reward(obs, reward, terminated, config.reward_shaping, step=current_step)
        shaped_bonus += (shaped_reward - reward)

        # Store experience (IDENTICAL for both modes)
        experience = Experience(
            state=state.detach().clone(),
            action=action.detach().clone(),
            reward=T.tensor(shaped_reward, dtype=T.float32),
            next_state=next_state.detach().clone(),
            done=T.tensor(terminated)
        )
        training_context.replay_buffer.push(experience)

        # Update state (IDENTICAL for both modes)
        state = next_state
        obs = next_obs
        rewards.append(float(reward))

        # Check termination (IDENTICAL for both modes)
        if terminated or truncated:
            episode_terminated = terminated
            episode_truncated = truncated
            running = False

    # Finalize frame recorder if active
    if recorder is not None:
        recorder.finalize()

    # Compute results (IDENTICAL for both modes)
    episode_steps = len(rewards)

    if user_quit:
        return None, episode_steps

    env_reward = float(np.sum(rewards))
    total_reward = env_reward + shaped_bonus
    duration_seconds = time.time() - episode_start_time

    timing_state.total_steps += episode_steps

    # Convert to arrays for behavior analysis (will be cleared after)
    actions_arr = np.array(actions)
    observations_arr = np.array(observations_list)

    # Clear local lists immediately to free memory
    del actions, observations_list, rewards

    episode_data = EpisodeData(
        episode_num=episode_num,
        total_reward=total_reward,
        env_reward=env_reward,
        shaped_bonus=shaped_bonus,
        steps=episode_steps,
        duration_seconds=duration_seconds,
        terminated=episode_terminated,
        truncated=episode_truncated,
        actions_array=actions_arr,
        observations_array=observations_arr,
        rendered=should_render
    )

    # Clear array references after creating EpisodeData
    del actions_arr, observations_arr

    result = _finalize_episode(episode_data, training_context, timing_state, output)

    # Clear trajectory arrays in EpisodeData to free memory
    episode_data.clear_arrays()

    return result, episode_steps


def run_training(config: Config, options: Optional[TrainingOptions] = None) -> TrainingResult:
    """Core training loop. Called by main.py CLI and sweep_runner.py.

    Args:
        config: Training configuration (hyperparameters, env settings, etc.)
        options: Execution options (output mode, directories, what to save, etc.)
                 If None, uses default TrainingOptions.

    Returns:
        TrainingResult with all training metrics and diagnostics.
    """
    from models import EpisodeData, TimingState, TrainingContext, PyGameContext

    options = options or TrainingOptions()

    start_time = time.time() if config.run.timing else None

    # Conditional pygame initialization
    pg = None
    pygame_ctx = None
    needs_pygame = options.require_pygame and config.run.render_mode != 'none'

    if needs_pygame:
        import pygame as pg
        pg.init()
        pg.font.init()
        font = pg.font.Font(None, config.display.font_size)
        screen = pg.display.set_mode((600, 400))
        pg.display.set_caption("Lunar Lander Training")
        clock = pg.time.Clock()
        pygame_ctx = PyGameContext(font=font, screen=screen, clock=clock)

    # Initialize trainer and replay buffer
    trainer = TD3Trainer(config.training, config.environment, config.run)

    # Load pre-trained model if specified
    if options.load_model_path is not None:
        model_path = str(options.load_model_path)
        logger.info(f"Loading pre-trained model from: {model_path}")
        trainer.load(model_path)
        logger.info("Model loaded successfully")

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

    noise = OUActionNoise(config.noise)
    diagnostics = DiagnosticsTracker(batch_size=options.diagnostics_batch_size)
    reporter = DiagnosticsReporter(diagnostics)

    output = OutputController.from_string(options.output_mode)

    # Initialize memory limiter to prevent system-wide memory exhaustion
    memory_limit = options.memory_limit_mb or get_recommended_limit_mb()
    memory_limiter = MemoryLimiter(limit_mb=memory_limit)
    memory_limiter.start()

    logger.info(f"Starting TD3 training for {config.run.num_episodes} episodes")
    if config.run.render_mode == 'all':
        logger.info("Using single environment with rendering")
    elif config.run.render_mode == 'none':
        logger.info("Using single environment without rendering (headless mode)")
    else:
        logger.info(f"Using single environment with selective rendering ({len(config.run.render_episodes)} episodes)")
    if not config.run.training_enabled:
        logger.warning("TRAINING DISABLED - running simulation only")

    # Training state
    completed_episodes = 0
    total_steps = 0
    training_started = False
    user_quit = False
    error_occurred = None

    # Create shared dataclasses
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

    # Setup directories for artifacts
    charts_folder = None
    text_folder = None
    run_folder = None
    run_logger = None
    aggregate_writer = None
    sim_dir = None

    if options.enable_logging:
        # Create simulation directory
        from data.abstractions import SimulationIdentifier
        from data.simulation_io import SimulationDirectory
        from data.config_serializer import create_config_snapshot
        from data.run_logger import RunLogger
        from data.aggregate_writer import AggregateWriter

        # Use script directory as base
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_id = SimulationIdentifier.create()
        sim_dir = SimulationDirectory(script_dir, sim_id)
        sim_dir.initialize()

        config_snapshot = create_config_snapshot(config, sim_id)
        sim_dir.write_config(config_snapshot)

        run_logger = RunLogger(sim_dir.runs_path)
        aggregate_writer = AggregateWriter(sim_dir.aggregates_path, write_interval=100)
        training_context.run_logger = run_logger

        text_folder = str(sim_dir.text_path)
        run_folder = str(sim_dir.root_path)
        logger.info(f"Created simulation directory: {run_folder}")

        # For normal runs (not experiments), save charts to logs/{timestamp}/ folder
        # logs/ is at project root (lunar-lander/logs), not src/logs
        if not options.is_experiment:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            project_root = Path(script_dir).parent
            run_charts_folder = project_root / "logs" / timestamp
            run_charts_folder.mkdir(parents=True, exist_ok=True)
            charts_folder = str(run_charts_folder)

    # Override with explicit directories from options
    # (charts_dir can be used for experiments too if explicitly provided)
    if options.charts_dir is not None:
        charts_folder = str(options.charts_dir)
        Path(charts_folder).mkdir(parents=True, exist_ok=True)
    if options.results_dir is not None:
        Path(options.results_dir).mkdir(parents=True, exist_ok=True)

    MAINTENANCE_INTERVAL = 25  # Reduced from 100 to prevent memory fragmentation

    try:
        with create_environments(config.run, config.environment, set(config.video.record_episodes)) as env_bundle:
            # Stop signal file for external termination
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stop_file = os.path.join(script_dir, ".stop_simulation")

            while completed_episodes < config.run.num_episodes and not user_quit:
                # Check for external stop signal
                if os.path.exists(stop_file):
                    logger.info("Stop signal received, shutting down gracefully...")
                    os.remove(stop_file)
                    user_quit = True
                    break

                # Periodic maintenance
                if completed_episodes > 0 and completed_episodes % MAINTENANCE_INTERVAL == 0:
                    if pg is not None:
                        pg.event.pump()
                    gc_counts_before = gc.get_count()
                    gc.collect()
                    gc_counts_after = gc.get_count()
                    logger.debug(f"GC at episode {completed_episodes}: before={gc_counts_before}, after={gc_counts_after}")
                    if T.cuda.is_available():
                        T.cuda.empty_cache()

                # Periodic memory monitoring and enforcement via MemoryLimiter
                if completed_episodes > 0 and completed_episodes % MEMORY_LOG_INTERVAL == 0:
                    memory_mb = memory_limiter.get_memory_mb()
                    logger.info(f"Memory usage at episode {completed_episodes}: {memory_mb:.1f} MB ({memory_limiter.get_memory_percent():.1f}% of limit)")

                # Memory threshold safeguard - check periodically using MemoryLimiter
                if completed_episodes > 0 and completed_episodes % MEMORY_CHECK_INTERVAL == 0:
                    is_critical = memory_limiter.check_and_cleanup()
                    if is_critical:
                        # Memory is critically high - save checkpoint and consider stopping
                        logger.error(f"Critical memory pressure at episode {completed_episodes}!")
                        _save_model_checkpoint(trainer, completed_episodes, options, sim_dir, is_final=False)
                        # Continue but warn - the limiter has already done aggressive cleanup

                # Determine if this episode should be rendered to screen (needs pygame)
                should_render = completed_episodes in env_bundle.render_episodes and pygame_ctx is not None

                # Determine if this episode should be recorded for video
                # Recording can happen without screen display - just needs frames_dir
                record_episode = (
                    completed_episodes in config.video.record_episodes
                    and options.frames_dir is not None
                )

                # Run episode (same logic for both rendered and unrendered)
                result, episode_steps = _run_episode(
                    env_bundle.env,
                    trainer,
                    noise,
                    completed_episodes,
                    config,
                    training_context,
                    timing_state,
                    output,
                    should_render=should_render,
                    pygame_ctx=pygame_ctx,
                    record_episode=record_episode,
                    frames_dir=options.frames_dir
                )

                if result is None:
                    logger.warning(f"Episode returned None at episode {completed_episodes} - user quit detected")
                    user_quit = True
                    break

                total_steps = timing_state.total_steps

                # Training after episode
                if config.run.training_enabled and replay_buffer.is_ready(config.training.min_experiences_before_training):
                    if not training_started:
                        logger.info(
                            f">>> TRAINING STARTED at episode {completed_episodes} "
                            f"with {len(replay_buffer)} experiences <<<"
                        )
                        training_started = True

                    if config.training.use_per:
                        progress = completed_episodes / config.run.num_episodes
                        replay_buffer.anneal_beta(progress)

                    updates_to_do = max(1, episode_steps // 4)
                    metrics = trainer.train_on_buffer(replay_buffer, updates_to_do)
                    trainer.step_schedulers()
                    timing_state.total_training_updates = trainer.training_steps

                    if completed_episodes % 10 == 0:
                        noise_scale = compute_noise_scale(
                            completed_episodes,
                            config.noise.noise_scale_initial,
                            config.noise.noise_scale_final,
                            config.noise.noise_decay_episodes
                        )
                        diagnostics.record_training_metrics(metrics)
                        reporter.log_training_update(metrics, noise_scale)

                # Reset noise for next episode
                noise.reset()
                completed_episodes += 1

                # Explicit cleanup: delete result reference to allow GC
                del result

                # Batch completion handling (interval based on diagnostics batch size)
                report_interval = options.diagnostics_batch_size * 2  # Report every 2 batches
                if completed_episodes % report_interval == 0:
                    batch_num = completed_episodes // options.diagnostics_batch_size
                    start_ep = (batch_num - 2) * options.diagnostics_batch_size + 1
                    output.print_batch_completed(batch_num, start_ep, completed_episodes, diagnostics)
                    _run_periodic_diagnostics(
                        reporter, diagnostics, charts_folder, text_folder,
                        completed_episodes, output, open_charts=options.show_final_charts,
                        batch_size=options.diagnostics_batch_size
                    )
                    if aggregate_writer:
                        aggregate_writer.maybe_write(completed_episodes, diagnostics)

                # Periodic model checkpoint saving
                if options.save_model and completed_episodes % CHECKPOINT_INTERVAL == 0:
                    _save_model_checkpoint(
                        trainer, completed_episodes, options, sim_dir, is_final=False
                    )

    except KeyboardInterrupt:
        error_occurred = "KeyboardInterrupt"
        logger.warning("\n\nTraining interrupted by user (Ctrl+C)")

    except Exception as e:
        error_occurred = f"{type(e).__name__}: {e}"
        logger.error(f"\n\nUnexpected error occurred: {e}")
        traceback.print_exc()

    finally:
        elapsed_time = time.time() - start_time if start_time is not None else 0.0

        if completed_episodes > 0:
            output.print_final_summary(
                completed_episodes=completed_episodes,
                error_occurred=error_occurred,
                elapsed_time=elapsed_time,
                total_steps=total_steps,
                training_steps=trainer.training_steps,
                tracker=diagnostics
            )

            if output.is_verbose():
                try:
                    reporter.print_summary()
                except Exception as e:
                    logger.error(f"Failed to print full diagnostics: {e}")

            # Write final aggregate
            if aggregate_writer:
                try:
                    aggregate_writer.write_final(diagnostics)
                    logger.info("Final aggregate saved")
                except Exception as e:
                    logger.error(f"Failed to write final aggregate: {e}")

            # Save final model (always attempt on exit, even if interrupted)
            if options.save_model:
                _save_model_checkpoint(
                    trainer, completed_episodes, options, sim_dir, is_final=True
                )

            # Generate final charts
            if charts_folder is not None:
                try:
                    chart_path = os.path.join(charts_folder, f"chart_final_{completed_episodes}.png")
                    chart_gen = ChartGenerator(diagnostics, batch_size=options.diagnostics_batch_size)
                    chart_gen.generate_to_file(chart_path)
                    logger.info(f"Final chart saved to {chart_path}")
                except Exception as e:
                    logger.error(f"Failed to generate charts: {e}")

            # Show final chart if requested
            if options.show_final_charts and output.is_verbose():
                try:
                    chart_generator = ChartGenerator(diagnostics, batch_size=options.diagnostics_batch_size)
                    chart_generator.generate_all(show=True, block=True)
                except Exception as e:
                    logger.error(f"Failed to show final charts: {e}")

            # Export chart data to run_data.jsonl for experiments
            if options.is_experiment and options.results_dir is not None:
                try:
                    import json
                    run_data_path = options.results_dir / "run_data.jsonl"
                    chart_data = diagnostics.to_chart_data()
                    with open(run_data_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(chart_data) + '\n')
                    logger.info(f"Chart data exported to {run_data_path}")
                except Exception as e:
                    logger.error(f"Failed to export chart data: {e}")
        else:
            if not output.is_silent():
                print("\nNo episodes completed - no diagnostics to show")

        # Cleanup pygame
        if pg is not None:
            try:
                pg.quit()
            except Exception:
                pass

    # Compute result metrics
    episode_rewards = np.array(diagnostics.env_rewards)
    successes = np.array(diagnostics.successes_bool)

    if len(episode_rewards) > 0:
        success_rate = float(np.mean(successes)) * 100
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        max_reward = float(np.max(episode_rewards))
        min_reward = float(np.min(episode_rewards))
        final_100_success_rate = float(np.mean(successes[-100:])) * 100 if len(successes) >= 100 else None
        final_100_mean_reward = float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else None
    else:
        success_rate = 0.0
        mean_reward = 0.0
        std_reward = 0.0
        max_reward = 0.0
        min_reward = 0.0
        final_100_success_rate = None
        final_100_mean_reward = None

    return TrainingResult(
        success_rate=success_rate,
        mean_reward=mean_reward,
        std_reward=std_reward,
        max_reward=max_reward,
        min_reward=min_reward,
        first_success_episode=diagnostics.first_success_episode,
        final_100_success_rate=final_100_success_rate,
        final_100_mean_reward=final_100_mean_reward,
        total_episodes=completed_episodes,
        total_successes=diagnostics.success_count,
        max_consecutive_successes=diagnostics.max_streak,
        elapsed_time=elapsed_time,
        user_quit=user_quit,
        error=error_occurred,
        diagnostics=diagnostics
    )
