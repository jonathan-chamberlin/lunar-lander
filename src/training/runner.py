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
import torch as T

from config import Config
from data_types import Experience
from training.environment import (
    create_environments,
    shape_reward,
    compute_noise_scale,
    EpisodeManager
)
from training.noise import OUActionNoise
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from training.trainer import TD3Trainer
from training.training_options import TrainingOptions
from training.training_result import TrainingResult
from analysis.diagnostics import DiagnosticsTracker, DiagnosticsReporter
from analysis.behavior_analysis import BehaviorAnalyzer
from analysis.charts import ChartGenerator
from analysis.output_formatter import format_behavior_output
from constants import SAFE_LANDING_OUTCOMES, OUTCOME_TO_CATEGORY
from output_mode import OutputController

if TYPE_CHECKING:
    from models import EpisodeData, TimingState, TrainingContext, PyGameContext

logger = logging.getLogger(__name__)

# Module-level behavior analyzer instance
behavior_analyzer = BehaviorAnalyzer()


def _run_periodic_diagnostics(
    reporter: DiagnosticsReporter,
    diagnostics: DiagnosticsTracker,
    charts_folder: Optional[str],
    text_folder: Optional[str],
    completed_episodes: int,
    output: OutputController,
    open_charts: bool = False
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
        chart_gen = ChartGenerator(diagnostics, batch_size=50)
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

    # Analyze behaviors
    behavior_report = None
    outcome = 'UNKNOWN'
    outcome_category = 'crashed'
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


def _run_rendered_episode(
    render_env,
    trainer: TD3Trainer,
    noise: OUActionNoise,
    episode_num: int,
    config: Config,
    training_context: "TrainingContext",
    timing_state: "TimingState",
    pygame_ctx: "PyGameContext",
    output: OutputController
) -> tuple[Optional["EpisodeResult"], int]:
    """Run a single rendered episode with pygame display.

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
    import pygame as pg
    from models import EpisodeData
    from data_types import EpisodeResult

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

    # Pre-allocate surface for frame rendering
    frame_surface = pg.Surface((600, 400))
    frame_buffer = np.empty((600, 400, 3), dtype=np.uint8)

    # Pre-render text if overlay is enabled
    if config.display.show_run_overlay:
        text_surface = pygame_ctx.font.render(f"Run: {episode_num}", True, config.display.font_color)
        text_pos = (config.display.text_x, config.display.text_y)

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                logger.warning(f"pygame QUIT event received at episode {episode_num}")
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

        action_np = action.detach().cpu().numpy()
        actions.append(action_np)
        observations_list.append(obs.copy())

        # Step environment
        next_obs, reward, terminated, truncated, info = render_env.step(action_np)
        next_state = T.from_numpy(next_obs).float()

        # Render frame
        frame = render_env.render()
        frame_buffer[:] = frame.transpose(1, 0, 2)
        pg.surfarray.blit_array(frame_surface, frame_buffer)
        pygame_ctx.screen.blit(frame_surface, (0, 0))

        if config.display.show_run_overlay:
            pygame_ctx.screen.blit(text_surface, text_pos)

        pg.display.flip()
        if config.run.framerate is not None:
            pygame_ctx.clock.tick(config.run.framerate)

        # Apply reward shaping
        current_step = len(rewards)
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

    env_reward = float(np.sum(rewards))
    total_reward = env_reward + shaped_bonus
    duration_seconds = time.time() - episode_start_time

    timing_state.total_steps += episode_steps

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
    result = _finalize_episode(episode_data, training_context, timing_state, output)
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

    output = OutputController.from_string(options.output_mode)

    logger.info(f"Starting TD3 training for {config.run.num_episodes} episodes")
    if config.run.render_mode == 'all':
        logger.info("Using single rendered environment (render_mode='all')")
    elif config.run.render_mode == 'none':
        logger.info(f"Using {config.run.num_envs} parallel environments (render_mode='none')")
    else:
        logger.info(f"Using mixed mode: {len(config.run.render_episodes)} rendered, rest use {config.run.num_envs} parallel envs")
    if not config.run.training_enabled:
        logger.warning("TRAINING DISABLED - running simulation only")

    # Training state
    completed_episodes = 0
    steps_since_training = 0
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

        charts_folder = str(sim_dir.charts_path)
        text_folder = str(sim_dir.text_path)
        run_folder = str(sim_dir.root_path)
        logger.info(f"Created simulation directory: {run_folder}")

    # Override with explicit directories from options
    if options.charts_dir is not None:
        charts_folder = str(options.charts_dir)
        Path(charts_folder).mkdir(parents=True, exist_ok=True)
    if options.results_dir is not None:
        Path(options.results_dir).mkdir(parents=True, exist_ok=True)

    MAINTENANCE_INTERVAL = 100

    try:
        with create_environments(config.run, config.environment) as env_bundle:
            observations, _ = env_bundle.vec_env.reset()
            states = T.from_numpy(observations).float()
            episode_manager = EpisodeManager(config.run.num_envs)

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

                # Check if current episode should be rendered
                if completed_episodes in env_bundle.render_episodes:
                    if pygame_ctx is None:
                        # Need pygame but it wasn't initialized - skip rendered episodes
                        logger.warning(f"Skipping rendered episode {completed_episodes} - pygame not initialized")
                    else:
                        result, episode_steps = _run_rendered_episode(
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
                        total_steps = timing_state.total_steps

                        if result is None:
                            logger.warning(f"Rendered episode returned None at episode {completed_episodes} - user quit detected")
                            user_quit = True
                            break

                        # Training after rendered episode
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

                            updates_to_do = max(1, result.steps // 4)
                            metrics = trainer.train_on_buffer(replay_buffer, updates_to_do)
                            trainer.step_schedulers()

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

                        if completed_episodes % 100 == 0:
                            batch_num = completed_episodes // 100
                            output.print_batch_completed(batch_num, (batch_num - 1) * 100 + 1, completed_episodes, diagnostics)
                            _run_periodic_diagnostics(
                                reporter, diagnostics, charts_folder, text_folder,
                                completed_episodes, output, open_charts=options.show_final_charts
                            )
                            if aggregate_writer:
                                aggregate_writer.maybe_write(completed_episodes, diagnostics)

                        continue

                # Non-rendered vectorized training path
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
                        total_reward, env_reward, shaped_bonus, actions_array, observations_array, duration = \
                            episode_manager.get_episode_stats(i)

                        episode_data = EpisodeData(
                            episode_num=completed_episodes,
                            total_reward=total_reward,
                            env_reward=env_reward,
                            shaped_bonus=shaped_bonus,
                            steps=episode_manager.step_counts[i],
                            duration_seconds=duration,
                            actions_array=actions_array,
                            observations_array=observations_array,
                            terminated=terminateds[i],
                            truncated=truncateds[i],
                            rendered=False
                        )
                        _finalize_episode(episode_data, training_context, timing_state, output)

                        episode_manager.reset_env(i)
                        noise.reset(i)
                        completed_episodes += 1

                        if completed_episodes % 100 == 0:
                            batch_num = completed_episodes // 100
                            output.print_batch_completed(batch_num, (batch_num - 1) * 100 + 1, completed_episodes, diagnostics)
                            _run_periodic_diagnostics(
                                reporter, diagnostics, charts_folder, text_folder,
                                completed_episodes, output, open_charts=options.show_final_charts
                            )
                            if aggregate_writer:
                                aggregate_writer.maybe_write(completed_episodes, diagnostics)

                        if completed_episodes >= config.run.num_episodes:
                            break

                states = next_states
                observations = next_observations
                steps_since_training += config.run.num_envs
                total_steps += config.run.num_envs
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

                    if config.training.use_per:
                        progress = completed_episodes / config.run.num_episodes
                        replay_buffer.anneal_beta(progress)

                    updates_to_do = max(1, steps_since_training // 4)
                    metrics = trainer.train_on_buffer(replay_buffer, updates_to_do)
                    steps_since_training = 0
                    trainer.step_schedulers()

                    if completed_episodes % 10 == 0:
                        diagnostics.record_training_metrics(metrics)
                        reporter.log_training_update(metrics, noise_scale)

    except KeyboardInterrupt:
        error_occurred = "KeyboardInterrupt"
        logger.warning("\n\nTraining interrupted by user (Ctrl+C)")

    except Exception as e:
        if pg is not None:
            import pygame.error
            if isinstance(e, pygame.error):
                error_occurred = f"pygame.error: {e}"
                logger.error(f"\n\nPygame error occurred: {e}")
            else:
                error_occurred = f"{type(e).__name__}: {e}"
                logger.error(f"\n\nUnexpected error occurred: {e}")
                traceback.print_exc()
        else:
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

            # Save model
            if options.save_model and sim_dir is not None:
                try:
                    model_path = str(sim_dir.models_path / "final_model")
                    trainer.save(model_path)
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

            # Generate final charts
            if charts_folder is not None:
                try:
                    chart_path = os.path.join(charts_folder, f"chart_final_{completed_episodes}.png")
                    chart_gen = ChartGenerator(diagnostics, batch_size=50)
                    chart_gen.generate_to_file(chart_path)
                    logger.info(f"Final chart saved to {chart_path}")
                except Exception as e:
                    logger.error(f"Failed to generate charts: {e}")

            # Show final chart if requested
            if options.show_final_charts and output.is_verbose():
                try:
                    chart_generator = ChartGenerator(diagnostics, batch_size=50)
                    chart_generator.generate_all(show=True, block=True)
                except Exception as e:
                    logger.error(f"Failed to show final charts: {e}")
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
    else:
        success_rate = 0.0
        mean_reward = 0.0
        std_reward = 0.0
        max_reward = 0.0
        min_reward = 0.0
        final_100_success_rate = None

    return TrainingResult(
        success_rate=success_rate,
        mean_reward=mean_reward,
        std_reward=std_reward,
        max_reward=max_reward,
        min_reward=min_reward,
        first_success_episode=diagnostics.first_success_episode,
        final_100_success_rate=final_100_success_rate,
        total_episodes=completed_episodes,
        elapsed_time=elapsed_time,
        user_quit=user_quit,
        error=error_occurred,
        diagnostics=diagnostics
    )
