"""Video recording utilities for training episodes.

This module provides functionality to:
1. Save individual frames during episode execution
2. Compile frames into MP4 videos using imageio
"""

import logging
import os
from pathlib import Path
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class EpisodeRecorder:
    """Records frames from a single episode and saves them to disk.

    Usage:
        recorder = EpisodeRecorder(frames_dir, episode_num)
        for frame in episode_frames:
            recorder.save_frame(frame)
        recorder.finalize()
    """

    def __init__(self, frames_dir: Path, episode_num: int, frame_format: str = "png"):
        """Initialize recorder for a specific episode.

        Args:
            frames_dir: Directory to save frames (e.g., experiments/EXP_XXX/frames)
            episode_num: Episode number (used for subdirectory naming)
            frame_format: Image format for frames (png or jpg)
        """
        self.frames_dir = Path(frames_dir)
        self.episode_num = episode_num
        self.frame_format = frame_format
        self.frame_count = 0

        # Create episode-specific subdirectory
        self.episode_dir = self.frames_dir / f"episode_{episode_num:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Recording episode {episode_num} to {self.episode_dir}")

    def save_frame(self, frame: np.ndarray) -> None:
        """Save a single frame to disk.

        Args:
            frame: RGB array from env.render() with shape (H, W, 3)
        """
        import imageio

        frame_path = self.episode_dir / f"frame_{self.frame_count:06d}.{self.frame_format}"
        imageio.imwrite(frame_path, frame)
        self.frame_count += 1

    def finalize(self) -> int:
        """Finalize recording and return frame count.

        Returns:
            Number of frames saved for this episode
        """
        logger.info(f"Episode {self.episode_num}: saved {self.frame_count} frames to {self.episode_dir}")
        return self.frame_count


def compile_episode_video(
    frames_dir: Path,
    episode_num: int,
    video_dir: Path,
    framerate: int = 45,
    frame_format: str = "png"
) -> Optional[Path]:
    """Compile frames from a single episode into an MP4 video.

    Args:
        frames_dir: Directory containing episode frame subdirectories
        episode_num: Episode number to compile
        video_dir: Directory to save output video
        framerate: Video framerate in FPS
        frame_format: Frame image format

    Returns:
        Path to created video file, or None if no frames found
    """
    import imageio

    episode_dir = Path(frames_dir) / f"episode_{episode_num:06d}"
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Get sorted list of frame files
    frame_files = sorted(episode_dir.glob(f"frame_*.{frame_format}"))

    if not frame_files:
        logger.warning(f"No frames found for episode {episode_num} in {episode_dir}")
        return None

    # Create video
    video_path = video_dir / f"episode_{episode_num:06d}.mp4"

    logger.info(f"Compiling {len(frame_files)} frames to {video_path} at {framerate} FPS")

    with imageio.get_writer(video_path, fps=framerate, codec='libx264', quality=8) as writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            writer.append_data(frame)

    logger.info(f"Created video: {video_path}")
    return video_path


def compile_all_videos(
    frames_dir: Path,
    video_dir: Path,
    framerate: int = 45,
    frame_format: str = "png"
) -> List[Path]:
    """Compile all recorded episodes into individual videos.

    Args:
        frames_dir: Directory containing episode frame subdirectories
        video_dir: Directory to save output videos
        framerate: Video framerate in FPS
        frame_format: Frame image format

    Returns:
        List of paths to created video files
    """
    frames_dir = Path(frames_dir)
    video_paths = []

    # Find all episode directories
    episode_dirs = sorted(frames_dir.glob("episode_*"))

    if not episode_dirs:
        logger.warning(f"No episode directories found in {frames_dir}")
        return video_paths

    logger.info(f"Found {len(episode_dirs)} episodes to compile")

    for episode_dir in episode_dirs:
        # Extract episode number from directory name
        try:
            episode_num = int(episode_dir.name.split("_")[1])
        except (IndexError, ValueError):
            logger.warning(f"Could not parse episode number from {episode_dir.name}")
            continue

        video_path = compile_episode_video(
            frames_dir, episode_num, video_dir, framerate, frame_format
        )
        if video_path:
            video_paths.append(video_path)

    logger.info(f"Compiled {len(video_paths)} videos to {video_dir}")
    return video_paths


def create_compilation_video(
    video_dir: Path,
    output_name: str = "all_episodes.mp4",
    framerate: int = 45
) -> Optional[Path]:
    """Concatenate all episode videos into a single compilation.

    Args:
        video_dir: Directory containing individual episode videos
        output_name: Name for the compilation video
        framerate: Video framerate in FPS

    Returns:
        Path to compilation video, or None if no videos found
    """
    import imageio

    video_dir = Path(video_dir)
    episode_videos = sorted(video_dir.glob("episode_*.mp4"))

    if not episode_videos:
        logger.warning(f"No episode videos found in {video_dir}")
        return None

    output_path = video_dir / output_name

    logger.info(f"Creating compilation from {len(episode_videos)} episodes")

    with imageio.get_writer(output_path, fps=framerate, codec='libx264', quality=8) as writer:
        for video_path in episode_videos:
            logger.debug(f"Adding {video_path.name} to compilation")
            reader = imageio.get_reader(video_path)
            for frame in reader:
                writer.append_data(frame)
            reader.close()

    logger.info(f"Created compilation: {output_path}")
    return output_path
