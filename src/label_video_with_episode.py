"""Tracer bullet: Label single video with episode text overlay."""
import cv2
from pathlib import Path

# Configuration - set video folder here
VIDEO_FOLDER = Path(r"C:\Repositories for Git\lunar-lander-file-folder\lunar-lander\experiments\EXP_023_long_training_with_recording\video")
OUTPUT_FOLDER = VIDEO_FOLDER.parent / "video_labeled"

# Test with first episode video
test_video = VIDEO_FOLDER / "episode_000001.mp4"

# Create output directory
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Open video
cap = cv2.VideoCapture(str(test_video))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input: {test_video}")
print(f"Video properties: {width}x{height} @ {fps} FPS")

# Output video writer
output_path = OUTPUT_FOLDER / "episode_000001_labeled.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# Process frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add yellow text "Episode 0001" at top-left
    cv2.putText(
        frame,
        "Episode 0001",
        (10, 25),  # position (x, y from top-left) - adjusted for smaller text
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,  # font scale (half size)
        (0, 255, 255),  # BGR yellow
        2,  # thickness (reduced for smaller text)
        cv2.LINE_AA
    )

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print(f"Processed {frame_count} frames")
print(f"Created: {output_path}")
