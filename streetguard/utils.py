"""
Utility functions for video handling, time conversions, file validation, and snapshot persistence.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Stores video stream metadata."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    is_valid: bool = True
    error_message: Optional[str] = None


def seconds_to_hms(seconds: float) -> str:
    """Format floating point seconds into HH:MM:SS format."""
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Inspect and extract properties from a video file using OpenCV."""
    path = Path(video_path)
    if not path.is_file():
        return VideoMetadata(
            width=0,
            height=0,
            fps=0.0,
            total_frames=0,
            duration_seconds=0.0,
            is_valid=False,
            error_message=f"File not found: '{video_path}'",
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoMetadata(
            width=0,
            height=0,
            fps=0.0,
            total_frames=0,
            duration_seconds=0.0,
            is_valid=False,
            error_message=f"OpenCV could not open video stream: '{video_path}'",
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        fps = 30.0  # Safe fallback for non-standard video streams

    duration = total_frames / fps if fps > 0 else 0.0

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration,
        is_valid=True,
    )


def save_alert_screenshot(
    frame: np.ndarray,
    output_dir: Path,
    category: str,
    track_id: Optional[int],
    timestamp_sec: float,
) -> str:
    """
    Save annotated screenshot of an incident to disk and return the file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    tid_str = f"track_{track_id}" if track_id is not None else "notrack"
    filename = f"{category}_{tid_str}_{int(timestamp_sec)}s_{time_str}.jpg"
    full_path = output_dir / filename

    # Write JPEG image
    cv2.imwrite(str(full_path), frame)
    return str(full_path)
