"""
Tracker state and history management for StreetGuard AI.
Maintains object trajectories, temporal metrics, and handles missing or lost track IDs safely.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple
import numpy as np

from streetguard.detector import Detection


@dataclass
class TrackState:
    """Detailed temporal history and metrics for an individual tracked object."""
    track_id: int
    class_id: int
    class_name: str
    first_seen_time: float
    last_seen_time: float
    history: Deque[Tuple[float, Tuple[int, int], Tuple[int, int, int, int]]] = field(
        default_factory=lambda: deque(maxlen=150)
    )  # (timestamp, center, box)
    frames_tracked: int = 1
    missed_frames: int = 0
    current_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    current_center: Tuple[int, int] = (0, 0)
    current_footprint: Tuple[int, int] = (0, 0)
    is_active: bool = True

    def update(
        self,
        box: Tuple[int, int, int, int],
        timestamp: float,
        footprint: Tuple[int, int],
    ) -> None:
        """Update track with a newly observed detection."""
        self.current_box = box
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        self.current_center = (cx, cy)
        self.current_footprint = footprint
        self.history.append((timestamp, (cx, cy), box))
        self.last_seen_time = timestamp
        self.frames_tracked += 1
        self.missed_frames = 0
        self.is_active = True

    def mark_missed(self) -> None:
        """Increment missed frames count when track is not detected in current frame."""
        self.missed_frames += 1
        if self.missed_frames > 30:  # ~1 second at 30 fps
            self.is_active = False

    def get_displacement_over_window(self, duration_sec: float, current_time: float) -> float:
        """
        Compute maximum Euclidean distance between any recorded centroid in the time window
        [current_time - duration_sec, current_time] and the initial position in that window.
        Returns 0.0 if insufficient history.
        """
        window_points = [
            center for ts, center, _ in self.history if (current_time - ts) <= duration_sec
        ]
        if len(window_points) < 2:
            return 0.0

        ref_point = np.array(window_points[0], dtype=float)
        distances = [np.linalg.norm(np.array(pt, dtype=float) - ref_point) for pt in window_points]
        return float(np.max(distances))

    def observation_duration(self) -> float:
        """Total time in seconds this object has been tracked."""
        return max(0.0, self.last_seen_time - self.first_seen_time)


class TrackHistoryManager:
    """
    Maintains and updates tracks over time.
    Safely handles unassigned detections, reappearing tracks, and track eviction.
    """

    def __init__(self, max_missed_frames: int = 45):
        self.tracks: Dict[int, TrackState] = {}
        self.max_missed_frames = max_missed_frames

    def update(self, detections: List[Detection], current_time: float) -> List[TrackState]:
        """
        Update tracking state with current frame detections.
        Handles missing tracking IDs safely without crashing.
        """
        detected_track_ids = set()

        for det in detections:
            # Handle detections that do not yet have a track ID assigned by ByteTrack
            if det.track_id is None:
                continue

            tid = det.track_id
            detected_track_ids.add(tid)

            if tid not in self.tracks:
                # Brand new track
                state = TrackState(
                    track_id=tid,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    first_seen_time=current_time,
                    last_seen_time=current_time,
                    current_box=det.box,
                    current_center=det.center,
                    current_footprint=det.footprint,
                )
                state.history.append((current_time, det.center, det.box))
                self.tracks[tid] = state
            else:
                # Update existing track
                self.tracks[tid].update(det.box, current_time, det.footprint)

        # Mark missed frames for tracks that were not detected this frame
        inactive_ids = []
        for tid, track in self.tracks.items():
            if tid not in detected_track_ids:
                track.mark_missed()
                if track.missed_frames > self.max_missed_frames:
                    inactive_ids.append(tid)

        # Remove stale tracks to prevent memory accumulation
        for tid in inactive_ids:
            del self.tracks[tid]

        return [t for t in self.tracks.values() if t.is_active]

    def get_active_tracks_by_class(self, class_ids: List[int]) -> List[TrackState]:
        """Filter active tracks by class ID."""
        return [
            t for t in self.tracks.values()
            if t.is_active and t.class_id in class_ids
        ]

    def reset(self) -> None:
        """Clear all tracking state."""
        self.tracks.clear()
