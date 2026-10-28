"""
Restricted-area intrusion detection rule for StreetGuard AI.
Monitors user-defined polygonal regions and raises explainable intrusion alerts
when tracked persons enter the zone, strictly preventing duplicate alerts while inside.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from streetguard.config import IntrusionRuleConfig
from streetguard.tracker import TrackState


@dataclass
class IntrusionAlert:
    """Represents an intrusion violation event."""
    track_id: int
    entry_timestamp: float
    formatted_time: str
    point: Tuple[int, int]  # (x, y) ground or centroid coordinate
    zone_name: str
    explanation: str
    box: Tuple[int, int, int, int]
    object_category: str = "person"


class RestrictedAreaIntrusionDetector:
    """
    Evaluates tracked persons against a configurable polygonal boundary.
    Emits alerts on entry and avoids repeated alerts for the same person
    while they remain inside the perimeter.
    """

    def __init__(self, config: Optional[IntrusionRuleConfig] = None):
        self.config = config or IntrusionRuleConfig()
        # Per-track state: track_id -> {is_inside: bool, alerted: bool, frames_outside: int}
        self.track_states: Dict[int, Dict[str, Any]] = {}
        # Precomputed pixel polygon contour (updated when frame dimensions or polygon changes)
        self._cached_pixel_polygon: Optional[np.ndarray] = None
        self._cached_resolution: Optional[Tuple[int, int]] = None

    def set_polygon_normalized(self, polygon_normalized: List[Tuple[float, float]], zone_name: Optional[str] = None):
        """Update the normalized polygon vertices."""
        self.config.polygon_normalized = polygon_normalized
        if zone_name:
            self.config.zone_name = zone_name
        self._cached_pixel_polygon = None

    def get_pixel_polygon(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Calculate and cache pixel coordinates from normalized polygon."""
        if (
            self._cached_pixel_polygon is not None
            and self._cached_resolution == (frame_width, frame_height)
        ):
            return self._cached_pixel_polygon

        pts = [
            [int(x * frame_width), int(y * frame_height)]
            for x, y in self.config.polygon_normalized
        ]
        self._cached_pixel_polygon = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        self._cached_resolution = (frame_width, frame_height)
        return self._cached_pixel_polygon

    def is_point_inside(
        self, point: Tuple[int, int], frame_width: int, frame_height: int
    ) -> bool:
        """
        Check if a given pixel point is inside the restricted polygon.
        Uses cv2.pointPolygonTest (returns >= 0 when point is inside or on edge).
        """
        poly = self.get_pixel_polygon(frame_width, frame_height)
        # OpenCV expects (float(x), float(y))
        test = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), measureDist=False)
        return test >= 0

    def evaluate(
        self,
        tracks: List[TrackState],
        frame_width: int,
        frame_height: int,
        current_time: float,
        formatted_time: str,
    ) -> List[IntrusionAlert]:
        """
        Evaluate active tracks against the restricted area polygon.
        Generates alerts only when a person transitions from outside to inside,
        or upon first entering the restricted zone.
        """
        if not self.config.enabled or not self.config.polygon_normalized:
            return []

        alerts: List[IntrusionAlert] = []
        active_ids = set()

        for track in tracks:
            # Only monitor persons for perimeter intrusion
            if track.class_id != 0 and track.class_name.lower() != "person":
                continue

            active_ids.add(track.track_id)

            # Determine point of contact (footprint / ground contact point preferred)
            test_pt = track.current_footprint if self.config.use_footprint else track.current_center
            is_inside = self.is_point_inside(test_pt, frame_width, frame_height)

            # Retrieve or initialize track state
            state = self.track_states.get(
                track.track_id,
                {"is_inside": False, "alerted": False, "frames_outside": 0},
            )

            if is_inside:
                # Intrusion condition: person is inside and has not yet been alerted for this entry
                if not state["alerted"]:
                    explanation = (
                        f"Perimeter Intrusion: Track #{track.track_id} (Person) "
                        f"entered restricted zone '{self.config.zone_name}' at {formatted_time}. "
                        f"Observed ground footprint: ({test_pt[0]}, {test_pt[1]})."
                    )
                    alert = IntrusionAlert(
                        track_id=track.track_id,
                        entry_timestamp=current_time,
                        formatted_time=formatted_time,
                        point=test_pt,
                        zone_name=self.config.zone_name,
                        explanation=explanation,
                        box=track.current_box,
                        object_category="person",
                    )
                    alerts.append(alert)
                    state["alerted"] = True

                state["is_inside"] = True
                state["frames_outside"] = 0
            else:
                # Person is outside polygon
                state["frames_outside"] += 1
                state["is_inside"] = False
                # If person has remained outside for cooldown period, allow new alert upon re-entry
                if state["frames_outside"] >= self.config.reentry_cooldown_frames:
                    state["alerted"] = False

            self.track_states[track.track_id] = state

        # Clean up stale tracks from internal state
        stale_ids = [tid for tid in self.track_states if tid not in active_ids]
        for tid in stale_ids:
            # Keep track state briefly in case track re-appears shortly, or evict if outside long enough
            if self.track_states[tid]["frames_outside"] > 90:
                del self.track_states[tid]

        return alerts

    def reset(self) -> None:
        """Clear all active intrusion tracking states."""
        self.track_states.clear()
        self._cached_pixel_polygon = None
