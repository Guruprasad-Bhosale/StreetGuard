"""
Video analytics pipeline for StreetGuard AI.
Coordinates video decoding, YOLO detection, ByteTrack tracking,
rule evaluation, incident logging, and annotated visualization.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Generator, List, Optional
import cv2
import numpy as np

from streetguard.config import (
    IntrusionRuleConfig,
    ModelConfig,
    UnattendedRuleConfig,
    VideoProcessingConfig,
)
from streetguard.database import IncidentDatabase, IncidentRecord
from streetguard.detector import Detection, YOLODetector
from streetguard.rules.intrusion import IntrusionAlert, RestrictedAreaIntrusionDetector
from streetguard.rules.unattended import CandidateDiagnostic, UnattendedAlert, UnattendedObjectDetector
from streetguard.tracker import TrackHistoryManager, TrackState
from streetguard.utils import get_video_metadata, save_alert_screenshot, seconds_to_hms
from streetguard.visualizer import SurveillanceVisualizer


@dataclass
class FrameAnalyticsResult:
    """Telemetry and visual frame output for an individual processed frame."""
    frame_idx: int
    video_timestamp: float
    formatted_time: str
    fps: float  # End-to-end processing FPS
    inference_fps: float  # Pure YOLO + ByteTrack inference FPS
    progress_ratio: float
    annotated_frame: np.ndarray
    detections: List[Detection]
    active_tracks: List[TrackState]
    intrusion_alerts: List[IntrusionAlert]
    unattended_alerts: List[UnattendedAlert]
    person_count: int
    object_count: int
    candidate_diagnostics: List[CandidateDiagnostic] = field(default_factory=list)


class VideoAnalyticsPipeline:
    """
    Main pipeline engine orchestrating object tracking, spatial reasoning,
    heuristics, and database logging.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        intrusion_config: Optional[IntrusionRuleConfig] = None,
        unattended_config: Optional[UnattendedRuleConfig] = None,
        video_config: Optional[VideoProcessingConfig] = None,
        db: Optional[IncidentDatabase] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.intrusion_config = intrusion_config or IntrusionRuleConfig()
        self.unattended_config = unattended_config or UnattendedRuleConfig()
        self.video_config = video_config or VideoProcessingConfig()

        # Database and Storage
        self.db = db or IncidentDatabase(self.video_config.db_path)
        self.screenshot_dir = self.video_config.screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Core modules
        self.detector = YOLODetector(self.model_config)
        self.tracker_manager = TrackHistoryManager()
        self.intrusion_detector = RestrictedAreaIntrusionDetector(self.intrusion_config)
        self.unattended_detector = UnattendedObjectDetector(self.unattended_config)
        self.visualizer = SurveillanceVisualizer(self.video_config)

    def reset(self) -> None:
        """Reset internal trackers and state before starting a new video."""
        self.tracker_manager.reset()
        self.intrusion_detector.reset()
        self.unattended_detector.reset()

    def update_configs(
        self,
        intrusion_config: Optional[IntrusionRuleConfig] = None,
        unattended_config: Optional[UnattendedRuleConfig] = None,
    ) -> None:
        """Update runtime rule parameters dynamically from UI controls."""
        if intrusion_config is not None:
            self.intrusion_config = intrusion_config
            self.intrusion_detector.config = intrusion_config
        if unattended_config is not None:
            self.unattended_config = unattended_config
            self.unattended_detector.config = unattended_config

    def process_video_stream(
        self,
        video_path: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameAnalyticsResult, None, None]:
        """
        Stream frames from a video file, executing detection, tracking,
        rule verification, and yielding FrameAnalyticsResult for each frame.
        Safely releases all video resources upon termination or interruption.
        """
        self.reset()
        meta = get_video_metadata(video_path)
        if not meta.is_valid:
            raise ValueError(f"Invalid video file: {meta.error_message}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video: {video_path}")

        video_name = Path(video_path).name
        video_fps = meta.fps if meta.fps > 0 else 30.0
        total_frames = meta.total_frames
        frame_idx = 0
        last_alert_banner = None
        alert_banner_expiry = 0.0

        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_skip > 1 and (frame_idx % frame_skip != 0):
                    continue

                if max_frames and frame_idx > max_frames:
                    break

                # Video elapsed timestamp
                curr_video_time = frame_idx / video_fps
                formatted_time = seconds_to_hms(curr_video_time)

                # Resize high-resolution frame for fast CPU inference if configured
                orig_h, orig_w = frame.shape[:2]
                scale = 1.0
                if self.video_config.target_width and orig_w > self.video_config.target_width:
                    scale = self.video_config.target_width / float(orig_w)
                    new_w = self.video_config.target_width
                    new_h = int(orig_h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                h, w = frame.shape[:2]

                # Step 1: YOLO Detection + ByteTrack tracking (measured pure inference time)
                t_infer_0 = time.perf_counter()
                detections = self.detector.track(frame, persist=True)
                t_infer_1 = time.perf_counter()
                infer_fps = 1.0 / max(1e-5, (t_infer_1 - t_infer_0))

                # Step 2: Update temporal tracking state
                active_tracks = self.tracker_manager.update(detections, curr_video_time)

                # Step 3: Evaluate Perimeter Intrusion rule
                intrusion_alerts = self.intrusion_detector.evaluate(
                    active_tracks, w, h, curr_video_time, formatted_time
                )

                # Step 4: Evaluate Unattended Object rule with 7-stage diagnostic tracking
                unattended_alerts = self.unattended_detector.evaluate(
                    active_tracks, curr_video_time, formatted_time, detections=detections
                )

                # Aggregate active alert IDs for visualization highlights
                intruder_ids = [a.track_id for a in intrusion_alerts]
                unattended_ids = [a.track_id for a in unattended_alerts]

                # Step 5: Save screenshots and log incidents to SQLite
                for a in intrusion_alerts:
                    snap_path = save_alert_screenshot(
                        frame,
                        self.screenshot_dir,
                        "intrusion",
                        a.track_id,
                        curr_video_time,
                    )
                    record = IncidentRecord(
                        event_category="intrusion",
                        video_name=video_name,
                        video_timestamp=curr_video_time,
                        formatted_timestamp=formatted_time,
                        track_id=a.track_id,
                        object_category=a.object_category,
                        screenshot_path=snap_path,
                        reason=a.explanation,
                        details_json=json.dumps(
                            {"zone": a.zone_name, "point": a.point, "box": a.box}
                        ),
                    )
                    try:
                        self.db.log_incident(record)
                    except Exception as db_err:
                        print(f"Error persisting intrusion incident: {db_err}")

                    last_alert_banner = f"INTRUSION: Person #{a.track_id} in {a.zone_name}"
                    alert_banner_expiry = curr_video_time + 4.0

                for u in unattended_alerts:
                    snap_path = save_alert_screenshot(
                        frame,
                        self.screenshot_dir,
                        "unattended",
                        u.track_id,
                        curr_video_time,
                    )
                    record = IncidentRecord(
                        event_category="unattended_object",
                        video_name=video_name,
                        video_timestamp=curr_video_time,
                        formatted_timestamp=formatted_time,
                        track_id=u.track_id,
                        object_category=u.object_category,
                        screenshot_path=snap_path,
                        reason=u.explanation,
                        details_json=json.dumps(
                            {
                                "stationary_duration": u.stationary_duration,
                                "displacement": u.displacement_px,
                                "closest_person_dist": u.closest_person_dist,
                                "box": u.box,
                            }
                        ),
                    )
                    try:
                        inc_id = self.db.log_incident(record)
                        self.unattended_detector._log_diag(
                            stage="DB_PERSISTENCE",
                            track_id=u.track_id,
                            category=u.object_category,
                            video_time=curr_video_time,
                            formatted_time=formatted_time,
                            message=f"Persisted to SQLite incidents table with ID #{inc_id}",
                            details={"incident_id": inc_id, "db_path": str(self.video_config.db_path)},
                        )
                    except Exception as db_err:
                        self.unattended_detector._log_diag(
                            stage="DB_PERSISTENCE",
                            track_id=u.track_id,
                            category=u.object_category,
                            video_time=curr_video_time,
                            formatted_time=formatted_time,
                            message=f"DB persistence failed: {db_err}",
                            details={"error": str(db_err)},
                        )

                    last_alert_banner = f"Potentially unattended object — review required: {u.object_category.title()} #{u.track_id}"
                    alert_banner_expiry = curr_video_time + 4.0

                # Compute frame rendering FPS
                now = time.time()
                fps = 1.0 / max(1e-5, (now - prev_time))
                prev_time = now

                # Step 6: Render overlays on frame
                annotated = frame.copy()

                # Render polygon zone
                poly_pts = self.intrusion_detector.get_pixel_polygon(w, h)
                is_active_intrusion = any(
                    self.intrusion_detector.track_states.get(t.track_id, {}).get("is_inside", False)
                    for t in active_tracks
                    if t.class_id == 0
                )
                annotated = self.visualizer.draw_polygon_zone(
                    annotated,
                    poly_pts,
                    self.intrusion_config.zone_name,
                    is_active_intrusion=is_active_intrusion,
                )

                # Render tracking trails
                annotated = self.visualizer.draw_tracking_trails(annotated, active_tracks)

                # Candidate diagnostics for decision tracing
                candidate_diags = self.unattended_detector.get_candidate_diagnostics()

                # Render detection boxes, candidate status, and tags
                annotated = self.visualizer.draw_detections(
                    annotated,
                    detections,
                    unattended_track_ids=unattended_ids,
                    intrusion_track_ids=intruder_ids,
                    candidate_diagnostics=candidate_diags,
                )

                # Expire banner
                active_banner = last_alert_banner if curr_video_time <= alert_banner_expiry else None

                # Counts
                person_count = sum(1 for d in detections if d.class_id == 0)
                object_count = len(detections) - person_count

                # Draw surveillance HUD
                annotated = self.visualizer.draw_hud(
                    annotated,
                    formatted_time,
                    person_count,
                    object_count,
                    fps,
                    latest_alert_text=active_banner,
                )

                progress = frame_idx / float(max(1, total_frames))

                yield FrameAnalyticsResult(
                    frame_idx=frame_idx,
                    video_timestamp=curr_video_time,
                    formatted_time=formatted_time,
                    fps=fps,
                    inference_fps=infer_fps,
                    progress_ratio=min(1.0, progress),
                    annotated_frame=annotated,
                    detections=detections,
                    active_tracks=active_tracks,
                    intrusion_alerts=intrusion_alerts,
                    unattended_alerts=unattended_alerts,
                    person_count=person_count,
                    object_count=object_count,
                    candidate_diagnostics=candidate_diags,
                )

        finally:
            cap.release()
