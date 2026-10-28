"""
Unattended-object detection rule for StreetGuard AI.
Implements an explainable rule-based surveillance heuristic monitoring stationary luggage/items
and evaluating the spatial proximity of tracked persons over time.
Includes full 7-stage diagnostic decision tracing, dropout resilience, transient pedestrian filtering,
and false-positive caveats.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from streetguard.config import UnattendedRuleConfig
from streetguard.detector import Detection
from streetguard.tracker import TrackState

logger = logging.getLogger("streetguard.unattended")


@dataclass
class DiagnosticEvent:
    """Represents a diagnostic milestone in the unattended decision pipeline."""
    stage: str  # 'OBJECT_DETECTED', 'OBJECT_TRACKED', 'STATIONARY_TIMER', 'STATIONARY_CONFIRMED', 'PROXIMITY_EVAL', 'ALERT_DECISION', 'DB_PERSISTENCE'
    track_id: Optional[int]
    object_category: str
    video_timestamp: float
    formatted_time: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    wall_clock_time: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3])


@dataclass
class CandidateDiagnostic:
    """Detailed diagnostics for an individual candidate object."""
    candidate_key: str
    track_id: Optional[int]
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    first_seen_time: float
    stationary_duration: float
    movement_status: str
    nearby_person_status: str
    rule_state: str  # 'MONITORING', 'STATIONARY_CONFIRMED', 'ALERT_ACCEPTED', 'ALERT_SUPPRESSED_COOLDOWN', 'ALERT_SUPPRESSED_ATTENDANT', 'INSUFFICIENT_OBSERVATION'
    decision_reason: str
    formatted_time: str = ""


@dataclass
class UnattendedAlert:
    """Represents a review-required unattended luggage event."""
    track_id: int
    object_category: str
    timestamp: float
    formatted_time: str
    stationary_duration: float
    displacement_px: float
    closest_person_dist: Optional[float]
    closest_person_id: Optional[int]
    point: Tuple[int, int]
    box: Tuple[int, int, int, int]
    explanation: str
    alert_label: str = "Potentially unattended object — review required."
    heuristic_disclaimer: str = (
        "HEURISTIC NOTICE: YOLO and ByteTrack identify object location and tracking continuity, "
        "not legal ownership or intent. Occlusion, static clutter, or distant owners out of camera FOV "
        "may generate false alerts. Human operator review required. "
        "Proximity does not prove ownership, and the absence of a nearby person does not prove abandonment."
    )


class UnattendedObjectDetector:
    """
    Evaluates tracked luggage items (backpacks, suitcases, handbags) for:
    1. Detection of supported luggage classes.
    2. Spatial stillness (displacement <= threshold over duration) with jitter tolerance.
    3. Spatial anchoring resilient to tracking ID changes and brief detection dropouts.
    4. Person association evaluating proximity and temporal dwell (distinguishing transient passersby from attendants).
    5. Adequate observation history before raising alert.
    6. Structured 7-stage diagnostic logging and per-candidate decision tracing.
    """

    def __init__(self, config: Optional[UnattendedRuleConfig] = None):
        self.config = config or UnattendedRuleConfig()
        # Per-object state:
        # track_id or candidate_id -> {
        #   'stationary_start_time': float,
        #   'anchor_position': (cx, cy),
        #   'last_seen_time': float,
        #   'first_seen_time': float,
        #   'alerted': bool,
        #   'last_alert_time': float,
        #   'recent_positions': deque of (cx, cy),
        #   'person_near_timers': Dict[int, Dict[str, float]],
        #   'object_category': str,
        #   'box': Tuple[int, int, int, int],
        #   'confidence': float,
        #   'rule_state': str,
        #   'decision_reason': str,
        #   'movement_status': str,
        #   'nearby_person_status': str,
        # }
        self.object_states: Dict[Any, Dict[str, Any]] = {}
        # Ring buffer of recent diagnostic events for UI tracing and viva inspection
        self.diagnostics: deque[DiagnosticEvent] = deque(maxlen=200)
        # Snapshot of candidate diagnostics from the most recent evaluation
        self.candidate_diagnostics: Dict[Any, CandidateDiagnostic] = {}
        # Dropout grace period in seconds before evicting stationary state
        self.missed_grace_sec: float = getattr(self.config, "missed_grace_sec", 3.0)

    def _log_diag(
        self,
        stage: str,
        track_id: Optional[int],
        category: str,
        video_time: float,
        formatted_time: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record diagnostic event to internal ring buffer and standard logger."""
        event = DiagnosticEvent(
            stage=stage,
            track_id=track_id,
            object_category=category,
            video_timestamp=round(video_time, 2),
            formatted_time=formatted_time,
            message=message,
            details=details or {},
        )
        self.diagnostics.append(event)
        logger.debug(f"[{stage}] Track #{track_id} ({category}) @ {formatted_time}: {message}")

    def get_recent_diagnostics(self, limit: int = 50) -> List[DiagnosticEvent]:
        """Retrieve recent diagnostic events for dashboard and debugging."""
        return list(self.diagnostics)[-limit:]

    def get_candidate_diagnostics(self) -> List[CandidateDiagnostic]:
        """Retrieve decision diagnostics for all active candidate objects."""
        return list(self.candidate_diagnostics.values())

    def _compute_min_person_distance(
        self,
        object_center: Tuple[int, int],
        person_tracks: List[TrackState],
        object_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[Optional[float], Optional[int]]:
        """Find the Euclidean/ground perspective distance and track ID of the closest detected person."""
        if not person_tracks:
            return None, None

        min_dist = float("inf")
        closest_id = None
        obj_foot = (object_center[0], object_box[3]) if object_box is not None else object_center
        k_persp = getattr(self.config, "perspective_vertical_scale", 1.5)

        for p_track in person_tracks:
            p_center = np.array(p_track.current_center, dtype=float)
            p_foot = np.array(p_track.current_footprint, dtype=float)

            # Ground plane perspective distance & centroid distance
            dx = float(p_foot[0] - obj_foot[0])
            dy = float(p_foot[1] - obj_foot[1])
            d_ground = float(np.hypot(dx, k_persp * dy))
            d_center = float(np.hypot(p_center[0] - object_center[0], p_center[1] - object_center[1]))
            d = min(d_ground, d_center)

            if d < min_dist:
                min_dist = d
                closest_id = p_track.track_id

        return (min_dist if min_dist != float("inf") else None, closest_id)

    def evaluate(
        self,
        tracks: List[TrackState],
        current_time: float,
        formatted_time: str,
        detections: Optional[List[Detection]] = None,
    ) -> List[UnattendedAlert]:
        """
        Evaluate active tracks and luggage detections for potential unattended items.
        Executes and logs all 7 diagnostic verification stages with explainable logic.
        """
        if not self.config.enabled:
            return []

        alerts: List[UnattendedAlert] = []
        self.candidate_diagnostics.clear()

        # Stage 1: Object Detected logging (from raw detections if provided)
        if detections:
            for det in detections:
                if (
                    det.class_id in self.config.target_classes
                    or det.class_name.lower() in ["backpack", "handbag", "suitcase"]
                ):
                    self._log_diag(
                        stage="OBJECT_DETECTED",
                        track_id=det.track_id,
                        category=det.class_name,
                        video_time=current_time,
                        formatted_time=formatted_time,
                        message=f"Visual detection of {det.class_name} with confidence {det.confidence:.2f}",
                        details={"confidence": round(det.confidence, 3), "box": det.box},
                    )

        # Separate persons from candidate unattended luggage objects
        person_tracks = [
            t for t in tracks
            if t.class_id == 0 or t.class_name.lower() == "person"
        ]

        # Gather candidate luggage items from tracks AND raw detections
        # This guarantees luggage detections reach the rule evaluator even if ByteTrack has not assigned an ID
        candidate_items = []
        covered_centers = []

        # 1. From active luggage tracks
        for t in tracks:
            if (
                t.class_id in self.config.target_classes
                or t.class_name.lower() in ["backpack", "handbag", "suitcase"]
            ):
                candidate_items.append({
                    "track_id": t.track_id,
                    "class_id": t.class_id,
                    "class_name": t.class_name,
                    "confidence": 0.5,
                    "box": t.current_box,
                    "center": t.current_center,
                    "footprint": t.current_footprint,
                    "first_seen_time": t.first_seen_time,
                    "obs_duration": t.observation_duration(),
                    "frames_tracked": t.frames_tracked,
                })
                covered_centers.append(t.current_center)

        # 2. From raw detections (e.g. untracked detections or newly appeared items)
        if detections:
            for det in detections:
                if (
                    det.class_id in self.config.target_classes
                    or det.class_name.lower() in ["backpack", "handbag", "suitcase"]
                ):
                    # Check if already covered by an existing track
                    cx, cy = det.center
                    matched = False
                    for item in candidate_items:
                        if det.track_id is not None and item["track_id"] == det.track_id:
                            item["confidence"] = det.confidence
                            matched = True
                            break
                        icx, icy = item["center"]
                        if np.hypot(cx - icx, cy - icy) <= 20.0:
                            item["confidence"] = max(item["confidence"], det.confidence)
                            if item["track_id"] is None and det.track_id is not None:
                                item["track_id"] = det.track_id
                            matched = True
                            break

                    if not matched:
                        candidate_items.append({
                            "track_id": det.track_id,
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                            "box": det.box,
                            "center": det.center,
                            "footprint": det.footprint,
                            "first_seen_time": current_time,
                            "obs_duration": 0.0,
                            "frames_tracked": 1,
                        })
                        covered_centers.append(det.center)

        active_state_keys = set()

        for item in candidate_items:
            tid = item["track_id"]
            cname = item["class_name"]
            conf = item["confidence"]
            cx, cy = item["center"]
            box = item["box"]

            # Match to existing state:
            # 1. Direct match by track_id if tid is present
            matched_key = None
            if tid is not None and tid in self.object_states:
                matched_key = tid
            else:
                # 2. Spatial anchor match: find closest existing stationary anchor within tolerance
                best_dist = float("inf")
                best_key = None
                for ex_key, ex_st in self.object_states.items():
                    if ex_key not in active_state_keys:
                        anc_x, anc_y = ex_st["anchor_position"]
                        dist = float(np.hypot(cx - anc_x, cy - anc_y))
                        time_gap = current_time - ex_st["last_seen_time"]
                        if dist <= self.config.stationary_displacement_px and time_gap <= self.missed_grace_sec:
                            if dist < best_dist:
                                best_dist = dist
                                best_key = ex_key

                if best_key is not None:
                    matched_key = best_key
                    # Track handover or track ID migration
                    if tid is not None and matched_key != tid:
                        self.object_states[tid] = self.object_states.pop(matched_key)
                        self._log_diag(
                            stage="STATIONARY_TIMER",
                            track_id=tid,
                            category=cname,
                            video_time=current_time,
                            formatted_time=formatted_time,
                            message=f"Re-anchored track #{tid} to prior stationary luggage #{matched_key} (disp: {best_dist:.1f}px)",
                            details={"prior_track_id": matched_key, "displacement": round(best_dist, 1)},
                        )
                        matched_key = tid

            # Use track_id or generate spatial candidate key
            effective_key = matched_key if matched_key is not None else (tid if tid is not None else f"cand_{cx}_{cy}")
            active_state_keys.add(effective_key)

            if effective_key not in self.object_states:
                # Brand new candidate object
                self.object_states[effective_key] = {
                    "stationary_start_time": item["first_seen_time"],
                    "anchor_position": (cx, cy),
                    "last_seen_time": current_time,
                    "first_seen_time": item["first_seen_time"],
                    "alerted": False,
                    "last_alert_time": -999.0,
                    "consecutive_move_frames": 0,
                    "recent_positions": deque([(cx, cy)], maxlen=10),
                    "person_near_timers": {},
                    "object_category": cname,
                    "box": box,
                    "confidence": conf,
                    "track_id": tid,
                    "rule_state": "MONITORING",
                    "decision_reason": "New object candidate under observation",
                    "movement_status": "Stationary",
                    "nearby_person_status": "Evaluating proximity",
                }

            state = self.object_states[effective_key]
            state["last_seen_time"] = current_time
            state["box"] = box
            state["confidence"] = conf
            state["object_category"] = cname
            state["track_id"] = tid if tid is not None else state.get("track_id")
            state["recent_positions"].append((cx, cy))

            # Stage 2: Object Tracked
            obs_duration = current_time - state["first_seen_time"]
            display_tid = state.get("track_id")
            self._log_diag(
                stage="OBJECT_TRACKED",
                track_id=display_tid,
                category=cname,
                video_time=current_time,
                formatted_time=formatted_time,
                message=f"Object {cname} (ID: {display_tid}) active for {obs_duration:.2f}s",
                details={"track_id": display_tid, "obs_duration": round(obs_duration, 2)},
            )

            # Stage 3: Stationary displacement check
            anchor = np.array(state["anchor_position"], dtype=float)
            curr = np.array((cx, cy), dtype=float)
            curr_displacement = float(np.linalg.norm(curr - anchor))

            if curr_displacement > self.config.stationary_displacement_px:
                # Moving object resets stationary state
                state["anchor_position"] = (cx, cy)
                state["stationary_start_time"] = current_time
                state["alerted"] = False
                state["recent_positions"].clear()
                state["recent_positions"].append((cx, cy))
                state["movement_status"] = f"Moved ({curr_displacement:.1f}px > {self.config.stationary_displacement_px}px)"
                state["rule_state"] = "MONITORING"
                state["decision_reason"] = f"Object moved ({curr_displacement:.1f}px > {self.config.stationary_displacement_px}px). Resetting stationary timer."

                self._log_diag(
                    stage="STATIONARY_TIMER",
                    track_id=display_tid,
                    category=cname,
                    video_time=current_time,
                    formatted_time=formatted_time,
                    message=state["decision_reason"],
                    details={"displacement": round(curr_displacement, 1), "threshold": self.config.stationary_displacement_px},
                )
                self.object_states[effective_key] = state
                self._record_candidate_diagnostic(effective_key, state, current_time, formatted_time, obs_duration, 0.0)
                continue

            # Object is stationary within tolerance
            stationary_duration = current_time - state["stationary_start_time"]
            state["movement_status"] = f"Stationary (disp: {curr_displacement:.1f}px <= {self.config.stationary_displacement_px}px)"

            self._log_diag(
                stage="STATIONARY_TIMER",
                track_id=display_tid,
                category=cname,
                video_time=current_time,
                formatted_time=formatted_time,
                message=f"Stationary for {stationary_duration:.1f}s (disp: {curr_displacement:.1f}px)",
                details={"stationary_duration": round(stationary_duration, 2), "displacement": round(curr_displacement, 1)},
            )

            # Stage 4: Stationary threshold reached
            if stationary_duration < self.config.stationary_duration_sec:
                state["rule_state"] = "MONITORING"
                state["decision_reason"] = f"Stationary for {stationary_duration:.1f}s < threshold {self.config.stationary_duration_sec:.1f}s"
                self.object_states[effective_key] = state
                self._record_candidate_diagnostic(effective_key, state, current_time, formatted_time, obs_duration, stationary_duration)
                continue

            # Stationary duration threshold satisfied
            self._log_diag(
                stage="STATIONARY_CONFIRMED",
                track_id=display_tid,
                category=cname,
                video_time=current_time,
                formatted_time=formatted_time,
                message=f"Stationary threshold met: {stationary_duration:.1f}s >= {self.config.stationary_duration_sec}s",
                details={"stationary_duration": round(stationary_duration, 2)},
            )

            # Stage 5: Person proximity & temporal association evaluation
            min_dist, closest_pid = self._compute_min_person_distance((cx, cy), person_tracks, object_box=box)

            if "person_near_timers" not in state:
                state["person_near_timers"] = {}

            is_attended = False
            attendant_id = None
            attendant_dwell = 0.0
            transient_pedestrians = []
            active_near_pids = set()

            person_dwell_threshold = getattr(self.config, "person_dwell_sec", 1.5)
            ped_speed_threshold = getattr(self.config, "pedestrian_speed_threshold_px", 15.0)
            k_persp = getattr(self.config, "perspective_vertical_scale", 1.5)
            obj_foot = (cx, box[3])

            if min_dist is not None and min_dist <= self.config.nearby_person_radius_px:
                for p_track in person_tracks:
                    p_center = np.array(p_track.current_center, dtype=float)
                    p_foot = np.array(p_track.current_footprint, dtype=float)

                    dx = float(p_foot[0] - obj_foot[0])
                    dy = float(p_foot[1] - obj_foot[1])
                    d_p = min(float(np.hypot(dx, k_persp * dy)), float(np.hypot(p_center[0] - cx, p_center[1] - cy)))

                    if d_p <= self.config.nearby_person_radius_px:
                        pid = p_track.track_id
                        active_near_pids.add(pid)

                        if pid not in state["person_near_timers"]:
                            # Check if person has already been observed stationary at this spot
                            p_obs = p_track.observation_duration()
                            p_disp = p_track.get_displacement_over_window(
                                duration_sec=min(p_obs, 2.0), current_time=current_time
                            )
                            if p_obs >= person_dwell_threshold and p_disp < ped_speed_threshold:
                                entry_time = p_track.first_seen_time
                            else:
                                entry_time = current_time

                            state["person_near_timers"][pid] = {
                                "first_near": entry_time,
                                "last_near": current_time,
                            }
                        else:
                            state["person_near_timers"][pid]["last_near"] = current_time

                        dwell = current_time - state["person_near_timers"][pid]["first_near"]
                        p_disp_recent = p_track.get_displacement_over_window(
                            duration_sec=min(1.0, max(0.1, p_track.observation_duration())),
                            current_time=current_time,
                        )

                        # Classification: Attendant (lingering or stationary) vs Transient Pedestrian (walking past)
                        is_stationary_near = p_disp_recent < ped_speed_threshold
                        if dwell >= person_dwell_threshold or (is_stationary_near and p_track.observation_duration() >= person_dwell_threshold):
                            is_attended = True
                            attendant_id = pid
                            attendant_dwell = dwell
                            break
                        else:
                            transient_pedestrians.append((pid, d_p, dwell, p_disp_recent))

            # Evict expired person dwell timers
            stale_pids = [
                pid for pid, info in state["person_near_timers"].items()
                if pid not in active_near_pids and (current_time - info["last_near"]) > 1.0
            ]
            for pid in stale_pids:
                del state["person_near_timers"][pid]

            # Stage 6: Alert decision
            if is_attended:
                # Persistent attendant present near object
                state["alerted"] = False
                state["rule_state"] = "ALERT_SUPPRESSED_ATTENDANT"
                state["nearby_person_status"] = f"Attendant #{attendant_id} lingering at {min_dist:.1f}px (dwell {attendant_dwell:.1f}s >= {person_dwell_threshold:.1f}s)"
                state["decision_reason"] = f"Alert rejected: Attendant #{attendant_id} within radius ({min_dist:.1f}px <= {self.config.nearby_person_radius_px}px)"

                self._log_diag(
                    stage="PROXIMITY_EVAL",
                    track_id=display_tid,
                    category=cname,
                    video_time=current_time,
                    formatted_time=formatted_time,
                    message=f"Proximity result: Attendant #{attendant_id} lingering ({min_dist:.1f}px <= {self.config.nearby_person_radius_px}px, dwell: {attendant_dwell:.1f}s)",
                    details={"attendant_id": attendant_id, "min_dist": round(min_dist, 1), "dwell": round(attendant_dwell, 1)},
                )
                self._log_diag(
                    stage="ALERT_DECISION",
                    track_id=display_tid,
                    category=cname,
                    video_time=current_time,
                    formatted_time=formatted_time,
                    message=state["decision_reason"],
                    details={"closest_person_id": attendant_id, "min_dist": round(min_dist, 1)},
                )
            else:
                # No attendant present (either completely isolated or transient pedestrian walking past)
                if transient_pedestrians:
                    t_pid, t_dist, t_dwell, t_disp = transient_pedestrians[0]
                    state["nearby_person_status"] = f"Transient pedestrian #{t_pid} passing at {t_dist:.1f}px (dwell {t_dwell:.1f}s < {person_dwell_threshold:.1f}s)"
                    self._log_diag(
                        stage="PROXIMITY_EVAL",
                        track_id=display_tid,
                        category=cname,
                        video_time=current_time,
                        formatted_time=formatted_time,
                        message=f"Transient pedestrian #{t_pid} at {t_dist:.1f}px (walking past, dwell {t_dwell:.1f}s < {person_dwell_threshold:.1f}s) — does not suppress alert",
                        details={"transient_person_id": t_pid, "dist": round(t_dist, 1), "dwell": round(t_dwell, 1)},
                    )
                else:
                    state["nearby_person_status"] = f"Isolated ({min_dist:.1f}px to nearest person)" if min_dist is not None else "Isolated (no persons in frame)"
                    self._log_diag(
                        stage="PROXIMITY_EVAL",
                        track_id=display_tid,
                        category=cname,
                        video_time=current_time,
                        formatted_time=formatted_time,
                        message=f"Proximity result: {state['nearby_person_status']}. Isolated: True",
                        details={"min_dist": round(min_dist, 1) if min_dist is not None else None, "person_is_nearby": False},
                    )

                time_since_last_alert = current_time - state["last_alert_time"]
                can_alert = (not state["alerted"]) or (time_since_last_alert >= self.config.alert_cooldown_sec)

                if not can_alert:
                    state["rule_state"] = "ALERT_SUPPRESSED_COOLDOWN"
                    state["decision_reason"] = f"Alert suppressed: Cooldown active ({time_since_last_alert:.1f}s < {self.config.alert_cooldown_sec}s)"
                    self._log_diag(
                        stage="ALERT_DECISION",
                        track_id=display_tid,
                        category=cname,
                        video_time=current_time,
                        formatted_time=formatted_time,
                        message=state["decision_reason"],
                        details={"cooldown_remaining": round(self.config.alert_cooldown_sec - time_since_last_alert, 1)},
                    )
                elif obs_duration < self.config.min_observation_sec:
                    state["rule_state"] = "INSUFFICIENT_OBSERVATION"
                    state["decision_reason"] = f"Alert rejected: Insufficient observation time ({obs_duration:.1f}s < {self.config.min_observation_sec}s)"
                    self._log_diag(
                        stage="ALERT_DECISION",
                        track_id=display_tid,
                        category=cname,
                        video_time=current_time,
                        formatted_time=formatted_time,
                        message=state["decision_reason"],
                        details={"obs_duration": round(obs_duration, 2)},
                    )
                else:
                    # Alert accepted!
                    state["rule_state"] = "ALERT_ACCEPTED"
                    state["alerted"] = True
                    state["last_alert_time"] = current_time

                    closest_info = (
                        f"Nearest person ID #{closest_pid} is {min_dist:.1f}px away"
                        if min_dist is not None
                        else "No persons currently detected in frame"
                    )
                    explanation = (
                        f"Potentially unattended object — review required: Unattended Object Detected. Track #{display_tid} ({cname}) "
                        f"has remained stationary for {stationary_duration:.1f}s "
                        f"(movement {curr_displacement:.1f}px <= {self.config.stationary_displacement_px}px). "
                        f"{closest_info} (threshold: {self.config.nearby_person_radius_px}px)."
                    )
                    state["decision_reason"] = explanation

                    alert = UnattendedAlert(
                        track_id=display_tid if display_tid is not None else -1,
                        object_category=cname,
                        timestamp=current_time,
                        formatted_time=formatted_time,
                        stationary_duration=stationary_duration,
                        displacement_px=curr_displacement,
                        closest_person_dist=min_dist,
                        closest_person_id=closest_pid,
                        point=(cx, cy),
                        box=box,
                        explanation=explanation,
                    )
                    alerts.append(alert)

                    self._log_diag(
                        stage="ALERT_DECISION",
                        track_id=display_tid,
                        category=cname,
                        video_time=current_time,
                        formatted_time=formatted_time,
                        message=f"Alert ACCEPTED: {explanation}",
                        details={"alert_label": alert.alert_label},
                    )

            self.object_states[effective_key] = state
            self._record_candidate_diagnostic(effective_key, state, current_time, formatted_time, obs_duration, stationary_duration)

        # Evict stale states after grace period expires
        stale_keys = [
            k for k, v in self.object_states.items()
            if (k not in active_state_keys) and ((current_time - v["last_seen_time"]) > self.missed_grace_sec)
        ]
        for k in stale_keys:
            cat = self.object_states[k].get("object_category", "luggage")
            tid_k = self.object_states[k].get("track_id", k)
            self._log_diag(
                stage="OBJECT_TRACKED",
                track_id=tid_k if isinstance(tid_k, int) else None,
                category=cat,
                video_time=current_time,
                formatted_time=formatted_time,
                message=f"Track #{tid_k} evicted after {self.missed_grace_sec}s absence",
            )
            del self.object_states[k]

        return alerts

    def _record_candidate_diagnostic(
        self,
        key: Any,
        state: Dict[str, Any],
        current_time: float,
        formatted_time: str,
        obs_duration: float,
        stationary_duration: float,
    ) -> None:
        """Store structured decision telemetry for a candidate luggage object."""
        self.candidate_diagnostics[str(key)] = CandidateDiagnostic(
            candidate_key=str(key),
            track_id=state.get("track_id"),
            class_name=state.get("object_category", "luggage"),
            confidence=round(state.get("confidence", 0.0), 3),
            box=state.get("box", (0, 0, 0, 0)),
            center=state.get("anchor_position", (0, 0)),
            first_seen_time=round(state.get("first_seen_time", current_time), 2),
            stationary_duration=round(stationary_duration, 2),
            movement_status=state.get("movement_status", "Stationary"),
            nearby_person_status=state.get("nearby_person_status", "Isolated"),
            rule_state=state.get("rule_state", "MONITORING"),
            decision_reason=state.get("decision_reason", ""),
            formatted_time=formatted_time,
        )

    def reset(self) -> None:
        """Clear all unattended object states and diagnostics."""
        self.object_states.clear()
        self.diagnostics.clear()
        self.candidate_diagnostics.clear()
