"""
Surveillance visualization module for StreetGuard AI.
Renders CCTV-style overlays: bounding boxes, tracking trails, polygon zones,
proximity guides, and status heads-up displays (HUD).
"""

from typing import Any, List, Optional
import cv2
import numpy as np

from streetguard.config import VideoProcessingConfig
from streetguard.detector import Detection
from streetguard.tracker import TrackState

# Surveillance Color Palette (BGR for OpenCV)
COLOR_PERSON = (245, 180, 50)       # Cyan-Azure
COLOR_BAG = (50, 200, 245)          # Amber-Gold
COLOR_INTRUSION = (50, 50, 235)     # Vivid Red
COLOR_ZONE_NORMAL = (220, 160, 40)  # Calm Cyan/Teal
COLOR_ZONE_ALERT = (30, 30, 240)    # Warning Red
COLOR_DARK_BG = (20, 20, 24)        # Deep Charcoal
COLOR_UNATTENDED = (30, 140, 255)   # Intense Orange


class SurveillanceVisualizer:
    """Renders professional surveillance overlays on OpenCV video frames."""

    def __init__(self, config: Optional[VideoProcessingConfig] = None):
        self.config = config or VideoProcessingConfig()

    def draw_polygon_zone(
        self,
        frame: np.ndarray,
        polygon_pts: np.ndarray,
        zone_name: str,
        is_active_intrusion: bool = False,
    ) -> np.ndarray:
        """
        Draw a semi-transparent filled polygon for the restricted perimeter
        with a border and label.
        """
        if polygon_pts is None or len(polygon_pts) < 3:
            return frame

        overlay = frame.copy()
        color = COLOR_ZONE_ALERT if is_active_intrusion else COLOR_ZONE_NORMAL
        alpha = 0.28 if is_active_intrusion else 0.15

        # Fill polygon on overlay
        cv2.fillPoly(overlay, [polygon_pts], color)
        # Blend overlay with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw crisp outline
        border_thickness = 3 if is_active_intrusion else 2
        cv2.polylines(frame, [polygon_pts], isClosed=True, color=color, thickness=border_thickness)

        # Draw zone label at first polygon vertex
        v0 = polygon_pts[0][0]
        label = f" ZONE: {zone_name.upper()} "
        if is_active_intrusion:
            label += "[INTRUSION ALERT] "

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx, ly = max(10, v0[0]), max(20, v0[1] - 8)
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 6, ly + 4), color, -1)
        cv2.putText(
            frame,
            label,
            (lx + 3, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_DARK_BG if not is_active_intrusion else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_tracking_trails(
        self, frame: np.ndarray, tracks: List[TrackState]
    ) -> np.ndarray:
        """Draw fading historical centroid paths for tracked entities."""
        if not self.config.draw_tracking_trails:
            return frame

        for track in tracks:
            pts = [pt for _, pt, _ in track.history]
            if len(pts) < 2:
                continue

            color = COLOR_PERSON if track.class_id == 0 else COLOR_BAG
            # Take last N points
            recent_pts = pts[-self.config.trail_length :]
            for i in range(1, len(recent_pts)):
                # Progressive thickness and intensity
                thickness = int(np.sqrt(float(i + 1)))
                cv2.line(frame, recent_pts[i - 1], recent_pts[i], color, thickness)

        return frame

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        unattended_track_ids: Optional[List[int]] = None,
        intrusion_track_ids: Optional[List[int]] = None,
        candidate_diagnostics: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes, IDs, class names, and confidence scores."""
        unattended_set = set(unattended_track_ids or [])
        intrusion_set = set(intrusion_track_ids or [])

        # Build candidate lookup by track_id and spatial center
        cand_by_tid = {}
        cand_list = candidate_diagnostics or []
        for c in cand_list:
            if getattr(c, "track_id", None) is not None:
                cand_by_tid[c.track_id] = c

        for det in detections:
            x1, y1, x2, y2 = det.box
            tid = det.track_id
            is_unattended = tid in unattended_set
            is_intruder = tid in intrusion_set

            # Match candidate diagnostic info
            cand = cand_by_tid.get(tid)
            if cand is None and cand_list:
                dcx, dcy = det.center
                for c in cand_list:
                    ccx, ccy = getattr(c, "center", (0, 0))
                    if np.hypot(dcx - ccx, dcy - ccy) <= 30.0:
                        cand = c
                        break

            stat_dur = cand.stationary_duration if cand is not None else 0.0
            is_candidate = (
                cand is not None
                and stat_dur > 0.5
                and det.class_id in (24, 26, 28)
            )

            # Pick box color based on alert state and object category
            if is_intruder:
                box_color = COLOR_INTRUSION
            elif is_unattended or (cand is not None and getattr(cand, "rule_state", "") == "ALERT_ACCEPTED"):
                box_color = COLOR_UNATTENDED
                is_unattended = True
            elif is_candidate:
                box_color = (0, 215, 255)  # Bright Gold
            elif det.class_id == 0:
                box_color = COLOR_PERSON
            else:
                box_color = COLOR_BAG

            # Bounding box rectangle
            border_thick = 3 if is_unattended else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, border_thick)

            # Highlighting corners for surveillance visual flair
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 0:
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 3)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 3)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 3)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 3)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 3)

            # Build label text: Class, ID, Conf, Stationary Duration
            id_str = f"#{tid}" if tid is not None else "---"
            stat_str = f" [{stat_dur:.1f}s]" if stat_dur > 0.5 else ""
            label = f"{det.class_name.upper()} {id_str} {int(det.confidence * 100)}%{stat_str}"
            if is_intruder:
                label = f"! INTRUDER #{tid} !"
            elif is_unattended:
                label = f"! UNATTENDED #{tid} ({int(det.confidence * 100)}% | {stat_dur:.1f}s) !"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # Label background box
            label_y1 = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, label_y1 + th + 6), box_color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, label_y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0) if (not is_intruder and not is_unattended) else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw bottom center footprint dot
            fx, fy = det.footprint
            cv2.circle(frame, (fx, fy), 4, box_color, -1)

            # If unattended, draw alert beacon ring around item
            if is_unattended:
                cx, cy = det.center
                cv2.circle(frame, (cx, cy), max(25, det.width // 2 + 10), COLOR_UNATTENDED, 2)
                sub_label = "Potentially unattended object \u2014 review required."
                (stw, sth), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                sub_x1 = max(5, cx - stw // 2)
                sub_y1 = min(frame.shape[0] - 10, y2 + sth + 12)
                cv2.rectangle(frame, (sub_x1 - 4, sub_y1 - sth - 4), (sub_x1 + stw + 4, sub_y1 + 4), (20, 20, 180), -1)
                cv2.putText(frame, sub_label, (sub_x1, sub_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        formatted_time: str,
        person_count: int,
        object_count: int,
        fps: float,
        latest_alert_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw a modern dark heads-up display (HUD) banner at the top of the frame.
        """
        h, w = frame.shape[:2]
        hud_height = 42

        # Draw HUD bar
        sub_img = frame[0:hud_height, 0:w]
        dark_bar = np.full_like(sub_img, 18)
        cv2.addWeighted(sub_img, 0.25, dark_bar, 0.75, 0, sub_img)
        frame[0:hud_height, 0:w] = sub_img
        cv2.line(frame, (0, hud_height), (w, hud_height), (60, 60, 70), 1)

        # HUD Info Elements
        title = "STREETGUARD AI [LIVE]"
        cv2.putText(frame, title, (14, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        stats = f"TIME: {formatted_time}  |  PERSONS: {person_count}  |  ITEMS: {object_count}  |  {fps:.1f} FPS"
        cv2.putText(frame, stats, (220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Alert banner if active
        if latest_alert_text:
            alert_bar_y = h - 36
            alert_sub = frame[alert_bar_y:h, 0:w]
            red_bar = np.full_like(alert_sub, (20, 20, 160), dtype=np.uint8)
            cv2.addWeighted(alert_sub, 0.3, red_bar, 0.7, 0, alert_sub)
            frame[alert_bar_y:h, 0:w] = alert_sub
            cv2.putText(
                frame,
                f"ALERT: {latest_alert_text}",
                (16, alert_bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
