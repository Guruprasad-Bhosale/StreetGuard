"""
Diagnostic script for inspecting CCTV mall footage (mall1.mp4 vs mall2.mp4).
Analyzes detection, tracking continuity, class labels, confidences,
bounding box coordinates, tracking IDs, and person proximity.
Saves annotated diagnostic evidence frames to data/diagnostics/.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streetguard.config import (
    ModelConfig,
    UnattendedRuleConfig,
    VideoProcessingConfig,
)
from streetguard.pipeline import VideoAnalyticsPipeline


def run_diagnostics():
    out_dir = Path("data/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STREETGUARD AI — SURVEILLANCE CCTV FOOTAGE DIAGNOSTIC REPORT")
    print("=" * 80)

    # 1. Inspect Video Metadata
    videos = ["sample_data/mall1.mp4", "sample_data/mall2.mp4"]
    for vpath in videos:
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = total / fps if fps > 0 else 0
        cap.release()
        print(f"\n[VIDEO METADATA] {vpath}:")
        print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Frame Count: {total} | Duration: {dur:.2f}s")

    # 2. Run Pipeline on mall1.mp4 and log all detections of representative frames
    print("\n" + "=" * 80)
    print("ANALYSIS OF sample_data/mall1.mp4 (TARGET FOOTAGE CONTAINING FLOOR LUGGAGE)")
    print("=" * 80)

    m_cfg = ModelConfig(
        confidence_threshold=0.35,
        luggage_confidence_threshold=0.20,
    )
    u_cfg = UnattendedRuleConfig(
        enabled=True,
        stationary_duration_sec=4.0,
        nearby_person_radius_px=75.0,
        person_dwell_sec=1.5,
    )
    v_cfg = VideoProcessingConfig(target_width=960)

    pipeline = VideoAnalyticsPipeline(
        model_config=m_cfg,
        unattended_config=u_cfg,
        video_config=v_cfg,
    )

    representative_frames = [14, 72, 120, 166, 209, 235]
    saved_frames = {}

    alert_records = []

    for res in pipeline.process_video_stream("sample_data/mall1.mp4"):
        if res.unattended_alerts:
            for a in res.unattended_alerts:
                alert_records.append((res.frame_idx, res.formatted_time, a))

        if res.frame_idx in representative_frames:
            saved_frames[res.frame_idx] = res
            print(f"\n--- Representative Frame {res.frame_idx} (Time: {res.formatted_time}) ---")
            print("  Detections:")
            for d in res.detections:
                is_bag = d.class_id in (24, 26, 28)
                prefix = "  [LUGGAGE] " if is_bag else "  [PERSON]  "
                track_id_str = f"Track #{d.track_id}" if d.track_id is not None else "No Track ID (Unassigned)"
                thresh = m_cfg.luggage_confidence_threshold if is_bag else m_cfg.confidence_threshold
                print(
                    f"{prefix}Class: {d.class_name} (ID: {d.class_id}) | Conf: {d.confidence:.3f} (Thresh: {thresh}) | "
                    f"Box: {d.box} | {track_id_str}"
                )

            # Check candidate diagnostics
            cand_diags = pipeline.unattended_detector.get_candidate_diagnostics()
            if cand_diags:
                print("  Candidate Heuristic States:")
                for c in cand_diags:
                    print(
                        f"    Candidate #{c.track_id} ({c.class_name}): State={c.rule_state} | "
                        f"Stationary={c.stationary_duration:.1f}s | Proximity={c.nearby_person_status} | "
                        f"Decision={c.decision_reason}"
                    )

            # Save diagnostic annotated frame
            out_path = out_dir / f"mall1_diag_frame_{res.frame_idx:03d}_{int(res.video_timestamp)}s.jpg"
            cv2.imwrite(str(out_path), res.annotated_frame)
            print(f"  Saved diagnostic frame: {out_path}")

    print("\n" + "=" * 80)
    print("MALL1 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total alerts triggered: {len(alert_records)}")
    for f_idx, t_str, alert in alert_records:
        print(f"  Frame {f_idx} ({t_str}): {alert.explanation}")

    # 3. Quick comparison with mall2.mp4
    print("\n" + "=" * 80)
    print("COMPARATIVE AUDIT OF sample_data/mall2.mp4")
    print("=" * 80)
    pipeline_m2 = VideoAnalyticsPipeline(
        model_config=m_cfg,
        unattended_config=u_cfg,
        video_config=v_cfg,
    )
    m2_alerts = []
    for res in pipeline_m2.process_video_stream("sample_data/mall2.mp4", frame_skip=2):
        if res.unattended_alerts:
            m2_alerts.extend(res.unattended_alerts)
    print(f"Total unattended alerts in mall2.mp4: {len(m2_alerts)}")
    print(
        "Observation on mall2.mp4: No stationary luggage on floor. Handbags and backpacks detected "
        "are carried by walking shoppers whose positions move continuously."
    )
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()
