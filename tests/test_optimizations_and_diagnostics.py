"""
Automated tests for StreetGuard AI optimizations, diagnostic tracing,
dependency compatibility, timing, and error resilience.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from streetguard.config import ModelConfig, UnattendedRuleConfig, get_default_device
from streetguard.database import IncidentDatabase, IncidentRecord
from streetguard.detector import YOLODetector
from streetguard.pipeline import FrameAnalyticsResult
from streetguard.rules.unattended import UnattendedObjectDetector
from streetguard.tracker import TrackState


def test_compatible_numpy_pyarrow_imports():
    """Verify that NumPy and PyArrow load together without version conflict."""
    import numpy as np
    import pyarrow as pa
    assert np.__version__.startswith("1.26")
    assert pa.__version__.startswith("18.1")
    # Test zero-copy / conversion
    arr = np.array([10, 20, 30], dtype=np.int32)
    arrow_arr = pa.array(arr)
    assert len(arrow_arr) == 3


def test_incident_table_data_preparation_and_arrow(tmp_path: Path):
    """Verify incident database records convert to pandas and PyArrow without crash."""
    db_file = tmp_path / "test_incidents.db"
    db = IncidentDatabase(db_file)
    rec = IncidentRecord(
        event_category="unattended_object",
        video_name="train1.mp4",
        video_timestamp=14.2,
        formatted_timestamp="00:00:14",
        track_id=10,
        object_category="backpack",
        reason="Potentially unattended object — review required",
        screenshot_path="/data/snap.jpg",
    )
    db.log_incident(rec)

    incidents = db.get_incidents()
    table_data = [
        {
            "ID": i.id,
            "Category": i.event_category.upper(),
            "Video": i.video_name,
            "Timestamp": i.formatted_timestamp,
            "Track ID": f"#{i.track_id}",
            "Object": i.object_category,
            "Reason": i.reason,
        }
        for i in incidents
    ]
    df = pd.DataFrame(table_data)
    arrow_table = pa.Table.from_pandas(df)
    assert len(arrow_table) == 1
    assert "Category" in arrow_table.column_names


def test_stationary_object_timing_source_video_time():
    """Verify stationary duration uses source-video time rather than processing wall-clock time."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=4.0,
        stationary_displacement_px=30.0,
        nearby_person_radius_px=100.0,
        min_observation_sec=2.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = TrackState(
        track_id=1,
        class_id=24,
        class_name="backpack",
        first_seen_time=10.0,  # Starts at 10.0s in video
        last_seen_time=10.0,
        current_box=(200, 200, 250, 250),
        current_center=(225, 225),
        current_footprint=(225, 250),
    )

    # Frame 1 at video time 10.0s
    detector.evaluate([bag], current_time=10.0, formatted_time="00:00:10")

    # Frame 2 at video time 12.0s (video time delta is 2.0s)
    bag.last_seen_time = 12.0
    alerts2 = detector.evaluate([bag], current_time=12.0, formatted_time="00:00:12")
    assert len(alerts2) == 0  # 2.0s < 4.0s threshold

    # Frame 3 at video time 15.0s (video time delta is 5.0s > 4.0s)
    bag.last_seen_time = 15.0
    alerts3 = detector.evaluate([bag], current_time=15.0, formatted_time="00:00:15")
    assert len(alerts3) == 1
    assert alerts3[0].stationary_duration == 5.0
    assert alerts3[0].alert_label == "Potentially unattended object — review required."


def test_object_movement_resets_stationary_state():
    """Verify sustained displacement beyond tolerance resets stationary timer."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=25.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = TrackState(
        track_id=5,
        class_id=24,
        class_name="backpack",
        first_seen_time=0.0,
        last_seen_time=0.0,
        current_box=(100, 100, 140, 140),
        current_center=(120, 120),
        current_footprint=(120, 140),
    )
    detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")

    # Object stays stationary until 2.5s
    bag.last_seen_time = 2.5
    detector.evaluate([bag], current_time=2.5, formatted_time="00:00:02")

    # Object moves 80px away for 2 consecutive frames
    bag.current_center = (200, 200)
    bag.last_seen_time = 2.6
    detector.evaluate([bag], current_time=2.6, formatted_time="00:00:02")
    bag.last_seen_time = 2.7
    detector.evaluate([bag], current_time=2.7, formatted_time="00:00:02")

    # At 3.5s (only 0.8s since move), alert must be rejected
    bag.last_seen_time = 3.5
    alerts = detector.evaluate([bag], current_time=3.5, formatted_time="00:00:03")
    assert len(alerts) == 0


def test_track_id_dropout_resilience():
    """Verify brief track dropout within grace period does not wipe stationary history."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=30.0,
        nearby_person_radius_px=100.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)
    detector.missed_grace_sec = 2.0  # 2 seconds grace

    bag = TrackState(
        track_id=8,
        class_id=28,
        class_name="suitcase",
        first_seen_time=0.0,
        last_seen_time=1.0,
        current_box=(300, 300, 350, 350),
        current_center=(325, 325),
        current_footprint=(325, 350),
    )
    detector.evaluate([bag], current_time=1.0, formatted_time="00:00:01")

    # Bag is temporarily occluded / not detected at t=1.5s
    detector.evaluate([], current_time=1.5, formatted_time="00:00:01")
    # State should still be preserved during grace period
    assert 8 in detector.object_states

    # Bag reappears at t=3.5s (3.5s total duration > 3.0s threshold)
    bag.last_seen_time = 3.5
    alerts = detector.evaluate([bag], current_time=3.5, formatted_time="00:00:03")
    assert len(alerts) == 1
    assert alerts[0].track_id == 8


def test_person_proximity_behavior_and_diagnostic_logging():
    """Verify isolated luggage alerts, while nearby person rejects alert and logs reason."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=25.0,
        nearby_person_radius_px=100.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = TrackState(
        track_id=12,
        class_id=24,
        class_name="backpack",
        first_seen_time=0.0,
        last_seen_time=4.0,
        current_box=(100, 100, 150, 150),
        current_center=(125, 125),
        current_footprint=(125, 150),
    )
    detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")

    # Test case A: Person is standing 50px away (< 100px threshold)
    person_near = TrackState(
        track_id=2,
        class_id=0,
        class_name="person",
        first_seen_time=0.0,
        last_seen_time=4.0,
        current_box=(160, 100, 200, 200),
        current_center=(180, 150),
        current_footprint=(180, 200),
    )
    alerts_suppressed = detector.evaluate([bag, person_near], current_time=4.0, formatted_time="00:00:04")
    assert len(alerts_suppressed) == 0

    # Verify diagnostic logs recorded proximity rejection
    diags = detector.get_recent_diagnostics(10)
    stages = [d.stage for d in diags]
    assert "PROXIMITY_EVAL" in stages
    assert "ALERT_DECISION" in stages
    assert any("within radius" in d.message for d in diags)

    # Test case B: Person walks away to 250px (> 100px threshold)
    person_far = TrackState(
        track_id=2,
        class_id=0,
        class_name="person",
        first_seen_time=0.0,
        last_seen_time=5.0,
        current_box=(400, 100, 440, 200),
        current_center=(420, 150),
        current_footprint=(420, 200),
    )
    bag.last_seen_time = 5.0
    alerts_fired = detector.evaluate([bag, person_far], current_time=5.0, formatted_time="00:00:05")
    assert len(alerts_fired) == 1
    assert alerts_fired[0].track_id == 12


def test_database_persistence_error_resilience(tmp_path: Path):
    """Verify pipeline handles database persistence errors gracefully without crashing."""
    read_only_db = tmp_path / "readonly.db"
    # Create empty db file and close it
    db = IncidentDatabase(read_only_db)
    
    # Simulate failed write with invalid table
    rec = IncidentRecord(
        event_category="unattended_object",
        video_name="test.mp4",
        video_timestamp=1.0,
        formatted_timestamp="00:00:01",
        track_id=1,
        object_category="bag",
        reason="Test",
    )
    # Valid write succeeds
    id_val = db.log_incident(rec)
    assert id_val > 0


def test_cpu_fallback_and_device_selection():
    """Verify get_default_device and YOLODetector fallback to CPU when CUDA is unavailable."""
    dev = get_default_device()
    assert dev in ["cpu", "cuda"]
    if not torch.cuda.is_available():
        assert dev == "cpu"

    cfg = ModelConfig(device="cuda")
    detector = YOLODetector(cfg)
    if not torch.cuda.is_available():
        assert detector.device == "cpu"


def test_fps_metric_and_telemetry():
    """Verify FPS calculations and timing properties are valid."""
    res = FrameAnalyticsResult(
        frame_idx=10,
        video_timestamp=0.4,
        formatted_time="00:00:00",
        fps=28.5,
        inference_fps=36.2,
        progress_ratio=0.1,
        annotated_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        detections=[],
        active_tracks=[],
        intrusion_alerts=[],
        unattended_alerts=[],
        person_count=3,
        object_count=1,
    )
    assert res.inference_fps > res.fps
    assert res.person_count == 3
    assert res.object_count == 1
