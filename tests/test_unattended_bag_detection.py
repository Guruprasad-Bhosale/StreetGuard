"""
Comprehensive test suite for Unattended Bag Detection in StreetGuard AI.
Tests all 10 critical validation scenarios:
1. A stationary bag observed for the configured duration.
2. A bag with brief detector dropouts.
3. A bag whose tracking ID changes (track handover/re-anchoring).
4. A moving bag (resets stationary state).
5. A stationary bag with nearby unrelated pedestrians (transient shoppers do not suppress alert).
6. A person who remains near the bag (attendant suppresses alert).
7. A bag that leaves the frame (evicted after grace period).
8. Correct video-timestamp handling (uses video time, not wall clock).
9. Duplicate-alert suppression (cooldown window).
10. Database persistence & forensic snapshot.
"""

from pathlib import Path
import numpy as np
import pytest

from streetguard.config import UnattendedRuleConfig, VideoProcessingConfig
from streetguard.database import IncidentDatabase, IncidentRecord
from streetguard.detector import Detection
from streetguard.rules.unattended import UnattendedObjectDetector
from streetguard.tracker import TrackState


def make_track(tid, cid, cname, center, box, first_seen, last_seen, history_pts=None):
    """Helper to construct realistic TrackState objects for testing."""
    t = TrackState(
        track_id=tid,
        class_id=cid,
        class_name=cname,
        first_seen_time=first_seen,
        last_seen_time=last_seen,
        current_box=box,
        current_center=center,
        current_footprint=(center[0], box[3]),
    )
    if history_pts:
        for ts, pt, bx in history_pts:
            t.history.append((ts, pt, bx))
    else:
        t.history.append((first_seen, center, box))
        if last_seen > first_seen:
            t.history.append((last_seen, center, box))
    return t


def test_1_stationary_bag_observed_for_configured_duration():
    """Scenario 1: A stationary bag observed for >= configured duration triggers alert."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=4.0,
        stationary_displacement_px=30.0,
        nearby_person_radius_px=75.0,
        min_observation_sec=2.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(10, 28, "suitcase", (440, 460), (420, 440, 460, 480), 0.0, 0.0)

    # Frame at t=0.0s
    alerts = detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")
    assert len(alerts) == 0

    # Frame at t=2.0s
    bag.last_seen_time = 2.0
    alerts = detector.evaluate([bag], current_time=2.0, formatted_time="00:00:02")
    assert len(alerts) == 0

    # Frame at t=4.5s (stationary 4.5s >= 4.0s)
    bag.last_seen_time = 4.5
    alerts = detector.evaluate([bag], current_time=4.5, formatted_time="00:00:04")
    assert len(alerts) == 1
    assert alerts[0].track_id == 10
    assert alerts[0].object_category == "suitcase"
    assert alerts[0].stationary_duration >= 4.0
    assert alerts[0].alert_label == "Potentially unattended object — review required."


def test_2_bag_with_brief_detector_dropouts():
    """Scenario 2: Brief detector dropouts within missed_grace_sec do not erase valid history."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=30.0,
        missed_grace_sec=3.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(5, 24, "backpack", (300, 300), (280, 280, 320, 320), 0.0, 1.0)
    detector.evaluate([bag], current_time=1.0, formatted_time="00:00:01")

    # Detector dropout: bag occluded / missed at t=2.0s
    alerts_dropout = detector.evaluate([], current_time=2.0, formatted_time="00:00:02")
    assert len(alerts_dropout) == 0
    # State must be preserved during grace period
    assert 5 in detector.object_states

    # Bag reappears at t=3.5s (3.5s continuous stationary time)
    bag.last_seen_time = 3.5
    alerts = detector.evaluate([bag], current_time=3.5, formatted_time="00:00:03")
    assert len(alerts) == 1
    assert alerts[0].track_id == 5


def test_3_bag_whose_tracking_id_changes():
    """Scenario 3: Object whose ByteTrack tracking ID changes (e.g. #34 -> #228) re-anchors seamlessly."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=30.0,
        missed_grace_sec=3.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    # Initial track #34 at (500, 500) from t=0.0s to t=1.0s
    bag_v1 = make_track(34, 24, "backpack", (500, 500), (480, 480, 520, 520), 0.0, 1.0)
    detector.evaluate([bag_v1], current_time=1.0, formatted_time="00:00:01")

    # Brief gap at t=2.0s
    detector.evaluate([], current_time=2.0, formatted_time="00:00:02")

    # Reappears as track #228 at (502, 501) (displacement ~2.2px <= 30px) at t=3.5s (gap 2.5s <= grace 3.0s)
    bag_v2 = make_track(228, 28, "suitcase", (502, 501), (482, 481, 522, 521), 3.5, 3.5)
    alerts = detector.evaluate([bag_v2], current_time=3.5, formatted_time="00:00:03")

    # Must adopt prior history from track #34: 3.5s - 0.0s = 3.5s >= 3.0s
    assert len(alerts) == 1
    assert alerts[0].track_id == 228
    assert alerts[0].stationary_duration >= 3.0


def test_4_moving_bag_resets_stationary_state():
    """Scenario 4: Moving bag exceeding displacement tolerance resets stationary timer."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=25.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(15, 26, "handbag", (200, 200), (180, 180, 220, 220), 0.0, 2.5)
    detector.evaluate([bag], current_time=2.5, formatted_time="00:00:02")

    # Bag is picked up and moved 90px away to (290, 200) at t=2.8s
    bag.current_center = (290, 200)
    bag.current_box = (270, 180, 310, 220)
    bag.last_seen_time = 2.8
    detector.evaluate([bag], current_time=2.8, formatted_time="00:00:02")

    # At t=4.0s (only 1.2s since movement), alert must be rejected
    bag.last_seen_time = 4.0
    alerts = detector.evaluate([bag], current_time=4.0, formatted_time="00:00:04")
    assert len(alerts) == 0


def test_5_stationary_bag_with_nearby_unrelated_pedestrians():
    """Scenario 5: Unrelated pedestrians transiting past do NOT indefinitely suppress unattended alert."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=30.0,
        nearby_person_radius_px=80.0,
        person_dwell_sec=1.5,
        pedestrian_speed_threshold_px=15.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(20, 28, "suitcase", (400, 400), (380, 380, 420, 420), 0.0, 3.5)
    detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")

    # Transient pedestrian walking past at 50px away: entered zone at t=3.2s, now t=3.5s (dwell=0.3s < 1.5s)
    # Moving with displacement 40px in 1s
    history_ped = [
        (2.5, (310, 400), (290, 320, 330, 480)),
        (3.5, (350, 400), (330, 320, 370, 480)),
    ]
    pedestrian = make_track(101, 0, "person", (350, 400), (330, 320, 370, 480), 2.5, 3.5, history_pts=history_ped)

    bag.last_seen_time = 3.5
    alerts = detector.evaluate([bag, pedestrian], current_time=3.5, formatted_time="00:00:03")

    # Pedestrian is a transient passerby; unattended alert MUST fire!
    assert len(alerts) == 1
    assert alerts[0].track_id == 20


def test_6_person_who_remains_near_bag_suppresses_alert():
    """Scenario 6: Person who remains stationary/lingers near bag is classified as attendant and suppresses alert."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=30.0,
        nearby_person_radius_px=80.0,
        person_dwell_sec=1.5,
        pedestrian_speed_threshold_px=15.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(30, 24, "backpack", (400, 400), (380, 380, 420, 420), 0.0, 4.0)

    # Attendant standing 25px away since t=0.0s (stationary, disp=0, dwell=4.0s >= 1.5s)
    history_attendant = [
        (0.0, (425, 400), (410, 320, 440, 480)),
        (4.0, (425, 400), (410, 320, 440, 480)),
    ]
    attendant = make_track(50, 0, "person", (425, 400), (410, 320, 440, 480), 0.0, 4.0, history_pts=history_attendant)

    detector.evaluate([bag, attendant], current_time=0.0, formatted_time="00:00:00")
    alerts = detector.evaluate([bag, attendant], current_time=4.0, formatted_time="00:00:04")

    # Attendant present suppresses alert
    assert len(alerts) == 0
    # Candidate diagnostics confirm attendant suppression
    candidates = detector.get_candidate_diagnostics()
    assert any(c.rule_state == "ALERT_SUPPRESSED_ATTENDANT" for c in candidates)


def test_7_bag_leaves_frame():
    """Scenario 7: Bag removed/leaving frame is safely evicted after missed_grace_sec expires."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        missed_grace_sec=2.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(40, 24, "backpack", (100, 100), (80, 80, 120, 120), 0.0, 1.0)
    detector.evaluate([bag], current_time=1.0, formatted_time="00:00:01")
    assert 40 in detector.object_states

    # Bag leaves frame: absent at t=3.5s (gap 2.5s > missed_grace_sec 2.0s)
    detector.evaluate([], current_time=3.5, formatted_time="00:00:03")
    assert 40 not in detector.object_states


def test_8_correct_video_timestamp_handling():
    """Scenario 8: Stationary duration calculates using video timestamps, unaffected by execution speed."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=4.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(55, 28, "suitcase", (300, 300), (280, 280, 320, 320), 10.0, 10.0)
    # Starts at video timestamp 10.0s
    detector.evaluate([bag], current_time=10.0, formatted_time="00:00:10")

    # Step to 12.0s in video time (delta = 2.0s < 4.0s)
    bag.last_seen_time = 12.0
    alerts12 = detector.evaluate([bag], current_time=12.0, formatted_time="00:00:12")
    assert len(alerts12) == 0

    # Step to 14.5s in video time (delta = 4.5s >= 4.0s)
    bag.last_seen_time = 14.5
    alerts14 = detector.evaluate([bag], current_time=14.5, formatted_time="00:00:14")
    assert len(alerts14) == 1
    assert alerts14[0].stationary_duration == 4.5
    assert alerts14[0].timestamp == 14.5


def test_9_duplicate_alert_suppression():
    """Scenario 9: Cooldown window suppresses duplicate alerts for the same continuous stationary event."""
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        alert_cooldown_sec=30.0,
        min_observation_sec=1.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = make_track(60, 24, "backpack", (200, 200), (180, 180, 220, 220), 0.0, 0.0)
    detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")

    # Initial alert fires at t=3.5s
    bag.last_seen_time = 3.5
    alerts1 = detector.evaluate([bag], current_time=3.5, formatted_time="00:00:03")
    assert len(alerts1) == 1

    # Next frame at t=4.0s (only 0.5s later; cooldown active)
    bag.last_seen_time = 4.0
    alerts2 = detector.evaluate([bag], current_time=4.0, formatted_time="00:00:04")
    assert len(alerts2) == 0

    # Next frame at t=10.0s (only 6.5s later; cooldown still active)
    bag.last_seen_time = 10.0
    alerts3 = detector.evaluate([bag], current_time=10.0, formatted_time="00:00:10")
    assert len(alerts3) == 0


def test_10_database_persistence(tmp_path: Path):
    """Scenario 10: Persist unattended incident through database with screenshot and query history."""
    db_file = tmp_path / "test_streetguard.db"
    db = IncidentDatabase(db_file)

    rec = IncidentRecord(
        event_category="unattended_object",
        video_name="mall1.mp4",
        video_timestamp=8.7,
        formatted_timestamp="00:00:08",
        track_id=228,
        object_category="suitcase",
        screenshot_path=str(tmp_path / "snap.jpg"),
        reason="Potentially unattended object — review required: Track #228 stationary for 4.2s",
    )

    inc_id = db.log_incident(rec)
    assert inc_id > 0

    # Retrieve logged incidents
    rows = db.get_incidents(category="unattended_object")
    assert len(rows) == 1
    assert rows[0].track_id == 228
    assert rows[0].object_category == "suitcase"
    assert "Potentially unattended object — review required" in rows[0].reason
