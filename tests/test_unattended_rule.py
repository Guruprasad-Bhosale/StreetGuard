"""
Tests for Unattended object heuristic rule:
- Stationary object duration calculation
- Nearby person proximity check (person close suppresses alert, person distant allows alert)
- Observation window requirement
- Movement resetting the stationary timer
"""

from streetguard.config import UnattendedRuleConfig
from streetguard.rules.unattended import UnattendedObjectDetector
from streetguard.tracker import TrackState


def create_track(tid, cid, cname, center, box, first_seen, last_seen):
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
    t.history.append((first_seen, center, box))
    t.history.append((last_seen, center, box))
    return t


def test_unattended_alert_triggered_when_isolated_and_stationary():
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=20.0,
        nearby_person_radius_px=100.0,
        min_observation_sec=2.0,
    )
    detector = UnattendedObjectDetector(config)

    # Backpack stationary at (300, 300) since t=0.0
    bag = create_track(
        tid=10,
        cid=24,
        cname="backpack",
        center=(300, 300),
        box=(280, 280, 320, 320),
        first_seen=0.0,
        last_seen=0.0,
    )

    # Observation at t=0.0 (initial frame)
    alerts0 = detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")
    assert len(alerts0) == 0  # Needs stationary duration and min observation

    # Observation at t=1.0 (still under stationary duration)
    bag.last_seen_time = 1.0
    alerts1 = detector.evaluate([bag], current_time=1.0, formatted_time="00:00:01")
    assert len(alerts1) == 0

    # Observation at t=4.0 (stationary for 4.0s > 3.0s threshold, no person nearby)
    bag.last_seen_time = 4.0
    alerts4 = detector.evaluate([bag], current_time=4.0, formatted_time="00:00:04")
    assert len(alerts4) == 1
    alert = alerts4[0]
    assert alert.track_id == 10
    assert alert.object_category == "backpack"
    assert alert.stationary_duration >= 3.0
    assert "Unattended Object Detected" in alert.explanation


def test_nearby_person_suppresses_unattended_alert():
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=20.0,
        nearby_person_radius_px=150.0,
        min_observation_sec=2.0,
    )
    detector = UnattendedObjectDetector(config)

    # Backpack at (300, 300)
    bag = create_track(10, 24, "backpack", (300, 300), (280, 280, 320, 320), 0.0, 0.0)
    detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")

    # Person standing right next to bag at (320, 300) -> distance = 20px (< 150px)
    person = create_track(1, 0, "person", (320, 300), (300, 200, 340, 350), 0.0, 4.0)
    bag.last_seen_time = 4.0

    alerts = detector.evaluate([bag, person], current_time=4.0, formatted_time="00:00:04")
    # Alert should be suppressed because owner is within 150px
    assert len(alerts) == 0


def test_moving_object_resets_stationary_state():
    config = UnattendedRuleConfig(
        stationary_duration_sec=3.0,
        stationary_displacement_px=20.0,
        nearby_person_radius_px=100.0,
        min_observation_sec=2.0,
    )
    detector = UnattendedObjectDetector(config)

    # Backpack starts at (300, 300)
    bag = create_track(10, 24, "backpack", (300, 300), (280, 280, 320, 320), 0.0, 0.0)
    detector.evaluate([bag], current_time=0.0, formatted_time="00:00:00")

    # At t=2.5s, bag moves significantly to (450, 450) -> displacement ~212px
    bag.current_center = (450, 450)
    bag.current_box = (430, 430, 470, 470)
    bag.last_seen_time = 2.5
    detector.evaluate([bag], current_time=2.5, formatted_time="00:00:02")

    # At t=4.0s (only 1.5s since movement), should not alert because stationary timer was reset!
    bag.last_seen_time = 4.0
    alerts = detector.evaluate([bag], current_time=4.0, formatted_time="00:00:04")
    assert len(alerts) == 0
