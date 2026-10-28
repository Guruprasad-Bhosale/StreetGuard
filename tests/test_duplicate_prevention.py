"""
Tests for Duplicate alert prevention:
- Person continuously staying inside restricted zone triggers alert ONLY ONCE.
- Re-entering after leaving and cooldown allows new alert.
- Persistent stationary unattended object suppresses repeated duplicate alerts.
"""

from streetguard.config import IntrusionRuleConfig, UnattendedRuleConfig
from streetguard.rules.intrusion import RestrictedAreaIntrusionDetector
from streetguard.rules.unattended import UnattendedObjectDetector
from streetguard.tracker import TrackState


def test_intrusion_duplicate_suppression():
    config = IntrusionRuleConfig(
        zone_name="Secured Area",
        polygon_normalized=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],
        reentry_cooldown_frames=3,
    )
    detector = RestrictedAreaIntrusionDetector(config)

    # Person inside zone (center 500, 500 in 1000x1000)
    person = TrackState(
        track_id=42,
        class_id=0,
        class_name="person",
        first_seen_time=1.0,
        last_seen_time=1.0,
        current_box=(480, 400, 520, 500),
        current_center=(500, 450),
        current_footprint=(500, 500),
    )

    # Frame 1: Enters zone -> Alert #1 generated!
    alerts1 = detector.evaluate([person], 1000, 1000, 1.0, "00:00:01")
    assert len(alerts1) == 1
    assert alerts1[0].track_id == 42

    # Frame 2: Still inside zone -> NO duplicate alert!
    alerts2 = detector.evaluate([person], 1000, 1000, 2.0, "00:00:02")
    assert len(alerts2) == 0

    # Frame 3: Still inside zone -> NO duplicate alert!
    alerts3 = detector.evaluate([person], 1000, 1000, 3.0, "00:00:03")
    assert len(alerts3) == 0


def test_intrusion_reentry_after_cooldown():
    config = IntrusionRuleConfig(
        zone_name="Secured Area",
        polygon_normalized=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],
        reentry_cooldown_frames=2,  # Short cooldown for test
    )
    detector = RestrictedAreaIntrusionDetector(config)

    person = TrackState(
        track_id=42,
        class_id=0,
        class_name="person",
        first_seen_time=1.0,
        last_seen_time=1.0,
        current_box=(480, 400, 520, 500),
        current_center=(500, 450),
        current_footprint=(500, 500),
    )

    # Frame 1: Inside -> Alert #1
    detector.evaluate([person], 1000, 1000, 1.0, "00:00:01")

    # Person exits to (100, 100)
    person.current_center = (100, 100)
    person.current_footprint = (100, 120)

    # Outside frame 1
    detector.evaluate([person], 1000, 1000, 2.0, "00:00:02")
    # Outside frame 2 (cooldown satisfied)
    detector.evaluate([person], 1000, 1000, 3.0, "00:00:03")

    # Person re-enters to (500, 500)
    person.current_center = (500, 450)
    person.current_footprint = (500, 500)
    alerts_reentry = detector.evaluate([person], 1000, 1000, 4.0, "00:00:04")

    # New alert should trigger on re-entry
    assert len(alerts_reentry) == 1
    assert alerts_reentry[0].track_id == 42


def test_unattended_duplicate_suppression():
    config = UnattendedRuleConfig(
        stationary_duration_sec=2.0,
        stationary_displacement_px=20.0,
        nearby_person_radius_px=100.0,
        min_observation_sec=1.0,
        alert_cooldown_sec=10.0,
    )
    detector = UnattendedObjectDetector(config)

    bag = TrackState(
        track_id=99,
        class_id=24,
        class_name="backpack",
        first_seen_time=0.0,
        last_seen_time=3.0,
        current_box=(300, 300, 340, 340),
        current_center=(320, 320),
        current_footprint=(320, 340),
    )
    bag.history.append((0.0, (320, 320), (300, 300, 340, 340)))
    bag.history.append((3.0, (320, 320), (300, 300, 340, 340)))

    # Frame at t=0.0 (anchor)
    detector.evaluate([bag], 0.0, "00:00:00")

    # Frame at t=3.0 (triggers initial alert)
    alerts1 = detector.evaluate([bag], 3.0, "00:00:03")
    assert len(alerts1) == 1

    # Frame at t=4.0 (still unattended, within cooldown period of 10s)
    alerts2 = detector.evaluate([bag], 4.0, "00:00:04")
    # Must NOT generate repeated duplicate alert
    assert len(alerts2) == 0
