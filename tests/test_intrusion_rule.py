"""
Tests for Restricted area intrusion detection:
- Polygon containment tests
- Footprint entry into zone
- Alert generation and explanation format
- Non-person class filtering
"""

from streetguard.config import IntrusionRuleConfig
from streetguard.rules.intrusion import RestrictedAreaIntrusionDetector
from streetguard.tracker import TrackState


def test_polygon_containment():
    # Define a rectangular normalized polygon from (0.2, 0.2) to (0.8, 0.8)
    config = IntrusionRuleConfig(
        zone_name="Test Zone",
        polygon_normalized=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
    )
    detector = RestrictedAreaIntrusionDetector(config)

    # Frame 1000x1000: Zone is [200..800, 200..800]
    # Point inside: (500, 500)
    assert detector.is_point_inside((500, 500), 1000, 1000) is True

    # Point outside: (100, 100)
    assert detector.is_point_inside((100, 100), 1000, 1000) is False

    # Point on edge: (200, 500) -> pointPolygonTest returns >= 0
    assert detector.is_point_inside((200, 500), 1000, 1000) is True


def test_intrusion_alert_triggered_for_person_entering():
    config = IntrusionRuleConfig(
        zone_name="Perimeter Zone A",
        polygon_normalized=[(0.4, 0.4), (0.8, 0.4), (0.8, 0.8), (0.4, 0.8)],
        use_footprint=True,
    )
    detector = RestrictedAreaIntrusionDetector(config)

    # Person track with footprint at (600, 600) in 1000x1000 frame (Inside zone)
    person_track = TrackState(
        track_id=7,
        class_id=0,
        class_name="person",
        first_seen_time=10.0,
        last_seen_time=10.0,
        current_box=(580, 500, 620, 600),
        current_center=(600, 550),
        current_footprint=(600, 600),
    )

    alerts = detector.evaluate(
        tracks=[person_track],
        frame_width=1000,
        frame_height=1000,
        current_time=10.5,
        formatted_time="00:00:10",
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.track_id == 7
    assert alert.object_category == "person"
    assert "Perimeter Intrusion" in alert.explanation
    assert "Perimeter Zone A" in alert.explanation


def test_non_person_classes_do_not_trigger_intrusion():
    config = IntrusionRuleConfig(
        zone_name="Perimeter Zone A",
        polygon_normalized=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    )
    detector = RestrictedAreaIntrusionDetector(config)

    # Backpack inside zone
    bag_track = TrackState(
        track_id=3,
        class_id=24,
        class_name="backpack",
        first_seen_time=5.0,
        last_seen_time=5.0,
        current_box=(400, 400, 450, 450),
        current_center=(425, 425),
        current_footprint=(425, 450),
    )

    alerts = detector.evaluate(
        tracks=[bag_track],
        frame_width=1000,
        frame_height=1000,
        current_time=5.0,
        formatted_time="00:00:05",
    )
    # Luggage items alone do not violate person perimeter intrusion
    assert len(alerts) == 0
