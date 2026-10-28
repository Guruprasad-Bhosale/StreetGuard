"""
Tests for Detection output handling, coordinates, centers, footprints, and metrics.
"""

from streetguard.detector import Detection


def test_detection_dataclass_properties():
    # Box: x1=100, y1=150, x2=200, y2=350
    det = Detection(
        box=(100, 150, 200, 350),
        confidence=0.88,
        class_id=0,
        class_name="person",
        track_id=12,
    )

    assert det.width == 100
    assert det.height == 200
    assert det.center == (150, 250)
    # Footprint should be bottom-center: (cx, y2) = (150, 350)
    assert det.footprint == (150, 350)
    assert det.confidence == 0.88
    assert det.class_name == "person"
    assert det.track_id == 12


def test_detection_without_track_id():
    det = Detection(
        box=(50, 50, 80, 90),
        confidence=0.45,
        class_id=24,
        class_name="backpack",
        track_id=None,
    )
    assert det.track_id is None
    assert det.center == (65, 70)
    assert det.footprint == (65, 90)
