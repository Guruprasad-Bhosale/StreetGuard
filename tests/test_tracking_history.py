"""
Tests for Tracking history updates, trail accumulation, missing track ID safety, and motion displacement.
"""

from streetguard.detector import Detection
from streetguard.tracker import TrackHistoryManager, TrackState


def test_track_history_initialization_and_update():
    manager = TrackHistoryManager(max_missed_frames=10)

    # Frame 1: Detection at t=1.0s
    det1 = Detection(box=(100, 100, 140, 180), confidence=0.85, class_id=0, class_name="person", track_id=1)
    tracks = manager.update([det1], current_time=1.0)

    assert len(tracks) == 1
    t1 = tracks[0]
    assert t1.track_id == 1
    assert t1.frames_tracked == 1
    assert t1.observation_duration() == 0.0

    # Frame 2: Person moves to x=120, y=100 at t=2.0s
    det2 = Detection(box=(120, 100, 160, 180), confidence=0.87, class_id=0, class_name="person", track_id=1)
    tracks = manager.update([det2], current_time=2.0)

    assert len(tracks) == 1
    t1 = tracks[0]
    assert t1.frames_tracked == 2
    assert t1.observation_duration() == 1.0
    assert len(t1.history) == 2


def test_missing_track_ids_handled_safely():
    manager = TrackHistoryManager()

    # Detection with track_id=None (e.g. ByteTrack has not assigned an ID yet)
    det_untracked = Detection(box=(200, 200, 250, 270), confidence=0.5, class_id=24, class_name="backpack", track_id=None)
    tracks = manager.update([det_untracked], current_time=0.5)

    # Should not crash, and no tracks created for None track_id
    assert len(tracks) == 0


def test_displacement_over_window():
    state = TrackState(
        track_id=5,
        class_id=24,
        class_name="backpack",
        first_seen_time=0.0,
        last_seen_time=0.0,
        current_box=(100, 100, 120, 120),
        current_center=(110, 110),
        current_footprint=(110, 120),
    )
    # Append static observations at t=0, 1, 2, 3
    state.history.append((0.0, (100, 100), (90, 90, 110, 110)))
    state.history.append((1.0, (102, 101), (92, 91, 112, 111)))
    state.history.append((2.0, (101, 100), (91, 90, 111, 110)))
    state.history.append((3.0, (100, 101), (90, 91, 110, 111)))

    disp = state.get_displacement_over_window(duration_sec=3.0, current_time=3.0)
    assert disp < 5.0  # Stationary!

    # Add significant movement at t=4.0
    state.history.append((4.0, (200, 200), (190, 190, 210, 210)))
    disp_moved = state.get_displacement_over_window(duration_sec=3.0, current_time=4.0)
    assert disp_moved > 50.0  # Moved!
