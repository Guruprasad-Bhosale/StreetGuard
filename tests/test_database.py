"""
Tests for Incident database logging, querying, stats, and clearing using SQLite.
"""

from pathlib import Path
from streetguard.database import IncidentDatabase, IncidentRecord


def test_database_logging_and_retrieval(tmp_path: Path):
    db_file = tmp_path / "test_streetguard.db"
    db = IncidentDatabase(db_path=db_file)

    rec1 = IncidentRecord(
        event_category="intrusion",
        video_name="gate_camera.mp4",
        video_timestamp=14.5,
        formatted_timestamp="00:00:14",
        track_id=3,
        object_category="person",
        reason="Person ID #3 entered restricted perimeter",
        screenshot_path="/path/to/snap1.jpg",
    )
    rec2 = IncidentRecord(
        event_category="unattended_object",
        video_name="lobby_camera.mp4",
        video_timestamp=45.2,
        formatted_timestamp="00:00:45",
        track_id=8,
        object_category="backpack",
        reason="Backpack ID #8 left unattended for 10s",
        screenshot_path="/path/to/snap2.jpg",
    )

    id1 = db.log_incident(rec1)
    id2 = db.log_incident(rec2)
    assert id1 > 0
    assert id2 > id1

    # Fetch all
    incidents = db.get_incidents()
    assert len(incidents) == 2

    # Filter by category
    intrusions = db.get_incidents(category="intrusion")
    assert len(intrusions) == 1
    assert intrusions[0].track_id == 3
    assert intrusions[0].event_category == "intrusion"

    unattended = db.get_incidents(category="unattended_object")
    assert len(unattended) == 1
    assert unattended[0].track_id == 8

    # Stats
    stats = db.get_stats()
    assert stats["intrusion"] == 1
    assert stats["unattended_object"] == 1
    assert stats["total"] == 2

    # Clear
    db.clear_incidents()
    assert len(db.get_incidents()) == 0
    assert db.get_stats()["total"] == 0
