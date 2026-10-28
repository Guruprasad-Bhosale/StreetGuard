"""
Database persistence module for StreetGuard AI incidents using SQLite.
Stores detected surveillance incidents, associated metadata, and screenshot references.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from streetguard.config import DEFAULT_DB_PATH


@dataclass
class IncidentRecord:
    """Represents a logged security incident."""
    event_category: str  # 'intrusion' or 'unattended_object'
    video_timestamp: float  # Video elapsed time in seconds
    formatted_timestamp: str  # "00:01:23"
    track_id: Optional[int]
    object_category: str
    reason: str
    screenshot_path: Optional[str] = None
    video_name: str = "cctv_feed.mp4"
    details_json: str = "{}"
    wall_clock_time: str = ""
    id: Optional[int] = None

    def __post_init__(self):
        if not self.wall_clock_time:
            self.wall_clock_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class IncidentDatabase:
    """Manages SQLite storage and retrieval for surveillance events."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create incidents table and indices if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_category TEXT NOT NULL,
                    video_name TEXT NOT NULL,
                    video_timestamp REAL NOT NULL,
                    formatted_timestamp TEXT NOT NULL,
                    wall_clock_time TEXT NOT NULL,
                    track_id INTEGER,
                    object_category TEXT NOT NULL,
                    screenshot_path TEXT,
                    reason TEXT NOT NULL,
                    details_json TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_category ON incidents (event_category)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_video ON incidents (video_name)"
            )
            conn.commit()

    def log_incident(self, incident: IncidentRecord) -> int:
        """Insert a new incident record into the database and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO incidents (
                    event_category, video_name, video_timestamp,
                    formatted_timestamp, wall_clock_time, track_id,
                    object_category, screenshot_path, reason, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident.event_category,
                    incident.video_name,
                    incident.video_timestamp,
                    incident.formatted_timestamp,
                    incident.wall_clock_time,
                    incident.track_id,
                    incident.object_category,
                    incident.screenshot_path,
                    incident.reason,
                    incident.details_json,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_incidents(
        self,
        category: Optional[str] = None,
        video_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[IncidentRecord]:
        """Fetch incidents filtered by category and video name, ordered by most recent."""
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []

        if category:
            query += " AND event_category = ?"
            params.append(category)

        if video_name:
            query += " AND video_name = ?"
            params.append(video_name)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                IncidentRecord(
                    id=row["id"],
                    event_category=row["event_category"],
                    video_name=row["video_name"],
                    video_timestamp=row["video_timestamp"],
                    formatted_timestamp=row["formatted_timestamp"],
                    wall_clock_time=row["wall_clock_time"],
                    track_id=row["track_id"],
                    object_category=row["object_category"],
                    screenshot_path=row["screenshot_path"],
                    reason=row["reason"],
                    details_json=row["details_json"],
                )
                for row in rows
            ]

    def get_stats(self) -> Dict[str, int]:
        """Retrieve count of incidents grouped by category."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT event_category, COUNT(*) as count FROM incidents GROUP BY event_category"
            )
            counts = {row["event_category"]: row["count"] for row in cursor.fetchall()}
            cursor.execute("SELECT COUNT(*) as total FROM incidents")
            total = cursor.fetchone()["total"]
            counts["total"] = total
            return counts

    def clear_incidents(self) -> None:
        """Clear all stored incidents."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM incidents")
            conn.commit()
