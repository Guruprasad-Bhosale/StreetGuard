"""
Tests for video ingestion, metadata inspection, and invalid video error handling.
"""

from pathlib import Path
from streetguard.utils import get_video_metadata, seconds_to_hms


def test_time_formatting():
    assert seconds_to_hms(0) == "00:00:00"
    assert seconds_to_hms(65) == "00:01:05"
    assert seconds_to_hms(3661) == "01:01:01"
    assert seconds_to_hms(-5) == "00:00:00"


def test_invalid_nonexistent_video():
    meta = get_video_metadata("non_existent_footage_12345.mp4")
    assert meta.is_valid is False
    assert meta.error_message is not None
    assert "not found" in meta.error_message.lower()


def test_invalid_corrupted_file(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt_video.mp4"
    corrupt_file.write_text("This is plain text, not a valid video file.")

    meta = get_video_metadata(str(corrupt_file))
    assert meta.is_valid is False
    assert meta.error_message is not None
