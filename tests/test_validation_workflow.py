"""
Automated unit tests for the comprehensive validation workflow.
Tests dimension calculations, smoke testing, graceful error reporting,
and Markdown/JSON report generation.
"""

import json
from pathlib import Path
import pytest
from validate_pipeline import (
    calculate_inference_dimensions,
    generate_markdown_report,
    validate_video_pipeline,
)


def test_calculate_inference_dimensions_aspect_ratio_preservation():
    # 4K resolution (3840 x 2160) -> 16:9
    w, h, scale = calculate_inference_dimensions(3840, 2160, 960)
    assert w == 960
    assert h == 540
    assert scale == 0.25
    assert abs((w / h) - (3840 / 2160)) < 1e-4

    # 1080p resolution (1920 x 1080) -> 16:9
    w, h, scale = calculate_inference_dimensions(1920, 1080, 960)
    assert w == 960
    assert h == 540
    assert scale == 0.5

    # 720p resolution (1280 x 720) -> 16:9
    w, h, scale = calculate_inference_dimensions(1280, 720, 960)
    assert w == 960
    assert h == 540
    assert scale == 0.75

    # Smaller resolution than target (640 x 480) with target 960 -> unchanged
    w, h, scale = calculate_inference_dimensions(640, 480, 960)
    assert w == 640
    assert h == 480
    assert scale == 1.0


def test_validation_workflow_missing_video_error_handling():
    res = validate_video_pipeline("non_existent_stream_999.mp4", max_frames=10, verbose=False)
    assert res["success"] is False
    assert res["status"] == "FAILED"
    assert "error" in res
    assert "video_name" in res


def test_validation_workflow_smoke_execution():
    train1_path = Path("sample_data/train1.mp4")
    if not train1_path.is_file():
        pytest.skip("sample_data/train1.mp4 not found")

    res = validate_video_pipeline(
        str(train1_path),
        max_frames=5,
        downscale_width=960,
        verbose=False,
    )

    assert res["success"] is True
    assert res["status"] == "PASSED"
    assert res["processed_frames"] == 5
    assert res["input_resolution"] == "1920x1080"
    assert res["inference_resolution"] == "960x540"
    assert res["average_inference_fps"] > 0.0
    assert "class_detection_counts" in res
    assert "unique_tracking_ids_count" in res
    assert "active_tracks_stats" in res
    assert res["snapshot_path"] is not None


def test_markdown_and_json_report_generation(tmp_path: Path):
    dummy_data = {
        "generated_at": "2026-10-28 20:00:00",
        "device": "cpu",
        "total_videos": 1,
        "smoke_tests_passed": 1,
        "smoke_pass_rate": "100%",
        "full_tests_passed": 1,
        "full_pass_rate": "100%",
        "total_processed_frames": 75,
        "cumulative_processing_time_sec": 4.2,
        "full_results": [
            {
                "video_name": "train1.mp4",
                "success": True,
                "status": "PASSED",
                "input_resolution": "1920x1080",
                "inference_resolution": "960x540",
                "processed_frames": 75,
                "total_video_frames": 960,
                "average_inference_fps": 18.5,
                "processing_time_sec": 4.2,
                "unique_tracking_ids_count": 22,
                "class_detection_counts": {"person": 150, "backpack": 5},
                "total_detections": 155,
                "active_tracks_stats": {"min": 5, "max": 12, "mean": 8.5},
                "snapshot_path": str(tmp_path / "train1_snap.jpg"),
                "errors": [],
            }
        ],
    }

    # Test JSON serialization
    json_out = tmp_path / "test_report.json"
    json_out.write_text(json.dumps(dummy_data, indent=2))
    loaded = json.loads(json_out.read_text())
    assert loaded["total_videos"] == 1

    # Test Markdown report generation
    md_out = tmp_path / "test_report.md"
    md_content = generate_markdown_report(dummy_data, str(md_out))
    assert md_out.is_file()
    assert "StreetGuard AI - Comprehensive Multi-Video Validation Report" in md_content
    assert "train1.mp4" in md_content
    assert "18.5" in md_content
