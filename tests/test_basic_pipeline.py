"""
Tests for initial video processing pipeline:
- Video decoding and sequential frame iteration
- YOLO detection and ByteTrack tracking ID generation
- Person count and FPS telemetry
- Graceful handling of missing files and unsupported formats
"""

from pathlib import Path
import pytest
from streetguard.pipeline import VideoAnalyticsPipeline
from streetguard.utils import get_video_metadata


def test_missing_video_graceful_handling():
    meta = get_video_metadata("non_existent_footage_sample.xyz")
    assert meta.is_valid is False
    assert meta.error_message is not None

    pipeline = VideoAnalyticsPipeline()
    with pytest.raises(ValueError, match="Invalid video file"):
        # Attempting to process invalid video should raise clean descriptive ValueError
        next(pipeline.process_video_stream("non_existent_footage_sample.xyz"))


def test_train1_pipeline_execution():
    train1_path = Path("sample_data/train1.mp4")
    if not train1_path.is_file():
        pytest.skip("train1.mp4 not found in sample_data")

    meta = get_video_metadata(str(train1_path))
    assert meta.is_valid is True
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.fps == 25.0

    # Test processing 10 frames of train1
    pipeline = VideoAnalyticsPipeline()
    results = []
    for frame_res in pipeline.process_video_stream(str(train1_path), max_frames=10):
        results.append(frame_res)

    assert len(results) == 10
    # Frame index should increment sequentially
    assert results[0].frame_idx == 1
    assert results[-1].frame_idx == 10
    # Telemetry properties exist
    assert results[-1].fps > 0.0
    assert results[-1].annotated_frame is not None
    assert results[-1].annotated_frame.shape[0] > 0
    # In train1.mp4 there are real people on the platform
    assert any(res.person_count > 0 for res in results)
