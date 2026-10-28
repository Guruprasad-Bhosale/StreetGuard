"""
Configuration module for StreetGuard AI surveillance analytics.
Handles model parameters, rule thresholds, paths, and class definitions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
DEFAULT_DB_PATH = DATA_DIR / "streetguard.db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard COCO class mappings for surveillance objects
COCO_CLASSES: Dict[int, str] = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    25: "umbrella",
    56: "chair",
    63: "laptop",
    67: "cell phone",
}

# Default target categories to track in surveillance
DEFAULT_MONITORED_CLASSES: List[int] = [0, 24, 26, 28]

# Default restricted polygonal area (relative/normalized coords [0..1] x, y)
# Top-left, top-right, bottom-right, bottom-left trapezoid/quadrilateral
DEFAULT_RESTRICTED_POLYGON_NORMALIZED: List[Tuple[float, float]] = [
    (0.45, 0.40),
    (0.85, 0.40),
    (0.95, 0.85),
    (0.40, 0.85),
]


def get_default_device() -> str:
    """Detect available compute device (CUDA GPU if available, fallback to CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelConfig:
    """Configuration for YOLO detector and ByteTrack."""
    model_name: str = "yolo11n.pt"
    device: str = field(default_factory=get_default_device)
    confidence_threshold: float = 0.35
    luggage_confidence_threshold: float = 0.20  # Sensitive threshold for small/dark luggage classes
    iou_threshold: float = 0.45
    target_classes: List[int] = field(default_factory=lambda: list(DEFAULT_MONITORED_CLASSES))
    tracker_type: str = "bytetrack.yaml"


@dataclass
class IntrusionRuleConfig:
    """Configuration for Restricted Area Intrusion detection."""
    enabled: bool = True
    zone_name: str = "Restricted Perimeter"
    polygon_normalized: List[Tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_RESTRICTED_POLYGON_NORMALIZED)
    )
    # Check bottom-center of bounding box (feet/ground contact point)
    use_footprint: bool = True
    # Cooldown frames before a person leaving and re-entering triggers a new alert
    reentry_cooldown_frames: int = 45


@dataclass
class UnattendedRuleConfig:
    """
    Configuration for Unattended Object detection heuristic.
    Rule logic:
    - Object belongs to unattended-candidate classes (e.g. backpack, handbag, suitcase).
    - Object center displacement <= max_displacement_px over stationary_duration_sec.
    - No person's center/footprint is lingering within nearby_person_radius_px.
    - Observation duration >= min_observation_sec.
    - Transient pedestrians walking past do not suppress alerts indefinitely.
    """
    enabled: bool = True
    target_classes: List[int] = field(default_factory=lambda: [24, 26, 28])
    stationary_duration_sec: float = 4.0
    stationary_displacement_px: float = 30.0
    nearby_person_radius_px: float = 75.0
    perspective_vertical_scale: float = 1.5
    person_dwell_sec: float = 1.5  # Sustained dwell required before nearby person suppresses unattended status
    pedestrian_speed_threshold_px: float = 15.0  # Motion threshold to identify transient walking pedestrians
    min_observation_sec: float = 2.0
    missed_grace_sec: float = 3.0
    # Frames/seconds to wait before permitting a new alert for an ongoing unattended event
    alert_cooldown_sec: float = 30.0


@dataclass
class VideoProcessingConfig:
    """Configuration for video ingestion and rendering."""
    target_width: int = 960  # Scale down large 1080p/4K videos for responsive CPU execution
    draw_tracking_trails: bool = True
    trail_length: int = 30
    save_incident_screenshots: bool = True
    screenshot_dir: Path = SCREENSHOTS_DIR
    db_path: Path = DEFAULT_DB_PATH
