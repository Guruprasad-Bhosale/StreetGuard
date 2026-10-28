"""
Detector module for StreetGuard AI.
Wraps Ultralytics YOLO model for inference and ByteTrack integration.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from streetguard.config import COCO_CLASSES, ModelConfig


@dataclass
class Detection:
    """Standardized representation of a single object detection."""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        """Center point (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @property
    def footprint(self) -> Tuple[int, int]:
        """Bottom-center point (cx, y2) representing ground footprint."""
        x1, _, x2, y2 = self.box
        return int((x1 + x2) / 2), int(y2)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


class YOLODetector:
    """Manages YOLO model loading, inference, and ByteTrack execution."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = self._resolve_device(self.config.device)
        self.model = YOLO(self.config.model_name)
        # Move model to resolved device
        self.model.to(self.device)

    def _resolve_device(self, requested_device: str) -> str:
        """Ensure requested device is available, falling back safely to CPU."""
        if requested_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested_device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run pure detection on a single image/frame."""
        infer_conf = min(
            self.config.confidence_threshold,
            getattr(self.config, "luggage_confidence_threshold", self.config.confidence_threshold),
        )
        results = self.model.predict(
            source=frame,
            conf=infer_conf,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            device=self.device,
            verbose=False,
        )
        return self._parse_ultralytics_results(results[0])

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        """
        Run tracking using YOLO + ByteTrack.
        Preserves track IDs across consecutive frames.
        """
        infer_conf = min(
            self.config.confidence_threshold,
            getattr(self.config, "luggage_confidence_threshold", self.config.confidence_threshold),
        )
        results = self.model.track(
            source=frame,
            conf=infer_conf,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            tracker=self.config.tracker_type,
            persist=persist,
            device=self.device,
            verbose=False,
        )
        return self._parse_ultralytics_results(results[0])

    def _parse_ultralytics_results(self, result) -> List[Detection]:
        """Convert Ultralytics Boxes result into standardized Detection objects."""
        detections: List[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)

        names = result.names or COCO_CLASSES
        luggage_thresh = getattr(self.config, "luggage_confidence_threshold", self.config.confidence_threshold)

        for i in range(len(boxes)):
            cid = int(class_ids[i])
            conf = float(confidences[i])

            # Class-specific confidence filtering (allows sensitive luggage detection without person clutter)
            min_thresh = luggage_thresh if cid in (24, 26, 28) else self.config.confidence_threshold
            if conf < min_thresh:
                continue

            box = (
                int(boxes[i][0]),
                int(boxes[i][1]),
                int(boxes[i][2]),
                int(boxes[i][3]),
            )
            cname = names.get(cid, COCO_CLASSES.get(cid, f"class_{cid}"))
            tid = int(track_ids[i]) if track_ids is not None else None

            detections.append(
                Detection(
                    box=box,
                    confidence=conf,
                    class_id=cid,
                    class_name=cname,
                    track_id=tid,
                )
            )

        return detections
