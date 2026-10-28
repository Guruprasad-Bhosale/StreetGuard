"""
Performance benchmarking suite for StreetGuard AI.
Compares baseline vs optimized configurations on representative Railway and Mall CCTV footage.
Measures pure YOLO+ByteTrack inference FPS, end-to-end processing FPS, device selection,
and detection count impact.
"""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Optional
import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streetguard.config import DEFAULT_MONITORED_CLASSES, ModelConfig
from streetguard.detector import YOLODetector
from streetguard.tracker import TrackHistoryManager
from streetguard.utils import get_video_metadata
from streetguard.visualizer import SurveillanceVisualizer


def run_benchmark_run(
    video_path: str,
    target_width: Optional[int],
    num_frames: int = 150,
    detector: Optional[YOLODetector] = None,
) -> dict:
    """
    Benchmark an individual configuration on a video clip.
    Measures pure inference time vs total pipeline time separately.
    """
    meta = get_video_metadata(video_path)
    if not meta.is_valid:
        raise ValueError(f"Cannot open video: {meta.error_message}")

    if detector is None:
        cfg = ModelConfig(
            model_name="yolo11n.pt",
            confidence_threshold=0.30,
            target_classes=DEFAULT_MONITORED_CLASSES,
        )
        detector = YOLODetector(cfg)

    cap = cv2.VideoCapture(video_path)
    tracker = TrackHistoryManager()
    vis = SurveillanceVisualizer()

    total_infer_time = 0.0
    total_pipeline_time = 0.0
    processed_frames = 0
    class_counts = Counter()
    unique_tracks = set()

    orig_w, orig_h = meta.width, meta.height
    if target_width and orig_w > target_width:
        scale = target_width / float(orig_w)
        infer_w = target_width
        infer_h = int(round(orig_h * scale))
    else:
        infer_w, infer_h = orig_w, orig_h
        scale = 1.0

    try:
        while processed_frames < num_frames:
            f_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            curr_video_time = processed_frames / meta.fps

            if scale < 1.0:
                frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

            # Pure inference time
            t0 = time.perf_counter()
            detections = detector.track(frame, persist=True)
            t1 = time.perf_counter()
            total_infer_time += (t1 - t0)

            # Tracking & history update
            active_tracks = tracker.update(detections, curr_video_time)

            # Overlay drawing
            annotated = frame.copy()
            annotated = vis.draw_tracking_trails(annotated, active_tracks)
            annotated = vis.draw_detections(annotated, detections)

            for d in detections:
                class_counts[d.class_name] += 1
                if d.track_id is not None:
                    unique_tracks.add(d.track_id)

            f_end = time.perf_counter()
            total_pipeline_time += (f_end - f_start)

    finally:
        cap.release()

    avg_infer_fps = processed_frames / max(1e-5, total_infer_time)
    avg_e2e_fps = processed_frames / max(1e-5, total_pipeline_time)

    return {
        "video_name": Path(video_path).name,
        "input_resolution": f"{orig_w}x{orig_h}",
        "inference_resolution": f"{infer_w}x{infer_h}",
        "scale_factor": round(scale, 3),
        "processed_frames": processed_frames,
        "total_infer_time_sec": round(total_infer_time, 3),
        "total_pipeline_time_sec": round(total_pipeline_time, 3),
        "inference_fps": round(avg_infer_fps, 1),
        "end_to_end_fps": round(avg_e2e_fps, 1),
        "unique_track_ids": len(unique_tracks),
        "class_counts": dict(class_counts),
        "total_detections": sum(class_counts.values()),
        "device": detector.device,
    }


def execute_full_benchmark(num_frames: int = 150) -> dict:
    """
    Run baseline vs optimized comparisons on Railway and Mall footage.
    """
    print("\n" + "=" * 65)
    print(" STREETGUARD AI - PERFORMANCE BENCHMARKING SUITE")
    print("=" * 65)

    # 1. Device Diagnostics
    cuda_available = torch.cuda.is_available()
    torch_version = torch.__version__
    print(f"[DIAG] PyTorch Build       : {torch_version}")
    print(f"[DIAG] CUDA Enabled PyTorch: {cuda_available}")
    if not cuda_available:
        print("[DIAG] Hardware Fallback   : CPU (Active build is CPU-only)")
    else:
        print(f"[DIAG] GPU Device          : {torch.cuda.get_device_name(0)}")

    cfg = ModelConfig(
        model_name="yolo11n.pt",
        confidence_threshold=0.30,
        target_classes=DEFAULT_MONITORED_CLASSES,
    )
    detector = YOLODetector(cfg)

    test_matrix = [
        # Railway footage
        {"path": "sample_data/train1.mp4", "label": "Railway (train1.mp4)"},
        # Mall footage
        {"path": "sample_data/mall1.mp4", "label": "Mall Atrium (mall1.mp4)"},
        # 4K Mall footage
        {"path": "sample_data/mall2.mp4", "label": "Mall 4K (mall2.mp4)"},
    ]

    benchmark_records = []

    for item in test_matrix:
        vpath = item["path"]
        vlabel = item["label"]
        print(f"\n>>> Benchmarking Scene: {vlabel}")

        # Config A: Baseline Native Resolution (unscaled)
        print("    Running Baseline (Native Resolution)...")
        res_baseline = run_benchmark_run(
            vpath, target_width=None, num_frames=num_frames, detector=detector
        )
        res_baseline["config_name"] = "Baseline (Native Res)"
        res_baseline["scene_label"] = vlabel
        benchmark_records.append(res_baseline)
        print(
            f"    -> Baseline: Infer: {res_baseline['inference_fps']} FPS | "
            f"End-to-End: {res_baseline['end_to_end_fps']} FPS ({res_baseline['inference_resolution']})"
        )

        # Config B: Optimized Rescaling (960px width)
        print("    Running Optimized (960px Downscale)...")
        res_opt960 = run_benchmark_run(
            vpath, target_width=960, num_frames=num_frames, detector=detector
        )
        res_opt960["config_name"] = "Optimized (960px Width)"
        res_opt960["scene_label"] = vlabel
        benchmark_records.append(res_opt960)
        print(
            f"    -> Optimized 960px: Infer: {res_opt960['inference_fps']} FPS | "
            f"End-to-End: {res_opt960['end_to_end_fps']} FPS ({res_opt960['inference_resolution']})"
        )

        # Config C: High-Speed Rescaling (640px width)
        print("    Running High-Speed (640px Downscale)...")
        res_opt640 = run_benchmark_run(
            vpath, target_width=640, num_frames=num_frames, detector=detector
        )
        res_opt640["config_name"] = "High-Speed (640px Width)"
        res_opt640["scene_label"] = vlabel
        benchmark_records.append(res_opt640)
        print(
            f"    -> High-Speed 640px: Infer: {res_opt640['inference_fps']} FPS | "
            f"End-to-End: {res_opt640['end_to_end_fps']} FPS ({res_opt640['inference_resolution']})"
        )

    # Print Comparison Table
    print("\n" + "=" * 100)
    print(" PERFORMANCE BENCHMARK COMPARISON TABLE")
    print("=" * 100)
    header = f"{'Scene':<22} | {'Config':<24} | {'Res':<10} | {'Infer FPS':<10} | {'E2E FPS':<10} | {'Detections':<10}"
    print(header)
    print("-" * 100)
    for r in benchmark_records:
        row = (
            f"{r['scene_label'][:22]:<22} | "
            f"{r['config_name']:<24} | "
            f"{r['inference_resolution']:<10} | "
            f"{r['inference_fps']:<10} | "
            f"{r['end_to_end_fps']:<10} | "
            f"{r['total_detections']:<10}"
        )
        print(row)
    print("=" * 100 + "\n")

    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "device": detector.device,
        "cuda_available": cuda_available,
        "torch_version": torch_version,
        "tested_frames_per_run": num_frames,
        "benchmarks": benchmark_records,
    }

    out_file = BASE_DIR / "data" / "performance_benchmark_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report_payload, indent=2))
    print(f"[OK] Benchmark JSON results exported to: {out_file}")

    return report_payload


if __name__ == "__main__":
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    execute_full_benchmark(num_frames=frames)
