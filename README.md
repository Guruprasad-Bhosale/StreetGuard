# 🛡️ StreetGuard AI: Explainable Computer Vision Surveillance Application

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics%2011n-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracker-ByteTrack-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Testing](https://img.shields.io/badge/Tests-33%20Passed-success.svg)](file:///g:/StreetGurad/tests)
[![Performance](https://img.shields.io/badge/RealTime-30.5%20FPS%20(CPU)-blueviolet.svg)](file:///g:/StreetGurad/data/performance_benchmark_report.json)

**StreetGuard AI** is a lightweight, explainable computer vision surveillance system designed for automated CCTV video analytics. Built for academic research, demonstration, and final-year B.Tech AIML project viva defense, it pairs **Ultralytics YOLO** object detection with **ByteTrack** multi-object tracking to analyze prerecorded CCTV footage (`train1`, `train2`, `train3`, `mall1`, `mall2`).

The system detects people and luggage, tracks them across frames with persistent IDs, flags unauthorized entry into restricted polygonal zones, and identifies potentially unattended luggage using a transparent 7-stage diagnostic decision pipeline.

---

## 📹 Video Assets Inventory

All five surveillance videos are stored in `sample_data/`:

| Video Identifier | File Name | Native Resolution | Frame Rate | Total Frames | Duration | File Size | Scene Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train1`** | `train1.mp4` | **1920 x 1080** (FHD) | 25.0 FPS | 960 frames | ~38.4s | 42.1 MB | Railway station platform with moving passengers |
| **`train2`** | `train2.mp4` | **3840 x 2160** (4K) | 60.0 FPS | 620 frames | ~10.3s | 40.7 MB | High-framerate 4K railway platform overhead |
| **`train3`** | `train3.mp4` | **3840 x 2160** (4K) | 60.0 FPS | 1128 frames | ~18.8s | 73.5 MB | High-framerate 4K concourse passenger flow |
| **`mall1`** | `mall1.mp4` | **1280 x 720** (HD) | 24.0 FPS | 240 frames | ~10.0s | 3.95 MB | Shopping mall corridor with shoppers and items |
| **`mall2`** | `mall2.mp4` | **3840 x 2160** (4K) | 30.0 FPS | 300 frames | ~10.0s | 30.9 MB | 4K shopping mall atrium overhead view |
| **`cctv_demo`** | `cctv_demo.mp4`| **800 x 480** (SD) | 24.0 FPS | 288 frames | ~12.0s | 0.78 MB | Synthetic demo with clear intrusion & unattended bag |

---

## 📁 Project Architecture & File Responsibilities

StreetGuard AI follows a strict single-responsibility design. Every module has one clear owner:

```text
G:\StreetGurad\
├── app.py                         # Streamlit web dashboard, UI controls, live video feed & incident vault
├── benchmark_performance.py       # Performance benchmarking suite (Baseline vs. 960px vs. 640px)
├── validate_pipeline.py           # CLI validation workflow for smoke testing and markdown report generation
├── requirements.txt               # Pinned, tested dependency configuration
├── README.md                      # Comprehensive project guide and viva preparation reference
├── yolo11n.pt                     # Pre-trained Ultralytics YOLO model weights
├── streetguard/                   # Core surveillance Python package
│   ├── __init__.py                # Package initialization and exports
│   ├── config.py                  # Centralized configuration dataclasses and conservative defaults
│   ├── database.py                # SQLite persistence for incident records, evidence, and statistics
│   ├── detector.py                # YOLO object detection wrapper with CPU/GPU auto-detection
│   ├── tracker.py                 # ByteTrack integration and temporal track trajectory history
│   ├── pipeline.py                # Linear 8-step video processing pipeline coordinator
│   ├── visualizer.py              # OpenCV overlays: bounding boxes, tracking trails, polygon zones, HUD
│   ├── utils.py                   # Genuinely shared helpers: video metadata, time formatting, snapshots
│   └── rules/                     # Explainable surveillance analytic rules
│       ├── __init__.py
│       ├── intrusion.py           # Restricted polygonal area intrusion rule with ground-footprint logic
│       └── unattended.py          # 7-stage diagnostic unattended-object rule with dropout resilience
├── scripts/
│   └── generate_sample_cctv.py    # Synthetic realistic CCTV video generator for quick testing
├── sample_data/                   # Prerecorded MP4 surveillance footage
├── data/                          # Auto-generated runtime data (SQLite DB, reports, snapshots)
│   ├── streetguard.db             # Local SQLite security incident database
│   ├── performance_benchmark_report.json # Benchmark results across railway and mall scenes
│   ├── validation_report.json     # Machine-readable validation results
│   ├── validation_report.md       # Human-readable validation summary
│   ├── validation_snapshots/      # Pipeline verification frames
│   └── screenshots/               # Stored incident snapshot evidence (.jpg)
└── tests/                         # Comprehensive pytest test suite (33 passing unit tests)
    ├── test_basic_pipeline.py     # Sequential frame iteration, metadata, and graceful error handling
    ├── test_database.py           # SQLite logging, category filtering, statistics, and clearing
    ├── test_detection_output.py   # Bounding boxes, centers, and ground footprint coordinates
    ├── test_duplicate_prevention.py # Intrusion duplicate suppression and re-entry cooldown
    ├── test_intrusion_rule.py     # cv2.pointPolygonTest containment and person-only filtering
    ├── test_optimizations_and_diagnostics.py # NumPy/PyArrow compatibility, Arrow serialization, timing, dropout resilience
    ├── test_tracking_history.py   # Trajectory accumulation and missing track ID safety
    ├── test_unattended_rule.py    # Stationary duration, nearby-person suppression, movement reset
    ├── test_validation_workflow.py# Dimension scaling, smoke validation, and report generation
    └── test_video_handling.py     # Video time formatting and corrupted file handling
```

---

## 🔄 The 8-Step Video Processing Pipeline

In [`streetguard/pipeline.py`](file:///g:/StreetGurad/streetguard/pipeline.py), each video frame is processed through an explainable, linear sequence:

```text
Video Frame
   │
   ▼
[1. Ingest & Scale]   ──> Downscale high-resolution frames (e.g. 1080p -> 960px) for real-time CPU speed
   │
   ▼
[2. YOLO Detection]   ──> Run YOLO11n on target classes (person, backpack, handbag, suitcase)
   │
   ▼
[3. ByteTrack State]  ──> Associate detections with historical trajectories, assign persistent Track IDs
   │
   ▼
[4. Intrusion Rule]   ──> Test person ground-footprint against restricted polygon via cv2.pointPolygonTest
   │
   ▼
[5. Unattended Rule]  ──> Evaluate 7-stage diagnostic pipeline (stationarity, proximity, dropout tolerance)
   │
   ▼
[6. Incident Logging] ──> Persist valid alerts and JPEG evidence snapshots to SQLite database
   │
   ▼
[7. Frame Rendering]  ──> Draw semi-transparent zones, tracking motion trails, bounding boxes, and HUD
   │
   ▼
[8. Telemetry Yield]  ──> Yield annotated frame with pure inference FPS and end-to-end FPS metrics
```

---

## 🛑 Explainable Surveillance Heuristics

### 1. Restricted Area Perimeter Intrusion
- **Geometric Test**: Defined as an $N$-sided polygon in normalized image coordinates $[0 \dots 1]$. At runtime, vertices are converted to current frame pixel coordinates. OpenCV's `cv2.pointPolygonTest` tests whether a point is inside ($\ge 0$) or outside ($< 0$).
- **Ground-Footprint Heuristic**: Rather than testing the bounding box center $(cx, cy)$ (which sits at chest level and can falsely trigger when leaning over a line), StreetGuard evaluates the **bottom-center ground contact point** $(cx, y_2)$ representing the person's feet on the floor.
- **Duplicate Suppression**: Once a person enters the restricted zone, an alert is raised once. While the person remains inside, duplicate alerts are suppressed. If the person leaves the zone and remains outside for `reentry_cooldown_frames` (default: 45 frames), a new alert is permitted upon re-entry.

### 2. 7-Stage Unattended Object Detection Pipeline
Rather than a black-box model, unattended object detection uses an explainable 7-stage verification pipeline with live diagnostic auditing:

| Stage | Name | Verification Condition |
| :---: | :--- | :--- |
| **1** | `OBJECT_DETECTED` | YOLO detects candidate luggage (`backpack`, `handbag`, `suitcase`) with confidence $\ge 0.25$. |
| **2** | `OBJECT_TRACKED` | ByteTrack assigns a persistent tracking ID with minimum observation time ($t_{obs} \ge 2.0\text{s}$) to filter camera glitches. |
| **3** | `STATIONARY_TIMER` | Euclidean distance from anchor $\Delta d \le 30.0\text{px}$. If $\Delta d \le 30.0\text{px}$, stationary timer advances using source video timestamp. If $\Delta d > 30.0\text{px}$, the timer resets. |
| **4** | `STATIONARY_CONFIRMED` | Object has remained continuously stationary for $\ge 4.0\text{s}$ (configurable). |
| **5** | `PROXIMITY_EVAL` | Distance to nearest person footprint/center is measured. If a person is within $R \le 120.0\text{px}$, the alert is suppressed (owner is nearby). |
| **6** | `ALERT_DECISION` | Evaluates alert cooldown (10.0s). If isolated and stationary, alert is accepted and labeled: `"Potentially unattended object — review required."` |
| **7** | `DB_PERSISTENCE` | Persists alert record and JPEG snapshot to SQLite database with error handling. |

- **Detector Dropout Resilience**: If a stationary bag is temporarily missed by YOLO for 1–2 frames due to occlusion or lighting, a 3.0-second grace window (`missed_grace_sec = 3.0s`) prevents wiping the stationary timer.
- **Review-Required Wording**: The system flags observable physical stillness and absence of nearby people; it does not claim to know whether a bag has been abandoned or contains dangerous materials.

---

## ⚡ Installation & Launch Instructions

### 1. Environment & Dependencies
Tested on **Python 3.12** on Windows 11 (64-bit):
- `numpy==1.26.4`
- `pyarrow==18.1.0` (Pinned to resolve NumPy 2.0 compatibility conflicts)
- `torch>=2.0.0` (CPU build with automatic CUDA GPU detection)
- `ultralytics>=8.3.0`
- `streamlit>=1.30.0`
- `opencv-python>=4.8.0`

Install all dependencies:
```powershell
pip install -r requirements.txt
```

> **Windows C-Runtime Note (`c10.dll`)**: On Windows, `import torch` is placed at the top of entry points (`app.py`, `benchmark_performance.py`, `validate_pipeline.py`) to prevent MSVC runtime collisions with PyArrow.

### 2. Launching the Application
On Windows systems with **Windows Application Control (WDAC)** or **AppLocker**, launching via `streamlit.exe run app.py` may be blocked by OS policy.

Launch the application directly using the Python module command:
```powershell
python -m streamlit run app.py
```
This is fully supported, complies with OS security policies, and opens the dashboard at `http://localhost:8501`.

### 3. GPU / Hardware Diagnostics
To inspect compute hardware auto-detection:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Fallback')"
```
*(If an NVIDIA GPU such as the RTX 3050 is available and CUDA acceleration is desired, install the CUDA wheel: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`)*

---

## 🧪 Testing & Validation

### 1. Run the Complete Pytest Suite
Execute all 33 automated unit tests:
```powershell
python -m pytest -v
```
**Status: 33 passed, 0 failed in ~4.6 seconds.**

### 2. Validate CCTV Video Pipeline via CLI
Validate detection, ByteTrack tracking, and telemetry on single videos or the full benchmark suite:
```powershell
# Validate 75 frames of train1.mp4 (supports positional or flag arguments)
python validate_pipeline.py sample_data/train1.mp4 75

# Run comprehensive validation across all 5 videos
python validate_pipeline.py --all-videos
```
Generates reports in `data/validation_report.json` and `data/validation_report.md`.

### 3. Run Performance Benchmark
Evaluate pure inference FPS vs. end-to-end pipeline FPS across native and downscaled resolutions:
```powershell
python benchmark_performance.py
```

#### Measured Benchmark Results (Intel CPU / Torch 2.14.0+cpu):
| Scene | Native Resolution | Benchmark Configuration | Target Resolution | Pure Inference FPS (YOLO + ByteTrack) | End-to-End Pipeline FPS | Real-Time Target ($\ge 30$ FPS) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Railway (`train1.mp4`)** | 1920 x 1080 | Baseline (Native) | 1920 x 1080 | 24.6 FPS | 21.4 FPS | Below 30 FPS target |
| **Railway (`train1.mp4`)** | 1920 x 1080 | **Optimized (960px)** | **960 x 540** | **36.9 FPS** | **30.5 FPS** | **$\ge 30$ FPS Achieved!** |
| **Railway (`train1.mp4`)** | 1920 x 1080 | High-Speed (640px) | 640 x 360 | 37.2 FPS | 31.1 FPS | Exceeds real-time |
| **Mall Atrium (`mall1.mp4`)** | 1280 x 720 | Baseline (Native) | 1280 x 720 | 37.7 FPS | 33.6 FPS | **$\ge 30$ FPS Achieved!** |
| **Mall Atrium (`mall1.mp4`)** | 1280 x 720 | **Optimized (960px)** | **960 x 540** | **37.3 FPS** | **32.3 FPS** | **$\ge 30$ FPS Achieved!** |
| **Mall Atrium (`mall1.mp4`)** | 1280 x 720 | High-Speed (640px) | 640 x 360 | 37.5 FPS | 33.6 FPS | Exceeds real-time |
| **Mall 4K (`mall2.mp4`)** | 3840 x 2160 | Baseline (Native) | 3840 x 2160 | 34.1 FPS | 21.3 FPS | 4K video decoding bound |
| **Mall 4K (`mall2.mp4`)** | 3840 x 2160 | **Optimized (960px)** | **960 x 540** | **36.4 FPS** | **23.4 FPS** | Inference 36.4 FPS (CPU decode bound) |

---

## 🎓 Viva Preparation & Defense Reference

Use these straightforward answers during your project viva:

### Q1: What is the difference between Object Detection and Multi-Object Tracking?
> **Answer**: Object Detection (YOLO) identifies where objects are in an individual image frame and assigns class labels and confidence scores. Multi-Object Tracking (ByteTrack) connects detections across consecutive frames over time, associating bounding boxes to the same physical entity and assigning a persistent **Track ID**.

### Q2: Why use ByteTrack instead of SORT or DeepSORT?
> **Answer**: Classic SORT discards low-confidence detections. In surveillance, occluded or distant objects often have temporarily lower confidence scores. ByteTrack associates high-confidence detections first, then matches remaining unmatched tracks with **low-confidence detections**, maintaining tracking continuity through occlusions without the computational overhead of heavy deep appearance feature extractors.

### Q3: Why test the bottom-center footprint point instead of the bounding-box center for perimeter intrusion?
> **Answer**: The center of a person's bounding box $(cx, cy)$ is located at waist or chest height. If a person leans forward or reaches over a fence, their center might cross into the restricted zone while they are still standing safely outside. Testing the ground contact point $(cx, y_2)$ ensures we verify where the person's feet actually touch the floor.

### Q4: How does the system handle temporary detector dropouts for stationary luggage?
> **Answer**: Real CCTV footage often suffers from 1–2 frame detector dropouts. If the tracking state was deleted immediately on a missed frame, the stationary timer would reset back to zero every time. StreetGuard implements a 3.0-second grace window (`missed_grace_sec = 3.0s`) and spatial re-anchoring so that brief visual dropouts do not erase stationary progress.

### Q5: How do you achieve $\ge 30$ FPS on CPU without a GPU?
> **Answer**: We downscale high-resolution frames (1080p and 4K) to `960x540` while preserving aspect ratio before passing them to YOLO11n. This reduces the number of pixels processed by 75% compared to 1080p, raising inference speed from ~24 FPS to ~37 FPS and end-to-end pipeline speed to 30.5 FPS on standard CPU hardware.

### Q6: Can this system determine if a bag contains a bomb or is legally abandoned?
> **Answer**: No. Computer vision observes physical visual attributes: object classification, spatial stillness, and distance to detected persons. It cannot determine legal ownership, human intent, or internal contents. That is why all alerts are strictly labeled: *"Potentially unattended object — review required"*, designating it as a decision-support tool for human security operators.

### Q7: What are the main limitations of the system?
> **Answer**:
> 1. **Ownership Ambiguity**: The system cannot tell who owns an item. If Person A drops a bag and walks away while stranger Person B stands nearby ($< 120\text{px}$), the bag is not flagged as unattended.
> 2. **Camera Field of View**: If an owner sits on a bench just outside the camera's frame, the bag appears isolated to the algorithm.
> 3. **2D Perspective Distortion**: Distances are measured in 2D image pixels. A 120-pixel radius covers a much larger real-world area in the background than in the foreground. Future work could incorporate camera calibration/homography to compute real-world metric distances.

---

## 📜 License & Academic Integrity
Developed as an academic demonstration and research project for B.Tech Artificial Intelligence and Machine Learning.
All sample CCTV footage is utilized strictly for research, testing, and educational viva defense.
#   S t r e e t G u a r d  
 