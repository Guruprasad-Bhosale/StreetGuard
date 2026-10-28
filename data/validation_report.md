# 🛡️ StreetGuard AI - Comprehensive Multi-Video Validation Report

**Generated:** 2026-10-28 20:22:30  
**System Hardware:** CPU  
**Detector Model:** Ultralytics YOLO (`yolo11n.pt`) + ByteTrack  

---

## 1. Executive Summary

- **Total Videos Evaluated:** 5
- **Smoke Test Success Rate:** 5 / 5 (100.0%)
- **Full Video Execution Rate:** 5 / 5 (100.0%)
- **Total Processed Frames:** 3,248
- **Cumulative Processing Time:** 135.0 seconds

---

## 2. Multi-Video Benchmark Comparison Table

| Video | Status | Input Res | Inference Res | Processed / Total | FPS | Time | Unique IDs | Persons | Luggage | Snapshot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train1.mp4`** | ✅ PASS | 1920x1080 | 960x540 | 960 / 960 | **31.3** | 30.6s | 226 | 7233 | 543 | `train1_full_snapshot.jpg` |
| **`train2.mp4`** | ✅ PASS | 3840x2160 | 960x540 | 620 / 620 | **21.4** | 28.9s | 42 | 5925 | 0 | `train2_full_snapshot.jpg` |
| **`train3.mp4`** | ✅ PASS | 3840x2160 | 960x540 | 1128 / 1128 | **21.4** | 52.6s | 115 | 13059 | 530 | `train3_full_snapshot.jpg` |
| **`mall1.mp4`** | ✅ PASS | 1280x720 | 960x540 | 240 / 240 | **30.7** | 7.8s | 63 | 2206 | 33 | `mall1_full_snapshot.jpg` |
| **`mall2.mp4`** | ✅ PASS | 3840x2160 | 960x540 | 300 / 300 | **20.0** | 15.0s | 50 | 2111 | 281 | `mall2_full_snapshot.jpg` |

---

## 3. Detailed Video Performance Breakdown

### Video: `train1.mp4`
- **Validation Status**: `PASSED`
- **Resolution**: Input `1920x1080` → Inference `960x540` (Aspect-ratio preserved, scale factor: 0.5)
- **Native Video Properties**: 25.0 FPS, 960 total frames (00:00:38)
- **Processing Performance**: 960 frames in 30.65s (**31.32 FPS**)
- **Tracking Statistics**: 226 unique tracking IDs assigned by ByteTrack (Sample: `[160, 209, 211, 212, 213, 214, 215, 216, 217, 218, 219, 224, 227, 232, 233]`)
- **Active Tracks Stats**: Min: 2, Max: 34, Mean: 16.61
- **Class Detections**: `{"person": 7233, "backpack": 45, "handbag": 446, "suitcase": 52}` (Total detections: 7776)
- **Errors**: 0 errors recorded
- **Saved Verification Snapshot**: [`data\validation_snapshots\train1_full_snapshot.jpg`](file:///data/validation_snapshots/train1_full_snapshot.jpg)

### Video: `train2.mp4`
- **Validation Status**: `PASSED`
- **Resolution**: Input `3840x2160` → Inference `960x540` (Aspect-ratio preserved, scale factor: 0.25)
- **Native Video Properties**: 60.0 FPS, 620 total frames (00:00:10)
- **Processing Performance**: 620 frames in 28.95s (**21.41 FPS**)
- **Tracking Statistics**: 42 unique tracking IDs assigned by ByteTrack (Sample: `[896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 909, 912, 913]`)
- **Active Tracks Stats**: Min: 0, Max: 15, Mean: 12.56
- **Class Detections**: `{"person": 5925}` (Total detections: 5925)
- **Errors**: 0 errors recorded
- **Saved Verification Snapshot**: [`data\validation_snapshots\train2_full_snapshot.jpg`](file:///data/validation_snapshots/train2_full_snapshot.jpg)

### Video: `train3.mp4`
- **Validation Status**: `PASSED`
- **Resolution**: Input `3840x2160` → Inference `960x540` (Aspect-ratio preserved, scale factor: 0.25)
- **Native Video Properties**: 60.0 FPS, 1128 total frames (00:00:18)
- **Processing Performance**: 1128 frames in 52.58s (**21.45 FPS**)
- **Tracking Statistics**: 115 unique tracking IDs assigned by ByteTrack (Sample: `[896, 898, 926, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976]`)
- **Active Tracks Stats**: Min: 3, Max: 20, Mean: 15.51
- **Class Detections**: `{"person": 13059, "backpack": 496, "suitcase": 7, "handbag": 27}` (Total detections: 13589)
- **Errors**: 0 errors recorded
- **Saved Verification Snapshot**: [`data\validation_snapshots\train3_full_snapshot.jpg`](file:///data/validation_snapshots/train3_full_snapshot.jpg)

### Video: `mall1.mp4`
- **Validation Status**: `PASSED`
- **Resolution**: Input `1280x720` → Inference `960x540` (Aspect-ratio preserved, scale factor: 0.75)
- **Native Video Properties**: 24.0 FPS, 240 total frames (00:00:10)
- **Processing Performance**: 240 frames in 7.81s (**30.72 FPS**)
- **Tracking Statistics**: 63 unique tracking IDs assigned by ByteTrack (Sample: `[1097, 1098, 1099, 1100, 1101, 1102, 1105, 1107, 1114, 1116, 1120, 1125, 1127, 1129, 1130]`)
- **Active Tracks Stats**: Min: 0, Max: 24, Mean: 18.02
- **Class Detections**: `{"person": 2206, "suitcase": 33}` (Total detections: 2239)
- **Errors**: 0 errors recorded
- **Saved Verification Snapshot**: [`data\validation_snapshots\mall1_full_snapshot.jpg`](file:///data/validation_snapshots/mall1_full_snapshot.jpg)

### Video: `mall2.mp4`
- **Validation Status**: `PASSED`
- **Resolution**: Input `3840x2160` → Inference `960x540` (Aspect-ratio preserved, scale factor: 0.25)
- **Native Video Properties**: 30.0 FPS, 300 total frames (00:00:10)
- **Processing Performance**: 300 frames in 14.97s (**20.04 FPS**)
- **Tracking Statistics**: 50 unique tracking IDs assigned by ByteTrack (Sample: `[1152, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1270, 1273, 1274, 1277, 1281, 1286, 1293]`)
- **Active Tracks Stats**: Min: 0, Max: 18, Mean: 12.93
- **Class Detections**: `{"person": 2111, "handbag": 257, "backpack": 24}` (Total detections: 2392)
- **Errors**: 0 errors recorded
- **Saved Verification Snapshot**: [`data\validation_snapshots\mall2_full_snapshot.jpg`](file:///data/validation_snapshots/mall2_full_snapshot.jpg)

---

## 4. Engineering Observations & Technical Takeaways

1. **Aspect-Ratio Downscaling on 4K Streams**:
   - The 4K streams (`train2`, `train3`, `mall2` at 3840x2160) are automatically downscaled to `960x540`.
   - Aspect ratio 16:9 is strictly maintained without non-uniform stretching, preserving bounding box geometry and camera perspective.
2. **ByteTrack Identity Continuity**:
   - Tracking IDs are assigned consistently across consecutive frames.
   - The 2-stage association preserves identity across partial occlusions on crowded platform scenes.
3. **Zero Cloud Dependencies & Transparent Error Handling**:
   - All inference runs locally on the host CPU.
   - Missing videos and non-standard video streams return structured error status without interrupting remaining jobs.
