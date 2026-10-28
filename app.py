"""
StreetGuard AI - Explainable Computer Vision Surveillance Dashboard.
Streamlit application for CCTV analytics with YOLO, ByteTrack,
Real-Time Detection & Tracking Pipeline, and Incident Auditing.
"""

import torch  # noqa: F401 - Must be imported before pyarrow/pandas to avoid Windows C-runtime DLL collision (WinError 1114)
from pathlib import Path
import tempfile
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from streetguard.config import (
    DEFAULT_MONITORED_CLASSES,
    DEFAULT_RESTRICTED_POLYGON_NORMALIZED,
    IntrusionRuleConfig,
    ModelConfig,
    UnattendedRuleConfig,
    VideoProcessingConfig,
    get_default_device,
)
from streetguard.database import IncidentDatabase
from streetguard.pipeline import VideoAnalyticsPipeline
from streetguard.utils import get_video_metadata, seconds_to_hms

# ---------------------------------------------------------
# Streamlit Page Configuration & Dark Surveillance CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="StreetGuard AI - Intelligent Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Surveillance Styling
st.markdown(
    """
    <style>
    /* Dark Surveillance Base */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
    }
    
    /* Metrics and Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b26 0%, #1c2333 100%);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* Live Feed Container */
    .cctv-feed-frame {
        border: 2px solid #00e5ff;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
    }

    /* Alert Badges */
    .alert-card-intrusion {
        background: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #ef4444;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .alert-card-unattended {
        background: rgba(245, 158, 11, 0.15);
        border-left: 4px solid #f59e0b;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize persistent database instance
db = IncidentDatabase()

# ---------------------------------------------------------
# Sidebar Configuration & Parameters
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=64)
st.sidebar.title("StreetGuard AI")
st.sidebar.caption("Intelligent CCTV Video Analytics & Incident Auditing")

# Section 0: Pipeline Operation Mode
st.sidebar.header("🕹️ Operation Mode")
pipeline_mode = st.sidebar.radio(
    "Select Mode",
    [
        "Initial Pipeline: Detection & Tracking Only",
        "Full Surveillance: With Intrusion & Unattended Rules",
    ],
    index=1,
    help="Select Full Surveillance to monitor security rules including intrusion perimeter and unattended luggage heuristics.",
)
is_basic_mode = "Initial Pipeline" in pipeline_mode

# Section 1: Video Ingestion
st.sidebar.header("📹 Video Source")
video_source_mode = st.sidebar.radio(
    "Source Mode",
    ["Available CCTV Footage", "Upload Video", "Local File Path"],
    index=0,
)

video_path_to_process = None

# Scan sample videos in sample_data/
found_videos = {}
sample_dir = Path("sample_data")
if sample_dir.exists():
    for vid in sample_dir.glob("*.mp4"):
        found_videos[vid.name] = str(vid)

if video_source_mode == "Available CCTV Footage":
    if found_videos:
        # Prioritize train1.mp4 at top of list
        sorted_keys = sorted(found_videos.keys())
        if "train1.mp4" in sorted_keys:
            sorted_keys.remove("train1.mp4")
            sorted_keys.insert(0, "train1.mp4")

        selected_video_name = st.sidebar.selectbox(
            "Select CCTV Video",
            sorted_keys,
            index=0,
        )
        video_path_to_process = found_videos[selected_video_name]
    else:
        st.sidebar.warning("No video files detected. Upload a video or enter a file path.")

elif video_source_mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload CCTV MP4", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path_to_process = tfile.name
        st.sidebar.success(f"Loaded: {uploaded_file.name}")

elif video_source_mode == "Local File Path":
    default_path = found_videos.get("train1.mp4", "sample_data/train1.mp4")
    custom_path = st.sidebar.text_input("Absolute/Relative MP4 Path", default_path)
    if Path(custom_path).is_file():
        video_path_to_process = custom_path
    else:
        st.sidebar.error("File does not exist.")

# Section 2: Detector & Tracker Settings
st.sidebar.header("⚙️ Detection & Tracking")
device_choice = get_default_device()
st.sidebar.info(f"Compute Hardware: **{device_choice.upper()}** (Auto-detected)")

conf_thresh = st.sidebar.slider("Person / General Confidence", 0.15, 0.90, 0.35, 0.05)
luggage_conf_thresh = st.sidebar.slider(
    "Luggage Confidence Threshold",
    0.10,
    0.50,
    0.20,
    0.05,
    help="Sensitive threshold specifically for luggage (backpack, suitcase, handbag) without lowering person threshold.",
)
scale_res = st.sidebar.selectbox("Processing Width (Downscale)", [960, 720, 640, 1280, None], index=0)
frame_skip_step = st.sidebar.slider("Frame Step (1 = every frame)", 1, 4, 1)

# Section 3 & 4: Rules (Visible or Active only in Full Mode)
zone_enabled = False
unattended_enabled = False
poly_norm = DEFAULT_RESTRICTED_POLYGON_NORMALIZED
zone_name = "Secure Perimeter A"
stationary_duration = 4.0
nearby_person_radius = 75.0
stationary_disp_thresh = 30.0
person_dwell_thresh = 1.5

if not is_basic_mode:
    st.sidebar.header("🛑 Restricted Area Rules")
    zone_enabled = st.sidebar.checkbox("Enable Intrusion Detection", value=True)
    zone_name = st.sidebar.text_input("Zone Name", "Secure Perimeter A")
    zone_preset = st.sidebar.selectbox(
        "Perimeter Preset",
        ["Center-Right Quad", "Left Corridor", "Full Center Floor", "Custom Quad"],
        index=0,
    )
    if zone_preset == "Center-Right Quad":
        poly_norm = [(0.45, 0.40), (0.85, 0.40), (0.95, 0.88), (0.40, 0.88)]
    elif zone_preset == "Left Corridor":
        poly_norm = [(0.05, 0.35), (0.45, 0.35), (0.45, 0.90), (0.05, 0.90)]
    elif zone_preset == "Full Center Floor":
        poly_norm = [(0.25, 0.35), (0.75, 0.35), (0.85, 0.90), (0.15, 0.90)]
    else:
        x_min = st.sidebar.slider("Zone X Min", 0.0, 1.0, 0.35, 0.05)
        x_max = st.sidebar.slider("Zone X Max", 0.0, 1.0, 0.75, 0.05)
        y_min = st.sidebar.slider("Zone Y Min", 0.0, 1.0, 0.35, 0.05)
        y_max = st.sidebar.slider("Zone Y Max", 0.0, 1.0, 0.85, 0.05)
        poly_norm = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    st.sidebar.header("🧳 Unattended Object Rules")
    unattended_enabled = st.sidebar.checkbox("Enable Unattended Object Detection", value=True)
    stationary_duration = st.sidebar.slider("Stationary Duration (sec)", 2.0, 20.0, 4.0, 0.5)
    nearby_person_radius = st.sidebar.slider("Nearby-Person Radius (px)", 20.0, 200.0, 75.0, 5.0)
    stationary_disp_thresh = st.sidebar.slider("Movement Threshold (px)", 5.0, 50.0, 30.0, 5.0)
    person_dwell_thresh = st.sidebar.slider(
        "Attendant Dwell Threshold (sec)",
        0.5,
        5.0,
        1.5,
        0.5,
        help="Duration a nearby person must remain near the object before being considered an attendant/owner.",
    )

# ---------------------------------------------------------
# Main Page Header & Tabs
# ---------------------------------------------------------
st.title("🛡️ StreetGuard AI")
st.markdown(
    "**Explainable CCTV Surveillance Analytics** powered by Ultralytics YOLO & ByteTrack. "
    f"Current Mode: **{'Initial Video Pipeline (Detection & Tracking)' if is_basic_mode else 'Full Surveillance Mode'}**."
)

tab_live, tab_history, tab_viva = st.tabs(["🎥 Surveillance Feed", "📋 Incident Vault", "🎓 Viva & Architecture Guide"])

# ---------------------------------------------------------
# TAB 1: Live Surveillance Feed
# ---------------------------------------------------------
with tab_live:
    if not video_path_to_process:
        st.warning("⚠️ No video selected. Please specify a video file in the sidebar to begin.")
    else:
        # Validate video file existence and format gracefully
        meta = get_video_metadata(video_path_to_process)
        if not meta.is_valid:
            st.error(f"❌ Cannot process video: {meta.error_message}")
        else:
            # Display Video Stream Metadata Card
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Source File", Path(video_path_to_process).name)
            with col_m2:
                st.metric("Native Resolution", f"{meta.width} x {meta.height}")
            with col_m3:
                st.metric("Video Framerate", f"{meta.fps:.1f} FPS")
            with col_m4:
                st.metric("Total Frames", f"{meta.total_frames} ({seconds_to_hms(meta.duration_seconds)})")

            st.divider()

            # Operational Control Bar
            col_ctrl1, col_ctrl2 = st.columns([3, 7])
            with col_ctrl1:
                run_surveillance = st.button("▶ Start Video Analytics", type="primary", use_container_width=True)

            # Real-time Telemetry Metrics Row
            col_stat1, col_stat2, col_stat3, col_stat4, col_stat5, col_stat6 = st.columns(6)
            m_persons = col_stat1.empty()
            m_objects = col_stat2.empty()
            m_frames = col_stat3.empty()
            m_time = col_stat4.empty()
            m_infer_fps = col_stat5.empty()
            m_fps = col_stat6.empty()

            # Initial metrics display
            m_persons.metric("Persons Detected", 0)
            m_objects.metric("Tracked Objects", 0)
            m_frames.metric("Frames Processed", f"0 / {meta.total_frames}")
            m_time.metric("Video Time", "00:00:00")
            m_infer_fps.metric("Inference Speed", "0.0 FPS")
            m_fps.metric("End-to-End Speed", "0.0 FPS")

            # Layout: Video Monitor
            if is_basic_mode:
                video_frame_placeholder = st.empty()
                progress_bar = st.progress(0.0)
            else:
                col_player, col_feed = st.columns([7, 3])
                with col_player:
                    video_frame_placeholder = st.empty()
                    progress_bar = st.progress(0.0)
                with col_feed:
                    st.subheader("🚨 Security Alerts")
                    alerts_container = st.container(height=480)

            # Diagnostic decision tracer expander
            diag_expander = st.expander("🔍 Unattended Heuristic Decision Trace (7 Stages & Candidate Telemetry)", expanded=False)
            with diag_expander:
                candidate_table_placeholder = st.empty()
                diag_log_placeholder = st.empty()

            # Video Processing Execution Loop
            if run_surveillance:
                try:
                    # Build runtime configs
                    m_config = ModelConfig(
                        model_name="yolo11n.pt",
                        device=device_choice,
                        confidence_threshold=conf_thresh,
                        luggage_confidence_threshold=luggage_conf_thresh,
                        target_classes=DEFAULT_MONITORED_CLASSES,
                    )
                    i_config = IntrusionRuleConfig(
                        enabled=zone_enabled,
                        zone_name=zone_name,
                        polygon_normalized=poly_norm,
                    )
                    u_config = UnattendedRuleConfig(
                        enabled=unattended_enabled,
                        stationary_duration_sec=stationary_duration,
                        stationary_displacement_px=stationary_disp_thresh,
                        nearby_person_radius_px=nearby_person_radius,
                        person_dwell_sec=person_dwell_thresh,
                    )
                    v_config = VideoProcessingConfig(
                        target_width=scale_res,
                        draw_tracking_trails=True,
                    )

                    pipeline = VideoAnalyticsPipeline(
                        model_config=m_config,
                        intrusion_config=i_config,
                        unattended_config=u_config,
                        video_config=v_config,
                        db=db,
                    )
                except Exception as e:
                    st.error(f"❌ Failed to initialize YOLO detection or ByteTrack: {e}")
                    st.stop()

                try:
                    stream = pipeline.process_video_stream(
                        video_path_to_process,
                        frame_skip=frame_skip_step,
                    )

                    recent_alerts_list = []

                    for result in stream:
                        # Update telemetry metrics
                        m_persons.metric("Persons Detected", result.person_count)
                        m_objects.metric("Tracked Objects", len(result.active_tracks))
                        m_frames.metric("Frames Processed", f"{result.frame_idx} / {meta.total_frames}")
                        m_time.metric("Video Time", result.formatted_time)
                        m_infer_fps.metric("Inference Speed", f"{result.inference_fps:.1f} FPS")
                        m_fps.metric("End-to-End Speed", f"{result.fps:.1f} FPS")

                        # Convert BGR OpenCV image to RGB for Streamlit
                        rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                        video_frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                        progress_bar.progress(result.progress_ratio)

                        if not is_basic_mode:
                            for a in result.intrusion_alerts:
                                recent_alerts_list.insert(0, {
                                    "type": "PERIMETER INTRUSION",
                                    "time": result.formatted_time,
                                    "text": a.explanation,
                                })
                            for u in result.unattended_alerts:
                                recent_alerts_list.insert(0, {
                                    "type": "REVIEW REQUIRED: UNATTENDED",
                                    "time": result.formatted_time,
                                    "text": u.explanation,
                                })

                            with alerts_container:
                                if not recent_alerts_list:
                                    st.info("System Armed. Monitoring perimeter and luggage.", icon="🛡️")
                                else:
                                    for item in recent_alerts_list[:8]:
                                        is_intr = "INTRUSION" in item["type"]
                                        badge_style = "alert-card-intrusion" if is_intr else "alert-card-unattended"
                                        badge_icon = "🚨" if is_intr else "🧳"
                                        st.markdown(
                                            f"""
                                            <div class="{badge_style}">
                                                <strong>{badge_icon} {item['type']} @ {item['time']}</strong><br/>
                                                <small>{item['text']}</small>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                            # Refresh diagnostics trace periodically
                            if result.frame_idx % 10 == 0:
                                cand_list = pipeline.unattended_detector.get_candidate_diagnostics()
                                if cand_list:
                                    c_data = [
                                        {
                                            "Track ID": f"#{c.track_id}" if c.track_id is not None else "---",
                                            "Class": c.class_name.upper(),
                                            "Conf": f"{int(c.confidence * 100)}%",
                                            "Observed At": f"{c.first_seen_time:.1f}s",
                                            "Stationary": f"{c.stationary_duration:.1f}s",
                                            "Movement": c.movement_status,
                                            "Nearby Person": c.nearby_person_status,
                                            "Rule State": c.rule_state,
                                            "Decision Reason": c.decision_reason,
                                        }
                                        for c in cand_list
                                    ]
                                    candidate_table_placeholder.dataframe(pd.DataFrame(c_data), use_container_width=True, hide_index=True)

                                recent_diags = pipeline.unattended_detector.get_recent_diagnostics(6)
                                if recent_diags:
                                    diag_md = "\n".join([
                                        f"- **`{d.stage}`** | Track #{d.track_id} ({d.object_category}) @ {d.formatted_time}: {d.message}"
                                        for d in reversed(recent_diags)
                                    ])
                                    diag_log_placeholder.markdown(diag_md)

                    st.success("✅ Surveillance video processing completed.")
                except Exception as e:
                    st.error(f"❌ Error during video processing: {e}")

# ---------------------------------------------------------
# TAB 2: Incident Vault & Evidence Management
# ---------------------------------------------------------
with tab_history:
    st.subheader("🗄️ Security Incident History & Evidence Vault")
    st.caption("All security alerts logged locally in SQLite database with forensic timestamps and snapshots.")

    col_f1, col_f2, col_f3 = st.columns([3, 3, 4])
    with col_f1:
        category_filter = st.selectbox(
            "Filter Event Type",
            ["All Incidents", "intrusion", "unattended_object"],
            index=0,
        )
    with col_f2:
        if st.button("🗑️ Clear All Incidents", use_container_width=True):
            db.clear_incidents()
            st.success("Incident database cleared.")
            st.rerun()
    with col_f3:
        if st.button("🔄 Refresh History", use_container_width=True):
            st.rerun()

    filter_cat = None if category_filter == "All Incidents" else category_filter
    incidents = db.get_incidents(category=filter_cat, limit=100)

    if not incidents:
        st.info("No security incidents recorded yet. Run a surveillance analysis with alert rules enabled to log events.")
    else:
        table_data = []
        for inc in incidents:
            table_data.append({
                "ID": inc.id,
                "Category": inc.event_category.upper(),
                "Video": inc.video_name,
                "Timestamp": inc.formatted_timestamp,
                "Track ID": f"#{inc.track_id}" if inc.track_id is not None else "N/A",
                "Object": inc.object_category,
                "Reason / Explainability": inc.reason,
            })
        df_incidents = pd.DataFrame(table_data)
        try:
            st.dataframe(df_incidents, use_container_width=True, hide_index=True)
        except Exception as df_err:
            st.warning(f"Interactive table renderer note: {df_err}. Displaying standard table view.")
            st.table(df_incidents)

        st.divider()
        st.subheader("📸 Incident Snapshot Evidence Gallery")

        cols = st.columns(3)
        for idx, inc in enumerate(incidents[:12]):
            col = cols[idx % 3]
            with col:
                st.markdown(
                    f"**Incident #{inc.id}: {inc.event_category.upper()}**  \n"
                    f"⏱️ `{inc.formatted_timestamp}` | Track `#{inc.track_id}` ({inc.object_category})"
                )
                if inc.screenshot_path and Path(inc.screenshot_path).is_file():
                    img = Image.open(inc.screenshot_path)
                    st.image(img, use_container_width=True)
                else:
                    st.caption("Snapshot not available.")
                st.caption(f"ℹ️ {inc.reason}")
                st.write("---")

# ---------------------------------------------------------
# TAB 3: Viva & Explainability Guide
# ---------------------------------------------------------
with tab_viva:
    st.subheader("🎓 Technical Architecture & Viva Defense Reference")
    st.markdown(
        """
        ### 1. Object Detection (Ultralytics YOLO)
        - **Model Architecture**: Ultralytics YOLO (e.g. `yolo11n.pt`), a lightweight single-stage convolutional/attention vision model.
        - **Surveillance Classes Monitored**: `person (0)`, `backpack (24)`, `handbag (26)`, `suitcase (28)`.
        - **Explainability**: Confidence scores indicate detection certainty, with bounding boxes displaying spatial location.

        ### 2. Multi-Object Tracking (ByteTrack)
        - **Algorithm**: ByteTrack associates detections across consecutive frames using motion prediction (Kalman filtering) and visual overlap (IoU association).
        - **Low-Score Association**: Unlike SORT which discards low-confidence detections, ByteTrack splits detections into high and low scores to recover occluded objects.
        - **Track IDs**: Track IDs are dynamic tracking identifiers (`#1`, `#2`, ...); StreetGuard safely handles missing IDs and avoids treating them as permanent biometric identities.

        ### 3. Video Pipeline Telemetry
        - **Frame-by-Frame Tracking**: Shows real-time video FPS, processed frame counts, elapsed video time, and active person counts.
        - **Adaptive Downscaling**: High-resolution 1080p and 4K CCTV streams are automatically downscaled for smooth real-time CPU execution.
        """
    )
