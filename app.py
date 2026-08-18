"""
app.py — Streamlit web app for Object Detection + Distance Estimation.

Modes:
  1. Image Upload  — upload a photo → detect + annotate → side-by-side view.
  2. Live Webcam   — browser camera capture → detect + annotate per frame.

Models:
  • YOLOv8s (COCO)  — 80 fixed classes, fast and reliable.
  • YOLO-World      — Open-vocabulary; detects ANY object you specify
                      (tree, flower, pen, furniture, etc.).
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from detector import smart_detect
from distance import (
    estimate_distance,
    get_real_world_height,
    compute_hybrid_distances,
    calibrate_focal_length,
    DEFAULT_FOCAL_LENGTH_PX,
)
from depth_estimator import get_depth_estimator, depth_map_to_colormap

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DistVision — Object Detection & Distance Estimation",
    page_icon="📷",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, modern look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide default Streamlit hamburger & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }

    /* Title bar */
    .title-container {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .title-container h1 {
        background: linear-gradient(90deg, #a78bfa, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .title-container p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.25rem;
    }

    /* Detection stat cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        backdrop-filter: blur(6px);
    }
    .stat-card h3 {
        color: #a78bfa;
        font-size: 2rem;
        margin: 0;
    }
    .stat-card p {
        color: #94a3b8;
        margin: 0.2rem 0 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Colour palette for bounding boxes (20 distinct colours, cycled)
# ---------------------------------------------------------------------------
BOX_COLORS = [
    (167, 139, 250),   # violet
    (109, 213, 237),   # cyan
    (252, 211, 77),    # amber
    (248, 113, 113),   # red
    (52, 211, 153),    # emerald
    (251, 146, 60),    # orange
    (129, 140, 248),   # indigo
    (244, 114, 182),   # pink
    (163, 230, 53),    # lime
    (56, 189, 248),    # sky
    (232, 121, 249),   # fuchsia
    (250, 204, 21),    # yellow
    (74, 222, 128),    # green
    (253, 164, 175),   # rose
    (147, 197, 253),   # blue
    (253, 186, 116),   # light orange
    (134, 239, 172),   # light green
    (196, 181, 253),   # light violet
    (254, 202, 202),   # light red
    (165, 243, 252),   # light cyan
]


def _color_for_class(class_id: int):
    return BOX_COLORS[class_id % len(BOX_COLORS)]


# ---------------------------------------------------------------------------
# Annotate image
# ---------------------------------------------------------------------------
def annotate_image(
    image: np.ndarray,
    detections: list,
    focal_length: float,
) -> np.ndarray:
    """Draw bounding boxes, labels, and distance on a copy of the image."""
    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]
    # Scale font/thickness relative to image size
    scale = max(h_img, w_img) / 1000
    thickness = max(int(2 * scale), 1)
    font_scale = max(0.5 * scale, 0.4)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls = det["class_name"]
        conf = det["confidence"]
        color = _color_for_class(det["class_id"])

        # Use hybrid distance if available, otherwise fall back
        dist = det.get("hybrid_distance", estimate_distance(cls, det["bbox_height"], focal_length))
        method = det.get("distance_method", "Estimated")
        prefix = "~" if method == "Estimated" else ""
        dist_str = f"{prefix}{dist:.1f}m" if dist > 0 else "?"

        label = f"{cls} {conf:.0%} | {dist_str}"

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# Sidebar — settings & calibration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.radio("Input mode", ["📸 Image Upload", "📹 Webcam Capture"], index=0)

    st.markdown("---")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower values detect more objects but may include false positives.",
    )

    st.markdown("---")
    st.markdown("## 📐 Focal Length Calibration")
    st.caption(
        "Provide a reference measurement to improve distance accuracy. "
        "Leave defaults to use the built-in estimate (≈ 700 px for a typical 1080p webcam)."
    )

    cal_height = st.number_input("Reference object height (m)", value=1.70, step=0.05, format="%.2f")
    cal_distance = st.number_input("Distance to reference object (m)", value=3.00, step=0.1, format="%.2f")
    cal_bbox = st.number_input("Bounding-box height in reference image (px)", value=400.0, step=10.0, format="%.0f")

    if st.button("Calibrate"):
        try:
            fl = calibrate_focal_length(cal_height, cal_distance, cal_bbox)
            st.session_state["focal_length"] = fl
            st.success(f"Focal length set to **{fl:.1f} px**")
        except ValueError as exc:
            st.error(str(exc))

    focal = st.session_state.get("focal_length", DEFAULT_FOCAL_LENGTH_PX)
    st.info(f"Current focal length: **{focal:.1f} px**")

    st.markdown("---")
    st.markdown("## 🧠 AI Depth Estimation")
    use_ai_depth = st.checkbox(
        "Enable AI depth (Depth Anything V2)",
        value=True,
        help=(
            "Uses a neural network to estimate per-pixel depth, then combines "
            "it with the similar-triangles method for more accurate distances. "
            "Adds ~1–2 s processing time per image on CPU."
        ),
    )
    show_depth_map = st.checkbox(
        "Show depth map",
        value=False,
        help="Display the AI-generated depth map alongside the detected image.",
    )
    if use_ai_depth:
        st.success("AI depth: **enabled** — distances will be AI-enhanced")
    else:
        st.caption("AI depth disabled — using similar-triangles only.")

    st.markdown("---")
    st.markdown(
        "Built with [YOLOv8](https://docs.ultralytics.com/) · "
        "[YOLO-World](https://docs.ultralytics.com/models/yolo-world/) · "
        "[Depth Anything V2](https://huggingface.co/depth-anything) · "
        "[Streamlit](https://streamlit.io)"
    )

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="title-container">
        <h1>📷 DistVision</h1>
        <p>Real-time Object Detection &amp; Monocular Distance Estimation powered by YOLOv8 &amp; YOLO-World</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper: run detection + display results
# ---------------------------------------------------------------------------
def run_detection_and_display(image_np_rgb, show_side_by_side=True):
    """Run detection on an RGB numpy image and display annotated results."""
    image_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Detecting objects…"):
        detections = smart_detect(
            image_bgr,
            confidence_threshold=confidence,
        )

    # Apply confidence threshold consistently to final displayed detections
    detections = [d for d in detections if d["confidence"] >= confidence]

    # --- AI Depth Estimation (hybrid) ---
    depth_map = None
    if use_ai_depth and detections:
        with st.spinner("🧠 Running AI depth estimation…"):
            try:
                estimator = get_depth_estimator()
                depth_map = estimator.get_depth_map(image_np_rgb)
            except Exception as e:
                st.warning(f"AI depth failed: {e}. Falling back to similar-triangles.")
                depth_map = None

    # Compute distances (hybrid if depth_map available, else similar-triangles)
    if detections:
        detections = compute_hybrid_distances(
            detections,
            depth_map=depth_map,
            focal_length_px=focal,
        )

    annotated_bgr = annotate_image(image_bgr, detections, focal)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Stats row — use hybrid distance when available
    unique_classes = set(d["class_name"] for d in detections)
    avg_dist_vals = [
        d.get("hybrid_distance", estimate_distance(d["class_name"], d["bbox_height"], focal))
        for d in detections
    ]
    avg_dist = np.mean([v for v in avg_dist_vals if v > 0]) if any(v > 0 for v in avg_dist_vals) else 0

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            f'<div class="stat-card"><h3>{len(detections)}</h3><p>Objects Detected</p></div>',
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f'<div class="stat-card"><h3>{len(unique_classes)}</h3><p>Unique Classes</p></div>',
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            f'<div class="stat-card"><h3>{avg_dist:.1f}m</h3><p>Avg Distance</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Image display ---
    if show_side_by_side:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original")
            st.image(image_np_rgb, width="stretch")
        with col2:
            st.markdown("#### Detected + Distance")
            st.image(annotated_rgb, width="stretch")
    else:
        st.markdown("#### Detected + Distance")
        st.image(annotated_rgb, width="stretch")

    # --- Depth map visualisation ---
    if show_depth_map and depth_map is not None:
        st.markdown("#### 🧠 AI Depth Map")
        depth_colored = depth_map_to_colormap(depth_map)
        st.image(depth_colored, width="stretch", caption="Warm = close, Cool = far")

    # --- Detection table ---
    if detections:
        st.markdown("#### Detection Details")
        table_data = []
        for d in detections:
            obj_height = get_real_world_height(d["class_name"])
            dist = d.get("hybrid_distance", estimate_distance(d["class_name"], d["bbox_height"], focal))
            method = d.get("distance_method", "Estimated")
            prefix = "≈ " if method == "Estimated" else ""
            dist_display = f"{prefix}{dist:.2f}" if dist > 0 else "N/A"
            table_data.append(
                {
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "BBox Height (px)": f"{d['bbox_height']:.0f}",
                    "Object Height Used (m)": f"{obj_height:.2f}",
                    "Est. Distance (m)": dist_display,
                    "Status": method,
                }
            )
        st.dataframe(table_data, width="stretch")
    elif image_np_rgb is not None:
        st.warning(
            "No objects detected. Try:\n"
            "- Lowering the confidence threshold\n"
            "- Adding specific object names in the **Extra Objects** box\n"
            "  (e.g. flower, tree, pen)"
        )


# ---------------------------------------------------------------------------
# Mode 1: Image Upload
# ---------------------------------------------------------------------------
if mode == "📸 Image Upload":
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")
        image_np = np.array(pil_image)
        run_detection_and_display(image_np, show_side_by_side=True)
    else:
        st.info("👆 Upload an image to get started.")

# ---------------------------------------------------------------------------
# Mode 2: Webcam Capture
# ---------------------------------------------------------------------------
elif mode == "📹 Webcam Capture":
    st.markdown(
        "> **Note:** Your browser will ask for camera permission. "
        "On Streamlit Community Cloud (HTTPS), this works out of the box. "
        "Locally on HTTP, use Chrome/Edge with `localhost`."
    )

    camera_image = st.camera_input("Capture a frame from your webcam")

    if camera_image is not None:
        pil_image = Image.open(camera_image).convert("RGB")
        image_np = np.array(pil_image)
        run_detection_and_display(image_np, show_side_by_side=False)
    else:
        st.info("📹 Click the camera button above to capture a frame.")
