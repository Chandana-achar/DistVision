# 📷 DistVision — Object Detection & Distance Estimation

**AI Intern Case Study — Basic Assessment**

---

## 1. Approach

- **Problem:** Design a system that detects objects (people, vehicles, etc.) from a camera feed and estimates their distance.
- **Solution:** A Streamlit web app with two input modes — image upload and live webcam capture — that runs object detection and monocular distance estimation in a single pipeline.

**Steps from camera input to final output:**

1. **Input** — User uploads an image or captures a webcam frame via the browser.
2. **Object Detection** — The frame is passed through YOLOv8s (80 COCO classes). If results are weak/empty, YOLO-World runs as an automatic fallback for open-vocabulary detection.
3. **AI Depth Estimation** — Depth Anything V2 (Small) generates a per-pixel relative depth map from the same frame.
4. **Distance Calculation** — A hybrid pipeline combines two methods:
   - **Similar-triangles:** uses known object heights and camera focal length.
   - **AI depth:** uses the depth map, anchored to a real-world scale via the most confident detection.
   - Final distance = weighted blend (60% AI + 40% similar-triangles).
5. **Annotation & Display** — Bounding boxes, class labels, confidence scores, and estimated distances are drawn on the image. A detection details table and optional depth-map visualisation are also shown.

---

## 2. Distance Estimation

**Method used:** Hybrid approach (Similar-Triangles + AI Depth Estimation).

### Similar-Triangles (Primary)

```
Distance = (Known real-world height × Focal length in px) / Bounding-box height in px
```

- A lookup table of approximate real-world heights for all 80 COCO classes + 100+ extra classes is maintained in `distance.py`.
- The focal length defaults to 700 px (typical for 1080p webcams) and can be calibrated in-app using a reference object.

### AI Depth Estimation (Enhancement)

- **Depth Anything V2** (Small variant) produces a relative depth map for the entire image.
- The median depth within each bounding box is extracted (center-cropped to avoid background contamination).
- The most confident detection is used as an **anchor** to convert relative depth → absolute metres.
- Final distance = `0.6 × AI_distance + 0.4 × Similar_triangles_distance`.

### Why this method?

- Similar-triangles is simple, fast, and requires no special hardware — works with any monocular camera.
- AI depth adds spatial awareness (objects at similar image sizes but different real depths get differentiated).
- The hybrid blend corrects errors from both methods — similar-triangles can be wrong due to pose/size variation, and AI depth is relative-only without an anchor.

---

## 3. Model Choice

| Purpose | Model | Why chosen |
|---------|-------|------------|
| **Object Detection** | **YOLOv8s** (Ultralytics) | Fast real-time inference, 80 pre-trained COCO classes, lightweight (~22 MB), well-documented, easy to integrate via the `ultralytics` Python package. |
| **Open-Vocabulary Fallback** | **YOLO-World v2** (Small) | Detects ANY user-specified object (flower, pen, tree, etc.) beyond the 80 COCO classes. Auto-triggered only when YOLOv8s results are weak. |
| **Depth Estimation** | **Depth Anything V2** (Small) | State-of-the-art monocular depth model, ~100 MB, works on CPU (~1 s/image) and GPU (~50 ms/image). Available on Hugging Face for easy download. |

**Why YOLOv8 over alternatives?**

- **vs. YOLOv5:** YOLOv8 has better accuracy (higher mAP on COCO) and a cleaner Python API.
- **vs. Faster R-CNN / SSD:** YOLO is significantly faster for real-time use; two-stage detectors are too slow for interactive apps.
- **vs. YOLOv8n (nano):** We use `v8s` (small) instead of nano for better accuracy with only a minor speed trade-off — still real-time on CPU.

---

## 4. Data

### What type of data is needed?

- **Pre-trained weights** — The models come pre-trained:
  - YOLOv8s: trained on **COCO dataset** (330K images, 80 object categories).
  - Depth Anything V2: trained on large-scale synthetic + real depth datasets.
- **No custom training data is required** — the system works out-of-the-box with pre-trained weights.

### How is data obtained?

- **YOLOv8s weights** (`yolov8s.pt`) are auto-downloaded from Ultralytics on first run.
- **YOLO-World weights** (`yolov8s-worldv2.pt`) are auto-downloaded similarly.
- **Depth Anything V2 weights** are auto-downloaded from Hugging Face (`depth-anything/Depth-Anything-V2-Small-hf`).
- **User input data** — images are provided by the user at runtime via upload or webcam; no dataset collection is needed.
- **Object height lookup** — a manually curated table of approximate real-world heights for 160+ object classes is maintained in `distance.py`.

---

## 5. Deployment Idea

### Where will the system run?

- **Primary target:** Runs as a **Streamlit web app** on a laptop or cloud server.
- **Cloud deployment:** Deploys directly to **Streamlit Community Cloud** — just push the repo to GitHub and connect.
- **Local use:** `streamlit run app.py` on any machine with Python 3.9+.

### How is it made efficient?

- **Model caching:** All models (YOLOv8s, YOLO-World, Depth Anything) are loaded once (singleton pattern) and reused across requests.
- **Smart fallback:** YOLO-World only runs when YOLOv8s results are weak (quality-based gating), avoiding unnecessary computation.
- **Small model variants:** YOLOv8s (~22 MB) and Depth Anything V2-Small (~100 MB) are chosen for speed on CPU.
- **Headless OpenCV:** Uses `opencv-python-headless` — no GUI dependencies needed on cloud VMs.
- **Lazy imports:** Heavy libraries (PyTorch, Transformers) are imported only when needed.

---

## 6. Challenges

### Challenge 1: Inaccurate distance without camera calibration

- **Problem:** The similar-triangles method requires an accurate focal length. Default values can be significantly off for different cameras.
- **Solution:** Built an in-app calibration tool — the user provides a reference object's known height, distance, and bounding-box height to compute the correct focal length. Additionally, the AI depth model provides a second independent estimate, and blending the two reduces calibration sensitivity.

### Challenge 2: Detecting objects outside the 80 COCO classes

- **Problem:** YOLOv8 only recognises 80 classes. Common objects like flowers, trees, pens, or furniture items are missed.
- **Solution:** Implemented a smart auto-fallback to YOLO-World (open-vocabulary detector) with a curated list of 40+ extra classes. A semantic merge pipeline deduplicates overlapping detections — more specific labels (e.g., "rose") win over generic ones (e.g., "plant").

### Challenge 3: AI depth map gives relative values, not absolute metres

- **Problem:** Depth Anything V2 outputs relative depth (higher = farther) but doesn't give real-world distances in metres.
- **Solution:** The most confident detection's similar-triangles distance is used as an **anchor** to convert the entire relative depth map into absolute metres. This anchoring approach lets the AI depth model improve relative ordering between objects while the similar-triangles method grounds it in real-world scale.

---

## Project Structure

```
DistVision/
├── app.py              # Streamlit UI (image upload + webcam modes)
├── detector.py         # YOLOv8s + YOLO-World detection with smart fallback
├── distance.py         # Distance estimation (similar-triangles + hybrid AI)
├── depth_estimator.py  # Depth Anything V2 wrapper (monocular depth maps)
├── requirements.txt    # Python dependencies
└── README.md           # This file (case study answers)
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd DistVision

# 2. Create a virtual env (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.
Model weights are downloaded automatically on first run.

---

## Tech Stack

- **Python 3.9+**
- **Streamlit** — Web UI framework
- **Ultralytics YOLOv8** — Object detection
- **YOLO-World** — Open-vocabulary detection
- **Depth Anything V2** — Monocular depth estimation (via Hugging Face Transformers)
- **OpenCV** — Image processing and annotation
- **NumPy / Pillow** — Array and image utilities

---

## License

MIT
