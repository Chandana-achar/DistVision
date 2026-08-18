"""
detector.py — Object detection module supporting two modes:

1. YOLO COCO mode  — YOLOv8s pretrained on 80 COCO classes.
2. YOLO-World mode — Open-vocabulary detection; can detect ANY object
                     the user specifies (tree, flower, pen, furniture, etc.).
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any, Optional, Set

# ---------------------------------------------------------------------------
# COCO class names (80 classes) — used to auto-route between models
# ---------------------------------------------------------------------------
COCO_CLASS_NAMES: Set[str] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}

# ---------------------------------------------------------------------------
# Model cache — keyed by (model_path, mode) to avoid reloading
# ---------------------------------------------------------------------------
_models: Dict[str, YOLO] = {}


def load_model(model_path: str = "yolov8s.pt") -> YOLO:
    """Load a YOLO model (downloads automatically on first run)."""
    if model_path not in _models:
        _models[model_path] = YOLO(model_path)
    return _models[model_path]


def detect_objects(
    image: np.ndarray,
    confidence_threshold: float = 0.25,
    model_path: str = "yolov8s.pt",
    custom_classes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run YOLO inference on an image.

    Parameters
    ----------
    image : np.ndarray
        BGR or RGB image (H, W, 3).
    confidence_threshold : float
        Minimum confidence to keep a detection.
    model_path : str
        Path/name of the YOLO weights file.
    custom_classes : list[str] or None
        If provided AND model is YOLO-World, set these as detection targets.
        Ignored for standard YOLO COCO models.

    Returns
    -------
    list[dict]
        Each dict has keys:
          - "class_id"   : int
          - "class_name" : str
          - "confidence" : float
          - "bbox"       : (x1, y1, x2, y2) in pixels
          - "bbox_height": float  (y2 - y1)
          - "bbox_width" : float  (x2 - x1)
    """
    model = load_model(model_path)

    # If using YOLO-World and custom classes are specified, set them
    is_world = "world" in model_path.lower()
    if is_world and custom_classes:
        model.set_classes(custom_classes)

    results = model.predict(source=image, conf=confidence_threshold, verbose=False)

    detections: List[Dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "bbox_height": y2 - y1,
                    "bbox_width": x2 - x1,
                }
            )

    return detections


def get_class_names(model_path: str = "yolov8s.pt") -> Dict[int, str]:
    """Return the full mapping of class-id → class-name from the loaded model."""
    model = load_model(model_path)
    return dict(model.names)


# ---------------------------------------------------------------------------
# Smart detection — automatic model routing
# ---------------------------------------------------------------------------

def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Semantic grouping — classes in the same group compete for the same region
# ---------------------------------------------------------------------------
_SEMANTIC_GROUPS: Dict[str, set] = {
    "flora":       {"flower", "rose", "plant", "potted plant", "tree", "leaf", "grass"},
    "container":   {"vase", "cup", "bowl", "bottle", "jar", "can", "bucket"},
    "vehicle":     {"car", "truck", "bus", "motorcycle", "bicycle"},
    "furniture":   {"chair", "couch", "bench", "bed"},
    "electronics": {"tv", "laptop", "monitor", "screen", "cell phone"},
}

# Reverse index: class_name → group key
_CLASS_TO_GROUP: Dict[str, str] = {}
for _grp, _classes in _SEMANTIC_GROUPS.items():
    for _cls in _classes:
        _CLASS_TO_GROUP[_cls] = _grp

# Specificity score — higher = more specific = preferred in overlapping region
_SPECIFICITY: Dict[str, int] = {
    "rose": 5, "flower": 4,                       # most specific flora
    "leaf": 2, "tree": 3, "grass": 2,
    "plant": 1, "potted plant": 1,                 # least specific flora
    "vase": 1,                                     # often a false-positive for flowers
}


def _are_related(class_a: str, class_b: str) -> bool:
    """True if both classes belong to the same semantic group."""
    ga = _CLASS_TO_GROUP.get(class_a)
    gb = _CLASS_TO_GROUP.get(class_b)
    return ga is not None and ga == gb


def _is_contained(inner: tuple, outer: tuple, threshold: float = 0.70) -> bool:
    """True if ≥ threshold of inner box's area lies inside outer box."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if inner_area <= 0:
        return False
    return (inter / inner_area) >= threshold


def _filter_small_detections(
    detections: List[Dict[str, Any]],
    img_h: int,
    img_w: int,
    min_area_frac: float = 0.004,   # bbox must be ≥ 0.4 % of image area
    min_dim_px: int = 25,           # each bbox side must be ≥ 25 px
) -> List[Dict[str, Any]]:
    """Remove detections whose bounding box is too small to be reliable."""
    img_area = img_h * img_w
    min_area = img_area * min_area_frac

    out = []
    for d in detections:
        area = d["bbox_height"] * d["bbox_width"]
        if area >= min_area and d["bbox_height"] >= min_dim_px and d["bbox_width"] >= min_dim_px:
            out.append(d)
    return out


def _merge_detections(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
    related_iou_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Merge two detection lists with semantic awareness.

    Uses a lower IoU threshold for semantically related classes (e.g.
    flower/plant/leaf) so overlapping detections within the same semantic
    group are properly deduplicated.  When two related detections overlap,
    the more specific class wins; ties go to higher confidence.
    """
    merged = list(primary)

    for det in secondary:
        best_idx = -1
        best_iou = 0.0

        for i, existing in enumerate(merged):
            iou = _compute_iou(det["bbox"], existing["bbox"])
            related = _are_related(det["class_name"], existing["class_name"])
            threshold = related_iou_threshold if related else iou_threshold

            # Also check containment — a small box inside a bigger one
            contained = _is_contained(det["bbox"], existing["bbox"]) or \
                        _is_contained(existing["bbox"], det["bbox"])

            if (iou > threshold or contained) and iou > best_iou:
                best_idx = i
                best_iou = iou

        if best_idx >= 0:
            existing = merged[best_idx]
            det_spec = _SPECIFICITY.get(det["class_name"], 2)
            exist_spec = _SPECIFICITY.get(existing["class_name"], 2)

            if det_spec > exist_spec:
                merged[best_idx] = det          # more specific wins
            elif det_spec == exist_spec and det["confidence"] > existing["confidence"]:
                merged[best_idx] = det          # same specificity → higher conf
            # else keep existing
        else:
            merged.append(det)

    return merged


# Small, focused class list for YOLO-World auto-fallback.
# Only non-COCO objects — kept short for speed + accuracy.
_FALLBACK_WORLD_CLASSES: List[str] = [
    "flower", "rose", "tree", "grass", "leaf", "plant",
    "pen", "pencil", "notebook", "paper",
    "shoe", "hat", "glasses", "sunglasses", "helmet", "bag", "purse",
    "door", "window", "sign", "flag", "pole",
    "food", "fruit", "vegetable", "bread",
    "lamp", "mirror", "pillow", "candle",
    "wheel", "tire", "building", "house",
    "guitar", "drum", "camera", "drone",
    "hammer", "brush", "shirt", "jacket", "dress",
]

# Quality thresholds — if COCO results are below these, YOLO-World kicks in
_QUALITY_MAX_CONF = 0.40   # best detection must exceed this
_QUALITY_AVG_CONF = 0.30   # average confidence must exceed this


def smart_detect(
    image: np.ndarray,
    confidence_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Fully automatic smart detection with quality-based model fallback.

    Pipeline:
      1. Run YOLOv8s COCO first (fast, 80 classes).
      2. Quality check — if results are confident → return them.
      3. If weak/empty → run YOLO-World with a small focused list → merge
         with semantic awareness (more specific class wins).
      4. Post-process: filter tiny bounding boxes that produce unreliable
         distance estimates.
    """
    img_h, img_w = image.shape[:2]

    # Pass 1: YOLOv8s COCO (fast)
    coco_detections = detect_objects(
        image,
        confidence_threshold=confidence_threshold,
        model_path="yolov8s.pt",
        custom_classes=None,
    )

    # Quality check — decide if YOLO-World fallback is needed
    needs_fallback = False
    if not coco_detections:
        needs_fallback = True
    else:
        confidences = [d["confidence"] for d in coco_detections]
        max_conf = max(confidences)
        avg_conf = sum(confidences) / len(confidences)
        if max_conf < _QUALITY_MAX_CONF or avg_conf < _QUALITY_AVG_CONF:
            needs_fallback = True

    if needs_fallback:
        # Pass 2: YOLO-World with small focused class list (auto-fallback)
        world_threshold = max(confidence_threshold - 0.05, 0.10)
        world_detections = detect_objects(
            image,
            confidence_threshold=world_threshold,
            model_path="yolov8s-worldv2.pt",
            custom_classes=_FALLBACK_WORLD_CLASSES,
        )
        # Semantic merge — more specific class wins in overlapping regions
        detections = _merge_detections(coco_detections, world_detections)
    else:
        detections = coco_detections

    # Post-process: drop tiny bboxes that produce unreliable distances
    detections = _filter_small_detections(detections, img_h, img_w)

    return detections


# ---------------------------------------------------------------------------
# Default YOLO-World class list — covers COCO + many common extras
# ---------------------------------------------------------------------------
DEFAULT_WORLD_CLASSES: List[str] = [
    # COCO classes
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
    # ---- Extra classes (not in COCO) ----
    "tree", "flower", "rose", "grass", "pen", "pencil", "marker",
    "notebook", "paper", "bag", "shoe", "hat", "cap", "glasses",
    "sunglasses", "watch", "ring", "earring", "necklace", "bracelet",
    "pillow", "blanket", "curtain", "lamp", "candle", "mirror",
    "door", "window", "wall", "floor", "ceiling", "roof",
    "fence", "gate", "bridge", "road", "sidewalk", "building",
    "house", "tower", "sign", "flag", "pole", "wire", "cable",
    "pipe", "wheel", "tire", "engine", "battery", "charger",
    "plug", "socket", "switch", "button", "screen", "monitor",
    "speaker", "headphone", "earphone", "camera", "drone",
    "guitar", "piano", "drum", "violin", "trumpet",
    "ball", "bat", "racket", "net", "goal", "basket",
    "food", "fruit", "vegetable", "meat", "bread", "rice",
    "noodle", "soup", "salad", "cheese", "egg", "milk",
    "juice", "water", "coffee", "tea", "soda", "beer", "wine",
    "plate", "tray", "jar", "can", "box", "basket", "bucket",
    "towel", "cloth", "rope", "chain", "key", "lock",
    "hammer", "screwdriver", "wrench", "pliers", "drill",
    "brush", "comb", "razor", "soap", "shampoo",
    "medicine", "syringe", "bandage", "mask", "glove",
    "helmet", "vest", "jacket", "shirt", "pants", "dress",
    "skirt", "coat", "scarf", "belt", "wallet", "purse",
]
