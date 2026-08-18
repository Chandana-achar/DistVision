"""
distance.py — Monocular distance estimation using the similar-triangles method.

Distance = (Known real-world height × Focal length) / Bounding-box height in pixels

Includes:
  • A calibration helper to compute focal length from a reference image.
  • A comprehensive lookup of approximate real-world heights (meters) for
    ALL 80 COCO classes, with a sensible fallback for any unlisted class.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# 1.  Default / calibrated focal length
# ---------------------------------------------------------------------------
# This constant is computed once via `calibrate_focal_length()`.
# Default value is a reasonable starting point for a 1080p webcam (~3.6 mm
# lens, sensor pixel size ≈ 2.8 µm → focal ≈ 700 px).  Override via
# calibration for best accuracy.
DEFAULT_FOCAL_LENGTH_PX: float = 700.0


def calibrate_focal_length(
    known_height_m: float,
    known_distance_m: float,
    bbox_height_px: float,
) -> float:
    """
    Compute the camera's focal length in pixels from a single reference
    measurement.

    Parameters
    ----------
    known_height_m : float
        Real-world height of the reference object (metres).
    known_distance_m : float
        Real-world distance from camera to the reference object (metres).
    bbox_height_px : float
        Height of the object's bounding box in the reference image (pixels).

    Returns
    -------
    float
        Estimated focal length in pixels.
    """
    if bbox_height_px <= 0:
        raise ValueError("Bounding-box height must be > 0 pixels.")
    focal_length = (bbox_height_px * known_distance_m) / known_height_m
    return focal_length


# ---------------------------------------------------------------------------
# 2.  Real-world heights for ALL 80 COCO classes (metres)
# ---------------------------------------------------------------------------
# Sources: average adult human height, vehicle specs, common furniture
# dimensions, typical animal sizes, etc.  Values are intentionally
# approximate — monocular distance estimation is inherently rough.

COCO_CLASS_HEIGHTS: Dict[str, float] = {
    # ---- People ----
    "person": 1.70,

    # ---- Vehicles ----
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.10,
    "airplane": 12.0,
    "bus": 3.00,
    "train": 3.50,
    "truck": 2.50,
    "boat": 2.00,

    # ---- Traffic / Street ----
    "traffic light": 0.90,
    "fire hydrant": 0.50,
    "stop sign": 0.75,
    "parking meter": 1.20,

    # ---- Animals ----
    "bird": 0.20,
    "cat": 0.30,
    "dog": 0.50,
    "horse": 1.60,
    "sheep": 0.80,
    "cow": 1.50,
    "elephant": 3.00,
    "bear": 1.50,
    "zebra": 1.40,
    "giraffe": 5.50,

    # ---- Accessories ----
    "backpack": 0.50,
    "umbrella": 1.00,
    "handbag": 0.35,
    "tie": 0.55,
    "suitcase": 0.60,

    # ---- Sports ----
    "frisbee": 0.03,
    "skis": 1.70,
    "snowboard": 1.50,
    "sports ball": 0.22,
    "kite": 1.00,
    "baseball bat": 1.05,
    "baseball glove": 0.30,
    "skateboard": 0.15,
    "surfboard": 2.00,
    "tennis racket": 0.69,

    # ---- Kitchen / Food ----
    "bottle": 0.25,
    "wine glass": 0.23,
    "cup": 0.12,
    "fork": 0.19,
    "knife": 0.25,
    "spoon": 0.18,
    "bowl": 0.10,
    "banana": 0.20,
    "apple": 0.08,
    "sandwich": 0.10,
    "orange": 0.08,
    "broccoli": 0.20,
    "carrot": 0.20,
    "hot dog": 0.15,
    "pizza": 0.30,
    "donut": 0.10,
    "cake": 0.15,

    # ---- Furniture / Indoor ----
    "chair": 0.90,
    "couch": 0.90,
    "potted plant": 0.40,
    "bed": 0.60,
    "dining table": 0.75,
    "toilet": 0.45,

    # ---- Electronics ----
    "tv": 0.60,
    "laptop": 0.25,
    "mouse": 0.04,
    "remote": 0.20,
    "keyboard": 0.05,
    "cell phone": 0.15,

    # ---- Household ----
    "microwave": 0.30,
    "oven": 0.60,
    "toaster": 0.20,
    "sink": 0.25,
    "refrigerator": 1.80,
    "book": 0.24,
    "clock": 0.30,
    "vase": 0.30,
    "scissors": 0.20,
    "teddy bear": 0.40,
    "hair drier": 0.25,
    "toothbrush": 0.19,

    # ==== Extra classes for YOLO-World (not in COCO) ====
    # Nature
    "tree": 5.00,
    "flower": 0.30,
    "rose": 0.40,
    "grass": 0.15,

    # Stationery / Office
    "pen": 0.15,
    "pencil": 0.18,
    "marker": 0.15,
    "notebook": 0.30,
    "paper": 0.30,

    # Clothing / Accessories
    "bag": 0.40,
    "shoe": 0.12,
    "hat": 0.15,
    "cap": 0.15,
    "glasses": 0.05,
    "sunglasses": 0.05,
    "watch": 0.04,
    "ring": 0.02,
    "earring": 0.03,
    "necklace": 0.30,
    "bracelet": 0.05,
    "helmet": 0.25,
    "vest": 0.60,
    "jacket": 0.70,
    "shirt": 0.70,
    "pants": 1.00,
    "dress": 1.20,
    "skirt": 0.60,
    "coat": 0.90,
    "scarf": 1.50,
    "belt": 0.05,
    "wallet": 0.10,
    "purse": 0.25,

    # Home
    "pillow": 0.40,
    "blanket": 1.50,
    "curtain": 2.00,
    "lamp": 0.50,
    "candle": 0.20,
    "mirror": 0.60,
    "door": 2.00,
    "window": 1.20,
    "towel": 0.70,
    "cloth": 0.50,
    "rope": 1.00,
    "key": 0.06,
    "lock": 0.08,

    # Building / Outdoor
    "building": 10.0,
    "house": 6.00,
    "tower": 15.0,
    "sign": 0.60,
    "flag": 1.00,
    "pole": 4.00,
    "fence": 1.50,
    "gate": 2.00,
    "bridge": 5.00,
    "road": 3.00,
    "sidewalk": 2.00,

    # Electronics (extra)
    "screen": 0.40,
    "monitor": 0.45,
    "speaker": 0.30,
    "headphone": 0.20,
    "earphone": 0.05,
    "camera": 0.12,
    "drone": 0.30,
    "charger": 0.08,

    # Tools
    "hammer": 0.30,
    "screwdriver": 0.25,
    "wrench": 0.25,
    "pliers": 0.20,
    "drill": 0.30,
    "brush": 0.25,
    "comb": 0.18,

    # Food (extra)
    "food": 0.15,
    "fruit": 0.10,
    "vegetable": 0.15,
    "meat": 0.10,
    "bread": 0.15,
    "rice": 0.08,
    "cheese": 0.10,
    "egg": 0.06,
    "milk": 0.25,
    "juice": 0.25,
    "water": 0.25,
    "coffee": 0.12,
    "tea": 0.12,
    "plate": 0.03,
    "tray": 0.05,
    "jar": 0.20,
    "can": 0.12,
    "box": 0.30,
    "bucket": 0.35,

    # Musical instruments
    "guitar": 1.00,
    "piano": 1.00,
    "drum": 0.50,
    "violin": 0.60,

    # Sports (extra)
    "ball": 0.22,
    "bat": 0.90,
    "racket": 0.65,
    "net": 1.00,

    # Vehicles (extra)
    "wheel": 0.65,
    "tire": 0.65,
    "bench": 0.45,
}

# Fallback height for any class not explicitly listed above.
_FALLBACK_HEIGHT_M: float = 0.50


def get_real_world_height(class_name: str) -> float:
    """Return the approximate real-world height (m) for a COCO class name."""
    return COCO_CLASS_HEIGHTS.get(class_name, _FALLBACK_HEIGHT_M)


# ---------------------------------------------------------------------------
# 3.  Distance estimation
# ---------------------------------------------------------------------------

def estimate_distance(
    class_name: str,
    bbox_height_px: float,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
) -> float:
    """
    Estimate the distance (metres) from the camera to an object using
    the similar-triangles method.

    Distance = (real_height × focal_length) / bbox_height

    Parameters
    ----------
    class_name : str
        COCO class name (e.g. "person", "car").
    bbox_height_px : float
        Height of the object's bounding box in pixels.
    focal_length_px : float
        Camera focal length in pixels (from calibration).

    Returns
    -------
    float
        Estimated distance in metres, or -1.0 if bbox_height_px ≤ 0.
    """
    if bbox_height_px <= 0:
        return -1.0

    real_height = get_real_world_height(class_name)
    distance = (real_height * focal_length_px) / bbox_height_px
    return round(distance, 2)


# ---------------------------------------------------------------------------
# 4.  Hybrid distance estimation (AI depth + similar-triangles)
# ---------------------------------------------------------------------------

def compute_hybrid_distances(
    detections: list,
    depth_map,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    ai_weight: float = 0.6,
) -> list:
    """
    Combine AI depth estimation with similar-triangles for more accurate
    distance measurements.

    Algorithm:
      1. Compute similar-triangles distance for every detection.
      2. Extract AI relative depth (median within bbox) for each detection.
      3. Use the most confident detection as an **anchor** to convert
         AI relative depth → absolute metres.
      4. Blend the two estimates:  final = ai_weight×AI + (1-ai_weight)×ST.

    Parameters
    ----------
    detections : list[dict]
        Detection dicts from detector.py (must have 'bbox', 'bbox_height',
        'class_name', 'confidence' keys).
    depth_map : np.ndarray or None
        Per-pixel depth map from DepthEstimator.get_depth_map().
        If None, falls back to similar-triangles only.
    focal_length_px : float
        Camera focal length in pixels.
    ai_weight : float
        Weight for the AI depth estimate in the blend (0.0–1.0).
        Default 0.6 means 60% AI, 40% similar-triangles.

    Returns
    -------
    list[dict]
        Same detections list, with added keys:
          - "hybrid_distance" : float  (final blended distance in metres)
          - "st_distance"     : float  (similar-triangles distance)
          - "ai_distance"     : float  (AI-scaled distance, or -1 if N/A)
          - "distance_method" : str    ("AI-Enhanced" or "Estimated")
    """
    import numpy as np
    from depth_estimator import get_depth_estimator

    if not detections:
        return detections

    # Step 1: Compute similar-triangles distance for each detection
    for det in detections:
        st_dist = estimate_distance(
            det["class_name"], det["bbox_height"], focal_length_px
        )
        det["st_distance"] = st_dist
        det["ai_distance"] = -1.0
        det["distance_method"] = "Estimated"

    # If no depth map is available, return similar-triangles only
    if depth_map is None:
        for det in detections:
            det["hybrid_distance"] = det["st_distance"]
        return detections

    # Step 2: Extract AI relative depth for each detection
    estimator = get_depth_estimator()
    for det in detections:
        ai_depth = estimator.get_object_depth(depth_map, det["bbox"])
        det["_ai_raw_depth"] = ai_depth

    # Step 3: Find the anchor — highest confidence detection with valid data
    valid_dets = [
        d for d in detections
        if d["st_distance"] > 0 and d["_ai_raw_depth"] > 0
    ]

    if not valid_dets:
        # No valid anchor — fall back to similar-triangles
        for det in detections:
            det["hybrid_distance"] = det["st_distance"]
        return detections

    anchor = max(valid_dets, key=lambda d: d["confidence"])

    # Scale factor: convert AI relative depth → metres
    scale = anchor["st_distance"] / anchor["_ai_raw_depth"]

    # Step 4: Compute hybrid distance for each detection
    for det in detections:
        raw = det["_ai_raw_depth"]
        if raw > 0:
            ai_metric = raw * scale
            det["ai_distance"] = round(ai_metric, 2)
            det["hybrid_distance"] = round(
                ai_weight * ai_metric + (1 - ai_weight) * det["st_distance"],
                2,
            )
            det["distance_method"] = "AI-Enhanced"
        else:
            det["hybrid_distance"] = det["st_distance"]

        # Clean up internal key
        del det["_ai_raw_depth"]

    return detections

