"""
test_pipeline.py — End-to-end test of detection + distance estimation.

Downloads two sample images from the internet, runs YOLOv8 detection,
annotates them with bounding boxes + distance labels, and saves the
annotated output to verify the pipeline works.
"""

import sys
import os
import urllib.request
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from detector import detect_objects
from distance import estimate_distance, DEFAULT_FOCAL_LENGTH_PX

# Sample images (public domain / Ultralytics demo images)
SAMPLE_URLS = {
    "street_scene": "https://ultralytics.com/images/bus.jpg",
    "mixed_objects": "https://ultralytics.com/images/zidane.jpg",
}

BOX_COLORS = [
    (167, 139, 250), (109, 213, 237), (252, 211, 77), (248, 113, 113),
    (52, 211, 153), (251, 146, 60), (129, 140, 248), (244, 114, 182),
]

def download_image(url: str, save_path: str) -> np.ndarray:
    """Download image from URL and return as BGR numpy array."""
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, save_path)
    img = cv2.imread(save_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {save_path}")
    return img


def test_image(name: str, url: str, output_dir: str):
    """Run full detection + distance pipeline on one image."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    input_path = os.path.join(output_dir, f"{name}_input.jpg")
    output_path = os.path.join(output_dir, f"{name}_output.jpg")

    # Download
    img = download_image(url, input_path)
    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h}")

    # Detect
    detections = detect_objects(img, confidence_threshold=0.5)
    print(f"  Detections: {len(detections)}")

    if not detections:
        print("  ⚠ No objects detected!")
        return

    # Annotate
    annotated = img.copy()
    scale = max(h, w) / 1000
    thickness = max(int(2 * scale), 1)
    font_scale = max(0.5 * scale, 0.4)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls = det["class_name"]
        conf = det["confidence"]
        color = BOX_COLORS[det["class_id"] % len(BOX_COLORS)]

        dist = estimate_distance(cls, det["bbox_height"], DEFAULT_FOCAL_LENGTH_PX)
        label = f"{cls} {conf:.0%} | {dist:.1f}m"

        print(f"  [{i+1}] {cls:15s}  conf={conf:.2f}  "
              f"bbox_h={det['bbox_height']:.0f}px  dist={dist:.2f}m")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th_text - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    print(f"  ✓ Saved annotated output: {output_path}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(output_dir, exist_ok=True)

    print("DistVision — Pipeline Test")
    print("=" * 60)

    for name, url in SAMPLE_URLS.items():
        test_image(name, url, output_dir)

    print(f"\n{'='*60}")
    print("✓ All tests completed! Check test_output/ for results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
