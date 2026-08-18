"""
depth_estimator.py — AI-powered monocular depth estimation using Depth Anything V2.

Provides a wrapper around the Depth Anything V2 model (Small variant) from
Hugging Face Transformers.  The model produces a dense, per-pixel relative
depth map which is used by the hybrid distance pipeline in distance.py.

Key features:
  • Lazy model loading with caching (loaded once, reused across frames).
  • Automatic model download from Hugging Face on first run (~100 MB).
  • Works on CPU (≈ 1 s per image) and GPU (≈ 50 ms per image).
  • Depth map is resized to match the input image dimensions.
"""

import numpy as np
from PIL import Image
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# The Small variant balances accuracy and speed.  Alternatives:
#   "depth-anything/Depth-Anything-V2-Base-hf"   (~400 MB, more accurate)
#   "depth-anything/Depth-Anything-V2-Large-hf"  (~1.3 GB, most accurate)
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ---------------------------------------------------------------------------
# Singleton estimator — avoids reloading the model on every call
# ---------------------------------------------------------------------------
_estimator_instance: Optional["DepthEstimator"] = None


class DepthEstimator:
    """Wrapper around Depth Anything V2 for monocular depth estimation."""

    def __init__(self, model_id: str = MODEL_ID):
        """
        Load the Depth Anything V2 model and image processor.

        Parameters
        ----------
        model_id : str
            Hugging Face model identifier.
        """
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def get_depth_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Generate a per-pixel depth map from an RGB image.

        Parameters
        ----------
        image_rgb : np.ndarray
            Input image in RGB format, shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Depth map of shape (H, W) with float32 values.
            Higher values = farther from camera.
            Values are relative (not metric metres).
        """
        import torch

        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Squeeze batch dimension → (h, w)
        depth = predicted_depth.squeeze().cpu().numpy()

        # Resize depth map to match the original image dimensions
        h, w = image_rgb.shape[:2]
        if depth.shape != (h, w):
            depth_pil = Image.fromarray(depth)
            depth_pil = depth_pil.resize((w, h), Image.BILINEAR)
            depth = np.array(depth_pil, dtype=np.float32)

        return depth

    def get_object_depth(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[float, float, float, float],
        margin_frac: float = 0.1,
    ) -> float:
        """
        Extract the estimated depth for a single detected object.

        Uses the median depth value within the central region of the
        bounding box (excluding edges, which often contain background).

        Parameters
        ----------
        depth_map : np.ndarray
            Full-image depth map from ``get_depth_map()``.
        bbox : tuple
            (x1, y1, x2, y2) bounding box in pixels.
        margin_frac : float
            Fraction of bbox width/height to crop from each edge before
            computing the median.  Reduces background contamination.

        Returns
        -------
        float
            Median depth value within the ROI, or 0.0 if the ROI is empty.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape[:2]

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Shrink by margin to focus on the object center
        bw, bh = x2 - x1, y2 - y1
        mx = int(bw * margin_frac)
        my = int(bh * margin_frac)
        cx1, cy1 = x1 + mx, y1 + my
        cx2, cy2 = x2 - mx, y2 - my

        if cx2 <= cx1 or cy2 <= cy1:
            # Margin too large — fall back to full bbox
            roi = depth_map[y1:y2, x1:x2]
        else:
            roi = depth_map[cy1:cy2, cx1:cx2]

        if roi.size == 0:
            return 0.0

        return float(np.median(roi))


def get_depth_estimator() -> DepthEstimator:
    """
    Return the singleton DepthEstimator instance.

    The model is loaded lazily on the first call and cached for
    subsequent calls within the same process.
    """
    global _estimator_instance
    if _estimator_instance is None:
        _estimator_instance = DepthEstimator()
    return _estimator_instance


def depth_map_to_colormap(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert a raw depth map to a coloured visualisation (RGB).

    Uses the Inferno colourmap: warm colours = close, cool colours = far.

    Parameters
    ----------
    depth_map : np.ndarray
        Raw depth map of shape (H, W), float32.

    Returns
    -------
    np.ndarray
        Colour-mapped image of shape (H, W, 3), uint8 RGB.
    """
    import cv2

    # Normalise to 0–255
    d = depth_map.copy()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 0:
        d = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d = np.zeros_like(d, dtype=np.uint8)

    # Apply INFERNO colourmap (warm = close, cool = far)
    # Invert so closer objects appear warmer
    d = 255 - d
    colored = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return colored_rgb
