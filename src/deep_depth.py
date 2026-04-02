"""Monocular deep depth estimation utilities (Depth Anything)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "LiheYoung/depth-anything-small-hf"


def load_depth_model(model_id: str = MODEL_ID):
    """Load and return image processor and depth model."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    return processor, model


def predict_relative_depth(
    bgr_image: np.ndarray,
    processor,
    model,
) -> np.ndarray:
    """Infer a relative depth map and return a uint8 visualization."""
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input image must be valid and non-empty.")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = processor(images=rgb_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    pred = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=rgb_image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = pred.detach().cpu().numpy()
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return depth_vis.astype(np.uint8)
