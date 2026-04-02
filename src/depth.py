"""Dense stereo disparity computation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def compute_disparity(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Compute a dense disparity map from a stereo image pair.

    Args:
        img1: Left stereo image in BGR or grayscale format.
        img2: Right stereo image in BGR or grayscale format.

    Returns:
        A tuple of (raw_disparity, vis_disparity) where raw_disparity is a
        float32 array in pixel units and vis_disparity is a normalized uint8
        map suitable for saving/display.

    Raises:
        ValueError: If either image is invalid or if image sizes do not match.
    """
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        raise ValueError("Both stereo images must be valid and non-empty.")

    if img1.shape[:2] != img2.shape[:2]:
        raise ValueError("Stereo images must have the same height and width.")

    gray1 = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = img2 if img2.ndim == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 10,
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=150,
        speckleRange=2,
        preFilterCap=63,
    )

    # StereoSGBM returns disparity in 16x fixed-point; divide by 16 for pixel units.
    raw = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    raw = cv2.medianBlur(raw, 5)
    disp_norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX)
    return raw, disp_norm.astype(np.uint8)
