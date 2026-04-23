"""Dense stereo disparity computation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def compute_sgbm(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a dense disparity map with OpenCV SGBM.

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

    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=256,  # 16 * 16, divisible by 16; catches close objects
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # StereoSGBM returns disparity in 16x fixed-point; divide by 16 for pixel units.
    raw = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    # Median removes salt-and-pepper noise; bilateral smooths while preserving depth edges.
    raw = cv2.medianBlur(raw, 5)
    raw = cv2.bilateralFilter(raw, 9, 75, 75)
    disp_norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX)
    return raw, disp_norm.astype(np.uint8)


def compute_unimatch(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Placeholder for a future UniMatch stereo backend."""
    raise NotImplementedError("UniMatch disparity is not implemented yet.")


def compute_disparity(
    method: str | np.ndarray = "sgbm",
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the requested disparity backend.

    The function keeps backward compatibility with the historical
    compute_disparity(left, right) call shape while also supporting the
    explicit compute_disparity("sgbm", left, right) form.
    """
    if isinstance(method, np.ndarray):
        left_image = method
        right_image = left
        method_name = "sgbm"
    else:
        left_image = left
        right_image = right
        method_name = str(method).lower()

    if left_image is None or right_image is None:
        raise ValueError("Both stereo images must be provided.")

    if method_name == "sgbm":
        return compute_sgbm(left_image, right_image)
    if method_name == "unimatch":
        return compute_unimatch(left_image, right_image)
    raise ValueError(f"Unsupported disparity method: {method_name}")
