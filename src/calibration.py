"""Single-camera calibration utilities using checkerboard images."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _detect_checkerboard(
    gray: np.ndarray,
    pattern_size: Tuple[int, int],
    use_slow_fallback: bool = False,
) -> tuple[bool, np.ndarray | None]:
    """Try multiple checkerboard detection strategies for robustness."""
    base_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    # First pass: vanilla grayscale.
    found, corners = cv2.findChessboardCorners(gray, pattern_size, base_flags)
    if found:
        return True, corners

    # Second pass: contrast-limited adaptive histogram equalization.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    found, corners = cv2.findChessboardCorners(enhanced, pattern_size, base_flags)
    if found:
        return True, corners

    # Third pass: newer OpenCV detector that often recovers difficult boards.
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
        if found:
            return True, corners
        found, corners = cv2.findChessboardCornersSB(enhanced, pattern_size, sb_flags)
        if found:
            return True, corners

        # Optional slower fallback for hard datasets.
        if use_slow_fallback:
            exhaustive_flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, exhaustive_flags)
            if found:
                return True, corners
            found, corners = cv2.findChessboardCornersSB(
                enhanced,
                pattern_size,
                exhaustive_flags,
            )
            if found:
                return True, corners

    return False, None


def find_checkerboard_corners(
    images: List[np.ndarray],
    pattern_size: Tuple[int, int],
    verbose: bool = False,
    use_slow_fallback: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Detect and refine checkerboard corners across a list of images.

    Args:
        images: List of BGR images.
        pattern_size: Number of inner corners as (cols, rows).
        verbose: If True, print progress for each processed image.
        use_slow_fallback: If True, enable slower exhaustive detection fallback.

    Returns:
        A tuple containing:
            - objpoints: 3D points in checkerboard coordinate space.
            - imgpoints: 2D corner points detected in image coordinates.
            - valid_images: Subset of images where corners were successfully found.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    valid_images: List[np.ndarray] = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for idx, image in enumerate(images, start=1):
        if image is None or image.size == 0:
            if verbose:
                print(f"[Corners] Skipping invalid image {idx}/{len(images)}", flush=True)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = _detect_checkerboard(
            gray,
            pattern_size,
            use_slow_fallback=use_slow_fallback,
        )

        if not found or corners is None:
            if verbose:
                print(f"[Corners] Not found in image {idx}/{len(images)}", flush=True)
            continue

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

        objpoints.append(objp.copy())
        imgpoints.append(refined_corners)
        valid_images.append(image)
        if verbose:
            print(f"[Corners] Found in image {idx}/{len(images)}", flush=True)

    return objpoints, imgpoints, valid_images


def compute_reprojection_error(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """Compute mean reprojection error across all valid calibration images."""
    if not objpoints or not imgpoints:
        return float("inf")

    total_error = 0.0
    total_points = 0

    for objp, observed_imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected_imgp, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        error = cv2.norm(observed_imgp, projected_imgp, cv2.NORM_L2)
        total_error += error
        total_points += len(projected_imgp)

    if total_points == 0:
        return float("inf")

    return total_error / total_points


def calibrate_camera(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]:
    """Calibrate camera intrinsics from checkerboard correspondences.

    Args:
        objpoints: 3D checkerboard points per valid image.
        imgpoints: 2D detected corners per valid image.
        image_shape: Image shape as (height, width).

    Returns:
        Tuple containing camera matrix, distortion coefficients, rotation vectors,
        translation vectors, and mean reprojection error.

    Raises:
        ValueError: If no valid points are provided.
        RuntimeError: If OpenCV calibration fails.
    """
    if not objpoints or not imgpoints:
        raise ValueError("Calibration requires at least one valid checkerboard detection.")

    height, width = image_shape
    success, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (width, height),
        None,
        None,
    )

    if not success:
        raise RuntimeError("Camera calibration failed.")

    reprojection_error = compute_reprojection_error(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        K,
        dist,
    )

    return K, dist, rvecs, tvecs, reprojection_error


def undistort_image(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort a BGR image using camera intrinsics and distortion coefficients."""
    return cv2.undistort(image, K, dist)
