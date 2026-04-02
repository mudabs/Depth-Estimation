"""Geometric estimation utilities for stereo vision."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def compute_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate the Fundamental Matrix using RANSAC.

    Args:
        pts1: Matched points from the first image, shape (N, 2).
        pts2: Matched points from the second image, shape (N, 2).

    Returns:
        A tuple (F, mask) where:
            - F is the 3x3 Fundamental Matrix, or None if estimation fails.
            - mask is an inlier mask of shape (N, 1), or None if estimation fails.
    """
    if pts1 is None or pts2 is None or len(pts1) < 8 or len(pts2) < 8:
        return None, None

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99,
    )

    if F is None or mask is None:
        return None, None

    return F, mask


def filter_inliers(
    pts1: np.ndarray,
    pts2: np.ndarray,
    mask: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter point correspondences using a binary inlier mask.

    Args:
        pts1: Matched points from the first image, shape (N, 2).
        pts2: Matched points from the second image, shape (N, 2).
        mask: Inlier mask from robust estimation, shape (N, 1) or (N,).

    Returns:
        Filtered (pts1_inliers, pts2_inliers) as float32 arrays.
    """
    if mask is None or pts1 is None or pts2 is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()

    inlier_mask = mask.ravel().astype(bool)
    return pts1[inlier_mask].astype(np.float32), pts2[inlier_mask].astype(np.float32)


def compute_essential_matrix(F: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Compute Essential Matrix from Fundamental Matrix and camera intrinsics."""
    if F is None or K is None or F.shape != (3, 3) or K.shape != (3, 3):
        return None
    return K.T @ F @ K


def recover_pose_from_essential(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Recover relative pose (R, t) from the Essential Matrix.

    Returns:
        A tuple (R, t, pose_mask). If recovery fails, all values are None.
    """
    if E is None or K is None or len(pts1) < 5 or len(pts2) < 5:
        return None, None, None

    try:
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    except cv2.error:
        return None, None, None

    return R, t, pose_mask


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from two-view correspondences.

    Args:
        pts1: Inlier points from the first image, shape (N, 2).
        pts2: Inlier points from the second image, shape (N, 2).
        K: Camera intrinsic matrix, shape (3, 3).
        R: Relative rotation from camera 1 to camera 2, shape (3, 3).
        t: Relative translation from camera 1 to camera 2, shape (3, 1) or (3,).

    Returns:
        Triangulated 3D points in camera-1 coordinates, shape (N, 3).
        Returns an empty array if inputs are invalid.
    """
    if (
        pts1 is None
        or pts2 is None
        or K is None
        or R is None
        or t is None
        or len(pts1) == 0
        or len(pts2) == 0
    ):
        return np.empty((0, 3), dtype=np.float64)

    I = np.eye(3, dtype=np.float64)
    zero = np.zeros((3, 1), dtype=np.float64)
    P1 = K @ np.hstack((I, zero))
    t_col = t.reshape(3, 1).astype(np.float64)
    P2 = K @ np.hstack((R.astype(np.float64), t_col))

    pts1_2xN = pts1.astype(np.float64).T
    pts2_2xN = pts2.astype(np.float64).T

    points_4d = cv2.triangulatePoints(P1, P2, pts1_2xN, pts2_2xN)
    points_4d /= points_4d[3]
    points_3d = points_4d[:3].T
    return points_3d


def compute_rectification(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification transforms and projection matrices."""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=K,
        distCoeffs1=None,
        cameraMatrix2=K,
        distCoeffs2=None,
        imageSize=image_size,
        R=R,
        T=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    return R1, R2, P1, P2, Q


def rectify_images(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify a stereo pair with precomputed rectification transforms."""
    map1x, map1y = cv2.initUndistortRectifyMap(
        K,
        None,
        R1,
        P1,
        img1.shape[:2][::-1],
        cv2.CV_32FC1,
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K,
        None,
        R2,
        P2,
        img2.shape[:2][::-1],
        cv2.CV_32FC1,
    )

    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    return rect1, rect2
