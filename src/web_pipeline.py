"""Backend helpers for Streamlit stereo vision UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import OUTPUT_PATH
from src.depth import compute_disparity
from src.feature_matching import (
    detect_and_compute,
    draw_matches,
    extract_matched_points,
    match_features,
)
from src.geometry import (
    compute_essential_matrix,
    compute_fundamental_matrix,
    compute_rectification,
    filter_inliers,
    recover_pose_from_essential,
    triangulate_points,
    rectify_images,
)
from src.pointcloud import export_colored_pointcloud_from_depth


@dataclass
class CalibrationResult:
    """Container for camera calibration output."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    reprojection_error: float
    valid_images: int


@dataclass
class PipelineResult:
    """Container for classical stereo pipeline outputs."""

    keypoints_left: int
    keypoints_right: int
    num_matches: int
    num_inliers: int
    fundamental_matrix: np.ndarray
    essential_matrix: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    match_visualization: np.ndarray
    rectified_left: np.ndarray
    rectified_right: np.ndarray
    disparity_unrectified: np.ndarray
    disparity_rectified: np.ndarray
    depth_raw_vis: np.ndarray
    depth_clean_vis: np.ndarray
    depth_failure_mask: np.ndarray
    depth_min: float
    depth_max: float
    depth_mean: float
    depth_clean_min: float
    depth_clean_max: float
    depth_clean_mean: float
    depth_raw_valid_count: int
    depth_clean_valid_count: int
    depth_total_count: int
    depth_p5: float
    depth_p50: float
    depth_p95: float
    depth_hist_counts: np.ndarray
    depth_hist_bin_centers: np.ndarray
    preview_points_xyz: np.ndarray
    preview_points_rgb: np.ndarray
    pointcloud_path: Path
    sparse_points: np.ndarray


def _label_image(image: np.ndarray, label: str) -> np.ndarray:
    """Draw a title bar over an image for side-by-side comparisons."""
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (0, 0, 0), thickness=-1)
    cv2.putText(
        vis,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def create_side_by_side(left: np.ndarray, right: np.ndarray, left_label: str, right_label: str) -> np.ndarray:
    """Create a labeled side-by-side image panel."""
    left_labeled = _label_image(left, left_label)
    right_labeled = _label_image(right, right_label)

    if left_labeled.shape[:2] != right_labeled.shape[:2]:
        right_labeled = cv2.resize(
            right_labeled,
            (left_labeled.shape[1], left_labeled.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.hstack([left_labeled, right_labeled])


def run_classical_pipeline(left_image: np.ndarray, right_image: np.ndarray, camera_matrix: np.ndarray) -> PipelineResult:
    """Run the existing classical stereo pipeline and return display-ready outputs."""
    kp1, desc1 = detect_and_compute(left_image, method="ORB")
    kp2, desc2 = detect_and_compute(right_image, method="ORB")

    matches = match_features(desc1, desc2, method="ORB")
    pts1, pts2 = extract_matched_points(kp1, kp2, matches)
    if len(pts1) < 8:
        raise RuntimeError("Not enough matches to estimate geometry. Try better stereo images.")

    F, mask = compute_fundamental_matrix(pts1, pts2)
    if F is None or mask is None:
        raise RuntimeError("Failed to compute the fundamental matrix.")

    pts1_inliers, pts2_inliers = filter_inliers(pts1, pts2, mask)
    inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if int(keep) == 1]
    if len(pts1_inliers) < 5:
        raise RuntimeError("Not enough inlier matches after RANSAC.")

    E = compute_essential_matrix(F, camera_matrix)
    if E is None:
        raise RuntimeError("Failed to compute the essential matrix.")

    R, t, _pose_mask = recover_pose_from_essential(E, pts1_inliers, pts2_inliers, camera_matrix)
    if R is None or t is None:
        raise RuntimeError("Failed to recover camera pose from the essential matrix.")

    image_size = (left_image.shape[1], left_image.shape[0])
    R1, R2, P1, P2, Q = compute_rectification(camera_matrix, R, t, image_size)
    rect_left, rect_right = rectify_images(left_image, right_image, camera_matrix, R1, R2, P1, P2)

    _raw_unrect, disparity_unrect = compute_disparity(left_image, right_image)
    raw_disparity, disparity_rect = compute_disparity(rect_left, rect_right)

    points_3d_dense = cv2.reprojectImageTo3D(raw_disparity, Q)
    depth_map = points_3d_dense[:, :, 2]

    depth_raw_for_vis = depth_map.copy()
    depth_raw_for_vis[~np.isfinite(depth_raw_for_vis)] = 0
    depth_raw_vis = cv2.normalize(depth_raw_for_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    valid_depth_mask = np.isfinite(depth_map) & (depth_map > 0)
    valid_depth_values = depth_map[valid_depth_mask]

    # Robust cleaned depth: keep only central percentile band of positive finite depths.
    depth_clean = np.zeros_like(depth_map, dtype=np.float32)
    clean_mask = np.zeros_like(valid_depth_mask, dtype=bool)
    if valid_depth_values.size >= 20:
        lo = float(np.percentile(valid_depth_values, 1))
        hi = float(np.percentile(valid_depth_values, 99))
        if hi > lo:
            clean_mask = valid_depth_mask & (depth_map >= lo) & (depth_map <= hi)
    elif valid_depth_values.size > 0:
        # For tiny valid sets, keep positives as-is.
        clean_mask = valid_depth_mask

    depth_clean[clean_mask] = depth_map[clean_mask]
    depth_clean_vis = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    invalid_mask = ~valid_depth_mask
    depth_failure_mask = np.zeros(depth_map.shape, dtype=np.uint8)
    depth_failure_mask[invalid_mask] = 255

    # Use all positive finite depths for stats to avoid frequent all-zero outputs.
    valid_depth = valid_depth_values
    if valid_depth.size == 0:
        depth_min = 0.0
        depth_max = 0.0
        depth_mean = 0.0
        depth_p5 = 0.0
        depth_p50 = 0.0
        depth_p95 = 0.0
        depth_hist_counts = np.zeros(20, dtype=np.int64)
        depth_hist_bin_centers = np.linspace(0, 1, 20, dtype=np.float32)
    else:
        depth_min = float(np.min(valid_depth))
        depth_max = float(np.max(valid_depth))
        depth_mean = float(np.mean(valid_depth))
        depth_p5 = float(np.percentile(valid_depth, 5))
        depth_p50 = float(np.percentile(valid_depth, 50))
        depth_p95 = float(np.percentile(valid_depth, 95))
        depth_hist_counts, bin_edges = np.histogram(valid_depth, bins=20)
        depth_hist_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    clean_values = depth_clean[clean_mask]
    if clean_values.size == 0:
        depth_clean_min = 0.0
        depth_clean_max = 0.0
        depth_clean_mean = 0.0
    else:
        depth_clean_min = float(np.min(clean_values))
        depth_clean_max = float(np.max(clean_values))
        depth_clean_mean = float(np.mean(clean_values))

    depth_raw_valid_count = int(valid_depth.size)
    depth_clean_valid_count = int(clean_values.size)
    depth_total_count = int(depth_map.size)

    pointcloud_path = OUTPUT_PATH / "pointcloud" / "scene.ply"
    export_colored_pointcloud_from_depth(
        points_3d_dense,
        rect_left,
        pointcloud_path,
        z_min=0.0,
        z_max=100.0,
    )

    # Sample a subset for responsive in-app 3D preview.
    z = points_3d_dense[:, :, 2]
    preview_mask = np.isfinite(z) & (z > 0.0) & (z <= 100.0)
    preview_points = points_3d_dense[preview_mask]
    preview_colors_bgr = rect_left[preview_mask]
    preview_colors_rgb = preview_colors_bgr[:, ::-1].astype(np.uint8)

    max_preview_points = 12000
    if len(preview_points) > max_preview_points:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(preview_points), size=max_preview_points, replace=False)
        preview_points = preview_points[idx]
        preview_colors_rgb = preview_colors_rgb[idx]

    sparse_points = triangulate_points(pts1_inliers, pts2_inliers, camera_matrix, R, t)
    if len(sparse_points) > 0:
        sparse_points = sparse_points[sparse_points[:, 2] > 0]

    match_vis = draw_matches(left_image, right_image, kp1, kp2, inlier_matches)

    return PipelineResult(
        keypoints_left=len(kp1),
        keypoints_right=len(kp2),
        num_matches=len(matches),
        num_inliers=len(pts1_inliers),
        fundamental_matrix=F,
        essential_matrix=E,
        rotation_matrix=R,
        translation_vector=t,
        match_visualization=match_vis,
        rectified_left=rect_left,
        rectified_right=rect_right,
        disparity_unrectified=disparity_unrect,
        disparity_rectified=disparity_rect,
        depth_raw_vis=depth_raw_vis,
        depth_clean_vis=depth_clean_vis,
        depth_failure_mask=depth_failure_mask,
        depth_min=depth_min,
        depth_max=depth_max,
        depth_mean=depth_mean,
        depth_clean_min=depth_clean_min,
        depth_clean_max=depth_clean_max,
        depth_clean_mean=depth_clean_mean,
        depth_raw_valid_count=depth_raw_valid_count,
        depth_clean_valid_count=depth_clean_valid_count,
        depth_total_count=depth_total_count,
        depth_p5=depth_p5,
        depth_p50=depth_p50,
        depth_p95=depth_p95,
        depth_hist_counts=depth_hist_counts,
        depth_hist_bin_centers=depth_hist_bin_centers,
        preview_points_xyz=preview_points.astype(np.float32),
        preview_points_rgb=preview_colors_rgb,
        pointcloud_path=pointcloud_path,
        sparse_points=sparse_points,
    )
