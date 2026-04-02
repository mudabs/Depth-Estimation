"""Entry point for stereo feature detection and matching."""

import cv2
import numpy as np

from src.config import CAMERA_MATRIX, OUTPUT_PATH, STEREO_PATH
from src.depth import compute_disparity
from src.feature_matching import (
    detect_and_compute,
    draw_matches,
    extract_matched_points,
    match_features,
)
from src.geometry import compute_fundamental_matrix, filter_inliers
from src.geometry import compute_essential_matrix, recover_pose_from_essential
from src.geometry import compute_rectification, rectify_images
from src.geometry import triangulate_points
from src.io_utils import load_images_from_folder
from src.pointcloud import export_colored_pointcloud_from_depth
from src.visualization import plot_3d_points, save_side_by_side, show_image


def main() -> None:
    """Load a stereo pair, match features, and visualize correspondences."""
    stereo_folder = STEREO_PATH

    print(f"Loading stereo images from: {stereo_folder}", flush=True)
    try:
        images, filenames = load_images_from_folder(stereo_folder)
    except Exception as exc:
        print(f"Failed to load stereo images: {exc}")
        return

    if len(images) < 2:
        print(
            f"Need at least 2 images in '{stereo_folder}', "
            f"but found {len(images)} image(s)."
        )
        return

    left_image, right_image = images[0], images[1]
    left_name, right_name = filenames[0], filenames[1]

    print(f"Running ORB feature detection on: {left_name} and {right_name}", flush=True)
    kp1, desc1 = detect_and_compute(left_image)
    kp2, desc2 = detect_and_compute(right_image)

    print(f"Keypoints in {left_name}: {len(kp1)}")
    print(f"Keypoints in {right_name}: {len(kp2)}")

    matches = match_features(desc1, desc2)
    print(f"Number of matches: {len(matches)}")

    pts1, pts2 = extract_matched_points(kp1, kp2, matches)
    if len(matches) == 0 or len(pts1) == 0 or len(pts2) == 0:
        print("No valid feature matches were found between the stereo images.")
        return

    F, mask = compute_fundamental_matrix(pts1, pts2)
    if F is None or mask is None:
        print("Failed to estimate Fundamental Matrix. Not enough reliable correspondences.")
        return

    pts1_inliers, pts2_inliers = filter_inliers(pts1, pts2, mask)
    inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if int(keep) == 1]

    print(f"Total matches: {len(matches)}")
    print(f"Inliers after RANSAC: {len(pts1_inliers)}")
    print("Fundamental Matrix (F):")
    print(F)

    E = compute_essential_matrix(F, CAMERA_MATRIX)
    if E is None:
        print("Failed to compute Essential Matrix. Check camera intrinsics.")
        return

    R, t, _pose_mask = recover_pose_from_essential(
        E,
        pts1_inliers,
        pts2_inliers,
        CAMERA_MATRIX,
    )
    if R is None or t is None:
        print("Failed to recover relative pose (R, t) from Essential Matrix.")
        return

    print("Essential Matrix (E):")
    print(E)
    print("Rotation Matrix (R):")
    print(R)
    print("Translation Vector (t, unit-norm direction):")
    print(t)

    K = CAMERA_MATRIX
    image_size = (left_image.shape[1], left_image.shape[0])
    try:
        R1, R2, P1, P2, Q = compute_rectification(K, R, t, image_size)
        rect1, rect2 = rectify_images(left_image, right_image, K, R1, R2, P1, P2)
    except cv2.error as exc:
        print(f"Failed to rectify stereo images: {exc}")
        return

    rectified_left_output = OUTPUT_PATH / "disparity" / "rectified_left.png"
    rectified_right_output = OUTPUT_PATH / "disparity" / "rectified_right.png"
    rectified_left_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(rectified_left_output), rect1):
        print(f"Failed to save rectified left image to: {rectified_left_output}")
        return
    if not cv2.imwrite(str(rectified_right_output), rect2):
        print(f"Failed to save rectified right image to: {rectified_right_output}")
        return

    print(f"Saved rectified left image to: {rectified_left_output}")
    print(f"Saved rectified right image to: {rectified_right_output}")

    try:
        _raw_unrectified, disparity_unrectified = compute_disparity(left_image, right_image)
        raw_disparity, disparity = compute_disparity(rect1, rect2)
    except Exception as exc:
        print(f"Failed to compute disparity map: {exc}")
        return

    disparity_unrect_output_path = OUTPUT_PATH / "disparity" / "disparity_map_unrectified.png"
    if not cv2.imwrite(str(disparity_unrect_output_path), disparity_unrectified):
        print(f"Failed to save unrectified disparity map to: {disparity_unrect_output_path}")
        return

    failure_dir = OUTPUT_PATH / "failure"
    failure_dir.mkdir(parents=True, exist_ok=True)
    noisy_disparity_path = failure_dir / "noisy_unrectified_disparity.png"
    if not cv2.imwrite(str(noisy_disparity_path), disparity_unrectified):
        print(f"Failed to save failure disparity example to: {noisy_disparity_path}")
        return
    print(f"Saved failure example (noisy disparity) to: {noisy_disparity_path}")

    disparity_output_path = OUTPUT_PATH / "disparity" / "disparity_map_rectified.png"
    if not cv2.imwrite(str(disparity_output_path), disparity):
        print(f"Failed to save disparity map to: {disparity_output_path}")
        return

    print(f"Saved rectified disparity map to: {disparity_output_path}")

    comparisons_dir = OUTPUT_PATH / "comparisons"
    disparity_comparison_path = comparisons_dir / "disparity_comparison.png"
    save_side_by_side(
        disparity_unrectified,
        disparity,
        title1="Unrectified Disparity",
        title2="Rectified Disparity",
        output_path=disparity_comparison_path,
        figure_title="Disparity Comparison",
    )
    print(f"Saved disparity comparison to: {disparity_comparison_path}")

    # ---- Depth reconstruction from disparity + Q matrix ----
    points_3D_dense = cv2.reprojectImageTo3D(raw_disparity, Q)
    depth_map = points_3D_dense[:, :, 2]

    # Save a raw depth visualization before any filtering to document failure modes.
    depth_raw_for_vis = depth_map.copy()
    depth_raw_for_vis[~np.isfinite(depth_raw_for_vis)] = 0
    depth_raw_vis = cv2.normalize(depth_raw_for_vis, None, 0, 255, cv2.NORM_MINMAX)
    depth_raw_vis = depth_raw_vis.astype(np.uint8)
    raw_depth_failure_path = failure_dir / "raw_depth_before_filtering.png"
    if not cv2.imwrite(str(raw_depth_failure_path), depth_raw_vis):
        print(f"Failed to save raw depth failure example to: {raw_depth_failure_path}")
        return
    print(f"Saved failure example (raw depth) to: {raw_depth_failure_path}")

    # Optional helper image: highlights where depth is invalid or extreme.
    invalid_regions = (depth_map <= 0) | (depth_map > 100) | (~np.isfinite(depth_map))
    invalid_vis = np.zeros(depth_map.shape, dtype=np.uint8)
    invalid_vis[invalid_regions] = 255
    invalid_depth_regions_path = failure_dir / "depth_failure_regions.png"
    if cv2.imwrite(str(invalid_depth_regions_path), invalid_vis):
        print(f"Saved failure regions mask to: {invalid_depth_regions_path}")

    pointcloud_output_path = OUTPUT_PATH / "pointcloud" / "scene.ply"
    try:
        n_vertices = export_colored_pointcloud_from_depth(
            points_3D_dense,
            rect1,
            pointcloud_output_path,
            z_min=0.0,
            z_max=100.0,
        )
    except Exception as exc:
        print(f"Failed to export point cloud: {exc}")
        return

    print(f"Saved colored point cloud ({n_vertices} points) to: {pointcloud_output_path}")

    # Mask out invalid depth (where disparity <= 0 → depth is meaningless).
    valid_mask = raw_disparity > 0
    depth_map_masked = np.where(valid_mask, depth_map, np.nan)

    depth_output_dir = OUTPUT_PATH / "depth"
    depth_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(depth_output_dir / "depth_map.npy"), depth_map_masked)
    print(f"Saved raw depth map → {depth_output_dir / 'depth_map.npy'}")

    # ---- Depth inspection ----
    h, w = depth_map.shape
    center_y, center_x = h // 2, w // 2
    print("\n--- Depth at Center Pixel ---")
    print(f"Pixel: ({center_x}, {center_y})")
    print(f"Depth (Z): {depth_map[center_y, center_x]:.4f}")

    valid_ys, valid_xs = np.where(depth_map > 0)
    print("\n--- Sample Depth Values ---")
    if len(valid_ys) >= 5:
        rng = np.random.default_rng(seed=0)
        idxs = rng.integers(0, len(valid_ys), size=5)
        for idx in idxs:
            y, x = int(valid_ys[idx]), int(valid_xs[idx])
            print(f"Pixel ({x}, {y}) -> Depth: {depth_map[y, x]:.4f}")
    else:
        print(f"Only {len(valid_ys)} valid depth pixel(s) found.")

    print("\n--- Depth Statistics ---")
    print(f"Min depth:  {float(np.nanmin(depth_map_masked)):.4f}")
    print(f"Max depth:  {float(np.nanmax(depth_map_masked)):.4f}")
    print(f"Mean depth: {float(np.nanmean(depth_map_masked)):.4f}")

    valid_depth = depth_map[(depth_map > 0) & (depth_map < 100)]
    print("\n--- Filtered Depth Statistics (0 < depth < 100) ---")
    if valid_depth.size > 0:
        print(f"Min depth:  {float(np.min(valid_depth)):.4f}")
        print(f"Max depth:  {float(np.max(valid_depth)):.4f}")
        print(f"Mean depth: {float(np.mean(valid_depth)):.4f}")
    else:
        print("No valid depth values found in range (0, 100).")

    # Normalize valid pixels for visualization.
    depth_vis = np.zeros(depth_map.shape, dtype=np.uint8)
    valid_depth = depth_map[valid_mask]
    if valid_depth.size > 0:
        d_min, d_max = float(valid_depth.min()), float(valid_depth.max())
        depth_vis_float = np.zeros(depth_map.shape, dtype=np.float32)
        if d_max > d_min:
            depth_vis_float[valid_mask] = (
                (depth_map[valid_mask] - d_min) / (d_max - d_min) * 255
            )
        depth_vis = depth_vis_float.astype(np.uint8)
        depth_vis_path = depth_output_dir / "depth_visualization.png"
        if cv2.imwrite(str(depth_vis_path), depth_vis):
            print(f"Saved depth visualization → {depth_vis_path}")
        else:
            print(f"Failed to save depth visualization to: {depth_vis_path}")
    else:
        print("Warning: no valid depth pixels found for visualization.")

    # ---- Improved depth visualization (depth_clean.png) ----
    valid_mask_clean = raw_disparity > 1
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask_clean] = 0
    depth_clean_vis = np.zeros(depth_map.shape, dtype=np.uint8)

    # Clip extreme values using the 95th percentile of valid pixels.
    valid_pixels = depth_clean[valid_mask_clean]
    if valid_pixels.size > 0:
        p95 = float(np.percentile(valid_pixels, 95))
        depth_clean = np.clip(depth_clean, 0, p95)
        depth_clean_vis = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_clean_vis = depth_clean_vis.astype(np.uint8)
        depth_clean_path = depth_output_dir / "depth_clean.png"
        if cv2.imwrite(str(depth_clean_path), depth_clean_vis):
            print(f"Saved clean depth visualization → {depth_clean_path}")
        else:
            print(f"Failed to save clean depth visualization to: {depth_clean_path}")
    else:
        print("Warning: no valid pixels for clean depth visualization.")

    depth_comparison_path = comparisons_dir / "depth_comparison.png"
    save_side_by_side(
        depth_vis,
        depth_clean_vis,
        title1="Raw Depth (Normalized)",
        title2="Filtered Depth (Clean)",
        output_path=depth_comparison_path,
        figure_title="Depth Comparison",
    )
    print(f"Saved depth comparison to: {depth_comparison_path}")

    points_3d = triangulate_points(pts1_inliers, pts2_inliers, K, R, t)
    if len(points_3d) == 0:
        print("Triangulation failed: no 3D points were reconstructed.")
        return

    # Keep points in front of camera 1.
    points_3d = points_3d[points_3d[:, 2] > 0]

    if len(points_3d) == 0:
        print("Triangulation produced only negative-depth points.")
        return

    print(f"Number of 3D points: {len(points_3d)}")
    print("First 5 3D points (X, Y, Z):")
    print(points_3d[:5])

    plot_3d_points(points_3d)

    if len(inlier_matches) == 0:
        print("No inlier matches available for visualization.")
        return

    match_vis = draw_matches(left_image, right_image, kp1, kp2, inlier_matches)
    output_path = OUTPUT_PATH / "feature_matches" / "feature_matches_inliers.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), match_vis):
        print(f"Failed to save match visualization to: {output_path}")
        return

    print(f"Saved inlier match visualization to: {output_path}")
    show_image(disparity, title="Dense Disparity Map")
    show_image(match_vis, title="Stereo Inlier Matches")


if __name__ == "__main__":
    main()
