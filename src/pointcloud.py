"""Point cloud export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_ascii_ply(points_rgb: np.ndarray, output_path: Path) -> None:
    """Write XYZRGB point cloud to an ASCII PLY file.

    Args:
        points_rgb: Array of shape (N, 6) with columns [x, y, z, r, g, b].
        output_path: Target .ply path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_points = points_rgb.shape[0]

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {n_points}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )

    with output_path.open("w", encoding="ascii", newline="\n") as f:
        f.write(header)
        f.write("\n")
        for x, y, z, r, g, b in points_rgb:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def export_colored_pointcloud_from_depth(
    points_3d_dense: np.ndarray,
    rectified_left_bgr: np.ndarray,
    output_path: Path,
    z_min: float = 0.0,
    z_max: float = 100.0,
) -> int:
    """Export a filtered colored point cloud from dense reprojected points.

    Args:
        points_3d_dense: Reprojected 3D array of shape (H, W, 3).
        rectified_left_bgr: Rectified left image of shape (H, W, 3) in BGR.
        output_path: Output path for .ply file.
        z_min: Minimum valid depth (exclusive).
        z_max: Maximum valid depth (inclusive).

    Returns:
        Number of vertices written to the PLY file.

    Raises:
        ValueError: If input shapes are invalid or incompatible.
    """
    if points_3d_dense is None or points_3d_dense.ndim != 3 or points_3d_dense.shape[2] != 3:
        raise ValueError("points_3d_dense must have shape (H, W, 3).")
    if rectified_left_bgr is None or rectified_left_bgr.ndim != 3 or rectified_left_bgr.shape[2] != 3:
        raise ValueError("rectified_left_bgr must have shape (H, W, 3).")
    if points_3d_dense.shape[:2] != rectified_left_bgr.shape[:2]:
        raise ValueError("3D points and color image must have the same height and width.")

    z = points_3d_dense[:, :, 2]
    valid = np.isfinite(z) & (z > z_min) & (z <= z_max)

    points = points_3d_dense[valid]
    colors_bgr = rectified_left_bgr[valid]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)

    if points.size == 0:
        _write_ascii_ply(np.empty((0, 6), dtype=np.float64), output_path)
        return 0

    points_rgb = np.hstack([points.astype(np.float64), colors_rgb.astype(np.float64)])
    _write_ascii_ply(points_rgb, output_path)
    return points_rgb.shape[0]
