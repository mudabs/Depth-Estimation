"""Visualization helpers using matplotlib for OpenCV-style images."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _show_or_close() -> None:
    """Show figures only on interactive backends, otherwise close them silently."""
    backend = plt.get_backend().lower()
    if "agg" in backend:
        plt.close()
        return
    plt.show()


def _to_display_image(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for matplotlib display."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image(image: np.ndarray, title: str = "Image") -> None:
    """Display a single image using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(_to_display_image(image), cmap="gray" if image.ndim == 2 else None)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    _show_or_close()


def show_side_by_side(img1: np.ndarray, img2: np.ndarray, title: str = "Comparison") -> None:
    """Display two images side by side using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    axes[0].imshow(_to_display_image(img1), cmap="gray" if img1.ndim == 2 else None)
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(_to_display_image(img2), cmap="gray" if img2.ndim == 2 else None)
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    plt.tight_layout()
    _show_or_close()


def save_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    title1: str,
    title2: str,
    output_path: Path,
    figure_title: str | None = None,
) -> None:
    """Save two images side-by-side with custom titles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if figure_title:
        fig.suptitle(figure_title)

    axes[0].imshow(_to_display_image(img1), cmap="gray" if img1.ndim == 2 else None)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(_to_display_image(img2), cmap="gray" if img2.ndim == 2 else None)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_3d_points(points_3d: np.ndarray) -> None:
    """Plot triangulated 3D points as a sparse scatter cloud."""
    if points_3d is None or len(points_3d) == 0:
        print("No 3D points to plot.")
        return

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstructed 3D Points")
    plt.tight_layout()
    _show_or_close()


# --- New: DL Dense Point Cloud Visualization ---
def plot_dense_pointcloud(points_3d: np.ndarray, colors: np.ndarray = None, sample: int = 1, z_min: float = 0.0, z_max: float = None, title: str = "Dense 3D Point Cloud") -> None:
    """
    Visualize a dense 3D point cloud with optional color and outlier filtering.

    Args:
        points_3d: (H, W, 3) or (N, 3) array of 3D points.
        colors: (H, W, 3) or (N, 3) array of uint8 RGB colors, or None.
        sample: Subsample factor for visualization (e.g., 10 = show every 10th point).
        z_min: Minimum Z value to keep (exclusive).
        z_max: Maximum Z value to keep (inclusive).
        title: Plot title.
    """
    pts = points_3d.reshape(-1, 3)
    if colors is not None:
        cols = colors.reshape(-1, 3)
    else:
        cols = None
    mask = np.isfinite(pts[:, 2]) & (pts[:, 2] > z_min)
    if z_max is not None:
        mask &= pts[:, 2] <= z_max
    pts = pts[mask]
    if cols is not None:
        cols = cols[mask]
    if sample > 1:
        idx = np.arange(0, len(pts), sample)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if cols is not None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols / 255.0, s=1)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    _show_or_close()


# --- New: Clean Disparity Visualization ---
def clean_disparity_vis(disp: np.ndarray, invalid_val: float = 0.0, cmap: str = "plasma") -> np.ndarray:
    """
    Create a clean visualization for a disparity map, masking invalid values and applying a colormap.

    Args:
        disp: Disparity map (float32 or uint8).
        invalid_val: Value to treat as invalid (e.g., 0 or <0).
        cmap: Matplotlib colormap name.

    Returns:
        RGB uint8 image for display.
    """
    disp_vis = disp.copy()
    mask = (disp_vis <= invalid_val) | ~np.isfinite(disp_vis)
    disp_vis[mask] = np.nan
    vmin = np.nanmin(disp_vis)
    vmax = np.nanmax(disp_vis)
    norm_disp = (disp_vis - vmin) / (vmax - vmin + 1e-6)
    norm_disp[mask] = np.nan
    cm = plt.get_cmap(cmap)
    colored = cm(norm_disp)
    colored[..., 3][mask] = 0  # Set alpha to 0 for invalid
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    return rgb


# --- New: Clean Point Cloud Utility ---
def filter_pointcloud(points_3d: np.ndarray, z_min: float = 0.0, z_max: float = None) -> np.ndarray:
    """
    Remove outliers from a 3D point cloud based on Z range and finite values.

    Args:
        points_3d: (N, 3) or (H, W, 3) array.
        z_min: Minimum Z value to keep (exclusive).
        z_max: Maximum Z value to keep (inclusive).

    Returns:
        Filtered (M, 3) array.
    """
    pts = points_3d.reshape(-1, 3)
    mask = np.isfinite(pts[:, 2]) & (pts[:, 2] > z_min)
    if z_max is not None:
        mask &= pts[:, 2] <= z_max
    return pts[mask]
