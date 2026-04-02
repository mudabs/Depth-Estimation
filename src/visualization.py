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
