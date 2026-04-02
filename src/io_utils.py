"""I/O helpers for loading stereo images, including HEIC support."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a NumPy array in BGR format.

    Args:
        path: Image file path.

    Returns:
        Image as a NumPy array in BGR channel order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is unsupported.
        RuntimeError: If the file cannot be decoded as an image.
    """
    image_path = Path(path)

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    extension = image_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported image format '{extension}' for file: {image_path}. "
            f"Supported formats: {supported}"
        )

    try:
        with Image.open(image_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode not in {"RGB", "RGBA", "L"}:
                pil_img = pil_img.convert("RGB")
            np_img = np.array(pil_img)
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Failed to decode image file: {image_path}") from exc

    if np_img.ndim == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)

    if np_img.shape[2] == 4:
        return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


def load_images_from_folder(folder_path: str | Path) -> Tuple[List[np.ndarray], List[str]]:
    """Load all supported images from a folder in deterministic order.

    Args:
        folder_path: Folder containing image files.

    Returns:
        A tuple containing:
            - List of loaded images in BGR format.
            - Corresponding list of filenames.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If no supported images are found.
        RuntimeError: If any image fails to load.
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_files = sorted(
        [
            file_path
            for file_path in folder.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda file_path: file_path.name.lower(),
    )

    if not image_files:
        raise ValueError(f"No supported images found in folder: {folder}")

    images: List[np.ndarray] = []
    filenames: List[str] = []

    for image_file in image_files:
        try:
            images.append(load_image(image_file))
            filenames.append(image_file.name)
        except Exception as exc:
            raise RuntimeError(f"Error loading '{image_file.name}': {exc}") from exc

    return images, filenames
