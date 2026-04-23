"""CREStereo stereo deep learning helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

MODEL_ID = "crestereo"


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class StereoDLBundle:
    """Loaded stereo DL components."""

    model: torch.nn.Module
    processor: object
    device: torch.device


def load_stereo_dl_model() -> StereoDLBundle:
    """Load the pretrained CREStereo model."""
    try:
        from stereo_matching import AutoProcessor, AutoStereoModel
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'stereo_matching'. Install with: pip install stereo_matching"
        ) from exc

    device = _get_device()
    model = AutoStereoModel.from_pretrained(MODEL_ID, device=str(device))
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return StereoDLBundle(model=model, processor=processor, device=device)


def preprocess_stereo_dl(img: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """Convert a BGR image into a normalized batch tensor."""
    if img is None or img.size == 0:
        raise ValueError("Input image must be valid and non-empty.")

    device = device or _get_device()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def _extract_disparity(output) -> np.ndarray:
    if hasattr(output, "disparity") and output.disparity is not None:
        disparity = output.disparity
    else:
        disparity = output

    if isinstance(disparity, torch.Tensor):
        return disparity.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.asarray(disparity, dtype=np.float32)


def compute_stereo_dl_disparity(
    left: np.ndarray,
    right: np.ndarray,
    model: StereoDLBundle | torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Infer a CREStereo disparity map as a float32 array."""
    if model is None:
        model = load_stereo_dl_model()

    if isinstance(model, StereoDLBundle):
        bundle = model
    else:
        bundle = StereoDLBundle(model=model, processor=None, device=device or _get_device())

    if bundle.processor is None:
        raise ValueError("CREStereo processor is unavailable.")

    device = device or bundle.device
    inputs = bundle.processor(left, right)
    left_t = inputs["left_values"].to(device)
    right_t = inputs["right_values"].to(device)

    with torch.no_grad():
        output = bundle.model(left_t, right_t)

    original_sizes = inputs.get("original_sizes")
    if original_sizes is not None:
        output = bundle.processor.postprocess(output, original_sizes, colorize=True)

    disparity = _extract_disparity(output)
    return np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_stereo_dl_depth(
    left: np.ndarray,
    right: np.ndarray,
    focal_length_px: float,
    baseline_meters: float,
    model: StereoDLBundle | torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Convert CREStereo disparity to metric depth using the existing calibration."""
    disp = compute_stereo_dl_disparity(left, right, model=model, device=device)
    disp_safe = disp.copy()
    disp_safe[disp_safe <= 2.0] = np.nan
    depth = (float(focal_length_px) * float(baseline_meters)) / disp_safe
    depth[~np.isfinite(depth)] = np.nan
    return depth.astype(np.float32)
