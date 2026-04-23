"""UniMatch stereo deep learning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from src.unimatch.unimatch import UniMatch

MODEL_REPO_ID = "haofeixu/unimatch"
MODEL_REPO_TYPE = "space"
MODEL_FILENAME = (
    "pretrained/"
    "gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth"
)

UNIMATCH_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
UNIMATCH_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

UNIMATCH_FEATURE_CHANNELS = 128
UNIMATCH_NUM_SCALES = 2
UNIMATCH_UPSAMPLE_FACTOR = 4
UNIMATCH_NUM_HEAD = 1
UNIMATCH_FFN_DIM_EXPANSION = 4
UNIMATCH_NUM_TRANSFORMER_LAYERS = 6
UNIMATCH_ATTENTION_TYPE = "self_swin2d_cross_swin1d"
UNIMATCH_ATTN_SPLITS_LIST = [2, 8]
UNIMATCH_CORR_RADIUS_LIST = [-1, 4]
UNIMATCH_PROP_RADIUS_LIST = [-1, 1]
UNIMATCH_NUM_REG_REFINE = 3
UNIMATCH_PADDING_FACTOR = 32
UNIMATCH_MAX_SIDE = 960


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download_checkpoint() -> Path:
    """Download and cache the published UniMatch stereo checkpoint."""
    cache_dir = Path.home() / ".cache" / "cv_proj" / "unimatch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        repo_type=MODEL_REPO_TYPE,
        filename=MODEL_FILENAME,
        cache_dir=str(cache_dir),
    )
    return Path(checkpoint)


@dataclass
class UniMatchBundle:
    """Loaded UniMatch stereo components."""

    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path


def load_unimatch_model() -> UniMatchBundle:
    """Load the pretrained UniMatch stereo model."""
    device = _get_device()
    checkpoint_path = _download_checkpoint()

    model = UniMatch(
        feature_channels=UNIMATCH_FEATURE_CHANNELS,
        num_scales=UNIMATCH_NUM_SCALES,
        upsample_factor=UNIMATCH_UPSAMPLE_FACTOR,
        num_head=UNIMATCH_NUM_HEAD,
        ffn_dim_expansion=UNIMATCH_FFN_DIM_EXPANSION,
        num_transformer_layers=UNIMATCH_NUM_TRANSFORMER_LAYERS,
        reg_refine=True,
        task="stereo",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return UniMatchBundle(model=model, device=device, checkpoint_path=checkpoint_path)


def preprocess_for_unimatch(img: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """Convert a BGR image into the normalized tensor format UniMatch expects."""
    if img is None or img.size == 0:
        raise ValueError("Input image must be valid and non-empty.")

    device = device or _get_device()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - UNIMATCH_IMAGE_MEAN) / UNIMATCH_IMAGE_STD
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def _pad_to_multiple(
    left: torch.Tensor,
    right: torch.Tensor,
    multiple: int = UNIMATCH_PADDING_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Pad a stereo pair symmetrically to a multiple of `multiple`."""
    if left.shape != right.shape:
        raise ValueError("Stereo images must have the same shape.")

    _, _, height, width = left.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    if any(padding):
        left = F.pad(left, padding, mode="replicate")
        right = F.pad(right, padding, mode="replicate")
    return left, right, padding


def _resize_to_max_side(
    left: torch.Tensor,
    right: torch.Tensor,
    max_side: int = UNIMATCH_MAX_SIDE,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Downscale a stereo pair when needed to keep UniMatch memory bounded."""
    _, _, height, width = left.shape
    longest_side = max(height, width)
    if longest_side <= max_side:
        return left, right, 1.0

    scale = float(max_side) / float(longest_side)
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    size = (target_h, target_w)
    left = F.interpolate(left, size=size, mode="bilinear", align_corners=True)
    right = F.interpolate(right, size=size, mode="bilinear", align_corners=True)
    return left, right, float(width) / float(target_w)


def _unpad(tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    """Remove symmetric padding applied by _pad_to_multiple."""
    pad_left, pad_right, pad_top, pad_bottom = padding
    if not any(padding):
        return tensor
    return tensor[
        ...,
        pad_top : tensor.shape[-2] - pad_bottom,
        pad_left : tensor.shape[-1] - pad_right,
    ]


def _extract_disparity(output) -> np.ndarray:
    if isinstance(output, dict):
        disp = output["flow_preds"][-1]
    elif isinstance(output, (list, tuple)):
        disp = output[-1]
    else:
        disp = output

    if isinstance(disp, torch.Tensor):
        disp = disp.detach().cpu().numpy()

    return np.asarray(disp, dtype=np.float32)


def compute_unimatch_disparity(
    left: np.ndarray,
    right: np.ndarray,
    model: UniMatchBundle | torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Infer a UniMatch disparity map as a float32 array."""
    if model is None:
        model = load_unimatch_model()

    if isinstance(model, UniMatchBundle):
        bundle = model
    else:
        bundle = UniMatchBundle(model=model, device=device or _get_device(), checkpoint_path=Path())

    device = device or bundle.device
    left_t = preprocess_for_unimatch(left, device=device)
    right_t = preprocess_for_unimatch(right, device=device)
    left_t, right_t, disparity_scale = _resize_to_max_side(left_t, right_t)
    left_t, right_t, padding = _pad_to_multiple(left_t, right_t, multiple=UNIMATCH_PADDING_FACTOR)

    with torch.no_grad():
        output = bundle.model(
            left_t,
            right_t,
            attn_type=UNIMATCH_ATTENTION_TYPE,
            attn_splits_list=UNIMATCH_ATTN_SPLITS_LIST,
            corr_radius_list=UNIMATCH_CORR_RADIUS_LIST,
            prop_radius_list=UNIMATCH_PROP_RADIUS_LIST,
            num_reg_refine=UNIMATCH_NUM_REG_REFINE,
            task="stereo",
        )

    disparity = _extract_disparity(output)
    if disparity.ndim == 3 and disparity.shape[0] == 1:
        disparity = disparity[0]

    disparity_tensor = torch.from_numpy(disparity).unsqueeze(0).unsqueeze(0).to(device)
    disparity_tensor = _unpad(disparity_tensor, padding)
    disparity = disparity_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    disparity *= float(disparity_scale)
    return np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)


def compute_unimatch_depth(
    left: np.ndarray,
    right: np.ndarray,
    focal_length_px: float,
    baseline_meters: float,
    model: UniMatchBundle | torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Convert UniMatch disparity to metric depth using the existing calibration."""
    disp = compute_unimatch_disparity(left, right, model=model, device=device)
    disp_safe = disp.copy()
    disp_safe[disp_safe <= 2.0] = np.nan
    depth = (float(focal_length_px) * float(baseline_meters)) / disp_safe
    depth[~np.isfinite(depth)] = np.nan
    return depth.astype(np.float32)
