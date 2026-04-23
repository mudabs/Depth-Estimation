"""Streamlit UI for the stereo vision depth estimation pipeline."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import streamlit as st
import torch
try:
    import plotly.graph_objects as go
except Exception:
    go = None
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.calibration import calibrate_camera, find_checkerboard_corners
from src.config import CALIB_PATTERN_SIZE, CAMERA_MATRIX, OUTPUT_PATH
from src.crestereo_depth import compute_stereo_dl_disparity, load_stereo_dl_model
from src.deep_depth import load_depth_model, predict_relative_depth_map
from src.unimatch_depth import compute_unimatch_disparity, load_unimatch_model
from src.web_pipeline import CalibrationResult, PipelineResult, create_side_by_side, run_classical_pipeline

register_heif_opener()


def _to_bgr(uploaded_file) -> np.ndarray:
    """Decode uploaded image bytes to BGR format."""
    data = uploaded_file.getvalue()
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in {"RGB", "RGBA", "L"}:
            img = img.convert("RGB")
        np_img = np.array(img)

    if np_img.ndim == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    if np_img.shape[2] == 4:
        return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _image_signature(files: Iterable) -> tuple:
    """Build a lightweight signature used to detect upload changes."""
    sig = []
    for f in files or []:
        sig.append((f.name, f.size))
    return tuple(sig)


def _save_calibration_to_disk(result: CalibrationResult) -> Path:
    out_dir = OUTPUT_PATH / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration_results.npz"
    np.savez(
        str(path),
        camera_matrix=result.camera_matrix,
        distortion=result.distortion,
        reprojection_error=np.array([result.reprojection_error], dtype=np.float64),
    )
    return path


def _npz_bytes(result: CalibrationResult) -> bytes:
    buf = io.BytesIO()
    np.savez(
        buf,
        camera_matrix=result.camera_matrix,
        distortion=result.distortion,
        reprojection_error=np.array([result.reprojection_error], dtype=np.float64),
    )
    return buf.getvalue()


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return b""
    return encoded.tobytes()


def _depth_values(depth: np.ndarray, *, max_depth: float | None = None) -> np.ndarray:
    """Return finite positive depth samples with optional upper-bound filtering."""
    values = np.asarray(depth, dtype=np.float32).ravel()
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if max_depth is not None:
        values = values[values < float(max_depth)]
    return values


def _depth_summary(depth: np.ndarray) -> tuple[float | None, float | None, float | None]:
    values = _depth_values(depth, max_depth=5.0)
    if values.size == 0:
        return None, None, None
    return float(values.min()), float(np.median(values)), float(values.max())


def _depth_from_disparity(disparity: np.ndarray, focal_length_px: float, baseline_meters: float) -> np.ndarray:
    """Convert disparity values to approximate metric depth."""
    disp = np.asarray(disparity, dtype=np.float32)
    disparity_safe = disp.copy()
    disparity_safe[~np.isfinite(disparity_safe)] = np.nan
    disparity_safe[disparity_safe <= 2.0] = np.nan
    depth = (float(focal_length_px) * float(baseline_meters)) / disparity_safe
    depth[~np.isfinite(depth)] = np.nan
    return depth


def _depth_to_colormap(depth: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """Render depth values with a simple inferno colormap."""
    depth_vis = np.asarray(depth, dtype=np.float32)
    depth_vis = np.where(np.isfinite(depth_vis) & (depth_vis > 0), depth_vis, 0.0)
    if max_depth is not None:
        depth_vis = np.clip(depth_vis, 0.0, float(max_depth))
    if np.any(depth_vis > 0):
        norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
    else:
        norm = np.zeros_like(depth_vis, dtype=np.float32)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)


def _format_depth_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} m"


@st.cache_resource(show_spinner=False)
def _get_cached_depth_model():
    """Load the deep depth model once per Streamlit process."""
    return load_depth_model()


@st.cache_resource(show_spinner=False)
def get_stereo_dl_model():
    """Load the stereo DL model once per Streamlit process."""
    return load_stereo_dl_model()


@st.cache_resource(show_spinner=False)
def get_unimatch_model():
    """Load the UniMatch stereo model once per Streamlit process."""
    return load_unimatch_model()


def _render_upload_panel() -> None:
    st.header("Upload Data")

    stereo_files = st.file_uploader(
        "Upload exactly 2 stereo images (left, right)",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "heic", "heif"],
        accept_multiple_files=True,
        key="stereo_upload",
    )
    calib_files = st.file_uploader(
        "Upload calibration images (checkerboard)",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "heic", "heif"],
        accept_multiple_files=True,
        key="calib_upload",
    )

    stereo_sig = _image_signature(stereo_files)
    calib_sig = _image_signature(calib_files)
    combined_sig = (stereo_sig, calib_sig)
    if st.session_state.get("upload_signature") != combined_sig:
        st.session_state["upload_signature"] = combined_sig
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("calibration_result", None)
        st.session_state.pop("dl_depth_raw", None)
        st.session_state.pop("dl_depth_sig", None)
        st.session_state.pop("stereo_dl_disp_raw", None)
        st.session_state.pop("stereo_dl_sig", None)
        st.session_state.pop("stereo_dl_status", None)
        st.session_state.pop("unimatch_disp_raw", None)
        st.session_state.pop("unimatch_sig", None)
        st.session_state.pop("unimatch_status", None)

    if len(stereo_files or []) != 2:
        st.warning("Exactly 2 stereo images are required.")
    else:
        try:
            left = _to_bgr(stereo_files[0])
            right = _to_bgr(stereo_files[1])
            st.session_state["stereo_left"] = left
            st.session_state["stereo_right"] = right
            st.success("Stereo pair loaded.")

            c1, c2 = st.columns(2)
            with c1:
                st.image(_bgr_to_rgb(left), caption=f"Left: {stereo_files[0].name}", use_container_width=True)
            with c2:
                st.image(_bgr_to_rgb(right), caption=f"Right: {stereo_files[1].name}", use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to decode stereo images: {exc}")

    if calib_files:
        if len(calib_files) < 5:
            st.warning("At least 5 calibration images are recommended for stable results.")

        calib_images = []
        for file in calib_files:
            try:
                calib_images.append(_to_bgr(file))
            except Exception as exc:
                st.error(f"Failed to decode calibration image {file.name}: {exc}")

        if calib_images:
            st.session_state["calib_images"] = calib_images
            st.info(f"Loaded {len(calib_images)} calibration images.")
            preview_cols = st.columns(min(3, len(calib_images)))
            for idx, img in enumerate(calib_images[:3]):
                with preview_cols[idx]:
                    st.image(_bgr_to_rgb(img), caption=f"Calibration #{idx + 1}", use_container_width=True)


def _render_calibration_panel() -> None:
    st.header("Calibration")

    col1, col2 = st.columns(2)
    with col1:
        pattern_cols = st.number_input("Checkerboard corners (columns)", min_value=3, value=CALIB_PATTERN_SIZE[0])
    with col2:
        pattern_rows = st.number_input("Checkerboard corners (rows)", min_value=3, value=CALIB_PATTERN_SIZE[1])

    if st.button("Run Calibration", type="primary"):
        calib_images = st.session_state.get("calib_images", [])
        if not calib_images:
            st.error("Upload calibration images first.")
            return

        with st.spinner("Running camera calibration..."):
            try:
                objpoints, imgpoints, valid_images = find_checkerboard_corners(
                    calib_images,
                    pattern_size=(int(pattern_cols), int(pattern_rows)),
                    verbose=False,
                    use_slow_fallback=False,
                )
                if len(valid_images) == 0:
                    st.error("No checkerboard was detected. Check pattern size and image quality.")
                    return

                image_shape = valid_images[0].shape[:2]
                K, dist, _rvecs, _tvecs, reproj_error = calibrate_camera(objpoints, imgpoints, image_shape)
                result = CalibrationResult(
                    camera_matrix=K,
                    distortion=dist,
                    reprojection_error=float(reproj_error),
                    valid_images=len(valid_images),
                )
                st.session_state["calibration_result"] = result
                st.success("Calibration completed.")
            except Exception as exc:
                st.error(f"Calibration failed: {exc}")
                return

    result: CalibrationResult | None = st.session_state.get("calibration_result")
    if result is None:
        st.info("No calibration in session. You can still run the classical pipeline using default intrinsics.")
        return

    st.subheader("Calibration Results")
    st.write(f"Valid images used: {result.valid_images}")
    st.write(f"Reprojection error: {result.reprojection_error:.6f}")
    st.text("Camera Matrix (K)")
    st.code(np.array2string(result.camera_matrix, precision=4), language="text")
    st.text("Distortion Coefficients")
    st.code(np.array2string(result.distortion, precision=6), language="text")

    if st.button("Save Calibration Results"):
        save_path = _save_calibration_to_disk(result)
        st.success(f"Calibration saved to {save_path}")


def _render_classical_pipeline_panel() -> None:
    st.header("Classical Pipeline")

    # --- Baseline & metric toggle ---
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        baseline_meters = st.number_input(
            "Baseline (meters)",
            min_value=0.01,
            max_value=0.50,
            value=st.session_state.get("baseline", 0.10),
            step=0.01,
            format="%.2f",
            help="Physical distance between stereo camera centers in meters.",
        )
        st.session_state["baseline"] = baseline_meters
    with col_b2:
        use_metric = st.toggle(
            "Use metric scaling (baseline)",
            value=st.session_state.get("use_metric_scaling", True),
            help="ON = depth in meters using baseline. OFF = relative depth (unit translation).",
        )
        st.session_state["use_metric_scaling"] = use_metric

    if st.button("Run Classical Pipeline", type="primary"):
        left = st.session_state.get("stereo_left")
        right = st.session_state.get("stereo_right")
        if left is None or right is None:
            st.error("Upload exactly 2 stereo images first.")
            return

        calib_result: CalibrationResult | None = st.session_state.get("calibration_result")
        if calib_result is None:
            camera_matrix = CAMERA_MATRIX
            st.warning("Using default camera intrinsics from config (no session calibration).")
        else:
            camera_matrix = calib_result.camera_matrix

        with st.spinner("Running full classical stereo pipeline..."):
            try:
                result = run_classical_pipeline(
                    left,
                    right,
                    camera_matrix,
                    baseline_meters=float(baseline_meters),
                    use_metric_scaling=use_metric,
                )
                st.session_state["pipeline_result"] = result
                st.success("Pipeline completed.")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

    result: PipelineResult | None = st.session_state.get("pipeline_result")
    if result is None:
        st.info("Run the classical pipeline to view outputs.")
        return

    unit = "m" if result.use_metric_scaling else "(relative)"

    # --- Validation Warnings ---
    _render_validation_warnings(result)

    st.subheader("Feature Matching")
    st.image(_bgr_to_rgb(result.match_visualization), caption="ORB inlier matches", use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Keypoints (Left)", result.keypoints_left)
    m2.metric("Keypoints (Right)", result.keypoints_right)
    m3.metric("Matches", result.num_matches)
    m4.metric("Inliers", result.num_inliers)

    st.subheader("Rectification")
    rect_pair = create_side_by_side(result.rectified_left, result.rectified_right, "Rectified Left", "Rectified Right")
    st.image(_bgr_to_rgb(rect_pair), caption="Rectified stereo pair", use_container_width=True)

    st.subheader("Disparity Map")
    st.image(result.disparity_rectified, caption="Brighter = closer, darker = farther", use_container_width=True, clamp=True)

    # --- Depth Visualization ---
    st.subheader("Depth Map")
    calibrated_depth = _depth_from_disparity(result.disparity_map_raw, result.focal_length_px, result.baseline_meters)
    st.image(_depth_to_colormap(calibrated_depth), caption="Calibrated stereo depth", use_container_width=True)
    calib_min, calib_median, calib_max = _depth_summary(calibrated_depth)
    depth_cols = st.columns(3)
    depth_cols[0].metric("Min Depth", _format_depth_metric(calib_min))
    depth_cols[1].metric("Median Depth", _format_depth_metric(calib_median))
    depth_cols[2].metric("Max Depth", _format_depth_metric(calib_max))

    # --- Pixel depth query ---
    st.subheader("Pixel Depth Query")
    st.caption("Enter pixel coordinates to read depth and disparity at that location.")
    qc1, qc2 = st.columns(2)
    with qc1:
        px_x = st.number_input("X (column)", min_value=0, max_value=max(0, result.depth_map_raw.shape[1] - 1), value=result.depth_map_raw.shape[1] // 2)
    with qc2:
        px_y = st.number_input("Y (row)", min_value=0, max_value=max(0, result.depth_map_raw.shape[0] - 1), value=result.depth_map_raw.shape[0] // 2)
    dval = calibrated_depth[int(px_y), int(px_x)]
    dispval = result.disparity_map_raw[int(px_y), int(px_x)]
    is_confident = bool(result.confidence_map[int(px_y), int(px_x)] > 0)
    if np.isfinite(dval) and dval > 0:
        conf_label = "valid" if is_confident else "low confidence"
        st.success(
            f"Pixel ({int(px_x)}, {int(px_y)}) → Depth: {dval:.4f} m | "
            f"Disparity: {dispval:.2f} px | Confidence: {conf_label}"
        )
    else:
        st.warning(f"Pixel ({int(px_x)}, {int(px_y)}) → No valid depth (invalid/occluded)")

    # --- 3D Visualization ---
    st.subheader("3D Visualization")
    if go is None:
        st.warning("Plotly is not installed. Install it with: pip install plotly")
    elif result.preview_points_xyz.size == 0:
        st.warning("No valid dense 3D points available for preview.")
    else:
        max_n = int(len(result.preview_points_xyz))
        n_show = st.slider(
            "Preview points",
            min_value=1000 if max_n >= 1000 else 100,
            max_value=max_n,
            value=min(5000, max_n),
            step=500 if max_n >= 5000 else 100,
        )

        pts = result.preview_points_xyz[:n_show]
        cols = result.preview_points_rgb[:n_show]
        colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in cols]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker={"size": 2, "color": colors, "opacity": 0.8},
                )
            ]
        )
        fig.update_layout(
            height=600,
            margin={"l": 0, "r": 0, "b": 0, "t": 30},
            scene={
                "xaxis_title": f"X ({unit})",
                "yaxis_title": f"Y ({unit})",
                "zaxis_title": f"Z ({unit})",
                "aspectmode": "data",
            },
            title="Point Cloud Preview",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Debug Panel ---
    with st.expander("Debug Panel", expanded=False):
        st.caption("Internal diagnostics.")

        db1, db2, db3 = st.columns(3)
        db1.metric("Baseline (m)", f"{result.baseline_meters:.3f}")
        db2.metric("Focal Length (px)", f"{result.focal_length_px:.1f}")
        db3.metric("Metric Scaling", "ON" if result.use_metric_scaling else "OFF")

        st.markdown("**Disparity Statistics (pixels)**")
        dd1, dd2, dd3 = st.columns(3)
        dd1.metric("Disp Min", f"{result.disparity_min:.2f}")
        dd2.metric("Disp Max", f"{result.disparity_max:.2f}")
        dd3.metric("Disp Mean", f"{result.disparity_mean:.2f}")

        st.markdown("**Geometry Matrices**")
        st.markdown("**F (Fundamental Matrix)**")
        st.code(np.array2string(result.fundamental_matrix, precision=6, suppress_small=False), language="text")
        st.markdown("**E (Essential Matrix)**")
        st.code(np.array2string(result.essential_matrix, precision=6, suppress_small=False), language="text")
        st.markdown("**R (Rotation Matrix)**")
        st.code(np.array2string(result.rotation_matrix, precision=6, suppress_small=False), language="text")
        st.markdown("**t (Translation Vector)** — scaled by baseline" if result.use_metric_scaling else "**t (Translation Vector)** — unit (no scaling)")
        st.code(np.array2string(result.translation_vector.reshape(-1, 1), precision=6, suppress_small=False), language="text")

    # Sparse info
    if result.sparse_points.size > 0:
        st.caption(f"Sparse triangulated points: {len(result.sparse_points)}")


def _render_validation_warnings(result: PipelineResult) -> None:
    """Show automatic warnings based on depth quality heuristics."""
    raw_ratio = 100.0 * result.depth_raw_valid_count / max(1, result.depth_total_count)

    if result.use_metric_scaling and result.depth_mean > 20.0:
        st.warning(
            f"⚠ Mean depth {result.depth_mean:.1f} m > 20 m — disparity may be too small or baseline incorrect."
        )

    if raw_ratio < 30.0:
        st.warning("Low depth coverage. Improve scene texture or tune SGBM parameters.")

    if result.disparity_valid_pct < 10.0:
        st.warning("Very low disparity coverage. Scene may lack texture or numDisparities is too small.")

    if result.depth_raw_valid_count > 0 and abs(result.depth_max - result.depth_min) < 1e-6:
        st.warning("Depth min/max are identical — indicates a degenerate or near-constant depth estimate.")


def _render_validation_warnings(result: PipelineResult) -> None:
    """Show a minimal set of depth quality warnings."""
    if result.use_metric_scaling and result.depth_mean > 20.0:
        st.warning(
            f"Mean depth {result.depth_mean:.1f} m > 20 m; disparity may be too small or baseline incorrect."
        )

    if result.depth_raw_valid_count > 0 and abs(result.depth_max - result.depth_min) < 1e-6:
        st.warning("Depth min/max are identical, which suggests a degenerate or near-constant depth estimate.")


def _render_comparison_panel() -> None:
    st.header("Comparison")

    result: PipelineResult | None = st.session_state.get("pipeline_result")
    if result is None:
        st.info("Run the classical pipeline first to populate comparisons.")
        return

    st.subheader("Depth Comparison")

    unrectified_disparity = getattr(
        result,
        "disparity_unrectified_raw",
        result.disparity_map_raw,
    )
    uncalibrated_depth = _depth_from_disparity(
        unrectified_disparity,
        result.focal_length_px,
        result.baseline_meters,
    )
    calibrated_depth = _depth_from_disparity(
        result.disparity_map_raw,
        result.focal_length_px,
        result.baseline_meters,
    )

    calibrated_values = _depth_values(calibrated_depth, max_depth=None)
    calibrated_mean = float(np.mean(calibrated_values)) if calibrated_values.size else None

    left_image = st.session_state.get("stereo_left")
    right_image = st.session_state.get("stereo_right")
    dl_depth_raw = None
    if left_image is not None:
        sig = st.session_state.get("upload_signature")
        should_recompute = (
            st.session_state.get("dl_depth_raw") is None
            or st.session_state.get("dl_depth_sig") != sig
        )
        if should_recompute:
            with st.spinner("Running Depth Anything on left image..."):
                try:
                    processor, model = _get_cached_depth_model()
                    st.session_state["dl_depth_raw"] = predict_relative_depth_map(
                        left_image,
                        processor,
                        model,
                    )
                    st.session_state["dl_depth_sig"] = sig
                except Exception as exc:
                    st.session_state["dl_depth_raw"] = None
                    st.error(f"Deep model inference failed: {exc}")

        dl_depth_raw = st.session_state.get("dl_depth_raw")

    dl_depth_metric = None
    if dl_depth_raw is not None:
        dl_values = _depth_values(dl_depth_raw, max_depth=None)
        dl_mean = float(np.mean(dl_values)) if dl_values.size else None
        if calibrated_mean is not None and dl_mean is not None and dl_mean > 0:
            scale = calibrated_mean / dl_mean
            dl_depth_metric = np.asarray(dl_depth_raw, dtype=np.float32) * scale

    stereo_dl_depth = None
    stereo_dl_note = "Stereo DL (CREStereo) uses learned matching and may improve results in low-texture regions."
    if left_image is not None and right_image is not None:
        stereo_dl_sig = st.session_state.get("upload_signature")
        should_recompute_stereo_dl = (
            st.session_state.get("stereo_dl_status") != "unavailable"
            and (
                st.session_state.get("stereo_dl_disp_raw") is None
                or st.session_state.get("stereo_dl_sig") != stereo_dl_sig
            )
        )
        if should_recompute_stereo_dl:
            try:
                with st.spinner("Running Stereo DL on stereo pair..."):
                    model = get_stereo_dl_model()
                    st.session_state["stereo_dl_disp_raw"] = compute_stereo_dl_disparity(
                        left_image,
                        right_image,
                        model,
                    )
                    st.session_state["stereo_dl_sig"] = stereo_dl_sig
                    st.session_state["stereo_dl_status"] = "available"
            except Exception as exc:
                st.session_state["stereo_dl_disp_raw"] = None
                st.session_state["stereo_dl_sig"] = stereo_dl_sig
                st.session_state["stereo_dl_status"] = "unavailable"
                st.warning("Stereo DL failed to run.")
                print(exc)

        stereo_dl_disp_raw = st.session_state.get("stereo_dl_disp_raw")
        if stereo_dl_disp_raw is not None:
            stereo_dl_depth = _depth_from_disparity(
                stereo_dl_disp_raw,
                result.focal_length_px,
                result.baseline_meters,
            )

    unimatch_depth = None
    unimatch_note = "UniMatch is a learned stereo matcher and can improve fine detail and low-texture regions."
    if left_image is not None and right_image is not None:
        unimatch_sig = st.session_state.get("upload_signature")
        should_recompute_unimatch = (
            st.session_state.get("unimatch_status") != "unavailable"
            and (
                st.session_state.get("unimatch_disp_raw") is None
                or st.session_state.get("unimatch_sig") != unimatch_sig
            )
        )
        if should_recompute_unimatch:
            try:
                with st.spinner("Running UniMatch on stereo pair..."):
                    model = get_unimatch_model()
                    st.session_state["unimatch_disp_raw"] = compute_unimatch_disparity(
                        left_image,
                        right_image,
                        model,
                    )
                    st.session_state["unimatch_sig"] = unimatch_sig
                    st.session_state["unimatch_status"] = "available"
            except Exception as exc:
                st.session_state["unimatch_disp_raw"] = None
                st.session_state["unimatch_sig"] = unimatch_sig
                st.session_state["unimatch_status"] = "unavailable"
                st.warning("UniMatch failed to run.")
                print(exc)

        unimatch_disp_raw = st.session_state.get("unimatch_disp_raw")
        if unimatch_disp_raw is not None:
            unimatch_depth = _depth_from_disparity(
                unimatch_disp_raw,
                result.focal_length_px,
                result.baseline_meters,
            )

    def _render_depth_card(
        column,
        title: str,
        depth_values: np.ndarray,
        image: np.ndarray | None,
        note: str | None = None,
    ) -> None:
        with column:
            st.markdown(f"#### {title}")
            if image is not None:
                st.image(image, use_container_width=True)

            min_val, median_val, max_val = _depth_summary(depth_values)
            st.metric("Min Depth", _format_depth_metric(min_val))
            st.metric("Median Depth", _format_depth_metric(median_val))
            st.metric("Max Depth", _format_depth_metric(max_val))

            if note:
                st.caption(note)

    col1, col2, col3, col4, col5 = st.columns(5)

    _render_depth_card(
        col1,
        "Uncalibrated Stereo",
        uncalibrated_depth,
        _depth_to_colormap(uncalibrated_depth),
        note="Uses stereo matching without calibration. Depth is inconsistent and not reliable in scale.",
    )

    _render_depth_card(
        col2,
        "Calibrated Stereo",
        calibrated_depth,
        _depth_to_colormap(calibrated_depth),
        note="Uses calibrated stereo geometry. Produces metric depth but may contain noise and missing regions.",
    )

    _render_depth_card(
        col3,
        "Enhanced Stereo (WLS Filtered)",
        stereo_dl_depth if stereo_dl_depth is not None else np.array([], dtype=np.float32),
        _depth_to_colormap(stereo_dl_depth) if stereo_dl_depth is not None else None,
        note="Applies edge-preserving WLS filtering to refine stereo results. Reduces noise and improves surface consistency.",
    )

    _render_depth_card(
        col4,
        "Stereo DL (UniMatch)",
        unimatch_depth if unimatch_depth is not None else np.array([], dtype=np.float32),
        _depth_to_colormap(unimatch_depth) if unimatch_depth is not None else None,
        note="Uses a deep learning stereo model for pixel matching. Produces smoother and more complete depth than classical methods.",
    )

    _render_depth_card(
        col5,
        "Monocular DL (Depth Anything)",
        dl_depth_metric if dl_depth_metric is not None else np.array([], dtype=np.float32),
        _depth_to_colormap(dl_depth_metric) if dl_depth_metric is not None else None,
        note="Estimates depth from a single image using deep learning. Produces smooth results but lacks true metric scale.",
    )   


def _render_download_panel() -> None:
    st.header("Downloads")

    result: PipelineResult | None = st.session_state.get("pipeline_result")
    calib_result: CalibrationResult | None = st.session_state.get("calibration_result")

    if result is None and calib_result is None:
        st.info("Run calibration and/or pipeline to enable downloads.")
        return

    if result is not None:
        st.subheader("Pipeline Artifacts")
        st.download_button(
            label="Download Depth Map (PNG)",
            data=_png_bytes(result.depth_clean_vis),
            file_name="depth_clean.png",
            mime="image/png",
        )
        st.download_button(
            label="Download Depth Colormap (PNG)",
            data=_png_bytes(result.depth_colored),
            file_name="depth_colormap.png",
            mime="image/png",
        )
        st.download_button(
            label="Download Disparity Map (PNG)",
            data=_png_bytes(result.disparity_rectified),
            file_name="disparity_map_rectified.png",
            mime="image/png",
        )

        if result.pointcloud_path.exists():
            st.download_button(
                label="Download Point Cloud (.ply)",
                data=result.pointcloud_path.read_bytes(),
                file_name="scene.ply",
                mime="application/octet-stream",
            )
        else:
            st.warning("Point cloud file not found. Run classical pipeline again.")

    if calib_result is not None:
        st.subheader("Calibration Artifact")
        st.download_button(
            label="Download Calibration File (.npz)",
            data=_npz_bytes(calib_result),
            file_name="calibration_results.npz",
            mime="application/octet-stream",
        )

    # --- Save / Load Calibration Settings ---
    st.subheader("Calibration Settings (.npz)")
    st.caption("Saves baseline, camera matrix, and metric toggle for reuse across sessions.")

    save_col, load_col = st.columns(2)
    with save_col:
        baseline_val = st.session_state.get("baseline", 0.10)
        metric_val = st.session_state.get("use_metric_scaling", True)
        cam = CAMERA_MATRIX
        if calib_result is not None:
            cam = calib_result.camera_matrix
        buf = io.BytesIO()
        np.savez(buf, baseline_meters=np.array([baseline_val]), camera_matrix=cam, use_metric_scaling=np.array([metric_val]))
        buf.seek(0)
        st.download_button(
            label="Save Calibration Settings (.npz)",
            data=buf.getvalue(),
            file_name="calibration_settings.npz",
            mime="application/octet-stream",
        )

    with load_col:
        uploaded_settings = st.file_uploader("Load Calibration Settings (.npz)", type=["npz"], key="settings_upload")
        if uploaded_settings is not None:
            try:
                data = np.load(io.BytesIO(uploaded_settings.getvalue()), allow_pickle=False)
                if "baseline_meters" in data:
                    st.session_state["baseline"] = float(data["baseline_meters"].item())
                if "use_metric_scaling" in data:
                    st.session_state["use_metric_scaling"] = bool(data["use_metric_scaling"].item())
                if "camera_matrix" in data:
                    loaded_cam = data["camera_matrix"]
                    if loaded_cam.shape == (3, 3):
                        st.session_state["calibration_result"] = CalibrationResult(
                            camera_matrix=loaded_cam,
                            distortion=np.zeros(5),
                            reprojection_error=0.0,
                            valid_images=0,
                        )
                st.success(f"Settings loaded — baseline={st.session_state['baseline']:.2f}m, metric={'ON' if st.session_state['use_metric_scaling'] else 'OFF'}")
            except Exception as exc:
                st.error(f"Failed to load settings: {exc}")


def _init_state() -> None:
    defaults = {
        "stereo_left": None,
        "stereo_right": None,
        "calib_images": [],
        "calibration_result": None,
        "pipeline_result": None,
        "upload_signature": None,
        "dl_depth_raw": None,
        "dl_depth_sig": None,
        "stereo_dl_disp_raw": None,
        "stereo_dl_sig": None,
        "stereo_dl_status": None,
        "baseline": 0.10,
        "use_metric_scaling": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="Stereo Vision Depth Estimation System", layout="wide")
    _init_state()

    st.title("Stereo Vision Depth Estimation System")
    st.caption("Interactive UI for calibration, classical stereo reconstruction, comparisons, and downloads.")

    nav = st.sidebar.radio(
        "Navigation",
        ["Upload Data", "Calibration", "Classical Pipeline", "Comparison", "Downloads"],
    )

    if nav == "Upload Data":
        _render_upload_panel()
    elif nav == "Calibration":
        _render_calibration_panel()
    elif nav == "Classical Pipeline":
        _render_classical_pipeline_panel()
    elif nav == "Comparison":
        _render_comparison_panel()
    elif nav == "Downloads":
        _render_download_panel()

    # Helpful status box at bottom.
    st.sidebar.markdown("---")
    status = {
        "stereo_loaded": st.session_state.get("stereo_left") is not None,
        "calib_images": len(st.session_state.get("calib_images", [])),
        "calibration_ready": st.session_state.get("calibration_result") is not None,
        "pipeline_ready": st.session_state.get("pipeline_result") is not None,
    }
    st.sidebar.caption("Session Status")
    st.sidebar.code(json.dumps(status, indent=2), language="json")


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        print("This is a Streamlit app.")
        print("Run it with:")
        print("  streamlit run app.py")
    else:
        main()
