"""Streamlit UI for the stereo vision depth estimation pipeline."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.calibration import calibrate_camera, find_checkerboard_corners
from src.config import CALIB_PATTERN_SIZE, CAMERA_MATRIX, OUTPUT_PATH
from src.deep_depth import load_depth_model, predict_relative_depth
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


@st.cache_resource(show_spinner=False)
def _get_cached_depth_model():
    """Load the deep depth model once per Streamlit process."""
    return load_depth_model()


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
        st.session_state.pop("dl_depth_vis", None)
        st.session_state.pop("dl_depth_sig", None)

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
                result = run_classical_pipeline(left, right, camera_matrix)
                st.session_state["pipeline_result"] = result
                st.success("Pipeline completed.")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

    result: PipelineResult | None = st.session_state.get("pipeline_result")
    if result is None:
        st.info("Run the classical pipeline to view outputs.")
        return

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

    st.subheader("Depth Map")
    st.image(result.depth_clean_vis, caption="Filtered depth map", use_container_width=True, clamp=True)
    st.caption("Depth stats below are computed from positive finite depth values.")
    d1, d2, d3 = st.columns(3)
    d1.metric("Min depth", f"{result.depth_min:.3f}")
    d2.metric("Max depth", f"{result.depth_max:.3f}")
    d3.metric("Mean depth", f"{result.depth_mean:.3f}")

    st.subheader("3D Visualization")
    st.write("Sparse 3D points (from triangulation) available for quick inspection.")
    if result.sparse_points.size > 0:
        st.write(f"Sparse points: {len(result.sparse_points)}")
    else:
        st.write("No sparse points reconstructed.")
    st.write("Dense point cloud has been exported as a PLY file and can be downloaded in the Downloads section.")


def _render_comparison_panel() -> None:
    st.header("Comparison")

    result: PipelineResult | None = st.session_state.get("pipeline_result")
    if result is None:
        st.info("Run the classical pipeline first to populate comparisons.")
        return

    st.subheader("Uncalibrated vs Calibrated vs Deep Learning")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Uncalibrated")
        st.image(result.disparity_unrectified, use_container_width=True, clamp=True)

    with col2:
        st.caption("Calibrated")
        st.image(result.disparity_rectified, use_container_width=True, clamp=True)

    with col3:
        st.caption("Deep Learning")
        left_image = st.session_state.get("stereo_left")
        if left_image is None:
            st.warning("Upload stereo images to run deep depth estimation.")
        else:
            sig = st.session_state.get("upload_signature")
            should_recompute = (
                st.session_state.get("dl_depth_vis") is None
                or st.session_state.get("dl_depth_sig") != sig
            )
            if should_recompute:
                with st.spinner("Running Depth Anything on left image..."):
                    try:
                        processor, model = _get_cached_depth_model()
                        st.session_state["dl_depth_vis"] = predict_relative_depth(
                            left_image,
                            processor,
                            model,
                        )
                        st.session_state["dl_depth_sig"] = sig
                    except Exception as exc:
                        st.session_state["dl_depth_vis"] = None
                        st.error(f"Deep model inference failed: {exc}")

            dl_depth = st.session_state.get("dl_depth_vis")
            if dl_depth is not None:
                st.image(dl_depth, use_container_width=True, clamp=True)
                st.caption("Deep learning depth is relative and not metric.")

    st.info("Rectification usually improves geometric consistency; learned refinement can further suppress local noise.")

    st.subheader("Depth Values")
    st.caption("Numerical depth summary for calibrated output.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw Min", f"{result.depth_min:.3f}")
    m2.metric("Raw Max", f"{result.depth_max:.3f}")
    m3.metric("Raw Mean", f"{result.depth_mean:.3f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Clean Min", f"{result.depth_clean_min:.3f}")
    m5.metric("Clean Max", f"{result.depth_clean_max:.3f}")
    m6.metric("Clean Mean", f"{result.depth_clean_mean:.3f}")

    p1, p2, p3 = st.columns(3)
    p1.metric("P5", f"{result.depth_p5:.3f}")
    p2.metric("P50 (Median)", f"{result.depth_p50:.3f}")
    p3.metric("P95", f"{result.depth_p95:.3f}")

    raw_ratio = 100.0 * result.depth_raw_valid_count / max(1, result.depth_total_count)
    clean_ratio = 100.0 * result.depth_clean_valid_count / max(1, result.depth_total_count)
    q1, q2, q3 = st.columns(3)
    q1.metric("Raw Valid Pixels", f"{result.depth_raw_valid_count:,}")
    q2.metric("Clean Valid Pixels", f"{result.depth_clean_valid_count:,}")
    q3.metric("Coverage", f"raw {raw_ratio:.2f}% | clean {clean_ratio:.2f}%")

    st.subheader("Depth Distribution Histogram")
    hist_df = pd.DataFrame(
        {
            "depth": result.depth_hist_bin_centers,
            "count": result.depth_hist_counts,
        }
    )
    st.bar_chart(hist_df.set_index("depth"))

    if result.depth_raw_valid_count == 0:
        st.error("No positive finite depth pixels were found. Disparity is likely invalid for this pair.")
    elif result.depth_clean_valid_count == 0:
        st.warning(
            "Clean depth contains no valid pixels after robust filtering. "
            "This usually means the disparity/depth estimate is highly unstable."
        )
    elif abs(result.depth_max - result.depth_min) < 1e-6:
        st.warning(
            "Raw depth min/max are identical, which indicates a near-constant or degenerate depth estimate."
        )

    if raw_ratio < 1.0:
        st.warning(
            "Very low valid depth coverage (<1%). Depth reliability is poor for this image pair."
        )

    st.subheader("Failure Cases Panel")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.caption("Noisy Disparity (Unrectified)")
        st.image(result.disparity_unrectified, use_container_width=True, clamp=True)
    with f2:
        st.caption("Raw Depth Before Filtering")
        st.image(result.depth_raw_vis, use_container_width=True, clamp=True)
    with f3:
        st.caption("Failure Regions Mask")
        st.image(result.depth_failure_mask, use_container_width=True, clamp=True)

    st.write(
        "Stereo fails in low-texture regions because local windows lack unique intensity patterns. "
        "Repetitive patterns create ambiguous correspondences that can pass local matching checks, "
        "leading to noisy disparity and invalid depth estimates."
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


def _init_state() -> None:
    defaults = {
        "stereo_left": None,
        "stereo_right": None,
        "calib_images": [],
        "calibration_result": None,
        "pipeline_result": None,
        "upload_signature": None,
        "dl_depth_vis": None,
        "dl_depth_sig": None,
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
