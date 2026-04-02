"""Project-wide configuration constants for paths."""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data"
CALIB_PATH = DATA_PATH / "calibration"
STEREO_PATH = DATA_PATH / "stereo"
OUTPUT_PATH = PROJECT_ROOT / "outputs"

# Number of inner checkerboard corners as (columns, rows).
CALIB_PATTERN_SIZE = (8, 5)

# Intrinsic matrix from single-camera calibration (same camera/lens used for stereo pair).
CAMERA_MATRIX = np.array(
	[
		[3117.47676, 0.0, 1498.35228],
		[0.0, 3111.26225, 2032.87103],
		[0.0, 0.0, 1.0],
	],
	dtype=np.float64,
)
