"""Feature detection and matching utilities for stereo image pairs."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def detect_and_compute(
    image: np.ndarray,
    method: str = "ORB",
) -> Tuple[List[cv2.KeyPoint], np.ndarray | None]:
    """Detect keypoints and compute descriptors for an image.

    Args:
        image: Input image in BGR or grayscale format.
        method: Feature method, either "ORB" or "SIFT".

    Returns:
        A tuple of keypoints and descriptors.

    Raises:
        ValueError: If an unsupported method is requested.
    """
    if image is None or image.size == 0:
        return [], None

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_method = method.upper()

    if normalized_method == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
    elif normalized_method == "SIFT":
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unsupported feature detection method: {method}")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(
    desc1: np.ndarray | None,
    desc2: np.ndarray | None,
    method: str = "ORB",
) -> List[cv2.DMatch]:
    """Match feature descriptors between two images.

    Args:
        desc1: Descriptors from the first image.
        desc2: Descriptors from the second image.
        method: Matching method, either "ORB" or "SIFT".

    Returns:
        A list of sorted matches. Empty if descriptors are missing or no matches exist.

    Raises:
        ValueError: If an unsupported method is requested.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    normalized_method = method.upper()

    if normalized_method == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc1, desc2)
    elif normalized_method == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first_match, second_match = pair
            if first_match.distance < 0.75 * second_match.distance:
                matches.append(first_match)
    else:
        raise ValueError(f"Unsupported feature matching method: {method}")

    return sorted(matches, key=lambda match: match.distance)


def extract_matched_points(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert match objects into paired point arrays.

    Args:
        kp1: Keypoints from the first image.
        kp2: Keypoints from the second image.
        matches: Matched feature pairs.

    Returns:
        Two float32 arrays of shape (N, 2).
    """
    if not matches:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()

    pts1 = np.float32([kp1[match.queryIdx].pt for match in matches])
    pts2 = np.float32([kp2[match.trainIdx].pt for match in matches])
    return pts1, pts2


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    max_matches: int = 50,
) -> np.ndarray:
    """Draw the strongest feature matches between two images.

    Args:
        img1: First BGR image.
        img2: Second BGR image.
        kp1: Keypoints for the first image.
        kp2: Keypoints for the second image.
        matches: Sorted matches.
        max_matches: Maximum number of matches to visualize.

    Returns:
        Visualization image in BGR format.
    """
    matches_to_draw = list(matches[:max_matches])
    return cv2.drawMatches(
        img1,
        list(kp1),
        img2,
        list(kp2),
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
