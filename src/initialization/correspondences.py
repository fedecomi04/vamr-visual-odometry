# correspondences.py
# functions to compute 2D-2D keypoint correspondences between two images

# - Detect & match SIFT features between img0 and img1.
# - Track KLT features from img0 to img1.
# Wrapper function to select between methods.

from typing import List, Tuple

import cv2 as cv
import numpy as np


def compute_keypoint_correspondences_sift(
    img0: np.ndarray,
    img1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[cv.DMatch]]:
    """
    Compute 2D-2D keypoint correspondences between img0 and img1.

    Args:
        img0: First grayscale image.
        img1: Second grayscale image.

    Returns:
        keypoints0: Nx2 array of keypoints in img0.
        keypoints1: Nx2 array of corresponding keypoints in img1.
        descriptors0: NxD array of SIFT descriptors in img0.
        descriptors1: NxD array of SIFT descriptors in img1.
        matches: List of cv2.DMatch objects representing the matches.
    """
    return compute_sift_correspondences(img0, img1)


def compute_keypoint_correspondences_klt(
    img0: np.ndarray,
    img1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D-2D keypoint correspondences between img0 and img1.

    Args:
        img0: First grayscale image.
        img1: Second grayscale image.
    Returns:
        keypoints0: Nx2 array of keypoints in img0.
        keypoints1: Nx2 array of corresponding keypoints in img1.
    """
    return compute_klt_correspondences(img0, img1)


def compute_sift_correspondences(
    img0: np.ndarray,
    img1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[cv.DMatch]]:
    """
    Compute SIFT keypoint correspondences between img0 and img1.

    Args:
        img0: First grayscale image.
        img1: Second grayscale image.

    Returns:
        keypoints0: Nx2 array of keypoints in img0.
        keypoints1: Nx2 array of corresponding keypoints in img1.
        descriptors0: NxD array of SIFT descriptors in img0.
        descriptors1: NxD array of SIFT descriptors in img1.
        matches: List of cv2.DMatch objects representing the matches.
    """

    # Initialize SIFT detector
    sift = cv.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0, None)
    kp1, des1 = sift.detectAndCompute(img1, None)

    # Create BFMatcher object and Match descriptors
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    keypoints0 = np.array([kp0[m.queryIdx].pt for m in good_matches])
    keypoints1 = np.array([kp1[m.trainIdx].pt for m in good_matches])
    descriptors0 = np.array([des0[m.queryIdx] for m in good_matches])
    descriptors1 = np.array([des1[m.trainIdx] for m in good_matches])

    return keypoints0, keypoints1, descriptors0, descriptors1, good_matches


def compute_klt_correspondences(
    img0: np.ndarray,
    img1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KLT keypoint correspondences between img0 and img1.

    Args:
        img0: First grayscale image.
        img1: Second grayscale image.
    Returns:
        keypoints0: Nx2 array of keypoints in img0.
        keypoints1: Nx2 array of corresponding keypoints in img1.
    """
    # Detect good features to track (Shi-Tomasi Corner Detector)
    feature_params = dict(
        maxCorners=2000, qualityLevel=0.01, minDistance=10, blockSize=3
    )
    p0 = cv.goodFeaturesToTrack(img0, mask=None, **feature_params)

    # Track features using Lucas-Kanade Optical Flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p1, status, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

    # 3. Filter only valid points (status == 1)
    good_p1 = p1[status == 1]
    good_p0 = p0[status == 1]

    # Reshape to ensure Nx2 format
    keypoints0 = good_p0.reshape(-1, 2)
    keypoints1 = good_p1.reshape(-1, 2)

    return keypoints0, keypoints1
