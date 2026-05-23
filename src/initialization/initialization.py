# initialization.py
from typing import List, Literal, Optional

import cv2
import numpy as np

from .correspondences import (
    compute_keypoint_correspondences_sift,
    compute_keypoint_correspondences_klt,
)
from .geometry import (
    estimate_relative_pose,
    triangulate_landmarks,
    two_view_bundle_adjustment,
)
from .structures import (
    CameraIntrinsics,
    VOInitializationResultSift,
    VOInitializationResultKLT,
)


def initialize_vo_sift(
    img0: np.ndarray,
    img1: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> VOInitializationResultSift:
    """
    Perform VO initialization:
      Given two frames (img0, img1)
      1. compute 2D-2D correspondences
      2. estimate relative pose with RANSAC
      3. triangulate landmarks

    Args:
        img0: First grayscale or RGB image.
        img1: Second grayscale or RGB image.
        intrinsics: Camera intrinsics (K).
        correspondence_method: 'sift' or 'klt'.

    Returns:
        VOInitializationResult: contains pose, 3D points, and inlier keypoints.
    """
    # TODO: Implement the actual initialization logic here.
    # 1. Compute 2D-2D correspondences
    kp0, kp1, ds0, ds1, _ = compute_keypoint_correspondences_sift(img0, img1)

    # 2. Estimate the relative pose with RANSAC
    R, t, inlier_mask = estimate_relative_pose(
        kp0, kp1, intrinsics, ransac_thresh=1.0, max_iters=2000
    )

    kp0 = kp0[inlier_mask]
    kp1 = kp1[inlier_mask]
    ds0 = ds0[inlier_mask]
    ds1 = ds1[inlier_mask]

    # 3. Triangulate landmarks
    points_3d = triangulate_landmarks(
        kp0,
        kp1,
        R,
        t,
        intrinsics,
    )

    return VOInitializationResultSift(
        R=R,
        t=t,
        points_3d=points_3d,
        keypoints0=kp0,
        keypoints1=kp1,
        descriptors0=ds0,
        descriptors1=ds1,
        inlier_mask=inlier_mask,
    )


def initialize_vo_klt(
    img0: np.ndarray,
    img1: np.ndarray,
    intrinsics: CameraIntrinsics,
    use_bundle_adjustment: bool = True,
) -> VOInitializationResultKLT:
    """
    Perform VO initialization:
      Given two frames (img0, img1)
      1. compute 2D-2D correspondences
      2. estimate relative pose with RANSAC
      3. triangulate landmarks
      4. (optional) refine with 2-view bundle adjustment
    Args:
        img0: First grayscale or RGB image.
        img1: Second grayscale or RGB image.
        intrinsics: Camera intrinsics (K).
        use_bundle_adjustment: Whether to run 2-view BA to refine pose and points.
    Returns:
        VOInitializationResultKLT: contains pose, 3D points, and inlier keypoints.
    """
    # 1. Compute 2D-2D correspondences
    kp0, kp1 = compute_keypoint_correspondences_klt(img0, img1)

    # 2. Estimate the relative pose with RANSAC
    R, t, inlier_mask = estimate_relative_pose(
        kp0, kp1, intrinsics, ransac_thresh=1.0, max_iters=2000
    )

    kp0 = kp0[inlier_mask]
    kp1 = kp1[inlier_mask]

    # 3. Triangulate landmarks
    points_3d = triangulate_landmarks(
        kp0,
        kp1,
        R,
        t,
        intrinsics,
    )

    # 4. Refine with 2-view bundle adjustment
    if use_bundle_adjustment:
        R, t, points_3d = two_view_bundle_adjustment(
            kp0, kp1, R, t, points_3d, intrinsics
        )

    return VOInitializationResultKLT(
        R=R,
        t=t,
        points_3d=points_3d,
        keypoints0=kp0,
        keypoints1=kp1,
        inlier_mask=inlier_mask,
    )
