# geometry.py
# All math that converts 2D–2D matches into geometry: F, E, R, t, 3D

# find fundamental matrix with RANSAC
# find essential matrix from fundamental + K
# find 3d pose from essential matrix

import numpy as np
from typing import Tuple
import cv2 as cv
from scipy.optimize import least_squares

from .structures import CameraIntrinsics
from utils.decompose_essential_matrix import decomposeEssentialMatrix
from utils.disambiguate_relative_pose import disambiguateRelativePose


def estimate_fundamental_ransac(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ransac_thresh: float = 1.0,
    max_iters: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the fundamental matrix using RANSAC.

    Args:
        pts0: Nx2 array of points from image 0.
        pts1: Nx2 array of points from image 1.
        ransac_thresh: RANSAC inlier threshold.
        max_iters: Maximum number of RANSAC iterations.

    Returns:
        F: Estimated fundamental matrix.
        inlier_mask: Boolean mask of inliers.
    """
    F, inlier_mask = cv.findFundamentalMat(
        pts0,
        pts1,
        method=cv.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iters,
    )
    return F, inlier_mask.astype(bool)


def estimate_relative_pose(
    pts0: np.ndarray,
    pts1: np.ndarray,
    intrinsics: CameraIntrinsics,
    ransac_thresh: float = 1.0,
    max_iters: int = 2000,
    method: str = "cvRecoverPose",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the relative pose (R, t) between two views from 2D-2D correspondences.

    Args:
        pts0: Nx2 array of points from image 0.
        pts1: Nx2 array of points from image 1.
        intrinsics: Camera intrinsics.
        ransac_thresh: RANSAC inlier threshold.
        max_iters: Maximum number of RANSAC iterations.
    Returns:
        R: Rotation matrix from view 0 to view 1.
        t: Translation vector from view 0 to view 1.
        inlier_mask: Boolean mask of inliers.
    """
    # Estimate essential matrix using RANSAC
    E, inlier_mask = cv.findEssentialMat(
        pts0,
        pts1,
        cameraMatrix=intrinsics.K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=ransac_thresh,
        maxIters=max_iters,
    )

    # Recover pose from essential matrix
    if method == "cvRecoverPose":
        _, R, t, inlier_mask_pose = cv.recoverPose(
            E,
            pts0,
            pts1,
            cameraMatrix=intrinsics.K,
            mask=inlier_mask,
        )
    elif method == "decomposeEssentialMatrix":
        Rots, u3 = decomposeEssentialMatrix(E)
        # Disambiguate among the four possible configurations
        R, t = disambiguateRelativePose(
            Rots, u3, pts0, pts1, intrinsics.K, intrinsics.K
        )
        # Use the inlier mask from essential matrix estimation
        inlier_mask_pose = inlier_mask

    else:
        raise ValueError(f"Unknown method for pose estimation: {method}")

    return R, t, inlier_mask_pose.astype(bool).reshape(-1)


def triangulate_landmarks(
    pts0: np.ndarray,
    pts1: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Triangulate 3D landmarks from 2D-2D correspondences and relative pose.

    Args:
        pts0: Nx2 array of points from image 0.
        pts1: Nx2 array of points from image 1.
        R: Rotation matrix from view 0 to view 1.
        t: Translation vector from view 0 to view 1.
        intrinsics: Camera intrinsics.
    Returns:
        points_3d: Nx3 array of triangulated 3D points.
    """
    # Projection matrix for first camera (identity pose)
    P0 = intrinsics.K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Projection matrix for second camera (R, t pose)
    P1 = intrinsics.K @ np.hstack((R, t))

    # Triangulate points
    points_4d_hom = cv.triangulatePoints(P0, P1, pts0.T, pts1.T)

    # Convert from homogeneous to Euclidean coordinates
    points_3d = (points_4d_hom[:3, :] / points_4d_hom[3, :]).T

    return points_3d


def two_view_bundle_adjustment(
    pts0: np.ndarray,
    pts1: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    points_3d: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_iterations: int = 50,
    ftol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform 2-view bundle adjustment to refine pose and 3D points by minimizing
    reprojection error.

    The first camera is fixed at the origin (identity pose). We optimize:
    - The relative pose (R, t) of the second camera
    - All 3D landmark positions

    Args:
        pts0: Nx2 array of 2D keypoints in image 0.
        pts1: Nx2 array of 2D keypoints in image 1.
        R: Initial rotation matrix from view 0 to view 1 (3x3).
        t: Initial translation vector from view 0 to view 1 (3x1).
        points_3d: Initial Nx3 array of triangulated 3D points.
        intrinsics: Camera intrinsics.
        max_iterations: Maximum number of optimization iterations.
        ftol: Function tolerance for convergence.

    Returns:
        R_opt: Optimized rotation matrix.
        t_opt: Optimized translation vector.
        points_3d_opt: Optimized Nx3 array of 3D points.
    """
    K = intrinsics.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    n_points = len(points_3d)

    # Filter out points with invalid depth (behind camera or too far)
    valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 500)
    if np.sum(valid_mask) < 8:
        # Not enough valid points, return original
        return R, t, points_3d

    pts0_valid = pts0[valid_mask]
    pts1_valid = pts1[valid_mask]
    points_3d_valid = points_3d[valid_mask]
    n_valid = len(points_3d_valid)

    # Build initial parameter vector
    # Format: [rvec (3), tvec (3), point1 (3), point2 (3), ...]
    # First camera is fixed at identity, so we only optimize second camera pose

    rvec, _ = cv.Rodrigues(R)
    x0 = np.zeros(6 + 3 * n_valid)
    x0[0:3] = rvec.flatten()
    x0[3:6] = t.flatten()
    x0[6:] = points_3d_valid.flatten()

    def residuals(x):
        # Extract pose
        rvec_opt = x[0:3]
        tvec_opt = x[3:6]
        R_opt = _rodrigues_to_matrix(rvec_opt)

        # Extract 3D points
        pts_3d = x[6:].reshape(-1, 3)

        # Project points to camera 0 (identity pose)
        z0 = pts_3d[:, 2]
        valid0 = z0 > 0.1
        z0_safe = np.where(valid0, z0, 1.0)
        x0_proj = fx * pts_3d[:, 0] / z0_safe + cx
        y0_proj = fy * pts_3d[:, 1] / z0_safe + cy

        # Project points to camera 1 (R, t pose)
        pts_cam1 = (R_opt @ pts_3d.T).T + tvec_opt  # (N, 3)
        z1 = pts_cam1[:, 2]
        valid1 = z1 > 0.1
        z1_safe = np.where(valid1, z1, 1.0)
        x1_proj = fx * pts_cam1[:, 0] / z1_safe + cx
        y1_proj = fy * pts_cam1[:, 1] / z1_safe + cy

        # Compute reprojection errors
        err_x0 = np.where(valid0, x0_proj - pts0_valid[:, 0], 100.0)
        err_y0 = np.where(valid0, y0_proj - pts0_valid[:, 1], 100.0)
        err_x1 = np.where(valid1, x1_proj - pts1_valid[:, 0], 100.0)
        err_y1 = np.where(valid1, y1_proj - pts1_valid[:, 1], 100.0)

        # Stack all residuals: [err_x0, err_y0, err_x1, err_y1] interleaved per point
        res = np.empty(4 * n_valid)
        res[0::4] = err_x0
        res[1::4] = err_y0
        res[2::4] = err_x1
        res[3::4] = err_y1

        return res

    # Run optimization
    try:
        result = least_squares(
            residuals,
            x0,
            verbose=0,
            ftol=ftol,
            max_nfev=max_iterations * len(x0),
            method="lm",  # Levenberg-Marquardt
        )
        x_opt = result.x

        # Extract optimized values
        rvec_opt = x_opt[0:3]
        t_opt = x_opt[3:6].reshape(3, 1)
        R_opt = _rodrigues_to_matrix(rvec_opt)
        points_3d_opt_valid = x_opt[6:].reshape(-1, 3)

        # Reconstruct full points array with optimized valid points
        points_3d_opt = points_3d.copy()
        points_3d_opt[valid_mask] = points_3d_opt_valid

        initial_cost = np.sum(residuals(x0) ** 2)
        final_cost = result.cost
        print(
            f"    [2-View BA] Optimized {n_valid} points. "
            f"Reprojection error: {np.sqrt(initial_cost / (4 * n_valid)):.2f} -> {np.sqrt(2 * final_cost / (4 * n_valid)):.2f} px"
        )

        return R_opt, t_opt, points_3d_opt

    except Exception as e:
        print(f"    [2-View BA] Optimization failed: {e}")
        return R, t, points_3d


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues vector to rotation matrix."""
    rvec = np.asarray(rvec).reshape(3, 1)
    R, _ = cv.Rodrigues(rvec)
    return R
