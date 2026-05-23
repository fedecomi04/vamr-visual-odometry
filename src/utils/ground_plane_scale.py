# ground_plane_scale.py
"""
Ground Plane Scale Correction Module

Uses road mask to identify ground points, fits a plane via RANSAC,
and corrects scale based on known camera height.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque

import cv2
import numpy as np


@dataclass
class GroundPlaneConfig:
    """Configuration for ground plane scale estimation."""

    enabled: bool = True
    camera_height: float = 1.65  # meters (KITTI camera height)
    deque_maxlen: int = 500  # Max ground points to accumulate
    ransac_thresh: float = 0.1  # RANSAC inlier distance threshold (meters)
    ransac_iterations: int = 500  # RANSAC iteration count
    min_ground_points: int = 30  # Min points needed for plane fit
    correction_alpha: float = 0.2  # Smoothing factor for gradual correction
    height_tolerance: float = 0.15  # Height error before correction (meters)
    correction_interval: int = 5  # Frames between corrections
    min_plane_normal_y: float = (
        0.7  # Minimum Y component of plane normal (should point up)
    )
    min_inlier_ratio: float = 0.5  # Minimum ratio of inliers to total points


@dataclass
class GroundPlaneResult:
    """Result of ground plane estimation."""

    plane_params: Optional[np.ndarray] = None  # (a, b, c, d) where ax + by + cz + d = 0
    inlier_mask: Optional[np.ndarray] = None
    camera_height: float = 0.0
    scale_factor: float = 1.0
    num_inliers: int = 0
    is_valid: bool = False
    skipped: bool = False  # True if estimation was skipped (not correction frame)


class GroundPlaneScaleEstimator:
    """
    Estimates ground plane from 3D landmarks and computes scale correction
    to maintain consistent camera height.
    """

    def __init__(self, config: GroundPlaneConfig = None):
        self.config = config or GroundPlaneConfig()

        # Deque to accumulate ground points across frames for robustness
        self.ground_points_deque: Deque[np.ndarray] = deque(
            maxlen=self.config.deque_maxlen
        )

        # Track the current estimated plane
        self.current_plane: Optional[np.ndarray] = None
        self.frame_counter = 0

        # For visualization: track points that would be added vs already in deque
        self.last_candidate_points: Optional[np.ndarray] = (
            None  # Points to potentially add
        )
        self.last_candidate_2d: Optional[np.ndarray] = (
            None  # 2D projections of candidates
        )

    def filter_ground_landmarks(
        self,
        landmarks_3d: np.ndarray,
        keypoints_2d: np.ndarray,
        landmark_indices: np.ndarray,
        road_mask: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter 3D landmarks that project onto road regions.

        Args:
            landmarks_3d: (M, 3) array of all 3D landmarks in world frame
            keypoints_2d: (N, 2) array of 2D keypoints in current frame
            landmark_indices: (N,) array mapping keypoints to landmark IDs (-1 if no landmark)
            road_mask: (H, W) binary mask where True/1 indicates road
            K: (3, 3) camera intrinsic matrix
            R: (3, 3) rotation matrix (world to camera)
            t: (3, 1) translation vector (world to camera)

        Returns:
            ground_points_3d: (G, 3) array of ground landmarks
            ground_keypoints_2d: (G, 2) array of corresponding 2D keypoints
            ground_indices: (G,) array of original landmark indices
        """
        if landmarks_3d.size == 0 or keypoints_2d.size == 0:
            return np.empty((0, 3)), np.empty((0, 2)), np.empty((0,), dtype=int)

        # Get landmarks that have valid 2D observations
        valid_mask = landmark_indices != -1
        if not np.any(valid_mask):
            return np.empty((0, 3)), np.empty((0, 2)), np.empty((0,), dtype=int)

        valid_kp_2d = keypoints_2d[valid_mask]
        valid_landmark_ids = landmark_indices[valid_mask]

        # Check which keypoints fall on road mask
        ground_mask = []
        for kp in valid_kp_2d:
            x, y = int(round(kp[0])), int(round(kp[1]))
            # Bounds check
            if 0 <= y < road_mask.shape[0] and 0 <= x < road_mask.shape[1]:
                is_road = road_mask[y, x] > 0
            else:
                is_road = False
            ground_mask.append(is_road)

        ground_mask = np.array(ground_mask, dtype=bool)

        if not np.any(ground_mask):
            return np.empty((0, 3)), np.empty((0, 2)), np.empty((0,), dtype=int)

        # Filter to get ground points
        ground_kp_2d = valid_kp_2d[ground_mask]
        ground_landmark_ids = valid_landmark_ids[ground_mask]
        ground_points_3d = landmarks_3d[ground_landmark_ids]

        # Additional filter: points should be in front of camera and below camera
        # Transform to camera frame
        X_cam = (R @ ground_points_3d.T + t).T  # (G, 3)

        # Keep points in front of camera (positive Z) and below camera (positive Y in camera frame)
        valid_depth = (X_cam[:, 2] > 0.5) & (
            X_cam[:, 2] < 100
        )  # Reasonable depth range
        valid_below = X_cam[:, 1] > 0  # Below camera in camera frame (Y points down)

        final_mask = valid_depth & valid_below

        if not np.any(final_mask):
            return np.empty((0, 3)), np.empty((0, 2)), np.empty((0,), dtype=int)

        return (
            ground_points_3d[final_mask],
            ground_kp_2d[final_mask],
            ground_landmark_ids[final_mask],
        )

    def estimate_ground_plane_ransac(
        self, points_3d: np.ndarray, min_points: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RANSAC plane fitting to 3D points.

        The plane equation is: ax + by + cz + d = 0
        where (a, b, c) is the unit normal.

        Args:
            points_3d: (N, 3) array of 3D points
            min_points: Minimum number of inliers required (defaults to config.min_ground_points)

        Returns:
            plane_params: (4,) array [a, b, c, d] or None if failed
            inlier_mask: (N,) boolean array or None if failed
        """
        if min_points is None:
            min_points = self.config.min_ground_points
            
        n_points = len(points_3d)
        if n_points < 3:
            return None, None

        best_plane = None
        best_inliers = None
        best_num_inliers = 0

        for _ in range(self.config.ransac_iterations):
            # Sample 3 random points
            sample_idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points_3d[sample_idx]

            # Compute plane from 3 points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue  # Degenerate case (collinear points)

            normal = normal / norm_len  # Unit normal
            d = -np.dot(normal, p1)

            # Ensure normal points "up" (positive Y component for KITTI coordinate system)
            # In KITTI, Y points down, so ground plane normal should have negative Y
            # Actually, let's check: if camera is above ground, ground normal should point towards camera
            # We'll validate this later with min_plane_normal_y check

            # Compute distances to plane
            distances = np.abs(points_3d @ normal + d)
            inliers = distances < self.config.ransac_thresh
            num_inliers = np.sum(inliers)

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d])

        if best_plane is None or best_num_inliers < min_points:
            return None, None

        # Refit plane using all inliers (least squares)
        inlier_points = points_3d[best_inliers]
        refined_plane = self._fit_plane_least_squares(inlier_points)

        if refined_plane is not None:
            # Recompute inliers with refined plane
            normal = refined_plane[:3]
            d = refined_plane[3]
            distances = np.abs(points_3d @ normal + d)
            best_inliers = distances < self.config.ransac_thresh
            best_plane = refined_plane

        return best_plane, best_inliers

    def _fit_plane_least_squares(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit plane to points using SVD (least squares).
        """
        if len(points) < 3:
            return None

        # Center the points
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # SVD to find best-fit plane
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # Last row of Vt is the normal to the best-fit plane

        # Ensure consistent normal direction (pointing "up")
        # In world coordinates, Y typically points down for KITTI
        # Ground plane normal should point up (negative Y in KITTI world frame)
        # But we'll handle this in validation

        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            return None
        normal = normal / norm_len

        d = -np.dot(normal, centroid)
        return np.array([normal[0], normal[1], normal[2], d])

    def get_camera_height_from_plane(
        self, R: np.ndarray, t: np.ndarray, plane: np.ndarray
    ) -> float:
        """
        Compute the height of camera above the ground plane.

        Args:
            R: (3, 3) rotation matrix (world to camera)
            t: (3, 1) translation vector (world to camera)
            plane: (4,) plane parameters [a, b, c, d]

        Returns:
            height: Signed distance from camera to plane (positive = above)
        """
        # Camera center in world coordinates
        C = (-R.T @ t).flatten()

        # Plane equation: ax + by + cz + d = 0
        # Distance from point to plane
        normal = plane[:3]
        d = plane[3]

        # Signed distance
        distance = (np.dot(normal, C) + d) / np.linalg.norm(normal)

        # Return absolute height (we expect camera to be above ground)
        return abs(distance)

    def compute_scale_correction(
        self, current_height: float, smooth: bool = True
    ) -> float:
        """
        Compute scale factor to correct camera height to target.

        Args:
            current_height: Current estimated camera height
            smooth: Whether to apply smoothing

        Returns:
            scale_factor: Factor to multiply landmarks and translation by
        """
        if current_height < 1e-6:
            return 1.0

        raw_scale = self.config.camera_height / current_height

        if smooth:
            # Blend towards 1.0 to avoid sudden jumps
            alpha = self.config.correction_alpha
            scale = alpha * raw_scale + (1 - alpha) * 1.0
            # Clamp to reasonable range only during continuous operation
            scale = np.clip(scale, 0.5, 2.0)
        else:
            # For initialization, use raw scale without clamping
            scale = raw_scale

        return scale

    def update_ground_points_deque(self, new_ground_points: np.ndarray) -> None:
        """
        Add new ground points to the deque.

        Args:
            new_ground_points: (N, 3) array of new ground points
        """
        # Store candidate points for visualization
        self.last_candidate_points = (
            new_ground_points.copy() if len(new_ground_points) > 0 else None
        )

        for pt in new_ground_points:
            self.ground_points_deque.append(pt.copy())

    def get_deque_points(self) -> np.ndarray:
        """Get all points currently in the deque as array."""
        if len(self.ground_points_deque) == 0:
            return np.empty((0, 3))
        return np.array(list(self.ground_points_deque))

    def validate_plane(self, plane: np.ndarray) -> bool:
        """
        Validate that the estimated plane is reasonable (roughly horizontal).

        For KITTI coordinate system:
        - X: right
        - Y: down
        - Z: forward

        Ground plane normal should point mostly up (negative Y direction in world).
        """
        if plane is None:
            return False

        normal = plane[:3]
        normal = normal / np.linalg.norm(normal)

        # The Y component of the normal should be significant
        # In KITTI, if Y points down, ground normal pointing up has negative Y
        # But after fitting, normal direction could be flipped
        # Check absolute value of Y component
        y_component = abs(normal[1])

        return y_component >= self.config.min_plane_normal_y

    def estimate_and_correct(
        self,
        landmarks_3d: np.ndarray,
        keypoints_2d: np.ndarray,
        landmark_indices: np.ndarray,
        road_mask: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        force_correction: bool = False,
    ) -> Tuple[GroundPlaneResult, np.ndarray, np.ndarray]:
        """
        Main method: estimate ground plane and compute scale correction.

        Args:
            landmarks_3d: (M, 3) all landmarks
            keypoints_2d: (N, 2) current frame keypoints
            landmark_indices: (N,) mapping keypoints to landmarks
            road_mask: (H, W) road segmentation mask
            K: (3, 3) intrinsic matrix
            R: (3, 3) rotation (world to camera)
            t: (3, 1) translation (world to camera)
            force_correction: Force scale correction this frame

        Returns:
            result: GroundPlaneResult with plane info and scale factor
            corrected_landmarks: Scaled landmarks (or original if no correction)
            corrected_t: Scaled translation (or original if no correction)
        """
        self.frame_counter += 1
        result = GroundPlaneResult()

        # Filter landmarks to get ground points
        ground_pts, ground_kp_2d, _ = self.filter_ground_landmarks(
            landmarks_3d, keypoints_2d, landmark_indices, road_mask, K, R, t
        )

        # Store for visualization
        self.last_candidate_points = ground_pts.copy() if len(ground_pts) > 0 else None
        self.last_candidate_2d = ground_kp_2d.copy() if len(ground_kp_2d) > 0 else None

        # Add to deque
        if len(ground_pts) > 0:
            self.update_ground_points_deque(ground_pts)

        # Check if we should attempt correction
        should_correct = force_correction or (
            self.frame_counter % self.config.correction_interval == 0
        )

        if not should_correct:
            result.scale_factor = 1.0
            result.skipped = True  # Mark as skipped, not failed
            return result, landmarks_3d, t

        # Get accumulated points from deque
        deque_points = self.get_deque_points()

        # For initialization (force_correction=True), skip the min_ground_points check
        # We want to use whatever points we have at init
        if not force_correction and len(deque_points) < self.config.min_ground_points:
            print(
                f"    [GroundPlane DEBUG] Not enough points in deque: {len(deque_points)} < {self.config.min_ground_points}"
            )
            result.scale_factor = 1.0
            return result, landmarks_3d, t
        
        # Still need at least 3 points to fit a plane
        if len(deque_points) < 3:
            print(f"    [GroundPlane DEBUG] Need at least 3 points to fit plane, have {len(deque_points)}")
            result.scale_factor = 1.0
            return result, landmarks_3d, t

        # RANSAC plane estimation
        # For initialization (force_correction=True), use relaxed min_points of 5
        init_min_points = 5 if force_correction else None
        plane, inlier_mask = self.estimate_ground_plane_ransac(deque_points, min_points=init_min_points)

        # Debug: plane estimation result
        if plane is None:
            min_pts_used = init_min_points if init_min_points is not None else self.config.min_ground_points
            print(
                f"    [GroundPlane DEBUG] RANSAC failed to find plane with >= {min_pts_used} inliers"
            )
        else:
            normal = plane[:3] / np.linalg.norm(plane[:3])
            num_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
            inlier_ratio = num_inliers / len(deque_points)
            print(
                f"    [GroundPlane DEBUG] RANSAC plane normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], "
                f"inliers: {num_inliers}/{len(deque_points)} ({inlier_ratio:.1%}), Y_component: {abs(normal[1]):.3f}"
            )

        # Validate plane: check normal direction and inlier ratio
        if plane is not None:
            num_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
            inlier_ratio = (
                num_inliers / len(deque_points) if len(deque_points) > 0 else 0
            )
            plane_valid = self.validate_plane(plane)
            ratio_valid = inlier_ratio >= self.config.min_inlier_ratio

            if not plane_valid:
                print(
                    f"    [GroundPlane DEBUG] Plane validation FAILED (normal not vertical enough)"
                )
            if not ratio_valid:
                print(
                    f"    [GroundPlane DEBUG] Plane validation FAILED (inlier ratio {inlier_ratio:.1%} < {self.config.min_inlier_ratio:.0%})"
                )

        if (
            plane is None
            or not self.validate_plane(plane)
            or inlier_ratio < self.config.min_inlier_ratio
        ):
            result.scale_factor = 1.0
            return result, landmarks_3d, t

        # Compute camera height
        height = self.get_camera_height_from_plane(R, t, plane)

        # Check if correction is needed
        height_error = abs(height - self.config.camera_height)

        if height_error < self.config.height_tolerance and not force_correction:
            result.plane_params = plane
            result.inlier_mask = inlier_mask
            result.camera_height = height
            result.scale_factor = 1.0
            result.num_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
            result.is_valid = True
            self.current_plane = plane
            return result, landmarks_3d, t

        # Compute and apply scale correction
        scale = self.compute_scale_correction(height, smooth=not force_correction)

        corrected_landmarks = landmarks_3d * scale
        corrected_t = t * scale

        # Update result
        result.plane_params = plane
        result.inlier_mask = inlier_mask
        result.camera_height = height
        result.scale_factor = scale
        result.num_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        result.is_valid = True
        self.current_plane = plane

        # Clear deque after correction to avoid accumulating scaled/unscaled mix
        # Actually, we should scale the deque points too or clear them
        # For simplicity, let's scale them
        if scale != 1.0:
            scaled_deque = [pt * scale for pt in self.ground_points_deque]
            self.ground_points_deque.clear()
            for pt in scaled_deque:
                self.ground_points_deque.append(pt)

        return result, corrected_landmarks, corrected_t

    def initialize_scale(
        self,
        landmarks_3d: np.ndarray,
        keypoints_2d: np.ndarray,
        landmark_indices: np.ndarray,
        road_mask: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Initialize scale at startup based on ground plane.
        Uses direct (non-smoothed) correction.

        Args:
            landmarks_3d: Initial triangulated landmarks
            keypoints_2d: 2D keypoints from bootstrap frame
            landmark_indices: Mapping to landmarks
            road_mask: Road mask for bootstrap frame
            K, R, t: Camera parameters

        Returns:
            scale: The computed scale factor
            scaled_landmarks: Landmarks scaled to correct height
            scaled_t: Translation scaled to correct height
        """
        result, scaled_landmarks, scaled_t = self.estimate_and_correct(
            landmarks_3d,
            keypoints_2d,
            landmark_indices,
            road_mask,
            K,
            R,
            t,
            force_correction=True,
        )

        if result.is_valid:
            print(f"[GroundPlane] Initialization:")
            print(f"  - Estimated height: {result.camera_height:.3f}m")
            print(f"  - Target height: {self.config.camera_height:.3f}m")
            print(f"  - Scale factor: {result.scale_factor:.4f}")
            print(f"  - Inliers: {result.num_inliers}")
        else:
            print("[GroundPlane] Warning: Could not estimate ground plane at init")
            return 1.0, landmarks_3d, t

        return result.scale_factor, scaled_landmarks, scaled_t

    def add_ground_points_to_deque(self, ground_pts: np.ndarray):
        """
        Add filtered ground points to the deque.
        
        Args:
            ground_pts: (N, 3) array of ground point 3D coordinates
        """
        if len(ground_pts) > 0:
            self.update_ground_points_deque(ground_pts)

    def estimate_scale_factor_only(
        self,
        R: np.ndarray,
        t: np.ndarray,
        force_estimation: bool = False,
    ) -> GroundPlaneResult:
        """
        Estimate scale factor from accumulated ground points in deque.
        
        This is a simplified method that only uses the deque - the main loop
        is responsible for filtering and adding ground points separately.
        The scale factor should be applied to relative motion.

        Args:
            R: (3, 3) current camera rotation (world to camera)
            t: (3, 1) current camera translation (world to camera)
            force_estimation: If True, estimate even if not at correction interval

        Returns:
            result: GroundPlaneResult with scale_factor (no landmark modification)
        """
        self.frame_counter += 1
        result = GroundPlaneResult()

        # Check if we should attempt estimation
        should_estimate = force_estimation or (self.frame_counter % self.config.correction_interval == 0)

        if not should_estimate:
            result.scale_factor = 1.0
            result.skipped = True
            return result

        # Get accumulated points from deque
        deque_points = self.get_deque_points()

        if len(deque_points) < self.config.min_ground_points:
            print(
                f"    [GroundPlane DEBUG] Not enough points in deque: {len(deque_points)} < {self.config.min_ground_points}"
            )
            result.scale_factor = 1.0
            return result

        # RANSAC plane estimation
        plane, inlier_mask = self.estimate_ground_plane_ransac(deque_points)

        # Debug output
        if plane is None:
            print(
                f"    [GroundPlane DEBUG] RANSAC failed to find plane with >= {self.config.min_ground_points} inliers"
            )
            result.scale_factor = 1.0
            return result

        normal = plane[:3] / np.linalg.norm(plane[:3])
        num_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        inlier_ratio = num_inliers / len(deque_points) if len(deque_points) > 0 else 0
        
        print(
            f"    [GroundPlane DEBUG] RANSAC plane normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], "
            f"inliers: {num_inliers}/{len(deque_points)} ({inlier_ratio:.1%}), Y_component: {abs(normal[1]):.3f}"
        )

        # Validate plane
        plane_valid = self.validate_plane(plane)
        ratio_valid = inlier_ratio >= self.config.min_inlier_ratio

        if not plane_valid:
            print(f"    [GroundPlane DEBUG] Plane validation FAILED (normal not vertical enough)")
        if not ratio_valid:
            print(f"    [GroundPlane DEBUG] Plane validation FAILED (inlier ratio {inlier_ratio:.1%} < {self.config.min_inlier_ratio:.0%})")

        if not plane_valid or not ratio_valid:
            result.scale_factor = 1.0
            return result

        # Compute camera height from plane using the proper method
        height = self.get_camera_height_from_plane(R, t, plane)
        
        # DEBUG: Show plane and camera info
        C = (-R.T @ t).flatten()
        print(f"    [DEBUG] Plane: a={plane[0]:.3f}, b={plane[1]:.3f}, c={plane[2]:.3f}, d={plane[3]:.3f}")
        print(f"    [DEBUG] Camera center: Y={C[1]:.3f}, computed height={height:.3f}m")
        
        # Compute scale factor (what we would need to correct)
        # scale = target_height / current_height
        if height > 1e-6:
            raw_scale = self.config.camera_height / height
        else:
            raw_scale = 1.0

        # Apply smoothing for gradual correction
        alpha = self.config.correction_alpha
        smoothed_scale = alpha * raw_scale + (1 - alpha) * 1.0
        smoothed_scale = np.clip(smoothed_scale, 0.8, 1.2)  # Limit per-frame correction

        # Fill result
        result.plane_params = plane
        result.inlier_mask = inlier_mask
        result.camera_height = height
        result.scale_factor = smoothed_scale
        result.num_inliers = num_inliers
        result.is_valid = True
        self.current_plane = plane

        return result


def load_road_mask(mask_dir: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load road mask for a specific frame.

    Args:
        mask_dir: Directory containing mask files
        frame_idx: Frame index

    Returns:
        mask: (H, W) binary mask or None if not found
    """
    import os

    mask_path = os.path.join(mask_dir, f"{frame_idx:06d}_mask.npy")

    if not os.path.exists(mask_path):
        return None

    mask = np.load(mask_path)

    # Handle different mask shapes:
    # - (1, 1, H, W) -> squeeze to (H, W)
    # - (N, H, W) -> multiple roads detected, merge with logical OR
    # - (1, H, W) -> squeeze to (H, W)
    # - (H, W) -> use as is

    if mask.ndim == 4:
        # Shape like (1, 1, H, W) or (1, N, H, W)
        mask = np.squeeze(mask)

    if mask.ndim == 3:
        # Shape like (N, H, W) - multiple road masks, merge them
        # Use logical OR to combine all detected roads
        mask = np.any(mask, axis=0)

    # Ensure it's 2D now
    if mask.ndim != 2:
        print(f"Warning: Road mask has unexpected shape {mask.shape} after processing")
        return None

    # Convert to uint8 if boolean
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    return mask
