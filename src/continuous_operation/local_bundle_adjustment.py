from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass
class BAObservation:
    """A 2D observation of a landmark in a specific frame."""

    frame_idx: int
    landmark_idx: int
    keypoint: np.ndarray  # (2,) pixel coordinates


@dataclass
class BAFrame:
    """Camera pose for a frame in the BA window."""

    R: np.ndarray  # (3, 3) rotation matrix
    t: np.ndarray  # (3, 1) translation vector


class LocalBundleAdjustment:
    """
    Sliding window bundle adjustment for visual odometry.

    Maintains a window of recent frames and their observations,
    and periodically optimizes poses and landmarks jointly.
    """

    def __init__(
        self,
        K: np.ndarray,
        window_size: int = 10,
        run_every_n_frames: int = 5,
        max_iterations: int = 10,
        ftol: float = 1e-4,
        fix_first_pose: bool = True,
    ):
        """
        Initialize the Local Bundle Adjustment module.

        Args:
            K: Camera intrinsic matrix (3x3)
            window_size: Number of recent frames to keep in the optimization window
            run_every_n_frames: Run BA every N frames
            max_iterations: Maximum iterations for the optimizer
            ftol: Function tolerance for convergence
            fix_first_pose: Whether to fix the oldest pose in the window (gauge fixing)
        """
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

        self.window_size = window_size
        self.run_every_n_frames = run_every_n_frames
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.fix_first_pose = fix_first_pose

        # Sliding window data
        self.frame_poses: deque = deque(maxlen=window_size)  # List of BAFrame
        self.frame_indices: deque = deque(maxlen=window_size)  # Global frame indices
        self.observations: List[BAObservation] = []  # All observations in window

        # Mapping from global landmark idx to local idx in optimization
        self.landmark_to_local: Dict[int, int] = {}
        self.local_to_landmark: Dict[int, int] = {}

        self.frame_counter = 0
        self._last_ba_frame = -1

    def add_frame(
        self,
        frame_idx: int,
        R: np.ndarray,
        t: np.ndarray,
        keypoints: np.ndarray,
        landmark_indices: np.ndarray,
    ):
        """
        Add a new frame to the BA window.

        Args:
            frame_idx: Global frame index
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            keypoints: 2D keypoints (Nx2)
            landmark_indices: Landmark IDs for each keypoint (-1 if not a landmark)
        """
        # Add frame pose
        self.frame_poses.append(BAFrame(R=R.copy(), t=t.copy()))
        self.frame_indices.append(frame_idx)

        # Add observations
        for i, lm_idx in enumerate(landmark_indices):
            if lm_idx != -1:
                self.observations.append(
                    BAObservation(
                        frame_idx=frame_idx,
                        landmark_idx=int(lm_idx),
                        keypoint=keypoints[i].copy(),
                    )
                )

        # Clean up old observations (not in current window)
        if len(self.frame_indices) > 0:
            min_frame = min(self.frame_indices)
            self.observations = [
                obs for obs in self.observations if obs.frame_idx >= min_frame
            ]

        self.frame_counter += 1

    def reset(self) -> None:
        """Clear the sliding window state (used after a global world reset)."""
        self.frame_poses.clear()
        self.frame_indices.clear()
        self.observations = []
        self.landmark_to_local.clear()
        self.local_to_landmark.clear()
        self.frame_counter = 0
        self._last_ba_frame = -1

    def should_run(self) -> bool:
        """Check if BA should run this frame."""
        if len(self.frame_poses) < 3:
            return False
        return (self.frame_counter - self._last_ba_frame) >= self.run_every_n_frames

    def run(
        self,
        landmarks_3d: np.ndarray,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Run local bundle adjustment.

        Args:
            landmarks_3d: All 3D landmarks (Mx3)

        Returns:
            optimized_poses: List of (R, t) tuples for frames in window
            optimized_landmarks: Updated landmarks_3d array
        """
        if len(self.frame_poses) < 2:
            return [(f.R, f.t) for f in self.frame_poses], landmarks_3d

        self._last_ba_frame = self.frame_counter

        # Build the optimization problem
        # 1. Collect landmarks that have observations in the window
        observed_landmarks = set()
        for obs in self.observations:
            if 0 <= obs.landmark_idx < len(landmarks_3d):
                observed_landmarks.add(obs.landmark_idx)

        if len(observed_landmarks) < 4:
            return [(f.R, f.t) for f in self.frame_poses], landmarks_3d

        # Create local<->global landmark mapping
        self.landmark_to_local = {
            lm: i for i, lm in enumerate(sorted(observed_landmarks))
        }
        self.local_to_landmark = {i: lm for lm, i in self.landmark_to_local.items()}

        n_frames = len(self.frame_poses)
        n_landmarks = len(observed_landmarks)

        # Filter observations to only include valid landmarks
        valid_observations = [
            obs
            for obs in self.observations
            if obs.landmark_idx in self.landmark_to_local
        ]

        if len(valid_observations) < 10:
            return [(f.R, f.t) for f in self.frame_poses], landmarks_3d

        # Build initial parameter vector
        # Format: [rvec1, tvec1, rvec2, tvec2, ..., point1, point2, ...]
        # Note: if fix_first_pose, first pose is not optimized

        n_pose_params = 6 * (n_frames - 1) if self.fix_first_pose else 6 * n_frames
        n_point_params = 3 * n_landmarks

        x0 = np.zeros(n_pose_params + n_point_params)

        # Fill in pose parameters (rotation as Rodrigues vector, then translation)
        for i, frame in enumerate(self.frame_poses):
            if self.fix_first_pose and i == 0:
                continue
            idx = i - 1 if self.fix_first_pose else i
            rvec, _ = cv2.Rodrigues(frame.R)
            x0[idx * 6 : idx * 6 + 3] = rvec.flatten()
            x0[idx * 6 + 3 : idx * 6 + 6] = frame.t.flatten()

        # Fill in landmark positions
        point_start = n_pose_params
        for local_idx in range(n_landmarks):
            global_idx = self.local_to_landmark[local_idx]
            x0[point_start + local_idx * 3 : point_start + local_idx * 3 + 3] = (
                landmarks_3d[global_idx]
            )

        # Build frame index mapping (global -> local in window)
        frame_to_local = {f_idx: i for i, f_idx in enumerate(self.frame_indices)}

        # Pre-build observation index arrays for vectorized computation
        n_obs = len(valid_observations)
        obs_frame_indices = np.zeros(n_obs, dtype=np.int32)
        obs_landmark_indices = np.zeros(n_obs, dtype=np.int32)
        obs_keypoints = np.zeros((n_obs, 2), dtype=np.float64)

        for i, obs in enumerate(valid_observations):
            obs_frame_indices[i] = frame_to_local[obs.frame_idx]
            obs_landmark_indices[i] = self.landmark_to_local[obs.landmark_idx]
            obs_keypoints[i] = obs.keypoint

        # Cache fixed pose if applicable
        if self.fix_first_pose:
            fixed_R = self.frame_poses[0].R
            fixed_t = self.frame_poses[0].t.flatten()

        # Define vectorized residual function
        def residuals(x):
            # Extract poses - build rotation matrices
            Rs = np.zeros((n_frames, 3, 3), dtype=np.float64)
            ts = np.zeros((n_frames, 3), dtype=np.float64)

            for i in range(n_frames):
                if self.fix_first_pose and i == 0:
                    Rs[0] = fixed_R
                    ts[0] = fixed_t
                else:
                    idx = i - 1 if self.fix_first_pose else i
                    rvec = x[idx * 6 : idx * 6 + 3]
                    tvec = x[idx * 6 + 3 : idx * 6 + 6]
                    Rs[i] = _rodrigues_to_matrix(rvec)
                    ts[i] = tvec

            # Extract all 3D points
            points = x[point_start:].reshape(-1, 3)

            # Gather data for all observations using advanced indexing
            R_obs = Rs[obs_frame_indices]  # (n_obs, 3, 3)
            t_obs = ts[obs_frame_indices]  # (n_obs, 3)
            points_obs = points[obs_landmark_indices]  # (n_obs, 3)

            # Vectorized projection: point_cam = R @ point + t
            # Using einsum for batch matrix-vector multiplication
            points_cam = np.einsum("ijk,ik->ij", R_obs, points_obs) + t_obs  # (n_obs, 3)

            # Handle points behind camera
            z = points_cam[:, 2]
            valid_mask = z > 0.1

            # Project to image plane (vectorized)
            z_safe = np.where(valid_mask, z, 1.0)  # Avoid division by zero
            x_proj = self.fx * points_cam[:, 0] / z_safe + self.cx
            y_proj = self.fy * points_cam[:, 1] / z_safe + self.cy

            # Compute reprojection errors
            err_x = x_proj - obs_keypoints[:, 0]
            err_y = y_proj - obs_keypoints[:, 1]

            # Apply large penalty for invalid points (behind camera)
            err_x = np.where(valid_mask, err_x, 100.0)
            err_y = np.where(valid_mask, err_y, 100.0)

            # Interleave x and y errors into output array
            residuals_out = np.empty(n_obs * 2, dtype=np.float64)
            residuals_out[0::2] = err_x
            residuals_out[1::2] = err_y

            return residuals_out

        # Build sparsity pattern for Jacobian
        n_residuals = len(valid_observations) * 2
        sparsity = lil_matrix((n_residuals, len(x0)), dtype=int)

        for obs_idx, obs in enumerate(valid_observations):
            if obs.frame_idx not in frame_to_local:
                continue

            local_frame_idx = frame_to_local[obs.frame_idx]
            local_lm_idx = self.landmark_to_local[obs.landmark_idx]

            row = obs_idx * 2

            # Pose parameters (if not fixed)
            if not (self.fix_first_pose and local_frame_idx == 0):
                pose_idx = (
                    local_frame_idx - 1 if self.fix_first_pose else local_frame_idx
                )
                sparsity[row : row + 2, pose_idx * 6 : (pose_idx + 1) * 6] = 1

            # Point parameters
            point_idx = point_start + local_lm_idx * 3
            sparsity[row : row + 2, point_idx : point_idx + 3] = 1

        # Run optimization
        try:
            result = least_squares(
                residuals,
                x0,
                jac_sparsity=sparsity,
                verbose=0,
                x_scale="jac",
                ftol=self.ftol,
                max_nfev=self.max_iterations * len(x0),
                method="trf",
            )
            x_opt = result.x

            initial_cost = np.sum(residuals(x0) ** 2)
            print(
                f"    [BA] Optimized {n_frames} frames, {n_landmarks} landmarks. "
                f"Cost: {initial_cost:.1f} -> {result.cost:.1f}"
            )

        except Exception as e:
            print(f"    [BA] Optimization failed: {e}")
            return [(f.R, f.t) for f in self.frame_poses], landmarks_3d

        # Extract optimized poses
        optimized_poses = []
        for i in range(n_frames):
            if self.fix_first_pose and i == 0:
                optimized_poses.append(
                    (self.frame_poses[0].R.copy(), self.frame_poses[0].t.copy())
                )
            else:
                idx = i - 1 if self.fix_first_pose else i
                rvec = x_opt[idx * 6 : idx * 6 + 3]
                tvec = x_opt[idx * 6 + 3 : idx * 6 + 6].reshape(3, 1)
                R = _rodrigues_to_matrix(rvec)
                optimized_poses.append((R, tvec))

        # Update frame poses in the window
        for i, (R, t) in enumerate(optimized_poses):
            self.frame_poses[i].R = R
            self.frame_poses[i].t = t

        # Extract optimized landmarks
        optimized_landmarks = landmarks_3d.copy()
        points_opt = x_opt[point_start:].reshape(-1, 3)
        for local_idx in range(n_landmarks):
            global_idx = self.local_to_landmark[local_idx]
            optimized_landmarks[global_idx] = points_opt[local_idx]

        return optimized_poses, optimized_landmarks

    def get_latest_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the most recent optimized pose."""
        if len(self.frame_poses) == 0:
            return None
        return (self.frame_poses[-1].R.copy(), self.frame_poses[-1].t.copy())


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues vector to rotation matrix."""
    rvec = np.asarray(rvec).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R
