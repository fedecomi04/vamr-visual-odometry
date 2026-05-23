# structure.py
# data structures used in continuous operation

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Import shared CameraIntrinsics from initialization module
from initialization.structures import CameraIntrinsics

# Re-export for backward compatibility
__all__ = ["CameraIntrinsics", "VOState"]


@dataclass
class VOState:
    """State object for Visual Odometry pipeline."""

    # Image from this frame
    image: np.ndarray

    # Current Pose (R and t from world to camera)
    R: np.ndarray  # Shape: (3, 3)
    t: np.ndarray  # Shape: (3, 1)

    # Features
    keypoints: np.ndarray  # Shape: (N, 2) - keypoints in this frame
    landmarks_3d: np.ndarray  # Shape: (M, 3) - 3D landmarks in world frame
    landmarks_tracks: List[dict]  # Track info for each landmark
    landmark_indices: np.ndarray  # Shape: (N,) - maps keypoints to landmark IDs
    candidates: List[dict]  # Candidate keypoints not yet triangulated

    # Trajectory and statistics
    trajectory: List[np.ndarray] = field(
        default_factory=list
    )  # Camera centers in world
    landmark_counts: List[int] = field(
        default_factory=list
    )  # Active landmark count per frame
    axis_limits: Optional[Tuple[float, float, float, float]] = None  # UI axis limits

    # Optional: descriptors for SIFT-based methods
    descriptors: Optional[np.ndarray] = None

    # Ground plane scale correction (optional)
    ground_plane_estimator: Optional[object] = (
        None  # GroundPlaneScaleEstimator instance
    )
    last_ground_result: Optional[object] = (
        None  # GroundPlaneResult from last estimation
    )

    # Loop-closing integration helpers (frontend rebasing)
    world_reset_pending: bool = False
    world_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    world_s: float = 1.0
