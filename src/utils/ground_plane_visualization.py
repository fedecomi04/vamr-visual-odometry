# ground_plane_visualization.py
"""
Visualization utilities for ground plane scale correction.

Visualizes:
- Points in the deque (accumulated ground points) - BLUE filled circles
- Points that might be added to deque (current candidates) - GREEN hollow circles
- Ground plane (optional 3D visualization)
"""

from typing import Optional, Tuple, List

import cv2
import numpy as np


class GroundPlaneVisualizer:
    """
    Visualizes ground plane estimation on images.

    Color scheme:
    - BLUE (filled): Points already in the deque (accumulated history)
    - GREEN (hollow circle): New candidate points from current frame
    - RED (optional): Outliers rejected by RANSAC
    """

    # Colors in BGR
    COLOR_DEQUE = (255, 100, 0)  # Blue - points in deque
    COLOR_CANDIDATE = (0, 255, 0)  # Green - candidate points to add
    COLOR_OUTLIER = (0, 0, 255)  # Red - RANSAC outliers
    COLOR_PLANE_LINE = (255, 255, 0)  # Cyan - ground plane intersection

    def __init__(
        self,
        deque_point_radius: int = 3,
        candidate_point_radius: int = 5,
        candidate_thickness: int = 2,
    ):
        """
        Args:
            deque_point_radius: Radius for deque point circles
            candidate_point_radius: Radius for candidate point circles
            candidate_thickness: Line thickness for hollow candidate circles
        """
        self.deque_radius = deque_point_radius
        self.candidate_radius = candidate_point_radius
        self.candidate_thickness = candidate_thickness

    def project_points_to_image(
        self,
        points_3d: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to image coordinates.

        Args:
            points_3d: (N, 3) array of 3D points in world frame
            K: (3, 3) camera intrinsic matrix
            R: (3, 3) rotation matrix (world to camera)
            t: (3, 1) translation vector
            image_shape: (H, W) image dimensions

        Returns:
            points_2d: (M, 2) array of valid projected points
            valid_mask: (N,) boolean mask of points that project into image
        """
        if len(points_3d) == 0:
            return np.empty((0, 2)), np.array([], dtype=bool)

        # Transform to camera frame
        X_cam = (R @ points_3d.T + t).T  # (N, 3)

        # Filter points behind camera
        in_front = X_cam[:, 2] > 0.1

        if not np.any(in_front):
            return np.empty((0, 2)), np.zeros(len(points_3d), dtype=bool)

        X_cam_valid = X_cam[in_front]

        # Project to image
        x_norm = X_cam_valid[:, 0] / X_cam_valid[:, 2]
        y_norm = X_cam_valid[:, 1] / X_cam_valid[:, 2]

        u = K[0, 0] * x_norm + K[0, 2]
        v = K[1, 1] * y_norm + K[1, 2]

        points_2d_all = np.column_stack([u, v])

        # Filter points inside image bounds
        H, W = image_shape
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # Create full valid mask
        valid_mask = np.zeros(len(points_3d), dtype=bool)
        in_front_indices = np.where(in_front)[0]
        valid_mask[in_front_indices[in_bounds]] = True

        return points_2d_all[in_bounds], valid_mask

    def visualize_ground_points(
        self,
        image: np.ndarray,
        deque_points_3d: np.ndarray,
        candidate_points_3d: Optional[np.ndarray],
        candidate_points_2d: Optional[np.ndarray],
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        inlier_mask: Optional[np.ndarray] = None,
        show_outliers: bool = False,
    ) -> np.ndarray:
        """
        Visualize ground plane points on image.

        Args:
            image: Input image (grayscale or BGR)
            deque_points_3d: (N, 3) accumulated ground points from deque
            candidate_points_3d: (M, 3) new candidate ground points (optional)
            candidate_points_2d: (M, 2) 2D positions of candidates (if available)
            K: Camera intrinsic matrix
            R: Rotation matrix (world to camera)
            t: Translation vector
            inlier_mask: Boolean mask for deque points (True = inlier)
            show_outliers: Whether to show RANSAC outliers in red

        Returns:
            vis: Visualization image with ground points drawn
        """
        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        H, W = vis.shape[:2]

        # 1. Draw deque points (BLUE filled circles)
        if len(deque_points_3d) > 0:
            deque_2d, valid_mask = self.project_points_to_image(
                deque_points_3d, K, R, t, (H, W)
            )

            # If we have inlier mask, use it
            if inlier_mask is not None and show_outliers:
                # Get inliers and outliers
                valid_indices = np.where(valid_mask)[0]

                for i, pt_2d in enumerate(deque_2d):
                    pt = (int(round(pt_2d[0])), int(round(pt_2d[1])))

                    # Find corresponding original index
                    orig_idx = valid_indices[i] if i < len(valid_indices) else 0

                    if orig_idx < len(inlier_mask) and inlier_mask[orig_idx]:
                        # Inlier - blue
                        cv2.circle(vis, pt, self.deque_radius, self.COLOR_DEQUE, -1)
                    else:
                        # Outlier - red
                        cv2.circle(vis, pt, self.deque_radius, self.COLOR_OUTLIER, -1)
            else:
                # All deque points in blue
                for pt_2d in deque_2d:
                    pt = (int(round(pt_2d[0])), int(round(pt_2d[1])))
                    cv2.circle(vis, pt, self.deque_radius, self.COLOR_DEQUE, -1)

        # 2. Draw candidate points (GREEN hollow circles)
        if candidate_points_2d is not None and len(candidate_points_2d) > 0:
            # Use provided 2D coordinates directly
            for pt_2d in candidate_points_2d:
                pt = (int(round(pt_2d[0])), int(round(pt_2d[1])))
                if 0 <= pt[0] < W and 0 <= pt[1] < H:
                    cv2.circle(
                        vis,
                        pt,
                        self.candidate_radius,
                        self.COLOR_CANDIDATE,
                        self.candidate_thickness,
                    )
        elif candidate_points_3d is not None and len(candidate_points_3d) > 0:
            # Project 3D candidates to 2D
            cand_2d, _ = self.project_points_to_image(
                candidate_points_3d, K, R, t, (H, W)
            )
            for pt_2d in cand_2d:
                pt = (int(round(pt_2d[0])), int(round(pt_2d[1])))
                cv2.circle(
                    vis,
                    pt,
                    self.candidate_radius,
                    self.COLOR_CANDIDATE,
                    self.candidate_thickness,
                )

        return vis

    def visualize_ground_plane_on_image(
        self,
        image: np.ndarray,
        plane_params: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        grid_extent: float = 50.0,
        grid_step: float = 5.0,
    ) -> np.ndarray:
        """
        Visualize the estimated ground plane as a grid projected onto the image.

        Args:
            image: Input image
            plane_params: (4,) plane parameters [a, b, c, d]
            K: Camera intrinsic matrix
            R: Rotation matrix
            t: Translation vector
            grid_extent: Size of grid to draw (meters)
            grid_step: Grid spacing (meters)

        Returns:
            vis: Image with ground plane grid overlay
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        if plane_params is None:
            return vis

        # Camera center in world
        C = (-R.T @ t).flatten()

        # Generate grid points on the plane
        # Plane: ax + by + cz + d = 0 => y = -(ax + cz + d) / b (assuming b != 0)
        a, b, c, d = plane_params

        if abs(b) < 1e-6:
            return vis  # Can't solve for Y

        H, W = vis.shape[:2]

        # Create grid around camera position
        x_range = np.arange(C[0] - grid_extent, C[0] + grid_extent, grid_step)
        z_range = np.arange(C[2] - grid_extent, C[2] + grid_extent, grid_step)

        # Draw grid lines
        for x in x_range:
            line_pts_3d = []
            for z in z_range:
                y = -(a * x + c * z + d) / b
                line_pts_3d.append([x, y, z])

            line_pts_3d = np.array(line_pts_3d)
            pts_2d, _ = self.project_points_to_image(line_pts_3d, K, R, t, (H, W))

            if len(pts_2d) >= 2:
                pts_int = pts_2d.astype(np.int32)
                cv2.polylines(vis, [pts_int], False, self.COLOR_PLANE_LINE, 1)

        for z in z_range:
            line_pts_3d = []
            for x in x_range:
                y = -(a * x + c * z + d) / b
                line_pts_3d.append([x, y, z])

            line_pts_3d = np.array(line_pts_3d)
            pts_2d, _ = self.project_points_to_image(line_pts_3d, K, R, t, (H, W))

            if len(pts_2d) >= 2:
                pts_int = pts_2d.astype(np.int32)
                cv2.polylines(vis, [pts_int], False, self.COLOR_PLANE_LINE, 1)

        return vis

    def create_legend(
        self, image: np.ndarray, position: str = "top-right"
    ) -> np.ndarray:
        """
        Add a legend to the visualization image.

        Args:
            image: Input image
            position: Legend position ('top-right', 'top-left', 'bottom-right', 'bottom-left')

        Returns:
            vis: Image with legend
        """
        vis = image.copy()
        H, W = vis.shape[:2]

        # Legend items
        items = [
            ("Deque (history)", self.COLOR_DEQUE, True),
            ("Candidates (new)", self.COLOR_CANDIDATE, False),
        ]

        # Legend dimensions
        legend_h = 25 * len(items) + 10
        legend_w = 150
        margin = 10

        # Position
        if position == "top-right":
            x0, y0 = W - legend_w - margin, margin
        elif position == "top-left":
            x0, y0 = margin, margin
        elif position == "bottom-right":
            x0, y0 = W - legend_w - margin, H - legend_h - margin
        else:
            x0, y0 = margin, H - legend_h - margin

        # Draw background
        cv2.rectangle(
            vis, (x0, y0), (x0 + legend_w, y0 + legend_h), (255, 255, 255), -1
        )
        cv2.rectangle(vis, (x0, y0), (x0 + legend_w, y0 + legend_h), (0, 0, 0), 1)

        # Draw items
        for i, (label, color, filled) in enumerate(items):
            cy = y0 + 20 + i * 25
            cx = x0 + 15

            if filled:
                cv2.circle(vis, (cx, cy), 5, color, -1)
            else:
                cv2.circle(vis, (cx, cy), 5, color, 2)

            cv2.putText(
                vis,
                label,
                (cx + 15, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        return vis


def visualize_ground_scale_info(
    image: np.ndarray,
    ground_estimator,  # GroundPlaneScaleEstimator
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    result=None,  # GroundPlaneResult
    show_legend: bool = True,
    show_plane_grid: bool = False,
) -> np.ndarray:
    """
    Convenience function to visualize ground plane estimation results.

    Args:
        image: Input image
        ground_estimator: GroundPlaneScaleEstimator instance
        K: Camera intrinsic matrix
        R: Rotation matrix
        t: Translation vector
        result: Optional GroundPlaneResult from estimation
        show_legend: Whether to add legend
        show_plane_grid: Whether to show ground plane grid

    Returns:
        vis: Visualization image
    """
    visualizer = GroundPlaneVisualizer()

    # Get points from estimator
    deque_points = ground_estimator.get_deque_points()
    candidate_points_3d = ground_estimator.last_candidate_points
    candidate_points_2d = ground_estimator.last_candidate_2d

    # Get inlier mask if available
    inlier_mask = result.inlier_mask if result is not None else None

    # Visualize points
    vis = visualizer.visualize_ground_points(
        image=image,
        deque_points_3d=deque_points,
        candidate_points_3d=candidate_points_3d,
        candidate_points_2d=candidate_points_2d,
        K=K,
        R=R,
        t=t,
        inlier_mask=inlier_mask,
        show_outliers=True,
    )

    # Optionally show ground plane grid
    if show_plane_grid and result is not None and result.plane_params is not None:
        vis = visualizer.visualize_ground_plane_on_image(
            vis, result.plane_params, K, R, t
        )

    # Add legend
    if show_legend:
        vis = visualizer.create_legend(vis)

    # Add text info
    if result is not None and result.is_valid:
        info_text = f"H: {result.camera_height:.2f}m | Scale: {result.scale_factor:.3f} | Inliers: {result.num_inliers}"
        cv2.putText(
            vis,
            info_text,
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    return vis
