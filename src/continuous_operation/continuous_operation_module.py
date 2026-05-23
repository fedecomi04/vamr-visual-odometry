import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


class ContinuousOperationModule:
    """
    Base class for continuous VO operation modules.
    """

    def __init__(
        self,
        intrinsics,
        ransac_thresh=1.0,
        max_iters=2000,
        parallax_thresh_deg=2.0,
        iterative_triangulation=False,
    ):
        self.intrinsics = intrinsics
        self.ransac_thresh = ransac_thresh
        self.max_iters = max_iters
        self.parallax_thresh = np.deg2rad(parallax_thresh_deg)
        self.iterative_triangulation = iterative_triangulation

    def _estimate_relative_pose(self, p_prev, p_curr):
        """Estimate relative pose using essential matrix with RANSAC (Fallback)."""
        if len(p_curr) < 5:
            return np.eye(3), np.zeros((3, 1)), np.zeros(len(p_curr), dtype=bool)

        E, inlier_mask = cv.findEssentialMat(
            p_prev,
            p_curr,
            cameraMatrix=self.intrinsics.K,
            method=cv.RANSAC,
            prob=0.999,
            threshold=self.ransac_thresh,
            maxIters=self.max_iters,
        )

        if E is None or E.shape != (3, 3):
            return np.eye(3), np.zeros((3, 1)), np.zeros(len(p_curr), dtype=bool)

        _, R, t, inlier_mask_pose = cv.recoverPose(
            E, p_prev, p_curr, cameraMatrix=self.intrinsics.K, mask=inlier_mask
        )
        return R, t, inlier_mask_pose.astype(bool).reshape(-1)

    def _estimate_pose_pnp(
        self, points_3d, points_2d, rvec_guess=None, tvec_guess=None
    ):
        """
        Estimate absolute pose using PnP (3D-2D) with RANSAC.

        Args:
            points_3d: Nx3 array of world coordinates.
            points_2d: Nx2 array of image coordinates.
            rvec_guess, tvec_guess: Optional initial guesses for the solver.

        Returns:
            R: 3x3 Rotation matrix (World -> Camera).
            t: 3x1 Translation vector (World -> Camera).
            inliers: Boolean mask of inliers.
            success: Boolean flag indicating if PnP solved successfully.
        """
        # PnP requires at least 4 points (EPnP/P3P)
        if len(points_2d) < 4:
            print(f"[PnP Error] Not enough points: {len(points_2d)} provided.")
            return (
                np.eye(3),
                np.zeros((3, 1)),
                np.zeros(len(points_2d), dtype=bool),
                False,
            )

        # Convert to contiguous arrays of correct type for OpenCV
        points_3d = np.ascontiguousarray(points_3d).astype(np.float32)
        points_2d = np.ascontiguousarray(points_2d).astype(np.float32)
        K = self.intrinsics.K.astype(np.float32)

        # Use iterations count from init
        flags = cv.SOLVEPNP_ITERATIVE
        use_extrinsic_guess = False

        if rvec_guess is not None and tvec_guess is not None:
            use_extrinsic_guess = True

        success, rvec, tvec, inliers_idx = cv.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            None,
            rvec=rvec_guess,
            tvec=tvec_guess,
            useExtrinsicGuess=use_extrinsic_guess,
            iterationsCount=self.max_iters,
            reprojectionError=self.ransac_thresh,
            flags=flags,
        )

        if not success or inliers_idx is None:
            return (
                np.eye(3),
                np.zeros((3, 1)),
                np.zeros(len(points_2d), dtype=bool),
                False,
            )

        # Convert rvec to Rotation Matrix
        R, _ = cv.Rodrigues(rvec)

        # Create boolean mask
        inlier_mask = np.zeros(len(points_2d), dtype=bool)
        inlier_mask[inliers_idx.flatten()] = True

        return R, tvec, inlier_mask, True

    def _compute_candidate_parallax(self, first_obs, curr_pt, curr_R):
        """Compute the angle between the ray at first observation and current ray."""
        uv_first = np.array(first_obs["first_keypoint"])
        R_first = first_obs["first_R"]

        ray_first = self._pixel2ray(uv_first)
        ray_curr = self._pixel2ray(np.array(curr_pt))

        ray_first_world = R_first.T @ ray_first
        ray_curr_world = curr_R.T @ ray_curr

        dot_prod = np.clip(np.dot(ray_first_world.T, ray_curr_world), -1.0, 1.0)
        return float(np.arccos(dot_prod))

    def _triangulate_single_point(self, first_obs, curr_pt, curr_R, curr_t):
        """Triangulate a single point using the first observation and current observation.

        Uses either DLT (Direct Linear Transform) or iterative refinement via
        scipy.optimize.least_squares (Levenberg-Marquardt) based on the
        `iterative_triangulation` flag.
        """
        K = self.intrinsics.K
        P_first = K @ np.hstack((first_obs["first_R"], first_obs["first_t"]))
        P_curr = K @ np.hstack((curr_R, curr_t))

        pts_first = (
            np.array(first_obs["first_keypoint"]).reshape(2, 1).astype(np.float32)
        )
        pts_curr = np.array(curr_pt).reshape(2, 1).astype(np.float32)

        # Initial DLT triangulation
        X_hom = cv.triangulatePoints(P_first, P_curr, pts_first, pts_curr)
        X = (X_hom[:3] / X_hom[3]).flatten()

        if self.iterative_triangulation:
            X = self._refine_triangulation_lm(
                X, pts_first.flatten(), pts_curr.flatten(), P_first, P_curr
            )

        return X

    def _refine_triangulation_lm(self, X_init, pt_first, pt_curr, P_first, P_curr):
        """Refine 3D point by minimizing reprojection error using Levenberg-Marquardt.

        Uses scipy.optimize.least_squares with method='lm'.

        Args:
            X_init: Initial 3D point estimate (3,)
            pt_first: 2D observation in first frame (2,)
            pt_curr: 2D observation in current frame (2,)
            P_first: Projection matrix for first frame (3x4)
            P_curr: Projection matrix for current frame (3x4)

        Returns:
            Refined 3D point (3,)
        """

        def reprojection_residual(X):
            """Compute reprojection error for both views."""
            X_hom = np.append(X, 1.0)

            # Project to first camera
            proj_first = P_first @ X_hom
            u1 = proj_first[0] / proj_first[2]
            v1 = proj_first[1] / proj_first[2]

            # Project to second camera
            proj_curr = P_curr @ X_hom
            u2 = proj_curr[0] / proj_curr[2]
            v2 = proj_curr[1] / proj_curr[2]

            # Residuals: predicted - observed
            return np.array(
                [u1 - pt_first[0], v1 - pt_first[1], u2 - pt_curr[0], v2 - pt_curr[1]]
            )

        # Run Levenberg-Marquardt optimization
        result = least_squares(
            reprojection_residual,
            X_init,
            method="lm",  # Levenberg-Marquardt
            ftol=1e-6,
            xtol=1e-6,
            max_nfev=100,
        )

        return result.x

    def _pixel2ray(self, uv):
        """Convert pixel coordinates to normalized ray."""
        uv_hom = np.array([uv[0], uv[1], 1.0])
        ray = np.linalg.inv(self.intrinsics.K) @ uv_hom
        return ray / np.linalg.norm(ray)
