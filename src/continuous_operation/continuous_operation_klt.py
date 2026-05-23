import cv2
import numpy as np

from .continuous_operation_module import ContinuousOperationModule
from .local_bundle_adjustment import LocalBundleAdjustment
from .structure import VOState


class ContinuousOperationKLT(ContinuousOperationModule):
    """
    Continuous VO using PnP with Velocity-Aided Fallback.
    Prevents 'Scale Collapse' by enforcing physical motion consistency.
    """

    def __init__(
        self,
        intrinsics,
        ransac_thresh=1.0,  # Tight threshold for high precision
        max_iters=2000,
        target_features=2000,
        parallax_thresh_deg=1.0,  # Lowered to 1.0 to triangulate sooner during turns
        min_distance=10,
        min_inliers_pnp=15,  # Increased: Don't trust PnP with fewer than 15 points
        enable_ba=True,  # Enable/disable bundle adjustment
        ba_window_size=10,  # Number of keyframes in BA window
        ba_every_n_frames=5,  # Run BA every N frames
        iterative_triangulation=False,  # Use LM refinement for triangulation
    ):
        super().__init__(
            intrinsics,
            ransac_thresh,
            max_iters,
            parallax_thresh_deg,
            iterative_triangulation,
        )
        self.target_features = target_features
        self.min_distance = min_distance
        self.min_inliers_pnp = min_inliers_pnp

        # Physics Memory
        self.last_velocity = 1.0  # Default startup speed (meters/frame approx)
        self.min_velocity_threshold = (
            0.1  # If PnP says we moved < 10cm, suspect failure
        )

        # Bundle Adjustment
        self.enable_ba = enable_ba
        self.frame_counter = 0
        self.ba_module = None
        if enable_ba:
            self.ba_module = LocalBundleAdjustment(
                K=intrinsics.K,
                window_size=ba_window_size,
                run_every_n_frames=ba_every_n_frames,
                max_iterations=10,
                ftol=1e-4,
                fix_first_pose=True,
            )
        
        # Ground plane scale correction factor (applied to relative motion)
        self.ground_scale_factor = 1.0

    def set_ground_scale_factor(self, scale_factor: float):
        """
        Set the scale correction factor from ground plane estimation.
        This will be applied to the relative translation in the next frame.
        """
        self.ground_scale_factor = scale_factor

    def process_frame(self, curr_img, prev_state: VOState, ground_scale_factor: float = None) -> VOState:
        """
        Process a frame with optional ground plane scale correction.
        
        Args:
            curr_img: Current frame image
            prev_state: Previous VO state
            ground_scale_factor: Scale factor to apply to relative motion (default: use internal value)
        """
        # Use provided scale factor or internal one
        if ground_scale_factor is not None:
            scale_factor = ground_scale_factor
        else:
            scale_factor = self.ground_scale_factor

        # After a global world correction, drop any motion priors / local optimizers
        # so the frontend continues from the corrected pose without snapping back.
        if bool(getattr(prev_state, "world_reset_pending", False)):
            self.last_velocity = 1.0
            if self.enable_ba and self.ba_module is not None:
                try:
                    self.ba_module.reset()
                except Exception:
                    pass
            
        # --- 0. Setup & Shape Definitions ---
        prev_img = prev_state.image
        h, w = curr_img.shape

        p0 = prev_state.keypoints
        candidates = prev_state.candidates
        all_landmarks = prev_state.landmarks_3d
        landmark_indices = prev_state.landmark_indices

        # --- Stationary Guard (Physics Constraint) ---
        # Only lock if stationary AND we have healthy tracks.
        # If tracks are empty (0), we MUST fall through to Step 4 to replenish them.
        if len(p0) > 50 and self._is_stationary(prev_img, curr_img):
            print(f"Stationary detected. Locking pose. (Tracks: {len(p0)})")

            return VOState(
                image=curr_img.copy(),
                R=prev_state.R,
                t=prev_state.t,
                keypoints=p0,
                descriptors=None,
                landmarks_3d=all_landmarks,
                landmarks_tracks=prev_state.landmarks_tracks,
                landmark_indices=landmark_indices,
                candidates=candidates,
                trajectory=prev_state.trajectory,
                landmark_counts=prev_state.landmark_counts,
                axis_limits=prev_state.axis_limits,
            )

        # --- 1. KLT Tracking ---
        # FIX: Handle empty p0 input to avoid calcOpticalFlowPyrLK crash
        if len(p0) > 0:
            p1, st_p, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, p0, None)
        else:
            p1, st_p = None, None

        # Track candidates
        if len(candidates) > 0:
            c0 = np.float32([c["keypoint"] for c in candidates])
            c1, st_c, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, c0, None)
        else:
            c1 = np.empty((0, 2), dtype=np.float32)
            st_c = np.empty((0, 1), dtype=np.uint8)

        # FIX: Handle NoneType returns from KLT (Total Failure Case)
        if p1 is None or st_p is None:
            print(
                "WARNING: Tracking failed completely (0 points). Attempting recovery."
            )
            p1 = np.empty((0, 2), dtype=np.float32)
            st_p = np.empty((0, 1), dtype=np.uint8)
            valid_p = np.zeros(0, dtype=bool)
        else:
            # Filter tracks using w, h (now safely defined)
            valid_p = self._filter_tracks(p1, st_p, w, h)

        p1_good = p1[valid_p]
        p0_good = p0[valid_p]
        curr_landmark_indices = landmark_indices[valid_p]

        # --- 2. Pose Estimation (PnP + Physics Check) ---
        has_3d = curr_landmark_indices != -1
        indices_with_3d = np.where(has_3d)[0]
        pnp_2d = p1_good[has_3d]
        pnp_3d_indices = curr_landmark_indices[has_3d]
        pnp_3d = all_landmarks[pnp_3d_indices]

        # Range-gate 3D points used by PnP to avoid far-out outliers influencing pose.
        C_prev = (-prev_state.R.T @ prev_state.t).flatten()
        if len(pnp_3d) > 0:
            d = np.linalg.norm(pnp_3d - C_prev.reshape(1, 3), axis=1)
            keep = d <= 350.0
            if np.any(keep):
                pnp_2d = pnp_2d[keep]
                pnp_3d = pnp_3d[keep]
                indices_with_3d = indices_with_3d[keep]
            else:
                pnp_2d = np.empty((0, 2), dtype=np.float32)
                pnp_3d = np.empty((0, 3), dtype=np.float64)
                indices_with_3d = np.empty((0,), dtype=np.int64)

        pose_solved = False
        curr_R = prev_state.R
        curr_t = prev_state.t
        p1_inliers = p1_good

        # Step 2a: Attempt PnP
        if len(pnp_2d) >= self.min_inliers_pnp:
            success, rvec, tvec, inliers_idx = cv2.solvePnPRansac(
                pnp_3d,
                pnp_2d,
                self.intrinsics.K,
                None,
                confidence=0.999,
                reprojectionError=self.ransac_thresh,
                iterationsCount=self.max_iters,
                flags=cv2.SOLVEPNP_EPNP,
            )

            if success:
                # PnP gives ABSOLUTE pose (camera-to-world transform)
                # We want to keep PnP's absolute localization but scale the motion
                R_pnp, _ = cv2.Rodrigues(rvec)
                t_pnp = tvec
                
                # Compute camera centers in world frame
                C_curr_pnp = (-R_pnp.T @ t_pnp).flatten()
                
                # Scale the MOTION (change in camera center), not the pose itself
                # This preserves PnP's absolute rotation while adjusting translation
                motion = C_curr_pnp - C_prev
                motion_scaled = motion * scale_factor
                C_curr = C_prev + motion_scaled
                
                # Keep PnP's absolute rotation, adjust translation for scaled motion
                curr_R = R_pnp
                curr_t = -R_pnp @ C_curr.reshape(3, 1)
                
                # Calculate velocity (distance moved in world frame)
                dist_moved = np.linalg.norm(motion_scaled)

                # Update Velocity Memory (Smooth it)
                self.last_velocity = 0.7 * self.last_velocity + 0.3 * dist_moved

                # Filter Inliers
                inliers_idx = inliers_idx.flatten()
                pnp_mask = np.zeros(len(p1_good), dtype=bool)
                pnp_mask[indices_with_3d[inliers_idx]] = True

                p1_inliers = p1_good[pnp_mask]
                curr_landmark_indices = curr_landmark_indices[pnp_mask]
                pose_solved = True

        # Step 2b: Fallback with Velocity Injection
        if not pose_solved:
            print(f"!! FALLBACK !! Using E-Matrix with Scale={self.last_velocity:.3f}")

            # Use ALL tracked points for E-Matrix (more robust than just 3D ones)
            R_rel, t_rel, inlier_mask = self._estimate_relative_pose(p0_good, p1_good)

            # INJECT VELOCITY: Scale the unit vector t_rel
            # Also apply ground plane scale correction
            t_rel_scaled = t_rel * self.last_velocity * scale_factor

            # Integrate
            curr_R = R_rel @ prev_state.R
            curr_t = R_rel @ prev_state.t + t_rel_scaled

            inliers = inlier_mask.astype(bool)
            p1_inliers = p1_good[inliers]
            curr_landmark_indices = curr_landmark_indices[inliers]

        # --- 3. Manage Candidates ---
        next_candidates = []
        landmarks_3d_new = all_landmarks
        landmarks_tracks = prev_state.landmarks_tracks.copy()

        # Only process candidates if we successfully tracked them
        if len(candidates) > 0 and c1 is not None and st_c is not None:
            valid_c = self._filter_tracks(c1, st_c, w, h)
            c1_good = c1[valid_c]
            valid_c_indices = np.where(valid_c)[0]

            for idx, mapped_idx in enumerate(valid_c_indices):
                cand_obj = candidates[mapped_idx]
                curr_pt = c1_good[idx]

                angle = self._compute_candidate_parallax(
                    cand_obj["first_obs"], curr_pt, curr_R
                )

                if angle > self.parallax_thresh:
                    landmark_3d = self._triangulate_single_point(
                        cand_obj["first_obs"], curr_pt, curr_R, curr_t
                    )

                    # Cheirality Check: Point must be in front of current camera
                    X_cam = curr_R @ landmark_3d.reshape(3, 1) + curr_t
                    if X_cam[2] > 0:
                        # Range-gate newly triangulated landmarks (world coords) to suppress huge outliers.
                        C_curr = (-curr_R.T @ curr_t).reshape(3)
                        if float(np.linalg.norm(landmark_3d.reshape(3) - C_curr)) > 350.0:
                            continue
                        landmarks_3d_new = np.vstack((landmarks_3d_new, landmark_3d))
                        landmarks_tracks.append(cand_obj["first_obs"])

                        # Add new landmark to tracking
                        new_id = len(landmarks_3d_new) - 1
                        p1_inliers = np.vstack((p1_inliers, curr_pt.reshape(1, 2)))
                        curr_landmark_indices = np.append(curr_landmark_indices, new_id)
                else:
                    cand_obj["keypoint"] = curr_pt
                    next_candidates.append(cand_obj)

        # --- 4. Replenish Features ---
        current_count = len(p1_inliers) + len(next_candidates)
        if current_count < self.target_features:
            new_candidates = self._detect_new_features(
                curr_img, p1_inliers, next_candidates, curr_R, curr_t
            )
            next_candidates.extend(new_candidates)

        # --- 5. Local Bundle Adjustment ---
        self.frame_counter += 1

        if self.enable_ba and self.ba_module is not None:
            # Add current frame to BA window
            self.ba_module.add_frame(
                frame_idx=self.frame_counter,
                R=curr_R,
                t=curr_t,
                keypoints=p1_inliers,
                landmark_indices=curr_landmark_indices,
            )

            # Run BA if triggered
            if self.ba_module.should_run():
                optimized_poses, landmarks_3d_new = self.ba_module.run(landmarks_3d_new)

                # Update current pose with optimized values
                if len(optimized_poses) > 0:
                    curr_R, curr_t = optimized_poses[-1]

        return VOState(
            image=curr_img.copy(),
            R=curr_R,
            t=curr_t,
            keypoints=p1_inliers,
            descriptors=None,
            landmarks_3d=landmarks_3d_new,
            landmarks_tracks=landmarks_tracks,
            landmark_indices=curr_landmark_indices,
            candidates=next_candidates,
            trajectory=prev_state.trajectory,
            landmark_counts=prev_state.landmark_counts,
            axis_limits=prev_state.axis_limits,
            world_reset_pending=False,
            world_R=np.asarray(getattr(prev_state, "world_R", np.eye(3)), dtype=np.float64),
            world_s=float(getattr(prev_state, "world_s", 1.0)),
        )

    def _filter_tracks(self, points, status, w, h):
        status = status.reshape(-1).astype(bool)
        inside = (
            (points[:, 0] >= 0)
            & (points[:, 0] < w)
            & (points[:, 1] >= 0)
            & (points[:, 1] < h)
        )
        return status & inside

    def _detect_new_features(self, curr_img, tracked_pts, candidates, curr_R, curr_t):
        current_count = len(tracked_pts) + len(candidates)
        mask = np.zeros_like(curr_img, dtype=np.uint8)
        mask[:] = 255

        # Mask out existing points
        if len(tracked_pts) > 0:
            for pt in tracked_pts:
                cv2.circle(mask, tuple(pt.astype(int)), self.min_distance, 0, -1)
        if len(candidates) > 0:
            for cand in candidates:
                cv2.circle(
                    mask, tuple(cand["keypoint"].astype(int)), self.min_distance, 0, -1
                )

        new_pts = cv2.goodFeaturesToTrack(
            curr_img,
            maxCorners=max(10, self.target_features - current_count),
            qualityLevel=0.01,
            minDistance=self.min_distance,
            mask=mask,
        )

        new_candidates = []
        if new_pts is not None:
            for pt in new_pts:
                pt_flat = pt[0]
                new_cand = {
                    "keypoint": pt_flat,
                    "first_obs": {
                        "first_keypoint": pt_flat,
                        "first_R": curr_R.copy(),
                        "first_t": curr_t.copy(),
                    },
                }
                new_candidates.append(new_cand)
        return new_candidates

    def _is_stationary(self, img1, img2, threshold=1.0):
        """
        Check if the camera is stationary based on image difference.
        Returns True if Mean Absolute Difference is below threshold.
        """
        if img1 is None or img2 is None:
            return False

        diff = cv2.absdiff(img1, img2)
        mean_diff = np.mean(diff)
        return mean_diff < threshold
