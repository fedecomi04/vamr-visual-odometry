import cv2 as cv
import numpy as np

from .continuous_operation_module import ContinuousOperationModule
from .structure import VOState


class ContinuousOperationSIFT(ContinuousOperationModule):
    """
    Continuous VO operation using SIFT feature matching.

    This class handles frame-to-frame visual odometry using SIFT descriptors
    for feature detection and matching.
    """

    def __init__(
        self,
        intrinsics,
        ransac_thresh=1.0,
        max_iters=2000,
        match_ratio=0.75,
        candidate_match_ratio=0.70,
        parallax_thresh_deg=2.0,
        min_angle_deg=1.0,
    ):
        """
        Initialize the SIFT-based continuous operation module.

        Args:
            intrinsics: Camera intrinsics object with K matrix.
            ransac_thresh: RANSAC threshold for essential matrix estimation.
            max_iters: Maximum iterations for RANSAC.
            match_ratio: Lowe's ratio test threshold for landmark matching.
            candidate_match_ratio: Lowe's ratio test threshold for candidate matching.
            parallax_thresh_deg: Parallax threshold in degrees for triangulation.
            min_angle_deg: Minimum angle for landmark update triangulation.
        """
        super().__init__(intrinsics, ransac_thresh, max_iters, parallax_thresh_deg)
        self.match_ratio = match_ratio
        self.candidate_match_ratio = candidate_match_ratio
        self.min_angle_deg = min_angle_deg

        # Initialize SIFT detector
        self.sift = cv.SIFT_create()
        self.bf = cv.BFMatcher()

    def process_frame(self, curr_img, prev_state: VOState) -> VOState:
        """
        Process a single frame for continuous VO.

        Args:
            curr_img: Current grayscale image.
            prev_state: VOState object containing previous frame's state.

        Returns:
            Updated VOState object for the current frame.
        """
        # 1. Extract keypoints and descriptors
        kp_curr, des_curr = self.sift.detectAndCompute(curr_img, None)

        if des_curr is None:
            return prev_state

        # Initialize the Association Map for this frame (-1 means "not a landmark yet")
        curr_landmark_indices = np.full(len(kp_curr), -1, dtype=int)

        # 2. Match to Existing Landmarks
        matches = self._compute_matches(des_curr, prev_state.descriptors)

        # Filter matches using the Previous Frame's Association Map
        good_matches = []
        kp_curr_matched = []
        kp_prev_matched = []
        matched_global_ids = []

        for m in matches:
            prev_idx = m.trainIdx
            curr_idx = m.queryIdx

            # Look up the Global Landmark ID from the previous frame
            global_id = prev_state.landmark_indices[prev_idx]

            if global_id != -1:
                good_matches.append(m)
                kp_curr_matched.append(kp_curr[curr_idx].pt)
                kp_prev_matched.append(prev_state.keypoints[prev_idx].pt)
                matched_global_ids.append(global_id)
                curr_landmark_indices[curr_idx] = global_id

        kp_curr_matched = np.float32(kp_curr_matched)
        kp_prev_matched = np.float32(kp_prev_matched)

        # 3. Estimate Relative Pose
        R_rel, t_rel, inlier_mask = self._estimate_relative_pose(
            kp_prev_matched, kp_curr_matched
        )

        # 4. Update Global Pose
        curr_R = R_rel @ prev_state.R
        curr_t = R_rel @ prev_state.t + t_rel

        # 5. Update existing landmarks (Triangulate Inliers)
        landmarks_3d = prev_state.landmarks_3d.copy()
        landmarks_tracks = prev_state.landmarks_tracks.copy()

        if np.sum(inlier_mask) > 0:
            inlier_indices = np.where(inlier_mask)[0]
            p_curr_inliers = kp_curr_matched[inlier_indices]
            valid_global_ids = [matched_global_ids[i] for i in inlier_indices]

            landmarks_3d = self._update_landmarks(
                landmarks_3d,
                landmarks_tracks,
                valid_global_ids,
                p_curr_inliers,
                curr_R,
                curr_t,
            )

        # 6. Manage Candidates
        candidates = prev_state.candidates
        next_candidates, landmarks_3d, landmarks_tracks, curr_landmark_indices = (
            self._manage_candidates(
                candidates,
                kp_curr,
                des_curr,
                curr_landmark_indices,
                curr_R,
                curr_t,
                landmarks_3d,
                landmarks_tracks,
            )
        )

        # 7. Return Updated State
        return VOState(
            image=curr_img.copy(),
            R=curr_R,
            t=curr_t,
            keypoints=kp_curr,  # Keep as cv2.KeyPoint objects for SIFT
            descriptors=des_curr,
            landmarks_3d=landmarks_3d,
            landmarks_tracks=landmarks_tracks,
            landmark_indices=curr_landmark_indices,
            candidates=next_candidates,
            # Preserve from previous state (updated in main.py)
            trajectory=prev_state.trajectory,
            landmark_counts=prev_state.landmark_counts,
            axis_limits=prev_state.axis_limits,
            world_reset_pending=False,
            world_R=np.asarray(getattr(prev_state, "world_R", np.eye(3)), dtype=np.float64),
            world_s=float(getattr(prev_state, "world_s", 1.0)),
        )

    def _compute_matches(self, des_curr, des_prev):
        """Compute matches between current and previous descriptors."""
        if des_prev is None or len(des_prev) == 0:
            return []
        matches = self.bf.knnMatch(des_curr, des_prev, k=2)
        good = []
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                good.append(m)
        return good

    def _update_landmarks(
        self,
        landmarks_3d,
        landmarks_tracks,
        matched_global_ids,
        uv_curr,
        curr_R,
        curr_t,
    ):
        """Update existing landmarks via triangulation."""
        K = self.intrinsics.K

        relevant_tracks = [landmarks_tracks[i] for i in matched_global_ids]

        uv_first = np.float32([t["first_keypoint"] for t in relevant_tracks])
        R_first = np.array([t["first_R"] for t in relevant_tracks])
        t_first = np.array([t["first_t"] for t in relevant_tracks])

        angles = self._compute_bearing_angle(uv_first, uv_curr, R_first, curr_R)
        valid_mask = angles > np.deg2rad(self.min_angle_deg)

        if np.sum(valid_mask) == 0:
            return landmarks_3d

        P_curr = K @ np.hstack((curr_R, curr_t))
        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            global_idx = matched_global_ids[idx]
            P_first = K @ np.hstack((R_first[idx], t_first[idx]))

            X_hom = cv.triangulatePoints(
                P_first, P_curr, uv_first[idx].reshape(2, 1), uv_curr[idx].reshape(2, 1)
            )
            landmarks_3d[global_idx] = (X_hom[:3] / X_hom[3]).flatten()

        return landmarks_3d

    def _compute_bearing_angle(self, uv1, uv2, R1_list, R2):
        """Compute bearing angles between corresponding points."""
        K_inv = np.linalg.inv(self.intrinsics.K)
        p1_norm = np.hstack((uv1, np.ones((uv1.shape[0], 1)))) @ K_inv.T
        p2_norm = np.hstack((uv2, np.ones((uv2.shape[0], 1)))) @ K_inv.T
        ray1 = p1_norm / np.linalg.norm(p1_norm, axis=1, keepdims=True)
        ray2 = p2_norm / np.linalg.norm(p2_norm, axis=1, keepdims=True)

        angles = []
        for i in range(len(uv1)):
            r1_world = R1_list[i].T @ ray1[i]
            r2_world = R2.T @ ray2[i]
            dot = np.clip(np.dot(r1_world, r2_world), -1.0, 1.0)
            angles.append(np.arccos(dot))
        return np.array(angles)

    def _manage_candidates(
        self,
        candidates,
        kp_curr,
        des_curr,
        curr_landmark_indices,
        curr_R,
        curr_t,
        landmarks_3d,
        landmarks_tracks,
    ):
        """Manage candidate keypoints: track, triangulate, or create new."""
        next_candidates = []
        used_indices = set(np.where(curr_landmark_indices != -1)[0])

        if len(candidates) > 0 and len(des_curr) > 0:
            cand_descriptors = np.array([c["descriptor"] for c in candidates])
            matches = self.bf.knnMatch(des_curr, cand_descriptors, k=2)

            for m, n in matches:
                if m.distance < self.candidate_match_ratio * n.distance:
                    curr_idx = m.queryIdx
                    cand_idx = m.trainIdx

                    if curr_idx in used_indices:
                        continue

                    candidate = candidates[cand_idx]
                    curr_pt = kp_curr[curr_idx].pt

                    angle = self._compute_candidate_parallax(
                        candidate["first_obs"], curr_pt, curr_R
                    )

                    if angle > self.parallax_thresh:
                        landmark_3d = self._triangulate_single_point(
                            candidate["first_obs"], curr_pt, curr_R, curr_t
                        )

                        landmarks_3d = np.vstack((landmarks_3d, landmark_3d))
                        landmarks_tracks.append(candidate["first_obs"])

                        new_id = len(landmarks_3d) - 1
                        curr_landmark_indices[curr_idx] = new_id
                        used_indices.add(curr_idx)
                    else:
                        candidate["descriptor"] = des_curr[curr_idx]
                        candidate["keypoint"] = curr_pt
                        next_candidates.append(candidate)
                        used_indices.add(curr_idx)

        # Create new candidates from unmatched keypoints
        for i in range(len(kp_curr)):
            if i not in used_indices:
                new_cand = {
                    "descriptor": des_curr[i],
                    "keypoint": kp_curr[i].pt,
                    "first_obs": {
                        "first_keypoint": kp_curr[i].pt,
                        "first_R": curr_R.copy(),
                        "first_t": curr_t.copy(),
                    },
                }
                next_candidates.append(new_cand)

        return next_candidates, landmarks_3d, landmarks_tracks, curr_landmark_indices
