from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from loop_closing.keyframes import Keyframe, KeyframeManager
from .sift_map import SIFTMap
from .sift_triangulation import triangulate_from_poses


class SIFTMapper:
    """
    SIFT-only mapping sidecar that uses KNOWN KLT poses for triangulation.

    - Does not estimate camera poses.
    - Runs SIFT detect/compute only when creating a SIFT keyframe.
    """

    def __init__(self, K: np.ndarray, max_kf_features: int = 800):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.max_kf_features = int(max_kf_features)

        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

        self.kfm = KeyframeManager(min_frame_gap=100, min_translation=5.0, min_rotation_deg=5.0)
        self.map = SIFTMap()
        self.last_kf: Optional[Keyframe] = None

    def _detect_and_compute(self, image_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kps, des = self.sift.detectAndCompute(image_gray, None)
        if kps is None or des is None or len(kps) == 0 or len(des) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

        responses = np.array([kp.response for kp in kps], dtype=np.float32)
        order = np.argsort(-responses)  # deterministic: strongest first
        if len(order) > self.max_kf_features:
            order = order[: self.max_kf_features]
        kps_sel = [kps[i] for i in order.tolist()]
        des_sel = np.asarray(des[order], dtype=np.float32)
        pts = np.asarray([kp.pt for kp in kps_sel], dtype=np.float32).reshape(-1, 2)
        return pts, des_sel

    def _ratio_matches(self, des_a: np.ndarray, des_b: np.ndarray, ratio: float = 0.7) -> np.ndarray:
        """
        Return good matches as an (M,2) int array [idx_a, idx_b].
        """
        if des_a.size == 0 or des_b.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        matches = self.bf.knnMatch(np.asarray(des_a, dtype=np.float32), np.asarray(des_b, dtype=np.float32), k=2)
        good = []
        for mn in matches:
            if len(mn) < 2:
                continue
            m, n = mn
            if m.distance < float(ratio) * n.distance:
                good.append((int(m.queryIdx), int(m.trainIdx)))
        if not good:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray(good, dtype=np.int64)

    def process_frame(
        self,
        frame_idx: int,
        image_gray: np.ndarray,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
    ) -> Optional[Keyframe]:
        if not self.kfm.should_create_keyframe(frame_idx, R_wc, t_wc):
            return None

        t0 = time.perf_counter()
        pts, des = self._detect_and_compute(image_gray)
        t_det = (time.perf_counter() - t0) * 1000.0

        landmark_ids = -np.ones((len(pts),), dtype=np.int64)
        kf = self.kfm.add_keyframe(frame_idx, R_wc, t_wc, pts, des, landmark_ids)
        kf.is_valid = True  # required by the spec (stored per keyframe)

        assoc3d = 0
        matches_n = 0
        tri_attempted = 0
        tri_kept = 0
        tri_rej_parallax = 0
        tri_rej_depth = 0

        new3d = 0

        t_match = 0.0
        t_tri = 0.0

        if self.last_kf is not None and len(self.last_kf.descriptors) > 0 and len(kf.descriptors) > 0:
            t1 = time.perf_counter()
            pairs = self._ratio_matches(self.last_kf.descriptors, kf.descriptors, ratio=0.7)
            t_match = (time.perf_counter() - t1) * 1000.0
            matches_n = int(pairs.shape[0])

            if matches_n > 0:
                idx_last = pairs[:, 0]
                idx_curr = pairs[:, 1]

                lm_last = np.asarray(self.last_kf.landmark_ids, dtype=np.int64).reshape(-1)

                known_mask = (lm_last[idx_last] != -1)
                if np.any(known_mask):
                    # Propagate existing 3D associations.
                    kf.landmark_ids[idx_curr[known_mask]] = lm_last[idx_last[known_mask]]
                    assoc3d = int(np.count_nonzero(known_mask))

                need_mask = ~known_mask
                if np.any(need_mask):
                    idx_last_new = idx_last[need_mask]
                    idx_curr_new = idx_curr[need_mask]

                    unassigned = lm_last[idx_last_new] == -1
                    idx_last_new = idx_last_new[unassigned]
                    idx_curr_new = idx_curr_new[unassigned]

                    if idx_last_new.size > 0:
                        pts0 = np.asarray(self.last_kf.keypoints, dtype=np.float32)[idx_last_new]
                        pts1 = np.asarray(kf.keypoints, dtype=np.float32)[idx_curr_new]

                        t2 = time.perf_counter()
                        Xw_valid, valid_mask, stats = triangulate_from_poses(
                            self.K,
                            self.last_kf.R_wc,
                            self.last_kf.t_wc,
                            kf.R_wc,
                            kf.t_wc,
                            pts0,
                            pts1,
                            min_parallax_deg=2.0,
                            max_reproj_err_px=None,
                            return_stats=True,
                        )
                        t_tri = (time.perf_counter() - t2) * 1000.0
                        tri_attempted += int(stats["attempted"])
                        tri_kept += int(stats["kept"])
                        tri_rej_parallax += int(stats["rej_parallax"])
                        tri_rej_depth += int(stats["rej_depth"])

                        if Xw_valid.shape[0] > 0:
                            new_ids = self.map.add_landmarks(Xw_valid)
                            new3d = int(new_ids.shape[0])
                            valid_idx_last = idx_last_new[valid_mask]
                            valid_idx_curr = idx_curr_new[valid_mask]
                            for lid, il, ic in zip(
                                new_ids.tolist(),
                                valid_idx_last.tolist(),
                                valid_idx_curr.tolist(),
                            ):
                                # Update BOTH keyframes so future loop-closure can form 3D–3D pairs.
                                if self.last_kf.landmark_ids[il] == -1:
                                    self.last_kf.landmark_ids[il] = int(lid)
                                if kf.landmark_ids[ic] == -1:
                                    kf.landmark_ids[ic] = int(lid)

        self.map.add_keyframe(kf)
        self.last_kf = kf

        total = t_det + t_match + t_tri
        assoc_total = int(np.count_nonzero(np.asarray(kf.landmark_ids, dtype=np.int64) != -1))
        print(
            f"[SIFTMap] kf frame={int(frame_idx)} feats={int(len(pts))} "
            f"matches={int(matches_n)} assoc3d={int(assoc_total)} new3d={int(new3d)}"
        )
        print(
            f"[SIFTTriStats] attempted={int(tri_attempted)} kept={int(tri_kept)} "
            f"rej_parallax={int(tri_rej_parallax)} rej_depth={int(tri_rej_depth)}"
        )
        print(
            f"[SIFTMapTiming] frame={int(frame_idx)} detect={t_det:.1f}ms match={t_match:.1f}ms "
            f"tri={t_tri:.1f}ms total={total:.1f}ms"
        )
        return kf
