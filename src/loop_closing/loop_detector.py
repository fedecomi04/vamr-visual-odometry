from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .keyframes import Keyframe, KeyframeManager
from .place_recognition import DescriptorDatabase


class LoopCandidate:
    def __init__(
        self,
        query_kf: Keyframe,
        match_kf: Keyframe,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        inlier_mask: np.ndarray,
        idx_query: np.ndarray,
        idx_db: np.ndarray,
    ):
        self.query_kf = query_kf
        self.match_kf = match_kf
        self.R_rel = R_rel
        self.t_rel = t_rel
        self.inlier_mask = inlier_mask
        self.idx_query = idx_query
        self.idx_db = idx_db


def _match_keypoints(
    des_query: np.ndarray,
    des_db: np.ndarray,
    pts_query: np.ndarray,
    pts_db: np.ndarray,
    ratio: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm = cv2.NORM_HAMMING if des_query.dtype == np.uint8 else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(des_db, des_query, k=2)
    p_db = []
    p_q = []
    idx_db = []
    idx_q = []
    for mn in matches:
        if len(mn) < 2:
            continue
        m, n = mn
        if m.distance < ratio * n.distance:
            p_db.append(pts_db[m.queryIdx])
            p_q.append(pts_query[m.trainIdx])
            idx_db.append(int(m.queryIdx))
            idx_q.append(int(m.trainIdx))
    if len(p_db) == 0:
        empty2 = np.empty((0, 2), dtype=np.float32)
        empty1 = np.empty((0,), dtype=np.int64)
        return empty2, empty2, empty1, empty1
    return (
        np.asarray(p_db, dtype=np.float32),
        np.asarray(p_q, dtype=np.float32),
        np.asarray(idx_q, dtype=np.int64),
        np.asarray(idx_db, dtype=np.int64),
    )


def find_loop_closure(
    keyframe_manager: KeyframeManager,
    db: DescriptorDatabase,
    K: np.ndarray,  # 3x3 intrinsics
    ransac_thresh: float = 1.0,
    max_iters: int = 2000,
    min_inliers: int = 80,
    top_k: int = 10,
    debug: bool = False,
) -> Optional[LoopCandidate]:
    """
    Use the most recent keyframe as query.
    1) Get top candidates from DescriptorDatabase.
    2) For each candidate, match keypoints and run Essential matrix RANSAC.
    3) If a candidate passes RANSAC with enough inliers, return a LoopCandidate.
    4) If none pass, return None.
    """
    query_kf = keyframe_manager.get_last_keyframe()
    if query_kf is None:
        return None

    candidates = db.query(query_kf, top_k=int(top_k))
    if debug:
        print(
            f"[LC] query kf={query_kf.frame_idx} feats={0 if query_kf.descriptors is None else len(query_kf.descriptors)} "
            f"cands={len(candidates)}"
        )
    if not candidates:
        return None

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    for cand_kf, _score in candidates:
        if query_kf.descriptors is None or cand_kf.descriptors is None:
            continue
        des_query = np.asarray(query_kf.descriptors)
        des_db = np.asarray(cand_kf.descriptors)
        if des_query.dtype != des_db.dtype:
            if debug:
                print(
                    f"[LC] skip cand kf={cand_kf.frame_idx} dtype mismatch db={des_db.dtype} q={des_query.dtype}"
                )
            continue

        p_db, p_q, idx_q, idx_db = _match_keypoints(
            des_query=des_query,
            des_db=des_db,
            pts_query=np.asarray(query_kf.keypoints, dtype=np.float32),
            pts_db=np.asarray(cand_kf.keypoints, dtype=np.float32),
            ratio=0.7,
        )
        if p_db.shape[0] < 8:
            if debug:
                print(f"[LC] cand kf={cand_kf.frame_idx} matches={int(p_db.shape[0])} (<8)")
            continue
        if debug:
            print(f"[LC] cand kf={cand_kf.frame_idx} matches={int(p_db.shape[0])} (pre-RANSAC)")

        E, mask = cv2.findEssentialMat(
            p_db,
            p_q,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=float(ransac_thresh),
            maxIters=int(max_iters),
        )
        if E is None or mask is None:
            if debug:
                print(f"[LC] cand kf={cand_kf.frame_idx} E/mask=None")
            continue

        inlier_mask = mask.ravel().astype(bool)
        if int(np.count_nonzero(inlier_mask)) < int(min_inliers):
            if debug:
                print(
                    f"[LC] cand kf={cand_kf.frame_idx} E_inl={int(np.count_nonzero(inlier_mask))} (<{int(min_inliers)})"
                )
            continue

        _n, R_rel, t_rel, mask_pose = cv2.recoverPose(E, p_db, p_q, K, mask=mask)
        mask_pose = (
            mask_pose.ravel().astype(bool) if mask_pose is not None else inlier_mask
        )
        if int(np.count_nonzero(mask_pose)) < int(min_inliers):
            if debug:
                print(
                    f"[LC] cand kf={cand_kf.frame_idx} pose_inl={int(np.count_nonzero(mask_pose))} (<{int(min_inliers)})"
                )
            continue
        if debug:
            print(
                f"[LC] cand kf={cand_kf.frame_idx} E_inl={int(np.count_nonzero(inlier_mask))} pose_inl={int(np.count_nonzero(mask_pose))} (PASS)"
            )

        return LoopCandidate(
            query_kf=query_kf,
            match_kf=cand_kf,
            R_rel=np.asarray(R_rel, dtype=np.float64),
            t_rel=np.asarray(t_rel, dtype=np.float64),
            inlier_mask=mask_pose,
            idx_query=idx_q,
            idx_db=idx_db,
        )

    return None
