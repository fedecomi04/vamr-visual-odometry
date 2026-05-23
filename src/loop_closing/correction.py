from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .loop_detector import LoopCandidate
from .pose_graph import PoseGraph
from .sim3 import ransac_sim3
from .sim3_lie import Sim3, sim3_compose, sim3_inverse, sim3_relative


@dataclass
class Sim3CorrectionStats:
    pairs: int
    inliers: int
    inlier_ratio: float
    scale: float
    applied: bool
    reason: str


def _camera_center_from_Twc_sim3(T_wc: np.ndarray) -> np.ndarray:
    """
    Camera center for a world->camera Sim(3) matrix:
        x_c = A x_w + t, with A = s R
        C_w = -(1/s) R^T t
    Works for SE(3) as a special case (s=1).
    """
    T_wc = np.asarray(T_wc, dtype=np.float64).reshape(4, 4)
    A = T_wc[:3, :3]
    t = T_wc[:3, 3].reshape(3, 1)
    detA = float(np.linalg.det(A))
    if not np.isfinite(detA) or abs(detA) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    s = float(np.cbrt(detA))
    if not np.isfinite(s) or abs(s) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    R = A / s
    C = (-(1.0 / s) * (R.T @ t)).reshape(3)
    return C


def build_3d_pairs_from_loop(
    loop_candidate: LoopCandidate, state: Any
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 3D–3D pairs from inlier 2D matches using landmark IDs.

    For each inlier match k:
      id_q = query_kf.landmark_ids[idx_query[k]]
      id_m = match_kf.landmark_ids[idx_db[k]]
    If both ids != -1, collect:
      Xq = landmarks_3d[id_q]
      Xm = landmarks_3d[id_m]
    Returns (X, Y) as Nx3 arrays in WORLD coordinates.
    """
    landmarks_3d = getattr(state, "landmarks_3d", None)
    if landmarks_3d is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    landmarks_3d = np.asarray(landmarks_3d, dtype=np.float64)

    q = loop_candidate.query_kf
    m = loop_candidate.match_kf

    idx_q = np.asarray(loop_candidate.idx_query, dtype=np.int64).reshape(-1)
    idx_m = np.asarray(loop_candidate.idx_db, dtype=np.int64).reshape(-1)
    inl = np.asarray(loop_candidate.inlier_mask, dtype=bool).reshape(-1)

    n = min(len(idx_q), len(idx_m), len(inl))
    idx_q = idx_q[:n]
    idx_m = idx_m[:n]
    inl = inl[:n]

    lm_q = np.asarray(getattr(q, "landmark_ids", []), dtype=np.int64).reshape(-1)
    lm_m = np.asarray(getattr(m, "landmark_ids", []), dtype=np.int64).reshape(-1)
    if lm_q.size == 0 or lm_m.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    X_list = []
    Y_list = []
    for k in np.where(inl)[0]:
        iq = int(idx_q[k])
        im = int(idx_m[k])
        if iq < 0 or iq >= lm_q.size or im < 0 or im >= lm_m.size:
            continue
        id_q = int(lm_q[iq])
        id_m = int(lm_m[im])
        if id_q == -1 or id_m == -1:
            continue
        if id_q < 0 or id_q >= landmarks_3d.shape[0]:
            continue
        if id_m < 0 or id_m >= landmarks_3d.shape[0]:
            continue
        X_list.append(landmarks_3d[id_q])
        Y_list.append(landmarks_3d[id_m])

    if not X_list:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    return np.asarray(X_list, dtype=np.float64), np.asarray(Y_list, dtype=np.float64)


def build_3d_pairs_in_camera_frames(
    loop_candidate: LoopCandidate, state: Any
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 3D–3D pairs in KEYFRAME CAMERA coordinates (not world coords).

    For each inlier match k:
      id_q = query_kf.landmark_ids[idx_query[k]]
      id_m = match_kf.landmark_ids[idx_db[k]]
    If both valid:
      Pq_w = landmarks_3d[id_q]
      Pm_w = landmarks_3d[id_m]

      X_cam = R_q @ Pq_w + t_q
      Y_cam = R_m @ Pm_w + t_m

    Returns (X_cam, Y_cam) as Nx3 arrays.
    """
    landmarks_3d = getattr(state, "landmarks_3d", None)
    if landmarks_3d is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    landmarks_3d = np.asarray(landmarks_3d, dtype=np.float64)

    q = loop_candidate.query_kf
    m = loop_candidate.match_kf

    Rq = np.asarray(q.R_wc, dtype=np.float64).reshape(3, 3)
    tq = np.asarray(q.t_wc, dtype=np.float64).reshape(3, 1)
    Rm = np.asarray(m.R_wc, dtype=np.float64).reshape(3, 3)
    tm = np.asarray(m.t_wc, dtype=np.float64).reshape(3, 1)

    idx_q = np.asarray(loop_candidate.idx_query, dtype=np.int64).reshape(-1)
    idx_m = np.asarray(loop_candidate.idx_db, dtype=np.int64).reshape(-1)
    inl = np.asarray(loop_candidate.inlier_mask, dtype=bool).reshape(-1)

    n = min(len(idx_q), len(idx_m), len(inl))
    idx_q = idx_q[:n]
    idx_m = idx_m[:n]
    inl = inl[:n]

    lm_q = np.asarray(getattr(q, "landmark_ids", []), dtype=np.int64).reshape(-1)
    lm_m = np.asarray(getattr(m, "landmark_ids", []), dtype=np.int64).reshape(-1)
    if lm_q.size == 0 or lm_m.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    X_list = []
    Y_list = []
    for k in np.where(inl)[0]:
        iq = int(idx_q[k])
        im = int(idx_m[k])
        if iq < 0 or iq >= lm_q.size or im < 0 or im >= lm_m.size:
            continue
        id_q = int(lm_q[iq])
        id_m = int(lm_m[im])
        if id_q == -1 or id_m == -1:
            continue
        if id_q < 0 or id_q >= landmarks_3d.shape[0]:
            continue
        if id_m < 0 or id_m >= landmarks_3d.shape[0]:
            continue
        Pq_w = np.asarray(landmarks_3d[id_q], dtype=np.float64).reshape(3, 1)
        Pm_w = np.asarray(landmarks_3d[id_m], dtype=np.float64).reshape(3, 1)
        X_list.append((Rq @ Pq_w + tq).reshape(3))
        Y_list.append((Rm @ Pm_w + tm).reshape(3))

    if not X_list:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    return np.asarray(X_list, dtype=np.float64), np.asarray(Y_list, dtype=np.float64)


def _invert_sim3(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 similarity transform with A = sR (s>0), mapping x' = A x + t.
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    A = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    detA = float(np.linalg.det(A))
    if not np.isfinite(detA) or abs(detA) < 1e-12:
        return np.eye(4, dtype=np.float64)
    s = float(np.cbrt(detA))
    if not np.isfinite(s) or abs(s) < 1e-12:
        return np.eye(4, dtype=np.float64)
    R = A / s
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = (1.0 / s) * R.T
    Tinv[:3, 3] = (-(1.0 / s) * (R.T @ t)).reshape(3)
    return Tinv


def compute_world_warp_from_anchor_keyframe(kf, pose_graph: PoseGraph) -> np.ndarray:
    """
    Compute a similarity warp S_map that maps VO world points -> optimized world points.

    Uses a single anchor keyframe k:
      x_c = T_wc_vo * P_w_vo
      x_c = T_wc_opt * P_w_opt
    => P_w_opt = inv(T_wc_opt) * T_wc_vo * P_w_vo
    => S_map = inv(T_wc_opt) * T_wc_vo
    """
    T_wc_vo = np.eye(4, dtype=np.float64)
    T_wc_vo[:3, :3] = np.asarray(kf.R_wc, dtype=np.float64).reshape(3, 3)
    T_wc_vo[:3, 3] = np.asarray(kf.t_wc, dtype=np.float64).reshape(3)

    T_wc_opt = np.asarray(pose_graph.get_optimized_T_wc(kf.id), dtype=np.float64).reshape(4, 4)
    Twc_opt = _invert_sim3(T_wc_opt)
    S_map = Twc_opt @ T_wc_vo
    return S_map


def apply_world_warp_to_landmarks(landmarks_3d: np.ndarray, S_map: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 similarity transform S_map to world points (Nx3).
    """
    P = np.asarray(landmarks_3d, dtype=np.float64).reshape(-1, 3)
    S_map = np.asarray(S_map, dtype=np.float64).reshape(4, 4)
    A = S_map[:3, :3]
    t = S_map[:3, 3]
    return (P @ A.T) + t.reshape(1, 3)


def compute_global_correction_sim3(
    loop_candidate: LoopCandidate,
    pose_graph: PoseGraph,
    state: Any,
    sim3_thresh: float = 0.2,
    max_iters: int = 2000,
    min_shared_landmarks: int = 30,
    min_inliers: int = 20,
    min_inlier_ratio: float = 0.5,
    scale_bounds: Tuple[float, float] = (0.5, 2.0),
) -> Tuple[Optional[np.ndarray], Sim3CorrectionStats]:
    """
    Estimate a scale-aware global correction using Sim(3) from 3D–3D correspondences built
    from inlier 2D matches (via landmark IDs -> world points).

    Returns (G, stats) where:
      - G is a 4x4 world->camera similarity matrix to left-multiply T_wc poses: T_wc_corr = G @ T_wc
      - If rejected, G is None and stats.applied == False.
    """
    query_kf = loop_candidate.query_kf
    match_kf = loop_candidate.match_kf

    Xq, Xm = build_3d_pairs_from_loop(loop_candidate, state)
    pairs = int(Xq.shape[0])
    if pairs < int(min_shared_landmarks):
        return None, Sim3CorrectionStats(
            pairs=pairs,
            inliers=0,
            inlier_ratio=0.0,
            scale=1.0,
            applied=False,
            reason="too_few_3d_pairs",
        )

    s, R, t, inlier_mask, ok = ransac_sim3(
        Xq,
        Xm,
        thresh=sim3_thresh,
        max_iters=max_iters,
        min_inliers=min_inliers,
        scale_bounds=scale_bounds,
    )
    inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inliers / max(1, pairs))

    if (not ok) or (inlier_ratio < float(min_inlier_ratio)):
        return None, Sim3CorrectionStats(
            pairs=pairs,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            scale=float(s),
            applied=False,
            reason="ransac_failed_or_low_inlier_ratio",
        )

    # S acts on WORLD coordinates: Y ≈ s R X + t
    S = np.eye(4, dtype=np.float64)
    S[:3, :3] = float(s) * np.asarray(R, dtype=np.float64).reshape(3, 3)
    S[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)

    # Update camera->world: Twc_opt = S @ Twc_old (use query as reference)
    Tq_old = pose_graph.get_keyframe_pose(query_kf.id)  # world->camera
    Twq_old = np.linalg.inv(Tq_old)  # camera->world (SE3)
    Twq_opt = S @ Twq_old  # camera->world (Sim3)
    Tq_opt = _invert_sim3(Twq_opt)  # world->camera (Sim3)
    G = Tq_opt @ np.linalg.inv(Tq_old)

    return G, Sim3CorrectionStats(
        pairs=pairs,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        scale=float(s),
        applied=True,
        reason="applied",
    )


@dataclass
class LoopSim3MeasStats:
    pairs: int
    inliers: int
    inlier_ratio: float
    scale: float
    accepted: bool
    reason: str


def compute_loop_sim3_measurement(
    loop_candidate: LoopCandidate,
    pose_graph: PoseGraph,
    state: Any,
    sim3_thresh: float = 0.2,
    max_iters: int = 2000,
    min_pairs: int = 50,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.5,
    scale_bounds: Tuple[float, float] = (0.5, 2.0),
    debug: bool = False,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, float]], LoopSim3MeasStats]:
    """
    Compute a loop edge measurement Z_ij directly between KEYFRAME CAMERA frames.

    1) Build 3D–3D pairs in each keyframe's camera coordinates:
         X_cam (query frame) and Y_cam (match frame)
    2) Estimate Sim(3) mapping query-camera -> match-camera:
         Y_cam ≈ s * R * X_cam + t
       This (R,t,s) is the desired pose-graph measurement Z_ij to constrain:
         Z_ij ≈ T_j ∘ inv(T_i)
    """
    X_cam, Y_cam = build_3d_pairs_in_camera_frames(loop_candidate, state)
    pairs = int(X_cam.shape[0])
    if debug:
        q = loop_candidate.query_kf
        m = loop_candidate.match_kf
        print(
            f"[LC-Sim3] q={q.frame_idx} m={m.frame_idx} pairs={pairs} min_pairs={int(min_pairs)}"
        )
    if pairs < int(min_pairs):
        return None, LoopSim3MeasStats(
            pairs=pairs,
            inliers=0,
            inlier_ratio=0.0,
            scale=1.0,
            accepted=False,
            reason="too_few_pairs",
        )

    s, R, t, inlier_mask, ok = ransac_sim3(
        X_cam,
        Y_cam,
        thresh=sim3_thresh,
        max_iters=max_iters,
        min_inliers=min_inliers,
        scale_bounds=scale_bounds,
    )
    inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inliers / max(1, pairs))
    if debug:
        print(
            f"[LC-Sim3] ransac ok={int(ok)} inl={inliers}/{pairs} ratio={inlier_ratio:.2f} s={float(s):.3f}"
        )

    if (not ok) or (inlier_ratio < float(min_inlier_ratio)):
        return None, LoopSim3MeasStats(
            pairs=pairs,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            scale=float(s),
            accepted=False,
            reason="ransac_failed_or_low_inlier_ratio",
        )

    return (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3), float(s)), LoopSim3MeasStats(
        pairs=pairs,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        scale=float(s),
        accepted=True,
        reason="accepted",
    )


def _debug_check_sim3_inverse_compose() -> bool:
    """Sanity check: inv(S) ∘ S ≈ identity."""
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.1, size=3)
    Rw, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Rw) < 0:
        Rw[:, 2] *= -1
    t = rng.normal(scale=1.0, size=3)
    s = float(np.exp(rng.normal(scale=0.1)))
    S = Sim3(R=Rw, t=t, s=s)
    I = sim3_compose(sim3_inverse(S), S)
    return (
        np.allclose(I.R, np.eye(3), atol=1e-6)
        and np.allclose(I.t, np.zeros(3), atol=1e-6)
        and abs(I.s - 1.0) < 1e-6
    )


__all__ = [
    "Sim3CorrectionStats",
    "compute_global_correction_sim3",
    "build_3d_pairs_from_loop",
    "build_3d_pairs_in_camera_frames",
    "LoopSim3MeasStats",
    "compute_loop_sim3_measurement",
    "_camera_center_from_Twc_sim3",
    "compute_world_warp_from_anchor_keyframe",
    "apply_world_warp_to_landmarks",
]
