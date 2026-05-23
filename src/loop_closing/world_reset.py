from __future__ import annotations

import numpy as np


def decompose_sim3_matrix(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Decompose a 4x4 similarity transform:
        x' = (s R) x + t
    Returns (s, R, t) where R is 3x3 rotation, t is (3,).
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    A = T[:3, :3]
    t = T[:3, 3].reshape(3)
    detA = float(np.linalg.det(A))
    if not np.isfinite(detA) or abs(detA) < 1e-12:
        return 1.0, np.eye(3, dtype=np.float64), t
    s = float(np.cbrt(detA))
    if not np.isfinite(s) or abs(s) < 1e-12:
        return 1.0, np.eye(3, dtype=np.float64), t
    R = A / s
    return s, R, t


def invert_sim3_matrix(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 similarity transform with A = sR (s>0):
        x' = A x + t
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    s, R, t = decompose_sim3_matrix(T)
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = (1.0 / s) * R.T
    Tinv[:3, 3] = (-(1.0 / s) * (R.T @ t.reshape(3, 1))).reshape(3)
    return Tinv


def normalize_Twc_sim3_to_se3(T_wc: np.ndarray) -> np.ndarray:
    """
    Convert a world->camera Sim(3) matrix (A=sR, t) into an SE(3)-like matrix (R, t/s).

    This keeps reprojections equivalent up to a global scaling of camera coordinates:
        x_c = s (R x_w + (t/s))
    """
    T_wc = np.asarray(T_wc, dtype=np.float64).reshape(4, 4)
    s, R, t = decompose_sim3_matrix(T_wc)
    T_se3 = np.eye(4, dtype=np.float64)
    T_se3[:3, :3] = R
    T_se3[:3, 3] = (t / s).reshape(3)
    return T_se3


def sim3_to_points(S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply 4x4 similarity transform S to Nx3 points."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    S = np.asarray(S, dtype=np.float64).reshape(4, 4)
    A = S[:3, :3]
    t = S[:3, 3]
    return (P @ A.T) + t.reshape(1, 3)


def compute_world_correction_from_anchor(kf, pose_graph) -> np.ndarray:
    """
    Compute a world correction S_W that maps the current (frontend) world -> pose-graph world,
    using a single anchor keyframe k:

      S_W = inv(T_wc_opt_k) @ T_wc_vo_k

    where T_wc_vo_k is SE3 from the keyframe fields, and T_wc_opt_k is pose-graph optimized
    world->camera Sim(3).
    """
    T_wc_vo = np.eye(4, dtype=np.float64)
    T_wc_vo[:3, :3] = np.asarray(kf.R_wc, dtype=np.float64).reshape(3, 3)
    T_wc_vo[:3, 3] = np.asarray(kf.t_wc, dtype=np.float64).reshape(3)

    T_wc_opt = np.asarray(pose_graph.get_optimized_T_wc(kf.id), dtype=np.float64).reshape(4, 4)
    return invert_sim3_matrix(T_wc_opt) @ T_wc_vo


def apply_world_correction_to_Twc_list(poses_wc: list[np.ndarray], S_W: np.ndarray) -> None:
    """
    In-place update for a list of world->camera SE3 4x4 poses.
    After correction, each pose remains SE3-like (R, t) via normalization.
    """
    inv_S = invert_sim3_matrix(S_W)
    for i in range(len(poses_wc)):
        Twc_old = np.asarray(poses_wc[i], dtype=np.float64).reshape(4, 4)
        Twc_sim = Twc_old @ inv_S
        poses_wc[i] = normalize_Twc_sim3_to_se3(Twc_sim)


def apply_world_correction_to_Rt(R_wc: np.ndarray, t_wc: np.ndarray, S_W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Update a single (R_wc, t_wc) pose via S_W and return (R_new, t_new) in SE3-like form."""
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    T_wc[:3, 3] = np.asarray(t_wc, dtype=np.float64).reshape(3)
    inv_S = invert_sim3_matrix(S_W)
    T_wc_sim = T_wc @ inv_S
    T_wc_se3 = normalize_Twc_sim3_to_se3(T_wc_sim)
    R_new = T_wc_se3[:3, :3].copy()
    t_new = T_wc_se3[:3, 3].reshape(3, 1).copy()
    return R_new, t_new


def apply_world_correction_to_keyframes(keyframes: list, S_W: np.ndarray) -> None:
    """In-place update for loop_closing.keyframes.Keyframe objects."""
    inv_S = invert_sim3_matrix(S_W)
    for kf in keyframes:
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = np.asarray(kf.R_wc, dtype=np.float64).reshape(3, 3)
        T_wc[:3, 3] = np.asarray(kf.t_wc, dtype=np.float64).reshape(3)
        T_wc_sim = T_wc @ inv_S
        T_wc_se3 = normalize_Twc_sim3_to_se3(T_wc_sim)
        kf.R_wc = T_wc_se3[:3, :3].copy()
        kf.t_wc = T_wc_se3[:3, 3].reshape(3, 1).copy()


def apply_world_correction_to_tracks_and_candidates(state, S_W: np.ndarray) -> None:
    """
    In-place update for VO state's stored reference poses inside:
      - state.landmarks_tracks: list of dicts with keys like {"first_R","first_t",...}
      - state.candidates: list of dicts; in KLT candidates keep cand["first_obs"]["first_R"/"first_t"]

    This avoids mixing "old world" reference poses with "new world" poses/landmarks after a loop reset.
    """
    inv_S = invert_sim3_matrix(S_W)

    def _update_first_pose(d: dict) -> None:
        if "first_R" not in d or "first_t" not in d:
            return
        R_wc = np.asarray(d["first_R"], dtype=np.float64).reshape(3, 3)
        t_wc = np.asarray(d["first_t"], dtype=np.float64).reshape(3, 1)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = t_wc.reshape(3)
        T_wc_sim = T_wc @ inv_S
        T_wc_se3 = normalize_Twc_sim3_to_se3(T_wc_sim)
        d["first_R"] = T_wc_se3[:3, :3].copy()
        d["first_t"] = T_wc_se3[:3, 3].reshape(3, 1).copy()

    tracks = getattr(state, "landmarks_tracks", None)
    if isinstance(tracks, list):
        for tr in tracks:
            if isinstance(tr, dict):
                _update_first_pose(tr)

    candidates = getattr(state, "candidates", None)
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            # KLT: cand["first_obs"] is a dict with first_R/first_t
            first_obs = cand.get("first_obs")
            if isinstance(first_obs, dict):
                _update_first_pose(first_obs)
            # SIFT variants may store first_obs directly
            _update_first_pose(cand)


__all__ = [
    "decompose_sim3_matrix",
    "invert_sim3_matrix",
    "normalize_Twc_sim3_to_se3",
    "sim3_to_points",
    "compute_world_correction_from_anchor",
    "apply_world_correction_to_Twc_list",
    "apply_world_correction_to_Rt",
    "apply_world_correction_to_keyframes",
    "apply_world_correction_to_tracks_and_candidates",
]
