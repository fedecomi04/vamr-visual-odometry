from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def _decompose_sim3_matrix(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Decompose a 4x4 similarity transform x' = (sR)x + t.
    Returns (R, t, s).
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    A = T[:3, :3]
    t = T[:3, 3].reshape(3)
    detA = float(np.linalg.det(A))
    if not np.isfinite(detA) or abs(detA) < 1e-12:
        return np.eye(3), t, 1.0
    s = float(np.cbrt(detA))
    if not np.isfinite(s) or abs(s) < 1e-12:
        return np.eye(3), t, 1.0
    R = A / s
    return R, t, s


def _compose_sim3_matrix(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = float(s) * np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _camera_center_from_Twc_sim3(T_wc: np.ndarray) -> np.ndarray:
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
    return (-(1.0 / s) * (R.T @ t)).reshape(3)


def slerp(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Rotation interpolation (slerp) between R0 and R1.
    Uses scipy if available, otherwise quaternion slerp fallback.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)

    try:
        from scipy.spatial.transform import Rotation as Rot
        from scipy.spatial.transform import Slerp

        key_rots = Rot.from_matrix([R0, R1])
        key_times = np.array([0.0, 1.0], dtype=np.float64)
        slerp_obj = Slerp(key_times, key_rots)
        return slerp_obj([a]).as_matrix()[0]
    except Exception:
        pass

    def _mat_to_quat(Rm: np.ndarray) -> np.ndarray:
        tr = np.trace(Rm)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (Rm[2, 1] - Rm[1, 2]) / S
            qy = (Rm[0, 2] - Rm[2, 0]) / S
            qz = (Rm[1, 0] - Rm[0, 1]) / S
        else:
            if Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
                S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2.0
                qw = (Rm[2, 1] - Rm[1, 2]) / S
                qx = 0.25 * S
                qy = (Rm[0, 1] + Rm[1, 0]) / S
                qz = (Rm[0, 2] + Rm[2, 0]) / S
            elif Rm[1, 1] > Rm[2, 2]:
                S = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2.0
                qw = (Rm[0, 2] - Rm[2, 0]) / S
                qx = (Rm[0, 1] + Rm[1, 0]) / S
                qy = 0.25 * S
                qz = (Rm[1, 2] + Rm[2, 1]) / S
            else:
                S = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2.0
                qw = (Rm[1, 0] - Rm[0, 1]) / S
                qx = (Rm[0, 2] + Rm[2, 0]) / S
                qy = (Rm[1, 2] + Rm[2, 1]) / S
                qz = 0.25 * S
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        return q / np.linalg.norm(q)

    def _quat_to_mat(q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )

    q0 = _mat_to_quat(R0)
    q1 = _mat_to_quat(R1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        q = q / np.linalg.norm(q)
        return _quat_to_mat(q)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * a
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = (s0 * q0) + (s1 * q1)
    return _quat_to_mat(q / np.linalg.norm(q))


def interpolate_sim3(G0: np.ndarray, G1: np.ndarray, alpha: float) -> np.ndarray:
    R0, t0, s0 = _decompose_sim3_matrix(G0)
    R1, t1, s1 = _decompose_sim3_matrix(G1)
    a = float(np.clip(alpha, 0.0, 1.0))
    R = slerp(R0, R1, a)
    t = (1.0 - a) * t0 + a * t1
    log_s = (1.0 - a) * np.log(max(1e-12, s0)) + a * np.log(max(1e-12, s1))
    s = float(np.exp(log_s))
    return _compose_sim3_matrix(R, t, s)


@dataclass
class WarpStats:
    max_center_error_at_kf: float
    mean_center_error_at_kf: float
    num_keyframe_checks: int


def compute_frame_corrections(
    frames_T_wc_vo: List[np.ndarray],
    keyframes: List,
    pose_graph,
    base_frame_idx: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], WarpStats]:
    """
    Compute per-frame corrected poses/centers by interpolating correction transforms between keyframes.
    Returns (poses_T_wc_corr, centers_corr, stats).
    """
    if not frames_T_wc_vo:
        return [], [], WarpStats(0.0, 0.0, 0)

    kfs = sorted(list(keyframes), key=lambda k: int(k.frame_idx))
    if not kfs:
        poses_corr = [np.asarray(T, dtype=np.float64).copy() for T in frames_T_wc_vo]
        centers = [_camera_center_from_Twc_sim3(T) for T in poses_corr]
        return poses_corr, centers, WarpStats(0.0, 0.0, 0)

    # Precompute per-keyframe correction G_k = T_opt_k @ inv(T_vo_k)
    kf_frames = np.array([int(k.frame_idx) for k in kfs], dtype=np.int64)
    G_list = []
    centers_opt = []
    for k in kfs:
        T_vo = np.eye(4, dtype=np.float64)
        T_vo[:3, :3] = np.asarray(k.R_wc, dtype=np.float64).reshape(3, 3)
        T_vo[:3, 3] = np.asarray(k.t_wc, dtype=np.float64).reshape(3)
        T_opt = np.asarray(pose_graph.get_optimized_T_wc(k.id), dtype=np.float64).reshape(4, 4)
        Gk = T_opt @ np.linalg.inv(T_vo)
        G_list.append(Gk)
        centers_opt.append(_camera_center_from_Twc_sim3(T_opt))
    G_list = list(G_list)
    centers_opt = np.asarray(centers_opt, dtype=np.float64)

    poses_corr: List[np.ndarray] = []
    centers_corr: List[np.ndarray] = []

    for i, T_vo_i in enumerate(frames_T_wc_vo):
        frame_idx = int(base_frame_idx + i)
        # bracket
        if frame_idx <= kf_frames[0]:
            G_i = G_list[0]
        elif frame_idx >= kf_frames[-1]:
            G_i = G_list[-1]
        else:
            j = int(np.searchsorted(kf_frames, frame_idx, side="right"))
            j0 = j - 1
            j1 = j
            f0 = int(kf_frames[j0])
            f1 = int(kf_frames[j1])
            alpha = 0.0 if f1 == f0 else float((frame_idx - f0) / (f1 - f0))
            G_i = interpolate_sim3(G_list[j0], G_list[j1], alpha)

        T_corr_i = np.asarray(G_i, dtype=np.float64) @ np.asarray(T_vo_i, dtype=np.float64)
        poses_corr.append(T_corr_i)
        centers_corr.append(_camera_center_from_Twc_sim3(T_corr_i))

    # Sanity: keyframe camera centers should match optimized centers (where frame exists)
    errs = []
    checks = 0
    for idx_kf, k in enumerate(kfs):
        rel = int(k.frame_idx) - int(base_frame_idx)
        if 0 <= rel < len(poses_corr):
            C_corr = centers_corr[rel]
            C_opt = centers_opt[idx_kf]
            errs.append(float(np.linalg.norm(np.asarray(C_corr) - np.asarray(C_opt))))
            checks += 1

    if errs:
        stats = WarpStats(float(np.max(errs)), float(np.mean(errs)), checks)
    else:
        stats = WarpStats(0.0, 0.0, 0)

    return poses_corr, centers_corr, stats

