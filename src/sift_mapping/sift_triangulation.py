from __future__ import annotations

import cv2
import numpy as np


def _project_matrix(K: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_wc, dtype=np.float64).reshape(3, 1)
    Rt = np.concatenate([R, t], axis=1)
    return K @ Rt


def _bearing_world(K: np.ndarray, R_wc: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """
    Pixel -> unit bearing vector in WORLD coords for each point.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    x = np.concatenate([pts, ones], axis=1).T  # 3xN
    rays_c = np.linalg.inv(K) @ x  # 3xN
    rays_c /= np.linalg.norm(rays_c, axis=0, keepdims=True).clip(min=1e-12)
    rays_w = R_wc.T @ rays_c
    rays_w /= np.linalg.norm(rays_w, axis=0, keepdims=True).clip(min=1e-12)
    return rays_w.T  # Nx3


def triangulate_from_poses(
    K: np.ndarray,
    R0_wc: np.ndarray,
    t0_wc: np.ndarray,
    R1_wc: np.ndarray,
    t1_wc: np.ndarray,
    pts0_px: np.ndarray,
    pts1_px: np.ndarray,
    min_parallax_deg: float = 2.0,
    max_reproj_err_px: float | None = None,
    return_stats: bool = False,
) -> tuple:
    """
    Batch triangulation using known world->camera poses (no pose estimation).

    Returns:
      Xw: (M,3) triangulated points in WORLD coords (only for valid_mask==True)
      valid_mask: (N,) mask over input correspondences
    """
    pts0 = np.asarray(pts0_px, dtype=np.float64).reshape(-1, 2)
    pts1 = np.asarray(pts1_px, dtype=np.float64).reshape(-1, 2)
    n = int(min(pts0.shape[0], pts1.shape[0]))
    pts0 = pts0[:n]
    pts1 = pts1[:n]
    if n == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)

    P0 = _project_matrix(K, R0_wc, t0_wc)
    P1 = _project_matrix(K, R1_wc, t1_wc)

    x0 = pts0.T  # 2xN
    x1 = pts1.T  # 2xN
    X_h = cv2.triangulatePoints(P0, P1, x0, x1)  # 4xN
    X_h /= X_h[3:4].clip(min=1e-12)
    Xw = X_h[:3].T  # Nx3 in world coords

    R0 = np.asarray(R0_wc, dtype=np.float64).reshape(3, 3)
    t0 = np.asarray(t0_wc, dtype=np.float64).reshape(3, 1)
    R1 = np.asarray(R1_wc, dtype=np.float64).reshape(3, 3)
    t1 = np.asarray(t1_wc, dtype=np.float64).reshape(3, 1)

    Xc0 = (R0 @ Xw.T + t0).T
    Xc1 = (R1 @ Xw.T + t1).T
    cheirality = (Xc0[:, 2] > 0) & (Xc1[:, 2] > 0)

    b0 = _bearing_world(K, R0, pts0)
    b1 = _bearing_world(K, R1, pts1)
    cosang = np.sum(b0 * b1, axis=1).clip(-1.0, 1.0)
    parallax_deg = np.degrees(np.arccos(cosang))
    parallax_ok = parallax_deg >= float(min_parallax_deg)

    def _reproj_err(P: np.ndarray, Xw_: np.ndarray, pts_: np.ndarray) -> np.ndarray:
        Xh = np.concatenate([Xw_, np.ones((Xw_.shape[0], 1), dtype=np.float64)], axis=1).T  # 4xN
        x = (P @ Xh).T  # Nx3
        u = x[:, 0] / x[:, 2].clip(min=1e-12)
        v = x[:, 1] / x[:, 2].clip(min=1e-12)
        du = u - pts_[:, 0]
        dv = v - pts_[:, 1]
        return np.sqrt(du * du + dv * dv)

    if max_reproj_err_px is None:
        reproj_ok = np.ones((n,), dtype=bool)
    else:
        err0 = _reproj_err(P0, Xw, pts0)
        err1 = _reproj_err(P1, Xw, pts1)
        reproj_ok = (err0 <= float(max_reproj_err_px)) & (err1 <= float(max_reproj_err_px))

    finite_ok = np.all(np.isfinite(Xw), axis=1)
    valid = cheirality & parallax_ok & reproj_ok & finite_ok

    if not return_stats:
        return Xw[valid], valid

    attempted = int(n)
    kept = int(np.count_nonzero(valid))
    rej_parallax = int(np.count_nonzero(~parallax_ok))
    rej_depth = int(np.count_nonzero(~cheirality))
    rej_reproj = int(np.count_nonzero(~reproj_ok)) if max_reproj_err_px is not None else 0
    stats = {
        "attempted": attempted,
        "kept": kept,
        "rej_parallax": rej_parallax,
        "rej_depth": rej_depth,
        "rej_reproj": rej_reproj,
    }
    return Xw[valid], valid, stats
