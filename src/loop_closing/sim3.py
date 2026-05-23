from __future__ import annotations

from typing import Tuple

import numpy as np


def umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama similarity alignment mapping X -> Y:
        Y ≈ s * R * X + t
    X, Y: (N,3)
    Returns: (s, R, t) with R (3,3), t (3,)
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 3)
    if X.shape[0] != Y.shape[0] or X.shape[0] < 3:
        raise ValueError(f"Need N>=3 correspondences with matching shapes, got {X.shape} vs {Y.shape}")

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    cov = (Yc.T @ Xc) / float(X.shape[0])
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    var_x = float(np.mean(np.sum(Xc * Xc, axis=1)))
    if var_x < 1e-12:
        raise ValueError("Degenerate configuration: variance of X is ~0")

    s = float(np.trace(np.diag(D) @ S) / var_x)
    t = mu_y - s * (R @ mu_x)
    return s, R, t


def ransac_sim3(
    X: np.ndarray,
    Y: np.ndarray,
    thresh: float = 0.2,
    max_iters: int = 2000,
    min_inliers: int = 20,
    scale_bounds: tuple[float, float] = (0.2, 5.0),
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    RANSAC over Sim(3) hypotheses (sample 3 correspondences).
    Returns (s, R, t, inlier_mask, success).
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 3)
    n = X.shape[0]
    if n != Y.shape[0] or n < 3:
        return 1.0, np.eye(3), np.zeros(3), np.zeros(n, dtype=bool), False

    thresh = float(thresh)
    lo_s, hi_s = float(scale_bounds[0]), float(scale_bounds[1])

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best = (1.0, np.eye(3), np.zeros(3))

    rng = np.random.default_rng()
    for _ in range(int(max_iters)):
        idx = rng.choice(n, size=3, replace=False)
        try:
            s, R, t = umeyama_sim3(X[idx], Y[idx])
        except Exception:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        if s < lo_s or s > hi_s:
            continue
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            continue

        Y_hat = (s * (X @ R.T)) + t  # (s R X + t) with row-vectors
        err = np.linalg.norm(Y - Y_hat, axis=1)
        inliers = err < thresh
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best = (s, R, t)

    if best_count < int(min_inliers):
        return best[0], best[1], best[2], best_inliers, False

    # Refit on inliers
    try:
        s, R, t = umeyama_sim3(X[best_inliers], Y[best_inliers])
    except Exception:
        s, R, t = best

    if not np.isfinite(s) or s <= 0 or s < lo_s or s > hi_s:
        return s, R, t, best_inliers, False

    return float(s), np.asarray(R), np.asarray(t), best_inliers, True

