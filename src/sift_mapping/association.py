from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None


def build_visible_landmark_kdtree(
    landmarks_3d: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    max_range_m: float | None = None,
) -> tuple[Optional["cKDTree"], np.ndarray, np.ndarray]:
    """
    Project all landmarks and build a KDTree in pixel space for visible landmarks.

    Returns:
      tree: cKDTree over (u,v) or None if unavailable/empty
      visible_landmark_ids: (M,) ids into landmarks_3d
      uv: (M,2) projected pixel coordinates
    """
    P = np.asarray(landmarks_3d, dtype=np.float64).reshape(-1, 3)
    if P.size == 0:
        return None, np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_wc, dtype=np.float64).reshape(3, 1)

    Xc = (R @ P.T + t).T  # Nx3
    z = Xc[:, 2]
    in_front = z > 0
    if not np.any(in_front):
        return None, np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

    ids = np.nonzero(in_front)[0].astype(np.int64)

    if max_range_m is not None:
        C = (-(R.T @ t)).reshape(3)
        d = np.linalg.norm(P[ids] - C.reshape(1, 3), axis=1)
        in_range = d <= float(max_range_m)
        if not np.any(in_range):
            return None, np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        ids = ids[in_range]
        Xc = Xc[in_front][in_range]
        z = Xc[:, 2]
    else:
        Xc = Xc[in_front]

    x = Xc[:, 0] / z.clip(min=1e-12)
    y = Xc[:, 1] / z.clip(min=1e-12)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    uv = np.stack([u, v], axis=1).astype(np.float32)

    h, w = int(image_shape[0]), int(image_shape[1])
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_bounds):
        return None, np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

    uv = uv[in_bounds]
    ids = ids[in_bounds]

    if cKDTree is None or uv.shape[0] == 0:
        return None, ids, uv

    tree = cKDTree(uv.astype(np.float64))
    return tree, ids, uv


def associate_2d_to_3d(
    kp_uv: np.ndarray,
    tree: Optional["cKDTree"],
    visible_ids: np.ndarray,
    max_px: float = 3.0,
) -> np.ndarray:
    """
    Vectorized nearest-neighbor association of 2D keypoints to projected 3D landmarks.

    Returns:
      landmark_ids: (N,) ids into global landmark array, -1 if none
    """
    pts = np.asarray(kp_uv, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    if tree is None or visible_ids.size == 0:
        return -np.ones((pts.shape[0],), dtype=np.int64)

    dist, nn = tree.query(pts, k=1, workers=-1)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    nn = np.asarray(nn, dtype=np.int64).reshape(-1)

    out = -np.ones((pts.shape[0],), dtype=np.int64)
    ok = dist <= float(max_px)
    if np.any(ok):
        out[ok] = np.asarray(visible_ids, dtype=np.int64).reshape(-1)[nn[ok]]
    return out
