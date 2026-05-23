from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None


def resolve_duplicates_by_radius(
    existing_xyz: np.ndarray,
    new_xyz: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    For each point in new_xyz, either:
      - resolve to an existing landmark id (if within radius), or
      - mark as new (id = -1).

    Returns:
      resolved_ids: (N,) existing ids or -1
      keep_new_mask: (N,) True if should be inserted as a new landmark
      merged_count: number of points resolved to existing landmarks
    """
    Xn = np.asarray(new_xyz, dtype=np.float64).reshape(-1, 3)
    Xe = np.asarray(existing_xyz, dtype=np.float64).reshape(-1, 3)
    if Xn.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.zeros((0,), dtype=bool), 0
    if Xe.shape[0] == 0 or cKDTree is None:
        resolved = -np.ones((Xn.shape[0],), dtype=np.int64)
        keep = np.ones((Xn.shape[0],), dtype=bool)
        return resolved, keep, 0

    tree = cKDTree(Xe)
    dist, nn = tree.query(Xn, k=1, workers=-1)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    nn = np.asarray(nn, dtype=np.int64).reshape(-1)

    resolved = -np.ones((Xn.shape[0],), dtype=np.int64)
    keep = dist > float(radius)
    resolved[~keep] = nn[~keep]
    merged = int(np.count_nonzero(~keep))
    return resolved, keep, merged

