from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np


def _get_attr(state: Any, name: str, default=None):
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def get_vo_selected_features(
    state, mode: str, image_gray=None
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Returns (pts2d, descriptors) to be used for loop closure database insertion.
    - pts2d: Nx2 float32
    - descriptors: NxD array or None

    For KLT mode, descriptors are computed ONLY at pts2d using ORB (no detection).
    """
    mode = str(mode).lower().strip()
    if mode not in {"sift", "klt"}:
        raise ValueError(f"mode must be 'sift' or 'klt', got {mode!r}")

    if mode == "sift":
        keypoints = _get_attr(state, "keypoints_prev", None)
        if keypoints is None:
            # fallback for VOState-based branch
            keypoints = _get_attr(state, "keypoints", None)
        descriptors = _get_attr(state, "descriptors_prev", None)
        if descriptors is None:
            descriptors = _get_attr(state, "descriptors", None)

        if keypoints is None or descriptors is None or len(descriptors) == 0:
            return np.empty((0, 2), dtype=np.float32), None, np.empty((0,), dtype=np.int64)

        pts2d = np.asarray([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
        descriptors = np.asarray(descriptors)

        landmark_indices = _get_attr(state, "landmark_indices", None)
        if landmark_indices is not None and len(landmark_indices) == len(pts2d):
            # "Selected" VO set: exclude unassociated points and candidate pools.
            mask = np.asarray(landmark_indices) != -1
            pts2d = pts2d[mask]
            descriptors = descriptors[mask]
            landmark_ids = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)[mask]
        else:
            landmark_ids = -np.ones((len(pts2d),), dtype=np.int64)

        return pts2d, descriptors, landmark_ids

    # KLT mode
    pts2d = _get_attr(state, "keypoints_prev", None)
    if pts2d is None:
        pts2d = _get_attr(state, "keypoints", None)
    if pts2d is None or len(pts2d) == 0:
        return np.empty((0, 2), dtype=np.float32), None, np.empty((0,), dtype=np.int64)

    pts2d = np.asarray(pts2d, dtype=np.float32).reshape(-1, 2)
    landmark_indices = _get_attr(state, "landmark_indices", None)
    if landmark_indices is not None and len(landmark_indices) == len(pts2d):
        landmark_ids = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
    else:
        landmark_ids = -np.ones((len(pts2d),), dtype=np.int64)

    if image_gray is None:
        image_gray = _get_attr(state, "image", None)
    if image_gray is None:
        return pts2d, None, landmark_ids

    orb = cv2.ORB_create()
    kps = []
    for idx, (x, y) in enumerate(pts2d):
        kp = cv2.KeyPoint(float(x), float(y), 20)
        kp.class_id = int(idx)
        kps.append(kp)
    kps2, des = orb.compute(image_gray, kps)
    if des is None or kps2 is None or len(kps2) == 0:
        return np.empty((0, 2), dtype=np.float32), None, np.empty((0,), dtype=np.int64)

    pts2d2 = np.asarray([kp.pt for kp in kps2], dtype=np.float32).reshape(-1, 2)
    kept = np.asarray([kp.class_id for kp in kps2], dtype=np.int64)
    kept = kept[(kept >= 0) & (kept < len(landmark_ids))]
    landmark_ids2 = landmark_ids[kept] if len(kept) == len(pts2d2) else -np.ones((len(pts2d2),), dtype=np.int64)
    return pts2d2, des, landmark_ids2
