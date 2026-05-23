from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from loop_closing.keyframes import Keyframe

from .map_utils import resolve_duplicates_by_radius


class SIFTMap:
    """
    Global SIFT landmark map in WORLD coordinates + stored SIFT keyframes.

    Notes:
    - Landmarks are stored in a dynamically resized contiguous array to support
      amortized O(1) batch appends and efficient in-place warps.
    - Keyframes are stored as `loop_closing.keyframes.Keyframe` objects for
      compatibility with existing loop-closing code. An extra attribute
      `is_valid: bool` is attached at creation time by the mapper.
    """

    def __init__(self, initial_capacity: int = 4096):
        self._cap = int(max(1, initial_capacity))
        self._n = 0
        self._landmarks = np.empty((self._cap, 3), dtype=np.float64)
        self.keyframes: List[Keyframe] = []

    @property
    def landmarks_3d(self) -> np.ndarray:
        return self._landmarks[: self._n]

    @property
    def num_landmarks(self) -> int:
        return int(self._n)

    def add_landmarks(self, Xw_batch: np.ndarray) -> np.ndarray:
        """
        Append a batch of world points (Nx3) and return their new landmark ids (N,).
        """
        Xw = np.asarray(Xw_batch, dtype=np.float64).reshape(-1, 3)
        n_new = int(Xw.shape[0])
        if n_new == 0:
            return np.empty((0,), dtype=np.int64)

        needed = self._n + n_new
        if needed > self._cap:
            new_cap = self._cap
            while new_cap < needed:
                new_cap *= 2
            new_arr = np.empty((new_cap, 3), dtype=np.float64)
            if self._n > 0:
                new_arr[: self._n] = self._landmarks[: self._n]
            self._landmarks = new_arr
            self._cap = new_cap

        start = self._n
        end = self._n + n_new
        self._landmarks[start:end] = Xw
        self._n = end
        return np.arange(start, end, dtype=np.int64)

    def resolve_or_add_landmarks(
        self, Xw_batch: np.ndarray, radius: float = 0.5
    ) -> tuple[np.ndarray, int, int]:
        """
        Add a batch of points while suppressing near-duplicates:
        - if a candidate is within `radius` of an existing landmark, reuse its id
        - otherwise insert it as a new landmark

        Returns:
          landmark_ids: (N,) assigned ids (existing or new)
          num_new: number of newly inserted landmarks
          num_merged: number of candidates merged into existing landmarks
        """
        Xw = np.asarray(Xw_batch, dtype=np.float64).reshape(-1, 3)
        if Xw.shape[0] == 0:
            return np.empty((0,), dtype=np.int64), 0, 0

        existing = self.landmarks_3d
        resolved, keep_new, merged = resolve_duplicates_by_radius(
            existing_xyz=existing, new_xyz=Xw, radius=float(radius)
        )

        ids = np.asarray(resolved, dtype=np.int64).reshape(-1)
        if np.any(keep_new):
            new_ids = self.add_landmarks(Xw[keep_new])
            ids[keep_new] = new_ids
            return ids, int(new_ids.shape[0]), int(merged)
        return ids, 0, int(merged)

    def add_keyframe(self, kf: Keyframe) -> None:
        self.keyframes.append(kf)

    def apply_sim3_inplace(self, S_W: np.ndarray) -> None:
        """
        Apply a 4x4 similarity transform to all landmarks in-place.
        """
        if self._n == 0:
            return
        S = np.asarray(S_W, dtype=np.float64).reshape(4, 4)
        A = S[:3, :3]
        t = S[:3, 3].reshape(1, 3)
        P = self._landmarks[: self._n]
        P[:] = (P @ A.T) + t
