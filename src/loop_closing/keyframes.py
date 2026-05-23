from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Keyframe:
    id: int  # running KF id
    frame_idx: int  # original frame index in sequence
    R_wc: np.ndarray  # 3x3 world->camera rotation
    t_wc: np.ndarray  # 3x1 world->camera translation
    keypoints: np.ndarray  # Nx2 array of 2D keypoints (image coordinates)
    descriptors: np.ndarray  # NxD descriptors (SIFT float32 or ORB uint8)
    landmark_ids: np.ndarray  # (N,) int landmark index, -1 if none
    is_loop_closure: bool = False  # optional flag


def _camera_center_world(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    t_wc = np.asarray(t_wc, dtype=float).reshape(3, 1)
    R_wc = np.asarray(R_wc, dtype=float).reshape(3, 3)
    return (-R_wc.T @ t_wc).reshape(3)


def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R1 = np.asarray(R1, dtype=float).reshape(3, 3)
    R2 = np.asarray(R2, dtype=float).reshape(3, 3)
    R_rel = R2 @ R1.T
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


class KeyframeManager:
    def __init__(self, min_translation=3, min_rotation_deg=10.0, min_frame_gap=200):
        self.min_translation = float(min_translation)
        self.min_rotation_deg = float(min_rotation_deg)
        self.min_frame_gap = int(min_frame_gap)
        self._keyframes: List[Keyframe] = []
        self._next_id: int = 0

    def should_create_keyframe(
        self, frame_idx: int, R_wc: np.ndarray, t_wc: np.ndarray
    ) -> bool:
        """Return True if this frame should become a keyframe based on motion and frame gap."""
        last = self.get_last_keyframe()
        if last is None:
            return True

        frame_gap = int(frame_idx) - int(last.frame_idx)
        if frame_gap >= self.min_frame_gap:
            return True

        c_last = _camera_center_world(last.R_wc, last.t_wc)
        c_curr = _camera_center_world(R_wc, t_wc)
        translation = float(np.linalg.norm(c_curr - c_last))
        if translation > self.min_translation:
            return True

        rot_deg = _rotation_angle_deg(last.R_wc, R_wc)
        if rot_deg > self.min_rotation_deg:
            return True

        return False

    def add_keyframe(
        self,
        frame_idx: int,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        landmark_ids: np.ndarray,
    ) -> Keyframe:
        """Create and store a new keyframe and return it."""
        kf = Keyframe(
            id=self._next_id,
            frame_idx=int(frame_idx),
            R_wc=np.asarray(R_wc, dtype=np.float64).reshape(3, 3).copy(),
            t_wc=np.asarray(t_wc, dtype=np.float64).reshape(3, 1).copy(),
            keypoints=np.asarray(keypoints, dtype=np.float32).reshape(-1, 2).copy(),
            descriptors=np.asarray(descriptors).copy(),
            landmark_ids=np.asarray(landmark_ids, dtype=np.int64).reshape(-1).copy(),
        )
        self._keyframes.append(kf)
        self._next_id += 1
        return kf

    def get_all_keyframes(self) -> List[Keyframe]:
        return list(self._keyframes)

    def get_last_keyframe(self) -> Optional[Keyframe]:
        return self._keyframes[-1] if self._keyframes else None
