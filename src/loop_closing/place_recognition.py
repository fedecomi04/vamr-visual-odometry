from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .keyframes import Keyframe


class DescriptorDatabase:
    def __init__(self, max_keyframes: int = 500, temporal_exclusion: int = 30, debug: bool = False):
        self.keyframes: List[Keyframe] = []
        self.max_keyframes = int(max_keyframes)
        self.temporal_exclusion = int(temporal_exclusion)
        self._total_descriptors: int = 0
        self.debug = bool(debug)

    def add_keyframe(self, kf: Keyframe):
        self.keyframes.append(kf)
        self._total_descriptors += int(0 if kf.descriptors is None else len(kf.descriptors))
        if len(self.keyframes) > self.max_keyframes:
            removed = self.keyframes[:-self.max_keyframes]
            self.keyframes = self.keyframes[-self.max_keyframes :]
            for old in removed:
                self._total_descriptors -= int(
                    0 if old.descriptors is None else len(old.descriptors)
                )

    @property
    def total_descriptors(self) -> int:
        return int(self._total_descriptors)

    def query(self, query_kf: Keyframe, top_k: int = 10) -> List[Tuple[Keyframe, float]]:
        """
        Return up to top_k candidate keyframes (older than temporal_exclusion)
        sorted by a simple similarity score (#good_matches / #query_features).
        """
        if query_kf.descriptors is None or len(query_kf.descriptors) == 0:
            return []

        query_des = np.asarray(query_kf.descriptors)
        denom = float(max(1, len(query_des)))
        norm = cv2.NORM_HAMMING if query_des.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm)

        scored: List[Tuple[Keyframe, float]] = []
        for db_kf in self.keyframes:
            if abs(db_kf.frame_idx - query_kf.frame_idx) < self.temporal_exclusion:
                if self.debug:
                    print(
                        f"[LC-DB] skip kf={db_kf.frame_idx} (temporal_exclusion {self.temporal_exclusion})"
                    )
                continue
            if db_kf.descriptors is None or len(db_kf.descriptors) == 0:
                if self.debug:
                    print(f"[LC-DB] skip kf={db_kf.frame_idx} (no descriptors)")
                continue

            db_des = np.asarray(db_kf.descriptors)
            if db_des.dtype != query_des.dtype:
                if self.debug:
                    print(
                        f"[LC-DB] skip kf={db_kf.frame_idx} (dtype mismatch db={db_des.dtype} q={query_des.dtype})"
                    )
                continue
            matches = bf.knnMatch(query_des, db_des, k=2)
            good = 0
            for mn in matches:
                if len(mn) < 2:
                    continue
                m, n = mn
                if m.distance < 0.7 * n.distance:
                    good += 1
            score = good / denom
            if self.debug:
                print(
                    f"[LC-DB] cand kf={db_kf.frame_idx} dtype_ok=1 good_matches={good} score={score:.3f}"
                )
            scored.append((db_kf, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: int(top_k)]
