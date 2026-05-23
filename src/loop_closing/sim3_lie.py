from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Sim3:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)
    s: float


def axis_angle_to_R(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(w)
    return R


def R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    w, _ = cv2.Rodrigues(R)
    return w.reshape(3)


def pack_sim3(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    w = R_to_axis_angle(R)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    log_s = float(np.log(float(s)))
    return np.concatenate([w, t, np.array([log_s], dtype=np.float64)])


def unpack_sim3(p: np.ndarray) -> Sim3:
    p = np.asarray(p, dtype=np.float64).reshape(7)
    w = p[:3]
    t = p[3:6]
    log_s = float(p[6])
    s = float(np.exp(log_s))
    R = axis_angle_to_R(w)
    return Sim3(R=R, t=t, s=s)


def sim3_to_matrix(S: Sim3) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = float(S.s) * np.asarray(S.R, dtype=np.float64).reshape(3, 3)
    M[:3, 3] = np.asarray(S.t, dtype=np.float64).reshape(3)
    return M


def sim3_from_se3_matrix(T: np.ndarray) -> Sim3:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return Sim3(R=T[:3, :3].copy(), t=T[:3, 3].copy(), s=1.0)


def sim3_compose(S2: Sim3, S1: Sim3) -> Sim3:
    """
    Composition for point maps: x' = S(x) = s R x + t
    Returns S = S2 ∘ S1.
    """
    s = float(S2.s) * float(S1.s)
    R = np.asarray(S2.R) @ np.asarray(S1.R)
    t = float(S2.s) * (np.asarray(S2.R) @ np.asarray(S1.t)) + np.asarray(S2.t)
    return Sim3(R=R, t=t.reshape(3), s=s)


def sim3_inverse(S: Sim3) -> Sim3:
    s_inv = 1.0 / float(S.s)
    R_inv = np.asarray(S.R).T
    t_inv = (-s_inv) * (R_inv @ np.asarray(S.t).reshape(3, 1))
    return Sim3(R=R_inv, t=t_inv.reshape(3), s=s_inv)


def sim3_relative(T_i: Sim3, T_j: Sim3) -> Sim3:
    """Relative transform Z_ij = T_j ∘ inv(T_i)."""
    return sim3_compose(T_j, sim3_inverse(T_i))


def sim3_residual(meas: Sim3, pred: Sim3) -> np.ndarray:
    """
    7D residual for an edge with measurement meas and prediction pred:
      - rotation: axis-angle of R_meas^T R_pred
      - translation: t_pred - t_meas
      - log-scale: log(s_pred) - log(s_meas)
    """
    R_err = np.asarray(meas.R).T @ np.asarray(pred.R)
    w_err = R_to_axis_angle(R_err)
    t_err = np.asarray(pred.t).reshape(3) - np.asarray(meas.t).reshape(3)
    log_s_err = float(np.log(float(pred.s)) - np.log(float(meas.s)))
    return np.concatenate([w_err, t_err, np.array([log_s_err], dtype=np.float64)])


def apply_sim3_to_point(S: Sim3, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(3)
    return (float(S.s) * (np.asarray(S.R) @ x.reshape(3, 1))).reshape(3) + np.asarray(
        S.t
    ).reshape(3)

