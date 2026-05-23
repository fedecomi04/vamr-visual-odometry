# strcutures.py
# simple classes / dataclasses used for inputs/outputs

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    K: np.ndarray


@dataclass
class VOInitializationResultSift:
    R: np.ndarray
    t: np.ndarray
    points_3d: np.ndarray
    keypoints0: np.ndarray
    keypoints1: np.ndarray
    descriptors0: np.ndarray
    descriptors1: np.ndarray
    inlier_mask: np.ndarray


@dataclass
class VOInitializationResultKLT:
    R: np.ndarray
    t: np.ndarray
    points_3d: np.ndarray
    keypoints0: np.ndarray
    keypoints1: np.ndarray
    inlier_mask: np.ndarray
