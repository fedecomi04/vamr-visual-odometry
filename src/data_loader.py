# data_loader.py
import os
from dataclasses import dataclass
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _parse_malaga_timestamp(image_path: str) -> float:
    name = os.path.basename(image_path)
    # e.g. img_CAMERA1_1261229981.580023_left.jpg
    parts = name.split("_")
    for part in parts:
        try:
            return float(part)
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp from Malaga image name: {name}")


def _load_malaga_ground_truth(malaga_root: str, left_images: List[str]) -> np.ndarray:
    gps_path = os.path.join(malaga_root, "malaga-urban-dataset-extract-07_all-sensors_GPS.txt")
    if not os.path.isfile(gps_path):
        raise FileNotFoundError(f"Missing Malaga GPS file for ground truth: {gps_path}")

    # Columns: Time, ..., Local X (col 8), Local Y (col 9)
    gps = np.loadtxt(gps_path, comments="%", dtype=float, usecols=(0, 8, 9))
    gps = np.atleast_2d(gps)
    if gps.shape[0] < 2 or gps.shape[1] != 3:
        raise ValueError(f"Unexpected GPS file format in {gps_path}: got {gps.shape}")

    t_gps = gps[:, 0]
    x_gps = gps[:, 1]
    y_gps = gps[:, 2]

    t_img = np.array([_parse_malaga_timestamp(p) for p in left_images], dtype=float)
    x_img = np.interp(t_img, t_gps, x_gps)
    y_img = np.interp(t_img, t_gps, y_gps)

    traj = np.column_stack([x_img, y_img])
    traj = traj - traj[0]
    # Convention fix: Malaga's provided Local X/Y is flipped vs the VO pose convention used here.
    return -traj


@dataclass
class DatasetPaths:
    kitti: str = "./data/kitti"
    malaga: str = "./data/malaga-urban-dataset-extract-07"
    parking: str = "./data/parking"
    own: str = "./data/own_dataset"


@dataclass
class DatasetInfo:
    ds: int
    K: np.ndarray
    last_frame: int  # inclusive index of last frame
    bootstrap_frames: Tuple[int, int]  # default bootstrap frame indices
    ground_truth: Optional[np.ndarray] = None
    left_images: Optional[List[str]] = None  # used for Malaga (ds == 1)


def load_dataset(ds: int, paths: DatasetPaths) -> DatasetInfo:
    """Load camera matrix K, ground-truth and basic info for the chosen dataset."""
    if ds == 0:
        # KITTI
        gt_path = os.path.join(paths.kitti, "poses", "05.txt")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Missing KITTI ground truth file: {gt_path}")
        ground_truth = np.loadtxt(gt_path)
        # same as MATLAB(:, [end-8 end])
        ground_truth = ground_truth[:, [-9, -1]]
        last_frame = 2760
        bootstrap_frames = (0, 7)
        K = np.array(
            [
                [7.18856e02, 0, 6.071928e02],
                [0, 7.18856e02, 1.852157e02],
                [0, 0, 1],
            ]
        )
        if ground_truth.shape[0] > 0:
            last_frame = min(last_frame, ground_truth.shape[0] - 1)
            ground_truth = ground_truth[: last_frame + 1]
        return DatasetInfo(
            ds=ds,
            K=K,
            last_frame=last_frame,
            bootstrap_frames=bootstrap_frames,
            ground_truth=ground_truth,
        )

    elif ds == 1:
        # Malaga
        img_dir = os.path.join(
            paths.malaga, "malaga-urban-dataset-extract-07_rectified_800x600_Images"
        )
        left_images = sorted(glob(os.path.join(img_dir, "*_left.jpg")))
        if len(left_images) < 2:
            raise FileNotFoundError(
                f"Malaga left images not found under {img_dir} (expected *_left.jpg)."
            )
        # last_frame is inclusive index
        last_frame = len(left_images) - 1
        bootstrap_frames = (0, 7)
        ground_truth = _load_malaga_ground_truth(paths.malaga, left_images)
        K = np.array(
            [
                [621.18428, 0, 404.0076],
                [0, 621.18428, 309.05989],
                [0, 0, 1],
            ]
        )
        return DatasetInfo(
            ds=ds,
            K=K,
            last_frame=last_frame,
            bootstrap_frames=bootstrap_frames,
            ground_truth=ground_truth,
            left_images=left_images,
        )

    elif ds == 2:
        # Parking
        last_frame = 598
        bootstrap_frames = (0, 7)

        # Read K from K.txt (comma-separated, trailing commas handled via usecols)
        k_path = os.path.join(paths.parking, "K.txt")
        if not os.path.isfile(k_path):
            raise FileNotFoundError(f"Missing Parking intrinsics file: {k_path}")
        K = np.genfromtxt(
            k_path,
            delimiter=",",
            usecols=(0, 1, 2),
            dtype=float,
        )

        gt_path = os.path.join(paths.parking, "poses.txt")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Missing Parking ground truth file: {gt_path}")
        ground_truth = np.loadtxt(gt_path)
        ground_truth = ground_truth[:, [-9, -1]]
        if ground_truth.shape[0] > 0:
            last_frame = min(last_frame, ground_truth.shape[0] - 1)
            ground_truth = ground_truth[: last_frame + 1]

        return DatasetInfo(
            ds=ds,
            K=K,
            last_frame=last_frame,
            bootstrap_frames=bootstrap_frames,
            ground_truth=ground_truth,
        )

    elif ds == 3:
        # Own dataset
        last_frame = 442  # frames are 1-indexed: frame_000001.jpg to frame_000442.jpg
        bootstrap_frames = (1, 8)  # 1-indexed
        K = np.array(
            [
                [4119.30, 0, 2855.11],
                [0, 4116.30, 2132.56],
                [0, 0, 1],
            ]
        )
        return DatasetInfo(
            ds=ds,
            K=K,
            last_frame=last_frame,
            bootstrap_frames=bootstrap_frames,
            ground_truth=None,
        )

    else:
        raise ValueError(f"Invalid dataset index: {ds}")


def load_bootstrap_images(
    dataset: DatasetInfo,
    paths: DatasetPaths,
    bootstrap_frames: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the two bootstrap frames (img0, img1) as grayscale."""
    i0, i1 = bootstrap_frames
    ds = dataset.ds

    if ds == 0:
        img0 = cv2.imread(
            os.path.join(paths.kitti, "05", "image_0", f"{i0:06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        img1 = cv2.imread(
            os.path.join(paths.kitti, "05", "image_0", f"{i1:06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )

    elif ds == 1:
        if dataset.left_images is None:
            raise RuntimeError("left_images is not loaded for Malaga dataset.")
        img0 = cv2.imread(dataset.left_images[i0], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(dataset.left_images[i1], cv2.IMREAD_GRAYSCALE)

    elif ds == 2:
        img0 = cv2.imread(
            os.path.join(paths.parking, "images", f"img_{i0:05d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        img1 = cv2.imread(
            os.path.join(paths.parking, "images", f"img_{i1:05d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )

    elif ds == 3:
        img0 = cv2.imread(
            os.path.join(paths.own, "img", f"frame_{i0:06d}.jpg"), cv2.IMREAD_GRAYSCALE
        )
        img1 = cv2.imread(
            os.path.join(paths.own, "img", f"frame_{i1:06d}.jpg"), cv2.IMREAD_GRAYSCALE
        )

    else:
        raise ValueError(f"Invalid dataset index: {ds}")

    return img0, img1


def load_frame_image(
    dataset: DatasetInfo,
    paths: DatasetPaths,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Load a single frame by index as grayscale. Returns None if not found."""
    ds = dataset.ds

    if ds == 0:
        image_path = os.path.join(paths.kitti, "05", "image_0", f"{frame_idx:06d}.png")
    elif ds == 1:
        if dataset.left_images is None:
            raise RuntimeError("left_images is not loaded for Malaga dataset.")
        image_path = dataset.left_images[frame_idx]
    elif ds == 2:
        image_path = os.path.join(paths.parking, "images", f"img_{frame_idx:05d}.png")
    elif ds == 3:
        image_path = os.path.join(paths.own, "img", f"frame_{frame_idx:06d}.jpg")
    else:
        raise ValueError(f"Invalid dataset index: {ds}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: could not read {image_path}")
        return None

    return image
