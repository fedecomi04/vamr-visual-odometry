# visualization.py
# functions to visualize keypoints, matches, 3D points, camera poses

from typing import Optional

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils.draw_camera import drawCamera

from initialization.structures import CameraIntrinsics

from .correspondences import compute_keypoint_correspondences_sift, compute_keypoint_correspondences_klt
from .geometry import estimate_relative_pose, triangulate_landmarks


def visualize_correspondences(
    img0: np.ndarray,
    img1: np.ndarray,
    method: str = "sift",
):
    """
    Draw correspondences between two images.

    Args:
        img0: First image.
        img1: Second image.
        method: Correspondence method ('sift' or 'klt').
    Returns:
        img_matches: Image showing matches.
    """
    if method == "sift":
        kp0, kp1, ds0, ds1, matches = compute_keypoint_correspondences_sift(
            img0, img1
        )
    elif method == "klt":
        kp0, kp1 = compute_keypoint_correspondences_klt(
            img0, img1
        )
        matches = [cv.DMatch(i, i, 0) for i in range(len(kp0))]
    else:
        raise ValueError(f"Unknown correspondence method: {method}")

    # Convert from numpy arrays to Keypoint objects
    kp0 = [cv.KeyPoint(float(p[0]), float(p[1]), 10) for p in kp0]
    kp1 = [cv.KeyPoint(float(p[0]), float(p[1]), 10) for p in kp1]

    remapped_matches = []
    for i in range(len(matches)):
        # cv.DMatch(_queryIdx, _trainIdx, _distance)
        new_match = cv.DMatch(i, i, matches[i].distance)
        remapped_matches.append(new_match)

    # Draw matches
    img_matches = cv.drawMatches(
        img0,
        kp0,
        img1,
        kp1,
        remapped_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img_matches)
    plt.savefig("images/correspondences_{}.png".format(method))
    plt.show()
    plt.close()


def visualize_3d_scene(
    img0: np.ndarray,
    img1: np.ndarray,
    kp0: np.ndarray,
    kp1: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    points_3d: np.ndarray,
    method: str = "sift"
):
    """
    Visualize 3D scene from two images by triangulating matched keypoints.

    Args:
        img0: First image.
        img1: Second image.
    """

    # Visualize 3D points
    fig = plt.figure()
    ax = fig.add_subplot(131, projection="3d")

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker="o")

    # Display camera pose
    drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale=2)
    ax.text(-0.1, -0.1, -0.1, "Cam 0")

    center_cam2_W = -R.T @ t
    center_cam2_W = center_cam2_W.reshape(
        3,
    )
    drawCamera(ax, center_cam2_W, R.T, length_scale=2)
    ax.text(
        center_cam2_W[0] - 0.1,
        center_cam2_W[1] - 0.1,
        center_cam2_W[2] - 0.1,
        "Cam 2",
    )

    # Display matched points
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(img0, cmap="gray")
    ax.scatter(kp0[:, 0], kp0[:, 1], color="cyan", marker="x", s=10)
    ax.set_title("Image 1")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img1, cmap="gray")
    ax.scatter(kp1[:, 0], kp1[:, 1], color="cyan", marker="x", s=10)
    ax.set_title("Image 2")

    plt.savefig("images/3d_scene_{}.png".format(method))
    plt.show()
    plt.close()
