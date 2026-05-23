import cv2
import numpy as np
from typing import Union
from .structure import VOState


def visualize_tracking(image, state: Union[VOState, dict]):
    """
    Visualize tracking results on the image.

    Args:
        image: Grayscale image to draw on.
        state: VOState object or legacy dictionary containing VO state.

    Returns:
        BGR image with visualization.
    """
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Support both VOState and legacy dict
    if isinstance(state, VOState):
        curr_indices = state.landmark_indices
        kp_curr = state.keypoints
        candidates = state.candidates
    else:
        curr_indices = state["landmark_indices"]
        kp_curr = state["keypoints_prev"]
        candidates = state["candidates"]

    # 1. Determine Data Type (SIFT vs KLT)
    is_numpy = isinstance(kp_curr, np.ndarray)

    # 2. Draw Tracked Keypoints
    for i, idx in enumerate(curr_indices):
        # Extract (x, y) depending on type
        if is_numpy:
            pt_vals = kp_curr[i]  # KLT: it's already [x, y]
            x, y = pt_vals[0], pt_vals[1]
        else:
            pt_vals = kp_curr[i].pt  # SIFT: it's a KeyPoint object
            x, y = pt_vals[0], pt_vals[1]

        pt = (int(x), int(y))

        if idx != -1:
            # Green: Successfully tracked landmark
            cv2.circle(vis_img, pt, 3, (0, 255, 0), -1)
        else:
            # Optional: Draw non-landmark keypoints (e.g. just detected) in another color?
            # For now, let's skip or draw small gray dot
            # cv2.circle(vis_img, pt, 2, (100, 100, 100), -1)
            pass

    # 3. Draw Candidates (Red)
    # Candidates are usually dicts with 'keypoint' which might be tuple or array
    for cand in candidates:
        c_kp = cand["keypoint"]

        # Candidate keypoints might also vary (tuple vs array)
        if isinstance(c_kp, (np.ndarray, list)):
            c_kp = np.array(c_kp).flatten()
            pt = (int(c_kp[0]), int(c_kp[1]))
        elif hasattr(c_kp, "pt"):  # KeyPoint object
            pt = (int(c_kp.pt[0]), int(c_kp.pt[1]))
        else:  # Tuple
            pt = (int(c_kp[0]), int(c_kp[1]))

        cv2.circle(vis_img, pt, 2, (0, 0, 255), -1)

    # 4. Info Text
    info = f"Landmarks: {np.sum(curr_indices != -1)} | Candidates: {len(candidates)}"
    cv2.putText(
        vis_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    return vis_img
