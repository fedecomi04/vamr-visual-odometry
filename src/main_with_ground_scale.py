# main_with_ground_scale.py
"""
Main script with Ground Plane Scale Correction.

This version uses road masks to:
1. Initialize scale so camera height = 1.65m
2. Combat scale drift during continuous operation
"""

import json
from pathlib import Path

import cv2
import numpy as np

from continuous_operation.continuous_operation_klt import ContinuousOperationKLT
from continuous_operation.continuous_operation_sift import ContinuousOperationSIFT
from continuous_operation.structure import VOState
from continuous_operation.visualization import visualize_tracking
from data_loader import (
    DatasetPaths,
    load_bootstrap_images,
    load_dataset,
    load_frame_image,
)
from evaluation.trajectory_error import compute_rte
from evaluation.metrics import (
    compute_scale_error,
    compute_kitti_metrics,
    RuntimeTracker,
    print_evaluation_summary,
    save_evaluation_results,
)
from initialization import (
    VOInitializationResultKLT,
    VOInitializationResultSift,
    initialize_vo_klt,
    initialize_vo_sift,
    visualize_3d_scene,
    visualize_correspondences,
)
from initialization.structures import CameraIntrinsics
from ui_tools.vo_ui import update_ui
from utils.ground_plane_scale import (
    GroundPlaneConfig,
    GroundPlaneScaleEstimator,
    load_road_mask,
)
from utils.ground_plane_visualization import (
    visualize_ground_scale_info,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

UI_activate = True
GROUND_SCALE_ENABLED = True  # Enable/disable ground plane scale correction
GROUND_SCALE_VISUALIZATION = True  # Show ground points visualization

# Debug: deterministic per-N-frame signature to compare runs across entry points.
SIGNATURE_EVERY_N = 50


def _print_signature(
    frame_idx: int,
    state: VOState,
    ground_scale_factor: float,
    ground_result,
):
    C = (-np.asarray(state.R).T @ np.asarray(state.t)).reshape(3)
    kps = getattr(state, "keypoints", None)
    if kps is None:
        tracked = 0
    elif isinstance(kps, np.ndarray):
        tracked = int(np.asarray(kps, dtype=float).reshape(-1, 2).shape[0])
    else:
        tracked = int(len(kps))
    inliers = int(getattr(ground_result, "num_inliers", 0)) if ground_result is not None else 0
    C_round = tuple(float(f"{v:.3f}") for v in C.tolist())
    print(
        f"[Sig] i={int(frame_idx)} C={C_round} scale={float(ground_scale_factor):.4f} "
        f"tracked={tracked} ground_inliers={inliers}"
    )

# --- Setup ---
ds = 1  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
DATASET_NAMES = {0: "kitti", 1: "malaga", 2: "parking", 3: "own"}
dataset_name = DATASET_NAMES.get(ds, f"dataset_{ds}")

paths = DatasetPaths(
    kitti=str(DATA_ROOT / "kitti"),
    malaga=str(DATA_ROOT / "malaga-urban-dataset-extract-07"),
    parking=str(DATA_ROOT / "parking"),
    own=str(DATA_ROOT / "own_dataset"),
)

# Road mask directory (only for KITTI currently)
ROAD_MASK_DIR = str(DATA_ROOT / "kitti" / "road_masks")

# Load dataset info (K, ground_truth, last_frame, etc.)
dataset = load_dataset(ds, paths)

print("Camera matrix K:\n", dataset.K)
if dataset.ground_truth is not None:
    print("Ground truth shape:", dataset.ground_truth.shape)

# --- Ground Plane Scale Estimator Setup ---
ground_config = GroundPlaneConfig(
    enabled=GROUND_SCALE_ENABLED and ds == 0,  # Only for KITTI
    camera_height=1.65,  # KITTI camera height
    deque_maxlen=100,  # Reduced deque size
    ransac_thresh=0.1,  # Tighter threshold for better plane fit
    ransac_iterations=1000,  # More iterations for robustness
    min_ground_points=15,  # Require more points for robust plane fit
    correction_alpha=0.1,  # More gradual corrections
    height_tolerance=0.15,  # Slightly tighter tolerance
    correction_interval=10,  # Less frequent corrections
    min_plane_normal_y=0.7,  # Require more horizontal planes
)
ground_estimator = GroundPlaneScaleEstimator(ground_config)

# --- Bootstrap ---
bootstrap_frames = dataset.bootstrap_frames

img0, img1 = load_bootstrap_images(dataset, paths, bootstrap_frames)
intrinsics = CameraIntrinsics(dataset.K)

initialization_method = "klt"  # choose between "sift" and "klt"

# --- Hyperparameters ---
# Continuous operation hyperparameters
RANSAC_THRESH = 1.0
MAX_ITERS = 2000
ITERATIVE_TRIANGULATION = True
ENABLE_BA = True

# Store all hyperparameters for logging
hyperparameters = {
    "initialization_method": initialization_method,
    "bootstrap_frames": list(bootstrap_frames),
    "ransac_thresh": RANSAC_THRESH,
    "max_iters": MAX_ITERS,
    "iterative_triangulation": ITERATIVE_TRIANGULATION,
    "enable_ba": ENABLE_BA,
    # Ground plane specific hyperparameters
    "ground_scale_enabled": ground_config.enabled,
    "camera_height": ground_config.camera_height,
    "ground_deque_maxlen": ground_config.deque_maxlen,
    "ground_ransac_thresh": ground_config.ransac_thresh,
    "ground_ransac_iterations": ground_config.ransac_iterations,
    "ground_min_points": ground_config.min_ground_points,
    "ground_correction_alpha": ground_config.correction_alpha,
    "ground_height_tolerance": ground_config.height_tolerance,
    "ground_correction_interval": ground_config.correction_interval,
}

# (Here you would normally initialize your VO / tracking with img0, img1)
if initialization_method == "sift":
    vo_init_result: VOInitializationResultSift = initialize_vo_sift(
        img0,
        img1,
        intrinsics,
    )

elif initialization_method == "klt":
    vo_init_result: VOInitializationResultKLT = initialize_vo_klt(
        img0,
        img1,
        intrinsics,
    )

else:
    raise ValueError(f"Unknown initialization method: {initialization_method}")

print("\nVO Initialization Result using {}:".format(initialization_method))
print("Rotation R:\n", vo_init_result.R)
print("Translation t:\n", vo_init_result.t)
print("Number of 3D points:", vo_init_result.points_3d.shape[0])

R = vo_init_result.R
t = vo_init_result.t
points_3d = vo_init_result.points_3d
keypoints0 = vo_init_result.keypoints0
keypoints1 = vo_init_result.keypoints1

if initialization_method == "sift":
    descriptors0 = vo_init_result.descriptors0
    descriptors1 = vo_init_result.descriptors1

# --- Ground Plane Scale Initialization ---
if ground_config.enabled:
    print("\n--- Ground Plane Scale Initialization ---")

    # Load road mask for bootstrap frame 1
    road_mask = load_road_mask(ROAD_MASK_DIR, bootstrap_frames[1])

    if road_mask is not None:
        print(f"Road mask loaded: shape={road_mask.shape}, dtype={road_mask.dtype}")
        print(
            f"Road mask stats: min={road_mask.min()}, max={road_mask.max()}, "
            f"road_pixels={np.sum(road_mask > 0)}, unique={np.unique(road_mask)}"
        )

        # Create landmark indices (all points have landmarks at init)
        init_landmark_indices = np.arange(len(points_3d), dtype=int)

        # Debug: check keypoint locations vs road mask
        print(
            f"Keypoints1 shape: {keypoints1.shape}, range: x=[{keypoints1[:, 0].min():.1f}, {keypoints1[:, 0].max():.1f}], "
            f"y=[{keypoints1[:, 1].min():.1f}, {keypoints1[:, 1].max():.1f}]"
        )

        # Check how many keypoints fall on road
        on_road_count = 0
        for kp in keypoints1:
            x, y = int(round(kp[0])), int(round(kp[1]))
            if 0 <= y < road_mask.shape[0] and 0 <= x < road_mask.shape[1]:
                if road_mask[y, x] > 0:
                    on_road_count += 1
        print(
            f"Keypoints on road mask (before 3D filtering): {on_road_count} / {len(keypoints1)}"
        )

        # Debug: filter ground landmarks first to see statistics
        ground_pts, ground_kp_2d, _ = ground_estimator.filter_ground_landmarks(
            landmarks_3d=points_3d,
            keypoints_2d=keypoints1,
            landmark_indices=init_landmark_indices,
            road_mask=road_mask,
            K=dataset.K,
            R=R,
            t=t,
        )
        print(
            f"Ground points after 3D filtering: {len(ground_pts)} / {len(points_3d)} total landmarks"
        )

        # Debug: show ground point 3D positions
        if len(ground_pts) > 0:
            print(f"\n  DEBUG: Ground points 3D statistics:")
            print(
                f"    World coords: X=[{ground_pts[:, 0].min():.2f}, {ground_pts[:, 0].max():.2f}], "
                f"Y=[{ground_pts[:, 1].min():.2f}, {ground_pts[:, 1].max():.2f}], "
                f"Z=[{ground_pts[:, 2].min():.2f}, {ground_pts[:, 2].max():.2f}]"
            )

        # If still 0, check the 3D filtering criteria
        if len(ground_pts) == 0 and on_road_count > 0:
            print("\n  DEBUG: Points on road but filtered out by 3D criteria:")
            # Check which points are on road and why they're filtered
            valid_mask = init_landmark_indices != -1
            valid_kp_2d = keypoints1[valid_mask]
            valid_landmark_ids = init_landmark_indices[valid_mask]

            for idx, (kp, lid) in enumerate(
                zip(valid_kp_2d[:10], valid_landmark_ids[:10])
            ):  # Check first 10
                x, y = int(round(kp[0])), int(round(kp[1]))
                if 0 <= y < road_mask.shape[0] and 0 <= x < road_mask.shape[1]:
                    is_road = road_mask[y, x] > 0
                    if is_road:
                        pt_3d = points_3d[lid]
                        X_cam = (R @ pt_3d.reshape(3, 1) + t).flatten()
                        print(f"    KP({x}, {y}): 3D_world={pt_3d}, 3D_cam={X_cam}")
                        print(
                            f"      Z_cam={X_cam[2]:.2f} (need >0.5 and <100), Y_cam={X_cam[1]:.2f} (need >0 for below camera)"
                        )

        # Initialize scale
        scale_factor, points_3d, t = ground_estimator.initialize_scale(
            landmarks_3d=points_3d,
            keypoints_2d=keypoints1,
            landmark_indices=init_landmark_indices,
            road_mask=road_mask,
            K=dataset.K,
            R=R,
            t=t,
        )

        print(f"Applied initial scale factor: {scale_factor:.4f}")
        print(f"Deque size after init: {len(ground_estimator.ground_points_deque)}")
    else:
        print("Warning: Road mask not found for bootstrap frame, skipping scale init")

# Visualize
visualize_3d_scene(
    img0=img0,
    img1=img1,
    kp0=keypoints0,
    kp1=keypoints1,
    R=R,
    t=t,
    points_3d=points_3d,
    method=initialization_method,
)

visualize_correspondences(
    img0=img0,
    img1=img1,
    method=initialization_method,
)

print("\nVO Initialization and Visualization complete.")


# Initialize State

# Build landmark tracking info
landmark_tracking = []
R0 = np.eye(3)
t0 = np.zeros((3, 1))

for i in range(len(points_3d)):
    landmark_tracking.append(
        {"first_keypoint": tuple(keypoints0[i]), "first_R": R0, "first_t": t0}
    )
initial_landmark_indices = np.arange(len(points_3d), dtype=int)

# Convert keypoints to appropriate format
if initialization_method == "klt":
    keypoints_for_state = np.float32(keypoints1)
else:  # sift - convert to cv2.KeyPoint objects
    keypoints_for_state = [
        cv2.KeyPoint(x=pt[0], y=pt[1], size=1.0) for pt in keypoints1
    ]

# Create VOState object
state = VOState(
    image=img1.copy(),
    R=R,
    t=t,
    keypoints=keypoints_for_state,
    landmarks_3d=points_3d,
    landmarks_tracks=landmark_tracking,
    landmark_indices=initial_landmark_indices,
    candidates=[],
    trajectory=[(-R.T @ t).flatten()],
    landmark_counts=[],
    axis_limits=None,
    descriptors=descriptors1 if initialization_method == "sift" else None,
    ground_plane_estimator=ground_estimator if ground_config.enabled else None,
    last_ground_result=None,
)

# --- Continuous operation ---

# Initialize the continuous operation module before the loop
if initialization_method == "klt":
    co_module = ContinuousOperationKLT(
        intrinsics=intrinsics,
        ransac_thresh=RANSAC_THRESH,
        max_iters=MAX_ITERS,
        iterative_triangulation=ITERATIVE_TRIANGULATION,
        enable_ba=ENABLE_BA,
    )
else:  # sift
    co_module = ContinuousOperationSIFT(
        intrinsics=intrinsics,
        ransac_thresh=RANSAC_THRESH,
        max_iters=MAX_ITERS,
    )

# Initialize runtime tracker
runtime_tracker = RuntimeTracker()

for i in range(bootstrap_frames[1] + 1, dataset.last_frame + 1):
    print(f"\n\nProcessing frame {i}/{dataset.last_frame}\n=====================")

    # Start timing this frame
    runtime_tracker.start_frame()

    image = load_frame_image(dataset, paths, i)

    # --- Use previously computed scale factor for this frame's motion ---
    # The scale factor was computed at the end of the previous frame iteration
    # using the updated deque (which includes the latest candidates)
    ground_scale_factor = getattr(ground_estimator, "next_frame_scale_factor", 1.0)
    ground_result = getattr(ground_estimator, "last_estimation_result", None)

    if ground_config.enabled and ground_result is not None and ground_result.is_valid:
        print(
            f"  [GroundPlane PRE] Scale factor: {ground_scale_factor:.4f} | "
            f"Height: {ground_result.camera_height:.3f}m | "
            f"Inliers: {ground_result.num_inliers}"
        )

    prev_state = state
    state = co_module.process_frame(
        image, prev_state, ground_scale_factor=ground_scale_factor
    )

    # End timing this frame
    runtime_tracker.end_frame()

    # --- Accumulate Ground Points to Deque (AFTER process_frame) ---
    # Now that we have the new pose and landmarks, find ground points and add to deque
    # THEN estimate the plane from the updated deque for the NEXT frame
    ground_result = None
    ground_points_px = None
    if ground_config.enabled:
        road_mask = load_road_mask(ROAD_MASK_DIR, i)

        if road_mask is not None:
            # Debug: count ground points before accumulation
            n_deque_before = len(ground_estimator.ground_points_deque)

            # Filter ground landmarks and add to deque (no correction, just accumulation)
            ground_pts, ground_kp_2d, ground_indices = (
                ground_estimator.filter_ground_landmarks(
                    landmarks_3d=state.landmarks_3d,
                    keypoints_2d=state.keypoints,
                    landmark_indices=state.landmark_indices,
                    road_mask=road_mask,
                    K=dataset.K,
                    R=state.R,
                    t=state.t,
                )
            )
            ground_points_px = ground_kp_2d if len(ground_kp_2d) > 0 else None

            # Add to deque
            ground_estimator.add_ground_points_to_deque(ground_pts)
            ground_estimator.last_candidate_points = (
                ground_pts if len(ground_pts) > 0 else None
            )
            ground_estimator.last_candidate_2d = (
                ground_kp_2d if len(ground_kp_2d) > 0 else None
            )

            # Debug: ground plane statistics
            # Note: when deque is full, n_added = 0 but points ARE being added (old ones removed)
            n_candidates = len(ground_pts)
            n_deque_after = len(ground_estimator.ground_points_deque)
            n_added = (
                min(
                    n_candidates,
                    ground_config.deque_maxlen - n_deque_before + n_candidates,
                )
                if n_deque_before < ground_config.deque_maxlen
                else n_candidates
            )

            print(
                f"  [GroundPlane POST] Candidates: {n_candidates:3d} | "
                f"Added/Replaced: {n_candidates:3d} | "
                f"Deque size: {n_deque_after:4d}/{ground_config.deque_maxlen}"
            )

            # DEBUG: Show ground point statistics and camera position
            if n_candidates > 0:
                cam_center = (-state.R.T @ state.t).flatten()
                print(
                    f"    [DEBUG] Camera center: X={cam_center[0]:.2f}, Y={cam_center[1]:.2f}, Z={cam_center[2]:.2f}"
                )
                print(
                    f"    [DEBUG] New ground pts Y: min={ground_pts[:, 1].min():.2f}, max={ground_pts[:, 1].max():.2f}, mean={ground_pts[:, 1].mean():.2f}"
                )

                deque_pts = ground_estimator.get_deque_points()
                if len(deque_pts) > 0:
                    print(
                        f"    [DEBUG] Deque pts Y: min={deque_pts[:, 1].min():.2f}, max={deque_pts[:, 1].max():.2f}, mean={deque_pts[:, 1].mean():.2f}"
                    )
                    print(
                        f"    [DEBUG] Deque pts Z: min={deque_pts[:, 2].min():.2f}, max={deque_pts[:, 2].max():.2f}"
                    )

            # NOW estimate scale factor from the UPDATED deque (includes new candidates)
            # This scale factor will be used for the NEXT frame
            ground_result = ground_estimator.estimate_scale_factor_only(
                R=state.R,
                t=state.t,
                force_estimation=False,  # Only estimate at correction intervals
            )

            # Store for next frame
            if ground_result is not None and ground_result.is_valid:
                ground_estimator.next_frame_scale_factor = ground_result.scale_factor
                ground_estimator.last_estimation_result = ground_result
            elif ground_result is not None and ground_result.skipped:
                # Keep using the previous scale factor
                pass
            else:
                # Estimation failed, reset to 1.0
                ground_estimator.next_frame_scale_factor = 1.0
                ground_estimator.last_estimation_result = ground_result

            # Store result for visualization
            state.last_ground_result = ground_result
        else:
            print(f"  [GroundPlane] Road mask not found for frame {i}")

    # Update trajectory (camera center in world) and landmark counts
    cam_center = (-state.R.T @ state.t).flatten()
    state.trajectory.append(cam_center)
    active_landmarks = np.count_nonzero(state.landmark_indices != -1)
    state.landmark_counts.append(active_landmarks)

    # Prepare data for UI (last 20 frames)
    full_trajectory_raw = np.array(state.trajectory)
    last20_slice = slice(
        max(0, len(full_trajectory_raw) - 20), len(full_trajectory_raw)
    )
    local_traj_last20_raw = full_trajectory_raw[last20_slice]

    if state.landmarks_3d.size > 0:
        # Use only landmarks currently observed in this frame (green points),
        # i.e. those referenced in landmark_indices, and keep only those
        # that are in front of the current camera.
        landmark_ids = state.landmark_indices
        visible_mask = landmark_ids != -1
        if np.any(visible_mask):
            unique_ids = np.unique(landmark_ids[visible_mask])
            landmarks_world = state.landmarks_3d[unique_ids]
            X_cam = (state.R @ landmarks_world.T + state.t).T
            front_mask = X_cam[:, 2] > 0
            visible_world = landmarks_world[front_mask]
            # Range-gate landmarks for visualization to suppress extreme outliers.
            cam_center = (-state.R.T @ state.t).reshape(3)
            d = np.linalg.norm(visible_world - cam_center.reshape(1, 3), axis=1)
            visible_world = visible_world[d <= 350.0]
            local_landmarks_last20 = (
                visible_world[:, [0, 2]] if visible_world.size > 0 else np.empty((0, 2))
            )
        else:
            local_landmarks_last20 = np.empty((0, 2))
    else:
        local_landmarks_last20 = np.empty((0, 2))

    # Scale factor for visualization (using ground truth if available, else 1.0)
    if dataset.ground_truth is not None and full_trajectory_raw.shape[0] > 1:
        gt = dataset.ground_truth
        gt_idx = min(i, gt.shape[0] - 1)
        gt_segment = gt[: gt_idx + 1]
        est_xy = full_trajectory_raw[:, [0, 2]]
        len_est = np.linalg.norm(est_xy[-1] - est_xy[0])
        len_gt = np.linalg.norm(gt_segment[-1] - gt_segment[0])
        if len_est > 1e-6:
            scale = float(len_gt / len_est)
        else:
            scale = 1.0
    else:
        scale = 1.0

    # Apply scale to trajectories and landmarks for visualization
    full_trajectory = scale * full_trajectory_raw[:, [0, 2]]
    local_traj_last20 = scale * local_traj_last20_raw[:, [0, 2]]
    local_landmarks_last20_scaled = scale * local_landmarks_last20

    # Visualization image with landmarks/candidates overlaid
    vis = visualize_tracking(image, state)

    # --- Ground Plane Visualization ---
    if GROUND_SCALE_VISUALIZATION and ground_config.enabled:
        vis = visualize_ground_scale_info(
            image=vis,
            ground_estimator=ground_estimator,
            K=dataset.K,
            R=state.R,
            t=state.t,
            result=ground_result,
            show_legend=True,
            show_plane_grid=False,  # Set True to see plane grid overlay
        )

    # Update axis limits for UI based on the 20-frame window (scaled),
    # centering on the first frame of the window, but only every 50 frames
    # to avoid excessive jitter.
    if UI_activate:
        if state.axis_limits is None or i % 50 == 0:
            traj_window = local_traj_last20
            anchor = traj_window[0]

            # Distances of window trajectory from anchor
            dx = traj_window[:, 0] - anchor[0]
            dz = traj_window[:, 1] - anchor[1]
            r_traj = np.sqrt(dx**2 + dz**2).max() if len(traj_window) > 0 else 0.0

            # Also account for local landmarks, if any
            if local_landmarks_last20_scaled.size > 0:
                dx_l = local_landmarks_last20_scaled[:, 0] - anchor[0]
                dz_l = local_landmarks_last20_scaled[:, 1] - anchor[1]
                r_lm = np.sqrt(dx_l**2 + dz_l**2).max()
            else:
                r_lm = 0.0

            radius = max(r_traj, r_lm, 1e-3)
            margin = 0.3 * radius
            r_total = radius + margin

            xmin = anchor[0] - r_total
            xmax = anchor[0] + r_total
            zmin = anchor[1] - r_total
            zmax = anchor[1] + r_total

            state.axis_limits = (xmin, xmax, zmin, zmax)

    if UI_activate:
        update_ui(
            image=vis,
            tracked_landmarks_count=state.landmark_counts[-20:],
            full_trajectory=full_trajectory,
            local_traj_last20=local_traj_last20,
            local_landmarks_last20=local_landmarks_last20_scaled,
            axis_limits=state.axis_limits,
            rte_errors=None,
            ground_points_px=ground_points_px,
            full_trajectory_gt=dataset.ground_truth,
        )
    else:
        cv2.imshow("Tracking", vis)
        cv2.waitKey(10)

    if SIGNATURE_EVERY_N > 0 and (i % SIGNATURE_EVERY_N == 0):
        _print_signature(
            frame_idx=i,
            state=state,
            ground_scale_factor=ground_scale_factor,
            ground_result=ground_result,
        )


# --- Trajectory evaluation (Relative Trajectory Error) ---
# Get runtime statistics
runtime_stats = runtime_tracker.get_statistics()
runtime_tracker.print_summary()

# Initialize result containers
rte_results = None
scale_results = None
kitti_results = None

if dataset.ground_truth is not None:
    print("\n" + "=" * 60)
    print("COMPUTING EVALUATION METRICS...")
    print("=" * 60)

    traj_est = np.array(state.trajectory)  # Nx3 positions (camera centers in world)
    traj_gt = dataset.ground_truth  # typically Nx2 or Nx3 positions

    try:
        # 1. Relative Trajectory Error (RTE)
        print("\n[1/3] Computing Relative Trajectory Error (RTE)...")
        rte_results = compute_rte(traj_est, traj_gt, delta=1, align_scale=True)
        print(f"  RTE mean:   {rte_results['mean_rte']:.4f} m")
        print(f"  RTE median: {rte_results['median_rte']:.4f} m")
        print(f"  RTE RMSE:   {rte_results['rmse_rte']:.4f} m")

        # 2. Scale Error
        print("\n[2/3] Computing Scale Error...")
        scale_results = compute_scale_error(traj_est, traj_gt, window_size=50)
        print(f"  Optimal scale:   {scale_results['optimal_scale']:.4f}")
        print(f"  Scale drift:     {scale_results['scale_drift_percent']:.2f}%")

        # 3. KITTI Benchmark Metrics (adaptive segment lengths based on trajectory)
        print("\n[3/3] Computing KITTI Benchmark Metrics...")

        # Estimate trajectory length to choose appropriate segment lengths
        traj_length = np.sum(np.linalg.norm(np.diff(traj_gt, axis=0), axis=1))
        print(f"  Trajectory length: {traj_length:.1f}m")

        # Adapt segment lengths based on trajectory length
        if traj_length >= 800:
            # Full KITTI benchmark segments
            seg_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        elif traj_length >= 100:
            # Use segments up to trajectory length
            seg_lengths = [
                s for s in [100, 200, 300, 400, 500] if s <= traj_length * 0.8
            ]
        else:
            # Short trajectory - use smaller segments
            seg_lengths = [s for s in [10, 20, 30, 50, 75] if s <= traj_length * 0.8]

        if seg_lengths:
            kitti_results = compute_kitti_metrics(
                traj_est,
                traj_gt,
                segment_lengths=seg_lengths,
            )
        else:
            kitti_results = {"translation_error_percent": None, "num_total_segments": 0}

        trans_err = kitti_results.get("translation_error_percent")
        if trans_err is not None:
            print(f"  Translation error: {trans_err:.2f}%")
        else:
            print("  Translation error: N/A (trajectory too short)")

        # Print comprehensive summary
        print_evaluation_summary(
            rte_results=rte_results,
            scale_results=scale_results,
            kitti_results=kitti_results,
            runtime_stats=runtime_stats,
        )

        # Save all results to JSON with timestamp and dataset name
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp and dataset name
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = Path(__file__).name
        eval_filename = f"eval_ground_scale_{dataset_name}_{timestamp}.json"
        eval_results_path = results_dir / eval_filename

        save_evaluation_results(
            str(eval_results_path),
            rte_results=rte_results,
            scale_results=scale_results,
            kitti_results=kitti_results,
            runtime_stats=runtime_stats,
            script_name=script_name,
            dataset_name=dataset_name,
            hyperparameters=hyperparameters,
        )

        # Also save RTE separately for backward compatibility
        rte_path = results_dir / "rte.json"
        with rte_path.open("w") as f:
            json.dump(
                {
                    "mean_rte": rte_results["mean_rte"],
                    "median_rte": rte_results["median_rte"],
                    "rmse_rte": rte_results["rmse_rte"],
                },
                f,
                indent=2,
            )
        print(f"RTE results saved to {rte_path}")

        # Show final RTE curve in the same UI window and keep it open
        if UI_activate:
            from ui_tools.vo_ui import update_ui as _update_ui_final

            # reuse last visualization state if available
            full_trajectory_raw = np.array(state.trajectory)
            full_trajectory = full_trajectory_raw[:, [0, 2]]
            last20_slice = slice(
                max(0, len(full_trajectory_raw) - 20), len(full_trajectory_raw)
            )
            local_traj_last20 = full_trajectory[last20_slice]

            # landmarks for the last frame (best-effort reuse)
            if state.landmarks_3d.size > 0:
                landmark_ids = state.landmark_indices
                visible_mask = landmark_ids != -1
                if np.any(visible_mask):
                    unique_ids = np.unique(landmark_ids[visible_mask])
                    landmarks_world = state.landmarks_3d[unique_ids]
                    X_cam = (state.R @ landmarks_world.T + state.t).T
                    front_mask = X_cam[:, 2] > 0
                    visible_world = landmarks_world[front_mask]
                    local_landmarks_last20 = (
                        visible_world[:, [0, 2]]
                        if visible_world.size > 0
                        else np.empty((0, 2))
                    )
                else:
                    local_landmarks_last20 = np.empty((0, 2))
            else:
                local_landmarks_last20 = np.empty((0, 2))

            _update_ui_final(
                image=vis,
                tracked_landmarks_count=state.landmark_counts[-20:],
                full_trajectory=full_trajectory,
                local_traj_last20=local_traj_last20,
                local_landmarks_last20=local_landmarks_last20,
                axis_limits=state.axis_limits,
                rte_errors=rte_results["per_pair_errors"] if rte_results else None,
                full_trajectory_gt=dataset.ground_truth,
            )

            print(
                "Press close on the matplotlib window or Ctrl+C in the terminal to exit."
            )
            import matplotlib.pyplot as plt

            plt.show(block=True)
    except Exception as e:
        print(f"Could not compute evaluation metrics: {e}")
        import traceback

        traceback.print_exc()

else:
    # No ground truth available - still show final visualization and runtime stats
    print_evaluation_summary(
        rte_results=None,
        scale_results=None,
        kitti_results=None,
        runtime_stats=runtime_stats,
    )

    # Save runtime stats even without ground truth
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp and dataset name
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = Path(__file__).name
    eval_filename = f"eval_ground_scale_{dataset_name}_{timestamp}.json"

    save_evaluation_results(
        str(results_dir / eval_filename),
        rte_results=None,
        scale_results=None,
        kitti_results=None,
        runtime_stats=runtime_stats,
        script_name=script_name,
        dataset_name=dataset_name,
        hyperparameters=hyperparameters,
    )

    if UI_activate:
        print("\nNo ground truth available. Showing final trajectory visualization...")

        full_trajectory_raw = np.array(state.trajectory)
        full_trajectory = full_trajectory_raw[:, [0, 2]]
        last20_slice = slice(
            max(0, len(full_trajectory_raw) - 20), len(full_trajectory_raw)
        )
        local_traj_last20 = full_trajectory[last20_slice]

        # landmarks for the last frame
        if state.landmarks_3d.size > 0:
            landmark_ids = state.landmark_indices
            visible_mask = landmark_ids != -1
            if np.any(visible_mask):
                unique_ids = np.unique(landmark_ids[visible_mask])
                landmarks_world = state.landmarks_3d[unique_ids]
                X_cam = (state.R @ landmarks_world.T + state.t).T
                front_mask = X_cam[:, 2] > 0
                visible_world = landmarks_world[front_mask]
                local_landmarks_last20 = (
                    visible_world[:, [0, 2]]
                    if visible_world.size > 0
                    else np.empty((0, 2))
                )
            else:
                local_landmarks_last20 = np.empty((0, 2))
        else:
            local_landmarks_last20 = np.empty((0, 2))

        update_ui(
            image=vis,
            tracked_landmarks_count=state.landmark_counts[-20:],
            full_trajectory=full_trajectory,
            local_traj_last20=local_traj_last20,
            local_landmarks_last20=local_landmarks_last20,
            axis_limits=state.axis_limits,
            rte_errors=None,
        )

        print("Press close on the matplotlib window or Ctrl+C in the terminal to exit.")
        import matplotlib.pyplot as plt

        plt.show(block=True)
