# main.py

import argparse
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
try:
    from evaluation.trajectory_error import compute_rte
except ModuleNotFoundError:
    compute_rte = None
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

from loop_closing.correction import (
    _camera_center_from_Twc_sim3,
    apply_world_warp_to_landmarks,
    compute_loop_sim3_measurement,
    compute_world_warp_from_anchor_keyframe,
)
from loop_closing.feature_selection import get_vo_selected_features
from loop_closing.keyframes import KeyframeManager
from loop_closing.loop_detector import find_loop_closure
from loop_closing.place_recognition import DescriptorDatabase
from loop_closing.pose_graph import PoseGraph
from loop_closing.trajectory_warp import compute_frame_corrections
from loop_closing.world_reset import (
    apply_world_correction_to_Rt,
    apply_world_correction_to_Twc_list,
    apply_world_correction_to_keyframes,
    apply_world_correction_to_tracks_and_candidates,
    compute_world_correction_from_anchor,
    decompose_sim3_matrix,
    sim3_to_points,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
LOOP_LOG_DIR = PROJECT_ROOT / "data_loopclosure"
LOOP_LOG_PATH = LOOP_LOG_DIR / "loops.txt"

UI_activate = True
APPLY_MAP_WARP = False  # Deprecated in favor of WORLD_RESET_AFTER_LOOP.
WORLD_RESET_AFTER_LOOP = True
PLOT_SINGLE_TRAJECTORY = False
loop_closing_active = False

# --- Runtime options ---
_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument(
    "--start-frame",
    type=int,
    default=0,
    help="Frame index to start VO from (bootstrap uses start_frame and start_frame+bootstrap_gap).",
)
_parser.add_argument(
    "--bootstrap-gap",
    type=int,
    default=4,
    help="Gap between the two bootstrap frames.",
)
_parser.add_argument(
    "--no-bootstrap-viz",
    action="store_true",
    help="Disable bootstrap visualizations (3D scene + correspondences).",
)
_args = _parser.parse_args()

START_FRAME = int(_args.start_frame)
BOOTSTRAP_GAP = int(_args.bootstrap_gap)
BOOTSTRAP_VIZ = not bool(getattr(_args, "no_bootstrap_viz", False))
UI_activate = False if bool(getattr(_args, "no_ui", False)) else UI_activate
STOP_AFTER_FIRST_FRAME = bool(getattr(_args, "stop_after_first_frame", False))

if START_FRAME < 0:
    raise ValueError("--start-frame must be >= 0")
if BOOTSTRAP_GAP <= 0:
    raise ValueError("--bootstrap-gap must be > 0")


def _append_loop_log(loopedge: int, loop_sim3: str | None, posegraph: str | None):
    LOOP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = (
        f"loopedge={int(loopedge)} | "
        f"{loop_sim3 or '[LoopSim3] None'} | "
        f"{posegraph or '[PoseGraph] None'}\n"
    )
    with LOOP_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)

# --- Setup ---
ds = 0  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
DATASET_NAMES = {0: "kitti", 1: "malaga", 2: "parking", 3: "own"}
dataset_name = DATASET_NAMES.get(ds, f"dataset_{ds}")

paths = DatasetPaths(
    kitti=str(DATA_ROOT / "kitti"),
    malaga=str(DATA_ROOT / "malaga-urban-dataset-extract-07"),
    parking=str(DATA_ROOT / "parking"),
    own=str(DATA_ROOT / "own_dataset"),
)

# Load dataset info (K, ground_truth, last_frame, etc.)
dataset = load_dataset(ds, paths)

print("Camera matrix K:\n", dataset.K)
if dataset.ground_truth is not None:
    print("Ground truth shape:", dataset.ground_truth.shape)

# --- Bootstrap ---
bootstrap_frames = (START_FRAME, START_FRAME + BOOTSTRAP_GAP)
if bootstrap_frames[1] > dataset.last_frame:
    raise ValueError(
        f"Bootstrap frames {bootstrap_frames} exceed last_frame={dataset.last_frame}. "
        "Reduce --start-frame or --bootstrap-gap."
    )

img0, img1 = load_bootstrap_images(dataset, paths, bootstrap_frames)
intrinsics = CameraIntrinsics(dataset.K)

initialization_method = "klt"  # choose between "sift" and "klt"

# --- Hyperparameters ---
# Continuous operation hyperparameters
RANSAC_THRESH = 0.9
MAX_ITERS = 2000
ITERATIVE_TRIANGULATION = False
ENABLE_BA = False

# Store all hyperparameters for logging
hyperparameters = {
    "initialization_method": initialization_method,
    "bootstrap_frames": list(bootstrap_frames),
    "ransac_thresh": RANSAC_THRESH,
    "max_iters": MAX_ITERS,
    "iterative_triangulation": ITERATIVE_TRIANGULATION,
    "enable_ba": ENABLE_BA,
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

# Visualize (blocking, optional)
if BOOTSTRAP_VIZ:
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
)

# --- Loop closing state (SIFT only for now) ---
kfm = KeyframeManager(min_frame_gap=100, min_translation=5.0, min_rotation_deg=5.0)
descriptor_db = DescriptorDatabase()
pose_graph = PoseGraph(odom_weight=1.0, loop_weight=3.0)
loops_detected = 0
last_loop_text = None
last_keyframe_features = 0
loop_sim3_text = None
pg_opt_text = None
map_warp_text = None

MAX_KF_FEATURES = 800

# --- Pose storage for corrected trajectory display ---
poses_wc = []
trajectory_corrected = []

# Seed pose lists with bootstrap second frame (img1 / frame bootstrap_frames[1])
T_wc0 = np.eye(4, dtype=np.float64)
T_wc0[:3, :3] = np.asarray(state.R, dtype=np.float64).reshape(3, 3)
T_wc0[:3, 3] = np.asarray(state.t, dtype=np.float64).reshape(3)
poses_wc.append(T_wc0)
cam_center0_corr = _camera_center_from_Twc_sim3(T_wc0)
trajectory_corrected.append(cam_center0_corr)

prev_kf = None
keyframes_created = []

# Essential graph: add covisibility edges if >= this many shared landmarks.
COVIS_MIN_SHARED = 30


def _count_shared_landmarks(kf_a, kf_b) -> int:
    a = np.asarray(getattr(kf_a, "landmark_ids", []), dtype=np.int64).reshape(-1)
    b = np.asarray(getattr(kf_b, "landmark_ids", []), dtype=np.int64).reshape(-1)
    a = a[a != -1]
    b = b[b != -1]
    if a.size == 0 or b.size == 0:
        return 0
    a = np.unique(a)
    b = np.unique(b)
    return int(np.intersect1d(a, b, assume_unique=True).size)

def _recompute_corrected_trajectory():
    global pg_opt_text
    # Always compute a pose-graph-consistent corrected trajectory by warping per-frame VO poses
    # using optimized keyframe corrections (interpolated between surrounding keyframes).
    poses_corr, centers_corr, warp_stats = compute_frame_corrections(
        frames_T_wc_vo=poses_wc,
        keyframes=keyframes_created,
        pose_graph=pose_graph,
        base_frame_idx=int(bootstrap_frames[1]),
    )
    trajectory_corrected.clear()
    trajectory_corrected.extend([np.asarray(c, dtype=np.float64) for c in centers_corr])
    if warp_stats.num_keyframe_checks > 0:
        extra = (
            f" warp_err max={warp_stats.max_center_error_at_kf:.2f} "
            f"mean={warp_stats.mean_center_error_at_kf:.2f}"
        )
        pg_opt_text = (pg_opt_text or "[PoseGraph]") + extra


if initialization_method in {"sift", "klt"}:
    pts0, des0, lm0 = get_vo_selected_features(
        state, mode=initialization_method, image_gray=state.image
    )
    if des0 is not None and len(des0) > MAX_KF_FEATURES:
        idx = np.random.choice(len(des0), size=MAX_KF_FEATURES, replace=False)
        pts0 = pts0[idx]
        des0 = des0[idx]
        lm0 = lm0[idx]
    last_keyframe_features = int(0 if des0 is None else len(des0))
    if des0 is not None and len(des0) > 0 and len(pts0) == len(des0):
        kf0 = kfm.add_keyframe(bootstrap_frames[1], state.R, state.t, pts0, des0, lm0)
        descriptor_db.add_keyframe(kf0)
        pose_graph.add_node_from_keyframe(kf0)
        prev_kf = kf0
        keyframes_created.append(kf0)
        _recompute_corrected_trajectory()

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
    # for i in range(2200, dataset.last_frame + 1):
    print(f"\n\nProcessing frame {i}/{dataset.last_frame}\n=====================")

    # Start timing this frame
    runtime_tracker.start_frame()

    image = load_frame_image(dataset, paths, i)

    prev_state = state
    state = co_module.process_frame(image, prev_state)

    # End timing this frame
    runtime_tracker.end_frame()

    # End timing this frame
    runtime_tracker.end_frame()

    # Update pose list + trajectories (camera center in world)
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = np.asarray(state.R, dtype=np.float64).reshape(3, 3)
    T_wc[:3, 3] = np.asarray(state.t, dtype=np.float64).reshape(3)
    poses_wc.append(T_wc)

    cam_center = (-state.R.T @ state.t).flatten()
    state.trajectory.append(cam_center)

    # Corrected trajectory is recomputed whenever keyframes/optimization update.
    # For incremental frames, append raw center (will be warped on next recompute).
    trajectory_corrected.append(_camera_center_from_Twc_sim3(T_wc))
    if loop_closing_active:
        _recompute_corrected_trajectory()

    active_landmarks = np.count_nonzero(state.landmark_indices != -1)
    state.landmark_counts.append(active_landmarks)

    # Keyframe creation + loop detection (SIFT only)
    new_keyframe_created = False
    if kfm.should_create_keyframe(i, state.R, state.t):
        mode = "sift" if initialization_method == "sift" else "klt"
        pts, des, landmark_ids = get_vo_selected_features(
            state, mode=mode, image_gray=state.image
        )
        if des is not None and len(des) > MAX_KF_FEATURES:
            idx = np.random.choice(len(des), size=MAX_KF_FEATURES, replace=False)
            pts = pts[idx]
            des = des[idx]
            landmark_ids = landmark_ids[idx]
        if des is not None and len(des) > 0 and len(pts) == len(des):
            last_keyframe_features = int(len(des))
            kf = kfm.add_keyframe(i, state.R, state.t, pts, des, landmark_ids)
            descriptor_db.add_keyframe(kf)
            pose_graph.add_node_from_keyframe(kf)
            if prev_kf is not None:
                # Essential graph spanning tree: connect only to parent at creation time.
                pose_graph.add_tree_edge(prev_kf, kf)
                # Add strong covisibility edges (>= N shared landmarks).
                for other in keyframes_created:
                    if other.id == prev_kf.id:
                        continue
                    if _count_shared_landmarks(other, kf) >= COVIS_MIN_SHARED:
                        pose_graph.add_covisibility_edge(other, kf)
            prev_kf = kf
            keyframes_created.append(kf)
            new_keyframe_created = True
            _recompute_corrected_trajectory()

    if new_keyframe_created:
        loop_candidate = find_loop_closure(
            kfm,
            descriptor_db,
            intrinsics.K,
            ransac_thresh=1.5,
            max_iters=3000,
            min_inliers=50,
            top_k=20,
        )
        if loop_candidate is not None:
            loops_detected += 1
            q_kf = loop_candidate.query_kf
            m_kf = loop_candidate.match_kf
            delta_frames = int(abs(int(q_kf.frame_idx) - int(m_kf.frame_idx)))
            last_loop_text = f"{q_kf.frame_idx} ↔ {m_kf.frame_idx} (Δ={delta_frames})"
            meas, mstats = compute_loop_sim3_measurement(
                loop_candidate,
                pose_graph,
                state,
                sim3_thresh=0.5,
                max_iters=2000,
                min_pairs=30,
                min_inliers=20,
                min_inlier_ratio=0.3,
                scale_bounds=(0.3, 3.0),
            )
            meas_text = ""
            if meas is not None:
                R_rel, t_rel, s_rel = meas
                # Summarize measurement magnitude for debugging.
                rot_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R_rel)[0].reshape(3))))
                t_norm = float(np.linalg.norm(np.asarray(t_rel, dtype=np.float64).reshape(3)))
                meas_text = f" rot={rot_deg:.1f}deg |t|={t_norm:.2f} s={float(s_rel):.3f}"
            loop_sim3_text = (
                f"[LoopSim3] {('ACCEPT' if mstats.accepted else 'REJECT')} "
                f"pairs={mstats.pairs} inl={mstats.inliers} "
                f"ratio={mstats.inlier_ratio:.2f} s={mstats.scale:.3f}"
                f" | q_kf={q_kf.id} m_kf={m_kf.id} Δf={delta_frames}"
                f"{meas_text}"
            )
            if meas is not None:
                pose_graph.add_loop_edge(
                    loop_candidate.query_kf, loop_candidate.match_kf, R_rel, t_rel, s_rel
                )
                opt = pose_graph.optimize(max_nfev=30, loss="huber", f_scale=2.0)
                pg_opt_text = (
                    f"[PoseGraph] cost {opt.cost_initial:.1f}→{opt.cost_final:.1f} "
                    f"shift max={opt.max_pose_shift:.2f} mean={opt.mean_pose_shift:.2f}"
                )

                if APPLY_MAP_WARP and getattr(state, "landmarks_3d", None) is not None:
                    try:
                        S_map = compute_world_warp_from_anchor_keyframe(q_kf, pose_graph)
                        state.landmarks_3d = apply_world_warp_to_landmarks(
                            state.landmarks_3d, S_map
                        )
                        detA = float(np.linalg.det(np.asarray(S_map[:3, :3])))
                        s_map = float(np.cbrt(detA)) if abs(detA) > 1e-12 else 1.0
                        map_warp_text = f"[MapWarp] applied s={s_map:.3f}"
                    except Exception as _e:
                        map_warp_text = "[MapWarp] failed"
                    if map_warp_text:
                        pg_opt_text = (pg_opt_text or "[PoseGraph]") + " " + map_warp_text

                if WORLD_RESET_AFTER_LOOP:
                    # Rebase frontend state + map into the optimized world so tracking continues
                    # without snapping back to the pre-loop VO coordinate system.
                    S_W = compute_world_correction_from_anchor(q_kf, pose_graph)
                    s_w, _R_w, _t_w = decompose_sim3_matrix(S_W)

                    # Pose-jump diagnostic on camera center (world coords).
                    C_old = (
                        -np.asarray(state.R, dtype=np.float64).T
                        @ np.asarray(state.t, dtype=np.float64)
                    ).reshape(3)

                    # Live pose (R,t)
                    state.R, state.t = apply_world_correction_to_Rt(state.R, state.t, S_W)

                    # Per-frame pose history (poses_wc)
                    apply_world_correction_to_Twc_list(poses_wc, S_W)

                    # Stored keyframe poses (used by loop measurement & warping)
                    apply_world_correction_to_keyframes(keyframes_created, S_W)

                    # Stored reference poses inside tracks/candidates (triangulation references).
                    apply_world_correction_to_tracks_and_candidates(state, S_W)

                    # Map landmarks (world points)
                    if getattr(state, "landmarks_3d", None) is not None and state.landmarks_3d.size > 0:
                        state.landmarks_3d = sim3_to_points(S_W, state.landmarks_3d)

                    # Existing trajectory points (camera centers in world)
                    if getattr(state, "trajectory", None) is not None and len(state.trajectory) > 0:
                        traj_pts = np.asarray(state.trajectory, dtype=np.float64).reshape(-1, 3)
                        traj_pts = sim3_to_points(S_W, traj_pts)
                        state.trajectory = [traj_pts[i] for i in range(traj_pts.shape[0])]

                    # Also rebase the corrected trajectory list (if it already exists)
                    if len(trajectory_corrected) > 0:
                        trajc = np.asarray(trajectory_corrected, dtype=np.float64).reshape(-1, 3)
                        trajc = sim3_to_points(S_W, trajc)
                        trajectory_corrected.clear()
                        trajectory_corrected.extend([trajc[i] for i in range(trajc.shape[0])])

                    C_new = (
                        -np.asarray(state.R, dtype=np.float64).T
                        @ np.asarray(state.t, dtype=np.float64)
                    ).reshape(3)
                    jump = float(np.linalg.norm(C_new - C_old))
                    # Tell the frontend to drop any motion priors next frame.
                    state.world_reset_pending = True
                    # Store the raw correction (for debugging only; not used directly in VO updates).
                    state.world_R = np.asarray(S_W[:3, :3], dtype=np.float64)
                    state.world_s = float(s_w)
                    rot_deg = float(
                        np.degrees(
                            np.arccos(
                                np.clip((np.trace(state.world_R) - 1.0) / 2.0, -1.0, 1.0)
                            )
                        )
                    )
                    print(
                        f"[WorldReset] applied S_W: scale={s_w:.3f} | pose_jump ||C_new-C_old||={jump:.3f}"
                    )
                    print(
                        f"[WorldAnchor] Updated increment anchor: rot_deg={rot_deg:.2f} scale={s_w:.3f}"
                    )
                    loop_closing_active = True
                _recompute_corrected_trajectory()

            # Persist loop info for every recognized loop
            _append_loop_log(
                loopedge=pose_graph.loop_edges_count,
                loop_sim3=loop_sim3_text,
                posegraph=pg_opt_text,
            )

    # Prepare data for UI (last 20 frames)
    # Before first accepted loop: show raw VO.
    # After: show the single active SLAM trajectory (rebased via WORLD_RESET).
    if loop_closing_active:
        full_trajectory_raw = np.array(trajectory_corrected)
    else:
        full_trajectory_raw = np.array(state.trajectory)
    full_trajectory_corrected_raw = np.array(trajectory_corrected)
    last20_slice = slice(
        max(0, len(full_trajectory_raw) - 20), len(full_trajectory_raw)
    )
    local_traj_last20_raw = full_trajectory_raw[last20_slice]
    local_traj_last20_corrected_raw = full_trajectory_corrected_raw[last20_slice]

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
            local_landmarks_last20 = (
                visible_world[:, [0, 2]] if visible_world.size > 0 else np.empty((0, 2))
            )
        else:
            local_landmarks_last20 = np.empty((0, 2))
    else:
        local_landmarks_last20 = np.empty((0, 2))

    # Determine a consistent scale using ground truth if available.
    # This keeps visualization physically meaningful without hard-coding.
    if dataset.ground_truth is not None and full_trajectory_raw.shape[0] > 1:
        gt = dataset.ground_truth
        # Use the same frame index for GT as for the current image, clamped to GT length.
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
    full_trajectory_corrected = scale * full_trajectory_corrected_raw[:, [0, 2]]
    local_traj_last20_corrected = scale * local_traj_last20_corrected_raw[:, [0, 2]]
    local_landmarks_last20_scaled = scale * local_landmarks_last20

    # Ground-truth trajectory overlay (grey, in the background)
    full_trajectory_gt = None
    if dataset.ground_truth is not None and np.asarray(dataset.ground_truth).size > 0:
        gt = np.asarray(dataset.ground_truth, dtype=float)
        gt_idx = min(i, gt.shape[0] - 1)
        gt_segment = gt[: gt_idx + 1]
        if gt_segment.shape[0] > 0 and full_trajectory.shape[0] > 0:
            full_trajectory_gt = (gt_segment - gt_segment[0]) + full_trajectory[0]

    # Visualization image with landmarks/candidates overlaid
    vis = visualize_tracking(image, state)

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
        if PLOT_SINGLE_TRAJECTORY:
            full_trajectory_corrected = None
            local_traj_last20_corrected = None
        update_ui(
            image=vis,
            tracked_landmarks_count=state.landmark_counts[-20:],
            full_trajectory=full_trajectory,
            full_trajectory_corrected=full_trajectory_corrected,
            local_traj_last20=local_traj_last20,
            local_traj_last20_corrected=local_traj_last20_corrected,
            local_landmarks_last20=local_landmarks_last20_scaled,
            axis_limits=state.axis_limits,
            rte_errors=None,
            full_trajectory_gt=full_trajectory_gt,
            loops_detected=loops_detected,
            keyframes_count=len(kfm.get_all_keyframes()),
            loop_edges_count=pose_graph.loop_edges_count,
            db_features_count=descriptor_db.total_descriptors,
            last_keyframe_features=last_keyframe_features,
            loop_sim3_text=loop_sim3_text,
            pg_opt_text=pg_opt_text,
            last_loop_text=last_loop_text,
            trajectory_mode=("SLAM" if loop_closing_active else "VO"),
        )
    else:
        cv2.imshow("Tracking", vis)
        cv2.waitKey(10)

    if STOP_AFTER_FIRST_FRAME:
        print("[SmokeTest] Stopping after first frame.")
        break


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
        if compute_rte is None:
            print("\n[1/3] Skipping RTE (missing optional dependency 'evo').")
        else:
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
        eval_filename = f"eval_{dataset_name}_{timestamp}.json"
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

        # Also save RTE separately for backward compatibility (only if available)
        if rte_results is not None:
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
                full_trajectory_corrected=None,
                local_traj_last20=local_traj_last20,
                local_landmarks_last20=local_landmarks_last20,
                axis_limits=state.axis_limits,
                rte_errors=rte["per_pair_errors"],
                full_trajectory_gt=dataset.ground_truth,
                loops_detected=loops_detected,
                keyframes_count=len(kfm.get_all_keyframes()),
                loop_edges_count=pose_graph.loop_edges_count,
                db_features_count=descriptor_db.total_descriptors,
                last_keyframe_features=last_keyframe_features,
                loop_sim3_text=loop_sim3_text,
                pg_opt_text=pg_opt_text,
                last_loop_text=last_loop_text,
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
    eval_filename = f"eval_{dataset_name}_{timestamp}.json"

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
