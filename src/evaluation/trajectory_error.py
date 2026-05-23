import numpy as np
from evo.core import transformations
from evo.core.metrics import RPE, PoseRelation
from evo.core.trajectory import PoseTrajectory3D


def _to_pose_matrices(traj):
    """
    Convert various trajectory formats to an array of 4x4 pose matrices.

    Accepted formats:
        - list/array of 4x4 matrices (N x 4 x 4)
        - list/array of 3D positions (N x 3) -> identity rotation
        - list/array of 2D positions (N x 2) -> identity rotation, y = 0
    """
    traj_arr = np.asarray(traj)

    if traj_arr.ndim == 3 and traj_arr.shape[1:] == (4, 4):
        return traj_arr

    if traj_arr.ndim == 2:
        if traj_arr.shape[1] == 3:
            positions = traj_arr
        elif traj_arr.shape[1] == 2:
            positions = np.column_stack([traj_arr, np.zeros(len(traj_arr))])
        else:
            raise ValueError(
                "Unsupported trajectory shape: expected Nx4x4, Nx3 or Nx2, "
                f"got {traj_arr.shape}"
            )

        poses = np.repeat(np.eye(4)[None, :, :], len(positions), axis=0)
        poses[:, :3, 3] = positions
        return poses

    raise ValueError(
        "Unsupported trajectory format: expected list/array of 4x4 poses or "
        "positions with shape Nx3 / Nx2."
    )


def _poses_to_trajectory(poses):
    """
    Convert an array of 4x4 pose matrices (T_i_wc) to a PoseTrajectory3D.
    """
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape Nx4x4")

    num_poses = poses.shape[0]
    positions = poses[:, :3, 3]
    quats = np.zeros((num_poses, 4))
    for i in range(num_poses):
        quats[i] = transformations.quaternion_from_matrix(poses[i])

    timestamps = np.arange(num_poses, dtype=float)
    return PoseTrajectory3D(
        positions_xyz=positions, orientations_quat_wxyz=quats, timestamps=timestamps
    )


def _scale_align_trajectory(
    traj_gt: PoseTrajectory3D, traj_est: PoseTrajectory3D
) -> PoseTrajectory3D:
    """
    Perform a simple scale alignment between two trajectories based on their
    translation components.

    This mimics the behavior of evo's ScaleCorrector for pure-translation
    trajectories: we find the scalar s that minimizes || s * p_est - p_gt ||^2.
    """
    p_gt = np.asarray(traj_gt.positions_xyz)
    p_est = np.asarray(traj_est.positions_xyz)

    n = min(len(p_gt), len(p_est))
    if n == 0:
        return traj_est

    p_gt = p_gt[:n]
    p_est = p_est[:n]

    denom = float(np.sum(p_est * p_est))
    if denom < 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(p_est * p_gt) / denom)

    scaled_positions = scale * np.asarray(traj_est.positions_xyz)

    return PoseTrajectory3D(
        positions_xyz=scaled_positions,
        orientations_quat_wxyz=np.asarray(traj_est.orientations_quat_wxyz),
        timestamps=np.asarray(traj_est.timestamps),
    )


def compute_rte(traj_est, traj_gt, delta=1, align_scale=True):
    """
    Computes Relative Trajectory Error (RTE/RPE) between estimated trajectory and ground truth.

    Parameters:
        traj_est: list or Nx4x4 / Nx3 / Nx2 numpy array of estimated camera poses or positions (T_i_wc)
        traj_gt:  list or Nx4x4 / Nx3 / Nx2 numpy array of ground truth poses or positions
        delta: frame interval for RPE (default: 1)
        align_scale: if True, perform scale alignment for monocular VO

    Returns:
        A dictionary with:
            - mean_rte
            - median_rte
            - rmse_rte
            - per_pair_errors   (numpy array)
    """
    if traj_est is None or traj_gt is None:
        raise ValueError("traj_est and traj_gt must not be None")

    poses_est = _to_pose_matrices(traj_est)
    poses_gt = _to_pose_matrices(traj_gt)

    n = min(len(poses_est), len(poses_gt))
    if n < 2:
        raise ValueError("Need at least 2 poses to compute RPE.")

    poses_est = poses_est[:n]
    poses_gt = poses_gt[:n]

    traj_gt_evo = _poses_to_trajectory(poses_gt)
    traj_est_evo = _poses_to_trajectory(poses_est)

    if align_scale:
        traj_est_evo = _scale_align_trajectory(traj_gt_evo, traj_est_evo)

    rpe_metric = RPE(pose_relation=PoseRelation.translation_part, delta=delta)
    rpe_metric.process_data((traj_gt_evo, traj_est_evo))
    result = rpe_metric.get_result()

    return {
        "mean_rte": float(result.stats["mean"]),
        "median_rte": float(result.stats["median"]),
        "rmse_rte": float(result.stats["rmse"]),
        "per_pair_errors": np.asarray(result.np_arrays["error_array"]),
    }


def plot_rte(errors):
    import matplotlib.pyplot as plt

    errors = np.asarray(errors)

    plt.figure()
    plt.plot(errors)
    plt.title("Relative Trajectory Error (RTE)")
    plt.xlabel("Frame index")
    plt.ylabel("Translation error [m]")
    plt.grid(True)
    plt.show()
