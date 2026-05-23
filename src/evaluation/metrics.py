# evaluation/metrics.py
"""
Comprehensive VO evaluation metrics:
- Scale Error
- KITTI Benchmark Metrics (translation %, rotation deg/100m)
- Runtime Statistics
"""

from typing import Dict, List, Optional

import numpy as np


def compute_scale_error(
    traj_est: np.ndarray, traj_gt: np.ndarray, window_size: int = 10
) -> Dict[str, float]:
    """
    Compute scale error between estimated and ground truth trajectories.

    The scale error measures how well the estimated trajectory maintains
    consistent scale compared to ground truth over time.

    Parameters:
        traj_est: Estimated trajectory (Nx3 or Nx2 positions)
        traj_gt: Ground truth trajectory (Nx3 or Nx2 positions)
        window_size: Size of sliding window for local scale estimation

    Returns:
        Dictionary with scale error statistics:
        - optimal_scale: Global scale factor to align trajectories
        - scale_drift: Change in local scale over trajectory (%)
        - mean_local_scale: Mean of local scale factors
        - std_local_scale: Standard deviation of local scale factors
    """
    traj_est = np.asarray(traj_est)
    traj_gt = np.asarray(traj_gt)

    # Handle 2D case (x, z) -> pad y with zeros
    if traj_est.ndim == 2 and traj_est.shape[1] == 2:
        traj_est = np.column_stack(
            [traj_est[:, 0], np.zeros(len(traj_est)), traj_est[:, 1]]
        )
    if traj_gt.ndim == 2 and traj_gt.shape[1] == 2:
        traj_gt = np.column_stack(
            [traj_gt[:, 0], np.zeros(len(traj_gt)), traj_gt[:, 1]]
        )

    n = min(len(traj_est), len(traj_gt))
    traj_est = traj_est[:n]
    traj_gt = traj_gt[:n]

    # Compute global optimal scale
    denom = np.sum(traj_est * traj_est)
    if denom < 1e-12:
        optimal_scale = 1.0
    else:
        optimal_scale = float(np.sum(traj_est * traj_gt) / denom)

    # Compute local scale factors using sliding windows
    local_scales = []
    for i in range(0, n - window_size, window_size // 2):
        end_idx = min(i + window_size, n)

        # Compute segment lengths
        seg_est = traj_est[i:end_idx]
        seg_gt = traj_gt[i:end_idx]

        len_est = np.sum(np.linalg.norm(np.diff(seg_est, axis=0), axis=1))
        len_gt = np.sum(np.linalg.norm(np.diff(seg_gt, axis=0), axis=1))

        if len_est > 1e-6:
            local_scales.append(len_gt / len_est)

    local_scales = np.array(local_scales) if local_scales else np.array([optimal_scale])

    # Scale drift: relative change from start to end
    if len(local_scales) > 1:
        scale_drift = (local_scales[-1] - local_scales[0]) / local_scales[0] * 100
    else:
        scale_drift = 0.0

    return {
        "optimal_scale": float(optimal_scale),
        "scale_drift_percent": float(scale_drift),
        "mean_local_scale": float(np.mean(local_scales)),
        "std_local_scale": float(np.std(local_scales)),
        "local_scales": local_scales.tolist(),
    }


def compute_kitti_metrics(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
    poses_est: Optional[np.ndarray] = None,
    poses_gt: Optional[np.ndarray] = None,
    segment_lengths: List[int] = [100, 200, 300, 400, 500, 600, 700, 800],
    step_size: int = 10,
) -> Dict[str, float]:
    """
    Compute KITTI benchmark metrics: translation error (%) and rotation error (deg/100m).

    The KITTI benchmark evaluates odometry over segments of different lengths
    (100m to 800m by default) and reports average errors.

    Parameters:
        traj_est: Estimated trajectory positions (Nx3 or Nx2)
        traj_gt: Ground truth trajectory positions (Nx3 or Nx2)
        poses_est: Optional Nx4x4 pose matrices for rotation error (if None, only translation computed)
        poses_gt: Optional Nx4x4 ground truth pose matrices
        segment_lengths: List of segment lengths in meters to evaluate
        step_size: Step size for starting frames

    Returns:
        Dictionary with KITTI metrics:
        - translation_error_percent: Average translation error (%)
        - rotation_error_deg_per_100m: Average rotation error (deg/100m)
        - errors_by_length: Dict mapping segment length to (trans_err, rot_err)
    """
    traj_est = np.asarray(traj_est)
    traj_gt = np.asarray(traj_gt)

    # Handle 2D case
    if traj_est.ndim == 2 and traj_est.shape[1] == 2:
        traj_est = np.column_stack(
            [traj_est[:, 0], np.zeros(len(traj_est)), traj_est[:, 1]]
        )
    if traj_gt.ndim == 2 and traj_gt.shape[1] == 2:
        traj_gt = np.column_stack(
            [traj_gt[:, 0], np.zeros(len(traj_gt)), traj_gt[:, 1]]
        )

    n = min(len(traj_est), len(traj_gt))
    traj_est = traj_est[:n]
    traj_gt = traj_gt[:n]

    # Scale alignment (for monocular VO)
    denom = np.sum(traj_est * traj_est)
    if denom > 1e-12:
        scale = float(np.sum(traj_est * traj_gt) / denom)
        traj_est_scaled = scale * traj_est
    else:
        traj_est_scaled = traj_est

    # Compute cumulative distances for ground truth
    dists = np.zeros(n)
    for i in range(1, n):
        dists[i] = dists[i - 1] + np.linalg.norm(traj_gt[i] - traj_gt[i - 1])

    errors_by_length = {}
    all_trans_errors = []
    all_rot_errors = []

    for seg_len in segment_lengths:
        trans_errors = []
        rot_errors = []

        for start in range(0, n, step_size):
            # Find end frame where cumulative distance >= seg_len
            target_dist = dists[start] + seg_len
            end = start + 1
            while end < n and dists[end] < target_dist:
                end += 1

            if end >= n:
                continue

            # Translation error
            delta_gt = traj_gt[end] - traj_gt[start]
            delta_est = traj_est_scaled[end] - traj_est_scaled[start]

            len_gt = np.linalg.norm(delta_gt)
            error_trans = np.linalg.norm(delta_gt - delta_est)

            if len_gt > 1e-6:
                trans_error_percent = (error_trans / len_gt) * 100
                trans_errors.append(trans_error_percent)
                all_trans_errors.append(trans_error_percent)

            # Rotation error (if poses provided)
            if poses_est is not None and poses_gt is not None:
                if start < len(poses_est) and end < len(poses_est):
                    R_gt_start = poses_gt[start][:3, :3]
                    R_gt_end = poses_gt[end][:3, :3]
                    R_est_start = poses_est[start][:3, :3]
                    R_est_end = poses_est[end][:3, :3]

                    # Relative rotations
                    dR_gt = R_gt_end @ R_gt_start.T
                    dR_est = R_est_end @ R_est_start.T

                    # Error rotation
                    dR_err = dR_gt @ dR_est.T

                    # Angle from rotation matrix
                    angle = np.arccos(np.clip((np.trace(dR_err) - 1) / 2, -1, 1))
                    angle_deg = np.degrees(angle)

                    # Convert to deg/100m
                    rot_error = angle_deg / (seg_len / 100)
                    rot_errors.append(rot_error)
                    all_rot_errors.append(rot_error)

        if trans_errors:
            errors_by_length[seg_len] = {
                "translation_error_percent": float(np.mean(trans_errors)),
                "rotation_error_deg_per_100m": (
                    float(np.mean(rot_errors)) if rot_errors else None
                ),
                "num_segments": len(trans_errors),
            }

    return {
        "translation_error_percent": (
            float(np.mean(all_trans_errors)) if all_trans_errors else None
        ),
        "rotation_error_deg_per_100m": (
            float(np.mean(all_rot_errors)) if all_rot_errors else None
        ),
        "errors_by_length": errors_by_length,
        "num_total_segments": len(all_trans_errors),
    }


class RuntimeTracker:
    """
    Track runtime statistics for VO pipeline.

    Usage:
        tracker = RuntimeTracker()

        for frame in frames:
            tracker.start_frame()
            # ... process frame ...
            tracker.end_frame()

        stats = tracker.get_statistics()
    """

    def __init__(self):
        self.frame_times: List[float] = []
        self._frame_start: Optional[float] = None
        self._component_times: Dict[str, List[float]] = {}
        self._component_start: Dict[str, float] = {}

    def start_frame(self):
        """Start timing a new frame."""
        import time

        self._frame_start = time.perf_counter()

    def end_frame(self):
        """End timing the current frame."""
        import time

        if self._frame_start is not None:
            elapsed = time.perf_counter() - self._frame_start
            self.frame_times.append(elapsed)
            self._frame_start = None

    def start_component(self, name: str):
        """Start timing a specific component (e.g., 'feature_extraction', 'pose_estimation')."""
        import time

        self._component_start[name] = time.perf_counter()

    def end_component(self, name: str):
        """End timing a specific component."""
        import time

        if name in self._component_start:
            elapsed = time.perf_counter() - self._component_start[name]
            if name not in self._component_times:
                self._component_times[name] = []
            self._component_times[name].append(elapsed)
            del self._component_start[name]

    def get_statistics(self) -> Dict[str, float]:
        """
        Get runtime statistics.

        Returns:
            Dictionary with:
            - total_time_sec: Total processing time
            - num_frames: Number of frames processed
            - avg_frame_time_ms: Average time per frame (milliseconds)
            - fps: Frames per second
            - min_frame_time_ms: Minimum frame time
            - max_frame_time_ms: Maximum frame time
            - std_frame_time_ms: Standard deviation of frame times
            - component_times: Dict of component timing statistics
        """
        if not self.frame_times:
            return {
                "total_time_sec": 0.0,
                "num_frames": 0,
                "avg_frame_time_ms": 0.0,
                "fps": 0.0,
            }

        times = np.array(self.frame_times)
        total_time = np.sum(times)

        stats = {
            "total_time_sec": float(total_time),
            "num_frames": len(times),
            "avg_frame_time_ms": float(np.mean(times) * 1000),
            "fps": float(len(times) / total_time) if total_time > 0 else 0.0,
            "min_frame_time_ms": float(np.min(times) * 1000),
            "max_frame_time_ms": float(np.max(times) * 1000),
            "std_frame_time_ms": float(np.std(times) * 1000),
            "median_frame_time_ms": float(np.median(times) * 1000),
        }

        # Component statistics
        component_stats = {}
        for name, comp_times in self._component_times.items():
            comp_times = np.array(comp_times)
            component_stats[name] = {
                "avg_ms": float(np.mean(comp_times) * 1000),
                "std_ms": float(np.std(comp_times) * 1000),
                "total_sec": float(np.sum(comp_times)),
            }

        if component_stats:
            stats["component_times"] = component_stats

        return stats

    def print_summary(self):
        """Print a formatted summary of runtime statistics."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("RUNTIME STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {stats['num_frames']}")
        print(f"Total processing time:  {stats['total_time_sec']:.2f} sec")
        print("-" * 60)
        print(f"Average frame time:     {stats['avg_frame_time_ms']:.2f} ms")
        print(f"FPS:                    {stats['fps']:.2f}")
        print(f"Min frame time:         {stats.get('min_frame_time_ms', 0):.2f} ms")
        print(f"Max frame time:         {stats.get('max_frame_time_ms', 0):.2f} ms")
        print(f"Std frame time:         {stats.get('std_frame_time_ms', 0):.2f} ms")
        print(f"Median frame time:      {stats.get('median_frame_time_ms', 0):.2f} ms")

        if "component_times" in stats:
            print("-" * 60)
            print("Component breakdown:")
            for name, comp_stats in stats["component_times"].items():
                print(f"  {name}: {comp_stats['avg_ms']:.2f} ms avg")

        print("=" * 60)


def print_evaluation_summary(
    rte_results: Optional[Dict] = None,
    scale_results: Optional[Dict] = None,
    kitti_results: Optional[Dict] = None,
    runtime_stats: Optional[Dict] = None,
):
    """
    Print a comprehensive evaluation summary.

    Parameters:
        rte_results: Results from compute_rte()
        scale_results: Results from compute_scale_error()
        kitti_results: Results from compute_kitti_metrics()
        runtime_stats: Results from RuntimeTracker.get_statistics()
    """
    print("\n")
    print("=" * 70)
    print("                    VO EVALUATION SUMMARY")
    print("=" * 70)

    if rte_results:
        print("\n📊 RELATIVE TRAJECTORY ERROR (RTE)")
        print("-" * 40)
        print(f"  Mean RTE:   {rte_results['mean_rte']:.4f} m")
        print(f"  Median RTE: {rte_results['median_rte']:.4f} m")
        print(f"  RMSE RTE:   {rte_results['rmse_rte']:.4f} m")

    if scale_results:
        print("\n📏 SCALE ERROR")
        print("-" * 40)
        print(f"  Optimal scale factor:  {scale_results['optimal_scale']:.4f}")
        print(f"  Scale drift:           {scale_results['scale_drift_percent']:.2f}%")
        print(f"  Mean local scale:      {scale_results['mean_local_scale']:.4f}")
        print(f"  Std local scale:       {scale_results['std_local_scale']:.4f}")

    if kitti_results:
        print("\n🏎️  KITTI BENCHMARK METRICS")
        print("-" * 40)
        trans_err = kitti_results.get("translation_error_percent")
        rot_err = kitti_results.get("rotation_error_deg_per_100m")

        if trans_err is not None:
            print(f"  Translation error:     {trans_err:.2f}%")
        if rot_err is not None:
            print(f"  Rotation error:        {rot_err:.4f} deg/100m")

        print(f"  Total segments eval'd: {kitti_results.get('num_total_segments', 0)}")

        if kitti_results.get("errors_by_length"):
            print("\n  Errors by segment length:")
            for seg_len, errs in sorted(kitti_results["errors_by_length"].items()):
                trans = errs.get("translation_error_percent")
                if trans is not None:
                    print(
                        f"    {seg_len:4d}m: {trans:.2f}% ({errs['num_segments']} segments)"
                    )

    if runtime_stats:
        print("\n⏱️  RUNTIME PERFORMANCE")
        print("-" * 40)
        print(f"  Total frames:          {runtime_stats['num_frames']}")
        print(f"  Total time:            {runtime_stats['total_time_sec']:.2f} sec")
        print(f"  Average frame time:    {runtime_stats['avg_frame_time_ms']:.2f} ms")
        print(f"  FPS:                   {runtime_stats['fps']:.2f}")
        if "min_frame_time_ms" in runtime_stats:
            print(
                f"  Min/Max frame time:    {runtime_stats['min_frame_time_ms']:.2f} / {runtime_stats['max_frame_time_ms']:.2f} ms"
            )

    print("\n" + "=" * 70)


def save_evaluation_results(
    filepath: str,
    rte_results: Optional[Dict] = None,
    scale_results: Optional[Dict] = None,
    kitti_results: Optional[Dict] = None,
    runtime_stats: Optional[Dict] = None,
    script_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    hyperparameters: Optional[Dict] = None,
):
    """
    Save all evaluation results to a JSON file.

    Parameters:
        filepath: Path to save the JSON file
        rte_results: Results from compute_rte()
        scale_results: Results from compute_scale_error()
        kitti_results: Results from compute_kitti_metrics()
        runtime_stats: Results from RuntimeTracker.get_statistics()
        script_name: Name of the script that generated these results
        dataset_name: Name of the dataset used
        hyperparameters: Dictionary of hyperparameters used in the run
    """
    import json
    from pathlib import Path

    results = {
        "timestamp": str(np.datetime64("now")),
    }

    # Add metadata
    if script_name:
        results["script_name"] = script_name

    if dataset_name:
        results["dataset"] = dataset_name

    if hyperparameters:
        results["hyperparameters"] = hyperparameters

    if rte_results:
        # Convert numpy arrays to lists for JSON serialization
        rte_copy = rte_results.copy()
        if "per_pair_errors" in rte_copy:
            rte_copy["per_pair_errors"] = (
                rte_copy["per_pair_errors"].tolist()
                if hasattr(rte_copy["per_pair_errors"], "tolist")
                else rte_copy["per_pair_errors"]
            )
        results["rte"] = rte_copy

    if scale_results:
        results["scale_error"] = scale_results

    if kitti_results:
        results["kitti_metrics"] = kitti_results

    if runtime_stats:
        results["runtime"] = runtime_stats

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation results saved to: {filepath}")
