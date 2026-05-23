import matplotlib.pyplot as plt
import numpy as np
import cv2


def _ensure_figure():
    if not plt.get_fignums():
        plt.figure("VO UI", figsize=(10, 8))
    else:
        plt.figure("VO UI")


def update_ui(
    image,
    tracked_landmarks_count,
    full_trajectory,
    local_traj_last20,
    local_landmarks_last20,
    axis_limits=None,
    rte_errors=None,
    full_trajectory_gt=None,
    full_trajectory_corrected=None,
    local_traj_last20_corrected=None,
    loops_detected: int = 0,
    keyframes_count: int = 0,
    db_features_count: int = 0,
    last_keyframe_features: int = 0,
    loop_sim3_text: str | None = None,
    loop_edges_count: int = 0,
    pg_opt_text: str | None = None,
    last_loop_text: str | None = None,
    trajectory_mode: str | None = None,
    ground_points_px: np.ndarray | None = None,
    show_debug_trajectory: bool = False,
):
    """
    Displays ONE window with four 2D subplots:
    1) Current image (grayscale or RGB)
    2) Number of tracked landmarks over last 20 frames (line plot)
    3) Full trajectory in 2D (axis equal)
    4) Trajectory of last 20 frames + landmarks (axis equal)
    """
    _ensure_figure()
    plt.clf()
    fig = plt.gcf()

    use_rte_subplot = rte_errors is not None and len(rte_errors) > 0
    if use_rte_subplot:
        grid_nrows, grid_ncols = 2, 3
    else:
        grid_nrows, grid_ncols = 2, 2

    # 1) Current image
    ax1 = plt.subplot(grid_nrows, grid_ncols, 1)
    ax1.set_title("Current Image")
    image_show = image
    if ground_points_px is not None and len(ground_points_px) > 0:
        img = image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pts = np.asarray(ground_points_px, dtype=float).reshape(-1, 2)
        h, w = img.shape[:2]
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(img, (xi, yi), 7, (0, 255, 0), 2)
        image_show = img

    if image_show.ndim == 2:
        ax1.imshow(image_show, cmap="gray")
    else:
        ax1.imshow(image_show[..., ::-1])  # BGR -> RGB if needed
    ax1.axis("off")

    # 2) Tracked landmarks count (last 20 frames)
    ax2 = plt.subplot(grid_nrows, grid_ncols, 2)
    ax2.set_title("# Tracked Landmarks (last 20)")
    counts = np.array(tracked_landmarks_count, dtype=float).ravel()
    if counts.size > 0:
        ax2.plot(np.arange(len(counts)), counts)
        ax2.set_xlabel("Frame offset")
        ax2.set_ylabel("Count")
    ax2.grid(True)

    # 3) Full trajectory (top-down)
    ax3 = plt.subplot(grid_nrows, grid_ncols, grid_ncols + 1)
    ax3.set_title("Full Trajectory")
    if full_trajectory_gt is not None and len(full_trajectory_gt) > 0:
        gt = np.asarray(full_trajectory_gt, dtype=float)
        ax3.plot(gt[:, 0], gt[:, 1], color="0.7", linewidth=2.0, zorder=1)
    if full_trajectory is not None and len(full_trajectory) > 0:
        traj = np.asarray(full_trajectory, dtype=float)
        ax3.plot(traj[:, 0], traj[:, 1], "-b", zorder=2)
        # Per-frame positions for jump/frame-switch diagnosis.
        ax3.scatter(traj[:, 0], traj[:, 1], c="r", s=4, alpha=0.25, zorder=2)
        ax3.scatter(traj[-1, 0], traj[-1, 1], c="r", s=20, zorder=3)
    if (
        show_debug_trajectory
        and full_trajectory_corrected is not None
        and len(full_trajectory_corrected) > 0
    ):
        traj_c = np.asarray(full_trajectory_corrected, dtype=float)
        ax3.plot(traj_c[:, 0], traj_c[:, 1], "--m", linewidth=1.5, zorder=2)

    def _truncate(s: str, n: int = 90) -> str:
        s = str(s)
        return s if len(s) <= n else (s[: n - 1] + "…")

    # Keep this box short so it doesn't shrink subplots under tight_layout.
    info_lines = [
        f"Loops: {int(loops_detected)} | Loop edges: {int(loop_edges_count)} | Keyframes: {int(keyframes_count)}",
        f"DB feats: {int(db_features_count)} | KF feats: {int(last_keyframe_features)}",
    ]
    if trajectory_mode:
        info_lines.append(_truncate(f"Trajectory: {trajectory_mode}", 110))
    if loop_sim3_text:
        info_lines.append(_truncate(loop_sim3_text, 110))
    if pg_opt_text:
        info_lines.append(_truncate(pg_opt_text, 110))
    if last_loop_text:
        info_lines.append(_truncate(f"Last loop: {last_loop_text}", 110))

    fig.text(
        0.01,
        0.99,
        "\n".join(info_lines[:5]),
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.6"),
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.axis("equal")
    ax3.grid(True)

    # 4) Local trajectory (last 20) + landmarks, with fixed axes
    ax4 = plt.subplot(grid_nrows, grid_ncols, grid_ncols + 2)
    ax4.set_title("Local Trajectory + Landmarks (last 20)")
    if local_traj_last20 is not None and len(local_traj_last20) > 0:
        local_traj = np.asarray(local_traj_last20, dtype=float)
        ax4.plot(local_traj[:, 0], local_traj[:, 1], "-g")
        ax4.scatter(local_traj[:, 0], local_traj[:, 1], c="r", s=6, alpha=0.35)
        ax4.scatter(local_traj[-1, 0], local_traj[-1, 1], c="r", s=20)
    if (
        show_debug_trajectory
        and local_traj_last20_corrected is not None
        and len(local_traj_last20_corrected) > 0
    ):
        local_traj_c = np.asarray(local_traj_last20_corrected, dtype=float)
        ax4.plot(local_traj_c[:, 0], local_traj_c[:, 1], "--m", linewidth=1.5)
    if local_landmarks_last20 is not None and len(local_landmarks_last20) > 0:
        lm = np.asarray(local_landmarks_last20, dtype=float)
        ax4.scatter(lm[:, 0], lm[:, 1], c="k", s=5, alpha=0.5)
    ax4.set_xlabel("x")
    ax4.set_ylabel("z")
    ax4.axis("equal")
    # Fix axes using externally provided limits, updated sparsely
    if axis_limits is not None:
        xmin, xmax, zmin, zmax = axis_limits
        ax4.set_xlim(xmin, xmax)
        ax4.set_ylim(zmin, zmax)

    ax4.grid(True)

    if use_rte_subplot:
        ax5 = plt.subplot(grid_nrows, grid_ncols, 3)
        ax5.set_title("Relative Trajectory Error (RTE)")
        errors = np.asarray(rte_errors, dtype=float).ravel()
        ax5.plot(np.arange(len(errors)), errors)
        ax5.set_xlabel("Frame index")
        ax5.set_ylabel("Translation error [m]")
        ax5.grid(True)

    plt.tight_layout()
    plt.pause(0.00001)
