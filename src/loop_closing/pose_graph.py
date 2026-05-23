from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .keyframes import Keyframe
from .sim3_lie import Sim3, axis_angle_to_R, pack_sim3, sim3_from_se3_matrix, sim3_residual, unpack_sim3, R_to_axis_angle


@dataclass
class Node:
    kf_id: int
    frame_idx: int
    p_init: np.ndarray  # (7,)
    p_opt: np.ndarray  # (7,)
    T_wc_init: np.ndarray  # (4,4) SE(3) at insertion time


@dataclass
class Edge:
    kind: str  # "tree" | "covisibility" | "loop"
    i_id: int
    j_id: int
    meas: np.ndarray  # (7,)
    weight: float = 1.0


@dataclass
class OptimizeStats:
    nodes: int
    tree_edges: int
    covis_edges: int
    loop_edges: int
    cost_initial: float
    cost_final: float
    success: bool
    n_params: int
    n_residuals: int
    nfev: int
    t_residual_s: float
    t_optimize_s: float
    # Backward-compatible: camera-center shift in world units.
    max_pose_shift: float
    mean_pose_shift: float
    rot_deg_max: float
    rot_deg_mean: float
    scale_log_max: float
    scale_log_mean: float
    dx_inf: float
    max_prior_deviation: float
    max_translation_jump_before: float
    max_translation_jump_after: float
    early_pose_shift_max: float


class PoseGraph:
    def __init__(
        self,
        odom_weight: float = 1.0,
        loop_weight: float = 3.0,
        pose_prior_weight: float | None = None,
        smoothness_weight_t: float | None = None,
        smoothness_weight_R: float | None = None,
        smoothness_weight_s: float | None = None,
    ):
        self.nodes: Dict[int, Node] = {}
        self._order: List[int] = []
        self.edges: List[Edge] = []
        self.odom_weight = float(odom_weight)
        self.loop_weight = float(loop_weight)
        self.pose_prior_weight = (
            float(1 * self.odom_weight)
            if pose_prior_weight is None
            else float(pose_prior_weight)
        )
        self.smoothness_weight_t = (
            float(10 * self.odom_weight)
            if smoothness_weight_t is None
            else float(smoothness_weight_t)
        )
        self.smoothness_weight_R = (
            float(1* self.odom_weight)
            if smoothness_weight_R is None
            else float(smoothness_weight_R)
        )
        self.smoothness_weight_s = (
            float(1 * self.odom_weight)
            if smoothness_weight_s is None
            else float(smoothness_weight_s)
        )
        self.last_opt_stats: Optional[OptimizeStats] = None

    def add_node_from_keyframe(self, kf: Keyframe):
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = np.asarray(kf.R_wc, dtype=np.float64).reshape(3, 3)
        T_wc[:3, 3] = np.asarray(kf.t_wc, dtype=np.float64).reshape(3)

        S = sim3_from_se3_matrix(T_wc)
        p = pack_sim3(S.R, S.t, S.s)
        node = Node(
            kf_id=int(kf.id),
            frame_idx=int(kf.frame_idx),
            p_init=p.copy(),
            p_opt=p.copy(),
            T_wc_init=T_wc.copy(),
        )
        if node.kf_id not in self.nodes:
            self.nodes[node.kf_id] = node
            self._order.append(node.kf_id)

    def add_tree_edge(self, parent_kf: Keyframe, child_kf: Keyframe):
        Ti = self.get_initial_T_wc(parent_kf.id)
        Tj = self.get_initial_T_wc(child_kf.id)
        Z = Tj @ np.linalg.inv(Ti)  # world->camera relative (SE3)
        meas = pack_sim3(Z[:3, :3], Z[:3, 3], 1.0)
        self.edges.append(
            Edge(kind="tree", i_id=int(parent_kf.id), j_id=int(child_kf.id), meas=meas, weight=self.odom_weight)
        )

    def add_covisibility_edge(self, kf_i: Keyframe, kf_j: Keyframe):
        Ti = self.get_initial_T_wc(kf_i.id)
        Tj = self.get_initial_T_wc(kf_j.id)
        Z = Tj @ np.linalg.inv(Ti)  # world->camera relative (SE3)
        meas = pack_sim3(Z[:3, :3], Z[:3, 3], 1.0)
        self.edges.append(
            Edge(kind="covisibility", i_id=int(kf_i.id), j_id=int(kf_j.id), meas=meas, weight=self.odom_weight)
        )

    def add_loop_edge(self, kf_i: Keyframe, kf_j: Keyframe, R: np.ndarray, t: np.ndarray, s: float):
        meas = pack_sim3(R, t, float(s))
        self.edges.append(
            Edge(kind="loop", i_id=int(kf_i.id), j_id=int(kf_j.id), meas=meas, weight=self.loop_weight)
        )

    def get_initial_T_wc(self, kf_id: int) -> np.ndarray:
        return self.nodes[int(kf_id)].T_wc_init

    def get_optimized_p(self, kf_id: int) -> np.ndarray:
        return self.nodes[int(kf_id)].p_opt

    def get_optimized_T_wc(self, kf_id: int) -> np.ndarray:
        S = unpack_sim3(self.nodes[int(kf_id)].p_opt)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = float(S.s) * np.asarray(S.R, dtype=np.float64)
        T[:3, 3] = np.asarray(S.t, dtype=np.float64)
        return T

    def reset_nodes_from_keyframes(self, keyframes: list) -> None:
        """
        Rebase existing nodes to the provided (corrected) keyframe poses.

        This is used after applying a global world correction to the frontend to
        keep subsequent odometry-edge initialization consistent with the corrected world.
        """
        for kf in keyframes:
            kf_id = int(getattr(kf, "id", -1))
            if kf_id not in self.nodes:
                continue
            T_wc = np.eye(4, dtype=np.float64)
            T_wc[:3, :3] = np.asarray(kf.R_wc, dtype=np.float64).reshape(3, 3)
            T_wc[:3, 3] = np.asarray(kf.t_wc, dtype=np.float64).reshape(3)
            p = pack_sim3(T_wc[:3, :3], T_wc[:3, 3], 1.0)
            node = self.nodes[kf_id]
            node.T_wc_init = T_wc.copy()
            node.p_init = p.copy()
            node.p_opt = p.copy()

    @property
    def loop_edges_count(self) -> int:
        return sum(1 for e in self.edges if e.kind == "loop")

    @property
    def tree_edges_count(self) -> int:
        return sum(1 for e in self.edges if e.kind == "tree")

    @property
    def covis_edges_count(self) -> int:
        return sum(1 for e in self.edges if e.kind == "covisibility")

    def optimize(
        self,
        max_nfev: int = 15,
        loss: str = "huber",
        f_scale: float = 1.0,
        verbose: int = 0,
    ) -> OptimizeStats:
        if len(self._order) < 2 or len(self.edges) == 0:
            stats = OptimizeStats(
                nodes=len(self._order),
                tree_edges=self.tree_edges_count,
                covis_edges=self.covis_edges_count,
                loop_edges=self.loop_edges_count,
                cost_initial=0.0,
                cost_final=0.0,
                success=True,
                n_params=0,
                n_residuals=0,
                nfev=0,
                t_residual_s=0.0,
                t_optimize_s=0.0,
                max_pose_shift=0.0,
                mean_pose_shift=0.0,
                rot_deg_max=0.0,
                rot_deg_mean=0.0,
                scale_log_max=0.0,
                scale_log_mean=0.0,
                dx_inf=0.0,
                max_prior_deviation=0.0,
                max_translation_jump_before=0.0,
                max_translation_jump_after=0.0,
                early_pose_shift_max=0.0,
            )
            self.last_opt_stats = stats
            return stats

        fixed_id = self._order[0]
        var_ids = self._order[1:]
        id_to_idx = {kf_id: i for i, kf_id in enumerate(var_ids)}

        x0 = np.concatenate([self.nodes[kf_id].p_opt for kf_id in var_ids], axis=0)

        # Precompute edge measurement components once (avoid per-eval unpack + Rodrigues).
        edges = list(self.edges)
        E = len(edges)
        meas_R = np.zeros((E, 3, 3), dtype=np.float64)
        meas_t = np.zeros((E, 3), dtype=np.float64)
        meas_log_s = np.zeros((E,), dtype=np.float64)
        meas_s = np.zeros((E,), dtype=np.float64)
        weights = np.zeros((E,), dtype=np.float64)
        i_pos = np.zeros((E,), dtype=np.int64)
        j_pos = np.zeros((E,), dtype=np.int64)

        all_ids = [fixed_id] + var_ids
        id_to_pos = {kf_id: i for i, kf_id in enumerate(all_ids)}

        for k, e in enumerate(edges):
            m = np.asarray(e.meas, dtype=np.float64).reshape(7)
            meas_R[k] = axis_angle_to_R(m[:3])
            meas_t[k] = m[3:6]
            meas_log_s[k] = m[6]
            meas_s[k] = float(np.exp(m[6]))
            weights[k] = float(e.weight)
            i_pos[k] = int(id_to_pos[e.i_id])
            j_pos[k] = int(id_to_pos[e.j_id])

        # Smoothness factors over consecutive keyframe triples (in insertion order).
        # 7 residuals per triple: 3 (t) + 3 (R) + 1 (log_s)
        n_smooth = max(0, len(self._order) - 2)

        # Jacobian sparsity:
        # - each edge depends on node i and node j (except fixed node0).
        # - each pose prior depends on its node.
        # - each smoothness factor depends on (i-1, i, i+1).
        n_params = int(x0.size)
        n_priors = int(len(var_ids))
        n_residuals = int(7 * E + 7 * n_priors + 7 * n_smooth)
        sparsity = lil_matrix((n_residuals, n_params), dtype=int)
        for k in range(E):
            row0 = 7 * k
            ii = int(edges[k].i_id)
            jj = int(edges[k].j_id)
            if ii != fixed_id:
                ci = int(id_to_idx[ii]) * 7
                sparsity[row0 : row0 + 7, ci : ci + 7] = 1
            if jj != fixed_id:
                cj = int(id_to_idx[jj]) * 7
                sparsity[row0 : row0 + 7, cj : cj + 7] = 1
        # Unary pose priors: each prior depends only on that node's 7 params.
        prior_row0 = 7 * E
        for i in range(n_priors):
            c = int(i) * 7
            r = prior_row0 + 7 * i
            sparsity[r : r + 7, c : c + 7] = 1

        smooth_row0 = 7 * E + 7 * n_priors
        for k in range(n_smooth):
            a_id = int(self._order[k])
            b_id = int(self._order[k + 1])
            c_id = int(self._order[k + 2])
            r0 = smooth_row0 + 7 * k
            for nid in (a_id, b_id, c_id):
                if nid == fixed_id:
                    continue
                col0 = int(id_to_idx[nid]) * 7
                sparsity[r0 : r0 + 7, col0 : col0 + 7] = 1

        fixed_p = np.asarray(self.nodes[fixed_id].p_opt, dtype=np.float64).reshape(7)
        p_init_var = np.vstack(
            [np.asarray(self.nodes[kf_id].p_init, dtype=np.float64).reshape(7) for kf_id in var_ids]
        )  # (Nvar,7)

        def residuals(x: np.ndarray) -> np.ndarray:
            # Unpack all node params once per call (avoid per-edge Rodrigues calls).
            x = np.asarray(x, dtype=np.float64)
            p_var = x.reshape(len(var_ids), 7)
            p_all = np.vstack([fixed_p, p_var])  # (N,7)

            w_all = p_all[:, :3]
            t_all = p_all[:, 3:6]
            log_s_all = p_all[:, 6]
            s_all = np.exp(log_s_all)

            N = p_all.shape[0]
            R_all = np.zeros((N, 3, 3), dtype=np.float64)
            for n in range(N):
                R_all[n] = axis_angle_to_R(w_all[n])

            # Precompute inverses for each node
            s_inv = 1.0 / s_all
            Rt_all = np.transpose(R_all, (0, 2, 1))
            t_inv = (-s_inv[:, None] * (Rt_all @ t_all[:, :, None])[:, :, 0]).reshape(N, 3)

            # Residuals:
            # - 7 per binary edge
            # - 7 per unary pose prior (soft regularization to initial VO pose)
            # - 7 per temporal smoothness factor (second differences)
            out = np.zeros((7 * E + 7 * n_priors + 7 * n_smooth,), dtype=np.float64)
            for k in range(E):
                ii = i_pos[k]
                jj = j_pos[k]

                # pred = T_j ∘ inv(T_i)
                s_pred = s_all[jj] * s_inv[ii]
                R_pred = R_all[jj] @ Rt_all[ii]
                t_pred = (s_all[jj] * (R_all[jj] @ t_inv[ii].reshape(3, 1))).reshape(3) + t_all[jj]

                # Group-consistent error: E = inv(Z_meas) ∘ Z_pred
                sm = meas_s[k]
                Rm = meas_R[k]
                tm = meas_t[k]
                s_err = s_pred / sm
                R_err = Rm.T @ R_pred
                t_err = (1.0 / sm) * (Rm.T @ (t_pred - tm).reshape(3, 1))
                w_err = R_to_axis_angle(R_err)
                log_s_err = float(np.log(float(s_err)))

                r = np.concatenate(
                    [w_err, t_err.reshape(3), np.array([log_s_err], dtype=np.float64)]
                )
                out[7 * k : 7 * k + 7] = np.sqrt(weights[k]) * r

            # Pose priors: r_prior_i = sqrt(w_prior) * (p_i - p_init_i)
            prior_start = 7 * E
            prior_end = 7 * E + 7 * n_priors
            smooth_start = prior_end

            if n_priors > 0 and self.pose_prior_weight > 0:
                w_prior = float(self.pose_prior_weight)
                scale = float(np.sqrt(w_prior))
                out[prior_start:prior_end] = scale * (
                    p_var.reshape(-1) - p_init_var.reshape(-1)
                )

            # Temporal smoothness: second difference on translation/log-scale, and
            # second difference on relative rotation in Lie algebra.
            if n_smooth > 0 and (
                self.smoothness_weight_t > 0
                or self.smoothness_weight_R > 0
                or self.smoothness_weight_s > 0
            ):
                wt = float(self.smoothness_weight_t)
                wR = float(self.smoothness_weight_R)
                ws = float(self.smoothness_weight_s)
                st = float(np.sqrt(wt)) if wt > 0 else 0.0
                sR = float(np.sqrt(wR)) if wR > 0 else 0.0
                ss = float(np.sqrt(ws)) if ws > 0 else 0.0

                for k in range(n_smooth):
                    a = int(id_to_pos[int(self._order[k])])
                    b = int(id_to_pos[int(self._order[k + 1])])
                    c = int(id_to_pos[int(self._order[k + 2])])

                    # Translation second difference in parameter space
                    dt2 = t_all[c] - 2.0 * t_all[b] + t_all[a]

                    # Rotation smoothness via relative rotations (axis-angle log map)
                    dR_prev = Rt_all[a] @ R_all[b]  # R_a^T R_b
                    dR_next = Rt_all[b] @ R_all[c]  # R_b^T R_c
                    w_prev = R_to_axis_angle(dR_prev)
                    w_next = R_to_axis_angle(dR_next)
                    dw = w_next - w_prev

                    # Scale smoothness on log-scale
                    dlog2 = log_s_all[c] - 2.0 * log_s_all[b] + log_s_all[a]

                    r0 = smooth_start + 7 * k
                    out[r0 : r0 + 3] = st * dt2
                    out[r0 + 3 : r0 + 6] = sR * dw
                    out[r0 + 6] = ss * float(dlog2)

            return out

        import time
        t0 = time.perf_counter()
        r0 = residuals(x0)
        t_res = time.perf_counter() - t0
        cost0 = float(np.sum(r0 * r0))

        t_opt0 = time.perf_counter()
        result = least_squares(
            residuals,
            x0,
            method="trf",
            jac_sparsity=sparsity,
            max_nfev=int(max_nfev),
            loss=loss,
            f_scale=float(f_scale),
            verbose=int(verbose),
        )
        t_opt = time.perf_counter() - t_opt0

        x_opt = result.x
        dx_inf = float(np.max(np.abs(x_opt - x0))) if x_opt.size else 0.0
        for i, kf_id in enumerate(var_ids):
            self.nodes[kf_id].p_opt = x_opt[i * 7 : (i + 1) * 7].copy()

        rf = residuals(x_opt)
        costf = float(np.sum(rf * rf))

        # Diagnostics: camera-center / rotation / scale changes.
        def _cam_center_from_p(p: np.ndarray) -> np.ndarray:
            p = np.asarray(p, dtype=np.float64).reshape(7)
            R = axis_angle_to_R(p[:3])
            t = p[3:6].reshape(3, 1)
            s = float(np.exp(p[6]))
            return (-(1.0 / s) * (R.T @ t)).reshape(3)

        def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
            R_rel = np.asarray(Ra) @ np.asarray(Rb).T
            cos_angle = (np.trace(R_rel) - 1.0) / 2.0
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            return float(np.degrees(np.arccos(cos_angle)))

        shiftC = []
        rot_deg = []
        scale_dlog = []
        for kf_id in self._order:
            p0 = np.asarray(self.nodes[kf_id].p_init, dtype=np.float64).reshape(7)
            p1 = np.asarray(self.nodes[kf_id].p_opt, dtype=np.float64).reshape(7)
            C0 = _cam_center_from_p(p0)
            C1 = _cam_center_from_p(p1)
            shiftC.append(float(np.linalg.norm(C1 - C0)))
            R0 = axis_angle_to_R(p0[:3])
            R1 = axis_angle_to_R(p1[:3])
            rot_deg.append(_rot_deg(R1, R0))
            scale_dlog.append(float(abs(p1[6] - p0[6])))
        max_shift = float(np.max(shiftC)) if shiftC else 0.0
        mean_shift = float(np.mean(shiftC)) if shiftC else 0.0
        rot_max = float(np.max(rot_deg)) if rot_deg else 0.0
        rot_mean = float(np.mean(rot_deg)) if rot_deg else 0.0
        slog_max = float(np.max(scale_dlog)) if scale_dlog else 0.0
        slog_mean = float(np.mean(scale_dlog)) if scale_dlog else 0.0

        prior_dev = []
        for kf_id in var_ids:
            p0 = np.asarray(self.nodes[kf_id].p_init, dtype=np.float64).reshape(7)
            p1 = np.asarray(self.nodes[kf_id].p_opt, dtype=np.float64).reshape(7)
            prior_dev.append(float(np.linalg.norm(p1 - p0)))
        max_prior_deviation = float(np.max(prior_dev)) if prior_dev else 0.0

        # Jump diagnostics in parameter-space translation (as requested).
        def _max_jump_from_p(get_p) -> float:
            jumps = []
            for k in range(len(self._order) - 1):
                a_id = int(self._order[k])
                b_id = int(self._order[k + 1])
                pa = np.asarray(get_p(a_id), dtype=np.float64).reshape(7)
                pb = np.asarray(get_p(b_id), dtype=np.float64).reshape(7)
                jumps.append(float(np.linalg.norm(pb[3:6] - pa[3:6])))
            return float(np.max(jumps)) if jumps else 0.0

        max_jump_before = _max_jump_from_p(lambda kf_id: self.nodes[int(kf_id)].p_init)
        max_jump_after = _max_jump_from_p(lambda kf_id: self.nodes[int(kf_id)].p_opt)

        # Early keyframes should move too (indicator for global propagation).
        early_k = min(10, len(self._order))
        early_shift = []
        for kf_id in self._order[:early_k]:
            p0 = np.asarray(self.nodes[int(kf_id)].p_init, dtype=np.float64).reshape(7)
            p1 = np.asarray(self.nodes[int(kf_id)].p_opt, dtype=np.float64).reshape(7)
            C0 = _cam_center_from_p(p0)
            C1 = _cam_center_from_p(p1)
            early_shift.append(float(np.linalg.norm(C1 - C0)))
        early_pose_shift_max = float(np.max(early_shift)) if early_shift else 0.0

        stats = OptimizeStats(
            nodes=len(self._order),
            tree_edges=self.tree_edges_count,
            covis_edges=self.covis_edges_count,
            loop_edges=self.loop_edges_count,
            cost_initial=cost0,
            cost_final=costf,
            success=bool(result.success),
            n_params=n_params,
            n_residuals=n_residuals,
            nfev=int(getattr(result, "nfev", 0)),
            t_residual_s=float(t_res),
            t_optimize_s=float(t_opt),
            max_pose_shift=max_shift,
            mean_pose_shift=mean_shift,
            rot_deg_max=rot_max,
            rot_deg_mean=rot_mean,
            scale_log_max=slog_max,
            scale_log_mean=slog_mean,
            dx_inf=dx_inf,
            max_prior_deviation=max_prior_deviation,
            max_translation_jump_before=max_jump_before,
            max_translation_jump_after=max_jump_after,
            early_pose_shift_max=early_pose_shift_max,
        )
        self.last_opt_stats = stats

        print(
            f"[PoseGraph] n_params={stats.n_params} n_residuals={stats.n_residuals} "
            f"nfev={stats.nfev} t_residual={stats.t_residual_s:.3f}s t_opt={stats.t_optimize_s:.3f}s "
            f"cost {stats.cost_initial:.1f} -> {stats.cost_final:.1f} "
            f"dx_inf={stats.dx_inf:.3e} "
            f"edges tree={stats.tree_edges} covis={stats.covis_edges} loop={stats.loop_edges} "
            f"shiftC max={stats.max_pose_shift:.3f} mean={stats.mean_pose_shift:.3f} "
            f"early_shift max={stats.early_pose_shift_max:.3f} "
            f"rot max={stats.rot_deg_max:.2f} mean={stats.rot_deg_mean:.2f} "
            f"dlogS max={stats.scale_log_max:.3f} mean={stats.scale_log_mean:.3f} "
            f"prior|max={stats.max_prior_deviation:.3f} "
            f"jump_t {stats.max_translation_jump_before:.2f}->{stats.max_translation_jump_after:.2f}"
        )
        return stats
