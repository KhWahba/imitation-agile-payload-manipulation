import argparse
import concurrent.futures as cf
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from payload_env import PayloadGymEnv
from expert_pcdbcbs_subprocess_dbg import PcDbCBSExpert, PcDbCBSPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
TEMPLATE_YAML = REPO_ROOT / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
PC_DBCBS_CFG = REPO_ROOT / "deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml"
OPT_CFG = REPO_ROOT / "deps/pc-dbCBS/configs/opt_training.yaml"
DYNOBENCH_BASE = REPO_ROOT / "deps/pc-dbCBS/deps/dynoplan/dynobench"
BUILD_DIR = REPO_ROOT / "deps/pc-dbCBS/build"


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_plan_dirs(run_tmp_dir: Path):
    rows = []
    for d in sorted(run_tmp_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.yaml"
        res_path = d / "result_subprocess_api.yaml"
        if not meta_path.exists() or not res_path.exists():
            continue
        meta = _load_yaml(meta_path) or {}
        res = _load_yaml(res_path) or {}
        x = np.asarray(res.get("X", []), dtype=np.float32)
        u = np.asarray(res.get("U", []), dtype=np.float32)
        if x.ndim != 2 or x.size == 0:
            continue
        rows.append(
            {
                "dir": str(d),
                "episode_id": int(meta.get("episode_id", -1)),
                "global_step": int(meta.get("global_step", -1)),
                "reason": str(meta.get("reason", "")),
                "dist_to_goal": float(meta.get("dist_to_goal", np.nan)),
                "warmstart": bool(meta.get("warmstart_optimization", False)),
                "X": x,
                "U": u if u.ndim == 2 else np.zeros((0, 0), dtype=np.float32),
            }
        )
    rows.sort(key=lambda r: (r["episode_id"], r["global_step"], r["dir"]))
    return rows


def _split_joint_state(X: np.ndarray):
    """Parse poses-first joint state layout: [7*n poses, 6*n vels]."""
    if X.ndim != 2 or X.shape[1] % 13 != 0:
        raise ValueError(f"Expected X shape (T, 13*n), got {X.shape}")
    n_bodies = X.shape[1] // 13
    poses = X[:, : 7 * n_bodies]
    vels = X[:, 7 * n_bodies :]

    payload_pos = poses[:, 0:3]
    payload_vel = vels[:, 0:3]
    quad_pos = []
    quad_vel = []
    for qi in range(n_bodies - 1):
        pbase = 7 * (1 + qi)
        vbase = 6 * (1 + qi)
        quad_pos.append(poses[:, pbase : pbase + 3])
        quad_vel.append(vels[:, vbase : vbase + 3])
    return payload_pos, payload_vel, quad_pos, quad_vel


def _extract_from_obs(raw_obs: np.ndarray, n_bodies: int):
    """Parse env raw_state prefix (13*n) with poses-first layout."""
    raw = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
    if raw.size < 13 * n_bodies:
        raise ValueError(f"raw obs too short: {raw.size} < {13*n_bodies}")
    x = raw[: 13 * n_bodies][None, :]
    payload_pos, payload_vel, quad_pos, quad_vel = _split_joint_state(x)
    return payload_pos[0], payload_vel[0], [q[0] for q in quad_pos], [q[0] for q in quad_vel]


def _plot_run(run_dir: Path, run_idx: int, plan_rows, rollout):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_payload_xy = axes[0, 0]
    ax_quads_xy = axes[0, 1]
    ax_z = axes[0, 2]
    ax_payload_speed = axes[1, 0]
    ax_quad_speed = axes[1, 1]
    ax_metrics = axes[1, 2]
    cmap = plt.get_cmap("viridis")
    n = max(len(plan_rows), 1)
    for i, row in enumerate(plan_rows):
        x = row["X"]
        color = cmap(i / max(n - 1, 1))
        payload_pos, payload_vel, quad_pos, quad_vel = _split_joint_state(x)

        ax_payload_xy.plot(payload_pos[:, 0], payload_pos[:, 1], color=color, alpha=0.9, lw=1.8)
        ax_payload_xy.scatter(payload_pos[0, 0], payload_pos[0, 1], color=color, s=12)
        ax_payload_xy.scatter(payload_pos[-1, 0], payload_pos[-1, 1], color=color, s=12, marker="x")

        styles = ["-", "--", ":", "-."]
        for qi, qp in enumerate(quad_pos):
            ax_quads_xy.plot(
                qp[:, 0], qp[:, 1],
                color=color, alpha=0.8, lw=1.2, linestyle=styles[qi % len(styles)]
            )
            ax_z.plot(qp[:, 2], color=color, alpha=0.7, lw=1.0, linestyle=styles[qi % len(styles)])
            ax_quad_speed.plot(
                np.linalg.norm(quad_vel[qi], axis=1),
                color=color, alpha=0.6, lw=1.0, linestyle=styles[qi % len(styles)]
            )

        ax_z.plot(payload_pos[:, 2], color=color, alpha=0.9, lw=1.8)
        ax_payload_speed.plot(np.linalg.norm(payload_vel, axis=1), color=color, alpha=0.8, lw=1.4)

    ax_payload_xy.scatter([0.0], [0.0], marker="*", s=120, c="red", label="goal")
    ax_payload_xy.set_title(f"Run {run_idx}: Payload XY (all replans)")
    ax_payload_xy.set_xlabel("x")
    ax_payload_xy.set_ylabel("y")
    ax_payload_xy.grid(True, alpha=0.3)
    ax_payload_xy.legend(loc="best")

    ax_quads_xy.set_title(f"Run {run_idx}: Quad XY (all replans)")
    ax_quads_xy.set_xlabel("x")
    ax_quads_xy.set_ylabel("y")
    ax_quads_xy.grid(True, alpha=0.3)

    ax_z.set_title("Z trajectories (payload + quads)")
    ax_z.set_xlabel("plan timestep")
    ax_z.set_ylabel("z")
    ax_z.grid(True, alpha=0.3)

    ax_payload_speed.set_title("Payload speed norm")
    ax_payload_speed.set_xlabel("plan timestep")
    ax_payload_speed.set_ylabel("||v_payload||")
    ax_payload_speed.grid(True, alpha=0.3)

    ax_quad_speed.set_title("Quad speed norms")
    ax_quad_speed.set_xlabel("plan timestep")
    ax_quad_speed.set_ylabel("||v_quad||")
    ax_quad_speed.grid(True, alpha=0.3)

    if plan_rows:
        r_idx = np.arange(len(plan_rows))
        dists = np.array([r["dist_to_goal"] for r in plan_rows], dtype=np.float32)
        horizons = np.array([int(r["U"].shape[0]) for r in plan_rows], dtype=np.float32)
        u0_norm = np.array(
            [float(np.linalg.norm(r["U"][0])) if r["U"].shape[0] else np.nan for r in plan_rows],
            dtype=np.float32,
        )
        ax_metrics.plot(r_idx, dists, "-o", ms=3, label="replan dist_to_goal")
        ax_metrics.plot(r_idx, horizons, "-s", ms=3, label="plan horizon |U|")
        ax_metrics.plot(r_idx, u0_norm, "-^", ms=3, label="||U[0]||")
        ax_metrics.set_title(f"Run {run_idx}: Replan Metrics ({len(plan_rows)} replans)")
        ax_metrics.set_xlabel("replan index")
        ax_metrics.grid(True, alpha=0.3)
        ax_metrics.legend(loc="best")
    else:
        ax_metrics.set_title("No replans found")

    # Highlight rollout replan instants and per-axis payload/quad velocities
    if rollout:
        t = np.array([r["t"] for r in rollout], dtype=int)
        replan_t = t[np.array([bool(r.get("just_replanned", False)) for r in rollout])]
        pvel = np.array([r["payload_vel"] for r in rollout], dtype=np.float32)
        q1vel = np.array([r["quad_vels"][0] for r in rollout], dtype=np.float32) if rollout[0]["quad_vels"] else None
        q2vel = np.array([r["quad_vels"][1] for r in rollout], dtype=np.float32) if rollout[0]["quad_vels"] and len(rollout[0]["quad_vels"]) > 1 else None
        ax_payload_speed.clear()
        ax_payload_speed.plot(t, pvel[:, 0], label="vx")
        ax_payload_speed.plot(t, pvel[:, 1], label="vy")
        ax_payload_speed.plot(t, pvel[:, 2], label="vz")
        for rt in replan_t:
            ax_payload_speed.axvline(rt, color="k", alpha=0.12, lw=1)
        ax_payload_speed.set_title("Rollout payload velocity (per axis)")
        ax_payload_speed.set_xlabel("env timestep")
        ax_payload_speed.set_ylabel("m/s")
        ax_payload_speed.grid(True, alpha=0.3)
        ax_payload_speed.legend(loc="best")

        ax_quad_speed.clear()
        if q1vel is not None:
            ax_quad_speed.plot(t, q1vel[:, 0], label="q1 vx", alpha=0.9)
            ax_quad_speed.plot(t, q1vel[:, 1], label="q1 vy", alpha=0.9)
            ax_quad_speed.plot(t, q1vel[:, 2], label="q1 vz", alpha=0.9)
        if q2vel is not None:
            ax_quad_speed.plot(t, q2vel[:, 0], "--", label="q2 vx", alpha=0.9)
            ax_quad_speed.plot(t, q2vel[:, 1], "--", label="q2 vy", alpha=0.9)
            ax_quad_speed.plot(t, q2vel[:, 2], "--", label="q2 vz", alpha=0.9)
        for rt in replan_t:
            ax_quad_speed.axvline(rt, color="k", alpha=0.12, lw=1)
        ax_quad_speed.set_title("Rollout quad velocities (per axis)")
        ax_quad_speed.set_xlabel("env timestep")
        ax_quad_speed.set_ylabel("m/s")
        ax_quad_speed.grid(True, alpha=0.3)
        ax_quad_speed.legend(loc="best", ncol=2, fontsize=8)

        for rt in replan_t:
            ax_metrics.axvline(np.where(r_idx == r_idx)[0][0] if plan_rows else 0, alpha=0)  # no-op keep axis live

    fig.suptitle(f"Expert Variance Test Run {run_idx} (env_steps={len(rollout)})", y=0.99)
    fig.tight_layout()
    out_path = run_dir / f"run_{run_idx:02d}_replans.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_combined(out_dir: Path, all_runs):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_payload_xy = axes[0, 0]
    ax_quad_xy = axes[0, 1]
    ax_u = axes[1, 0]
    ax_speed = axes[1, 1]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    max_replans = 0
    first_action_norms = []
    for run_idx, run_data in enumerate(all_runs):
        plan_rows = run_data["plans"]
        max_replans = max(max_replans, len(plan_rows))
        norms = []
        for j, row in enumerate(plan_rows):
            x = row["X"]
            payload_pos, payload_vel, quad_pos, quad_vel = _split_joint_state(x)
            alpha = 0.2 + 0.8 * ((j + 1) / max(len(plan_rows), 1))
            color = colors[run_idx % len(colors)]
            ax_payload_xy.plot(payload_pos[:, 0], payload_pos[:, 1], color=color, alpha=alpha, lw=1.2)
            styles = ["-", "--", ":", "-."]
            for qi, qp in enumerate(quad_pos):
                ax_quad_xy.plot(
                    qp[:, 0], qp[:, 1], color=color, alpha=alpha, lw=1.0,
                    linestyle=styles[qi % len(styles)]
                )
            ax_speed.plot(payload_vel[:, 0], color=color, alpha=0.25, lw=0.8)
            norms.append(float(np.linalg.norm(row["U"][0])) if row["U"].shape[0] else np.nan)
        first_action_norms.append(np.asarray(norms, dtype=np.float32))

    ax_payload_xy.scatter([0.0], [0.0], marker="*", s=120, c="red", label="goal")
    ax_payload_xy.set_title("All runs: Payload XY (all replans)")
    ax_payload_xy.set_xlabel("x")
    ax_payload_xy.set_ylabel("y")
    ax_payload_xy.grid(True, alpha=0.3)
    ax_payload_xy.legend(loc="best")

    ax_quad_xy.set_title("All runs: Quad XY (all replans)")
    ax_quad_xy.set_xlabel("x")
    ax_quad_xy.set_ylabel("y")
    ax_quad_xy.grid(True, alpha=0.3)

    for run_idx, norms in enumerate(first_action_norms):
        if norms.size == 0:
            continue
        ax_u.plot(np.arange(norms.size), norms, marker="o", ms=3, label=f"run{run_idx}")
    if max_replans > 0:
        # compare only shared prefix for variance signal
        shared = [a[: min(len(a) for a in first_action_norms if len(a) > 0)] for a in first_action_norms if len(a) > 0]
        if len(shared) >= 2 and shared[0].size > 0:
            shared_mat = np.stack(shared, axis=0)
            mean = np.nanmean(shared_mat, axis=0)
            std = np.nanstd(shared_mat, axis=0)
            t = np.arange(mean.size)
            ax_u.plot(t, mean, color="black", lw=2.0, label="mean ||U[0]||")
            ax_u.fill_between(t, mean - std, mean + std, color="gray", alpha=0.25, label="±1σ")
    ax_u.set_title("First-Action Norm by Replan (compare runs)")
    ax_u.set_xlabel("replan index")
    ax_u.set_ylabel("||U[0]||")
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(loc="best")

    ax_speed.set_title("Payload vx trajectories (all replans)")
    ax_speed.set_xlabel("plan timestep")
    ax_speed.set_ylabel("vx")
    ax_speed.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "all_runs_overlay.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_rollout_stats(out_dir: Path, all_runs):
    """Average/std across runs for rollout-level states (payload + quads) with replan markers."""
    # Align by rollout timestep prefix
    min_len = min(len(r["rollout"]) for r in all_runs)
    if min_len <= 0:
        return None
    ts = np.arange(min_len)
    payload_pos = np.array([[rr["payload_pos"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
    payload_vel = np.array([[rr["payload_vel"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
    # assume same number of quads across runs
    n_quads = len(all_runs[0]["rollout"][0]["quad_positions"])
    quad_pos = [
        np.array([[rr["quad_positions"][qi] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
        for qi in range(n_quads)
    ]
    quad_vel = [
        np.array([[rr["quad_vels"][qi] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
        for qi in range(n_quads)
    ]
    replan_mask = np.array([bool(rr.get("just_replanned", False)) for rr in all_runs[0]["rollout"][:min_len]])
    replan_ts = ts[replan_mask]

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    ax_pp = axes[0, 0]
    ax_pv = axes[0, 1]
    ax_qp = axes[1, 0]
    ax_qv = axes[1, 1]
    ax_dist = axes[2, 0]
    ax_rew = axes[2, 1]

    def plot_mean_std(ax, arr, labels, title, ylabel):
        # arr: [runs, T, D]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        for d in range(arr.shape[2]):
            ax.plot(ts, mean[:, d], label=labels[d])
            ax.fill_between(ts, mean[:, d] - std[:, d], mean[:, d] + std[:, d], alpha=0.2)
        for rt in replan_ts:
            ax.axvline(rt, color="k", alpha=0.08, lw=1)
        ax.set_title(title)
        ax.set_xlabel("env timestep")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plot_mean_std(ax_pp, payload_pos, ["px", "py", "pz"], "Payload position mean±std", "m")
    plot_mean_std(ax_pv, payload_vel, ["vx", "vy", "vz"], "Payload velocity mean±std", "m/s")

    if n_quads >= 1:
        plot_mean_std(ax_qp, quad_pos[0], ["q1 x", "q1 y", "q1 z"], "Quad1 position mean±std", "m")
        plot_mean_std(ax_qv, quad_vel[0], ["q1 vx", "q1 vy", "q1 vz"], "Quad1 velocity mean±std", "m/s")
    else:
        ax_qp.set_visible(False)
        ax_qv.set_visible(False)

    dists = np.array([[rr["payload_goal_dist"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
    rews = np.array([[rr["rew"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
    for ax, arr, title, ylabel in [
        (ax_dist, dists, "Payload goal distance mean±std", "m"),
        (ax_rew, rews, "Reward mean±std", "reward"),
    ]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(ts, mean, color="tab:blue")
        ax.fill_between(ts, mean - std, mean + std, color="tab:blue", alpha=0.2)
        for rt in replan_ts:
            ax.axvline(rt, color="k", alpha=0.08, lw=1)
        ax.set_title(title)
        ax.set_xlabel("env timestep")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "rollout_mean_std.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_diversity_style(out_dir: Path, all_runs):
    """Cleaner analyze_expert_trajs-style plots for rollout/replan variance."""
    min_len = min(len(r["rollout"]) for r in all_runs)
    if min_len <= 0:
        return []
    n_runs = len(all_runs)
    ts = np.arange(min_len)
    replan_mask = np.array([bool(rr.get("just_replanned", False)) for rr in all_runs[0]["rollout"][:min_len]])
    replan_ts = ts[replan_mask]

    payload_pos = np.array([[rr["payload_pos"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
    payload_vel = np.array([[rr["payload_vel"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
    n_quads = len(all_runs[0]["rollout"][0]["quad_positions"])
    quad_pos = [
        np.array([[rr["quad_positions"][qi] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
        for qi in range(n_quads)
    ]
    quad_vel = [
        np.array([[rr["quad_vels"][qi] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
        for qi in range(n_quads)
    ]
    rews = np.array([[rr["rew"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)
    dists = np.array([[rr["payload_goal_dist"] for rr in r["rollout"][:min_len]] for r in all_runs], dtype=np.float32)

    outputs = []

    def _plot_xyz(arr, title, yprefix, path_name):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        labels = ["x", "y", "z"]
        for k in range(3):
            for i in range(n_runs):
                axs[k].plot(ts, arr[i, :, k], color="tab:blue", alpha=0.12, lw=0.7)
            axs[k].plot(ts, mean[:, k], color="red", lw=1.8, label="mean")
            axs[k].fill_between(ts, mean[:, k] - 2 * std[:, k], mean[:, k] + 2 * std[:, k],
                                color="red", alpha=0.15, label="±2σ")
            for rt in replan_ts:
                axs[k].axvline(rt, color="k", alpha=0.08, lw=1)
            axs[k].set_title(f"{yprefix} {labels[k]}")
            axs[k].set_xlabel("env step")
            axs[k].grid(True, alpha=0.3)
            axs[k].legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        path = out_dir / path_name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)

    _plot_xyz(payload_pos, f"Payload Position Across {n_runs} Runs", "p", "rollout_payload_pos_diversity.png")
    _plot_xyz(payload_vel, f"Payload Velocity Across {n_runs} Runs", "v", "rollout_payload_vel_diversity.png")
    for qi in range(n_quads):
        _plot_xyz(quad_pos[qi], f"Quad {qi+1} Position Across {n_runs} Runs", f"q{qi+1} p", f"rollout_quad{qi+1}_pos_diversity.png")
        _plot_xyz(quad_vel[qi], f"Quad {qi+1} Velocity Across {n_runs} Runs", f"q{qi+1} v", f"rollout_quad{qi+1}_vel_diversity.png")

    # Reward and distance (cleaner than cluttered multi-axis plot)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, arr, title, ylabel, color in [
        (axs[0], dists, "Payload Goal Distance", "m", "tab:green"),
        (axs[1], rews, "Reward", "reward", "tab:purple"),
    ]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        for i in range(n_runs):
            ax.plot(ts, arr[i], color=color, alpha=0.08, lw=0.7)
        ax.plot(ts, mean, color="black", lw=1.8, label="mean")
        ax.fill_between(ts, mean - 2 * std, mean + 2 * std, color=color, alpha=0.15, label="±2σ")
        for rt in replan_ts:
            ax.axvline(rt, color="k", alpha=0.08, lw=1)
        ax.set_title(title)
        ax.set_xlabel("env step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "rollout_reward_distance_diversity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    # Replan first actions across runs (analyze_expert_trajs-like)
    min_replans = min(len(r["plans"]) for r in all_runs)
    std = None
    if min_replans > 0:
        u0s = []
        for r in all_runs:
            vecs = []
            for p in r["plans"][:min_replans]:
                U = p["U"]
                if U.shape[0] == 0:
                    vecs.append(np.full((U.shape[1] if U.ndim == 2 else 8,), np.nan, dtype=np.float32))
                else:
                    vecs.append(U[0].astype(np.float32))
            u0s.append(np.stack(vecs, axis=0))
        u0s = np.stack(u0s, axis=0)  # [runs, replans, nu]
        nu = u0s.shape[2]
        nrows = int(np.ceil(nu / 4))
        fig, axs = plt.subplots(nrows, 4, figsize=(16, 3.3 * nrows), sharex=True)
        axs = np.asarray(axs).reshape(nrows, 4)
        t_rep = np.arange(min_replans)
        mean = np.nanmean(u0s, axis=0)
        std = np.nanstd(u0s, axis=0)
        for j in range(nrows * 4):
            ax = axs[j // 4, j % 4]
            if j >= nu:
                ax.axis("off")
                continue
            for i in range(n_runs):
                ax.plot(t_rep, u0s[i, :, j], color="tab:blue", alpha=0.12, lw=0.7)
            ax.plot(t_rep, mean[:, j], color="red", lw=1.5)
            ax.fill_between(t_rep, mean[:, j] - 2 * std[:, j], mean[:, j] + 2 * std[:, j],
                            color="red", alpha=0.15)
            ax.set_title(f"u0[{j}]")
            ax.grid(True, alpha=0.3)
        for ax in axs[-1]:
            ax.set_xlabel("replan index")
        fig.suptitle("First Action Components Across Runs (mean ± 2σ)")
        fig.tight_layout()
        path = out_dir / "replan_u0_components_diversity.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)

        # σ over replan index
        fig, ax = plt.subplots(figsize=(10, 4))
        mean_sigma = np.nanmean(std, axis=1)
        max_sigma = np.nanmax(std, axis=1)
        ax.plot(t_rep, mean_sigma, "r-", lw=1.5, label="mean σ across motors")
        ax.plot(t_rep, max_sigma, "r--", lw=1.2, alpha=0.7, label="max σ")
        ax.set_xlabel("replan index")
        ax.set_ylabel("σ of u0 across runs")
        ax.set_title("First-Action Spread Over Replans")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / "replan_u0_std_over_index.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)

        # Pairwise u0 distance over replans
        pairs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                pairs.append(np.linalg.norm(u0s[i] - u0s[j], axis=1))
        if pairs:
            pair_d = np.stack(pairs, axis=0)
            fig, ax = plt.subplots(figsize=(10, 4))
            mean_pair = pair_d.mean(axis=0)
            p25 = np.percentile(pair_d, 25, axis=0)
            p75 = np.percentile(pair_d, 75, axis=0)
            ax.plot(t_rep, mean_pair, color="tab:blue", lw=1.5, label="mean pairwise L2")
            ax.fill_between(t_rep, p25, p75, color="tab:blue", alpha=0.15, label="25-75th pct")
            ax.set_xlabel("replan index")
            ax.set_ylabel("pairwise ||u0_i-u0_j||")
            ax.set_title("Pairwise First-Action Distance Across Runs")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = out_dir / "replan_pairwise_u0_distance.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(path)

    # Text report similar to analyze_expert_trajs
    report = []
    report.append("=" * 60)
    report.append("EXPERT VARIANCE REPORT (ROLLOUT + REPLANS)")
    report.append("=" * 60)
    report.append(f"Runs: {n_runs}")
    report.append(f"Aligned rollout length (min across runs): {min_len}")
    report.append(f"Replan instants (from run0 within aligned prefix): {replan_ts.tolist()}")
    report.append("")
    report.append("Payload position spread (σ across runs):")
    for k, name in enumerate(["x", "y", "z"]):
        s = payload_pos.std(axis=0)[:, k]
        report.append(f"  p{name}: mean_σ={s.mean():.6f}, max_σ={s.max():.6f} at t={int(s.argmax())}")
    report.append("Payload velocity spread (σ across runs):")
    for k, name in enumerate(["x", "y", "z"]):
        s = payload_vel.std(axis=0)[:, k]
        report.append(f"  v{name}: mean_σ={s.mean():.6f}, max_σ={s.max():.6f} at t={int(s.argmax())}")
    report.append("")
    if min_replans > 0 and std is not None:
        report.append(f"Aligned replans across runs: {min_replans}")
        report.append(f"u0 mean σ across motors: {float(np.nanmean(std)):.6f}")
        report.append(f"u0 max σ across motors/replans: {float(np.nanmax(std)):.6f}")
        report.append(f"u0 mean σ at first replan: {float(np.nanmean(std[0])):.6f}")
        report.append(f"u0 mean σ at last aligned replan: {float(np.nanmean(std[-1])):.6f}")
    report.append("=" * 60)
    report_text = "\n".join(report)
    (out_dir / "expert_variance_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    outputs.append(out_dir / "expert_variance_report.txt")

    return outputs


def _rollout_once(run_idx: int, out_dir_str: str, total_steps: int, replan_k: int, n_opt: int,
                  time_limit_ms: float, deterministic: bool, max_steps: int, stream_logs: bool):
    out_dir = Path(out_dir_str)
    run_dir = out_dir / f"run_{run_idx:02d}"
    tmp_dir = run_dir / "tmp_pcdbcbs"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Child planner subprocess inherits these env vars from this worker process.
    os.environ["PCDBCBS_EXPERT_SUBPROCESS_BIN"] = str(BUILD_DIR / "pc_dbcbs_expert_dbg")
    os.environ["PCDBCBS_EXPERT_SUBPROCESS_STREAM_LOGS"] = "1" if stream_logs else "0"
    os.environ["PCDBCBS_DETERMINISTIC"] = "1" if deterministic else "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    env = PayloadGymEnv(
        xml_path=str(XML_PATH),
        template_yaml_path=str(TEMPLATE_YAML),
        max_steps=max_steps,
    )
    env.terminate_on_success = True

    nu = env.action_dim
    act_low = np.zeros(nu, dtype=np.float32)
    act_high = np.ones(nu, dtype=np.float32) * 1.4

    expert = PcDbCBSExpert(
        paths=PcDbCBSPaths(
            bindings_path=str(BUILD_DIR),
            input_yaml=str(TEMPLATE_YAML),
            pc_dbcbs_cfg_yaml=str(PC_DBCBS_CFG),
            opt_cfg_yaml=str(OPT_CFG),
            dynobench_base=str(DYNOBENCH_BASE) + "/",
            motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
            time_limit=time_limit_ms,
            work_dir_root=str(tmp_dir),
            keep_files=True,
            warmstart_optimization=True,
            N_opt=n_opt,
        ),
        act_low=act_low,
        act_high=act_high,
        replan_every_k=replan_k,
    )

    obs, _ = env.reset(seed=0)
    qpos0 = env.data.qpos.copy().astype(np.float32)
    qvel0 = env.data.qvel.copy().astype(np.float32)
    expert.reset_episode()

    rollout = []
    acts = []
    n_bodies = env.n_bodies
    for t in range(total_steps):
        act = expert.act(obs)
        acts.append(np.asarray(act, dtype=np.float32).copy())
        next_obs, rew, terminated, truncated, info = env.step(act)
        payload_pos, payload_vel, quad_positions, quad_vels = _extract_from_obs(next_obs, n_bodies)
        rollout.append(
            {
                "t": t,
                "rew": float(rew),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "just_replanned": bool(expert.just_replanned),
                "reason": expert.last_replan_reason,
                "payload_pos": payload_pos.astype(float).tolist(),
                "payload_vel": payload_vel.astype(float).tolist(),
                "quad_positions": [q.astype(float).tolist() for q in quad_positions],
                "quad_vels": [q.astype(float).tolist() for q in quad_vels],
                "payload_goal_dist": float(np.linalg.norm(payload_pos - env.goal[:3])),
                "quad_oob": bool((info or {}).get("quad_out_of_bounds", False)),
                "payload_oob": bool((info or {}).get("payload_out_of_bounds", False)),
            }
        )
        obs = next_obs
        if terminated or truncated:
            break

    with open(run_dir / "rollout.json", "w", encoding="utf-8") as f:
        json.dump(rollout, f, indent=2)
    np.savez_compressed(
        run_dir / "rollout_actions.npz",
        qpos0=qpos0,
        qvel0=qvel0,
        u_policy=np.asarray(acts, dtype=np.float32),
    )

    plans = _parse_plan_dirs(tmp_dir)
    with open(run_dir / "plan_index.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "dir": p["dir"],
                    "episode_id": p["episode_id"],
                    "global_step": p["global_step"],
                    "reason": p["reason"],
                    "dist_to_goal": p["dist_to_goal"],
                    "x_shape": list(p["X"].shape),
                    "u_shape": list(p["U"].shape),
                }
                for p in plans
            ],
            f,
            indent=2,
        )

    _plot_run(run_dir, run_idx, plans, rollout)
    return {
        "run_idx": run_idx,
        "run_dir": str(run_dir),
        "rollout_steps": len(rollout),
        "plans": plans,
        "rollout": rollout,
    }


def _load_existing_results(out_dir: Path):
    results = []
    for run_dir in sorted(out_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        tmp_dir = run_dir / "tmp_pcdbcbs"
        rollout_path = run_dir / "rollout.json"
        if not tmp_dir.exists() or not rollout_path.exists():
            continue
        rollout = json.loads(rollout_path.read_text())
        plans = _parse_plan_dirs(tmp_dir)
        run_idx = int(run_dir.name.split("_")[-1])
        results.append(
            {
                "run_idx": run_idx,
                "run_dir": str(run_dir),
                "rollout_steps": len(rollout),
                "plans": plans,
                "rollout": rollout,
            }
        )
    results.sort(key=lambda r: r["run_idx"])
    return results


def main():
    ap = argparse.ArgumentParser(description="Run 3 expert-only rollouts in parallel and compare replans.")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Env rollout steps per run (stops earlier on env termination/truncation).",
    )
    ap.add_argument("--replan-k", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--n-opt", type=int, default=30)
    ap.add_argument("--time-limit-ms", type=float, default=50000.0)
    ap.add_argument("--deterministic", action="store_true", help="Set PCDBCBS_DETERMINISTIC=1")
    ap.add_argument("--stream-logs", action="store_true", help="Stream child planner logs (noisy).")
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--analyze-existing", type=str, default="", help="Only regenerate plots/reports from an existing output dir.")
    args = ap.parse_args()

    if args.analyze_existing:
        out_dir = Path(args.analyze_existing)
        results = _load_existing_results(out_dir)
        if not results:
            raise RuntimeError(f"No run_* results found in {out_dir}")
        _plot_combined(out_dir, results)
        _plot_rollout_stats(out_dir, results)
        _plot_diversity_style(out_dir, results)
        print(f"[expert_variance] re-analyzed existing outputs in: {out_dir}")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "runs" / "expert_variance" / f"k{args.replan_k}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    worker_args = dict(
        out_dir_str=str(out_dir),
        total_steps=args.steps,
        replan_k=args.replan_k,
        n_opt=args.n_opt,
        time_limit_ms=args.time_limit_ms,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        stream_logs=args.stream_logs,
    )

    results = []
    with cf.ProcessPoolExecutor(max_workers=args.runs) as ex:
        futs = [ex.submit(_rollout_once, i, **worker_args) for i in range(args.runs)]
        for fut in cf.as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["run_idx"])
    _plot_combined(out_dir, results)
    _plot_rollout_stats(out_dir, results)
    _plot_diversity_style(out_dir, results)

    summary = []
    for r in results:
        plan_rows = r["plans"]
        summary.append(
            {
                "run_idx": r["run_idx"],
                "rollout_steps": r["rollout_steps"],
                "replans": len(plan_rows),
                "reasons": [p["reason"] for p in plan_rows],
                "global_steps": [p["global_step"] for p in plan_rows],
                "first_action_norms": [
                    float(np.linalg.norm(p["U"][0])) if p["U"].shape[0] else None
                    for p in plan_rows
                ],
            }
        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[expert_variance] wrote outputs to: {out_dir}")
    for row in summary:
        print(
            f"  run{row['run_idx']}: steps={row['rollout_steps']} replans={row['replans']} "
            f"global_steps={row['global_steps']}"
        )


if __name__ == "__main__":
    main()
