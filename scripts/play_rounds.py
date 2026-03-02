#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _round_dirs(demos_root: Path) -> list[Path]:
    return sorted([p for p in demos_root.glob("round-*") if p.is_dir()])


def _demo_dirs(round_dir: Path) -> list[Path]:
    return sorted([p for p in round_dir.iterdir() if p.is_dir() and p.name.endswith(".npz")])


def _load_dataset(demo_dir: Path):
    from datasets import load_from_disk

    return load_from_disk(str(demo_dir))


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def _raw_state_dim_from_row(row: dict[str, Any]) -> int:
    # Current env uses raw_state prefix of length 39 for 2 quads.
    # Infer from action dim: state dim + learner_features_dim = obs_dim,
    # learner_features_dim = 6 + 12*n_quads + action_dim, action_dim=4*n_quads.
    # For now, use explicit prefix since this project is fixed to 2 quads in this training setup.
    obs0 = row["obs"][0]
    if len(obs0) >= 39:
        return 39
    raise ValueError(f"Unexpected obs dim {len(obs0)}; cannot infer raw state prefix.")


def _raw_state_to_qpos_qvel(raw_state: np.ndarray, n_bodies: int) -> tuple[np.ndarray, np.ndarray]:
    raw_state = np.asarray(raw_state, dtype=np.float64).reshape(-1)
    expected = 7 * n_bodies + 6 * n_bodies
    if raw_state.size != expected:
        raise ValueError(f"raw_state len={raw_state.size}, expected {expected}")
    poses = raw_state[: 7 * n_bodies].copy()  # pos + quat_xyzw
    vels = raw_state[7 * n_bodies :].copy()

    # Convert each quat from xyzw -> wxyz for MuJoCo qpos
    for i in range(n_bodies):
        b = 7 * i
        qx, qy, qz, qw = poses[b + 3 : b + 7]
        poses[b + 3 : b + 7] = [qw, qx, qy, qz]
    return poses, vels


def _map_policy_action_to_mujoco_ctrl(a_policy: np.ndarray, action_dim: int) -> np.ndarray:
    # Matches PayloadGymEnv.step() mapping.
    a = np.asarray(a_policy, dtype=np.float32).reshape(action_dim)
    # a = np.clip(a, -1.0, 1.0)
    planner_low = np.zeros((action_dim,), dtype=np.float32)
    planner_high = 1.4 * np.ones((action_dim,), dtype=np.float32)
    act_mid = 0.5 * (planner_low + planner_high)
    act_half = 0.5 * (planner_high - planner_low)
    action_planner = a * act_half + act_mid
    u_nominal = np.float32(0.034 * 9.81 / 4.0)
    return (action_planner * u_nominal).astype(np.float32)


def cmd_plot_rewards(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    demos_root = Path(args.demos_root)
    rounds = _round_dirs(demos_root)
    if args.round is not None:
        rounds = [demos_root / f"round-{int(args.round):03d}"]
    if not rounds:
        raise SystemExit(f"No round directories found in {demos_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rd in rounds:
        if not rd.exists():
            continue
        demos = _demo_dirs(rd)
        if not demos:
            continue
        curves: list[np.ndarray] = []
        plt.figure(figsize=(9, 5))
        for i, dd in enumerate(demos):
            ds = _load_dataset(dd)
            row = ds[0]
            rews = row.get("rews", [])
            if rews is None or len(rews) == 0:
                continue
            y = np.asarray(rews, dtype=np.float32)
            curves.append(y)
            if args.show_all:
                plt.plot(np.arange(len(y)), y, alpha=0.35, linewidth=1.0)

        if not curves:
            plt.close()
            continue

        max_len = max(len(c) for c in curves)
        M = np.full((len(curves), max_len), np.nan, dtype=np.float32)
        for i, c in enumerate(curves):
            M[i, : len(c)] = c
        mean = np.nanmean(M, axis=0)
        p25 = np.nanpercentile(M, 25, axis=0)
        p75 = np.nanpercentile(M, 75, axis=0)

        x = np.arange(max_len)
        plt.plot(x, mean, linewidth=2.5, label="mean reward")
        plt.fill_between(x, p25, p75, alpha=0.2, label="p25-p75")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.title(f"{rd.name}: per-timestep rewards ({len(curves)} demos)")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"{rd.name}_rewards.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"saved {out_path}")
    return 0


def cmd_plot_rewards_progress(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    demos_root = Path(args.demos_root)
    rounds = _round_dirs(demos_root)
    if not rounds:
        raise SystemExit(f"No round directories found in {demos_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    round_names: list[str] = []
    round_returns_mean: list[float] = []
    round_returns_std: list[float] = []
    round_success_rate: list[float] = []
    round_payload_oob_rate: list[float] = []
    round_quad_oob_rate: list[float] = []

    # Figure 1: per-round mean reward curve (+/- std), all rounds on one plot.
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("viridis")

    for ridx, rd in enumerate(rounds):
        demos = _demo_dirs(rd)
        if not demos:
            continue
        curves: list[np.ndarray] = []
        rets: list[float] = []
        succ = payload_oob = quad_oob = 0
        for dd in demos:
            ds = _load_dataset(dd)
            row = ds[0]
            rews = row.get("rews", [])
            infos = row.get("infos", [])
            if rews is None or len(rews) == 0:
                continue
            y = np.asarray(rews, dtype=np.float32)
            curves.append(y)
            rets.append(float(np.sum(y)))
            if infos:
                d = _safe_json_loads(infos[-1])
                if isinstance(d, dict):
                    succ += int(bool(d.get("is_success", False)))
                    payload_oob += int(bool(d.get("payload_out_of_bounds", False)))
                    quad_oob += int(bool(d.get("quad_out_of_bounds", False)))
        if not curves:
            continue

        max_len = max(len(c) for c in curves)
        M = np.full((len(curves), max_len), np.nan, dtype=np.float32)
        for i, c in enumerate(curves):
            M[i, : len(c)] = c
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(max_len)

        color = cmap(ridx / max(1, len(rounds) - 1))
        ax1.plot(x, mean, color=color, linewidth=1.8, alpha=0.9, label=rd.name)
        if args.show_bands:
            ax1.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)

        round_names.append(rd.name)
        round_returns_mean.append(float(np.mean(rets)))
        round_returns_std.append(float(np.std(rets)))
        n = max(1, len(curves))
        round_success_rate.append(float(succ) / n)
        round_payload_oob_rate.append(float(payload_oob) / n)
        round_quad_oob_rate.append(float(quad_oob) / n)

    ax1.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_title("Per-round mean per-timestep reward (all rounds)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.25)
    if len(round_names) <= 20:
        ax1.legend(ncol=2, fontsize=8)
    fig1.tight_layout()
    fig1_path = out_dir / "reward_curves_by_round.png"
    fig1.savefig(fig1_path, dpi=170)
    plt.close(fig1)
    print(f"saved {fig1_path}")

    # Figure 2: round return trend (mean +/- std across demos in round).
    if round_names:
        x = np.arange(len(round_names))
        fig2, ax2 = plt.subplots(figsize=(12, 4.5))
        m = np.asarray(round_returns_mean, dtype=np.float32)
        s = np.asarray(round_returns_std, dtype=np.float32)
        ax2.plot(x, m, color="tab:blue", linewidth=2.2, marker="o", label="mean return")
        ax2.fill_between(x, m - s, m + s, color="tab:blue", alpha=0.2, label="±1σ")
        ax2.set_title("Round return trend")
        ax2.set_xlabel("Round index")
        ax2.set_ylabel("Episode return")
        ax2.set_xticks(x)
        ax2.set_xticklabels(round_names, rotation=45, ha="right", fontsize=8)
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")
        fig2.tight_layout()
        fig2_path = out_dir / "round_return_trend.png"
        fig2.savefig(fig2_path, dpi=170)
        plt.close(fig2)
        print(f"saved {fig2_path}")

        # Figure 3: failure/success rates by round.
        fig3, ax3 = plt.subplots(figsize=(12, 4.5))
        ax3.plot(x, round_success_rate, marker="o", linewidth=2.0, label="success_rate")
        ax3.plot(x, round_payload_oob_rate, marker="o", linewidth=2.0, label="payload_oob_rate")
        ax3.plot(x, round_quad_oob_rate, marker="o", linewidth=2.0, label="quad_oob_rate")
        ax3.set_ylim(-0.02, 1.02)
        ax3.set_title("Round outcome rates")
        ax3.set_xlabel("Round index")
        ax3.set_ylabel("Rate")
        ax3.set_xticks(x)
        ax3.set_xticklabels(round_names, rotation=45, ha="right", fontsize=8)
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best")
        fig3.tight_layout()
        fig3_path = out_dir / "round_failure_rates.png"
        fig3.savefig(fig3_path, dpi=170)
        plt.close(fig3)
        print(f"saved {fig3_path}")

        metrics = {
            "rounds": round_names,
            "mean_return": round_returns_mean,
            "std_return": round_returns_std,
            "success_rate": round_success_rate,
            "payload_oob_rate": round_payload_oob_rate,
            "quad_oob_rate": round_quad_oob_rate,
        }
        metrics_path = out_dir / "round_progress_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"saved {metrics_path}")

    return 0


def _collect_round_obs(round_dir: Path) -> tuple[list[np.ndarray], int]:
    demos = _demo_dirs(round_dir)
    obs_curves: list[np.ndarray] = []
    raw_dim: int | None = None
    for dd in demos:
        ds = _load_dataset(dd)
        row = ds[0]
        obs = np.asarray(row.get("obs", []), dtype=np.float32)
        if obs.ndim != 2 or obs.shape[0] == 0:
            continue
        if raw_dim is None:
            raw_dim = _raw_state_dim_from_row(row)
        obs_curves.append(obs)
    if raw_dim is None:
        raise SystemExit(f"No valid obs found in {round_dir}")
    return obs_curves, raw_dim


def _pad_curves(curves: list[np.ndarray], value: float = np.nan) -> np.ndarray:
    max_len = max(c.shape[0] for c in curves)
    feat_dim = curves[0].shape[1]
    M = np.full((len(curves), max_len, feat_dim), value, dtype=np.float32)
    for i, c in enumerate(curves):
        M[i, : c.shape[0], :] = c
    return M


def cmd_plot_states(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    demos_root = Path(args.demos_root)
    rounds = _round_dirs(demos_root)
    if args.round is not None:
        rounds = [demos_root / f"round-{int(args.round):03d}"]
    if not rounds:
        raise SystemExit(f"No round directories found in {demos_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rd in rounds:
        if not rd.exists():
            continue
        obs_curves, raw_dim = _collect_round_obs(rd)
        if not obs_curves:
            continue
        n_bodies = raw_dim // 13
        if n_bodies < 1 or 13 * n_bodies != raw_dim:
            raise SystemExit(f"Unexpected raw_dim={raw_dim} in {rd}")
        n_quads = n_bodies - 1
        M = _pad_curves(obs_curves)

        pos = M[:, :, : 7 * n_bodies]
        vel = M[:, :, 7 * n_bodies : raw_dim]
        payload_pos = pos[:, :, 0:3]
        payload_v = vel[:, :, 0:3]
        payload_w = vel[:, :, 3:6]

        # Observation tails are learner features; last action_dim entries are prev action.
        action_dim = 4 * n_quads
        obs_tail = M[:, :, raw_dim:]
        prev_act = obs_tail[:, :, -action_dim:] if obs_tail.shape[2] >= action_dim and action_dim > 0 else None
        payload_pos_err = obs_tail[:, :, 0:3] if obs_tail.shape[2] >= 3 else None
        payload_vel_feat = obs_tail[:, :, 3:6] if obs_tail.shape[2] >= 6 else None

        def _plot_3axis(arr: np.ndarray, title: str, ylab: str, out_name: str) -> None:
            fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
            for ax_i, ax in enumerate(axs):
                c = arr[:, :, ax_i]
                mean = np.nanmean(c, axis=0)
                std = np.nanstd(c, axis=0)
                x = np.arange(mean.shape[0])
                ax.plot(x, mean, color="tab:red", linewidth=2.2, label="mean")
                ax.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color="tab:red", alpha=0.2, label="±2σ")
                if args.show_all:
                    for i in range(c.shape[0]):
                        ax.plot(x, c[i], color="tab:blue", alpha=0.18, linewidth=1.0)
                ax.set_title(["x", "y", "z"][ax_i])
                ax.set_xlabel("timestep")
                ax.grid(True, alpha=0.25)
                if ax_i == 0:
                    ax.set_ylabel(ylab)
                    ax.legend(loc="upper right")
            fig.suptitle(f"{rd.name}: {title} ({arr.shape[0]} demos)")
            fig.tight_layout()
            out_path = out_dir / f"{rd.name}_{out_name}.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            print(f"saved {out_path}")

        _plot_3axis(payload_pos, "payload position", "m", "payload_pos")
        _plot_3axis(payload_v, "payload linear velocity", "m/s", "payload_vel")
        _plot_3axis(payload_w, "payload angular velocity", "rad/s", "payload_angvel")

        for qi in range(n_quads):
            p0 = 7 * (1 + qi)
            v0 = 6 * (1 + qi)
            q_pos = pos[:, :, p0 : p0 + 3]
            q_v = vel[:, :, v0 : v0 + 3]
            q_w = vel[:, :, v0 + 3 : v0 + 6]
            _plot_3axis(q_pos, f"quad {qi+1} position", "m", f"q{qi+1}_pos")
            _plot_3axis(q_v, f"quad {qi+1} linear velocity", "m/s", f"q{qi+1}_vel")
            _plot_3axis(q_w, f"quad {qi+1} angular velocity", "rad/s", f"q{qi+1}_angvel")

        if prev_act is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            a_norm = np.linalg.norm(prev_act, axis=2)
            mean = np.nanmean(a_norm, axis=0)
            std = np.nanstd(a_norm, axis=0)
            x = np.arange(mean.shape[0])
            ax.plot(x, mean, color="k", linewidth=2.2, label="mean ||prev_action||")
            ax.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color="tab:purple", alpha=0.2, label="±2σ")
            if args.show_all:
                for i in range(a_norm.shape[0]):
                    ax.plot(x, a_norm[i], color="tab:purple", alpha=0.15, linewidth=1.0)
            ax.set_title(f"{rd.name}: prev-action norm")
            ax.set_xlabel("timestep")
            ax.set_ylabel("norm")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            fig.tight_layout()
            out_path = out_dir / f"{rd.name}_prev_action_norm.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            print(f"saved {out_path}")

        if obs_tail.shape[2] > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            tail_norm = np.linalg.norm(obs_tail, axis=2)
            mean = np.nanmean(tail_norm, axis=0)
            std = np.nanstd(tail_norm, axis=0)
            x = np.arange(mean.shape[0])
            ax.plot(x, mean, color="tab:green", linewidth=2.2, label="mean ||learner_obs_tail||")
            ax.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color="tab:green", alpha=0.2, label="±2σ")
            if args.show_all:
                for i in range(tail_norm.shape[0]):
                    ax.plot(x, tail_norm[i], color="tab:green", alpha=0.15, linewidth=1.0)
            ax.set_title(f"{rd.name}: learner-observation tail norm")
            ax.set_xlabel("timestep")
            ax.set_ylabel("norm")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            fig.tight_layout()
            out_path = out_dir / f"{rd.name}_learner_obs_norm.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            print(f"saved {out_path}")

        if payload_pos_err is not None:
            _plot_3axis(payload_pos_err, "learner feat: payload position error", "m", "learner_payload_pos_err")
        if payload_vel_feat is not None:
            _plot_3axis(payload_vel_feat, "learner feat: payload velocity", "m/s", "learner_payload_vel")

    return 0


def cmd_render_demo(args: argparse.Namespace) -> int:
    import sys

    sys.path.append("scripts")
    from videos_from_log import VideoConfig, render_from_actions, render_from_states
    from payload_env import PayloadGymEnv

    demos_root = Path(args.demos_root)
    round_dir = demos_root / f"round-{int(args.round):03d}"
    demos = _demo_dirs(round_dir)
    if not demos:
        raise SystemExit(f"No demos in {round_dir}")
    demo_dir = demos[int(args.demo_index)]

    ds = _load_dataset(demo_dir)
    row = ds[0]
    obs = np.asarray(row["obs"], dtype=np.float32)
    acts = np.asarray(row["acts"], dtype=np.float32)
    if obs.ndim != 2 or acts.ndim != 2:
        raise SystemExit("Unexpected dataset shapes for obs/acts")

    # Use env to get dimensions and workspace bounds.
    env = PayloadGymEnv(
        xml_path=str(args.xml_path),
        template_yaml_path=str(args.env_yaml),
        max_steps=int(args.max_steps),
    )
    raw_dim = env.state_dim
    raw0 = obs[0, :raw_dim]
    qpos0, qvel0 = _raw_state_to_qpos_qvel(raw0, env.n_bodies)
    u_traj = np.stack([_map_policy_action_to_mujoco_ctrl(a, env.action_dim) for a in acts], axis=0)

    out_dir = Path(args.out_dir) / f"round-{int(args.round):03d}_demo-{int(args.demo_index):03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VideoConfig(
        out_dir=str(out_dir),
        width=args.width,
        height=args.height,
        fps=args.fps,
        views=args.views if len(args.views) > 1 else args.views[0],
        env_min=env.workspace_bounds[0],
        env_max=env.workspace_bounds[1],
    )
    if args.replay_mode == "states":
        x_rows = []
        for t in range(obs.shape[0]):
            qt, vt = _raw_state_to_qpos_qvel(obs[t, :raw_dim], env.n_bodies)
            x_rows.append(np.concatenate([qt, vt], axis=0))
        x_traj = np.stack(x_rows, axis=0)
        written = render_from_states(
            xml_path=str(args.xml_path),
            x_traj_or_npz=x_traj,
            cfg=cfg,
        )
    else:
        written = render_from_actions(
            xml_path=str(args.xml_path),
            init_and_actions_or_npz=(qpos0, qvel0, u_traj),
            cfg=cfg,
            include_initial_frame=True,
        )

    print("demo_dir:", demo_dir)
    print("written:")
    for p in written:
        print(" ", p)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    demos_root = Path(args.demos_root)
    rounds = _round_dirs(demos_root)
    if args.round is not None:
        rounds = [demos_root / f"round-{int(args.round):03d}"]

    for rd in rounds:
        if not rd.exists():
            print(f"{rd.name}: missing")
            continue
        demos = _demo_dirs(rd)
        lengths = []
        rets = []
        succ = payload_oob = quad_oob = 0
        for dd in demos:
            ds = _load_dataset(dd)
            row = ds[0]
            acts = row.get("acts", [])
            rews = row.get("rews", [])
            infos = row.get("infos", [])
            lengths.append(len(acts))
            rets.append(float(sum(rews)) if rews else float("nan"))
            if infos:
                d = _safe_json_loads(infos[-1])
                if isinstance(d, dict):
                    succ += int(bool(d.get("is_success", False)))
                    payload_oob += int(bool(d.get("payload_out_of_bounds", False)))
                    quad_oob += int(bool(d.get("quad_out_of_bounds", False)))
        if not demos:
            print(f"{rd.name}: demos=0")
            continue
        print(
            f"{rd.name}: demos={len(demos)} "
            f"steps(mean/min/max)={np.mean(lengths):.1f}/{min(lengths)}/{max(lengths)} "
            f"return(mean)={np.nanmean(rets):.2f} success={succ} payload_oob={payload_oob} quad_oob={quad_oob}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect, plot, and render DAgger round demos.")
    ap.add_argument("--demos-root", default="runs/dagger_2000steps/scratch_dagger/demos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--round", type=int, default=None)

    p_plot = sub.add_parser("plot-rewards")
    p_plot.add_argument("--round", type=int, default=None, help="If omitted, plot all rounds.")
    p_plot.add_argument("--out-dir", default="runs/dagger_2000steps/round_reward_plots")
    p_plot.add_argument("--show-all", action="store_true", help="Plot all demo reward curves in addition to mean band.")

    p_prog = sub.add_parser("plot-rewards-progress")
    p_prog.add_argument("--out-dir", default="runs/dagger_2000steps/round_reward_progress")
    p_prog.add_argument("--show-bands", action="store_true", help="Show ±std band for per-round reward curves.")

    p_states = sub.add_parser("plot-states")
    p_states.add_argument("--round", type=int, default=None, help="If omitted, plot all rounds.")
    p_states.add_argument("--out-dir", default="runs/dagger_2000steps/round_state_plots")
    p_states.add_argument("--show-all", action="store_true", help="Overlay all demo curves.")

    p_render = sub.add_parser("render-demo")
    p_render.add_argument("--round", type=int, required=True)
    p_render.add_argument("--demo-index", type=int, default=0)
    p_render.add_argument("--out-dir", default="runs/dagger_2000steps/round_demo_videos")
    p_render.add_argument("--xml-path", default="deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml")
    p_render.add_argument("--env-yaml", default="deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml")
    p_render.add_argument("--max-steps", type=int, default=1000)
    p_render.add_argument("--width", type=int, default=1280)
    p_render.add_argument("--height", type=int, default=720)
    p_render.add_argument("--fps", type=int, default=60)
    p_render.add_argument("--views", nargs="+", default=["diag"])
    p_render.add_argument(
        "--replay-mode",
        choices=["actions", "states"],
        default="states",
        help="states: render logged trajectory; actions: rollout controls from initial state.",
    )

    args = ap.parse_args()
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.cmd == "plot-rewards":
        return cmd_plot_rewards(args)
    if args.cmd == "plot-rewards-progress":
        return cmd_plot_rewards_progress(args)
    if args.cmd == "plot-states":
        return cmd_plot_states(args)
    if args.cmd == "render-demo":
        return cmd_render_demo(args)
    raise RuntimeError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
