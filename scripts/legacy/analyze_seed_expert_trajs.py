#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_seed_trajs(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _split_raw_state(raw_state: np.ndarray):
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n_bodies = raw.size // 13
    poses = raw[: 7 * n_bodies]
    vels = raw[7 * n_bodies :]
    payload_pos = poses[:3]
    payload_vel = vels[:3]
    quad_pos = []
    quad_vel = []
    for qi in range(n_bodies - 1):
        pbase = 7 * (1 + qi)
        vbase = 6 * (1 + qi)
        quad_pos.append(poses[pbase : pbase + 3])
        quad_vel.append(vels[vbase : vbase + 3])
    return payload_pos, payload_vel, quad_pos, quad_vel


def _stack_rollouts(trajs):
    # infer dims from first trajectory action dim
    first = trajs[0]
    n_quads = first.acts.shape[1] // 4
    state_dim = 13 * (1 + n_quads)
    rolls = []
    for i, t in enumerate(trajs):
        obs = np.asarray(t.obs, dtype=np.float32)
        acts = np.asarray(t.acts, dtype=np.float32)
        raw = obs[:, :state_dim]
        payload_pos = []
        payload_vel = []
        quad_pos = [[] for _ in range(n_quads)]
        quad_vel = [[] for _ in range(n_quads)]
        for row in raw:
            pp, pv, qp, qv = _split_raw_state(row)
            payload_pos.append(pp)
            payload_vel.append(pv)
            for qi in range(n_quads):
                quad_pos[qi].append(qp[qi])
                quad_vel[qi].append(qv[qi])
        rolls.append(
            {
                "idx": i,
                "obs": obs,
                "acts": acts,
                "payload_pos": np.asarray(payload_pos, np.float32),
                "payload_vel": np.asarray(payload_vel, np.float32),
                "quad_pos": [np.asarray(v, np.float32) for v in quad_pos],
                "quad_vel": [np.asarray(v, np.float32) for v in quad_vel],
                "goal": raw[0, :3] * 0 + np.asarray(obs[0, state_dim : state_dim + 3], np.float32) + raw[0, :3],
                # goal reconstruction from learner feature payload pos error = p-goal => goal = p - e
            }
        )
    return rolls, n_quads, state_dim


def _load_seed_delta_values(seed_cfg_dir: Path, n: int):
    vals = [np.nan] * n
    if not seed_cfg_dir.exists():
        return vals
    for i in range(n):
        p = seed_cfg_dir / f"pc_dbcbs_seed_ep_{i:03d}.yaml"
        if not p.exists():
            continue
        try:
            y = yaml.safe_load(p.read_text())
            vals[i] = float(y["pc-dbcbs"]["default"]["delta_0"])
        except Exception:
            pass
    return vals


def _plot_xyz_family(out_path: Path, title: str, series_list: list[np.ndarray], replan_k: int | None = None):
    n = len(series_list)
    min_len = min(s.shape[0] for s in series_list)
    arr = np.stack([s[:min_len] for s in series_list], axis=0)  # [N,T,3]
    t = np.arange(min_len)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    labels = ["x", "y", "z"]
    for k in range(3):
        for i in range(n):
            axs[k].plot(t, arr[i, :, k], color="tab:blue", alpha=0.12, lw=0.8)
        axs[k].plot(t, mean[:, k], color="red", lw=1.8, label="mean")
        axs[k].fill_between(t, mean[:, k] - 2 * std[:, k], mean[:, k] + 2 * std[:, k],
                            color="red", alpha=0.15, label="±2σ")
        if replan_k and replan_k > 0:
            for rt in range(0, min_len, replan_k):
                axs[k].axvline(rt, color="k", alpha=0.06, lw=1)
        axs[k].set_title(labels[k])
        axs[k].set_xlabel("timestep")
        axs[k].grid(True, alpha=0.3)
        axs[k].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyze cached seed expert trajectories (.pkl) with delta_0 metadata.")
    ap.add_argument("--seed-cache-dir", default="runs/dagger_2000steps/round0_cache_obs_payload_rel_v2_seeddelta_0.70_0.90")
    ap.add_argument("--replan-k", type=int, default=0, help="For visual replan markers only (seed expert is typically 0).")
    args = ap.parse_args()

    seed_dir = Path(args.seed_cache_dir)
    pkl_path = seed_dir / "expert_trajs.pkl"
    out_dir = seed_dir / "seed_traj_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = _load_seed_trajs(pkl_path)
    rolls, n_quads, state_dim = _stack_rollouts(trajs)
    deltas = _load_seed_delta_values(seed_dir / "pcdbcbs_seed_cfgs", len(rolls))

    # Plot payload and quad states across seed episodes
    _plot_xyz_family(out_dir / "payload_pos.png", f"Seed Expert Payload Position ({len(rolls)} eps)", [r["payload_pos"] for r in rolls], args.replan_k)
    _plot_xyz_family(out_dir / "payload_vel.png", f"Seed Expert Payload Velocity ({len(rolls)} eps)", [r["payload_vel"] for r in rolls], args.replan_k)
    for qi in range(n_quads):
        _plot_xyz_family(out_dir / f"quad{qi+1}_pos.png", f"Seed Expert Quad {qi+1} Position ({len(rolls)} eps)", [r["quad_pos"][qi] for r in rolls], args.replan_k)
        _plot_xyz_family(out_dir / f"quad{qi+1}_vel.png", f"Seed Expert Quad {qi+1} Velocity ({len(rolls)} eps)", [r["quad_vel"][qi] for r in rolls], args.replan_k)

    # Goal-distance and action magnitude
    min_len = min(r["acts"].shape[0] for r in rolls)
    goal_dist = []
    act_norm = []
    for r in rolls:
        p = r["payload_pos"][:min_len]
        # reconstruct payload goal from learner feature on first obs
        raw0 = r["obs"][0, :state_dim]
        feat0 = r["obs"][0, state_dim:]
        e_p0 = feat0[:3]
        goal = raw0[:3] - e_p0
        goal_dist.append(np.linalg.norm(p - goal[None, :], axis=1))
        act_norm.append(np.linalg.norm(r["acts"][:min_len], axis=1))
    G = np.stack(goal_dist, axis=0)
    A = np.stack(act_norm, axis=0)
    t = np.arange(min_len)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, M, title, ylabel, color in [
        (axs[0], G, "Payload goal distance", "m", "tab:green"),
        (axs[1], A, "Policy action norm", "||a||", "tab:purple"),
    ]:
        mean = M.mean(axis=0); std = M.std(axis=0)
        for i in range(M.shape[0]):
            ax.plot(t, M[i], color=color, alpha=0.08, lw=0.8)
        ax.plot(t, mean, color="black", lw=1.8)
        ax.fill_between(t, mean - 2 * std, mean + 2 * std, color=color, alpha=0.15)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_dir / "goal_distance_and_action_norm.png", dpi=150)
    plt.close(fig)

    # delta_0 metadata scatter/summary
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(deltas))
    y = np.array(deltas, dtype=float)
    ax.plot(xs, y, "o-", label="delta_0 per seed episode")
    ax.set_xlabel("seed episode")
    ax.set_ylabel("delta_0")
    ax.set_title("Seed pc-dbCBS delta_0 values")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "seed_delta0_values.png", dpi=150)
    plt.close(fig)

    report = [
        "=" * 60,
        "SEED EXPERT TRAJECTORY ANALYSIS",
        "=" * 60,
        f"seed_cache_dir: {seed_dir}",
        f"episodes: {len(rolls)}",
        f"n_quads: {n_quads}",
        f"state_dim(raw): {state_dim}",
        f"episode_lengths(min/mean/max): {min(r['acts'].shape[0] for r in rolls)}/{np.mean([r['acts'].shape[0] for r in rolls]):.1f}/{max(r['acts'].shape[0] for r in rolls)}",
        f"delta_0(min/mean/max): {np.nanmin(y):.3f}/{np.nanmean(y):.3f}/{np.nanmax(y):.3f}",
        "plots:",
        "  payload_pos.png, payload_vel.png",
        "  quad1_pos/vel.png, quad2_pos/vel.png, ...",
        "  goal_distance_and_action_norm.png",
        "  seed_delta0_values.png",
        "=" * 60,
    ]
    txt = "\n".join(report)
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
