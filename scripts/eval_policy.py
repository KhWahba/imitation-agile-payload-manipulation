#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from imitation.algorithms import bc
from imitation.data import rollout

from payload_env import PayloadGymEnv


def _raw_state_dim_from_obs(obs: np.ndarray) -> int:
    # Fixed current setup: 2 quads -> 3 bodies -> 39 raw-state entries.
    if obs.shape[-1] >= 39:
        return 39
    raise ValueError(f"Unexpected obs dim {obs.shape[-1]}, expected at least 39.")


def _pad_2d(curves: list[np.ndarray], fill: float = np.nan) -> np.ndarray:
    tmax = max(c.shape[0] for c in curves)
    d = curves[0].shape[1]
    out = np.full((len(curves), tmax, d), fill, dtype=np.float32)
    for i, c in enumerate(curves):
        out[i, : c.shape[0], :] = c
    return out


def _plot_1d(curves: list[np.ndarray], out: Path, title: str, ylabel: str, show_all: bool) -> None:
    tmax = max(len(c) for c in curves)
    M = np.full((len(curves), tmax), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        M[i, : len(c)] = c
    mean = np.nanmean(M, axis=0)
    std = np.nanstd(M, axis=0)
    x = np.arange(tmax)
    plt.figure(figsize=(9, 4))
    if show_all:
        for i in range(M.shape[0]):
            plt.plot(x, M[i], alpha=0.18, linewidth=1.0, color="tab:blue")
    plt.plot(x, mean, color="tab:red", linewidth=2.2, label="mean")
    plt.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color="tab:red", alpha=0.2, label="±2σ")
    plt.title(title)
    plt.xlabel("timestep")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def _plot_3axis(curves: list[np.ndarray], out: Path, title: str, ylabel: str, show_all: bool) -> None:
    M = _pad_2d(curves)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for j, ax in enumerate(axs):
        c = M[:, :, j]
        mean = np.nanmean(c, axis=0)
        std = np.nanstd(c, axis=0)
        x = np.arange(mean.shape[0])
        if show_all:
            for i in range(c.shape[0]):
                ax.plot(x, c[i], alpha=0.16, linewidth=1.0, color="tab:blue")
        ax.plot(x, mean, color="tab:red", linewidth=2.2, label="mean")
        ax.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color="tab:red", alpha=0.2, label="±2σ")
        ax.set_title(["x", "y", "z"][j])
        ax.set_xlabel("timestep")
        if j == 0:
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _sample_reset(env: PayloadGymEnv, sigma: float, rng: np.random.Generator, base_start: np.ndarray) -> None:
    if sigma <= 0:
        env.start_state = base_start.copy()
        return
    s = base_start.copy()
    n = env.n_bodies
    lo, hi = env.workspace_bounds
    for bi in range(n):
        p0 = 7 * bi
        pos = s[p0 : p0 + 3] + rng.normal(0.0, sigma, size=3)
        s[p0 : p0 + 3] = np.clip(pos, lo, hi)
    env.start_state = s


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate saved DAgger/BC policy with statistical plots.")
    ap.add_argument("--policy-state-dict", required=True)
    ap.add_argument("--expert-trajs-pkl", default="runs/dagger_10000steps/round0_cache_obs_payload_rel_v2_seeddelta_0.70_0.75/expert_trajs.pkl")
    ap.add_argument("--out-dir", default="runs/policy_eval")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--reset-sigma", type=float, default=0.0, help="Gaussian sigma on per-body start positions at reset.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--show-all", action="store_true")
    ap.add_argument("--xml-path", default="deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml")
    ap.add_argument("--env-yaml", default="deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = PayloadGymEnv(xml_path=args.xml_path, template_yaml_path=args.env_yaml, max_steps=args.max_steps)
    env.terminate_on_success = True
    base_start = env.start_state.copy()

    with open(args.expert_trajs_pkl, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=min(256, len(transitions)),
    )
    state_dict = th.load(args.policy_state_dict, map_location="cpu")
    bc_trainer.policy.load_state_dict(state_dict)
    policy = bc_trainer.policy

    rews_curves: list[np.ndarray] = []
    acts_curves: list[np.ndarray] = []
    obs_curves: list[np.ndarray] = []
    payload_pos_curves: list[np.ndarray] = []
    payload_vel_curves: list[np.ndarray] = []
    payload_w_curves: list[np.ndarray] = []
    q1_pos_curves: list[np.ndarray] = []
    q1_vel_curves: list[np.ndarray] = []
    q1_w_curves: list[np.ndarray] = []
    q2_pos_curves: list[np.ndarray] = []
    q2_vel_curves: list[np.ndarray] = []
    q2_w_curves: list[np.ndarray] = []

    lengths, returns = [], []
    succ = payload_oob = quad_oob = 0

    for ep in range(args.n_episodes):
        _sample_reset(env, args.reset_sigma, rng, base_start)
        obs, _ = env.reset()
        done = False
        ep_rews = []
        ep_acts = []
        ep_obs = [obs.copy()]
        ep_infos: list[dict[str, Any]] = []
        while not done:
            act, _ = policy.predict(obs[None, :], deterministic=True)
            act = np.asarray(act[0], dtype=np.float32)
            next_obs, rew, terminated, truncated, info = env.step(act)
            ep_rews.append(float(rew))
            ep_acts.append(act.copy())
            ep_obs.append(next_obs.copy())
            ep_infos.append(dict(info))
            obs = next_obs
            done = bool(terminated or truncated)

        R = float(np.sum(ep_rews))
        T = len(ep_rews)
        lengths.append(T)
        returns.append(R)
        if ep_infos:
            final = ep_infos[-1]
            succ += int(bool(final.get("is_success", False)))
            payload_oob += int(bool(final.get("payload_out_of_bounds", False)))
            quad_oob += int(bool(final.get("quad_out_of_bounds", False)))

        rews = np.asarray(ep_rews, dtype=np.float32)
        acts = np.asarray(ep_acts, dtype=np.float32)
        obs_arr = np.asarray(ep_obs[:-1], dtype=np.float32)  # align with actions/rewards (T rows)
        raw_dim = _raw_state_dim_from_obs(obs_arr[0])
        raw = obs_arr[:, :raw_dim]
        n = raw_dim // 13
        pose = raw[:, : 7 * n]
        vel = raw[:, 7 * n : raw_dim]
        payload_pos_curves.append(pose[:, 0:3])
        payload_vel_curves.append(vel[:, 0:3])
        payload_w_curves.append(vel[:, 3:6])
        if n >= 3:
            q1_pos_curves.append(pose[:, 7:10])
            q2_pos_curves.append(pose[:, 14:17])
            q1_vel_curves.append(vel[:, 6:9])
            q2_vel_curves.append(vel[:, 12:15])
            q1_w_curves.append(vel[:, 9:12])
            q2_w_curves.append(vel[:, 15:18])
        rews_curves.append(rews)
        acts_curves.append(acts)
        obs_curves.append(obs_arr)

    _plot_1d(rews_curves, out / "rewards_per_timestep.png", "Per-timestep reward", "reward", args.show_all)
    _plot_1d([np.linalg.norm(a, axis=1) for a in acts_curves], out / "action_norm.png", "Action norm", "||a||", args.show_all)
    _plot_1d([np.linalg.norm(o[:, 39:], axis=1) for o in obs_curves], out / "learner_obs_norm.png", "Learner obs-tail norm", "norm", args.show_all)

    _plot_3axis(payload_pos_curves, out / "payload_pos.png", "Payload position", "m", args.show_all)
    _plot_3axis(payload_vel_curves, out / "payload_vel.png", "Payload velocity", "m/s", args.show_all)
    _plot_3axis(payload_w_curves, out / "payload_angvel.png", "Payload angular velocity", "rad/s", args.show_all)
    if q1_pos_curves:
        _plot_3axis(q1_pos_curves, out / "q1_pos.png", "Quad1 position", "m", args.show_all)
        _plot_3axis(q1_vel_curves, out / "q1_vel.png", "Quad1 velocity", "m/s", args.show_all)
        _plot_3axis(q1_w_curves, out / "q1_angvel.png", "Quad1 angular velocity", "rad/s", args.show_all)
    if q2_pos_curves:
        _plot_3axis(q2_pos_curves, out / "q2_pos.png", "Quad2 position", "m", args.show_all)
        _plot_3axis(q2_vel_curves, out / "q2_vel.png", "Quad2 velocity", "m/s", args.show_all)
        _plot_3axis(q2_w_curves, out / "q2_angvel.png", "Quad2 angular velocity", "rad/s", args.show_all)

    summary = {
        "n_episodes": int(args.n_episodes),
        "max_steps": int(args.max_steps),
        "reset_sigma": float(args.reset_sigma),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "success_rate": float(succ / max(1, args.n_episodes)),
        "payload_oob_rate": float(payload_oob / max(1, args.n_episodes)),
        "quad_oob_rate": float(quad_oob / max(1, args.n_episodes)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved plots + summary in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
