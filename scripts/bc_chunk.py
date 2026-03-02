#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch as th
import torch.nn as nn
import yaml
from imitation.algorithms import bc
from imitation.data.types import Trajectory, Transitions
from payload_env import PayloadGymEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from videos_from_log import VideoConfig, render_from_actions
    HAS_VIDEO = True
except Exception:
    HAS_VIDEO = False


REPO = Path(__file__).resolve().parents[1]
XML_PATH = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
TEMPLATE_YAML = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
DEFAULT_TRAJS_PKL = REPO / "runs/dagger_10000steps/round0_cache_obs_payload_rel_v2_seeddelta_0.70_0.75/expert_trajs.pkl"


@dataclass
class ChunkData:
    obs: np.ndarray      # [N, obs_dim]
    acts_flat: np.ndarray  # [N, H*act_dim]
    obs_dim: int
    act_dim: int
    horizon: int
    n_episodes: int
    n_windows: int


@dataclass
class ChunkPolicyBundle:
    policy: ActorCriticPolicy
    obs_dim: int
    act_dim: int
    horizon: int


def _pad_1d(curves: list[np.ndarray], fill: float = np.nan) -> np.ndarray:
    tmax = max(len(c) for c in curves)
    M = np.full((len(curves), tmax), fill, dtype=np.float32)
    for i, c in enumerate(curves):
        M[i, : len(c)] = c
    return M


def _plot_1d(curves: list[np.ndarray], out: Path, title: str, ylabel: str, show_all: bool = True) -> None:
    if not curves:
        return
    M = _pad_1d(curves)
    mean = np.nanmean(M, axis=0)
    std = np.nanstd(M, axis=0)
    x = np.arange(mean.shape[0])
    plt.figure(figsize=(9, 4))
    if show_all:
        for i in range(M.shape[0]):
            plt.plot(x, M[i], alpha=0.16, linewidth=1.0, color="tab:blue")
    plt.plot(x, mean, color="tab:red", linewidth=2.2, label="mean")
    plt.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color="tab:red", alpha=0.2, label="±2σ")
    plt.title(title)
    plt.xlabel("timestep")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def _plot_3axis(curves: list[np.ndarray], out: Path, title: str, ylabel: str, show_all: bool = True) -> None:
    if not curves:
        return
    tmax = max(c.shape[0] for c in curves)
    M = np.full((len(curves), tmax, 3), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        M[i, : c.shape[0], :] = c
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for j, ax in enumerate(axs):
        C = M[:, :, j]
        mean = np.nanmean(C, axis=0)
        std = np.nanstd(C, axis=0)
        x = np.arange(mean.shape[0])
        if show_all:
            for i in range(C.shape[0]):
                ax.plot(x, C[i], alpha=0.16, linewidth=1.0, color="tab:blue")
        ax.plot(x, mean, color="tab:red", linewidth=2.2, label="mean")
        ax.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color="tab:red", alpha=0.2, label="±2σ")
        ax.set_title(["x", "y", "z"][j])
        ax.set_xlabel("timestep")
        if j == 0:
            ax.set_ylabel(ylabel)
            ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _transform_reward_curves(
    curves: list[np.ndarray],
    clip_min: float | None,
    clip_max: float | None,
    normalize: bool,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for c in curves:
        x = np.asarray(c, dtype=np.float32).copy()
        if clip_min is not None or clip_max is not None:
            lo = -np.inf if clip_min is None else float(clip_min)
            hi = np.inf if clip_max is None else float(clip_max)
            x = np.clip(x, lo, hi)
        if normalize:
            denom = float(np.max(np.abs(x)))
            if denom > 1e-8:
                x = x / denom
        out.append(x)
    return out


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_trajs(paths: Iterable[Path]) -> list[Trajectory]:
    all_trajs: list[Trajectory] = []
    for p in paths:
        with open(p, "rb") as f:
            trajs = pickle.load(f)
        all_trajs.extend(trajs)
        print(f"[bc_chunk] loaded {len(trajs)} episodes from {p}")
    return all_trajs


def _build_chunk_data(trajs: list[Trajectory], horizon: int, max_windows: int = 0) -> ChunkData:
    if not trajs:
        raise RuntimeError("No trajectories loaded.")
    obs_dim = int(trajs[0].obs.shape[1])
    act_dim = int(trajs[0].acts.shape[1])

    obs_rows: list[np.ndarray] = []
    chunk_rows: list[np.ndarray] = []
    for ti, tr in enumerate(trajs):
        obs = np.asarray(tr.obs, dtype=np.float32)
        acts = np.asarray(tr.acts, dtype=np.float32)
        T = acts.shape[0]
        if obs.shape[0] != T + 1:
            raise ValueError(f"traj[{ti}] expected obs len T+1, got obs={obs.shape[0]} acts={T}")
        if T < horizon:
            continue

        for t in range(0, T - horizon + 1):
            obs_rows.append(obs[t].copy())
            chunk = acts[t : t + horizon].copy().reshape(-1)  # [H*U]
            chunk_rows.append(chunk)
            if max_windows > 0 and len(obs_rows) >= max_windows:
                break
        if max_windows > 0 and len(obs_rows) >= max_windows:
            break

    if not obs_rows:
        raise RuntimeError(f"No full windows found for horizon={horizon}.")

    X = np.stack(obs_rows, axis=0).astype(np.float32)
    Y = np.stack(chunk_rows, axis=0).astype(np.float32)

    return ChunkData(
        obs=X,
        acts_flat=Y,
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=int(horizon),
        n_episodes=len(trajs),
        n_windows=int(X.shape[0]),
    )


def _build_obs_horizon_windows(
    trajs: list[Trajectory],
    horizon: int,
    max_windows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    obs0_rows: list[np.ndarray] = []
    next_obs_rows: list[np.ndarray] = []
    for tr in trajs:
        obs = np.asarray(tr.obs, dtype=np.float32)
        acts = np.asarray(tr.acts, dtype=np.float32)
        T = acts.shape[0]
        if T < horizon:
            continue
        for t in range(0, T - horizon + 1):
            obs0_rows.append(obs[t].copy())
            next_obs_rows.append(obs[t + 1 : t + 1 + horizon].copy())  # [H, obs_dim]
            if max_windows > 0 and len(obs0_rows) >= max_windows:
                break
        if max_windows > 0 and len(obs0_rows) >= max_windows:
            break
    if not obs0_rows:
        raise RuntimeError(f"No windows for horizon={horizon}")
    return (
        np.stack(obs0_rows, axis=0).astype(np.float32),
        np.stack(next_obs_rows, axis=0).astype(np.float32),
    )


def _split_train_val(data: ChunkData, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(data.n_windows)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(data.n_windows * float(val_frac)))
    n_val = max(1, min(data.n_windows - 1, n_val))
    return idx[n_val:], idx[:n_val]


def _make_transitions(data: ChunkData, indices: np.ndarray) -> Transitions:
    obs = data.obs[indices]
    acts = data.acts_flat[indices]
    infos = np.array([{} for _ in range(obs.shape[0])], dtype=object)
    # BC only needs (obs, acts), but imitation's collate path expects transition fields.
    next_obs = obs.copy()
    dones = np.zeros((obs.shape[0],), dtype=bool)
    return Transitions(obs=obs, acts=acts, infos=infos, next_obs=next_obs, dones=dones)


def _make_env(max_steps: int) -> Monitor:
    env = PayloadGymEnv(xml_path=str(XML_PATH), template_yaml_path=str(TEMPLATE_YAML), max_steps=max_steps)
    env.terminate_on_success = True
    return Monitor(env)


def _make_policy(
    obs_space: gym.Space,
    chunk_action_space: gym.Space,
    lr: float,
    hidden: list[int],
    activation: str,
) -> ActorCriticPolicy:
    act_cls = nn.Tanh if activation.lower() == "tanh" else nn.ReLU
    return ActorCriticPolicy(
        observation_space=obs_space,
        action_space=chunk_action_space,
        lr_schedule=lambda _: lr,
        net_arch=hidden,
        activation_fn=act_cls,
        ortho_init=False,
    )


def _policy_mse_on_indices(
    policy: ActorCriticPolicy,
    data: ChunkData,
    idx: np.ndarray,
    batch_size: int,
) -> tuple[float, list[float]]:
    if len(idx) == 0:
        return float("nan"), []

    mse_sum = 0.0
    n = 0
    per_h_sum = np.zeros((data.horizon,), dtype=np.float64)

    for s in range(0, len(idx), batch_size):
        bi = idx[s : s + batch_size]
        obs = data.obs[bi]
        tgt = data.acts_flat[bi]
        pred, _ = policy.predict(obs, deterministic=True)
        diff = pred - tgt
        sq = diff * diff
        mse_sum += float(np.mean(sq)) * len(bi)
        n += len(bi)

        sq_h = sq.reshape(len(bi), data.horizon, data.act_dim).mean(axis=2)  # [B,H]
        per_h_sum += sq_h.sum(axis=0)

    mse = mse_sum / max(1, n)
    per_h = (per_h_sum / max(1, n)).tolist()
    return float(mse), per_h


def _save_checkpoint(
    out_dir: Path,
    trainer: bc.BC,
    args: argparse.Namespace,
    data: ChunkData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_mse: float,
    val_mse: float,
    val_h_mse: list[float],
) -> Path:
    cfg = dict(vars(args))
    cfg.pop("func", None)
    ckpt = {
        "policy_state_dict": trainer.policy.state_dict(),
        "obs_dim": data.obs_dim,
        "act_dim": data.act_dim,
        "horizon": data.horizon,
        "hidden": _parse_csv_ints(args.hidden),
        "activation": args.activation,
        "config": cfg,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "val_horizon_mse": val_h_mse,
    }
    ckpt_path = out_dir / "bc_chunk_best.pt"
    th.save(ckpt, ckpt_path)
    return ckpt_path


def _load_bundle(checkpoint: str, device: str = "auto") -> ChunkPolicyBundle:
    map_dev = "cpu" if device == "auto" else device
    ckpt = th.load(checkpoint, map_location=map_dev, weights_only=False)

    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    horizon = int(ckpt["horizon"])
    hidden = list(ckpt["hidden"])
    activation = str(ckpt.get("activation", "tanh"))
    lr = float(ckpt.get("config", {}).get("lr", 3e-4))

    env = _make_env(max_steps=10)
    obs_space = env.observation_space
    chunk_action_space = gym.spaces.Box(
        low=-np.ones((horizon * act_dim,), dtype=np.float32),
        high=np.ones((horizon * act_dim,), dtype=np.float32),
        dtype=np.float32,
    )
    policy = _make_policy(obs_space, chunk_action_space, lr=lr, hidden=hidden, activation=activation)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return ChunkPolicyBundle(policy=policy, obs_dim=obs_dim, act_dim=act_dim, horizon=horizon)


def cmd_train(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    th.manual_seed(args.seed)

    traj_paths = [Path(args.trajs_pkl)]
    if args.extra_trajs_pkl.strip():
        traj_paths.extend(Path(p.strip()) for p in args.extra_trajs_pkl.split(",") if p.strip())
    missing = [str(p) for p in traj_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectory caches: {missing}")

    trajs = _load_trajs(traj_paths)
    data = _build_chunk_data(trajs, horizon=args.horizon, max_windows=args.max_windows)
    train_idx, val_idx = _split_train_val(data, val_frac=args.val_frac, seed=args.seed)

    train_transitions = _make_transitions(data, train_idx)

    env = _make_env(max_steps=args.max_steps)
    obs_space = env.observation_space
    chunk_action_space = gym.spaces.Box(
        low=-np.ones((data.horizon * data.act_dim,), dtype=np.float32),
        high=np.ones((data.horizon * data.act_dim,), dtype=np.float32),
        dtype=np.float32,
    )

    policy = _make_policy(obs_space, chunk_action_space, lr=args.lr, hidden=_parse_csv_ints(args.hidden), activation=args.activation)
    bc_trainer = bc.BC(
        observation_space=obs_space,
        action_space=chunk_action_space,
        demonstrations=train_transitions,
        policy=policy,
        rng=rng,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.lr},
        device=args.device,
    )
    # Keep BC flow consistent with imitation docs and previous train_bc scripts:
    # one direct call to train for n_epochs.
    bc_trainer.train(n_epochs=args.epochs, progress_bar=True)
    train_mse, _ = _policy_mse_on_indices(bc_trainer.policy, data, train_idx, args.batch_size)
    val_mse, val_h_mse = _policy_mse_on_indices(bc_trainer.policy, data, val_idx, args.batch_size)
    print(f"[bc_chunk] final train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

    ckpt = _save_checkpoint(
        out_dir=out_dir,
        trainer=bc_trainer,
        args=args,
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        train_mse=train_mse,
        val_mse=val_mse,
        val_h_mse=val_h_mse,
    )

    summary = {
        "obs_dim": data.obs_dim,
        "act_dim": data.act_dim,
        "horizon": data.horizon,
        "n_episodes_loaded": data.n_episodes,
        "n_windows": data.n_windows,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "val_horizon_mse": val_h_mse,
        "checkpoint": str(ckpt),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


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


def _set_env_from_obs(base_env: PayloadGymEnv, obs: np.ndarray) -> None:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n = base_env.n_bodies
    state_dim = base_env.state_dim
    raw = obs[:state_dim].astype(np.float64)
    expected = 7 * n + 6 * n
    if raw.size != expected:
        raise ValueError(f"raw_state len={raw.size}, expected={expected}")

    poses = raw[: 7 * n].copy()  # [pos, quat_xyzw]
    vels = raw[7 * n :].copy()
    for i in range(n):
        p0 = 7 * i
        qx, qy, qz, qw = poses[p0 + 3 : p0 + 7]
        poses[p0 + 3 : p0 + 7] = [qw, qx, qy, qz]  # xyzw -> wxyz

    base_env.data.qpos[:] = poses
    base_env.data.qvel[:] = vels
    mujoco.mj_forward(base_env.model, base_env.data)

    base_env.prev_action[:] = obs[-base_env.action_dim :].astype(np.float32)
    base_env.step_count = 0
    base_env.success_hold_count = 0


def cmd_eval(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint:
        raise ValueError("Missing checkpoint. Provide --checkpoint or set eval.checkpoint in --config YAML.")

    bundle = _load_bundle(args.checkpoint, device=args.device)

    env = _make_env(max_steps=args.max_steps)
    base_start = env.unwrapped.start_state.copy()  # type: ignore[attr-defined]
    rng = np.random.default_rng(args.seed)

    returns: list[float] = []
    lengths: list[int] = []
    replan_counts: list[int] = []
    success = 0
    payload_oob = 0
    quad_oob = 0
    reward_curves: list[np.ndarray] = []
    action_norm_curves: list[np.ndarray] = []
    payload_goal_dist_curves: list[np.ndarray] = []
    payload_vel_curves: list[np.ndarray] = []
    q1_vel_curves: list[np.ndarray] = []
    q2_vel_curves: list[np.ndarray] = []
    npz_log_dir = out_dir / "npz_eval"
    video_dir = out_dir / "videos_eval"
    if args.render_video:
        npz_log_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        if not HAS_VIDEO:
            print("[bc_chunk eval] WARN: videos_from_log not available, disabling video render.")
            args.render_video = False

    for ep in range(args.n_episodes):
        base_env: PayloadGymEnv = env.unwrapped  # type: ignore[assignment]
        _sample_reset(base_env, args.reset_sigma, rng, base_start)
        obs, _ = env.reset()
        qpos0 = base_env.data.qpos.copy()
        qvel0 = base_env.data.qvel.copy()
        ep_u_mujoco: list[np.ndarray] = []

        done = False
        ret = 0.0
        t = 0
        replans = 0
        since_replan = args.replan_k
        buffer: list[np.ndarray] = []
        last_info: dict[str, Any] = {}
        ep_rewards: list[float] = []
        ep_act_norms: list[float] = []
        ep_goal_dists: list[float] = []
        ep_payload_vel: list[np.ndarray] = []
        ep_q1_vel: list[np.ndarray] = []
        ep_q2_vel: list[np.ndarray] = []

        while not done:
            need_replan = (len(buffer) == 0) or (since_replan >= args.replan_k)
            if need_replan:
                pred_flat, _ = bundle.policy.predict(obs[None, :], deterministic=True)
                chunk = pred_flat.reshape(bundle.horizon, bundle.act_dim)
                buffer = [chunk[i].astype(np.float32) for i in range(chunk.shape[0])]
                since_replan = 0
                replans += 1

            act = buffer.pop(0)
            obs, rew, terminated, truncated, info = env.step(act)
            ret += float(rew)
            t += 1
            since_replan += 1
            done = bool(terminated or truncated)
            last_info = dict(info) if isinstance(info, dict) else {}
            ep_rewards.append(float(rew))
            ep_act_norms.append(float(np.linalg.norm(act)))
            ep_u_mujoco.append(np.asarray(base_env.action_mujoco, dtype=np.float32).copy())

            raw = np.asarray(obs[: base_env.state_dim], dtype=np.float32)
            n = base_env.n_bodies
            poses = raw[: 7 * n]
            vels = raw[7 * n :]
            pL = poses[0:3]
            pL_goal = np.asarray(base_env.goal[:3], dtype=np.float32)
            ep_goal_dists.append(float(np.linalg.norm(pL - pL_goal)))
            ep_payload_vel.append(vels[0:3].copy())
            if n >= 3:
                ep_q1_vel.append(vels[6:9].copy())
                ep_q2_vel.append(vels[12:15].copy())

        returns.append(ret)
        lengths.append(t)
        replan_counts.append(replans)
        reward_curves.append(np.asarray(ep_rewards, dtype=np.float32))
        action_norm_curves.append(np.asarray(ep_act_norms, dtype=np.float32))
        payload_goal_dist_curves.append(np.asarray(ep_goal_dists, dtype=np.float32))
        payload_vel_curves.append(np.asarray(ep_payload_vel, dtype=np.float32))
        if ep_q1_vel:
            q1_vel_curves.append(np.asarray(ep_q1_vel, dtype=np.float32))
        if ep_q2_vel:
            q2_vel_curves.append(np.asarray(ep_q2_vel, dtype=np.float32))
        success += int(bool(last_info.get("is_success", False)))
        payload_oob += int(bool(last_info.get("payload_out_of_bounds", False)))
        quad_oob += int(bool(last_info.get("quad_out_of_bounds", False)))

        if args.print_each_episode:
            print(
                f"[bc_chunk eval] ep={ep:03d} T={t} R={ret:.2f} replans={replans} "
                f"succ={int(last_info.get('is_success', False))} "
                f"payload_oob={int(last_info.get('payload_out_of_bounds', False))} "
                f"quad_oob={int(last_info.get('quad_out_of_bounds', False))}"
            )
        if args.render_video:
            npz_path = npz_log_dir / f"bc_chunk_eval_ep_{ep:03d}.npz"
            np.savez_compressed(
                npz_path,
                qpos0=qpos0,
                qvel0=qvel0,
                u_traj=np.asarray(ep_u_mujoco, dtype=np.float32),
            )
            views = [v.strip() for v in args.video_views.split(",") if v.strip()]
            low, high = base_env.workspace_bounds
            cfg = VideoConfig(
                out_dir=str(video_dir / f"ep_{ep:03d}"),
                fps=int(args.video_fps),
                views=views if views else ["diag"],
                env_min=np.asarray(low, dtype=float),
                env_max=np.asarray(high, dtype=float),
            )
            render_from_actions(
                xml_path=str(XML_PATH),
                init_and_actions_or_npz=str(npz_path),
                cfg=cfg,
            )

    summary = {
        "n_episodes": int(args.n_episodes),
        "max_steps": int(args.max_steps),
        "reset_sigma": float(args.reset_sigma),
        "replan_k": int(args.replan_k),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "replans_mean": float(np.mean(replan_counts)),
        "success_rate": float(success / max(1, args.n_episodes)),
        "payload_oob_rate": float(payload_oob / max(1, args.n_episodes)),
        "quad_oob_rate": float(quad_oob / max(1, args.n_episodes)),
        "checkpoint": str(args.checkpoint),
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    reward_curves_plot = _transform_reward_curves(
        reward_curves,
        clip_min=args.reward_clip_min,
        clip_max=args.reward_clip_max,
        normalize=args.reward_normalize,
    )
    reward_ylabel = "reward (normalized)" if args.reward_normalize else "reward"
    _plot_1d(reward_curves_plot, out_dir / "rewards_per_timestep.png", "Per-timestep reward", reward_ylabel)
    _plot_1d(action_norm_curves, out_dir / "action_norm.png", "Action norm", "||a||")
    _plot_1d(payload_goal_dist_curves, out_dir / "payload_goal_distance.png", "Payload goal distance", "m")
    _plot_3axis(payload_vel_curves, out_dir / "payload_velocity.png", "Payload velocity", "m/s")
    _plot_3axis(q1_vel_curves, out_dir / "q1_velocity.png", "Quad 1 velocity", "m/s")
    _plot_3axis(q2_vel_curves, out_dir / "q2_velocity.png", "Quad 2 velocity", "m/s")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval_horizon(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint:
        raise ValueError("Missing checkpoint. Provide --checkpoint or set eval-horizon.checkpoint in --config YAML.")

    bundle = _load_bundle(args.checkpoint, device=args.device)

    traj_paths = [Path(args.trajs_pkl)]
    if args.extra_trajs_pkl.strip():
        traj_paths.extend(Path(p.strip()) for p in args.extra_trajs_pkl.split(",") if p.strip())
    missing = [str(p) for p in traj_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectory caches: {missing}")

    trajs = _load_trajs(traj_paths)
    obs0_all, expert_next_all = _build_obs_horizon_windows(
        trajs,
        horizon=bundle.horizon,
        max_windows=args.max_windows,
    )

    env = PayloadGymEnv(
        xml_path=str(XML_PATH),
        template_yaml_path=str(TEMPLATE_YAML),
        max_steps=max(bundle.horizon + 5, args.max_steps),
    )
    env.terminate_on_success = True
    base_env: PayloadGymEnv = env
    state_dim = env.state_dim

    N = obs0_all.shape[0]
    H = bundle.horizon
    pred_next_all = np.full_like(expert_next_all, np.nan, dtype=np.float32)
    terminated_early = 0

    for i in range(N):
        obs0 = obs0_all[i]
        _set_env_from_obs(base_env, obs0)
        pred_flat, _ = bundle.policy.predict(obs0[None, :], deterministic=True)
        chunk = pred_flat.reshape(H, bundle.act_dim).astype(np.float32)
        obs_cur = obs0
        for k in range(H):
            obs_cur, rew, terminated, truncated, info = env.step(chunk[k])
            pred_next_all[i, k] = obs_cur
            if terminated or truncated:
                terminated_early += 1
                break

    E = pred_next_all[:, :, :state_dim] - expert_next_all[:, :, :state_dim]
    valid = np.sum(~np.isnan(E))
    raw_state_mse = float(np.nansum(E * E) / max(1, valid))
    raw_state_mae = float(np.nansum(np.abs(E)) / max(1, valid))

    mse_h = np.nanmean(E * E, axis=(0, 2))  # [H]
    d_payload = pred_next_all[:, :, 0:3] - expert_next_all[:, :, 0:3]
    payload_pos_mae_h = np.nanmean(np.linalg.norm(d_payload, axis=2), axis=0)  # [H]

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_episodes_loaded": len(trajs),
        "n_windows": int(N),
        "horizon": int(H),
        "state_dim_compared": int(state_dim),
        "raw_state_mse": raw_state_mse,
        "raw_state_mae": raw_state_mae,
        "mse_per_horizon_step": mse_h.tolist(),
        "payload_pos_mae_per_horizon_step": payload_pos_mae_h.tolist(),
        "terminated_or_truncated_early_windows": int(terminated_early),
    }
    (out_dir / "horizon_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    xh = np.arange(H)
    plt.figure(figsize=(8, 4))
    plt.plot(xh, mse_h, marker="o", linewidth=2.0)
    plt.xlabel("horizon step k")
    plt.ylabel("MSE (raw state)")
    plt.title("Predicted-horizon raw-state MSE vs expert")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "horizon_raw_state_mse.png", dpi=170)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(xh, payload_pos_mae_h, marker="o", linewidth=2.0)
    plt.xlabel("horizon step k")
    plt.ylabel("MAE payload pos [m]")
    plt.title("Predicted-horizon payload position error vs expert")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "horizon_payload_pos_mae.png", dpi=170)
    plt.close()

    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval_offline(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint:
        raise ValueError("Missing checkpoint. Provide --checkpoint or set eval-offline.checkpoint in --config YAML.")

    bundle = _load_bundle(args.checkpoint, device=args.device)

    traj_paths = [Path(args.trajs_pkl)]
    if args.extra_trajs_pkl.strip():
        traj_paths.extend(Path(p.strip()) for p in args.extra_trajs_pkl.split(",") if p.strip())
    missing = [str(p) for p in traj_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectory caches: {missing}")
    trajs = _load_trajs(traj_paths)
    data = _build_chunk_data(trajs, horizon=bundle.horizon, max_windows=args.max_windows)

    pred_all: list[np.ndarray] = []
    tgt_all: list[np.ndarray] = []
    for s in range(0, data.n_windows, args.batch_size):
        obs = data.obs[s : s + args.batch_size]
        tgt = data.acts_flat[s : s + args.batch_size]
        pred, _ = bundle.policy.predict(obs, deterministic=True)
        pred_all.append(np.asarray(pred, dtype=np.float32))
        tgt_all.append(np.asarray(tgt, dtype=np.float32))

    P = np.concatenate(pred_all, axis=0)  # [N, H*U]
    T = np.concatenate(tgt_all, axis=0)   # [N, H*U]
    E = P - T
    absE = np.abs(E)
    sqE = E * E

    mse = float(np.mean(sqE))
    mae = float(np.mean(absE))
    maxae = float(np.max(absE))

    N, HU = P.shape
    H = bundle.horizon
    U = bundle.act_dim
    sq_h = sqE.reshape(N, H, U).mean(axis=2)  # [N,H]
    mae_h = absE.reshape(N, H, U).mean(axis=2)
    mse_h = sq_h.mean(axis=0).tolist()
    mae_h_mean = mae_h.mean(axis=0).tolist()

    sq_u = sqE.reshape(N, H, U).mean(axis=1)  # [N,U]
    mae_u = absE.reshape(N, H, U).mean(axis=1)
    mse_u = sq_u.mean(axis=0).tolist()
    mae_u_mean = mae_u.mean(axis=0).tolist()

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_episodes_loaded": len(trajs),
        "n_windows": int(N),
        "horizon": int(H),
        "act_dim": int(U),
        "mse": mse,
        "mae": mae,
        "max_abs_error": maxae,
        "mse_per_horizon_step": mse_h,
        "mae_per_horizon_step": mae_h_mean,
        "mse_per_action_dim": mse_u,
        "mae_per_action_dim": mae_u_mean,
    }
    (out_dir / "offline_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot 1: per-horizon-step error
    xh = np.arange(H)
    plt.figure(figsize=(8, 4))
    plt.plot(xh, mse_h, marker="o", linewidth=2.0, label="MSE")
    plt.plot(xh, mae_h_mean, marker="o", linewidth=2.0, label="MAE")
    plt.xlabel("horizon step k")
    plt.ylabel("error")
    plt.title("Chunk prediction error vs horizon step")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "offline_error_by_horizon.png", dpi=170)
    plt.close()

    # Plot 2: per-action-dimension MAE
    xu = np.arange(U)
    plt.figure(figsize=(8, 4))
    plt.bar(xu, mae_u_mean)
    plt.xlabel("action dimension")
    plt.ylabel("MAE")
    plt.title("Chunk prediction MAE by action dimension")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "offline_mae_by_action_dim.png", dpi=170)
    plt.close()

    # Plot 3: predicted vs target for a sampled window (all actions across horizon)
    rng = np.random.default_rng(args.seed)
    i = int(rng.integers(0, N))
    p_i = P[i].reshape(H, U)
    t_i = T[i].reshape(H, U)
    fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    axs = axs.flatten()
    for u in range(min(U, 8)):
        ax = axs[u]
        ax.plot(np.arange(H), t_i[:, u], label="expert", linewidth=2.0)
        ax.plot(np.arange(H), p_i[:, u], label="pred", linewidth=1.8, linestyle="--")
        ax.set_title(f"u{u}")
        ax.grid(True, alpha=0.25)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Sampled chunk: predicted vs expert")
    fig.tight_layout()
    fig.savefig(out_dir / "offline_sample_chunk_overlay.png", dpi=170)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Chunked-action BC using imitation.algorithms.bc.BC.")
    ap.add_argument("--config", type=str, default="", help="YAML config file with command defaults.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train chunked BC from expert trajectories.")
    tr.add_argument("--config", type=str, default="", help=argparse.SUPPRESS)
    tr.add_argument("--trajs-pkl", type=str, default=str(DEFAULT_TRAJS_PKL))
    tr.add_argument("--extra-trajs-pkl", type=str, default="")
    tr.add_argument("--out-dir", type=str, default="runs/bc_chunk")
    tr.add_argument("--horizon", type=int, default=10)
    tr.add_argument("--hidden", type=str, default="512,512,256")
    tr.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--val-frac", type=float, default=0.1)
    tr.add_argument("--max-windows", type=int, default=0)
    tr.add_argument("--max-steps", type=int, default=200)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", type=str, default="auto")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="Closed-loop chunk-buffer evaluation.")
    ev.add_argument("--config", type=str, default="", help=argparse.SUPPRESS)
    ev.add_argument("--checkpoint", type=str, default="")
    ev.add_argument("--out-dir", type=str, default="runs/bc_chunk_eval")
    ev.add_argument("--n-episodes", type=int, default=20)
    ev.add_argument("--max-steps", type=int, default=200)
    ev.add_argument("--replan-k", type=int, default=10)
    ev.add_argument("--reset-sigma", type=float, default=0.0)
    ev.add_argument("--seed", type=int, default=123)
    ev.add_argument("--device", type=str, default="auto")
    ev.add_argument("--print-each-episode", action="store_true")
    ev.add_argument("--reward-clip-min", type=float, default=-25.0)
    ev.add_argument("--reward-clip-max", type=float, default=5.0)
    ev.add_argument("--reward-normalize", action="store_true")
    ev.add_argument("--render-video", action="store_true")
    ev.add_argument("--video-views", type=str, default="diag")
    ev.add_argument("--video-fps", type=int, default=50)
    ev.set_defaults(func=cmd_eval)

    evo = sub.add_parser("eval-offline", help="Offline chunk comparison on expert-observation windows.")
    evo.add_argument("--config", type=str, default="", help=argparse.SUPPRESS)
    evo.add_argument("--checkpoint", type=str, default="")
    evo.add_argument("--trajs-pkl", type=str, default=str(DEFAULT_TRAJS_PKL))
    evo.add_argument("--extra-trajs-pkl", type=str, default="")
    evo.add_argument("--out-dir", type=str, default="runs/bc_chunk_offline_eval")
    evo.add_argument("--max-windows", type=int, default=0)
    evo.add_argument("--batch-size", type=int, default=1024)
    evo.add_argument("--seed", type=int, default=123)
    evo.add_argument("--device", type=str, default="auto")
    evo.set_defaults(func=cmd_eval_offline)

    eh = sub.add_parser("eval-horizon", help="Compare predicted H-step observation rollout vs expert observations.")
    eh.add_argument("--config", type=str, default="", help=argparse.SUPPRESS)
    eh.add_argument("--checkpoint", type=str, default="")
    eh.add_argument("--trajs-pkl", type=str, default=str(DEFAULT_TRAJS_PKL))
    eh.add_argument("--extra-trajs-pkl", type=str, default="")
    eh.add_argument("--out-dir", type=str, default="runs/bc_chunk_horizon_eval")
    eh.add_argument("--max-windows", type=int, default=500)
    eh.add_argument("--max-steps", type=int, default=200)
    eh.add_argument("--device", type=str, default="auto")
    eh.set_defaults(func=cmd_eval_horizon)

    return ap, {"train": tr, "eval": ev, "eval-offline": evo, "eval-horizon": eh}


def _load_config_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg)}")
    return cfg


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre.add_argument("cmd", nargs="?", choices=["train", "eval", "eval-offline", "eval-horizon"])
    pre_args, _ = pre.parse_known_args(sys.argv[1:])

    parser, subparsers = build_parser()
    if pre_args.config:
        cfg = _load_config_yaml(pre_args.config)

        common = cfg.get("common", {})
        if common:
            if not isinstance(common, dict):
                raise ValueError("`common` in config must be a mapping.")
            parser.set_defaults(**common)

        if pre_args.cmd:
            cmd_cfg = cfg.get(pre_args.cmd, {})
            if cmd_cfg:
                if not isinstance(cmd_cfg, dict):
                    raise ValueError(f"`{pre_args.cmd}` in config must be a mapping.")
                subparsers[pre_args.cmd].set_defaults(**cmd_cfg)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
