#!/usr/bin/env python3
"""
BC initialization tuning script for DAgger warm-start.

Sweeps over: policy architecture, learning rate, batch size, epochs,
data quantity, and whether to include/zero-out prev_action in observations.

Usage examples:
  # Full grid sweep (default)
  python train_bc_init_tuning.py

  # Quick single run
  python train_bc_init_tuning.py --no-sweep

  # Test if prev_action matters
  python train_bc_init_tuning.py --ablate-prev-action

  # Use more data from extra pkl
  python train_bc_init_tuning.py --extra-trajs-pkl path/to/more.pkl

  # Override architecture
  python train_bc_init_tuning.py --archs "128x128,256x256"
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch as th
import torch.nn as nn
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Trajectory
from payload_env import PayloadGymEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
XML_PATH = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
TEMPLATE_YAML = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
DEFAULT_TRAJS_PKL = (
    REPO / "runs/dagger_10000steps"
    / "round0_cache_obs_payload_rel_v2_seeddelta_0.70_0.75"
    / "expert_trajs.pkl"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_trajs(paths: Iterable[Path]) -> list[Trajectory]:
    all_trajs: list[Trajectory] = []
    for p in paths:
        with open(p, "rb") as f:
            trajs = pickle.load(f)
        if not trajs:
            continue
        all_trajs.extend(trajs)
        total_steps = sum(t.acts.shape[0] for t in trajs)
        print(f"[bc] loaded {len(trajs)} episodes ({total_steps} steps) from {p}")
    return all_trajs


def _make_env(max_steps: int, zero_prev_action: bool = False) -> DummyVecEnv:
    """Create a vectorized env.  If zero_prev_action=True, we patch the env
    so that prev_action is always zeros (ablation experiment)."""

    def _thunk():
        env = PayloadGymEnv(
            xml_path=str(XML_PATH),
            template_yaml_path=str(TEMPLATE_YAML),
            max_steps=max_steps,
        )
        env.terminate_on_success = True
        if zero_prev_action:
            # Monkey-patch step so prev_action stays zero
            _orig_step = env.step

            def _patched_step(action):
                result = _orig_step(action)
                env.prev_action[:] = 0.0
                return result

            env.step = _patched_step
        return Monitor(env)

    return DummyVecEnv([_thunk])


def _zero_prev_action_in_trajs(trajs: list[Trajectory], act_dim: int) -> list[Trajectory]:
    """Return a copy of trajectories with the prev_action portion of obs zeroed out."""
    new_trajs = []
    for traj in trajs:
        obs = traj.obs.copy()
        obs[:, -act_dim:] = 0.0
        new_trajs.append(
            Trajectory(obs=obs, acts=traj.acts.copy(), infos=traj.infos, terminal=traj.terminal)
        )
    return new_trajs


def _subsample_trajs(trajs: list[Trajectory], max_eps: int, seed: int) -> list[Trajectory]:
    if max_eps <= 0 or max_eps >= len(trajs):
        return trajs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(trajs), size=max_eps, replace=False)
    idx.sort()
    return [trajs[i] for i in idx]


def _build_policy(obs_space, act_space, lr: float, arch: list[int], activation: str):
    act_cls = nn.Tanh if activation.lower() == "tanh" else nn.ReLU
    return ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: lr,
        net_arch=arch,
        activation_fn=act_cls,
        ortho_init=False,
    )


def _evaluate_rollouts(policy, venv, n_episodes: int = 10):
    """Evaluate and return detailed metrics beyond just mean return."""
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs = venv.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            act, _ = policy.predict(obs, deterministic=True)
            obs, rew, dones, infos = venv.step(act)
            ep_reward += float(rew[0])
            ep_len += 1
            done = dones[0]
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    return {
        "mean_return": float(rewards.mean()),
        "std_return": float(rewards.std()),
        "min_return": float(rewards.min()),
        "max_return": float(rewards.max()),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std()),
        "success_rate": float(np.mean(lengths < 200)),  # finished before max_steps
    }


def _run_one(
    *,
    venv: DummyVecEnv,
    transitions,
    rng: np.random.Generator,
    lr: float,
    batch_size: int,
    epochs: int,
    arch: list[int],
    activation: str,
    ent_weight: float,
    l2_weight: float,
    eval_eps: int,
):
    policy = _build_policy(
        venv.observation_space, venv.action_space,
        lr=lr, arch=arch, activation=activation,
    )
    trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=policy,
        rng=rng,
        batch_size=batch_size,
        optimizer_kwargs=dict(lr=lr),
        ent_weight=ent_weight,
        l2_weight=l2_weight,
    )

    t0 = time.perf_counter()
    trainer.train(n_epochs=epochs)
    train_time = time.perf_counter() - t0

    metrics = _evaluate_rollouts(trainer.policy, venv, n_episodes=eval_eps)
    metrics["train_time_s"] = round(train_time, 1)
    return trainer, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="BC initialization tuning for DAgger warm-start.")

    # Data
    ap.add_argument("--trajs-pkl", type=str, default=str(DEFAULT_TRAJS_PKL),
                    help="Primary expert trajectory cache (pkl).")
    ap.add_argument("--extra-trajs-pkl", type=str, default="",
                    help="Comma-separated extra pkl paths to merge.")
    ap.add_argument("--max-episodes-list", type=str, default="0",
                    help="Comma-separated episode counts to try (0=all). E.g. '10,20,50'")

    # Env
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=str, default="runs/bc_init_tuning")

    # Sweep knobs
    ap.add_argument("--lrs", type=str, default="1e-3,5e-4,1e-4")
    ap.add_argument("--batch-sizes", type=str, default="128,256")
    ap.add_argument("--epochs-list", type=str, default="200,400,800")
    ap.add_argument("--archs", type=str, default="128x128,256x256,256x256x128")
    ap.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
    ap.add_argument("--ent-weight", type=float, default=1e-3,
                    help="Entropy bonus to prevent variance collapse.")
    ap.add_argument("--l2-weight", type=float, default=0.0)

    # Ablations
    ap.add_argument("--ablate-prev-action", action="store_true",
                    help="Also run each config with prev_action zeroed to test its effect.")
    ap.add_argument("--no-sweep", action="store_true",
                    help="Train a single config (first values only).")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    th.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    traj_paths = [Path(args.trajs_pkl)]
    if args.extra_trajs_pkl.strip():
        traj_paths.extend(Path(p.strip()) for p in args.extra_trajs_pkl.split(",") if p.strip())
    missing = [str(p) for p in traj_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectory caches: {missing}")

    all_trajs = _load_trajs(traj_paths)
    if not all_trajs:
        raise RuntimeError("No trajectories loaded.")

    act_dim = all_trajs[0].acts.shape[1]
    obs_dim = all_trajs[0].obs.shape[1]
    total_steps = sum(t.acts.shape[0] for t in all_trajs)
    print(f"\n[bc] Total data: {len(all_trajs)} episodes, {total_steps} transitions")
    print(f"[bc] obs_dim={obs_dim}  act_dim={act_dim}")

    # Summarize obs range
    all_obs = np.concatenate([t.obs for t in all_trajs], axis=0)
    all_acts = np.concatenate([t.acts for t in all_trajs], axis=0)
    print(f"[bc] Obs range: [{all_obs.min():.3f}, {all_obs.max():.3f}]  mean={all_obs.mean():.4f}")
    print(f"[bc] Act range: [{all_acts.min():.3f}, {all_acts.max():.3f}]  mean={all_acts.mean():.4f}")
    print(f"[bc] Prev_action (last {act_dim} obs dims) range: "
          f"[{all_obs[:, -act_dim:].min():.3f}, {all_obs[:, -act_dim:].max():.3f}]  "
          f"mean={all_obs[:, -act_dim:].mean():.4f}  std={all_obs[:, -act_dim:].std():.4f}")

    # ------------------------------------------------------------------
    # Build sweep grid
    # ------------------------------------------------------------------
    lrs = _parse_csv_floats(args.lrs)
    bss = _parse_csv_ints(args.batch_sizes)
    eps_list = _parse_csv_ints(args.epochs_list)
    archs = [[int(x) for x in a.split("x")] for a in args.archs.split(",") if a.strip()]
    max_ep_list = _parse_csv_ints(args.max_episodes_list)

    prev_action_modes = ["normal"]
    if args.ablate_prev_action:
        prev_action_modes.append("zeroed")

    if args.no_sweep:
        grid = [(lrs[0], bss[0], eps_list[0], archs[0], max_ep_list[0], prev_action_modes[0])]
    else:
        grid = [
            (lr, bs, ep, arch, max_eps, pa_mode)
            for lr in lrs
            for bs in bss
            for ep in eps_list
            for arch in archs
            for max_eps in max_ep_list
            for pa_mode in prev_action_modes
        ]

    print(f"\n[bc] Grid size: {len(grid)} configurations")
    print(f"[bc] LRs: {lrs}")
    print(f"[bc] Batch sizes: {bss}")
    print(f"[bc] Epochs: {eps_list}")
    print(f"[bc] Architectures: {archs}")
    print(f"[bc] Max episodes: {max_ep_list}")
    print(f"[bc] Prev action modes: {prev_action_modes}")

    # ------------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------------
    results = []
    best = None
    best_trainer = None

    venv_normal = _make_env(max_steps=args.max_steps, zero_prev_action=False)
    venv_zeroed = _make_env(max_steps=args.max_steps, zero_prev_action=True) if args.ablate_prev_action else None

    for i, (lr, bs, ep, arch, max_eps, pa_mode) in enumerate(grid):
        tag = f"lr{lr}_bs{bs}_ep{ep}_arch{'x'.join(map(str,arch))}_neps{max_eps}_pa{pa_mode}"
        print(f"\n{'='*60}")
        print(f"[bc] Trial {i+1}/{len(grid)}: {tag}")
        print(f"{'='*60}")

        # Subsample if requested
        trajs = _subsample_trajs(all_trajs, max_eps, args.seed) if max_eps > 0 else all_trajs

        # Ablate prev_action if requested
        if pa_mode == "zeroed":
            trajs = _zero_prev_action_in_trajs(trajs, act_dim)
            venv = venv_zeroed
        else:
            venv = venv_normal

        n_steps = sum(t.acts.shape[0] for t in trajs)
        print(f"[bc] Using {len(trajs)} episodes, {n_steps} transitions, prev_action={pa_mode}")

        transitions = rollout.flatten_trajectories(trajs)

        # Clamp batch size to available transitions
        actual_bs = min(bs, len(transitions))

        trainer, metrics = _run_one(
            venv=venv,
            transitions=transitions,
            rng=np.random.default_rng(args.seed + i),
            lr=lr,
            batch_size=actual_bs,
            epochs=ep,
            arch=arch,
            activation=args.activation,
            ent_weight=args.ent_weight,
            l2_weight=args.l2_weight,
            eval_eps=args.eval_episodes,
        )

        row = {
            "trial": i,
            "tag": tag,
            "lr": lr,
            "batch_size": actual_bs,
            "epochs": ep,
            "arch": arch,
            "activation": args.activation,
            "n_episodes": len(trajs),
            "n_transitions": n_steps,
            "prev_action_mode": pa_mode,
            "ent_weight": args.ent_weight,
            "l2_weight": args.l2_weight,
            **metrics,
        }
        results.append(row)

        print(f"[bc] Result: mean_ret={metrics['mean_return']:.1f}  "
              f"std_ret={metrics['std_return']:.1f}  "
              f"success={metrics['success_rate']:.0%}  "
              f"mean_len={metrics['mean_length']:.0f}  "
              f"train_time={metrics['train_time_s']:.1f}s")

        if best is None or row["mean_return"] > best["mean_return"]:
            best = row
            best_trainer = trainer
            th.save(trainer.policy.state_dict(), out_root / "best_policy_state_dict.pt")
            print(f"[bc] ** New best! Saved to {out_root / 'best_policy_state_dict.pt'}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results.sort(key=lambda r: r["mean_return"], reverse=True)
    (out_root / "bc_tuning_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (out_root / "bc_tuning_best.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("BC TUNING RESULTS (sorted by mean return, best first)")
    print(f"{'='*80}")
    print(f"{'#':>3} {'arch':<18} {'lr':>8} {'bs':>4} {'ep':>4} {'n_ep':>5} {'pa':>7} "
          f"{'ret':>8} {'std':>6} {'succ':>5} {'len':>5} {'time':>6}")
    print("-" * 80)
    for r in results:
        arch_str = "x".join(map(str, r["arch"]))
        print(f"{r['trial']:>3} {arch_str:<18} {r['lr']:>8.0e} {r['batch_size']:>4} "
              f"{r['epochs']:>4} {r['n_episodes']:>5} {r['prev_action_mode']:>7} "
              f"{r['mean_return']:>8.1f} {r['std_return']:>6.1f} "
              f"{r['success_rate']:>5.0%} {r['mean_length']:>5.0f} "
              f"{r['train_time_s']:>6.1f}")

    print(f"\n[bc] Best config: {best['tag']}")
    print(f"[bc] Best return: {best['mean_return']:.1f} +/- {best['std_return']:.1f}")
    print(f"[bc] Best success rate: {best['success_rate']:.0%}")
    print(f"[bc] Results saved to: {out_root / 'bc_tuning_results.json'}")
    print(f"[bc] Best policy saved to: {out_root / 'best_policy_state_dict.pt'}")

    # Cleanup
    venv_normal.close()
    if venv_zeroed is not None:
        venv_zeroed.close()


if __name__ == "__main__":
    main()
