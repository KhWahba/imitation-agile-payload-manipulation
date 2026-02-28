#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
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


REPO = Path(__file__).resolve().parents[1]
XML_PATH = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
TEMPLATE_YAML = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
DEFAULT_TRAJS_PKL = REPO / "runs/dagger_10000steps/round0_cache_obs_payload_rel_v2_seeddelta_0.70_0.75/expert_trajs.pkl"


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
        print(f"[bc] loaded {len(trajs)} episodes from {p}")
    return all_trajs


def _make_env(max_steps: int) -> DummyVecEnv:
    def _thunk():
        env = PayloadGymEnv(xml_path=str(XML_PATH), template_yaml_path=str(TEMPLATE_YAML), max_steps=max_steps)
        env.terminate_on_success = True
        return Monitor(env)

    return DummyVecEnv([_thunk])


def _build_policy(obs_space, act_space, lr: float, arch: list[int], activation: str):
    act_cls = nn.Tanh if activation.lower() == "tanh" else nn.ReLU
    # BC optimizer is configured in bc.BC, but ActorCriticPolicy still requires lr_schedule.
    return ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: lr,
        net_arch=arch,
        activation_fn=act_cls,
        ortho_init=False,
    )


def _subsample_trajectories(trajs: list[Trajectory], max_eps: int | None, seed: int) -> list[Trajectory]:
    if max_eps is None or max_eps <= 0 or max_eps >= len(trajs):
        return trajs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(trajs), size=max_eps, replace=False)
    idx.sort()
    return [trajs[i] for i in idx]


def _flatten(trajs: list[Trajectory]):
    transitions = rollout.flatten_trajectories(trajs)
    print(f"[bc] transitions: {len(transitions)} obs={transitions.obs.shape} acts={transitions.acts.shape}")
    return transitions


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
    eval_eps: int,
):
    policy = _build_policy(venv.observation_space, venv.action_space, lr=lr, arch=arch, activation=activation)
    trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=policy,
        rng=rng,
        batch_size=batch_size,
        optimizer_kwargs=dict(lr=lr),
    )
    trainer.train(n_epochs=epochs)
    mean_ret, std_ret = evaluate_policy(trainer.policy, venv, n_eval_episodes=eval_eps, deterministic=True)
    return trainer, float(mean_ret), float(std_ret)


def main():
    ap = argparse.ArgumentParser(description="BC initializer tuning for current payload setup.")
    ap.add_argument("--trajs-pkl", type=str, default=str(DEFAULT_TRAJS_PKL))
    ap.add_argument("--extra-trajs-pkl", type=str, default="", help="Comma-separated extra pkl paths to merge.")
    ap.add_argument("--duplicate-data-factor", type=int, default=1, help="Repeat merged traj list N times (copy-paste augmentation).")
    ap.add_argument("--max-episodes", type=int, default=0, help="If >0, subsample this many trajectories.")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=str, default="runs/bc_init_tuning")
    # Sweep knobs
    ap.add_argument("--lrs", type=str, default="1e-3,5e-4,2e-4")
    ap.add_argument("--batch-sizes", type=str, default="128,256,512")
    ap.add_argument("--epochs-list", type=str, default="200,400")
    ap.add_argument("--archs", type=str, default="256x256,512x512,512x512x256")
    ap.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
    ap.add_argument("--no-sweep", action="store_true", help="Train a single config (first values only).")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    th.manual_seed(args.seed)

    traj_paths = [Path(args.trajs_pkl)]
    if args.extra_trajs_pkl.strip():
        traj_paths.extend(Path(p.strip()) for p in args.extra_trajs_pkl.split(",") if p.strip())
    missing = [str(p) for p in traj_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectory caches: {missing}")

    trajs = _load_trajs(traj_paths)
    if not trajs:
        raise RuntimeError("No trajectories loaded.")
    if args.max_episodes > 0:
        trajs = _subsample_trajectories(trajs, args.max_episodes, args.seed)
        print(f"[bc] after subsample: {len(trajs)} episodes")
    if args.duplicate_data_factor > 1:
        trajs = trajs * int(args.duplicate_data_factor)
        print(f"[bc] after duplication factor={args.duplicate_data_factor}: {len(trajs)} episodes")

    transitions = _flatten(trajs)
    venv = _make_env(max_steps=args.max_steps)

    lrs = _parse_csv_floats(args.lrs)
    bss = _parse_csv_ints(args.batch_sizes)
    eps = _parse_csv_ints(args.epochs_list)
    archs = [[int(x) for x in a.split("x")] for a in args.archs.split(",") if a.strip()]

    if args.no_sweep:
        grid = [(lrs[0], bss[0], eps[0], archs[0])]
    else:
        grid = [(lr, bs, ep, arch) for lr in lrs for bs in bss for ep in eps for arch in archs]

    results = []
    best = None
    for i, (lr, bs, ep, arch) in enumerate(grid):
        print(f"[bc] trial {i+1}/{len(grid)}: lr={lr} bs={bs} epochs={ep} arch={arch}")
        trainer, mean_ret, std_ret = _run_one(
            venv=venv,
            transitions=transitions,
            rng=np.random.default_rng(args.seed + i),
            lr=lr,
            batch_size=bs,
            epochs=ep,
            arch=arch,
            activation=args.activation,
            eval_eps=args.eval_episodes,
        )
        row = {
            "trial": i,
            "lr": lr,
            "batch_size": bs,
            "epochs": ep,
            "arch": arch,
            "activation": args.activation,
            "mean_return": mean_ret,
            "std_return": std_ret,
        }
        results.append(row)
        if best is None or row["mean_return"] > best["mean_return"]:
            best = row
            th.save(trainer.policy.state_dict(), out_root / "best_policy_state_dict.pt")

    results.sort(key=lambda r: r["mean_return"], reverse=True)
    (out_root / "bc_tuning_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_root / "bc_tuning_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("[bc] best:", best)
    print(f"[bc] wrote: {out_root / 'bc_tuning_results.json'}")
    print(f"[bc] wrote: {out_root / 'best_policy_state_dict.pt'}")


if __name__ == "__main__":
    main()
