#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _last_info_dict(infos: list[str]) -> dict[str, Any] | None:
    if not infos:
        return None
    for s in reversed(infos):
        d = _safe_json_loads(s)
        if isinstance(d, dict):
            return d
    return None


@dataclass
class DemoStats:
    name: str
    steps: int
    return_sum: float
    terminal: bool
    final_dist_goal: float | None
    out_of_bounds: bool | None
    payload_oob: bool | None
    quad_oob: bool | None
    is_success: bool | None
    time_limit_truncated: bool | None


def _demo_stats(demo_dir: Path) -> DemoStats:
    ds = _load_dataset(demo_dir)
    row = ds[0]
    obs = row["obs"]
    acts = row["acts"]
    rews = row.get("rews", [])
    infos = row.get("infos", [])
    terminal = bool(row.get("terminal", False))
    # trajectory convention: len(obs)=len(acts)+1 in many cases
    steps = len(acts)
    if steps == 0 and len(rews):
        steps = len(rews)
    ret = float(sum(rews)) if rews is not None else float("nan")
    info_last = _last_info_dict(infos)

    def _get(k: str):
        if not isinstance(info_last, dict):
            return None
        return info_last.get(k)

    return DemoStats(
        name=demo_dir.name,
        steps=steps,
        return_sum=ret,
        terminal=terminal,
        final_dist_goal=float(_get("dist_goal")) if _get("dist_goal") is not None else None,
        out_of_bounds=bool(_get("out_of_bounds")) if _get("out_of_bounds") is not None else None,
        payload_oob=bool(_get("payload_out_of_bounds")) if _get("payload_out_of_bounds") is not None else None,
        quad_oob=bool(_get("quad_out_of_bounds")) if _get("quad_out_of_bounds") is not None else None,
        is_success=bool(_get("is_success")) if _get("is_success") is not None else None,
        time_limit_truncated=bool(_get("TimeLimit.truncated")) if _get("TimeLimit.truncated") is not None else None,
    )


def cmd_list(args: argparse.Namespace) -> int:
    demos_root = Path(args.demos_root)
    rounds = _round_dirs(demos_root)
    print(f"demos_root: {demos_root}")
    print(f"rounds: {len(rounds)}")
    for rd in rounds:
        demos = _demo_dirs(rd)
        print(f"{rd.name}: demos={len(demos)}")
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
        demo_dirs = _demo_dirs(rd)
        stats = [_demo_stats(d) for d in demo_dirs]
        if not stats:
            print(f"{rd.name}: demos=0")
            continue
        n = len(stats)
        steps = [s.steps for s in stats]
        returns = [s.return_sum for s in stats]
        succ = sum(bool(s.is_success) for s in stats if s.is_success is not None)
        payload_oob = sum(bool(s.payload_oob) for s in stats if s.payload_oob is not None)
        quad_oob = sum(bool(s.quad_oob) for s in stats if s.quad_oob is not None)
        tl = sum(bool(s.time_limit_truncated) for s in stats if s.time_limit_truncated is not None)
        print(
            f"{rd.name}: demos={n} "
            f"steps(mean/min/max)={sum(steps)/n:.1f}/{min(steps)}/{max(steps)} "
            f"return(mean)={sum(returns)/n:.2f} "
            f"success={succ} payload_oob={payload_oob} quad_oob={quad_oob} time_limit={tl}"
        )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    demos_root = Path(args.demos_root)
    round_dir = demos_root / f"round-{int(args.round):03d}"
    demos = _demo_dirs(round_dir)
    if not demos:
        raise SystemExit(f"No demos in {round_dir}")
    idx = int(args.demo_index)
    if idx < 0 or idx >= len(demos):
        raise SystemExit(f"demo_index out of range [0,{len(demos)-1}]")
    demo_dir = demos[idx]
    ds = _load_dataset(demo_dir)
    row = ds[0]
    stats = _demo_stats(demo_dir)
    print(f"demo_dir: {demo_dir}")
    print(f"stats: {stats}")
    print(f"columns: {ds.column_names}")
    print(f"obs_len={len(row['obs'])} acts_len={len(row['acts'])} rews_len={len(row['rews'])}")
    if row["obs"]:
        print(f"obs[0]_dim={len(row['obs'][0])}")
        print(f"obs[0][:16]={row['obs'][0][:16]}")
    if row["acts"]:
        print(f"acts[0]_dim={len(row['acts'][0])}")
        print(f"acts[0]={row['acts'][0]}")
    if row["infos"]:
        print("first_info:", row["infos"][0])
        print("last_info:", row["infos"][-1])
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    import sys
    import numpy as np

    sys.path.append("scripts")
    from payload_env import PayloadGymEnv

    demos_root = Path(args.demos_root)
    round_dir = demos_root / f"round-{int(args.round):03d}"
    demos = _demo_dirs(round_dir)
    demo_dir = demos[int(args.demo_index)]
    ds = _load_dataset(demo_dir)
    row = ds[0]

    xml_path = Path(args.xml_path)
    yaml_path = Path(args.env_yaml)
    env = PayloadGymEnv(xml_path=str(xml_path), template_yaml_path=str(yaml_path), max_steps=args.max_steps)
    obs, _ = env.reset()
    acts = np.asarray(row["acts"], dtype=np.float32)
    rews = np.asarray(row["rews"], dtype=np.float32)
    obs_saved = np.asarray(row["obs"], dtype=np.float32)

    print(f"Replaying {demo_dir.name}: steps={len(acts)}")
    print(f"saved_obs_dim={obs_saved.shape[1] if obs_saved.ndim==2 else 'n/a'} env_obs_dim={obs.shape[0]}")
    max_steps = min(len(acts), args.limit_steps if args.limit_steps > 0 else len(acts))

    for t in range(max_steps):
        a = acts[t]
        obs2, rew, terminated, truncated, info = env.step(a)
        # Only compare raw-state prefix because learner features can evolve if code changed.
        raw_dim = env.state_dim
        raw_err = float(np.max(np.abs(obs_saved[t + 1][:raw_dim] - obs2[:raw_dim]))) if t + 1 < len(obs_saved) else float("nan")
        rew_err = float(abs(rews[t] - rew)) if t < len(rews) else float("nan")
        print(
            f"t={t:03d} rew={rew:+.4f} rew_err={rew_err:.3e} raw_obs_max_err={raw_err:.3e} "
            f"term={terminated} trunc={truncated} dist={info.get('dist_goal')}"
        )
        if terminated or truncated:
            break
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect imitation DAgger round demos (Arrow datasets).")
    ap.add_argument("--demos-root", default="runs/dagger_2000steps/scratch_dagger/demos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--round", type=int, default=None)

    p_insp = sub.add_parser("inspect")
    p_insp.add_argument("--round", type=int, required=True)
    p_insp.add_argument("--demo-index", type=int, default=0)

    p_rep = sub.add_parser("replay")
    p_rep.add_argument("--round", type=int, required=True)
    p_rep.add_argument("--demo-index", type=int, default=0)
    p_rep.add_argument("--xml-path", default="deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml")
    p_rep.add_argument("--env-yaml", default="deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml")
    p_rep.add_argument("--max-steps", type=int, default=1000)
    p_rep.add_argument("--limit-steps", type=int, default=20)

    args = ap.parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.cmd == "inspect":
        return cmd_inspect(args)
    if args.cmd == "replay":
        return cmd_replay(args)
    raise RuntimeError(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
