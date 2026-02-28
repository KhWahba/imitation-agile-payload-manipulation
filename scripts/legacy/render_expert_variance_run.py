#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _map_policy_action_to_mujoco_ctrl(a_policy: np.ndarray, action_dim: int) -> np.ndarray:
    # Matches PayloadGymEnv.step() mapping.
    a = np.asarray(a_policy, dtype=np.float32).reshape(action_dim)
    planner_low = np.zeros((action_dim,), dtype=np.float32)
    planner_high = 1.4 * np.ones((action_dim,), dtype=np.float32)
    act_mid = 0.5 * (planner_low + planner_high)
    act_half = 0.5 * (planner_high - planner_low)
    action_planner = a * act_half + act_mid
    u_nominal = np.float32(0.034 * 9.81 / 4.0)
    return (action_planner * u_nominal).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render one expert_variance run rollout to video.")
    ap.add_argument("--run-dir", required=True, help="Path to runs/expert_variance/.../run_XX")
    ap.add_argument("--xml-path", default="deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml")
    ap.add_argument("--env-yaml", default="deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml")
    ap.add_argument("--out-dir", default="", help="Default: <run-dir>/videos")
    ap.add_argument("--views", nargs="+", default=["diag", "side", "top"])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    import sys
    sys.path.append("scripts")
    from payload_env import PayloadGymEnv
    from videos_from_log import VideoConfig, render_from_actions

    run_dir = Path(args.run_dir)
    npz_path = run_dir / "rollout_actions.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing {npz_path}. Re-run scripts/test_expert_variance_k10.py after latest patch.")

    d = np.load(npz_path)
    qpos0 = np.asarray(d["qpos0"], dtype=np.float32)
    qvel0 = np.asarray(d["qvel0"], dtype=np.float32)
    u_policy = np.asarray(d["u_policy"], dtype=np.float32)

    env = PayloadGymEnv(
        xml_path=str(args.xml_path),
        template_yaml_path=str(args.env_yaml),
        max_steps=200,
    )
    u_mujoco = np.stack([_map_policy_action_to_mujoco_ctrl(a, env.action_dim) for a in u_policy], axis=0)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VideoConfig(
        out_dir=str(out_dir),
        width=args.width,
        height=args.height,
        fps=args.fps,
        views=args.views,
        env_min=env.workspace_bounds[0],
        env_max=env.workspace_bounds[1],
    )
    written = render_from_actions(
        xml_path=str(args.xml_path),
        init_and_actions_or_npz=(qpos0, qvel0, u_mujoco),
        cfg=cfg,
        include_initial_frame=True,
    )
    print(f"run_dir: {run_dir}")
    print("written:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
