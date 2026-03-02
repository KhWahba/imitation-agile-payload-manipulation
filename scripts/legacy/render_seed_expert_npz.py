#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _default_seed_cache_dir() -> Path:
    # Prefer current 10k run cache if present, otherwise fall back to 2k cache.
    cands = [
        Path("runs/dagger_10000steps/round0_cache_obs_payload_rel_v2_seeddelta_0.70_0.75"),
        Path("runs/dagger_2000steps/round0_cache_obs_payload_rel_v2_seeddelta_0.70_0.90"),
    ]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a seed expert episode NPZ to video.")
    ap.add_argument("--seed-cache-dir", default=str(_default_seed_cache_dir()))
    ap.add_argument("--ep", type=int, default=None, help="Episode index, e.g. 3 -> expert_ep_003.npz")
    ap.add_argument("--npz", type=str, default="", help="Direct path to expert_ep_XXX.npz (overrides --ep)")
    ap.add_argument("--xml-path", default="deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Default: <seed-cache-dir-parent>/videos_round0/expert_ep_XXX",
    )
    ap.add_argument("--views", nargs="+", default=["diag", "side", "top"])
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--env-min", nargs=3, type=float, default=[-2.0, -2.0, 0.0])
    ap.add_argument("--env-max", nargs=3, type=float, default=[2.0, 2.0, 2.0])
    args = ap.parse_args()

    import sys
    sys.path.append("scripts")
    from videos_from_log import VideoConfig, render_from_actions

    seed_cache_dir = Path(args.seed_cache_dir)
    logs_dir = seed_cache_dir / "npz_logs"
    if args.npz:
        npz_path = Path(args.npz)
    else:
        if args.ep is None:
            raise SystemExit("Provide either --ep or --npz")
        npz_path = logs_dir / f"expert_ep_{int(args.ep):03d}.npz"
    if not npz_path.exists():
        raise SystemExit(f"NPZ not found: {npz_path}")

    ep_tag = npz_path.stem
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Mirror collect_expert_trajs(video_root_dir=out_root/videos_round0)
        out_root = seed_cache_dir.parent
        out_dir = out_root / "videos_round0" / ep_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = VideoConfig(
        out_dir=str(out_dir),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        views=args.views,
        env_min=np.asarray(args.env_min, dtype=float),
        env_max=np.asarray(args.env_max, dtype=float),
    )
    written = render_from_actions(
        xml_path=str(args.xml_path),
        init_and_actions_or_npz=str(npz_path),
        cfg=cfg,
    )

    print(f"npz: {npz_path}")
    print("written:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
