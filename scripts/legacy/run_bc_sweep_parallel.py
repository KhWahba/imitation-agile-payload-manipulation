#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRAIN_BC = REPO / "scripts/train_bc.py"


def _parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_archs(s: str):
    return [a.strip() for a in s.split(",") if a.strip()]


def _run_trial(
    idx: int,
    out_root: str,
    trajs_pkl: str,
    lr: float,
    batch_size: int,
    epochs: int,
    arch: str,
    activation: str,
    eval_episodes: int,
    max_steps: int,
):
    trial_dir = Path(out_root) / f"trial_{idx:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO / ".venv/bin/python"),
        str(TRAIN_BC),
        "--no-sweep",
        "--trajs-pkl",
        trajs_pkl,
        "--lrs",
        str(lr),
        "--batch-sizes",
        str(batch_size),
        "--epochs-list",
        str(epochs),
        "--archs",
        arch,
        "--activation",
        activation,
        "--eval-episodes",
        str(eval_episodes),
        "--max-steps",
        str(max_steps),
        "--out-root",
        str(trial_dir),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return {
            "trial": idx,
            "ok": False,
            "returncode": proc.returncode,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "arch": arch,
            "activation": activation,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    best_path = trial_dir / "bc_tuning_best.json"
    if not best_path.exists():
        return {
            "trial": idx,
            "ok": False,
            "returncode": -1,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "arch": arch,
            "activation": activation,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    row = json.loads(best_path.read_text(encoding="utf-8"))
    row["trial_id"] = idx
    row["ok"] = True
    row["trial_dir"] = str(trial_dir)
    return row


def main():
    ap = argparse.ArgumentParser(description="Run BC tuning trials in parallel and aggregate ranking.")
    ap.add_argument("--trajs-pkl", required=True)
    ap.add_argument("--out-root", default="runs/bc_init_tuning_parallel")
    ap.add_argument("--lrs", default="5e-4,2e-4")
    ap.add_argument("--batch-sizes", default="256,512")
    ap.add_argument("--epochs-list", default="40,80,120")
    ap.add_argument("--archs", default="512x512,512x512x256")
    ap.add_argument("--activation", default="tanh", choices=["tanh", "relu"])
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lrs = _parse_csv_floats(args.lrs)
    bss = _parse_csv_ints(args.batch_sizes)
    eps = _parse_csv_ints(args.epochs_list)
    archs = _parse_archs(args.archs)

    grid = []
    idx = 0
    for lr in lrs:
        for bs in bss:
            for ep in eps:
                for arch in archs:
                    grid.append((idx, lr, bs, ep, arch))
                    idx += 1

    jobs = max(1, min(int(args.jobs), len(grid)))
    print(f"[bc_parallel] trials={len(grid)} jobs={jobs}")

    rows = []
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = []
        for idx, lr, bs, ep, arch in grid:
            futs.append(
                ex.submit(
                    _run_trial,
                    idx,
                    str(out_root),
                    args.trajs_pkl,
                    lr,
                    bs,
                    ep,
                    arch,
                    args.activation,
                    args.eval_episodes,
                    args.max_steps,
                )
            )
        for fut in cf.as_completed(futs):
            row = fut.result()
            rows.append(row)
            if row.get("ok"):
                print(
                    f"[bc_parallel] done trial={row['trial_id']} "
                    f"ret={row['mean_return']:.3f} lr={row['lr']} bs={row['batch_size']} "
                    f"ep={row['epochs']} arch={row['arch']}"
                )
            else:
                print(
                    f"[bc_parallel] FAIL trial={row['trial']} rc={row['returncode']} "
                    f"lr={row['lr']} bs={row['batch_size']} ep={row['epochs']} arch={row['arch']}"
                )

    ok_rows = [r for r in rows if r.get("ok")]
    ok_rows.sort(key=lambda r: float(r["mean_return"]), reverse=True)
    (out_root / "bc_tuning_results.json").write_text(json.dumps(ok_rows, indent=2), encoding="utf-8")
    (out_root / "bc_tuning_all_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not ok_rows:
        raise RuntimeError("No successful trials.")

    best = ok_rows[0]
    (out_root / "bc_tuning_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    src = Path(best["trial_dir"]) / "best_policy_state_dict.pt"
    dst = out_root / "best_policy_state_dict.pt"
    if src.exists():
        shutil.copy2(src, dst)

    print(f"[bc_parallel] best: {best}")
    print(f"[bc_parallel] wrote: {out_root / 'bc_tuning_results.json'}")


if __name__ == "__main__":
    main()
