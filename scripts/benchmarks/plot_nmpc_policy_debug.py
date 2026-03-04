#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def _to_array(rows, key, default_len=None):
    vals = []
    for r in rows:
        v = r.get(key, [])
        vals.append(v)
    if default_len is not None:
        arr = np.full((len(rows), default_len), np.nan, dtype=float)
        for i, v in enumerate(vals):
            if not v:
                continue
            vv = np.asarray(v, dtype=float)
            n = min(default_len, vv.size)
            arr[i, :n] = vv[:n]
        return arr
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot NMPC policy debug JSONL")
    ap.add_argument("--debug-jsonl", required=True, help="path to *_policy_debug.jsonl")
    ap.add_argument("--out-dir", required=True, help="output directory for plots")
    args = ap.parse_args()

    path = Path(args.debug_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(path)

    ks = np.asarray([int(r["k"]) for r in rows], dtype=int)
    gd = np.asarray([float(r.get("goal_distance", np.nan)) for r in rows], dtype=float)
    did_solve = np.asarray([bool(r.get("did_solve", False)) for r in rows], dtype=bool)

    u_applied = _to_array(rows, "u_applied")
    nu = len(u_applied[0]) if u_applied and u_applied[0] else 0
    u_applied_arr = _to_array(rows, "u_applied", default_len=nu) if nu > 0 else None
    ws_u0_arr = _to_array(rows, "warm_start_u0", default_len=nu) if nu > 0 else None
    nmpc_u0_arr = _to_array(rows, "nmpc_u0", default_len=nu) if nu > 0 else None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ks, gd, color="black", lw=2, label="goal distance")
    solve_ks = ks[did_solve]
    if solve_ks.size > 0:
        ax.vlines(solve_ks, ymin=np.nanmin(gd), ymax=np.nanmax(gd), color="tab:red", alpha=0.2, label="solve step")
    ax.set_title("NMPC Goal Distance")
    ax.set_xlabel("step")
    ax.set_ylabel("distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "goal_distance.png", dpi=180)
    plt.close(fig)

    if nu > 0:
        fig, axes = plt.subplots(nu, 1, figsize=(10, max(3, 1.7 * nu)), sharex=True)
        if nu == 1:
            axes = [axes]
        for j, ax in enumerate(axes):
            ax.plot(ks, u_applied_arr[:, j], color="tab:blue", lw=1.8, label=f"u_applied[{j}]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("step")
        fig.suptitle("Applied Control")
        fig.tight_layout()
        fig.savefig(out_dir / "applied_actions.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(nu, 1, figsize=(10, max(3, 1.7 * nu)), sharex=True)
        if nu == 1:
            axes = [axes]
        for j, ax in enumerate(axes):
            if ws_u0_arr is not None:
                ax.plot(ks, ws_u0_arr[:, j], color="tab:orange", lw=1.2, label=f"warm_start_u0[{j}]")
            if nmpc_u0_arr is not None:
                ax.plot(ks, nmpc_u0_arr[:, j], color="tab:green", lw=1.2, label=f"nmpc_u0[{j}]")
            ax.plot(ks, u_applied_arr[:, j], color="tab:blue", lw=1.8, alpha=0.75, label=f"u_applied[{j}]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("step")
        fig.suptitle("Policy/Warm-Start/NMPC First Control Comparison")
        fig.tight_layout()
        fig.savefig(out_dir / "u0_comparison.png", dpi=180)
        plt.close(fig)

    solve_rows = [r for r in rows if r.get("did_solve", False) and "nmpc_actions_window" in r]
    if solve_rows:
        # Window action slicing plot (norm over horizon for each solve step).
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in solve_rows:
            wa = np.asarray(r["warm_start_actions_window"], dtype=float)
            sa = np.asarray(r["nmpc_actions_window"], dtype=float)
            if wa.ndim == 2 and wa.shape[1] > 0:
                ax.plot(np.linalg.norm(wa, axis=1), color="tab:orange", alpha=0.08, lw=1.0)
            if sa.ndim == 2 and sa.shape[1] > 0:
                ax.plot(np.linalg.norm(sa, axis=1), color="tab:green", alpha=0.12, lw=1.0)
        ax.plot([], [], color="tab:orange", label="warm-start action norm (all solve steps)")
        ax.plot([], [], color="tab:green", label="nmpc solved action norm (all solve steps)")
        ax.set_title("Solve-Window Action Slices")
        ax.set_xlabel("horizon index")
        ax.set_ylabel("||u||")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "window_actions_slices.png", dpi=180)
        plt.close(fig)

        # Window payload-state slicing plot (p_x,p_y,p_z and v_z).
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        axes = axes.ravel()
        labels = ["payload p_x", "payload p_y", "payload p_z", "payload v_z"]
        idxs = [0, 1, 2, 5]  # from [px,py,pz,vx,vy,vz]
        for r in solve_rows:
            ws = np.asarray(r.get("warm_start_payload_pv_window", []), dtype=float)
            ns = np.asarray(r.get("nmpc_payload_pv_window", []), dtype=float)
            if ws.ndim == 2 and ws.shape[1] >= 6:
                for a, ii in zip(axes, idxs):
                    a.plot(ws[:, ii], color="tab:orange", alpha=0.08, lw=1.0)
            if ns.ndim == 2 and ns.shape[1] >= 6:
                for a, ii in zip(axes, idxs):
                    a.plot(ns[:, ii], color="tab:green", alpha=0.12, lw=1.0)
        for a, lab in zip(axes, labels):
            a.set_title(lab)
            a.grid(True, alpha=0.3)
        axes[-1].set_xlabel("horizon index")
        axes[-2].set_xlabel("horizon index")
        fig.suptitle("Solve-Window Payload State Slices")
        fig.tight_layout()
        fig.savefig(out_dir / "window_payload_state_slices.png", dpi=180)
        plt.close(fig)

    summary = {
        "debug_jsonl": str(path),
        "steps": int(len(rows)),
        "solve_count": int(did_solve.sum()),
        "solve_ratio": float(did_solve.mean()),
        "goal_distance_final": float(gd[-1]),
        "goal_distance_min": float(np.nanmin(gd)),
        "goal_distance_max": float(np.nanmax(gd)),
        "plots": [
            "goal_distance.png",
            "applied_actions.png" if nu > 0 else None,
            "u0_comparison.png" if nu > 0 else None,
            "window_actions_slices.png" if solve_rows else None,
            "window_payload_state_slices.png" if solve_rows else None,
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[plot_nmpc_policy_debug] wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
