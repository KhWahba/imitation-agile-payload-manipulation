#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _arch_label(arch) -> str:
    if isinstance(arch, (list, tuple)):
        return "x".join(str(int(x)) for x in arch)
    return str(arch)


def main():
    ap = argparse.ArgumentParser(description="Plot BC tuning results from bc_tuning_results.json.")
    ap.add_argument(
        "--results-json",
        type=str,
        required=True,
        help="Path to bc_tuning_results.json produced by scripts/train_bc.py",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Default: sibling folder 'plots' next to the results json",
    )
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    results_path = Path(args.results_json)
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError(f"No rows in {results_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (results_path.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure sorted (best first) for ranking plots.
    rows = sorted(rows, key=lambda r: float(r["mean_return"]), reverse=True)

    # Extract arrays.
    mean_ret = np.array([float(r["mean_return"]) for r in rows], dtype=float)
    std_ret = np.array([float(r.get("std_return", 0.0)) for r in rows], dtype=float)
    lrs = np.array([float(r["lr"]) for r in rows], dtype=float)
    bss = np.array([int(r["batch_size"]) for r in rows], dtype=int)
    eps = np.array([int(r["epochs"]) for r in rows], dtype=int)
    arch = [_arch_label(r["arch"]) for r in rows]

    # 1) Ranking plot (top-k)
    k = min(int(args.top_k), len(rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(k)
    ax.barh(y, mean_ret[:k], xerr=std_ret[:k], color="tab:blue", alpha=0.85)
    labels = [
        f"#{i} lr={rows[i]['lr']} bs={rows[i]['batch_size']} ep={rows[i]['epochs']} arch={_arch_label(rows[i]['arch'])}"
        for i in range(k)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Eval Return (higher is better)")
    ax.set_title(f"BC Tuning Ranking (top {k})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bc_tuning_ranking_topk.png", dpi=150)
    plt.close(fig)

    # 2) Parameter trends panel
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax_lr, ax_bs, ax_ep, ax_arch = axs.ravel()

    ax_lr.scatter(np.log10(lrs), mean_ret, s=35, alpha=0.8)
    ax_lr.set_xlabel("log10(lr)")
    ax_lr.set_ylabel("mean_return")
    ax_lr.set_title("Return vs Learning Rate")
    ax_lr.grid(True, alpha=0.3)

    ax_bs.scatter(bss, mean_ret, s=35, alpha=0.8)
    ax_bs.set_xlabel("batch_size")
    ax_bs.set_ylabel("mean_return")
    ax_bs.set_title("Return vs Batch Size")
    ax_bs.grid(True, alpha=0.3)

    ax_ep.scatter(eps, mean_ret, s=35, alpha=0.8)
    ax_ep.set_xlabel("epochs")
    ax_ep.set_ylabel("mean_return")
    ax_ep.set_title("Return vs Epochs")
    ax_ep.grid(True, alpha=0.3)

    uniq_arch = sorted(set(arch))
    x = np.arange(len(uniq_arch))
    for i, a in enumerate(uniq_arch):
        vals = mean_ret[np.array([aa == a for aa in arch])]
        ax_arch.scatter(np.full(vals.shape, i), vals, s=25, alpha=0.7, color="tab:orange")
        ax_arch.plot([i - 0.2, i + 0.2], [vals.mean(), vals.mean()], color="black", lw=2)
    ax_arch.set_xticks(x)
    ax_arch.set_xticklabels(uniq_arch, rotation=25, ha="right")
    ax_arch.set_ylabel("mean_return")
    ax_arch.set_title("Return vs Architecture")
    ax_arch.grid(True, axis="y", alpha=0.3)

    fig.suptitle("BC Hyperparameter Sensitivity")
    fig.tight_layout()
    fig.savefig(out_dir / "bc_tuning_param_trends.png", dpi=150)
    plt.close(fig)

    # 3) Compact summary text
    best = rows[0]
    summary = [
        "BC TUNING SUMMARY",
        f"trials: {len(rows)}",
        f"best_mean_return: {best['mean_return']:.6f}",
        (
            "best_config: "
            f"lr={best['lr']}, batch_size={best['batch_size']}, epochs={best['epochs']}, "
            f"arch={_arch_label(best['arch'])}, activation={best.get('activation', 'n/a')}"
        ),
    ]
    (out_dir / "bc_tuning_plot_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"[bc_plots] wrote: {out_dir / 'bc_tuning_ranking_topk.png'}")
    print(f"[bc_plots] wrote: {out_dir / 'bc_tuning_param_trends.png'}")
    print(f"[bc_plots] wrote: {out_dir / 'bc_tuning_plot_summary.txt'}")


if __name__ == "__main__":
    main()
