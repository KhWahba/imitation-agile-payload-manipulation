#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path("/home/khaledwahba94/imitation-agile-payload-manipulation")
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

from analyze_pcdbcbs_runs import analyze_run_dir  # noqa: E402


DEFAULT_INPUT = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
DEFAULT_PC_CFG = REPO / "deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml"
DEFAULT_OPT_BASE = REPO / "deps/pc-dbCBS/configs/opt_training_empty_time_sweep.yaml"
DEFAULT_BIN = REPO / "deps/pc-dbCBS/build/pc_dbcbs_expert_dbg"
DEFAULT_DYNOBENCH_BASE = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench"
DEFAULT_MOTION_BASE = Path("/home/khaledwahba94/pc-dbCBS/motion_primitives")


def parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return obj


def run_expert(
    bin_path: Path,
    input_yaml: Path,
    pc_cfg: Path,
    opt_cfg: Path,
    out_dir: Path,
    dynobench_base: Path,
    motion_base: Path,
    time_limit_ms: int,
    n_opt: int,
    warmstart: bool,
) -> Tuple[int, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bin_path),
        "--input_yaml", str(input_yaml),
        "--output_yaml", str(out_dir / "result_dbcbs.yaml"),
        "--optimization_yaml", str(out_dir / "result_dbcbs_opt.yaml"),
        "--pc_dbcbs_cfg_yaml", str(pc_cfg),
        "--opt_cfg_yaml", str(opt_cfg),
        "--time_limit", str(float(time_limit_ms)),
        "--dynobench_base", str(dynobench_base) + "/",
        "--motion_primitives_base", str(motion_base) + "/",
        "--warmstart_optimization", "1" if warmstart else "0",
        "--override_visualize_mujoco", "1",
        "--visualize_mujoco", "0",
        "--N_opt", str(n_opt),
        "--result_yaml", str(out_dir / "result_api.yaml"),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    dt = time.time() - t0
    (out_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    return proc.returncode, dt


def score_run(report: Dict[str, Any]) -> Dict[str, float]:
    opt = report.get("optimized_joint", {})
    initg = report.get("init_guess_joint", {})
    api_flags = report.get("api_result_flags", {})

    feasible = 1.0 if bool(api_flags.get("feasible", False)) else 0.0
    solved_opt = 1.0 if bool(api_flags.get("solved_opt", False)) else 0.0
    present = 1.0 if bool(opt.get("present", False)) else 0.0
    if not (present and opt.get("nx") == 39 and "payload_path_len" in opt):
        return {
            "feasible": feasible,
            "solved_opt": solved_opt,
            "score": 1e9,
            "path_excess": 1e6,
            "payload_speed_mean": 1e6,
            "u_sat_frac": 1.0,
            "cable_slack_mean": 1.0,
            "cable_stretch_mean": 1.0,
        }

    # Empty env expected payload displacement is ~1.5m in x.
    payload_path = float(opt["payload_path_len"])
    path_excess = max(0.0, payload_path - 1.5)
    payload_speed_mean = float(opt["payload_speed"]["mean"])
    u_sat_frac = float(opt.get("u_sat_frac_ge_1p35", 0.0))

    cables = opt.get("cables", []) or []
    slack_fracs = [float(c.get("slack_frac_gt_1cm", 0.0)) for c in cables]
    stretch_fracs = [float(c.get("stretch_frac_gt_1cm", 0.0)) for c in cables]
    cable_slack_mean = float(np.mean(slack_fracs)) if slack_fracs else 0.0
    cable_stretch_mean = float(np.mean(stretch_fracs)) if stretch_fracs else 0.0

    # Penalize optimizer making controls noisier than init.
    init_u = initg.get("u_norm", {})
    init_u_mean = float(init_u.get("mean", payload_speed_mean))
    opt_u_mean = float(opt["u_norm"]["mean"])
    control_norm_increase = max(0.0, opt_u_mean - init_u_mean)

    # Lower is better. Strongly penalize non-feasible outcomes.
    base = (
        6.0 * path_excess +
        0.8 * payload_speed_mean +
        4.0 * u_sat_frac +
        2.5 * cable_slack_mean +
        6.0 * cable_stretch_mean +
        0.5 * control_norm_increase
    )
    score = base + (0.0 if feasible > 0.5 else 1000.0)

    return {
        "feasible": feasible,
        "solved_opt": solved_opt,
        "score": float(score),
        "path_excess": float(path_excess),
        "payload_speed_mean": float(payload_speed_mean),
        "u_sat_frac": float(u_sat_frac),
        "cable_slack_mean": float(cable_slack_mean),
        "cable_stretch_mean": float(cable_stretch_mean),
    }


def make_heatmap(
    values: np.ndarray,
    xlabels: List[str],
    ylabels: List[str],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    annotate: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(0.9 * len(xlabels) + 2, 0.7 * len(ylabels) + 2))
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("time_weight")
    ax.set_ylabel("time_ref")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                txt = "nan" if np.isnan(v) else f"{v:.3g}"
                ax.text(j, i, txt, ha="center", va="center", color="white" if not np.isnan(v) else "black", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep time_weight/time_ref for empty mujoco payload expert runs.")
    ap.add_argument("--out-root", type=Path, default=REPO / "runs/sweeps/time_param_sweep")
    ap.add_argument("--input-yaml", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--pc-cfg", type=Path, default=DEFAULT_PC_CFG)
    ap.add_argument("--opt-base-cfg", type=Path, default=DEFAULT_OPT_BASE)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--dynobench-base", type=Path, default=DEFAULT_DYNOBENCH_BASE)
    ap.add_argument("--motion-base", type=Path, default=DEFAULT_MOTION_BASE)
    ap.add_argument("--time-limit-ms", type=int, default=18000)
    ap.add_argument("--n-opt", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--time-weights", type=str, default="0.0,0.05,0.1,0.2,0.4")
    ap.add_argument("--time-refs", type=str, default="0.5,0.8,1.0,1.2")
    ap.add_argument("--keep-run-dirs", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true", help="If run dir already contains result_api.yaml, skip rerun and only re-analyze.")
    args = ap.parse_args()

    time_weights = parse_csv_floats(args.time_weights)
    time_refs = parse_csv_floats(args.time_refs)

    base_opt = load_yaml(args.opt_base_cfg)
    args.out_root.mkdir(parents=True, exist_ok=True)
    configs_dir = args.out_root / "configs"
    runs_dir = args.out_root / "runs"
    plots_dir = args.out_root / "plots"
    configs_dir.mkdir(exist_ok=True, parents=True)
    runs_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)

    records: List[Dict[str, Any]] = []
    total = len(time_weights) * len(time_refs) * args.repeats
    idx = 0

    for tref in time_refs:
        for tw in time_weights:
            cfg = copy.deepcopy(base_opt)
            cfg["solver_id"] = 1
            cfg["time_weight"] = float(tw)
            cfg["time_ref"] = float(tref)
            cfg_name = f"opt_tref_{tref:.2f}_tw_{tw:.2f}.yaml".replace("-", "m")
            cfg_path = configs_dir / cfg_name
            write_yaml(cfg_path, cfg)

            for rep in range(args.repeats):
                idx += 1
                run_dir = runs_dir / f"tref_{tref:.2f}_tw_{tw:.2f}_rep{rep:02d}".replace("-", "m")
                if run_dir.exists() and not args.reuse_existing:
                    shutil.rmtree(run_dir)
                print(f"[{idx}/{total}] run tw={tw} tref={tref} rep={rep} -> {run_dir.name}", flush=True)

                tstart = time.time()
                if args.reuse_existing and (run_dir / "result_api.yaml").exists():
                    rc, wall = 0, 0.0
                    print("    reusing existing artifacts", flush=True)
                else:
                    rc, wall = run_expert(
                        args.bin,
                        args.input_yaml,
                        args.pc_cfg,
                        cfg_path,
                        run_dir,
                        args.dynobench_base,
                        args.motion_base,
                        args.time_limit_ms,
                        args.n_opt,
                        True,
                    )

                rec: Dict[str, Any] = {
                    "time_weight": tw,
                    "time_ref": tref,
                    "repeat": rep,
                    "returncode": rc,
                    "wall_sec": wall,
                    "run_dir": str(run_dir),
                }
                try:
                    report = analyze_run_dir(run_dir)
                    rec["analysis"] = report
                    rec["metrics"] = score_run(report)
                except Exception as e:
                    rec["analysis_error"] = str(e)
                    rec["metrics"] = {
                        "feasible": 0.0,
                        "solved_opt": 0.0,
                        "score": 1e9,
                        "path_excess": 1e6,
                        "payload_speed_mean": 1e6,
                        "u_sat_frac": 1.0,
                        "cable_slack_mean": 1.0,
                        "cable_stretch_mean": 1.0,
                    }

                m = rec["metrics"]
                print(
                    f"    rc={rc} wall={wall:.1f}s feasible={m['feasible']:.0f} "
                    f"score={m['score']:.3f} path_excess={m['path_excess']:.3f} "
                    f"u_sat={m['u_sat_frac']:.3f}",
                    flush=True,
                )
                records.append(rec)

                if not args.keep_run_dirs and rc == 0:
                    # keep logs and key artifacts for best configs only later; for now keep all to inspect
                    pass

    # Aggregate
    combos = {(r["time_ref"], r["time_weight"]) for r in records}
    agg_rows: List[Dict[str, Any]] = []
    for tref, tw in sorted(combos):
        rows = [r for r in records if r["time_ref"] == tref and r["time_weight"] == tw]
        scores = np.array([r["metrics"]["score"] for r in rows], dtype=float)
        feas = np.array([r["metrics"]["feasible"] for r in rows], dtype=float)
        wall = np.array([r["wall_sec"] for r in rows], dtype=float)
        payload_speed = np.array([r["metrics"]["payload_speed_mean"] for r in rows], dtype=float)
        path_excess = np.array([r["metrics"]["path_excess"] for r in rows], dtype=float)
        u_sat = np.array([r["metrics"]["u_sat_frac"] for r in rows], dtype=float)
        agg_rows.append({
            "time_ref": tref,
            "time_weight": tw,
            "n": len(rows),
            "feasible_rate": float(np.mean(feas)),
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_best": float(np.min(scores)),
            "wall_sec_mean": float(np.mean(wall)),
            "payload_speed_mean": float(np.mean(payload_speed)),
            "path_excess_mean": float(np.mean(path_excess)),
            "u_sat_mean": float(np.mean(u_sat)),
        })

    # Pick best by feasible_rate desc, then score_median asc
    agg_rows_sorted = sorted(agg_rows, key=lambda r: (-r["feasible_rate"], r["score_median"], r["score_mean"]))
    best = agg_rows_sorted[0] if agg_rows_sorted else None

    # Write CSV/JSON
    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "best_combo": best,
        "aggregate": agg_rows_sorted,
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_root / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    with (args.out_root / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows_sorted[0].keys()) if agg_rows_sorted else [])
        if agg_rows_sorted:
            writer.writeheader()
            writer.writerows(agg_rows_sorted)

    # Plot heatmaps
    yvals = time_refs
    xvals = time_weights
    def grid_of(key: str) -> np.ndarray:
        M = np.full((len(yvals), len(xvals)), np.nan, dtype=float)
        for i, tref in enumerate(yvals):
            for j, tw in enumerate(xvals):
                row = next((r for r in agg_rows if r["time_ref"] == tref and r["time_weight"] == tw), None)
                if row:
                    M[i, j] = float(row[key])
        return M

    xlabels = [f"{x:g}" for x in xvals]
    ylabels = [f"{y:g}" for y in yvals]
    make_heatmap(grid_of("feasible_rate"), xlabels, ylabels, "Feasible Rate", plots_dir / "feasible_rate.png", cmap="viridis")
    make_heatmap(grid_of("score_median"), xlabels, ylabels, "Median Score (lower better)", plots_dir / "score_median.png", cmap="magma")
    make_heatmap(grid_of("path_excess_mean"), xlabels, ylabels, "Mean Payload Path Excess [m]", plots_dir / "path_excess_mean.png", cmap="plasma")
    make_heatmap(grid_of("u_sat_mean"), xlabels, ylabels, "Mean Control Saturation Fraction", plots_dir / "u_sat_mean.png", cmap="cividis")

    # Best-vs-baseline scatter (score by repeat)
    if best is not None:
        best_rows = [r for r in records if r["time_ref"] == best["time_ref"] and r["time_weight"] == best["time_weight"]]
        plt.figure(figsize=(6, 4))
        plt.scatter(
            [r["metrics"]["path_excess"] for r in best_rows],
            [r["metrics"]["u_sat_frac"] for r in best_rows],
            c=[r["metrics"]["score"] for r in best_rows],
            cmap="viridis",
            s=70,
        )
        plt.xlabel("Payload path excess [m]")
        plt.ylabel("Control saturation fraction")
        plt.title(f"Best combo repeats: tw={best['time_weight']}, tref={best['time_ref']}")
        plt.colorbar(label="Score")
        plt.tight_layout()
        plt.savefig(plots_dir / "best_combo_repeats.png", dpi=160)
        plt.close()

    print("\nBest combo:")
    print(json.dumps(best, indent=2))
    print(f"Plots saved under: {plots_dir}")


if __name__ == "__main__":
    main()
