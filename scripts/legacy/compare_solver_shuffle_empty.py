#!/usr/bin/env python3
import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path("/home/khaledwahba94/imitation-agile-payload-manipulation")
import sys
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

from analyze_pcdbcbs_runs import analyze_run_dir  # noqa: E402


DEFAULT_INPUT = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
DEFAULT_PC_CFG = REPO / "deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml"
DEFAULT_BIN = REPO / "deps/pc-dbCBS/build/pc_dbcbs_expert_dbg"
DEFAULT_DYNOBENCH_BASE = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench"
DEFAULT_MOTION_BASE = Path("/home/khaledwahba94/pc-dbCBS/motion_primitives")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return d


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


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
        "--warmstart_optimization", "1",
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


def load_opt_yaml_metrics(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "result_dbcbs_opt.yaml"
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text()) or {}
    if not isinstance(d, dict):
        return {}
    out = {}
    for k in ["goal_distance", "max_collision", "x_bound_distance", "u_bound_distance", "cost", "feasible"]:
        if k in d:
            try:
                out[k] = float(d[k])
            except Exception:
                pass
    return out


def score(report: Dict[str, Any], opt_metrics: Dict[str, float]) -> Dict[str, float]:
    opt = report.get("optimized_joint", {})
    feasible = 1.0 if bool(report.get("api_result_flags", {}).get("feasible", False)) else 0.0
    if not (opt.get("present") and "payload_path_len" in opt and opt.get("nx") == 39):
        return {"feasible": feasible, "score": 1e9}
    path_excess = max(0.0, float(opt["payload_path_len"]) - 1.5)
    payload_speed_mean = float(opt["payload_speed"]["mean"])
    u_sat_frac = float(opt.get("u_sat_frac_ge_1p35", 0.0))
    cables = opt.get("cables", []) or []
    cable_slack_mean = float(np.mean([c.get("slack_frac_gt_1cm", 0.0) for c in cables])) if cables else 0.0
    cable_stretch_mean = float(np.mean([c.get("stretch_frac_gt_1cm", 0.0) for c in cables])) if cables else 0.0
    goal_distance = float(opt_metrics.get("goal_distance", 1e3))
    score_val = (
        4.0 * path_excess +
        0.6 * payload_speed_mean +
        2.0 * u_sat_frac +
        1.5 * cable_slack_mean +
        5.0 * cable_stretch_mean +
        40.0 * goal_distance +
        (0.0 if feasible else 1000.0)
    )
    return {
        "feasible": feasible,
        "score": float(score_val),
        "path_excess": float(path_excess),
        "payload_path_len": float(opt["payload_path_len"]),
        "payload_speed_mean": float(payload_speed_mean),
        "u_sat_frac": float(u_sat_frac),
        "cable_slack_mean": float(cable_slack_mean),
        "cable_stretch_mean": float(cable_stretch_mean),
        "goal_distance": float(goal_distance),
        "duration_sec": float(opt.get("duration_sec_est", 0.0)),
    }


def extract_xy_series(run_dir: Path) -> Dict[str, np.ndarray]:
    d = yaml.safe_load((run_dir / "result_dbcbs_opt.yaml").read_text())
    X = np.asarray(d["states"], dtype=float)
    n_bodies = 3
    qpos = X[:, :7 * n_bodies]
    p0 = qpos[:, 0:3]
    p1 = qpos[:, 7:10]
    p2 = qpos[:, 14:17]
    return {"p0": p0, "p1": p1, "p2": p2}


def make_side_by_side_gif(run_a: Path, run_b: Path, label_a: str, label_b: str, out_path: Path, fps: int = 15) -> None:
    A = extract_xy_series(run_a)
    B = extract_xy_series(run_b)
    T = max(len(A["p0"]), len(B["p0"]))
    xmin = min(np.min(A["p0"][:, 0]), np.min(B["p0"][:, 0]), -1.6) - 0.1
    xmax = max(np.max(A["p0"][:, 0]), np.max(B["p0"][:, 0]), 0.1) + 0.1
    ymin = min(np.min(A["p0"][:, 1]), np.min(B["p0"][:, 1]), np.min(A["p1"][:, 1]), np.min(B["p1"][:, 1]), np.min(A["p2"][:, 1]), np.min(B["p2"][:, 1])) - 0.2
    ymax = max(np.max(A["p0"][:, 1]), np.max(B["p0"][:, 1]), np.max(A["p1"][:, 1]), np.max(B["p1"][:, 1]), np.max(A["p2"][:, 1]), np.max(B["p2"][:, 1])) + 0.2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    artists = []
    panels = [(axes[0], A, label_a), (axes[1], B, label_b)]
    for ax, data, title in panels:
        ax.set_title(title)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.scatter([-1.5], [0.0], c="k", marker="o", s=20)
        ax.scatter([0.0], [0.0], c="k", marker="x", s=30)
        (trail_p0,) = ax.plot([], [], color="tab:blue", lw=1.5)
        (trail_p1,) = ax.plot([], [], color="tab:orange", lw=1.0, alpha=0.8)
        (trail_p2,) = ax.plot([], [], color="tab:green", lw=1.0, alpha=0.8)
        payload_pt = ax.scatter([], [], c="tab:blue", s=30)
        q1_pt = ax.scatter([], [], c="tab:orange", s=25)
        q2_pt = ax.scatter([], [], c="tab:green", s=25)
        (cable1,) = ax.plot([], [], color="gray", lw=1)
        (cable2,) = ax.plot([], [], color="gray", lw=1)
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
        artists.append((trail_p0, trail_p1, trail_p2, payload_pt, q1_pt, q2_pt, cable1, cable2, txt, data))

    def init():
        return []

    def update(frame):
        out = []
        for tpl in artists:
            trail_p0, trail_p1, trail_p2, payload_pt, q1_pt, q2_pt, cable1, cable2, txt, data = tpl
            k = min(frame, len(data["p0"]) - 1)
            p0 = data["p0"]; p1 = data["p1"]; p2 = data["p2"]
            trail_p0.set_data(p0[:k + 1, 0], p0[:k + 1, 1])
            trail_p1.set_data(p1[:k + 1, 0], p1[:k + 1, 1])
            trail_p2.set_data(p2[:k + 1, 0], p2[:k + 1, 1])
            payload_pt.set_offsets([p0[k, 0], p0[k, 1]])
            q1_pt.set_offsets([p1[k, 0], p1[k, 1]])
            q2_pt.set_offsets([p2[k, 0], p2[k, 1]])
            cable1.set_data([p0[k, 0], p1[k, 0]], [p0[k, 1], p1[k, 1]])
            cable2.set_data([p0[k, 0], p2[k, 0]], [p0[k, 1], p2[k, 1]])
            txt.set_text(f"t={k*0.02:.2f}s")
            out.extend([trail_p0, trail_p1, trail_p2, payload_pt, q1_pt, q2_pt, cable1, cable2, txt])
        return out

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare solver_id 0 vs 1 (and shuffle true/false) on empty payload env.")
    ap.add_argument("--out-root", type=Path, default=REPO / "runs/compare_solver_shuffle_empty")
    ap.add_argument("--input-yaml", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--pc-cfg", type=Path, default=DEFAULT_PC_CFG)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--dynobench-base", type=Path, default=DEFAULT_DYNOBENCH_BASE)
    ap.add_argument("--motion-base", type=Path, default=DEFAULT_MOTION_BASE)
    ap.add_argument("--time-limit-ms", type=int, default=20000)
    ap.add_argument("--n-opt", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    cfg_dir = args.out_root / "configs"
    run_dir = args.out_root / "runs"
    plot_dir = args.out_root / "plots"
    gif_dir = args.out_root / "gifs"
    for p in [cfg_dir, run_dir, plot_dir, gif_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Build pc-dbcbs configs (shuffle true/false)
    pc_base = load_yaml(args.pc_cfg)
    pc_shuf_true = copy.deepcopy(pc_base)
    pc_shuf_false = copy.deepcopy(pc_base)
    pc_shuf_true["pc-dbcbs"]["default"]["shuffle"] = True
    pc_shuf_false["pc-dbcbs"]["default"]["shuffle"] = False
    pc_true_path = cfg_dir / "pc_dbcbs_shuffle_true.yaml"
    pc_false_path = cfg_dir / "pc_dbcbs_shuffle_false.yaml"
    save_yaml(pc_true_path, pc_shuf_true)
    save_yaml(pc_false_path, pc_shuf_false)

    # Build opt configs for solver0 and solver1 (nonzero time term)
    solver_common = {
        "time_weight": 0.1,
        "time_ref": 0.6,
        "check_with_finite_diff": False,
        "soft_control_bounds": False,
        "CALLBACKS": True,
        "use_finite_diff": False,
        "use_warmstart": True,
        "rollout_warmstart": False,
        "repair_init_guess": False,
        "control_bounds": True,
        "states_reg": True,
        "disturbance": 0.0,
        "num_threads": 1,
        "th_stop": 0.01,
        "init_reg": 1e3,
        "th_acceptnegstep": 0.01,
        "noise_level": 0.0,
        "u_bound_scale": 1,
        "max_iter": 220,
        "debug_file_name": "/tmp/debug_file.yaml",
        "weight_goal": 250.0,
        "collision_weight": 120.0,
        "smooth_traj": False,
        "shift_repeat": False,
        "reg_control": True,
        "control_reg_weight": 0.2,
        "penalty_iterations": 1,
    }
    opt0 = dict(solver_common, solver_id=0)
    opt1 = dict(solver_common, solver_id=1)
    opt0_path = cfg_dir / "opt_solver0.yaml"
    opt1_path = cfg_dir / "opt_solver1_tw0p1_tref0p6.yaml"
    save_yaml(opt0_path, opt0)
    save_yaml(opt1_path, opt1)

    conditions = [
        {"name": "s0_shufT", "solver_id": 0, "shuffle": True, "pc_cfg": pc_true_path, "opt_cfg": opt0_path},
        {"name": "s1_shufT", "solver_id": 1, "shuffle": True, "pc_cfg": pc_true_path, "opt_cfg": opt1_path},
        {"name": "s0_shufF", "solver_id": 0, "shuffle": False, "pc_cfg": pc_false_path, "opt_cfg": opt0_path},
        {"name": "s1_shufF", "solver_id": 1, "shuffle": False, "pc_cfg": pc_false_path, "opt_cfg": opt1_path},
    ]

    records: List[Dict[str, Any]] = []
    total = len(conditions) * args.repeats
    kk = 0
    for cond in conditions:
        for rep in range(args.repeats):
            kk += 1
            rdir = run_dir / cond["name"] / f"rep{rep:02d}"
            print(f"[{kk}/{total}] {cond['name']} rep={rep}", flush=True)
            if not (args.reuse_existing and (rdir / "result_api.yaml").exists()):
                if rdir.exists():
                    shutil.rmtree(rdir)
                rc, wall = run_expert(
                    args.bin,
                    args.input_yaml,
                    cond["pc_cfg"],
                    cond["opt_cfg"],
                    rdir,
                    args.dynobench_base,
                    args.motion_base,
                    args.time_limit_ms,
                    args.n_opt,
                )
            else:
                rc, wall = 0, 0.0
            rec: Dict[str, Any] = {
                "condition": cond["name"],
                "solver_id": cond["solver_id"],
                "shuffle": cond["shuffle"],
                "repeat": rep,
                "run_dir": str(rdir),
                "returncode": rc,
                "wall_sec": wall,
            }
            try:
                rep_json = analyze_run_dir(rdir)
                optm = load_opt_yaml_metrics(rdir)
                rec["analysis"] = rep_json
                rec["opt_metrics"] = optm
                rec["metrics"] = score(rep_json, optm)
                m = rec["metrics"]
                print(f"    feas={m['feasible']:.0f} score={m['score']:.3f} path_excess={m.get('path_excess',np.nan):.3f} u_sat={m.get('u_sat_frac',np.nan):.3f}", flush=True)
            except Exception as e:
                rec["error"] = str(e)
                rec["metrics"] = {"feasible": 0.0, "score": 1e9}
                print(f"    analyze error: {e}", flush=True)
            records.append(rec)

    # Aggregate
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["condition"], []).append(r)

    def vals(name: str, key: str) -> List[float]:
        out = []
        for r in groups[name]:
            v = r.get("metrics", {}).get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    aggregate: List[Dict[str, Any]] = []
    for cond in conditions:
        name = cond["name"]
        score_v = np.array(vals(name, "score"), float)
        feasible_v = np.array(vals(name, "feasible"), float)
        path_excess_v = np.array(vals(name, "path_excess"), float) if vals(name, "path_excess") else np.array([])
        u_sat_v = np.array(vals(name, "u_sat_frac"), float) if vals(name, "u_sat_frac") else np.array([])
        goal_v = np.array(vals(name, "goal_distance"), float) if vals(name, "goal_distance") else np.array([])
        aggregate.append({
            "condition": name,
            "solver_id": cond["solver_id"],
            "shuffle": cond["shuffle"],
            "n": len(groups.get(name, [])),
            "feasible_rate": float(np.mean(feasible_v)) if len(feasible_v) else 0.0,
            "score_median": float(np.median(score_v)) if len(score_v) else 1e9,
            "score_mean": float(np.mean(score_v)) if len(score_v) else 1e9,
            "path_excess_mean": float(np.mean(path_excess_v)) if len(path_excess_v) else np.nan,
            "u_sat_mean": float(np.mean(u_sat_v)) if len(u_sat_v) else np.nan,
            "goal_distance_mean": float(np.mean(goal_v)) if len(goal_v) else np.nan,
        })

    # Save raw
    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "aggregate": aggregate,
    }
    (args.out_root / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    labels = [c["name"] for c in conditions]
    pretty = {
        "s0_shufT": "solver0, shuffle=T",
        "s1_shufT": "solver1, shuffle=T",
        "s0_shufF": "solver0, shuffle=F",
        "s1_shufF": "solver1, shuffle=F",
    }
    xs_labels = [pretty[l] for l in labels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, title in [
        (axes[0, 0], "score", "Score (lower better)"),
        (axes[0, 1], "path_excess", "Payload Path Excess [m]"),
        (axes[1, 0], "u_sat_frac", "Control Saturation Fraction"),
        (axes[1, 1], "goal_distance", "Final Goal Distance"),
    ]:
        data = [vals(lbl, key) for lbl in labels]
        ax.boxplot(data, labels=xs_labels, showfliers=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "boxplots_metrics.png", dpi=180)
    plt.close(fig)

    # Feasible rate bars
    fig, ax = plt.subplots(figsize=(8, 4))
    rates = [next(a["feasible_rate"] for a in aggregate if a["condition"] == lbl) for lbl in labels]
    ax.bar(xs_labels, rates, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Feasible Rate")
    ax.set_title("Feasibility by Solver / Shuffle")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "feasible_rate_bars.png", dpi=180)
    plt.close(fig)

    # Scatter path_excess vs u_sat
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = {"s0_shufT": "tab:blue", "s1_shufT": "tab:orange", "s0_shufF": "tab:green", "s1_shufF": "tab:red"}
    for lbl in labels:
        x = vals(lbl, "path_excess")
        y = vals(lbl, "u_sat_frac")
        ax.scatter(x, y, label=pretty[lbl], color=cmap[lbl], s=45, alpha=0.8)
    ax.set_xlabel("Payload Path Excess [m]")
    ax.set_ylabel("Control Saturation Fraction")
    ax.set_title("Trajectory Quality Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_path_vs_usat.png", dpi=180)
    plt.close(fig)

    # Representative GIFs: best feasible solver0 vs solver1 under shuffle=false
    def best_run(cond_name: str) -> Dict[str, Any]:
        rows = [r for r in groups.get(cond_name, []) if r.get("metrics", {}).get("feasible", 0.0) > 0.5]
        if not rows:
            rows = groups.get(cond_name, [])
        return min(rows, key=lambda r: r.get("metrics", {}).get("score", 1e9))

    best_s0f = best_run("s0_shufF")
    best_s1f = best_run("s1_shufF")
    make_side_by_side_gif(
        Path(best_s0f["run_dir"]),
        Path(best_s1f["run_dir"]),
        "solver0, shuffle=F (best)",
        "solver1, shuffle=F (best)",
        gif_dir / "solver0_vs_solver1_shuffle_false_best.gif",
    )
    make_side_by_side_gif(
        Path(best_run("s0_shufT")["run_dir"]),
        Path(best_run("s1_shufT")["run_dir"]),
        "solver0, shuffle=T (best)",
        "solver1, shuffle=T (best)",
        gif_dir / "solver0_vs_solver1_shuffle_true_best.gif",
    )

    print("\nAggregate:")
    print(json.dumps(aggregate, indent=2))
    print(f"Plots: {plot_dir}")
    print(f"GIFs: {gif_dir}")


if __name__ == "__main__":
    main()

