import argparse
import concurrent.futures as cf
import json
import os
import shutil
import time
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from payload_env import PayloadGymEnv
from expert_pcdbcbs_subprocess_dbg import PcDbCBSExpert, PcDbCBSPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
TEMPLATE_YAML = REPO_ROOT / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
PC_DBCBS_CFG = REPO_ROOT / "deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml"
BASE_OPT_CFG = REPO_ROOT / "deps/pc-dbCBS/configs/opt_training.yaml"
DYNOBENCH_BASE = REPO_ROOT / "deps/pc-dbCBS/deps/dynoplan/dynobench"
BUILD_DIR = REPO_ROOT / "deps/pc-dbCBS/build"
MOTION_PRIMS_BASE = Path("/home/khaledwahba94/pc-dbCBS/motion_primitives/")


def _split_joint_state_row(raw_state: np.ndarray):
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n_bodies = raw.size // 13
    poses = raw[: 7 * n_bodies]
    vels = raw[7 * n_bodies :]
    payload_pos = poses[:3]
    payload_vel = vels[:3]
    quad_positions = []
    quad_vels = []
    for qi in range(n_bodies - 1):
        pbase = 7 * (1 + qi)
        vbase = 6 * (1 + qi)
        quad_positions.append(poses[pbase : pbase + 3])
        quad_vels.append(vels[vbase : vbase + 3])
    return payload_pos, payload_vel, quad_positions, quad_vels


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _iter_plan_dirs(tmp_dir: Path):
    for d in sorted(tmp_dir.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "meta.yaml"
        res = d / "result_subprocess_api.yaml"
        if meta.exists() and res.exists():
            yield d, meta, res


def _parse_plans(tmp_dir: Path):
    rows = []
    for d, meta_p, res_p in _iter_plan_dirs(tmp_dir):
        meta = _load_yaml(meta_p) or {}
        res = _load_yaml(res_p) or {}
        X = np.asarray(res.get("X", []), dtype=np.float32)
        U = np.asarray(res.get("U", []), dtype=np.float32)
        if X.ndim != 2 or X.size == 0:
            continue
        rows.append(
            {
                "dir": str(d),
                "global_step": int(meta.get("global_step", -1)),
                "reason": str(meta.get("reason", "")),
                "dist_to_goal": float(meta.get("dist_to_goal", np.nan)),
                "X": X,
                "U": U if U.ndim == 2 else np.zeros((0, 0), dtype=np.float32),
            }
        )
    rows.sort(key=lambda r: (r["global_step"], r["dir"]))
    return rows


def _rollout_eval(config_row: dict, out_dir_str: str, steps: int, replan_k: int, max_steps: int,
                  deterministic: bool, time_limit_ms: float, n_opt: int, stream_logs: bool):
    out_dir = Path(out_dir_str)
    cfg_id = config_row["id"]
    run_dir = out_dir / f"cfg_{cfg_id:03d}"
    tmp_dir = run_dir / "tmp_pcdbcbs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_cfg = _load_yaml(BASE_OPT_CFG)
        cfg = dict(base_cfg)
        cfg.update(config_row.get("overrides", {}))
        opt_cfg_path = run_dir / "opt_training_tuned.yaml"
        _write_yaml(opt_cfg_path, cfg)
        pc_cfg = _load_yaml(PC_DBCBS_CFG)
        pc_overrides = config_row.get("pc_overrides", {})
        if pc_overrides:
            _deep_update(pc_cfg, pc_overrides)
        pc_cfg_path = run_dir / "pc_dbcbs_tuned.yaml"
        _write_yaml(pc_cfg_path, pc_cfg)

        os.environ["PCDBCBS_EXPERT_SUBPROCESS_BIN"] = str(BUILD_DIR / "pc_dbcbs_expert_dbg")
        os.environ["PCDBCBS_EXPERT_SUBPROCESS_STREAM_LOGS"] = "1" if stream_logs else "0"
        os.environ["PCDBCBS_DETERMINISTIC"] = "1" if deterministic else "0"
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        env = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=max_steps)
        env.terminate_on_success = True
        nu = env.action_dim
        expert = PcDbCBSExpert(
            paths=PcDbCBSPaths(
                bindings_path=str(BUILD_DIR),
                input_yaml=str(TEMPLATE_YAML),
                pc_dbcbs_cfg_yaml=str(pc_cfg_path),
                opt_cfg_yaml=str(opt_cfg_path),
                dynobench_base=str(DYNOBENCH_BASE) + "/",
                motion_primitives_base=str(MOTION_PRIMS_BASE) + "/",
                time_limit=time_limit_ms,
                work_dir_root=str(tmp_dir),
                keep_files=True,
                warmstart_optimization=True,
                N_opt=n_opt,
            ),
            act_low=np.zeros(nu, dtype=np.float32),
            act_high=np.ones(nu, dtype=np.float32) * 1.4,
            replan_every_k=replan_k,
        )

        t0 = time.perf_counter()
        obs, _ = env.reset(seed=0)
        expert.reset_episode()
        rollout = []
        term_info = {"terminated": False, "truncated": False, "payload_oob": False, "quad_oob": False, "success": False}
        for t in range(steps):
            act = expert.act(obs)
            next_obs, rew, terminated, truncated, info = env.step(act)
            raw_state = next_obs[: env.state_dim]
            payload_pos, payload_vel, quad_pos, quad_vels = _split_joint_state_row(raw_state)
            rollout.append(
                {
                    "t": t,
                    "rew": float(rew),
                    "just_replanned": bool(expert.just_replanned),
                    "payload_pos": payload_pos.tolist(),
                    "payload_vel": payload_vel.tolist(),
                    "quad_vels": [q.tolist() for q in quad_vels],
                    "payload_goal_dist": float(np.linalg.norm(payload_pos - env.goal[:3])),
                }
            )
            obs = next_obs
            if terminated or truncated:
                term_info = {
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "payload_oob": bool((info or {}).get("payload_out_of_bounds", False)),
                    "quad_oob": bool((info or {}).get("quad_out_of_bounds", False)),
                    "success": bool((info or {}).get("is_success", False)),
                }
                break
        dur_s = time.perf_counter() - t0
        plans = _parse_plans(tmp_dir)
    except Exception as e:
        metrics = {
            "cfg_id": cfg_id,
            "name": config_row["name"],
            "overrides": config_row["overrides"],
            "pc_overrides": config_row.get("pc_overrides", {}),
            "error": str(e),
            "score": 1e9,
            "rollout_steps": 0,
            "replans": 0,
            "runtime_s": 0.0,
            "final_goal_dist": float("inf"),
            "payload_vel_rms": float("nan"),
            "payload_vel_axis_abs_mean": [float("nan")] * 3,
            "payload_vel_axis_abs_max": [float("nan")] * 3,
            "quad_vel_rms_mean": float("nan"),
            "plan_du_mean": float("nan"),
            "plan_du_max": float("nan"),
            "plan_payload_vel_mean": float("nan"),
            "plan_payload_vel_max": float("nan"),
            "terminated": False,
            "truncated": False,
            "payload_oob": False,
            "quad_oob": False,
            "success": False,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
    payload_vel_arr = np.array([r["payload_vel"] for r in rollout], dtype=np.float32) if rollout else np.zeros((0, 3), np.float32)
    quad_vel_arr = []
    if rollout and rollout[0]["quad_vels"]:
        n_q = len(rollout[0]["quad_vels"])
        for qi in range(n_q):
            quad_vel_arr.append(np.array([r["quad_vels"][qi] for r in rollout], dtype=np.float32))

    # Smoothness / quality metrics from optimized plans
    plan_du_norms = []
    plan_payload_vel_norms = []
    for p in plans:
        U = p["U"]
        X = p["X"]
        if U.shape[0] >= 2:
            du = np.diff(U, axis=0)
            plan_du_norms.extend(np.linalg.norm(du, axis=1).tolist())
        # payload vel in X (poses-first -> payload v at qvel block start)
        nb = X.shape[1] // 13
        qpos_dim = 7 * nb
        if X.shape[1] >= qpos_dim + 3:
            plan_payload_vel_norms.extend(np.linalg.norm(X[:, qpos_dim:qpos_dim + 3], axis=1).tolist())

    metrics = {
        "cfg_id": cfg_id,
        "name": config_row["name"],
        "overrides": config_row["overrides"],
        "pc_overrides": config_row.get("pc_overrides", {}),
        "rollout_steps": len(rollout),
        "replans": len(plans),
        "runtime_s": dur_s,
        "final_goal_dist": float(rollout[-1]["payload_goal_dist"]) if rollout else np.inf,
        "payload_vel_rms": float(np.sqrt(np.mean(np.square(payload_vel_arr)))) if payload_vel_arr.size else np.nan,
        "payload_vel_axis_abs_mean": (
            np.mean(np.abs(payload_vel_arr), axis=0).tolist() if payload_vel_arr.size else [np.nan, np.nan, np.nan]
        ),
        "payload_vel_axis_abs_max": (
            np.max(np.abs(payload_vel_arr), axis=0).tolist() if payload_vel_arr.size else [np.nan, np.nan, np.nan]
        ),
        "quad_vel_rms_mean": (
            float(np.mean([np.sqrt(np.mean(np.square(qv))) for qv in quad_vel_arr])) if quad_vel_arr else np.nan
        ),
        "plan_du_mean": float(np.mean(plan_du_norms)) if plan_du_norms else np.nan,
        "plan_du_max": float(np.max(plan_du_norms)) if plan_du_norms else np.nan,
        "plan_payload_vel_mean": float(np.mean(plan_payload_vel_norms)) if plan_payload_vel_norms else np.nan,
        "plan_payload_vel_max": float(np.max(plan_payload_vel_norms)) if plan_payload_vel_norms else np.nan,
        **term_info,
    }

    # Lower is better (heuristic ranking)
    penalty = 0.0
    penalty += 100.0 if metrics.get("payload_oob") else 0.0
    penalty += 40.0 if metrics.get("quad_oob") else 0.0
    penalty += 15.0 if metrics.get("truncated") and not metrics.get("quad_oob") else 0.0
    penalty += 3.0 * metrics["final_goal_dist"]
    if np.isfinite(metrics["payload_vel_rms"]):
        penalty += 8.0 * metrics["payload_vel_rms"]
    if np.isfinite(metrics["quad_vel_rms_mean"]):
        penalty += 4.0 * metrics["quad_vel_rms_mean"]
    if np.isfinite(metrics["plan_du_mean"]):
        penalty += 3.0 * metrics["plan_du_mean"]
    if np.isfinite(metrics["plan_du_max"]):
        penalty += 0.5 * metrics["plan_du_max"]
    penalty -= 0.05 * metrics["rollout_steps"]
    metrics["score"] = float(penalty)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _default_candidates():
    # Curated, not full Cartesian explosion.
    # Includes baseline and progressively stronger smoothing/regularization.
    return [
        {"name": "baseline", "overrides": {}},
        {"name": "acc_x4", "overrides": {"mujoco_payload_k_acc": 0.02}},
        {"name": "acc_x10", "overrides": {"mujoco_payload_k_acc": 0.05}},
        {"name": "vel_ang_x10", "overrides": {"mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.01}},
        {"name": "quat_vel", "overrides": {"mujoco_payload_reg_w_quat": 0.01, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02}},
        {"name": "acc+quat_vel", "overrides": {"mujoco_payload_k_acc": 0.02, "mujoco_payload_reg_w_quat": 0.01, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02}},
        {"name": "acc+quad_pos", "overrides": {"mujoco_payload_k_acc": 0.02, "mujoco_payload_reg_w_quad_pos": 0.01, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02}},
        {"name": "goal500_coll250", "overrides": {"weight_goal": 500.0, "collision_weight": 250.0, "mujoco_payload_k_acc": 0.02, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02}},
    ]


def _broad_candidates():
    cands = [{"name": "baseline", "overrides": {}}]
    # Goal/collision sweeps first (you asked to include these)
    for wg, cw in [(300, 200), (500, 200), (800, 200), (1200, 200), (500, 100), (500, 350), (800, 350)]:
        cands.append({
            "name": f"goal{wg}_coll{cw}",
            "overrides": {"weight_goal": float(wg), "collision_weight": float(cw)},
        })
    # Payload smoothing/regularization sweeps
    for k_acc in [0.01, 0.02, 0.05]:
        cands.append({"name": f"kacc_{k_acc:g}", "overrides": {"mujoco_payload_k_acc": float(k_acc)}})
    for wv, ww in [(0.005, 0.01), (0.01, 0.02), (0.02, 0.05)]:
        cands.append({
            "name": f"vel{wv:g}_ang{ww:g}",
            "overrides": {
                "mujoco_payload_reg_w_vel": float(wv),
                "mujoco_payload_reg_w_ang_vel": float(ww),
            },
        })
    # Combined settings (likely strongest)
    combos = [
        {"weight_goal": 500.0, "collision_weight": 200.0, "mujoco_payload_k_acc": 0.02, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02},
        {"weight_goal": 800.0, "collision_weight": 200.0, "mujoco_payload_k_acc": 0.02, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02},
        {"weight_goal": 800.0, "collision_weight": 350.0, "mujoco_payload_k_acc": 0.02, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02},
        {"weight_goal": 800.0, "collision_weight": 200.0, "mujoco_payload_k_acc": 0.05, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02, "mujoco_payload_reg_w_quat": 0.01},
        {"weight_goal": 500.0, "collision_weight": 200.0, "mujoco_payload_k_acc": 0.02, "mujoco_payload_reg_w_vel": 0.01, "mujoco_payload_reg_w_ang_vel": 0.02, "mujoco_payload_reg_w_quad_pos": 0.01},
    ]
    for i, ov in enumerate(combos):
        cands.append({"name": f"combo_{i}", "overrides": ov})
    return cands


def _wide_random_candidates(n_trials: int = 48, seed: int = 42):
    """Randomized broad sweep across high-impact optimizer knobs.

    Not exhaustive over every YAML field (too large), but covers the main continuous
    solver + cost parameters and payload regularization knobs.
    """
    rng = np.random.default_rng(seed)
    cands = [{"name": "baseline", "overrides": {}}]
    # Add a few anchors from previous good performers
    anchors = [
        {"name": "anchor_goal1200_coll200", "overrides": {"weight_goal": 1200.0, "collision_weight": 200.0}},
        {"name": "anchor_kacc005", "overrides": {"mujoco_payload_k_acc": 0.05}},
        {"name": "anchor_goal1200_kacc005", "overrides": {"weight_goal": 1200.0, "collision_weight": 200.0, "mujoco_payload_k_acc": 0.05}},
    ]
    cands.extend(anchors)

    # Log-spaced helper
    def sample_log(low, high):
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    for i in range(n_trials):
        # sample many knobs from opt_training.yaml + payload regularization overrides
        wg = float(rng.choice([300.0, 500.0, 800.0, 1200.0, 1600.0, 2200.0]))
        cw = float(rng.choice([100.0, 150.0, 200.0, 300.0, 400.0]))

        # solver params
        init_reg = sample_log(1e2, 1e5)
        th_stop = sample_log(1e-3, 5e-2)
        th_acceptnegstep = float(rng.choice([0.005, 0.01, 0.02, 0.05, 0.1]))
        max_iter = int(rng.choice([100, 150, 200, 300, 400]))
        penalty_iterations = int(rng.choice([1, 2, 3]))

        # payload smoothing / regularization (broader than before)
        k_acc = float(rng.choice([0.005, 0.01, 0.02, 0.05, 0.1, 0.2]))
        w_vel = float(rng.choice([0.001, 0.005, 0.01, 0.02, 0.05]))
        w_ang = float(rng.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1]))
        w_quat = float(rng.choice([0.0, 0.001, 0.005, 0.01, 0.02]))
        w_qpos = float(rng.choice([0.0, 0.005, 0.01, 0.02]))
        w_ppos = float(rng.choice([0.0, 0.002, 0.005, 0.01]))

        ov = {
            "weight_goal": wg,
            "collision_weight": cw,
            "init_reg": init_reg,
            "th_stop": th_stop,
            "th_acceptnegstep": th_acceptnegstep,
            "max_iter": max_iter,
            "penalty_iterations": penalty_iterations,
            "mujoco_payload_k_acc": k_acc,
            "mujoco_payload_reg_w_vel": w_vel,
            "mujoco_payload_reg_w_ang_vel": w_ang,
            "mujoco_payload_reg_w_quat": w_quat,
            "mujoco_payload_reg_w_quad_pos": w_qpos,
            "mujoco_payload_reg_w_payload_pos": w_ppos,
            # Keep these fixed for sweep stability unless explicitly exploring them
            "states_reg": True,
            "reg_control": False,
            "control_reg_weight": 0.0,
            "num_threads": 1,
        }
        cands.append({"name": f"wide_{i:03d}", "overrides": ov})

    return cands


def _focused_candidates():
    """Second-stage sweep around current winner, with higher values + delta variants."""
    cands = []
    base_opt = {"weight_goal": 1200.0, "collision_weight": 200.0}
    # Baseline winner and stronger goal weights
    for wg in [1200.0, 1600.0, 2200.0, 3000.0]:
        cands.append({"name": f"goal{int(wg)}_base", "overrides": {**base_opt, "weight_goal": wg}})
    # Stronger payload smoothing / regularization basins
    for k_acc in [0.05, 0.1, 0.2, 0.4]:
        for w_vel, w_ang in [(0.01, 0.02), (0.02, 0.05), (0.05, 0.1)]:
            cands.append({
                "name": f"goal1200_k{k_acc:g}_v{w_vel:g}_w{w_ang:g}",
                "overrides": {
                    **base_opt,
                    "mujoco_payload_k_acc": k_acc,
                    "mujoco_payload_reg_w_vel": w_vel,
                    "mujoco_payload_reg_w_ang_vel": w_ang,
                    "mujoco_payload_reg_w_quat": 0.01,
                    "mujoco_payload_reg_w_quad_pos": 0.01,
                    "mujoco_payload_reg_w_payload_pos": 0.005,
                },
            })
    # Solver hyperparams around plausible ranges
    for init_reg, th_stop, max_iter in [
        (1e3, 1e-2, 200),
        (1e4, 1e-2, 300),
        (1e4, 5e-3, 300),
        (1e5, 5e-3, 400),
    ]:
        cands.append({
            "name": f"solver_ir{int(init_reg):d}_ts{th_stop:g}_mi{max_iter}",
            "overrides": {
                **base_opt,
                "init_reg": float(init_reg),
                "th_stop": float(th_stop),
                "max_iter": int(max_iter),
                "th_acceptnegstep": 0.01,
                "penalty_iterations": 2,
                "mujoco_payload_k_acc": 0.1,
                "mujoco_payload_reg_w_vel": 0.02,
                "mujoco_payload_reg_w_ang_vel": 0.05,
                "mujoco_payload_reg_w_quat": 0.01,
            },
        })
    # Delta sensitivity (you called this out)
    for d0, dr, dmin in [
        (0.70, 0.95, 0.20),
        (0.85, 0.95, 0.20),   # current default
        (1.00, 0.95, 0.20),
        (1.20, 0.95, 0.20),
        (1.20, 0.98, 0.30),
        (1.50, 0.98, 0.40),
    ]:
        cands.append({
            "name": f"delta_d0{d0:g}_r{dr:g}_m{dmin:g}",
            "overrides": {
                **base_opt,
                "mujoco_payload_k_acc": 0.1,
                "mujoco_payload_reg_w_vel": 0.02,
                "mujoco_payload_reg_w_ang_vel": 0.05,
                "mujoco_payload_reg_w_quat": 0.01,
            },
            "pc_overrides": {
                "pc-dbcbs": {
                    "default": {
                        "delta_0": float(d0),
                        "delta_rate": float(dr),
                        "delta_min": float(dmin),
                    }
                }
            },
        })
    # Deduplicate by serialized overrides
    seen = set()
    out = []
    for c in cands:
        key = json.dumps({"o": c.get("overrides", {}), "p": c.get("pc_overrides", {})}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _delta_only_candidates():
    """Sweep only pc-dbCBS delta schedule, preserving current opt_training.yaml values."""
    cands = [{"name": "baseline", "overrides": {}}]
    for d0, dr, dmin in [
        (0.50, 0.95, 0.20),
        (0.60, 0.95, 0.20),
        (0.70, 0.95, 0.20),
        (0.85, 0.95, 0.20),  # current default
        (1.00, 0.95, 0.20),
        (1.10, 0.95, 0.20),
        (1.20, 0.95, 0.20),
        (1.00, 0.98, 0.30),
        (1.20, 0.98, 0.30),
    ]:
        cands.append({
            "name": f"delta_d0{d0:g}_r{dr:g}_m{dmin:g}",
            "overrides": {},
            "pc_overrides": {
                "pc-dbcbs": {
                    "default": {
                        "delta_0": float(d0),
                        "delta_rate": float(dr),
                        "delta_min": float(dmin),
                    }
                }
            },
        })
    return cands


def _plot_results(out_dir: Path, metrics_rows):
    rows = sorted(metrics_rows, key=lambda r: r["score"])
    names = [f"{r['cfg_id']:02d}:{r['name']}" for r in rows]
    scores = [r["score"] for r in rows]
    pvr = [r["payload_vel_rms"] for r in rows]
    du = [r["plan_du_mean"] for r in rows]
    dfin = [r["final_goal_dist"] for r in rows]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    ax = axs[0, 0]
    ax.barh(names, scores, color="tab:blue")
    ax.set_title("Heuristic Tuning Score (lower better)")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axs[0, 1]
    ax.scatter(pvr, du, c=scores, cmap="viridis", s=70)
    for r in rows:
        ax.text(r["payload_vel_rms"], r["plan_du_mean"], str(r["cfg_id"]), fontsize=8)
    ax.set_xlabel("payload_vel_rms")
    ax.set_ylabel("plan_du_mean")
    ax.set_title("Smoothness Tradeoff")
    ax.grid(True, alpha=0.3)

    ax = axs[1, 0]
    x = np.arange(len(rows))
    ax.plot(x, pvr, "-o", label="payload_vel_rms")
    ax.plot(x, du, "-s", label="plan_du_mean")
    ax.plot(x, dfin, "-^", label="final_goal_dist")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["cfg_id"]) for r in rows], rotation=45)
    ax.set_title("Key Metrics by Ranked Config")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[1, 1]
    # categorical outcomes
    payload_oob = [1 if r.get("payload_oob") else 0 for r in rows]
    quad_oob = [1 if r.get("quad_oob") else 0 for r in rows]
    success = [1 if r.get("success") else 0 for r in rows]
    trunc = [1 if r.get("truncated") else 0 for r in rows]
    ax.plot(x, success, "g-o", label="success")
    ax.plot(x, payload_oob, "r-o", label="payload_oob")
    ax.plot(x, quad_oob, "m-o", label="quad_oob")
    ax.plot(x, trunc, "k--o", label="truncated")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["cfg_id"]) for r in rows], rotation=45)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Outcome Flags by Ranked Config")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = out_dir / "tuning_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Parallel tuning sweep for pc-dbCBS payload optimizer.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--replan-k", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--n-opt", type=int, default=30)
    ap.add_argument("--time-limit-ms", type=float, default=100000.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--nondeterministic", action="store_true", help="Disable deterministic expert mode during tuning.")
    ap.add_argument("--preset", choices=["quick", "broad", "wide", "focused", "delta_only"], default="wide")
    ap.add_argument("--n-trials", type=int, default=40, help="For --preset wide: number of random configs (plus anchors).")
    ap.add_argument("--seed", type=int, default=42, help="For --preset wide random sampling.")
    ap.add_argument("--stream-logs", action="store_true")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "runs" / "opt_tuning" / f"k{args.replan_k}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preset == "focused":
        candidates = _focused_candidates()
    elif args.preset == "delta_only":
        candidates = _delta_only_candidates()
    elif args.preset == "wide":
        candidates = _wide_random_candidates(n_trials=args.n_trials, seed=args.seed)
    elif args.preset == "broad":
        candidates = _broad_candidates()
    else:
        candidates = _default_candidates()
    for i, c in enumerate(candidates):
        c["id"] = i
    (out_dir / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    worker_kwargs = dict(
        out_dir_str=str(out_dir),
        steps=args.steps,
        replan_k=args.replan_k,
        max_steps=args.max_steps,
        deterministic=not args.nondeterministic,
        time_limit_ms=args.time_limit_ms,
        n_opt=args.n_opt,
        stream_logs=args.stream_logs,
    )
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_rollout_eval, cand, **worker_kwargs) for cand in candidates]
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r["cfg_id"])
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    ranked = sorted(results, key=lambda r: r["score"])
    _plot_results(out_dir, results)

    print(f"[tune] wrote outputs to {out_dir}")
    for r in ranked:
        print(
            f"cfg {r['cfg_id']:02d} {r['name']}: score={r['score']:.3f} "
            f"steps={r['rollout_steps']} replans={r['replans']} "
            f"goal={r['final_goal_dist']:.3f} pvel_rms={r['payload_vel_rms']:.3f} "
            f"du_mean={r['plan_du_mean']:.3f} "
            f"flags(succ/p_oob/q_oob/trunc)={int(bool(r.get('success')))}{int(bool(r.get('payload_oob')))}{int(bool(r.get('quad_oob')))}{int(bool(r.get('truncated')))}"
        )


if __name__ == "__main__":
    main()
