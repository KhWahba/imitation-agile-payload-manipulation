#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


def _safe_load_yaml(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else None


def _to_np_2d(seq: Any) -> Optional[np.ndarray]:
    if not isinstance(seq, list):
        return None
    if len(seq) == 0:
        return np.zeros((0, 0), dtype=float)
    if not isinstance(seq[0], list):
        return None
    return np.asarray(seq, dtype=float)


def _traj_from_yaml_dict(data: Optional[dict]) -> Optional[Dict[str, np.ndarray]]:
    if not data:
        return None
    if "result" in data and isinstance(data["result"], dict):
        data = data["result"]
    states = _to_np_2d(data.get("states"))
    actions = _to_np_2d(data.get("actions"))
    if states is None:
        return None
    if actions is None:
        actions = np.zeros((0, 0), dtype=float)
    return {"states": states, "actions": actions}


def _dbcbs_multi_from_yaml(data: Optional[dict]) -> Optional[List[Dict[str, np.ndarray]]]:
    if not data:
        return None
    res = data.get("result")
    if not isinstance(res, list):
        return None
    out: List[Dict[str, np.ndarray]] = []
    for item in res:
        if not isinstance(item, dict):
            continue
        states = _to_np_2d(item.get("states"))
        actions = _to_np_2d(item.get("actions"))
        if states is None:
            continue
        if actions is None:
            actions = np.zeros((0, 0), dtype=float)
        out.append({"states": states, "actions": actions})
    return out or None


def _round_sig(arr: np.ndarray, n: int = 64) -> str:
    if arr.size == 0:
        return "empty"
    flat = np.round(arr.reshape(-1)[:n], 4)
    h = hashlib.sha1(flat.tobytes()).hexdigest()
    return h[:12]


def _norm_rows(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.zeros((0,), dtype=float)
    return np.linalg.norm(a, axis=1)


def _quat_norm_err_xyzw(q: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return np.zeros((0,), dtype=float)
    return np.abs(np.linalg.norm(q, axis=1) - 1.0)


def _quat_identity_dist_xyzw(q: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return np.zeros((0,), dtype=float)
    # q and -q represent same orientation; identity is [0,0,0,1]
    dots = np.abs(q[:, 3])
    dots = np.clip(dots, -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def _float_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p95": 0.0}
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "p95": float(np.percentile(x, 95)),
    }


@dataclass
class JointLayout:
    n_quads: int
    cable_lengths: List[float]

    @property
    def n_bodies(self) -> int:
        return 1 + self.n_quads

    @property
    def nx(self) -> int:
        return 13 * self.n_bodies


def _infer_layout(run_dir: Path, input_data: Optional[dict], env_data: Optional[dict]) -> Optional[JointLayout]:
    del run_dir
    src = input_data or env_data
    if not src:
        return None
    group = src.get("joint_robot")
    if not (isinstance(group, list) and group):
        # generated env.yaml stores the joint robot under "robots"
        cand = src.get("robots")
        if isinstance(cand, list) and cand:
            first = cand[0]
            if isinstance(first, dict) and isinstance(first.get("type"), str) and "payload" in first.get("type", ""):
                group = cand
    if not isinstance(group, list) or not group:
        return None
    jr0 = group[0]
    if not isinstance(jr0, dict):
        return None
    n_quads = int(jr0.get("quadsNum", 0))
    if n_quads <= 0:
        # fallback from l
        l = jr0.get("l")
        if isinstance(l, list):
            n_quads = len(l)
    if n_quads <= 0:
        return None
    l_raw = jr0.get("l", [0.5] * n_quads)
    if not isinstance(l_raw, list):
        l_raw = [0.5] * n_quads
    cable_lengths = [float(v) for v in l_raw[:n_quads]]
    if len(cable_lengths) < n_quads:
        cable_lengths.extend([0.5] * (n_quads - len(cable_lengths)))
    return JointLayout(n_quads=n_quads, cable_lengths=cable_lengths)


def _joint_metrics(traj: Optional[Dict[str, np.ndarray]], layout: Optional[JointLayout], dt: float = 0.02) -> Dict[str, Any]:
    if traj is None:
        return {"present": False}
    X = traj["states"]
    U = traj["actions"]
    out: Dict[str, Any] = {
        "present": True,
        "num_states": int(X.shape[0]),
        "num_actions": int(U.shape[0]),
        "nx": int(X.shape[1]) if X.ndim == 2 and X.size else 0,
        "nu": int(U.shape[1]) if U.ndim == 2 and U.size else 0,
        "duration_sec_est": float(U.shape[0] * dt),
        "action_sig": _round_sig(U),
        "state_sig": _round_sig(X),
    }
    if U.size:
        u_norm = _norm_rows(U)
        out["u_norm"] = _float_stats(u_norm)
        out["u_min"] = float(np.min(U))
        out["u_max"] = float(np.max(U))
        out["u_sat_frac_ge_1p35"] = float(np.mean(U >= 1.35))
        out["u_zero_frac_le_1e3"] = float(np.mean(np.abs(U) <= 1e-3))
    if not layout or X.size == 0:
        return out
    n = layout.n_bodies
    expected = 13 * n
    if X.shape[1] != expected:
        out["layout_mismatch"] = {"expected_nx": expected, "got_nx": int(X.shape[1])}
        return out

    qpos_dim = 7 * n
    qvel_dim = 6 * n
    poses = X[:, :qpos_dim]
    vels = X[:, qpos_dim:qpos_dim + qvel_dim]
    payload_pos = poses[:, 0:3]
    payload_quat = poses[:, 3:7]
    payload_vel = vels[:, 0:3]
    payload_w = vels[:, 3:6]

    out["payload_path_len"] = float(np.sum(np.linalg.norm(np.diff(payload_pos, axis=0), axis=1))) if len(payload_pos) > 1 else 0.0
    out["payload_speed"] = _float_stats(_norm_rows(payload_vel))
    out["payload_ang_speed"] = _float_stats(_norm_rows(payload_w))
    out["payload_quat_norm_err"] = _float_stats(_quat_norm_err_xyzw(payload_quat))
    out["payload_quat_identity_angle_rad"] = _float_stats(_quat_identity_dist_xyzw(payload_quat))
    out["payload_z"] = _float_stats(payload_pos[:, 2])

    quad_positions = []
    quad_vels = []
    quad_ws = []
    cable_stats = []
    for i in range(layout.n_quads):
        qpos_base = 7 * (1 + i)
        qvel_base = 6 * (1 + i)
        qpos_i = poses[:, qpos_base:qpos_base + 7]
        qvel_i = vels[:, qvel_base:qvel_base + 6]
        quad_pos = qpos_i[:, 0:3]
        quad_quat = qpos_i[:, 3:7]
        quad_v = qvel_i[:, 0:3]
        quad_w = qvel_i[:, 3:6]
        quad_positions.append(quad_pos)
        quad_vels.append(quad_v)
        quad_ws.append(quad_w)

        d = np.linalg.norm(quad_pos - payload_pos, axis=1)
        l = float(layout.cable_lengths[i])
        slack = np.maximum(0.0, l - d)
        stretch = np.maximum(0.0, d - l)
        cable_stats.append({
            "target_l": l,
            "dist": _float_stats(d),
            "slack": _float_stats(slack),
            "stretch": _float_stats(stretch),
            "taut_frac_eps_1cm": float(np.mean(np.abs(d - l) <= 0.01)),
            "slack_frac_gt_1cm": float(np.mean((l - d) > 0.01)),
            "stretch_frac_gt_1cm": float(np.mean((d - l) > 0.01)),
            "quat_norm_err": _float_stats(_quat_norm_err_xyzw(quad_quat)),
            "quat_identity_angle_rad": _float_stats(_quat_identity_dist_xyzw(quad_quat)),
            "speed": _float_stats(_norm_rows(quad_v)),
            "ang_speed": _float_stats(_norm_rows(quad_w)),
        })
    out["cables"] = cable_stats
    if len(quad_positions) >= 2:
        d12 = np.linalg.norm(quad_positions[0] - quad_positions[1], axis=1)
        out["quad_pair_dist"] = _float_stats(d12)
    out["init_like_zero_vel_frac"] = float(np.mean(np.abs(vels) < 1e-6))
    return out


def _discrete_metrics(multi: Optional[List[Dict[str, np.ndarray]]], dt: float = 0.02) -> Dict[str, Any]:
    if multi is None:
        return {"present": False}
    per_robot = []
    for i, tr in enumerate(multi):
        X = tr["states"]
        U = tr["actions"]
        item: Dict[str, Any] = {
            "robot": i,
            "num_states": int(X.shape[0]),
            "num_actions": int(U.shape[0]),
            "nx": int(X.shape[1]) if X.size else 0,
            "nu": int(U.shape[1]) if U.size else 0,
            "duration_sec_est": float(U.shape[0] * dt),
            "state_sig": _round_sig(X),
            "action_sig": _round_sig(U),
        }
        if X.shape[0] > 1 and X.shape[1] >= 3:
            item["path_len_xyz"] = float(np.sum(np.linalg.norm(np.diff(X[:, :3], axis=0), axis=1)))
        if U.size:
            item["u_norm"] = _float_stats(_norm_rows(U))
            item["u_min"] = float(np.min(U))
            item["u_max"] = float(np.max(U))
        if X.shape[1] >= 13:
            item["quat_norm_err"] = _float_stats(_quat_norm_err_xyzw(X[:, 3:7]))
            item["speed"] = _float_stats(_norm_rows(X[:, 7:10]))
            item["ang_speed"] = _float_stats(_norm_rows(X[:, 10:13]))
        per_robot.append(item)
    return {"present": True, "num_robots": len(per_robot), "per_robot": per_robot}


def _first_existing(run_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    return None


def analyze_run_dir(run_dir: Path) -> Dict[str, Any]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    meta = _safe_load_yaml(run_dir / "meta.yaml")
    input_yaml = _safe_load_yaml(_first_existing(run_dir, ["input.yaml"]) or Path("/nonexistent"))
    env_yaml = _safe_load_yaml(_first_existing(run_dir, ["env.yaml"]) or Path("/nonexistent"))
    layout = _infer_layout(run_dir, input_yaml, env_yaml)

    result_dbcbs = _safe_load_yaml(run_dir / "result_dbcbs.yaml")
    init_guess = _safe_load_yaml(run_dir / "init_guess_mujoco.yaml")
    opt_yaml = _safe_load_yaml(_first_existing(run_dir, ["result_dbcbs_opt.yaml", "optimization.yaml"]) or Path("/nonexistent"))
    api_result = _safe_load_yaml(_first_existing(run_dir, ["result_subprocess_api.yaml", "result_api.yaml"]) or Path("/nonexistent"))

    discrete = _dbcbs_multi_from_yaml(result_dbcbs)
    init_traj = _traj_from_yaml_dict(init_guess)
    opt_traj = _traj_from_yaml_dict(opt_yaml)
    api_traj = None
    if api_result and isinstance(api_result.get("X"), list):
        X = _to_np_2d(api_result.get("X"))
        U = _to_np_2d(api_result.get("U"))
        if X is not None and U is not None:
            api_traj = {"states": X, "actions": U}

    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "meta": meta or {},
        "layout": ({
            "n_quads": layout.n_quads,
            "n_bodies": layout.n_bodies,
            "cable_lengths": layout.cable_lengths,
            "nx_expected_joint": layout.nx,
        } if layout else None),
        "files_present": {
            "meta_yaml": (run_dir / "meta.yaml").exists(),
            "input_yaml": (run_dir / "input.yaml").exists(),
            "env_yaml": (run_dir / "env.yaml").exists(),
            "result_dbcbs_yaml": (run_dir / "result_dbcbs.yaml").exists(),
            "init_guess_mujoco_yaml": (run_dir / "init_guess_mujoco.yaml").exists(),
            "result_dbcbs_opt_yaml": (run_dir / "result_dbcbs_opt.yaml").exists(),
            "optimization_yaml": (run_dir / "optimization.yaml").exists(),
            "result_subprocess_api_yaml": (run_dir / "result_subprocess_api.yaml").exists(),
            "result_api_yaml": (run_dir / "result_api.yaml").exists(),
        },
        "discrete": _discrete_metrics(discrete),
        "init_guess_joint": _joint_metrics(init_traj, layout),
        "optimized_joint": _joint_metrics(opt_traj, layout),
        "api_result_joint": _joint_metrics(api_traj, layout) if api_traj else {"present": False},
    }

    if api_result:
        out["api_result_flags"] = {
            "ok": bool(api_result.get("ok", False)),
            "solved_db": bool(api_result.get("solved_db", False)),
            "solved_opt": bool(api_result.get("solved_opt", False)),
            "feasible": bool(api_result.get("feasible", False)),
            "cost": float(api_result.get("cost", 0.0)) if api_result.get("cost") is not None else None,
            "duration_discrete_sec": api_result.get("duration_discrete_sec"),
            "duration_opt_sec": api_result.get("duration_opt_sec"),
            "info": api_result.get("info"),
        }

    return out


def _pick(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def compare_runs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    paths = [
        "meta.global_step",
        "meta.t_in_plan_before",
        "meta.reason",
        "meta.warmstart_optimization",
        "discrete.num_robots",
        "init_guess_joint.num_actions",
        "optimized_joint.num_actions",
        "optimized_joint.duration_sec_est",
        "optimized_joint.payload_path_len",
        "optimized_joint.payload_z.mean",
        "optimized_joint.payload_speed.mean",
        "optimized_joint.u_norm.mean",
        "optimized_joint.u_sat_frac_ge_1p35",
    ]
    diffs = {}
    for p in paths:
        av = _pick(a, p)
        bv = _pick(b, p)
        diffs[p] = {"a": av, "b": bv}
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            diffs[p]["delta_b_minus_a"] = float(bv - av)

    # Discrete signatures per robot for "different search result" detection
    da = _pick(a, "discrete.per_robot") or []
    db = _pick(b, "discrete.per_robot") or []
    per_robot = []
    for i in range(max(len(da), len(db))):
        ai = da[i] if i < len(da) else {}
        bi = db[i] if i < len(db) else {}
        per_robot.append({
            "robot": i,
            "a_action_sig": ai.get("action_sig"),
            "b_action_sig": bi.get("action_sig"),
            "same_discrete_action_prefix_sig": ai.get("action_sig") == bi.get("action_sig"),
            "a_num_actions": ai.get("num_actions"),
            "b_num_actions": bi.get("num_actions"),
            "a_path_len_xyz": ai.get("path_len_xyz"),
            "b_path_len_xyz": bi.get("path_len_xyz"),
        })

    # Cable metrics (optimized)
    ca = _pick(a, "optimized_joint.cables") or []
    cb = _pick(b, "optimized_joint.cables") or []
    cable_cmp = []
    for i in range(max(len(ca), len(cb))):
        ai = ca[i] if i < len(ca) else {}
        bi = cb[i] if i < len(cb) else {}
        cable_cmp.append({
            "cable": i,
            "a_taut_frac_eps_1cm": ai.get("taut_frac_eps_1cm"),
            "b_taut_frac_eps_1cm": bi.get("taut_frac_eps_1cm"),
            "a_slack_frac_gt_1cm": ai.get("slack_frac_gt_1cm"),
            "b_slack_frac_gt_1cm": bi.get("slack_frac_gt_1cm"),
            "a_stretch_frac_gt_1cm": ai.get("stretch_frac_gt_1cm"),
            "b_stretch_frac_gt_1cm": bi.get("stretch_frac_gt_1cm"),
        })

    return {
        "summary_diffs": diffs,
        "discrete_per_robot": per_robot,
        "optimized_cable_compare": cable_cmp,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze/compare pc-dbCBS run artifacts (read-only).")
    ap.add_argument("--run", type=Path, help="Analyze a single run directory")
    ap.add_argument("--run-a", type=Path, help="First run dir for comparison")
    ap.add_argument("--run-b", type=Path, help="Second run dir for comparison")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional output JSON path")
    args = ap.parse_args()

    if args.run:
        report = analyze_run_dir(args.run)
    else:
        if not (args.run_a and args.run_b):
            ap.error("Provide either --run or both --run-a and --run-b")
        ra = analyze_run_dir(args.run_a)
        rb = analyze_run_dir(args.run_b)
        report = {"run_a": ra, "run_b": rb, "comparison": compare_runs(ra, rb)}

    txt = json.dumps(report, indent=2, sort_keys=False)
    print(txt)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
