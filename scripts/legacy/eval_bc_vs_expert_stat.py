#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml
from stable_baselines3.common.policies import ActorCriticPolicy

from payload_env import PayloadGymEnv
from expert_pcdbcbs_subprocess import PcDbCBSExpert as ExpertSubproc
from expert_pcdbcbs_subprocess import PcDbCBSPaths


REPO = Path(__file__).resolve().parents[1]
XML_PATH = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
TEMPLATE_YAML = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
PC_DBCBS_CFG = REPO / "deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml"
OPT_CFG = REPO / "deps/pc-dbCBS/configs/opt_training.yaml"
DYNOBENCH_BASE = REPO / "deps/pc-dbCBS/deps/dynoplan/dynobench"
BUILD_DIR = REPO / "deps/pc-dbCBS/build"


def _set_env_state_from_raw(env: PayloadGymEnv, raw_state: np.ndarray):
    """raw_state layout: poses-first [pos3, quat_xyzw4]*n then [lin3, ang3]*n."""
    s = np.asarray(raw_state, dtype=np.float64).reshape(-1)
    n = env.n_bodies
    if s.size != 13 * n:
        raise ValueError(f"Expected raw_state size {13*n}, got {s.size}")
    poses = s[: 7 * n].copy()
    vels = s[7 * n :].copy()
    # xyzw -> wxyz for MuJoCo qpos
    for i in range(n):
        b = 7 * i
        qx, qy, qz, qw = poses[b + 3 : b + 7]
        poses[b + 3 : b + 7] = [qw, qx, qy, qz]
    env.data.qpos[:] = poses
    env.data.qvel[:] = vels
    env.prev_action[:] = 0.0
    env.step_count = 0
    env.success_hold_count = 0
    import mujoco

    mujoco.mj_forward(env.model, env.data)


def _obs_raw_state(obs: np.ndarray, state_dim: int) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)[:state_dim].copy()


def _build_expert(replan_k: int, n_opt: int, time_limit_ms: float, keep_files: bool) -> ExpertSubproc:
    env_tmp = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=200)
    nu = env_tmp.action_dim
    act_low = np.zeros(nu, dtype=np.float32)
    act_high = np.ones(nu, dtype=np.float32) * 1.4
    return ExpertSubproc(
        paths=PcDbCBSPaths(
            bindings_path=str(BUILD_DIR),
            input_yaml=str(TEMPLATE_YAML),
            pc_dbcbs_cfg_yaml=str(PC_DBCBS_CFG),
            opt_cfg_yaml=str(OPT_CFG),
            dynobench_base=str(DYNOBENCH_BASE) + "/",
            motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
            time_limit=float(time_limit_ms),
            work_dir_root="runs/_tmp_pcdbcbs_eval",
            keep_files=bool(keep_files),
            warmstart_optimization=True,
            N_opt=int(n_opt),
        ),
        act_low=act_low,
        act_high=act_high,
        replan_every_k=int(replan_k),
    )


def _build_bc_policy(obs_space, act_space, arch: list[int], activation: str, ckpt: Path) -> ActorCriticPolicy:
    act_cls = th.nn.Tanh if activation.lower() == "tanh" else th.nn.ReLU
    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 1e-3,
        net_arch=arch,
        activation_fn=act_cls,
        ortho_init=False,
    )
    sd = th.load(str(ckpt), map_location="cpu")
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def _policy_act(policy: ActorCriticPolicy, obs: np.ndarray) -> np.ndarray:
    with th.no_grad():
        x = th.as_tensor(obs[None, :], dtype=th.float32)
        act, _, _ = policy.forward(x, deterministic=True)
    return act.squeeze(0).cpu().numpy().astype(np.float32)


def _rollout_with_agent(env: PayloadGymEnv, init_raw_state: np.ndarray, K: int, act_fn):
    _set_env_state_from_raw(env, init_raw_state)
    obs = env._get_obs()
    acts = []
    payload_pos = []
    done_t = None
    for t in range(K):
        a = act_fn(obs)
        obs, rew, term, trunc, info = env.step(a)
        acts.append(np.asarray(a, dtype=np.float32).copy())
        payload_pos.append(_obs_raw_state(obs, env.state_dim)[:3])
        if term or trunc:
            done_t = t + 1
            break
    return {
        "acts": np.asarray(acts, dtype=np.float32),
        "payload_pos": np.asarray(payload_pos, dtype=np.float32),
        "done_t": done_t,
    }


def _eval_one_init_worker(
    init_idx: int,
    s0: np.ndarray,
    K: int,
    expert_repeats: int,
    replan_k: int,
    n_opt: int,
    time_limit_ms: float,
    bc_best_json: str,
    bc_policy_pt: str,
    worker_tag: str,
):
    # Deterministic expert behavior per worker.
    os.environ["PCDBCBS_DETERMINISTIC"] = "1"
    os.environ.setdefault("PCDBCBS_EXPERT_SUBPROCESS_BIN", str(BUILD_DIR / "pc_dbcbs_expert"))
    os.environ.setdefault("PCDBCBS_EXPERT_SUBPROCESS_STREAM_LOGS", "0")

    env_ref = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=K)
    env_ref.terminate_on_success = True
    policy_cfg = json.loads(Path(bc_best_json).read_text(encoding="utf-8"))
    arch = policy_cfg["arch"]
    activation = policy_cfg.get("activation", "tanh")
    bc_policy = _build_bc_policy(env_ref.observation_space, env_ref.action_space, arch, activation, Path(bc_policy_pt))

    expert = ExpertSubproc(
        paths=PcDbCBSPaths(
            bindings_path=str(BUILD_DIR),
            input_yaml=str(TEMPLATE_YAML),
            pc_dbcbs_cfg_yaml=str(PC_DBCBS_CFG),
            opt_cfg_yaml=str(OPT_CFG),
            dynobench_base=str(DYNOBENCH_BASE) + "/",
            motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
            time_limit=float(time_limit_ms),
            work_dir_root=f"runs/_tmp_pcdbcbs_eval_{worker_tag}",
            keep_files=False,
            warmstart_optimization=True,
            N_opt=int(n_opt),
        ),
        act_low=np.zeros(env_ref.action_dim, dtype=np.float32),
        act_high=np.ones(env_ref.action_dim, dtype=np.float32) * 1.4,
        replan_every_k=int(replan_k),
    )

    env_bc = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=K)
    env_bc.terminate_on_success = True
    bc_roll = _rollout_with_agent(env_bc, s0, K, lambda obs: _policy_act(bc_policy, obs))

    ex_rolls = []
    for _ in range(expert_repeats):
        env_ex = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=K)
        env_ex.terminate_on_success = True
        expert.reset_episode()
        ex_rolls.append(_rollout_with_agent(env_ex, s0, K, lambda obs: expert.act(obs)))

    ex_acts_lens = [r["acts"].shape[0] for r in ex_rolls]
    ex_pos_lens = [r["payload_pos"].shape[0] for r in ex_rolls]
    aligned = min([bc_roll["acts"].shape[0], bc_roll["payload_pos"].shape[0], *ex_acts_lens, *ex_pos_lens])
    if aligned == 0:
        return None

    ex_acts = np.stack([r["acts"][:aligned] for r in ex_rolls], axis=0)
    ex_pos = np.stack([r["payload_pos"][:aligned] for r in ex_rolls], axis=0)
    ex_acts_mean = ex_acts.mean(axis=0)
    ex_pos_mean = ex_pos.mean(axis=0)

    act_err_t = np.linalg.norm(bc_roll["acts"][:aligned] - ex_acts_mean, axis=1)
    pos_err_t = np.linalg.norm(bc_roll["payload_pos"][:aligned] - ex_pos_mean, axis=1)

    ex_u_std = np.std(ex_acts, axis=0)
    ex_u_std_mean = float(ex_u_std.mean())
    ex_u_std_max = float(ex_u_std.max())

    done_vals = [r["done_t"] for r in ex_rolls if r["done_t"] is not None]
    expert_done_t_mean = float(np.mean(done_vals)) if done_vals else None

    return {
        "init_idx": init_idx,
        "aligned_len": int(aligned),
        "bc_done_t": bc_roll["done_t"],
        "expert_done_t_mean": expert_done_t_mean,
        "act_err_mean": float(act_err_t.mean()),
        "act_err_p95": float(np.percentile(act_err_t, 95)),
        "payload_pos_err_mean": float(pos_err_t.mean()),
        "payload_pos_err_p95": float(np.percentile(pos_err_t, 95)),
        "expert_u_std_mean": ex_u_std_mean,
        "expert_u_std_max": ex_u_std_max,
        "act_err_t": act_err_t.tolist(),
        "payload_pos_err_t": pos_err_t.tolist(),
    }


def _sample_init_states_from_dataset(trajs_pkl: Path, n_inits: int, seed: int) -> np.ndarray:
    trajs = pickle.loads(trajs_pkl.read_bytes())
    all_states = []
    state_dim = None
    for tr in trajs:
        if state_dim is None:
            # infer state dim from 2-quads setup
            state_dim = 39
        all_states.append(np.asarray(tr.obs[:, :state_dim], dtype=np.float32))
    all_states = np.concatenate(all_states, axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(all_states.shape[0], size=min(n_inits, all_states.shape[0]), replace=False)
    return all_states[idx]


def _goal_state_from_yaml(template_yaml: Path) -> np.ndarray:
    y = yaml.safe_load(template_yaml.read_text(encoding="utf-8"))
    return np.asarray(y["joint_robot"][0]["goal"], dtype=np.float32)


def _aggregate_and_plot(out_dir: Path, per_init_rows: list[dict], K: int):
    if not per_init_rows:
        return

    # Time-series arrays aligned by min available length per init.
    min_len = min(r["aligned_len"] for r in per_init_rows if r["aligned_len"] > 0)
    if min_len <= 0:
        return
    t = np.arange(min_len)
    act_err = np.stack([np.asarray(r["act_err_t"][:min_len], dtype=np.float32) for r in per_init_rows], axis=0)
    p_err = np.stack([np.asarray(r["payload_pos_err_t"][:min_len], dtype=np.float32) for r in per_init_rows], axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, arr, title, ylabel, color in [
        (axs[0], act_err, "||u_BC - u_Expert||", "L2", "tab:blue"),
        (axs[1], p_err, "||p_BC - p_Expert||", "m", "tab:green"),
    ]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        for i in range(arr.shape[0]):
            ax.plot(t, arr[i], color=color, alpha=0.12, lw=0.8)
        ax.plot(t, mean, color="black", lw=1.8, label="mean")
        ax.fill_between(t, mean - 2 * std, mean + 2 * std, color=color, alpha=0.2, label="±2σ")
        ax.set_title(title)
        ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bc_vs_expert_errors.png", dpi=150)
    plt.close(fig)

    # Failure histogram
    bc_done = [r["bc_done_t"] if r["bc_done_t"] is not None else K for r in per_init_rows]
    ex_done = [r["expert_done_t_mean"] if r["expert_done_t_mean"] is not None else K for r in per_init_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bc_done, bins=20, alpha=0.6, label="BC done_t")
    ax.hist(ex_done, bins=20, alpha=0.6, label="Expert done_t mean")
    ax.set_xlabel("timestep of termination/truncation (or K)")
    ax.set_ylabel("count")
    ax.set_title("Failure/Completion Timestep Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "done_t_hist.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Statistical BC-vs-Expert comparison from same initial states.")
    ap.add_argument("--trajs-pkl", type=str, required=True)
    ap.add_argument("--bc-best-json", type=str, required=True, help="Contains arch/activation fields.")
    ap.add_argument("--bc-policy-pt", type=str, required=True)
    ap.add_argument("--n-inits", type=int, default=30)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--expert-repeats", type=int, default=10)
    ap.add_argument("--replan-k", type=int, default=5)
    ap.add_argument("--time-limit-ms", type=float, default=50000.0)
    ap.add_argument("--n-opt", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--goal-hover-test", action="store_true", help="Also evaluate from exact goal state.")
    ap.add_argument("--goal-hover-steps", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers over sampled initial states.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (REPO / "runs" / "bc_vs_expert_eval" / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    init_states = _sample_init_states_from_dataset(Path(args.trajs_pkl), args.n_inits, args.seed)
    per_init_rows = []
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for i, s0 in enumerate(init_states):
            row = _eval_one_init_worker(
                init_idx=i,
                s0=s0,
                K=args.K,
                expert_repeats=args.expert_repeats,
                replan_k=args.replan_k,
                n_opt=args.n_opt,
                time_limit_ms=args.time_limit_ms,
                bc_best_json=args.bc_best_json,
                bc_policy_pt=args.bc_policy_pt,
                worker_tag=f"w{i:03d}",
            )
            if row is not None:
                per_init_rows.append(row)
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = []
            for i, s0 in enumerate(init_states):
                futs.append(
                    ex.submit(
                        _eval_one_init_worker,
                        i,
                        s0,
                        args.K,
                        args.expert_repeats,
                        args.replan_k,
                        args.n_opt,
                        args.time_limit_ms,
                        args.bc_best_json,
                        args.bc_policy_pt,
                        f"w{i:03d}",
                    )
                )
            for fut in cf.as_completed(futs):
                row = fut.result()
                if row is not None:
                    per_init_rows.append(row)
        per_init_rows.sort(key=lambda r: r["init_idx"])

    # Goal hover test (optional)
    hover = None
    if args.goal_hover_test:
        os.environ["PCDBCBS_DETERMINISTIC"] = "1"
        os.environ.setdefault("PCDBCBS_EXPERT_SUBPROCESS_BIN", str(BUILD_DIR / "pc_dbcbs_expert"))
        os.environ.setdefault("PCDBCBS_EXPERT_SUBPROCESS_STREAM_LOGS", "0")
        goal_state = _goal_state_from_yaml(TEMPLATE_YAML)
        env_ref = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=max(args.K, args.goal_hover_steps))
        env_ref.terminate_on_success = False
        policy_cfg = json.loads(Path(args.bc_best_json).read_text(encoding="utf-8"))
        arch = policy_cfg["arch"]
        activation = policy_cfg.get("activation", "tanh")
        bc_policy = _build_bc_policy(env_ref.observation_space, env_ref.action_space, arch, activation, Path(args.bc_policy_pt))
        expert = _build_expert(replan_k=args.replan_k, n_opt=args.n_opt, time_limit_ms=args.time_limit_ms, keep_files=False)

        env_bc = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=args.goal_hover_steps)
        env_bc.terminate_on_success = False
        bc_roll = _rollout_with_agent(env_bc, goal_state, args.goal_hover_steps, lambda obs: _policy_act(bc_policy, obs))

        ex_rolls = []
        for _ in range(args.expert_repeats):
            env_ex = PayloadGymEnv(str(XML_PATH), str(TEMPLATE_YAML), max_steps=args.goal_hover_steps)
            env_ex.terminate_on_success = False
            expert.reset_episode()
            ex_rolls.append(_rollout_with_agent(env_ex, goal_state, args.goal_hover_steps, lambda obs: expert.act(obs)))

        min_len = min([bc_roll["payload_pos"].shape[0], *[r["payload_pos"].shape[0] for r in ex_rolls]])
        ex_pos = np.stack([r["payload_pos"][:min_len] for r in ex_rolls], axis=0)
        ex_pos_mean = ex_pos.mean(axis=0)
        p_goal = goal_state[:3]
        bc_goal_dist = np.linalg.norm(bc_roll["payload_pos"][:min_len] - p_goal, axis=1)
        ex_goal_dist = np.linalg.norm(ex_pos_mean - p_goal, axis=1)
        hover = {
            "steps_used": int(min_len),
            "bc_goal_dist_mean": float(bc_goal_dist.mean()),
            "bc_goal_dist_p95": float(np.percentile(bc_goal_dist, 95)),
            "expert_goal_dist_mean": float(ex_goal_dist.mean()),
            "expert_goal_dist_p95": float(np.percentile(ex_goal_dist, 95)),
            "bc_done_t": bc_roll["done_t"],
            "expert_done_t_vals": [r["done_t"] for r in ex_rolls],
        }

    # Aggregate summary
    if not per_init_rows:
        raise RuntimeError("No valid initial states evaluated.")
    agg = {
        "n_inits_evaluated": len(per_init_rows),
        "K": int(args.K),
        "expert_repeats": int(args.expert_repeats),
        "replan_k": int(args.replan_k),
        "act_err_mean_of_means": float(np.mean([r["act_err_mean"] for r in per_init_rows])),
        "act_err_mean_p95": float(np.percentile([r["act_err_mean"] for r in per_init_rows], 95)),
        "payload_pos_err_mean_of_means": float(np.mean([r["payload_pos_err_mean"] for r in per_init_rows])),
        "payload_pos_err_mean_p95": float(np.percentile([r["payload_pos_err_mean"] for r in per_init_rows], 95)),
        "expert_u_std_mean_of_means": float(np.mean([r["expert_u_std_mean"] for r in per_init_rows])),
        "expert_u_std_max_overall": float(np.max([r["expert_u_std_max"] for r in per_init_rows])),
        "bc_done_t_median": float(np.median([r["bc_done_t"] if r["bc_done_t"] is not None else args.K for r in per_init_rows])),
    }

    (out_dir / "per_init_metrics.json").write_text(json.dumps(per_init_rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"aggregate": agg, "goal_hover": hover}, indent=2), encoding="utf-8")
    _aggregate_and_plot(out_dir, per_init_rows, args.K)

    print(json.dumps({"out_dir": str(out_dir), "aggregate": agg, "goal_hover": hover}, indent=2))


if __name__ == "__main__":
    import os

    main()
