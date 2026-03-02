#!/usr/bin/env python3
"""
test_policy_payload.py — Diagnostic evaluation of learned policy vs expert.

Usage:
    # Basic: just run the learned policy and render video
    python test_policy_payload.py --policy_path runs/bc_payload/policy_state_dict.pt

    # Compare against expert (requires pc-dbCBS bindings)
    python test_policy_payload.py --policy_path runs/bc_payload/policy_state_dict.pt --run_expert

    # Longer rollout, custom output dir
    python test_policy_payload.py --policy_path runs/bc_payload/policy_state_dict.pt --run_expert --steps 400 --out_dir runs/diag_test1

    # Skip video rendering (faster, just plots + metrics)
    python test_policy_payload.py --policy_path runs/bc_payload/policy_state_dict.pt --run_expert --no_video
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from imitation.algorithms import bc
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "scripts"))
from payload_env import PayloadGymEnv
from videos_from_log import VideoConfig, render_from_actions

np.set_printoptions(precision=4, suppress=True, linewidth=140)


# ═══════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Diagnostic: learned policy vs expert")
    p.add_argument("--policy_path", type=str, required=True,
                   help="Path to policy_state_dict.pt")
    p.add_argument("--xml_path", type=str,
                   default="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml")
    p.add_argument("--template_yaml", type=str,
                   default="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml")
    p.add_argument("--steps", type=int, default=300, help="Max rollout steps")
    p.add_argument("--out_dir", type=str, default="runs/diag_policy_vs_expert",
                   help="Output directory for plots, metrics, videos")
    p.add_argument("--run_expert", action="store_true",
                   help="Also run expert rollout for comparison (requires pc-dbCBS)")
    p.add_argument("--no_video", action="store_true",
                   help="Skip video rendering (faster)")
    p.add_argument("--replan_every_k", type=int, default=100,
                   help="Expert replanning interval")
    p.add_argument("--N_opt", type=int, default=100,
                   help="Expert optimization steps per plan")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# ROLLOUT FUNCTION
# ═══════════════════════════════════════════════════════════════════

def rollout(env, policy_fn, T, name="policy"):
    """
    Roll out a policy and record everything needed for diagnostics.
    policy_fn: obs (np.ndarray) -> action (np.ndarray) in [-1, 1]
    Returns dict with all logged data.
    """
    obs, _ = env.reset()
    qpos0 = env.data.qpos.copy()
    qvel0 = env.data.qvel.copy()

    obs_log = [obs.copy()]
    act_log = []
    ctrl_log = []          # MuJoCo ctrl (Newtons)
    state_log = []
    reward_log = []
    dist_log = []
    info_log = []

    for t in range(T):
        act = np.asarray(policy_fn(obs), dtype=np.float32).reshape(-1)
        obs, rew, terminated, truncated, info = env.step(act)

        act_log.append(act.copy())
        ctrl_log.append(env.action_mujoco.copy())
        state_log.append(obs[: env.state_dim].copy())
        reward_log.append(float(rew))
        dist_log.append(float(info.get("dist_goal", np.inf)))
        obs_log.append(obs.copy())
        info_log.append(info)

        if terminated or truncated:
            break

    return {
        "name": name,
        "qpos0": qpos0,
        "qvel0": qvel0,
        "obs": np.array(obs_log, dtype=np.float32),     # (T+1, obs_dim)
        "acts": np.array(act_log, dtype=np.float32),     # (T, act_dim)
        "ctrls": np.array(ctrl_log, dtype=np.float32),   # (T, act_dim) in Newtons
        "states": np.array(state_log, dtype=np.float32), # (T, state_dim)
        "rewards": np.array(reward_log, dtype=np.float32),
        "dists": np.array(dist_log, dtype=np.float32),
        "infos": info_log,
        "steps": len(act_log),
        "total_return": float(np.sum(reward_log)),
        "final_dist": float(dist_log[-1]) if dist_log else float("inf"),
        "out_of_bounds": bool(info_log[-1].get("out_of_bounds", False)) if info_log else False,
    }


# ═══════════════════════════════════════════════════════════════════
# STEP-BY-STEP EXPERT vs LEARNER ON SAME STATES
# ═══════════════════════════════════════════════════════════════════

def expert_labels_on_learner_states(env, expert, learner_rollout_data):
    """
    Re-roll the learner trajectory (using the same initial state),
    but at each step query BOTH the expert and the learner.
    This gives us the action error at every state the learner actually visits.
    """
    obs_seq = learner_rollout_data["obs"]  # (T+1, obs_dim)
    T = learner_rollout_data["steps"]

    expert_acts = []
    for t in range(T):
        obs_t = obs_seq[t]
        a_exp = expert.act(obs_t).astype(np.float32)
        expert_acts.append(a_exp)

    return np.array(expert_acts, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS (generic for N robots)
# ═══════════════════════════════════════════════════════════════════

def get_body_positions_from_states(states, n_bodies):
    """
    states: (T, 13*n_bodies)
    Returns: dict of body_name -> (T, 3) positions
    """
    T = states.shape[0]
    positions = {}
    positions["payload"] = states[:, 0:3]
    for i in range(1, n_bodies):
        positions[f"quad_{i}"] = states[:, 7*i:7*i+3]
    return positions


def get_body_quats_from_states(states, n_bodies):
    """Returns: dict of body_name -> (T, 4) quaternions (xyzw)"""
    quats = {}
    quats["payload"] = states[:, 3:7]
    for i in range(1, n_bodies):
        quats[f"quad_{i}"] = states[:, 7*i+3:7*i+7]
    return quats


def get_body_velocities_from_states(states, n_bodies):
    """Returns: dict of body_name -> (T, 3) linear velocities"""
    vel_offset = 7 * n_bodies
    vels = {}
    vels["payload"] = states[:, vel_offset:vel_offset+3]
    for i in range(1, n_bodies):
        vels[f"quad_{i}"] = states[:, vel_offset+6*i:vel_offset+6*i+3]
    return vels


def plot_3d_trajectories(rollouts, n_bodies, goal, out_path):
    """3D plot of payload + quad trajectories for each rollout."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors_policy = plt.cm.tab10(np.linspace(0, 1, n_bodies))
    styles = {"policy": "-", "expert": "--"}

    for rd in rollouts:
        name = rd["name"]
        ls = styles.get(name, "-")
        positions = get_body_positions_from_states(rd["states"], n_bodies)

        for idx, (body_name, pos) in enumerate(positions.items()):
            label = f"{name}/{body_name}" if rd is rollouts[0] or name == "expert" else None
            alpha = 1.0 if name == "policy" else 0.5
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], ls, color=colors_policy[idx],
                    alpha=alpha, linewidth=1.5, label=label)
            # mark start and end
            ax.scatter(*pos[0], marker="o", color=colors_policy[idx], s=40, alpha=alpha)
            ax.scatter(*pos[-1], marker="x", color=colors_policy[idx], s=60, alpha=alpha)

    # mark goal
    goal_pos = goal[:3]
    ax.scatter(*goal_pos, marker="*", color="gold", s=200, edgecolors="black", zorder=10, label="goal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Trajectories: Policy (solid) vs Expert (dashed)")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved 3D trajectory: {out_path}")


def plot_positions_over_time(rollouts, n_bodies, goal, out_path):
    """Per-axis position plots for payload and each quad."""
    axes_names = ["x", "y", "z"]
    fig, axs = plt.subplots(n_bodies, 3, figsize=(14, 3.5 * n_bodies), sharex=True)
    if n_bodies == 1:
        axs = axs[np.newaxis, :]

    for rd in rollouts:
        name = rd["name"]
        ls = "-" if name == "policy" else "--"
        lw = 1.5 if name == "policy" else 1.0
        positions = get_body_positions_from_states(rd["states"], n_bodies)
        T = rd["steps"]
        time = np.arange(T)

        for idx, (body_name, pos) in enumerate(positions.items()):
            for ax_i in range(3):
                axs[idx, ax_i].plot(time, pos[:, ax_i], ls, linewidth=lw,
                                    label=name, alpha=0.9 if name == "policy" else 0.6)
                # goal line
                goal_val = goal[7 * idx + ax_i]
                axs[idx, ax_i].axhline(goal_val, color="gold", linestyle=":", linewidth=1, alpha=0.7)

    for idx in range(n_bodies):
        body_name = "payload" if idx == 0 else f"quad_{idx}"
        for ax_i in range(3):
            axs[idx, ax_i].set_ylabel(f"{body_name} {axes_names[ax_i]} [m]")
            axs[idx, ax_i].grid(True, alpha=0.3)
            if idx == 0 and ax_i == 0:
                axs[idx, ax_i].legend(fontsize=7)
    for ax_i in range(3):
        axs[-1, ax_i].set_xlabel("step")

    fig.suptitle("Positions Over Time", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved positions: {out_path}")


def plot_velocities_over_time(rollouts, n_bodies, out_path):
    """Per-axis linear velocity plots."""
    axes_names = ["vx", "vy", "vz"]
    fig, axs = plt.subplots(n_bodies, 3, figsize=(14, 3.5 * n_bodies), sharex=True)
    if n_bodies == 1:
        axs = axs[np.newaxis, :]

    for rd in rollouts:
        name = rd["name"]
        ls = "-" if name == "policy" else "--"
        lw = 1.5 if name == "policy" else 1.0
        vels = get_body_velocities_from_states(rd["states"], n_bodies)
        T = rd["steps"]
        time = np.arange(T)

        for idx, (body_name, vel) in enumerate(vels.items()):
            for ax_i in range(3):
                axs[idx, ax_i].plot(time, vel[:, ax_i], ls, linewidth=lw,
                                    label=name, alpha=0.9 if name == "policy" else 0.6)

    for idx in range(n_bodies):
        body_name = "payload" if idx == 0 else f"quad_{idx}"
        for ax_i in range(3):
            axs[idx, ax_i].set_ylabel(f"{body_name} {axes_names[ax_i]} [m/s]")
            axs[idx, ax_i].axhline(0, color="gray", linestyle=":", linewidth=0.5)
            axs[idx, ax_i].grid(True, alpha=0.3)
            if idx == 0 and ax_i == 0:
                axs[idx, ax_i].legend(fontsize=7)
    for ax_i in range(3):
        axs[-1, ax_i].set_xlabel("step")

    fig.suptitle("Linear Velocities Over Time", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved velocities: {out_path}")


def plot_actions_over_time(rollouts, n_quads, out_path):
    """Per-motor action in [-1, 1] policy space."""
    n_motors = 4 * n_quads
    fig, axs = plt.subplots(n_quads, 4, figsize=(16, 3.5 * n_quads), sharex=True, sharey=True)
    if n_quads == 1:
        axs = axs[np.newaxis, :]

    for rd in rollouts:
        name = rd["name"]
        ls = "-" if name == "policy" else "--"
        lw = 1.5 if name == "policy" else 1.0
        acts = rd["acts"]
        T = acts.shape[0]
        time = np.arange(T)

        for q in range(n_quads):
            for m in range(4):
                motor_idx = 4 * q + m
                axs[q, m].plot(time, acts[:, motor_idx], ls, linewidth=lw,
                               label=name, alpha=0.9 if name == "policy" else 0.6)

    for q in range(n_quads):
        for m in range(4):
            axs[q, m].set_ylabel(f"Q{q+1} M{m+1}")
            axs[q, m].axhline(0, color="gray", linestyle=":", linewidth=0.5)
            axs[q, m].set_ylim(-1.15, 1.15)
            axs[q, m].grid(True, alpha=0.3)
            if q == 0 and m == 0:
                axs[q, m].legend(fontsize=7)
    for m in range(4):
        axs[-1, m].set_xlabel("step")

    fig.suptitle("Actions (policy space [-1, 1])", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved actions: {out_path}")


def plot_ctrls_over_time(rollouts, n_quads, out_path):
    """Per-motor MuJoCo ctrl in Newtons."""
    fig, axs = plt.subplots(n_quads, 4, figsize=(16, 3.5 * n_quads), sharex=True, sharey=True)
    if n_quads == 1:
        axs = axs[np.newaxis, :]

    for rd in rollouts:
        name = rd["name"]
        ls = "-" if name == "policy" else "--"
        lw = 1.5 if name == "policy" else 1.0
        ctrls = rd["ctrls"]
        T = ctrls.shape[0]
        time = np.arange(T)

        for q in range(n_quads):
            for m in range(4):
                motor_idx = 4 * q + m
                axs[q, m].plot(time, ctrls[:, motor_idx], ls, linewidth=lw,
                               label=name, alpha=0.9 if name == "policy" else 0.6)

    u_nominal = 0.034 * 9.81 / 4
    for q in range(n_quads):
        for m in range(4):
            axs[q, m].set_ylabel(f"Q{q+1} M{m+1} [N]")
            axs[q, m].axhline(u_nominal, color="green", linestyle=":", linewidth=1, alpha=0.5, label="hover" if (q==0 and m==0) else None)
            axs[q, m].grid(True, alpha=0.3)
            if q == 0 and m == 0:
                axs[q, m].legend(fontsize=7)
    for m in range(4):
        axs[-1, m].set_xlabel("step")

    fig.suptitle("MuJoCo ctrl (Newtons)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved ctrls: {out_path}")


def plot_dist_and_reward(rollouts, out_path):
    """Distance to goal and cumulative reward."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for rd in rollouts:
        name = rd["name"]
        ls = "-" if name == "policy" else "--"
        T = rd["steps"]
        time = np.arange(T)

        ax1.plot(time, rd["dists"], ls, linewidth=1.5, label=name)
        ax2.plot(time, np.cumsum(rd["rewards"]), ls, linewidth=1.5, label=name)

    ax1.set_ylabel("dist to goal [m]")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Distance to Goal")

    ax2.set_ylabel("cumulative reward")
    ax2.set_xlabel("step")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved dist+reward: {out_path}")


def plot_action_error(learner_acts, expert_acts_on_learner_states, n_quads, out_path):
    """
    Per-motor absolute action error at each step.
    This is THE key diagnostic: where does the policy disagree with the expert?
    """
    T = min(learner_acts.shape[0], expert_acts_on_learner_states.shape[0])
    error = np.abs(learner_acts[:T] - expert_acts_on_learner_states[:T])
    time = np.arange(T)

    fig, axs = plt.subplots(n_quads, 4, figsize=(16, 3.5 * n_quads), sharex=True, sharey=True)
    if n_quads == 1:
        axs = axs[np.newaxis, :]

    for q in range(n_quads):
        for m in range(4):
            motor_idx = 4 * q + m
            axs[q, m].plot(time, error[:, motor_idx], "r-", linewidth=1.0, alpha=0.8)
            axs[q, m].fill_between(time, 0, error[:, motor_idx], color="red", alpha=0.15)
            axs[q, m].set_ylabel(f"Q{q+1} M{m+1} |err|")
            axs[q, m].grid(True, alpha=0.3)
    for m in range(4):
        axs[-1, m].set_xlabel("step")

    fig.suptitle("Per-Motor |Action Error| (policy vs expert at learner states)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved action error: {out_path}")


def plot_action_error_summary(learner_acts, expert_acts_on_learner_states, out_path):
    """Mean and max action error over time — the divergence signal."""
    T = min(learner_acts.shape[0], expert_acts_on_learner_states.shape[0])
    error = np.abs(learner_acts[:T] - expert_acts_on_learner_states[:T])

    mean_err = error.mean(axis=1)
    max_err = error.max(axis=1)
    time = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, mean_err, "r-", linewidth=1.5, label="mean |error|")
    ax.plot(time, max_err, "r--", linewidth=1.0, alpha=0.6, label="max |error|")
    ax.fill_between(time, 0, mean_err, color="red", alpha=0.1)
    ax.set_xlabel("step")
    ax.set_ylabel("|action error|")
    ax.set_title("Action Error Summary Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved error summary: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# METRICS REPORT
# ═══════════════════════════════════════════════════════════════════

def print_metrics(rollouts, expert_acts_on_learner=None, learner_acts=None):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC METRICS REPORT")
    print("=" * 70)

    for rd in rollouts:
        name = rd["name"].upper()
        print(f"\n--- {name} ---")
        print(f"  Steps completed:    {rd['steps']}")
        print(f"  Total return:       {rd['total_return']:.2f}")
        print(f"  Final dist to goal: {rd['final_dist']:.4f} m")
        print(f"  Out of bounds:      {rd['out_of_bounds']}")
        print(f"  Act min/mean/max:   {rd['acts'].min():.4f} / {rd['acts'].mean():.4f} / {rd['acts'].max():.4f}")
        print(f"  Act std per motor:  {rd['acts'].std(axis=0)}")
        print(f"  Ctrl min/mean/max:  {rd['ctrls'].min():.6f} / {rd['ctrls'].mean():.6f} / {rd['ctrls'].max():.6f}")

        # check for action saturation
        sat_low = (rd['acts'] <= -0.99).sum()
        sat_high = (rd['acts'] >= 0.99).sum()
        total = rd['acts'].size
        print(f"  Action saturation:  {sat_low} low (-1) + {sat_high} high (+1) / {total} total "
              f"({100*(sat_low+sat_high)/total:.1f}%)")

    if expert_acts_on_learner is not None and learner_acts is not None:
        T = min(learner_acts.shape[0], expert_acts_on_learner.shape[0])
        error = np.abs(learner_acts[:T] - expert_acts_on_learner[:T])

        print(f"\n--- ACTION ERROR (policy vs expert at learner-visited states) ---")
        print(f"  Timesteps compared: {T}")
        print(f"  MAE (overall):      {error.mean():.6f}")
        print(f"  MSE (overall):      {(error**2).mean():.6f}")
        print(f"  Max error:          {error.max():.6f}")
        print(f"  MAE per motor:      {error.mean(axis=0)}")

        # Error at key timesteps
        for t_check in [0, 1, 2, 5, 10, 20, 50]:
            if t_check < T:
                print(f"  MAE at t={t_check:3d}:       {error[t_check].mean():.6f}  "
                      f"(max motor err: {error[t_check].max():.6f})")

        # When does error first exceed thresholds?
        mean_err = error.mean(axis=1)
        for thresh in [0.05, 0.1, 0.2, 0.5]:
            exceed = np.where(mean_err > thresh)[0]
            t_exceed = int(exceed[0]) if len(exceed) > 0 else -1
            print(f"  Mean |err| > {thresh:.2f}:   first at t={t_exceed}" if t_exceed >= 0
                  else f"  Mean |err| > {thresh:.2f}:   never exceeded")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- build env ---
    env = PayloadGymEnv(
        xml_path=args.xml_path,
        template_yaml_path=args.template_yaml,
        max_steps=args.steps,
    )
    env.terminate_on_success = False  # let it run full horizon for diagnostics

    print(f"Environment: n_quads={env.n_quads}, n_bodies={env.n_bodies}")
    print(f"  state_dim={env.state_dim}, action_dim={env.action_dim}")
    print(f"  obs_space={env.observation_space.shape}, act_space={env.action_space.shape}")
    print(f"  u_nominal={env.u_nominal:.6f} N")

    # --- load learned policy ---
    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )

    print(f"\nLoading policy from: {args.policy_path}")
    state_dict = th.load(args.policy_path, map_location="cpu")
    missing, unexpected = bc_trainer.policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")
    policy = bc_trainer.policy
    policy.eval()

    def learner_fn(obs):
        act, _ = policy.predict(obs, deterministic=True)
        return np.asarray(act, dtype=np.float32).reshape(-1)

    # --- roll out learner ---
    print("\n>>> Rolling out LEARNED POLICY...")
    learner_rd = rollout(env, learner_fn, args.steps, name="policy")
    print(f"    Completed {learner_rd['steps']} steps, final dist={learner_rd['final_dist']:.4f}")

    rollouts = [learner_rd]
    expert_acts_on_learner = None

    # --- optionally roll out expert ---
    if args.run_expert:
        from expert_pcdbcbs import PcDbCBSExpert, PcDbCBSPaths

        paths = PcDbCBSPaths(
            bindings_path="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
            input_yaml=args.template_yaml,
            pc_dbcbs_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
            opt_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt_training.yaml",
            dynobench_base="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
            motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
            work_dir_root=str(out_dir / "_tmp_pcdbcbs"),
            keep_files=False,
            warmstart_optimization=True,
            N_opt=args.N_opt,
        )

        nu = env.action_dim
        act_low = 0.0 * np.ones(nu, dtype=np.float32)
        act_high = 1.4 * np.ones(nu, dtype=np.float32)

        expert = PcDbCBSExpert(
            paths=paths,
            act_low=act_low,
            act_high=act_high,
            replan_every_k=args.replan_every_k,
        )

        def expert_fn(obs):
            return expert.act(obs)

        # Expert rollout from same initial state
        print("\n>>> Rolling out EXPERT...")
        expert.reset_episode()
        expert_rd = rollout(env, expert_fn, args.steps, name="expert")
        print(f"    Completed {expert_rd['steps']} steps, final dist={expert_rd['final_dist']:.4f}")
        rollouts.append(expert_rd)

        # Expert labels on learner-visited states
        print("\n>>> Querying expert on learner-visited states...")
        expert.reset_episode()
        expert_acts_on_learner = expert_labels_on_learner_states(env, expert, learner_rd)
        print(f"    Got {expert_acts_on_learner.shape[0]} expert labels")

    # --- print metrics ---
    print_metrics(rollouts,
                  expert_acts_on_learner=expert_acts_on_learner,
                  learner_acts=learner_rd["acts"] if expert_acts_on_learner is not None else None)

    # --- save metrics to file ---
    metrics_path = out_dir / "metrics.txt"
    import io
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    print_metrics(rollouts,
                  expert_acts_on_learner=expert_acts_on_learner,
                  learner_acts=learner_rd["acts"] if expert_acts_on_learner is not None else None)
    sys.stdout = old_stdout
    with open(metrics_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nMetrics saved to: {metrics_path}")

    # --- generate plots ---
    print("\n>>> Generating plots...")

    plot_3d_trajectories(rollouts, env.n_bodies, env.goal, str(plots_dir / "traj_3d.png"))
    plot_positions_over_time(rollouts, env.n_bodies, env.goal, str(plots_dir / "positions.png"))
    plot_velocities_over_time(rollouts, env.n_bodies, str(plots_dir / "velocities.png"))
    plot_actions_over_time(rollouts, env.n_quads, str(plots_dir / "actions.png"))
    plot_ctrls_over_time(rollouts, env.n_quads, str(plots_dir / "ctrls.png"))
    plot_dist_and_reward(rollouts, str(plots_dir / "dist_reward.png"))

    if expert_acts_on_learner is not None:
        plot_action_error(learner_rd["acts"], expert_acts_on_learner, env.n_quads,
                          str(plots_dir / "action_error_per_motor.png"))
        plot_action_error_summary(learner_rd["acts"], expert_acts_on_learner,
                                  str(plots_dir / "action_error_summary.png"))

    # --- render videos ---
    if not args.no_video:
        print("\n>>> Rendering videos...")
        for rd in rollouts:
            name = rd["name"]
            video_dir = out_dir / f"videos_{name}"
            cfg = VideoConfig(
                out_dir=str(video_dir),
                fps=50,
                views=["diag", "side"],
                env_min=np.array([-2.5, -2.5, 0.0]),
                env_max=np.array([+2.5, +2.5, 1.5]),
            )
            u_traj_mujoco = rd["ctrls"]  # already in Newtons
            written = render_from_actions(
                xml_path=args.xml_path,
                init_and_actions_or_npz=(rd["qpos0"], rd["qvel0"], u_traj_mujoco),
                cfg=cfg,
            )
            print(f"    [{name}] videos: {written}")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
