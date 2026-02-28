#!/usr/bin/env python3
"""
analyze_expert_trajs.py — Check trajectory diversity in cached expert demonstrations.

Observation layout (from PayloadGymEnv._get_obs):
  obs = [state (state_dim), learner_features (learner_obs_dim)]

  state_dim = 13 * n_bodies  (poses-first layout):
    poses block [0 : 7*n_bodies]:
      payload: pos(3) + quat_xyzw(4)  -> obs[0:7]
      quad i:  pos(3) + quat_xyzw(4)  -> obs[7*(1+i) : 7*(2+i)]
    vels block [7*n_bodies : 13*n_bodies]:
      payload: linvel(3) + angvel(3)   -> obs[7*n : 7*n+6]
      quad i:  linvel(3) + angvel(3)   -> obs[7*n + 6*(1+i) : 7*n + 6*(2+i)]

  learner_obs_dim = 6 + n_quads*12 + action_dim:
    payload_pos_err(3) + payload_linvel(3)         [feats+0 : feats+6]
    per quad i (12 each):
      e_p_rel(3) + e_v_rel(3) + e_q_vec(3) + e_w(3)
    prev_action(action_dim)                        [feats+6+n_quads*12 : end]

Usage:
    python analyze_expert_trajs.py --trajs_pkl runs/bc_payload/round0_cache/expert_trajs.pkl
    python analyze_expert_trajs.py --trajs_pkl runs/bc_payload/round0_cache/expert_trajs.pkl --out_dir runs/traj_analysis
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajs_pkl", type=str, required=True, help="Path to expert_trajs.pkl")
    p.add_argument("--out_dir", type=str, default="runs/traj_analysis")
    p.add_argument("--n_bodies", type=int, default=3, help="1 payload + N quads")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.trajs_pkl, "rb") as f:
        trajs = pickle.load(f)

    n_eps = len(trajs)
    print(f"Loaded {n_eps} trajectories")

    # ── Basic stats ──
    lengths = [t.acts.shape[0] for t in trajs]
    obs_dim = trajs[0].obs.shape[1]
    act_dim = trajs[0].acts.shape[1]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  Episode lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")

    n_bodies = args.n_bodies
    n_quads = n_bodies - 1
    state_dim = 13 * n_bodies
    feats_start = state_dim  # learner features start here

    # Derived obs slice helpers (all index into a single obs row)
    def payload_pos_slice():
        return slice(0, 3)

    def payload_quat_slice():
        return slice(3, 7)

    def quad_pos_slice(qi):
        base = 7 * (1 + qi)
        return slice(base, base + 3)

    def payload_linvel_slice():
        base = 7 * n_bodies
        return slice(base, base + 3)

    def payload_angvel_slice():
        base = 7 * n_bodies + 3
        return slice(base, base + 3)

    def quad_linvel_slice(qi):
        base = 7 * n_bodies + 6 * (1 + qi)
        return slice(base, base + 3)

    # Learner feature slices (relative to feats_start)
    # payload_pos_err(3) + payload_linvel(3)
    def feat_payload_pos_err_slice():
        return slice(feats_start, feats_start + 3)

    def feat_payload_linvel_slice():
        return slice(feats_start + 3, feats_start + 6)

    # per-quad: each block is 12 wide
    def feat_quad_e_p_rel_slice(qi):
        b = feats_start + 6 + 12 * qi
        return slice(b, b + 3)

    def feat_quad_e_v_rel_slice(qi):
        b = feats_start + 6 + 12 * qi + 3
        return slice(b, b + 3)

    def feat_quad_e_q_vec_slice(qi):
        b = feats_start + 6 + 12 * qi + 6
        return slice(b, b + 3)

    def feat_quad_e_w_slice(qi):
        b = feats_start + 6 + 12 * qi + 9
        return slice(b, b + 3)

    def feat_prev_action_slice():
        b = feats_start + 6 + 12 * n_quads
        return slice(b, b + act_dim)

    min_len = min(lengths)
    time = np.arange(min_len)

    # ══════════════════════════════════════════════════════════════
    # 1. PAYLOAD POSITION TRAJECTORIES
    # ══════════════════════════════════════════════════════════════
    axis_names = ["x", "y", "z"]
    payload_pos_all = np.stack(
        [t.obs[:min_len, payload_pos_slice()] for t in trajs], axis=0
    ).astype(np.float32)  # (n_eps, min_len, 3)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    mean_pos = payload_pos_all.mean(axis=0)
    std_pos = payload_pos_all.std(axis=0)
    for ax_i in range(3):
        for i in range(n_eps):
            axs[ax_i].plot(payload_pos_all[i, :, ax_i], alpha=0.15, linewidth=0.5, color="tab:blue")
        axs[ax_i].plot(time, mean_pos[:, ax_i], "r-", linewidth=2, label="mean")
        axs[ax_i].fill_between(time,
                                mean_pos[:, ax_i] - 2 * std_pos[:, ax_i],
                                mean_pos[:, ax_i] + 2 * std_pos[:, ax_i],
                                color="red", alpha=0.15, label="±2σ")
        axs[ax_i].set_xlabel("step")
        axs[ax_i].set_ylabel(f"payload {axis_names[ax_i]} [m]")
        axs[ax_i].legend(fontsize=7)
        axs[ax_i].grid(True, alpha=0.3)
    fig.suptitle(f"Payload Position Across {n_eps} Expert Trajectories", fontsize=12)
    plt.tight_layout()
    path = out_dir / "payload_pos_diversity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 2. QUAD POSITIONS FROM STATE
    # ══════════════════════════════════════════════════════════════
    fig, axs = plt.subplots(n_quads, 3, figsize=(15, 4 * n_quads), sharex=True)
    if n_quads == 1:
        axs = axs[np.newaxis, :]

    for qi in range(n_quads):
        quad_pos_all = np.stack(
            [t.obs[:min_len, quad_pos_slice(qi)] for t in trajs], axis=0
        ).astype(np.float32)  # (n_eps, min_len, 3)
        mean_qp = quad_pos_all.mean(axis=0)
        std_qp = quad_pos_all.std(axis=0)
        for ax_i in range(3):
            for i in range(n_eps):
                axs[qi, ax_i].plot(quad_pos_all[i, :, ax_i], alpha=0.15, linewidth=0.5, color="tab:orange")
            axs[qi, ax_i].plot(time, mean_qp[:, ax_i], "r-", linewidth=2, label="mean")
            axs[qi, ax_i].fill_between(time,
                                        mean_qp[:, ax_i] - 2 * std_qp[:, ax_i],
                                        mean_qp[:, ax_i] + 2 * std_qp[:, ax_i],
                                        color="red", alpha=0.15, label="±2σ")
            axs[qi, ax_i].set_ylabel(f"Q{qi+1} {axis_names[ax_i]} [m]")
            axs[qi, ax_i].legend(fontsize=7)
            axs[qi, ax_i].grid(True, alpha=0.3)
    for ax_i in range(3):
        axs[-1, ax_i].set_xlabel("step")
    fig.suptitle(f"Quad Positions Across {n_eps} Expert Trajectories", fontsize=12)
    plt.tight_layout()
    path = out_dir / "quad_pos_diversity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 3. PAYLOAD VELOCITY (from state vels block)
    # ══════════════════════════════════════════════════════════════
    payload_linvel_all = np.stack(
        [t.obs[:min_len, payload_linvel_slice()] for t in trajs], axis=0
    ).astype(np.float32)  # (n_eps, min_len, 3)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    mean_lv = payload_linvel_all.mean(axis=0)
    std_lv = payload_linvel_all.std(axis=0)
    for ax_i in range(3):
        for i in range(n_eps):
            axs[ax_i].plot(payload_linvel_all[i, :, ax_i], alpha=0.15, linewidth=0.5, color="tab:green")
        axs[ax_i].plot(time, mean_lv[:, ax_i], "r-", linewidth=2, label="mean")
        axs[ax_i].fill_between(time,
                                mean_lv[:, ax_i] - 2 * std_lv[:, ax_i],
                                mean_lv[:, ax_i] + 2 * std_lv[:, ax_i],
                                color="red", alpha=0.15, label="±2σ")
        axs[ax_i].set_xlabel("step")
        axs[ax_i].set_ylabel(f"payload v{axis_names[ax_i]} [m/s]")
        axs[ax_i].legend(fontsize=7)
        axs[ax_i].grid(True, alpha=0.3)
    fig.suptitle(f"Payload Linear Velocity Across {n_eps} Expert Trajectories", fontsize=12)
    plt.tight_layout()
    path = out_dir / "payload_linvel_diversity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 4. LEARNER FEATURES — payload position error + linvel
    # ══════════════════════════════════════════════════════════════
    payload_pos_err_all = np.stack(
        [t.obs[:min_len, feat_payload_pos_err_slice()] for t in trajs], axis=0
    ).astype(np.float32)
    payload_feat_linvel_all = np.stack(
        [t.obs[:min_len, feat_payload_linvel_slice()] for t in trajs], axis=0
    ).astype(np.float32)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax_i in range(3):
        arr = payload_pos_err_all
        m, s = arr.mean(axis=0), arr.std(axis=0)
        for i in range(n_eps):
            axs[0, ax_i].plot(arr[i, :, ax_i], alpha=0.15, linewidth=0.5, color="tab:purple")
        axs[0, ax_i].plot(time, m[:, ax_i], "r-", linewidth=2, label="mean")
        axs[0, ax_i].fill_between(time, m[:, ax_i] - 2*s[:, ax_i], m[:, ax_i] + 2*s[:, ax_i],
                                   color="red", alpha=0.15, label="±2σ")
        axs[0, ax_i].set_ylabel(f"pos_err {axis_names[ax_i]} [m]")
        axs[0, ax_i].legend(fontsize=7)
        axs[0, ax_i].grid(True, alpha=0.3)
        axs[0, ax_i].axhline(0, color="k", linewidth=0.5)

        arr = payload_feat_linvel_all
        m, s = arr.mean(axis=0), arr.std(axis=0)
        for i in range(n_eps):
            axs[1, ax_i].plot(arr[i, :, ax_i], alpha=0.15, linewidth=0.5, color="tab:green")
        axs[1, ax_i].plot(time, m[:, ax_i], "r-", linewidth=2, label="mean")
        axs[1, ax_i].fill_between(time, m[:, ax_i] - 2*s[:, ax_i], m[:, ax_i] + 2*s[:, ax_i],
                                   color="red", alpha=0.15, label="±2σ")
        axs[1, ax_i].set_ylabel(f"linvel {axis_names[ax_i]} [m/s]")
        axs[1, ax_i].set_xlabel("step")
        axs[1, ax_i].legend(fontsize=7)
        axs[1, ax_i].grid(True, alpha=0.3)
        axs[1, ax_i].axhline(0, color="k", linewidth=0.5)

    axs[0, 0].set_title("Payload pos error (pL - pL_goal)")
    axs[1, 0].set_title("Payload linear velocity")
    fig.suptitle(f"Learner Features: Payload — {n_eps} Trajectories", fontsize=12)
    plt.tight_layout()
    path = out_dir / "learner_feat_payload.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 5. LEARNER FEATURES — per-quad errors (e_p_rel, e_v_rel, e_q_vec, e_w)
    # ══════════════════════════════════════════════════════════════
    feat_labels = [
        ("e_p_rel", feat_quad_e_p_rel_slice, "tab:blue",   "[m]"),
        ("e_v_rel", feat_quad_e_v_rel_slice, "tab:orange", "[m/s]"),
        ("e_q_vec", feat_quad_e_q_vec_slice, "tab:red",    "[-]"),
        ("e_w",     feat_quad_e_w_slice,     "tab:green",  "[rad/s]"),
    ]
    for qi in range(n_quads):
        fig, axs = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
        for row, (label, slice_fn, color, unit) in enumerate(feat_labels):
            arr = np.stack(
                [t.obs[:min_len, slice_fn(qi)] for t in trajs], axis=0
            ).astype(np.float32)
            m, s = arr.mean(axis=0), arr.std(axis=0)
            for ax_i in range(3):
                for i in range(n_eps):
                    axs[row, ax_i].plot(arr[i, :, ax_i], alpha=0.12, linewidth=0.4, color=color)
                axs[row, ax_i].plot(time, m[:, ax_i], "k-", linewidth=1.8, label="mean")
                axs[row, ax_i].fill_between(time, m[:, ax_i] - 2*s[:, ax_i], m[:, ax_i] + 2*s[:, ax_i],
                                             color="red", alpha=0.15, label="±2σ")
                axs[row, ax_i].set_ylabel(f"{label}_{axis_names[ax_i]} {unit}")
                axs[row, ax_i].axhline(0, color="k", linewidth=0.5, linestyle="--")
                axs[row, ax_i].grid(True, alpha=0.3)
                if row == 0:
                    axs[row, ax_i].legend(fontsize=7)
        for ax_i in range(3):
            axs[-1, ax_i].set_xlabel("step")
        fig.suptitle(f"Learner Features: Quad {qi+1} Errors — {n_eps} Trajectories", fontsize=12)
        plt.tight_layout()
        path = out_dir / f"learner_feat_quad{qi+1}_errors.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 6. PREVIOUS ACTION (learner feature)
    # ══════════════════════════════════════════════════════════════
    prev_act_all = np.stack(
        [t.obs[:min_len, feat_prev_action_slice()] for t in trajs], axis=0
    ).astype(np.float32)  # (n_eps, min_len, act_dim)

    fig, axs = plt.subplots(n_quads, 4, figsize=(16, 3.5 * n_quads), sharex=True, sharey=True)
    if n_quads == 1:
        axs = axs[np.newaxis, :]
    m_pa = prev_act_all.mean(axis=0)
    s_pa = prev_act_all.std(axis=0)
    for q in range(n_quads):
        for mi in range(4):
            idx = 4 * q + mi
            for i in range(n_eps):
                axs[q, mi].plot(prev_act_all[i, :, idx], alpha=0.08, linewidth=0.3, color="tab:blue")
            axs[q, mi].plot(time, m_pa[:, idx], "r-", linewidth=1.5)
            axs[q, mi].fill_between(time, m_pa[:, idx] - 2*s_pa[:, idx], m_pa[:, idx] + 2*s_pa[:, idx],
                                     color="red", alpha=0.15)
            axs[q, mi].set_ylabel(f"Q{q+1}M{mi+1}")
            axs[q, mi].set_ylim(-1.15, 1.15)
            axs[q, mi].axhline(0, color="k", linewidth=0.5, linestyle="--")
            axs[q, mi].grid(True, alpha=0.3)
    for mi in range(4):
        axs[-1, mi].set_xlabel("step")
    fig.suptitle(f"Prev Action (obs feature) Across {n_eps} Episodes", fontsize=12)
    plt.tight_layout()
    path = out_dir / "prev_action_feature.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 7. ACTION TRAJECTORIES (per-motor spread)
    # ══════════════════════════════════════════════════════════════
    acts_all = np.zeros((n_eps, min_len, act_dim), dtype=np.float32)
    for i, t in enumerate(trajs):
        acts_all[i] = t.acts[:min_len]

    fig, axs = plt.subplots(n_quads, 4, figsize=(16, 3.5 * n_quads), sharex=True, sharey=True)
    if n_quads == 1:
        axs = axs[np.newaxis, :]

    mean_acts = acts_all.mean(axis=0)
    std_acts = acts_all.std(axis=0)

    for q in range(n_quads):
        for m in range(4):
            mi = 4 * q + m
            for i in range(n_eps):
                axs[q, m].plot(acts_all[i, :, mi], alpha=0.08, linewidth=0.3, color="tab:blue")
            axs[q, m].plot(time, mean_acts[:, mi], "r-", linewidth=1.5)
            axs[q, m].fill_between(time,
                                    mean_acts[:, mi] - 2 * std_acts[:, mi],
                                    mean_acts[:, mi] + 2 * std_acts[:, mi],
                                    color="red", alpha=0.15)
            axs[q, m].set_ylabel(f"Q{q+1}M{m+1}")
            axs[q, m].set_ylim(-1.15, 1.15)
            axs[q, m].grid(True, alpha=0.3)
    for m in range(4):
        axs[-1, m].set_xlabel("step")

    fig.suptitle(f"Action Trajectories Across {n_eps} Episodes (mean ± 2σ in red)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "action_diversity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 8. PER-TIMESTEP ACTION STD
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 4))
    mean_std_per_t = std_acts.mean(axis=1)
    max_std_per_t = std_acts.max(axis=1)
    ax.plot(time, mean_std_per_t, "r-", linewidth=1.5, label="mean σ (across motors)")
    ax.plot(time, max_std_per_t, "r--", linewidth=1.0, alpha=0.6, label="max σ")
    ax.set_xlabel("step")
    ax.set_ylabel("action σ across episodes")
    ax.set_title("Action Standard Deviation Over Time (lower = more consistent)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "action_std_over_time.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 9. PAIRWISE TRAJECTORY DISTANCE
    # ══════════════════════════════════════════════════════════════
    n_pairs = min(500, n_eps * (n_eps - 1) // 2)
    print(f"Computing pairwise action distances for {n_pairs} random pairs of trajectories...")
    rng = np.random.default_rng(42)
    pair_dists = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_eps, size=2, replace=False)
        d = np.sqrt(((acts_all[i] - acts_all[j]) ** 2).sum(axis=1))
        pair_dists.append(d)
    pair_dists = np.array(pair_dists)

    fig, ax = plt.subplots(figsize=(10, 4))
    mean_pair = pair_dists.mean(axis=0)
    p25 = np.percentile(pair_dists, 25, axis=0)
    p75 = np.percentile(pair_dists, 75, axis=0)
    ax.plot(time, mean_pair, "b-", linewidth=1.5, label="mean pairwise L2")
    ax.fill_between(time, p25, p75, color="blue", alpha=0.15, label="25-75th percentile")
    ax.set_xlabel("step")
    ax.set_ylabel("pairwise action L2 distance")
    ax.set_title("Pairwise Action Distance Over Time (higher = more diverse = harder for BC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "pairwise_action_distance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")

    # ══════════════════════════════════════════════════════════════
    # 10. NUMERIC SUMMARY
    # ══════════════════════════════════════════════════════════════
    report = []
    report.append("=" * 60)
    report.append("EXPERT TRAJECTORY DIVERSITY REPORT")
    report.append("=" * 60)
    report.append(f"Episodes: {n_eps}")
    report.append(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    report.append(f"n_bodies={n_bodies}, n_quads={n_quads}, state_dim={state_dim}")
    report.append(f"obs_dim={obs_dim}, act_dim={act_dim}")
    report.append(f"learner_obs_dim (inferred)={obs_dim - state_dim}")
    report.append("")

    report.append("── Obs layout sanity check ──")
    report.append(f"  expected learner_obs_dim = 6 + {n_quads}*12 + {act_dim} = {6 + n_quads*12 + act_dim}")
    report.append(f"  actual   learner_obs_dim = {obs_dim - state_dim}")
    if obs_dim - state_dim != 6 + n_quads * 12 + act_dim:
        report.append("  WARNING: mismatch — check --n_bodies argument!")
    report.append("")

    report.append("── Payload Position Spread (σ across episodes) ──")
    for ax_i, name in enumerate(axis_names):
        s = std_pos[:, ax_i]
        report.append(f"  {name}: mean_σ={s.mean():.4f}, max_σ={s.max():.4f} "
                       f"(at t={s.argmax()})")

    report.append("")
    report.append("── Payload Position Error Spread (learner feat) ──")
    for ax_i, name in enumerate(axis_names):
        s = payload_pos_err_all.std(axis=0)[:, ax_i]
        report.append(f"  {name}: mean_σ={s.mean():.4f}, max_σ={s.max():.4f}")

    report.append("")
    report.append("── Action Spread (σ across episodes) ──")
    report.append(f"  Overall mean σ: {std_acts.mean():.4f}")
    report.append(f"  Overall max σ:  {std_acts.max():.4f} "
                   f"(motor {std_acts.max(axis=0).argmax()}, t={std_acts.max(axis=1).argmax()})")
    report.append(f"  Mean σ at t=0:   {std_acts[0].mean():.4f}")
    report.append(f"  Mean σ at t=10:  {std_acts[min(10, min_len-1)].mean():.4f}")
    report.append(f"  Mean σ at t=50:  {std_acts[min(50, min_len-1)].mean():.4f}")
    report.append(f"  Mean σ at t=100: {std_acts[min(100, min_len-1)].mean():.4f}")
    report.append(f"  Mean σ at t=end: {std_acts[-1].mean():.4f}")

    report.append("")
    report.append("── Pairwise Action Distance ──")
    report.append(f"  Mean pairwise L2: {mean_pair.mean():.4f}")
    report.append(f"  Max pairwise L2:  {mean_pair.max():.4f} (at t={mean_pair.argmax()})")

    report.append("")
    report.append("── Interpretation Guide ──")
    report.append("  σ < 0.05: Very consistent — BC should learn this well")
    report.append("  σ 0.05-0.15: Moderate spread — BC will average, likely OK")
    report.append("  σ > 0.15: High diversity — BC will average, may hurt stability")
    report.append("  σ > 0.3: Multi-modal — consider filtering trajectories by cost")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    report_path = out_dir / "diversity_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    print(f"All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
