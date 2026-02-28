#!/usr/bin/env python3
"""
eval_bc_on_training_data.py — Evaluate BC policy on the training distribution.

This script answers: "Did BC successfully learn to imitate the expert demonstrations?"

Usage:
    python eval_bc_on_training_data.py \
        --policy_path runs/bc_payload_v0/policy_state_dict.pt \
        --trajs_path runs/bc_payload_v0/round0_cache/expert_trajs.pkl \
        --out_dir runs/bc_eval_on_training
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imitation.algorithms import bc

# Add project scripts to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate BC policy on training data")
    p.add_argument("--policy_path", type=str, required=True,
                   help="Path to policy_state_dict.pt")
    p.add_argument("--trajs_path", type=str, required=True,
                   help="Path to expert_trajs.pkl (the training data)")
    p.add_argument("--xml_path", type=str,
                   default="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml")
    p.add_argument("--template_yaml", type=str,
                   default="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml")
    p.add_argument("--out_dir", type=str, default="runs/bc_eval_on_training",
                   help="Output directory for plots and metrics")
    return p.parse_args()


def load_training_data(trajs_path):
    """Load expert trajectories and extract (obs, acts) pairs."""
    print(f"Loading trajectories from: {trajs_path}")
    with open(trajs_path, "rb") as f:
        expert_trajs = pickle.load(f)
    
    print(f"Loaded {len(expert_trajs)} trajectories")
    
    # Extract data per trajectory (for per-trajectory analysis)
    traj_data = []
    for i, traj in enumerate(expert_trajs):
        T = traj.acts.shape[0]
        traj_data.append({
            "idx": i,
            "length": T,
            "obs": traj.obs[:T].astype(np.float32),  # (T, obs_dim)
            "acts": traj.acts.astype(np.float32),     # (T, act_dim)
            "terminal": traj.terminal,
        })
    
    # Concatenate all
    obs_all = np.concatenate([td["obs"] for td in traj_data], axis=0)
    acts_all = np.concatenate([td["acts"] for td in traj_data], axis=0)
    
    return traj_data, obs_all, acts_all


def evaluate_policy_on_training_data(policy, obs_all, acts_expert):
    """Query policy on all training observations and compute errors."""
    print(f"\nQuerying policy on {len(obs_all)} training observations...")
    
    # Query policy (batch)
    acts_policy, _ = policy.predict(obs_all, deterministic=True)
    acts_policy = np.asarray(acts_policy, dtype=np.float32)
    
    # Compute errors
    error = acts_policy - acts_expert
    abs_error = np.abs(error)
    sq_error = error ** 2
    
    results = {
        "n_samples": len(obs_all),
        "acts_policy": acts_policy,
        "acts_expert": acts_expert,
        "error": error,
        "abs_error": abs_error,
        
        # Summary statistics
        "mae": float(abs_error.mean()),
        "mse": float(sq_error.mean()),
        "rmse": float(np.sqrt(sq_error.mean())),
        "max_error": float(abs_error.max()),
        "mae_per_motor": abs_error.mean(axis=0),
        "std_per_motor": error.std(axis=0),
        
        # Ranges
        "expert_min": float(acts_expert.min()),
        "expert_max": float(acts_expert.max()),
        "expert_mean": float(acts_expert.mean()),
        "policy_min": float(acts_policy.min()),
        "policy_max": float(acts_policy.max()),
        "policy_mean": float(acts_policy.mean()),
        
        # Percentiles
        "p50_error": float(np.percentile(abs_error, 50)),
        "p90_error": float(np.percentile(abs_error, 90)),
        "p95_error": float(np.percentile(abs_error, 95)),
        "p99_error": float(np.percentile(abs_error, 99)),
    }
    
    # Per-motor correlation
    n_motors = acts_expert.shape[1]
    correlations = []
    for m in range(n_motors):
        corr = np.corrcoef(acts_expert[:, m], acts_policy[:, m])[0, 1]
        correlations.append(corr)
    results["correlations"] = np.array(correlations)
    
    return results


def evaluate_per_trajectory(policy, traj_data):
    """Evaluate policy on each trajectory separately."""
    per_traj_results = []
    
    for td in traj_data:
        obs = td["obs"]
        acts_expert = td["acts"]
        
        acts_policy, _ = policy.predict(obs, deterministic=True)
        acts_policy = np.asarray(acts_policy, dtype=np.float32)
        
        error = acts_policy - acts_expert
        abs_error = np.abs(error)
        
        # Error at different timesteps
        T = len(obs)
        t_checks = [0, 1, 5, 10, 20, 50, T-1]
        error_at_t = {}
        for t in t_checks:
            if t < T:
                error_at_t[t] = float(abs_error[t].mean())
        
        per_traj_results.append({
            "idx": td["idx"],
            "length": td["length"],
            "mae": float(abs_error.mean()),
            "max_error": float(abs_error.max()),
            "error_at_t": error_at_t,
            "acts_policy": acts_policy,
            "acts_expert": acts_expert,
        })
    
    return per_traj_results


def print_report(results, per_traj_results, traj_data):
    """Print comprehensive evaluation report."""
    print("\n" + "=" * 70)
    print("BC POLICY EVALUATION ON TRAINING DATA")
    print("=" * 70)
    
    print(f"\n--- DATASET SUMMARY ---")
    print(f"Total samples:     {results['n_samples']}")
    print(f"Num trajectories:  {len(traj_data)}")
    lengths = [td["length"] for td in traj_data]
    print(f"Traj lengths:      min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    print(f"\n--- OVERALL ERROR METRICS ---")
    print(f"MAE:               {results['mae']:.6f}")
    print(f"MSE:               {results['mse']:.6f}")
    print(f"RMSE:              {results['rmse']:.6f}")
    print(f"Max abs error:     {results['max_error']:.6f}")
    
    print(f"\n--- ERROR PERCENTILES ---")
    print(f"50th percentile:   {results['p50_error']:.6f}")
    print(f"90th percentile:   {results['p90_error']:.6f}")
    print(f"95th percentile:   {results['p95_error']:.6f}")
    print(f"99th percentile:   {results['p99_error']:.6f}")
    
    print(f"\n--- PER-MOTOR ANALYSIS ---")
    n_motors = len(results["mae_per_motor"])
    print(f"{'Motor':<8} {'MAE':<10} {'Std':<10} {'Corr':<10}")
    print("-" * 38)
    for m in range(n_motors):
        print(f"{m:<8} {results['mae_per_motor'][m]:<10.6f} "
              f"{results['std_per_motor'][m]:<10.6f} "
              f"{results['correlations'][m]:<10.4f}")
    
    print(f"\n--- ACTION RANGE COMPARISON ---")
    print(f"Expert actions:    [{results['expert_min']:.4f}, {results['expert_max']:.4f}], mean={results['expert_mean']:.4f}")
    print(f"Policy actions:    [{results['policy_min']:.4f}, {results['policy_max']:.4f}], mean={results['policy_mean']:.4f}")
    
    range_diff = (results['policy_max'] - results['policy_min']) - (results['expert_max'] - results['expert_min'])
    print(f"Range difference:  {range_diff:+.4f} (policy - expert)")
    
    print(f"\n--- PER-TRAJECTORY ERROR ---")
    print(f"{'Traj':<6} {'Length':<8} {'MAE':<10} {'MaxErr':<10} {'MAE@t=0':<10} {'MAE@t=end':<10}")
    print("-" * 54)
    for ptr in per_traj_results[:10]:  # Show first 10
        mae_t0 = ptr["error_at_t"].get(0, float("nan"))
        mae_end = ptr["error_at_t"].get(ptr["length"]-1, float("nan"))
        print(f"{ptr['idx']:<6} {ptr['length']:<8} {ptr['mae']:<10.6f} "
              f"{ptr['max_error']:<10.6f} {mae_t0:<10.6f} {mae_end:<10.6f}")
    
    if len(per_traj_results) > 10:
        print(f"... and {len(per_traj_results) - 10} more trajectories")
    
    # Summary statistics across trajectories
    traj_maes = [ptr["mae"] for ptr in per_traj_results]
    print(f"\nTrajectory MAE:    min={min(traj_maes):.6f}, max={max(traj_maes):.6f}, "
          f"mean={np.mean(traj_maes):.6f}, std={np.std(traj_maes):.6f}")
    
    print(f"\n--- DIAGNOSIS ---")
    if results["mae"] < 0.02:
        print("✓ EXCELLENT: MAE < 0.02 — BC learned the training data very well")
        print("  If rollout still fails, the problem is COVARIATE SHIFT")
    elif results["mae"] < 0.05:
        print("✓ GOOD: MAE < 0.05 — BC learned reasonably well")
        print("  Small errors may compound during rollout (covariate shift)")
    elif results["mae"] < 0.10:
        print("⚠ MODERATE: MAE 0.05-0.10 — BC partially learned the data")
        print("  Consider: more training epochs, larger network, or data filtering")
    else:
        print("❌ POOR: MAE > 0.10 — BC failed to fit training data")
        print("  Possible causes:")
        print("    - High action variance in data (multi-modal actions)")
        print("    - Insufficient training")
        print("    - Network capacity too small")
    
    mean_corr = results["correlations"].mean()
    if mean_corr > 0.95:
        print(f"✓ Correlation {mean_corr:.3f} — policy tracks expert well")
    elif mean_corr > 0.8:
        print(f"⚠ Correlation {mean_corr:.3f} — moderate tracking")
    else:
        print(f"❌ Correlation {mean_corr:.3f} — poor tracking, BC may not have learned the mapping")
    
    print("=" * 70)


def plot_scatter_expert_vs_policy(results, out_dir, n_motors):
    """Scatter plot of expert vs policy actions for each motor."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    acts_expert = results["acts_expert"]
    acts_policy = results["acts_policy"]
    
    for m in range(min(n_motors, 8)):
        ax = axes[m]
        ax.scatter(acts_expert[:, m], acts_policy[:, m], alpha=0.1, s=1)
        ax.plot([-1, 1], [-1, 1], 'r--', linewidth=1, label='ideal')
        ax.set_xlabel(f"Expert Motor {m}")
        ax.set_ylabel(f"Policy Motor {m}")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        corr = results["correlations"][m]
        ax.set_title(f"Motor {m} (corr={corr:.3f})")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Expert vs Policy Actions (each point = one training sample)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_expert_vs_policy.png", dpi=150)
    plt.close()
    print(f"[plot] saved scatter: {out_dir / 'scatter_expert_vs_policy.png'}")


def plot_error_histogram(results, out_dir):
    """Histogram of action errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall error histogram
    ax = axes[0]
    abs_error = results["abs_error"].flatten()
    ax.hist(abs_error, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(results["mae"], color='r', linestyle='--', label=f'MAE={results["mae"]:.4f}')
    ax.axvline(results["p90_error"], color='orange', linestyle='--', label=f'P90={results["p90_error"]:.4f}')
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution (all motors)")
    ax.legend()
    ax.set_xlim(0, min(1.0, abs_error.max() * 1.1))
    
    # Per-motor MAE bar chart
    ax = axes[1]
    n_motors = len(results["mae_per_motor"])
    x = np.arange(n_motors)
    ax.bar(x, results["mae_per_motor"], edgecolor='black', alpha=0.7)
    ax.axhline(results["mae"], color='r', linestyle='--', label=f'Overall MAE={results["mae"]:.4f}')
    ax.set_xlabel("Motor")
    ax.set_ylabel("MAE")
    ax.set_title("MAE per Motor")
    ax.set_xticks(x)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "error_histogram.png", dpi=150)
    plt.close()
    print(f"[plot] saved histogram: {out_dir / 'error_histogram.png'}")


def plot_error_vs_action_magnitude(results, out_dir):
    """Plot error vs expert action magnitude to see if errors are larger for extreme actions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    acts_expert = results["acts_expert"].flatten()
    abs_error = results["abs_error"].flatten()
    
    # Bin by expert action magnitude
    bins = np.linspace(-1, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(acts_expert, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    bin_means = []
    bin_stds = []
    bin_counts = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(abs_error[mask].mean())
            bin_stds.append(abs_error[mask].std())
            bin_counts.append(mask.sum())
        else:
            bin_means.append(0)
            bin_stds.append(0)
            bin_counts.append(0)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    ax.bar(bin_centers, bin_means, width=0.08, alpha=0.7, edgecolor='black')
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none', color='black', capsize=2)
    ax.set_xlabel("Expert Action Value")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Error vs Expert Action Magnitude\n(Are extreme actions harder to predict?)")
    ax.axhline(results["mae"], color='r', linestyle='--', label=f'Overall MAE={results["mae"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_action_magnitude.png", dpi=150)
    plt.close()
    print(f"[plot] saved error vs magnitude: {out_dir / 'error_vs_action_magnitude.png'}")


def plot_trajectory_comparison(per_traj_results, out_dir, n_trajs=3):
    """Plot policy vs expert actions for a few trajectories."""
    fig, axes = plt.subplots(n_trajs, 2, figsize=(14, 4 * n_trajs))
    for i, ptr in enumerate(per_traj_results[:n_trajs]):
        print(i)
        acts_expert = ptr["acts_expert"]
        acts_policy = ptr["acts_policy"]
        T = ptr["length"]
        
        # Left: all motors
        ax = axes[i, 0]
        for m in range(acts_expert.shape[1]):
            ax.plot(acts_expert[:, m], '--', alpha=0.5, label=f'exp m{m}' if i == 0 else None)
            ax.plot(acts_policy[:, m], '-', alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Action")
        ax.set_title(f"Trajectory {ptr['idx']} (len={T}, MAE={ptr['mae']:.4f})")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Right: error over time
        ax = axes[i, 1]
        error_over_time = np.abs(acts_policy - acts_expert).mean(axis=1)
        ax.plot(error_over_time, 'b-', linewidth=1.5)
        ax.fill_between(range(T), 0, error_over_time, alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("MAE")
        ax.set_title(f"Error over time (Traj {ptr['idx']})")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Policy vs Expert Actions on Training Trajectories\n(dashed=expert, solid=policy)", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_comparison.png", dpi=150)
    plt.close()
    print(f"[plot] saved trajectory comparison: {out_dir / 'trajectory_comparison.png'}")


def plot_correlation_matrix(results, out_dir):
    """Plot correlation between expert and policy for each motor."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_motors = len(results["correlations"])
    x = np.arange(n_motors)
    colors = ['green' if c > 0.9 else 'orange' if c > 0.7 else 'red' for c in results["correlations"]]
    
    bars = ax.bar(x, results["correlations"], color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='Good (0.9)')
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Moderate (0.7)')
    
    ax.set_xlabel("Motor")
    ax.set_ylabel("Correlation (Expert vs Policy)")
    ax.set_title("Per-Motor Correlation")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add correlation values on bars
    for bar, corr in zip(bars, results["correlations"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_per_motor.png", dpi=150)
    plt.close()
    print(f"[plot] saved correlation: {out_dir / 'correlation_per_motor.png'}")


def save_report(results, per_traj_results, traj_data, out_dir):
    """Save text report to file."""
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    print_report(results, per_traj_results, traj_data)
    sys.stdout = old_stdout
    
    report_path = out_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nReport saved to: {report_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid issues if env not needed
    from payload_env import PayloadGymEnv
    
    # Create env just to get observation/action spaces
    env = PayloadGymEnv(
        xml_path=args.xml_path,
        template_yaml_path=args.template_yaml,
        max_steps=200,
    )
    
    print(f"Environment: obs_dim={env.observation_space.shape[0]}, act_dim={env.action_space.shape[0]}")
    
    # Load policy
    print(f"\nLoading policy from: {args.policy_path}")
    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )
    state_dict = th.load(args.policy_path, map_location="cpu")
    bc_trainer.policy.load_state_dict(state_dict)
    policy = bc_trainer.policy
    policy.eval()
    
    # Load training data
    traj_data, obs_all, acts_all = load_training_data(args.trajs_path)
    print(f"Total training samples: {len(obs_all)}")
    print(f"Observation shape: {obs_all.shape}")
    print(f"Action shape: {acts_all.shape}")
    
    # Verify dimensions match
    assert obs_all.shape[1] == env.observation_space.shape[0], \
        f"Obs dim mismatch: data={obs_all.shape[1]}, env={env.observation_space.shape[0]}"
    assert acts_all.shape[1] == env.action_space.shape[0], \
        f"Act dim mismatch: data={acts_all.shape[1]}, env={env.action_space.shape[0]}"
    
    # Evaluate on all training data
    results = evaluate_policy_on_training_data(policy, obs_all, acts_all)
    
    # Evaluate per trajectory
    per_traj_results = evaluate_per_trajectory(policy, traj_data)
    
    # Print report
    print_report(results, per_traj_results, traj_data)
    
    # Save report
    save_report(results, per_traj_results, traj_data, out_dir)
    
    # Generate plots
    print("\n>>> Generating plots...")
    n_motors = acts_all.shape[1]
    plot_scatter_expert_vs_policy(results, out_dir, n_motors)
    plot_error_histogram(results, out_dir)
    plot_error_vs_action_magnitude(results, out_dir)
    plot_trajectory_comparison(per_traj_results, out_dir, n_trajs=min(3, len(traj_data)))
    plot_correlation_matrix(results, out_dir)
    
    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()