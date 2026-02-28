# test_policy_range.py
"""
Diagnostic: Check if BC policy can output full action range [-1, 1]
"""

import numpy as np
import torch as th
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "scripts"))
from payload_env import PayloadGymEnv
from imitation.algorithms import bc


def main():
    # --- Setup ---
    xml_path = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    template_yaml = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
    policy_path = Path("runs/dagger_payload/dagger_policy.pt")

    env = PayloadGymEnv(xml_path=xml_path, template_yaml_path=template_yaml, max_steps=200)
    
    print("=" * 60)
    print("POLICY OUTPUT RANGE DIAGNOSTIC")
    print("=" * 60)
    print(f"obs_dim: {env.observation_space.shape[0]}")
    print(f"act_dim: {env.action_space.shape[0]}")
    print(f"n_bodies: {env.n_bodies}, n_quads: {env.n_quads}")
    
    # --- Load Policy ---
    print(f"\nLoading policy from: {policy_path}")
    dummy_bc = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=np.random.default_rng(0),
    )
    dummy_bc.policy.load_state_dict(th.load(policy_path, map_location="cpu"))
    policy = dummy_bc.policy
    policy.eval()
    
    # --- Get baseline observation ---
    obs_baseline, _ = env.reset()
    state_dim = env.state_dim
    vel_offset = 7 * env.n_bodies  # velocities start after poses
    
    print(f"\nBaseline state (first 20 values): {obs_baseline[:20]}")
    
    # --- Test suite ---
    tests = []
    
    # Test 1: Baseline
    tests.append(("Baseline (start state)", obs_baseline.copy()))
    
    # Test 2: Payload falling fast
    obs = obs_baseline.copy()
    obs[vel_offset + 2] = -3.0  # vz = -3 m/s
    tests.append(("Payload falling (vz=-3)", obs))
    
    # Test 3: Payload rising fast
    obs = obs_baseline.copy()
    obs[vel_offset + 2] = +3.0  # vz = +3 m/s
    tests.append(("Payload rising (vz=+3)", obs))
    
    # Test 4: Payload far left (needs to go right)
    obs = obs_baseline.copy()
    obs[0] = -3.0  # x = -3
    tests.append(("Payload at x=-3", obs))
    
    # Test 5: Payload at goal
    obs = obs_baseline.copy()
    obs[0] = 0.0  # x = 0 (goal)
    tests.append(("Payload at x=0 (goal)", obs))
    
    # Test 6: Payload past goal
    obs = obs_baseline.copy()
    obs[0] = +1.0  # x = +1 (overshot)
    tests.append(("Payload at x=+1 (overshot)", obs))
    
    # Test 7: High velocity toward goal
    obs = obs_baseline.copy()
    obs[vel_offset] = +3.0  # vx = +3 m/s
    tests.append(("High vx=+3 (toward goal)", obs))
    
    # Test 8: Combined perturbation
    obs = obs_baseline.copy()
    obs[0] = +0.5  # overshot
    obs[vel_offset] = +2.0  # still moving toward goal
    obs[vel_offset + 2] = -1.0  # and falling
    tests.append(("Overshot + fast + falling", obs))
    
    # --- Run tests ---
    print("\n" + "-" * 60)
    print("PERTURBATION TESTS")
    print("-" * 60)
    
    all_actions = []
    for name, obs in tests:
        act, _ = policy.predict(obs[None].astype(np.float32), deterministic=True)
        act = act.flatten()
        all_actions.append(act)
        print(f"\n{name}:")
        print(f"  Action: [{act.min():+.4f}, {act.max():+.4f}], mean={act.mean():+.4f}")
    
    # --- Random stress test ---
    print("\n" + "-" * 60)
    print("RANDOM OBSERVATION STRESS TEST")
    print("-" * 60)
    
    n_random = 1000
    random_obs = np.random.uniform(-5, 5, size=(n_random, env.observation_space.shape[0])).astype(np.float32)
    random_acts, _ = policy.predict(random_obs, deterministic=True)
    
    print(f"Across {n_random} random observations:")
    print(f"  Action min:  {random_acts.min():+.4f}")
    print(f"  Action max:  {random_acts.max():+.4f}")
    print(f"  Action mean: {random_acts.mean():+.4f}")
    print(f"  Action std:  {random_acts.std():.4f}")
    
    # Per-motor analysis
    print("\nPer-motor statistics:")
    for i in range(env.action_dim):
        motor_acts = random_acts[:, i]
        print(f"  Motor {i}: [{motor_acts.min():+.4f}, {motor_acts.max():+.4f}], std={motor_acts.std():.4f}")
    
    # --- Diagnosis ---
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    action_range = random_acts.max() - random_acts.min()
    action_std = random_acts.std()
    
    print(f"Total action range: {action_range:.3f} (expected: ~2.0 for [-1, 1])")
    print(f"Action std: {action_std:.3f}")
    
    if action_range < 0.3:
        print("\n❌ SEVERE: Policy output nearly constant regardless of input")
        print("   The policy has NOT learned state-dependent behavior")
        print("   Root cause: Training data has zero state diversity")
    elif action_range < 1.0:
        print("\n⚠️  WARNING: Policy output range is narrow")
        print("   The policy responds weakly to state changes")
        print("   Root cause: Training data covers limited state space")
    elif action_range < 1.5:
        print("\n⚠️  MODERATE: Policy uses partial action range")
        print("   May work for some states but not extreme corrections")
    else:
        print("\n✓ GOOD: Policy uses wide action range")
    
    # Check if all test actions are similar
    all_actions = np.array(all_actions)
    test_variance = all_actions.var()
    print(f"\nVariance across perturbation tests: {test_variance:.6f}")
    if test_variance < 0.001:
        print("❌ Policy outputs nearly identical actions for all test states!")


if __name__ == "__main__":
    main()