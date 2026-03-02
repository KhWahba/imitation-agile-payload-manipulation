# train_bc.py

import os
import tempfile
import pickle
from pathlib import Path
from typing import Any, Tuple, List

import numpy as np
import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper

# your env + expert
from scripts.payload_env_old import PayloadGymEnv
from expert_pcdbcbs import PcDbCBSExpert, PcDbCBSPaths, ExpertPolicySB3
from imitation.data import rollout

# your video logger (the file you pasted)
from videos_from_log import VideoConfig, render_from_actions


def _unwrap_env(e: Any) -> Any:
    """
    Unwrap nested wrappers until we reach the base env (PayloadGymEnv).
    Works for RolloutInfoWrapper and other gym-style wrappers that expose `.env`.
    """
    cur = e
    for _ in range(20):
        if hasattr(cur, "env"):
            cur = cur.env
        elif hasattr(cur, "unwrapped"):
            nxt = cur.unwrapped
            if nxt is cur:
                break
            cur = nxt
        else:
            break
    return cur


def collect_expert_trajs_and_logs(
    env_fn,
    expert: PcDbCBSExpert,
    n_episodes: int,
    *,
    xml_path: str,
    logs_dir: str,
    video_root_dir: str,
    render_videos: bool = True,
) -> Tuple[List[Trajectory], List[Path]]:
    """
    Collect n_episodes expert rollouts and:
      - return imitation Trajectory objects (for SimpleDAggerTrainer expert_trajs seed)
      - save per-episode .npz logs (qpos0, qvel0, u_traj) for deterministic replay videos
      - optionally render videos via render_from_actions(xml_path, npz_path, cfg)

    Returns:
      trajs: list[Trajectory]
      npz_paths: list[Path] saved (one per episode)
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    video_root_dir = Path(video_root_dir)
    video_root_dir.mkdir(parents=True, exist_ok=True)

    trajs: List[Trajectory] = []
    npz_paths: List[Path] = []

    for ep in range(n_episodes):
        env = env_fn()  # e.g. RolloutInfoWrapper(PayloadGymEnv(...))

        base_env = _unwrap_env(env)
        print("terminate_on_success(base_env) =", getattr(base_env, "terminate_on_success", None))
        # reset expert’s internal plan cache
        if hasattr(expert, "reset_episode"):
            expert.reset_episode()

        obs, info = env.reset()

        # capture init mujoco state for replay
        # PayloadGymEnv in your repo exposes `model` and `data` (MuJoCo handles)
        qpos0 = base_env.data.qpos.copy()
        qvel0 = base_env.data.qvel.copy()

        obs_list = [np.array(obs, dtype=np.float32)]
        act_list: List[np.ndarray] = []
        u_traj_mujoco: List[np.ndarray] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            act = expert.act(obs).astype(np.float32)
            next_obs, rew, terminated, truncated, step_info = env.step(act)

            act_list.append(act.copy())
            u_traj_mujoco.append(base_env.action_mujoco.copy())

            obs = next_obs
            obs_list.append(np.array(obs, dtype=np.float32))

        # --- save NPZ log ---
        u_traj_mujoco_np = np.asarray(u_traj_mujoco, dtype=np.float32)
        npz_path = logs_dir / f"expert_ep_{ep:03d}.npz"
        np.savez_compressed(
            npz_path,
            qpos0=qpos0,
            qvel0=qvel0,
            u_traj=u_traj_mujoco_np,  # save the mujoco-scaled actions for direct replay in renderer
        )
        npz_paths.append(npz_path)
        print(f"[round0] saved log: {npz_path} (T={len(act_list)})",
            "terminated=", terminated,
            "truncated=", truncated)
        # --- render videos (one directory per episode to avoid overwriting side.mp4/top.mp4/etc) ---
        if render_videos:
            ep_video_dir = video_root_dir / f"expert_ep_{ep:03d}"
            cfg = VideoConfig(
                out_dir=str(ep_video_dir),
                fps=50,
                views=["diag"],
                env_min=np.array([-2.0, -2.0, 0.0], dtype=float),
                env_max=np.array([+2.0, +2.0, 2.0], dtype=float),
            )
            written = render_from_actions(
                xml_path=xml_path,
                init_and_actions_or_npz=str(npz_path),  # pass the npz path directly
                cfg=cfg,
            )
            print(f"[round0] rendered: {written}")
        # --- build imitation Trajectory ---
        trajs.append(
            Trajectory(
                obs=np.stack(obs_list, axis=0),          # (T+1, obs_dim)
                acts=np.stack(act_list, axis=0),         # (T, act_dim)
                infos=None, # (T,)
                terminal = bool(terminated)   # True only if env says terminal, not time limit
,
            )
        )

        try:
            env.close()
        except Exception:
            pass

    return trajs, npz_paths


def main():
    rng = np.random.default_rng(0)
    th.manual_seed(0)

    xml_path = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    template_yaml = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"

    paths = PcDbCBSPaths(
        bindings_path="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
        input_yaml=template_yaml,
        pc_dbcbs_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
        opt_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt_training.yaml",
        dynobench_base="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
        motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
        time_limit=50000.0,
        work_dir_root="runs/_tmp_pcdbcbs",
        keep_files=False,
        warmstart_optimization=True,
        N_opt=100,
    )

    # --- env factory (IMPORTANT: keep the same as training) ---
    def make_env(max_steps=200):
        env = PayloadGymEnv(xml_path=xml_path, template_yaml_path=template_yaml, max_steps=max_steps)
        env.terminate_on_success = False          # <-- set on PayloadGymEnv
        env = Monitor(env)
        env = RolloutInfoWrapper(env)
        return env

    venv = DummyVecEnv([lambda: make_env(max_steps=200)])
    # --- expert (policy-space actions in [-1,1]) ---
    tmp_env = PayloadGymEnv(xml_path=xml_path, template_yaml_path=template_yaml, max_steps=200)
    nu = tmp_env.action_dim
    act_low = -1. * np.ones(nu, dtype=np.float32)
    act_high = 1. * np.ones(nu, dtype=np.float32)

    expert = PcDbCBSExpert(paths=paths, act_low=act_low, act_high=act_high, replan_every_k=0)
    expert_policy = ExpertPolicySB3(venv.observation_space, venv.action_space, expert, device="cpu")

    # --- cache expert trajs ---
    out_root = Path("runs/bc_payload")
    out_root.mkdir(parents=True, exist_ok=True)
    round0_dir = out_root / "round0_cache"
    round0_dir.mkdir(parents=True, exist_ok=True)
    trajs_pkl = round0_dir / "expert_trajs.pkl"

    # if you changed expert/env logic, delete this file once
    if trajs_pkl.exists():
        with open(trajs_pkl, "rb") as f:
            expert_trajs = pickle.load(f)
        print(f"Loaded cached expert_trajs: {trajs_pkl} (n={len(expert_trajs)})")
    else:
        expert_trajs, _ = collect_expert_trajs_and_logs(
            env_fn=lambda: make_env(max_steps=200),
            expert=expert,
            n_episodes=300,  # <-- increase this first; 10 can be too low
            xml_path=xml_path,
            logs_dir=str(round0_dir / "npz_logs"),
            video_root_dir=str(out_root / "videos_round0"),
            render_videos=False,
        )
        with open(trajs_pkl, "wb") as f:
            pickle.dump(expert_trajs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved cached expert_trajs: {trajs_pkl} (n={len(expert_trajs)})")

    transitions = rollout.flatten_trajectories(expert_trajs)
    print("num transitions:", len(transitions))
    print("n_eps:", len(expert_trajs))
    print("n_transitions:", len(transitions))
    print("obs shape:", transitions.obs.shape, "acts shape:", transitions.acts.shape)

    # --- BC trainer (make policy explicit so it’s stable across versions) ---
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=64,
        optimizer_kwargs=dict(lr=1e-3),
    )

    bc_trainer.train(n_epochs=400)  # start 200; adjust later
    th.save(bc_trainer.policy.state_dict(), out_root / "policy_state_dict.pt")
    print("Saved BC policy to:", out_root / "policy_state_dict.pt")

    # --- evaluate BC vs Expert ---
    bc_ret, _ = evaluate_policy(bc_trainer.policy, venv, n_eval_episodes=10, deterministic=True)
    ex_ret, _ = evaluate_policy(expert_policy, venv, n_eval_episodes=10, deterministic=True)
    print("BC mean return:", bc_ret)
    print("Expert mean return:", ex_ret)

    # --- Most important diagnostic: action MAE on expert states ---
    # Roll out expert for 1 episode, record obs and expert acts; query BC on same obs.
    env0 = make_env(max_steps=200)
    expert.reset_episode()
    obs, _ = env0.reset()
    obs_list = []
    exp_acts = []
    done = False
    while not done:
        obs_list.append(obs.copy())
        a_exp = expert.act(obs).astype(np.float32)
        exp_acts.append(a_exp.copy())
        obs, _, terminated, truncated, _ = env0.step(a_exp)
        done = bool(terminated or truncated)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    exp_arr = np.asarray(exp_acts, dtype=np.float32)

    # bc policy predict expects batch
    bc_arr, _ = bc_trainer.policy.predict(obs_arr, deterministic=True)
    bc_arr = np.asarray(bc_arr, dtype=np.float32)

    mae = float(np.mean(np.abs(bc_arr - exp_arr)))
    mse = float(np.mean((bc_arr - exp_arr) ** 2))
    print("BC vs Expert action MAE:", mae, "MSE:", mse)
    print("Expert act min/mean/max:", exp_arr.min(), exp_arr.mean(), exp_arr.max())
    print("BC act     min/mean/max:", bc_arr.min(), bc_arr.mean(), bc_arr.max())


if __name__ == "__main__":
    main()