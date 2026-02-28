# train_dagger_payload_with_cache_and_videos.py

import os
import tempfile
import pickle
from pathlib import Path
from typing import Any, Tuple, List

import numpy as np
import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

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
        info_list: List[dict] = []
        u_traj: List[np.ndarray] = []
        u_traj_mujoco: List[np.ndarray] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            act = expert.act(obs).astype(np.float32)
            next_obs, rew, terminated, truncated, step_info = env.step(act)

            act_list.append(act.copy())
            u_traj.append(act.copy())
            act_mujoco = act.copy() * 0.034 * 9.81 / 4
            u_traj_mujoco.append(act_mujoco)
            info_list.append(step_info)

            obs = next_obs
            obs_list.append(np.array(obs, dtype=np.float32))
        # --- save NPZ log ---
        u_traj_np = np.asarray(u_traj, dtype=np.float32)
        u_traj_mujoco_np = np.asarray(u_traj_mujoco, dtype=np.float32)
        npz_path = logs_dir / f"expert_ep_{ep:03d}.npz"
        np.savez_compressed(
            npz_path,
            qpos0=qpos0,
            qvel0=qvel0,
            u_traj=u_traj_mujoco_np,  # save the mujoco-scaled actions for direct replay in renderer
        )
        npz_paths.append(npz_path)
        print(f"[round0] saved log: {npz_path} (T={u_traj_np.shape[0]})")

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
                terminal = bool(terminated or truncated)   # True only if env says terminal, not time limit
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

    # --------- paths ----------
    xml_path = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    template_yaml = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"

    paths = PcDbCBSPaths(
        bindings_path="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
        input_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml",
        pc_dbcbs_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
        opt_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt_training.yaml",
        dynobench_base="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
        motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
        time_limit=50000.0,
        work_dir_root="runs/_tmp_pcdbcbs",  # temp root
        keep_files=False,                   # <== no file clutter
        warmstart_optimization=True,    # <== disable warmstart for optimization-only mode
        N_opt=100,                        # <== number of optimization steps
    )

    # --------- env factory ----------
    def make_env(max_steps: int = 200):
        env = PayloadGymEnv(
            xml_path=xml_path,
            template_yaml_path=template_yaml,
            max_steps=max_steps,
        )
        return RolloutInfoWrapper(env)

    # VecEnv for DAgger trainer + eval
    venv = DummyVecEnv([lambda: make_env(max_steps=200)])

    # --------- expert ----------
    tmp_env = PayloadGymEnv(xml_path=xml_path, template_yaml_path=template_yaml, max_steps=200)
    ctrlrange = tmp_env.model.actuator_ctrlrange.copy()
    nu = tmp_env.action_dim
    act_low = -1*np.ones(nu)  #ctrlrange[:, 0].astype(np.float32)
    act_high = 1*np.ones(nu) #ctrlrange[:, 1].astype(np.float32)

    expert = PcDbCBSExpert(
        paths=paths,
        act_low=act_low,
        act_high=act_high,
        replan_every_k=0,  # "periodic" replans disabled, but it can still replan when plan ends
    )


    # --------- round-0 cache + videos ----------
    out_root = Path("runs/bc_payload")
    out_root.mkdir(parents=True, exist_ok=True)

    round0_dir = out_root / "round0_cache"
    round0_dir.mkdir(parents=True, exist_ok=True)

    trajs_pkl = round0_dir / "expert_trajs.pkl"
    logs_dir = round0_dir / "npz_logs"
    video_root = out_root / "videos_round0"

    if trajs_pkl.exists():
        with open(trajs_pkl, "rb") as f:
            expert_trajs = pickle.load(f)
        print(f"Loaded cached expert_trajs: {trajs_pkl} (n={len(expert_trajs)})")
    else:
        expert_trajs, npz_paths = collect_expert_trajs_and_logs(
            env_fn=lambda: make_env(max_steps=200),
            expert=expert,
            n_episodes=10,  # <-- only seeds round-0; DAgger training still runs many episodes later
            xml_path=xml_path,
            logs_dir=str(logs_dir),
            video_root_dir=str(video_root),
            render_videos=True,
        )
        with open(trajs_pkl, "wb") as f:
            pickle.dump(expert_trajs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved cached expert_trajs: {trajs_pkl} (n={len(expert_trajs)})")

    print(f"Collected/loaded {len(expert_trajs)} expert trajectories for round-0 seed.")
    transitions = rollout.flatten_trajectories(expert_trajs)

    # print(expert_trajs[0].acts[0])  # print first traj for sanity check
    # print(expert_trajs[0].obs[0])  # print first traj for sanity check
    # exit()
    # --------- BC learner ----------
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=32,

    )

    bc_trainer.train(n_epochs=500)
    reward, _ = evaluate_policy(bc_trainer.policy, venv, n_eval_episodes=2)
    print("Reward:", reward)
    # reward_expert, _ = evaluate_policy(ExpertPolicySB3(venv.observation_space, venv.action_space, expert), venv, n_eval_episodes=2)
    # print("Expert reward:", reward_expert)


    th.save(bc_trainer.policy.state_dict(), out_root / "policy_state_dict.pt")
    print("Saved learner policy to:", out_root / "policy_state_dict.pt")


if __name__ == "__main__":
    main()
