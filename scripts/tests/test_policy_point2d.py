from pathlib import Path
import numpy as np
import mujoco

import torch as th
from imitation.algorithms import bc
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root / "scripts"))
from point2d_env import Point2DEnv

from videos_from_log import (
    VideoConfig,
    render_from_actions,
    render_from_states,
)


def _joint_state(model, data, joint_name: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint '{joint_name}' not found in model.")
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]
    q = float(data.qpos[qadr])
    v = float(data.qvel[dadr])
    return q, v


def main():
    
    xml_path = str(project_root / "envs/point2d.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Build venv only to get observation_space/action_space
    def make_env():
        return RolloutInfoWrapper(
            Point2DEnv(
                xml_path=xml_path,
                goal=(-0.9, -0.2),
                max_steps=300,
                
            )
        )

    venv = DummyVecEnv([make_env])

    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
    )

    state_dict_path = project_root / "scripts/runs/dagger_point2d/policy_state_dict.pt"
    state_dict = th.load(state_dict_path, map_location="cpu")
    bc_trainer.policy.load_state_dict(state_dict)

    policy = bc_trainer.policy

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # initial condition
    jx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jx")
    jy_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jy")
    data.qpos[model.jnt_qposadr[jx_id]] = -0.9
    data.qpos[model.jnt_qposadr[jy_id]] = -0.9
    mujoco.mj_forward(model, data)

    goal = np.array([1.0, 1.0], dtype=np.float32)

    # -----------------------------
    # LOGGING
    # -----------------------------
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()

    u_traj = []
    x_traj = []  # optional (state replay mode)

    steps = 300
    for t in range(steps):
        x, xdot = _joint_state(model, data, "jx")
        y, ydot = _joint_state(model, data, "jy")

        obs = np.array([x, y, xdot, ydot, goal[0], goal[1], 10, 10, 10], dtype=np.float32)
        
        u, _ = policy.predict(obs, deterministic=True)
        u = np.asarray(u, dtype=np.float32).reshape(-1)  # ensure (act_dim,)
        
        data.ctrl[:] = u
        mujoco.mj_step(model, data)

        u_traj.append(u.copy())
        x_traj.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))

    u_traj = np.asarray(u_traj, dtype=np.float32)
    x_traj = np.asarray(x_traj, dtype=np.float32)

    print("Simulation done.")

    # -----------------------------
    # VIDEO CONFIG
    # -----------------------------
    cfg = VideoConfig(
        out_dir="videos/test_pd_point2d_goal2",
        fps=100,
        views=["side", "top"],
        env_min=[-1.5, -1.5, 0],
        env_max=[+1.5, +1.5, 1.5],
    )

    # ==========================================================
    # OPTION A (RECOMMENDED): ROLLOUT ACTIONS → VIDEO
    # ==========================================================
    render_from_actions(
        xml_path=xml_path,
        init_and_actions_or_npz=(qpos0, qvel0, u_traj),
        cfg=cfg,
    )

    # ==========================================================
    # OPTION B: REPLAY STATES → VIDEO
    # (comment Option A and uncomment this if you want)
    # ==========================================================
    # render_from_states(
    #     xml_path=xml_path,
    #     x_traj_or_npz=x_traj,
    #     cfg=cfg,
    # )

    print("Videos written to:", cfg.out_dir)


if __name__ == "__main__":
    main()
