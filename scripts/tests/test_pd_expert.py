import numpy as np
import mujoco

from expert_pd_controller import PDExpert

# NEW: import the video utility
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
    xml_path = "../envs/point2d.xml"

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # actuator bounds
    ctrlrange = model.actuator_ctrlrange.copy()
    act_low = ctrlrange[:, 0].astype(np.float32)
    act_high = ctrlrange[:, 1].astype(np.float32)

    expert = PDExpert(kp=6.0, kd=3.0, act_low=act_low, act_high=act_high)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # initial condition
    jx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jx")
    jy_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jy")
    data.qpos[model.jnt_qposadr[jx_id]] = -0.5
    data.qpos[model.jnt_qposadr[jy_id]] = 0.3
    mujoco.mj_forward(model, data)

    goal = np.array([0.8, -0.2], dtype=np.float32)

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

        obs = np.array([x, y, xdot, ydot, goal[0], goal[1]], dtype=np.float32)
        u = expert.act(obs)

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
        out_dir="videos/test_pd_point2d",
        fps=100,
        views=["side", "top"],
        env_min=[-2, -2, 0],
        env_max=[+2, +2, 2],
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
