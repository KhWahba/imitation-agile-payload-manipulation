# /mnt/data/test_pcdbcbs_expert.py
from xml.parsers.expat import model
import numpy as np
import mujoco
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "scripts"))
from expert_pcdbcbs import PcDbCBSExpert, PcDbCBSPaths

# Same video utility your PD test uses 
from videos_from_log import VideoConfig, render_from_actions

from utils import set_mujoco_state_from_joint_robot_start



def get_obs_from_qpos_qvel(data, n_bodies: int, quat_out: str = "xyzw") -> np.ndarray:
    """
    Returns obs = [pose_all (7*n), vel_all (6*n)] where
      pose_i = [pos3, quat4] for i=0..n-1
      vel_i  = [lin3, ang3] for i=0..n-1
    Assumes qpos layout is free joints: [pos3, quat(wxyz)] repeated.
    Assumes qvel layout is free joints: [lin3, ang3] repeated.

    quat_out: "wxyz" or "xyzw"
    """
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)

    poses_wxyz = qpos[:7 * n_bodies].copy()   # (7n,)
    vels       = qvel[:6 * n_bodies].copy()   # (6n,)

    if quat_out == "wxyz":
        poses = poses_wxyz
    elif quat_out == "xyzw":
        poses = poses_wxyz.copy()
        for i in range(n_bodies):
            p0 = 7 * i
            qw, qx, qy, qz = poses[p0+3:p0+7]
            poses[p0+3:p0+7] = [qx, qy, qz, qw]
    else:
        raise ValueError("quat_out must be 'wxyz' or 'xyzw'")

    obs = np.concatenate([poses, vels], axis=0).astype(np.float32)
    return obs

def main():
    xml_path = "/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # actuator bounds
    ctrlrange = model.actuator_ctrlrange.copy()
    act_low = ctrlrange[:, 0].astype(np.float32)
    act_high = ctrlrange[:, 1].astype(np.float32)  # normalize by nominal thrust

    # Pick bodies (or infer)
    paths = PcDbCBSPaths(
        bindings_path="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
        input_yaml="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_empty2.yaml",
        pc_dbcbs_cfg_yaml="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
        opt_cfg_yaml="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt.yaml",
        dynobench_base="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
        motion_primitives_base="/home/khaledwahba94/inria/pc-dbCBS/motion_primitives/",
        time_limit=350000.0,
        work_dir_root="runs/_tmp_pcdbcbs",  # temp root
        keep_files=False,                   # <== no file clutter
        warmstart_optimization=False,    # <== disable warmstart for optimization-only mode
    )
    K = 0
    expert = PcDbCBSExpert(paths=paths, act_low=act_low, act_high=act_high, replan_every_k=K)
    expert.reset_episode()

    mujoco.mj_resetData(model, data)


    qpos0, qvel0 = set_mujoco_state_from_joint_robot_start(model, data, paths.input_yaml)
    data.qpos = qpos0.copy()
    data.qvel = qvel0.copy()
    mujoco.mj_forward(model, data)
    u_traj = []
    x_traj = []
    n_bodies = model.nq // 7 # free joints

    # first action triggers planning-from-obs
    obs = get_obs_from_qpos_qvel(data, n_bodies, quat_out="xyzw")
    # u0 = expert.act(obs)

    T_total = 200   # or 5*K
    replan_steps = []
    for t in range(T_total):
        obs = get_obs_from_qpos_qvel(data, n_bodies, quat_out="xyzw")
        u = expert.act(obs)
        if expert.just_replanned:
            print(f"[replan] t={t}")
            replan_steps.append(t)

        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        u_traj.append(u.copy())
        x_traj.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
    print("Replans at:", replan_steps)

    u_traj = np.asarray(u_traj, dtype=np.float32)
    x_traj = np.asarray(x_traj, dtype=np.float32)

    print("Simulation done. u_traj shape:", u_traj.shape)


    cfg = VideoConfig(
        out_dir="videos/test_pcdbcbs_payload",
        fps=50,
        views=["side", "top"],
        env_min=[-1.5, -1.5, 0],
        env_max=[+1.5, +1.5, 2],
    )

    render_from_actions(
        xml_path=xml_path,
        init_and_actions_or_npz=(qpos0, qvel0, u_traj),
        cfg=cfg,
    )

    # print("Videos written to:", cfg.out_dir)


if __name__ == "__main__":
    main()
