from pathlib import Path
import numpy as np

import torch as th
from imitation.algorithms import bc
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root / "scripts"))
from payload_env import PayloadGymEnv
import yaml
from videos_from_log import (
    VideoConfig,
    render_from_actions,
    render_from_states,
)



def main():
    
    xml_path = "/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    template_yaml = (
        "/home/khaledwahba94/inria/imitation-agile-payload-manipulation/"
        "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/"
        "mujocoquadspayload_zerogoal.yaml"
    )

    env = PayloadGymEnv(
        xml_path=xml_path,
        template_yaml_path=template_yaml,
        max_steps=500,
    )

    
    env.reset()    
    qpos0 = env.data.qpos.copy()
    qvel0 = env.data.qvel.copy()

    
    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )

    print("env information:" )
    print("obs space:", env.observation_space)
    print("act space:", env.action_space)
    print("act low:", env.action_space.low)
    print("act high:", env.action_space.high)
    print("-----------")
    print("previous action (should be zeros):", env.prev_action)
    print("initial qpos:", qpos0)
    print("initial qvel:", qvel0)
    print("-----------")
    # exit()
    print("Loading policy...")

    state_dict_path = project_root / "scripts/runs/dagger_payload_hover/policy_state_dict.pt"
    state_dict = th.load(state_dict_path, map_location="cpu")
    bc_trainer.policy.load_state_dict(state_dict)

    policy = bc_trainer.policy


    # -----------------------------
    # LOGGING
    # -----------------------------

    u_traj = []
    x_traj = []  # optional (state replay mode)

    steps = 100
    for t in range(steps):

        obs = env._get_obs()        
        u, _ = policy.predict(obs, deterministic=True)
        u = np.asarray(u, dtype=np.float32).reshape(-1)  # ensure (act_dim,)
        # u/=(0.034*9.81)
        # env.prev_action = u
        print(f"t={t}, obs={obs}, \n u={u}")
        env.step(u)

        u_traj.append(u.copy())
        x_traj.append(np.concatenate([env.data.qpos.copy(), env.data.qvel.copy()]))

    u_traj = np.asarray(u_traj, dtype=np.float32)
    x_traj = np.asarray(x_traj, dtype=np.float32)

    print("Simulation done.")

    # exit()
    # -----------------------------
    # VIDEO CONFIG
    # -----------------------------
    cfg = VideoConfig(
        out_dir="videos/test_payload_policy",
        fps=50,
        views=["diag", "side", "top"],
        env_min=[-2.0, -2.5, 0],
        env_max=[+2.0, +2.5, 1.5],
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
