# payload_gym_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import yaml
from utils import *
np.set_printoptions(
    precision=3,      # number of decimals
    suppress=True,    # turn off scientific notation
    linewidth=120,    # avoid line wrapping
)

class PayloadGymEnv(gym.Env):
    """
    Observation:
        obs = [state, goal, prev_action]
    where:
        state = 13 * (1 + quadsNum)
        goal  = same dimension as state (from template YAML joint_robot[0]["goal"])
        prev_action = nu = 4 * quadsNum

    Action:
        u_t in R^{nu}
    """

    def __init__(
        self,
        xml_path: str,
        template_yaml_path: str,
        max_steps: int = 400,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.max_steps = int(max_steps)
        self.step_count = 0

        # Read quadsNum and goal from the SAME YAML the expert uses
        with open(template_yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        jr0 = cfg["joint_robot"][0]
        self.n_quads = int(jr0.get("quadsNum", 1))
        self.n_bodies = 1 + self.n_quads

        self.state_dim = 13 * self.n_bodies
        self.action_dim = 4 * self.n_quads

        self.start_state = np.asarray(jr0["start"], dtype=np.float64).reshape(-1)

        if self.start_state.shape[0] != self.state_dim:
            raise ValueError(
                f"start dim {self.start_state.shape[0]} != expected {self.state_dim}"
            )
        self.goal = np.asarray(jr0["goal"], dtype=np.float32).reshape(-1)
        if self.goal.shape[0] != self.state_dim:
            raise ValueError(
                f"goal dim {self.goal.shape[0]} != expected {self.state_dim} (=13*(1+quadsNum))."
            )

        # Action space from MuJoCo ctrlrange if possible
        if self.model.nu == self.action_dim and self.model.actuator_ctrlrange is not None:
            ctrlrange = self.model.actuator_ctrlrange.astype(np.float32)
            self.action_space = spaces.Box(
                low=ctrlrange[:, 0],
                high=ctrlrange[:, 1],
                shape=(self.action_dim,),
                dtype=np.float32,
            )
        else:
            # fallback (adjust later if needed)
            self.action_space = spaces.Box(
                low=np.zeros(self.action_dim, dtype=np.float32),
                high=np.ones(self.action_dim, dtype=np.float32) * 2.0,
                dtype=np.float32,
            )

        # Obs space: [state, goal, prev_action]
        obs_dim = self.state_dim + self.state_dim + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.prev_action = np.zeros((self.action_dim,), dtype=np.float32)
        self.payload_bounds = (
            np.array([-1.0, -1.0, 0.48], dtype=np.float32),   # low
            np.array([-0.95, -0.95, 0.5], dtype=np.float32), # high
        )
    def _get_state(self) -> np.ndarray:
        # EXACT same mapping you used before
        return get_obs_from_qpos_qvel(self.data, self.n_bodies, quat_out="xyzw")

    def _get_obs(self) -> np.ndarray:
        state = self._get_state()
        return np.concatenate([state, self.goal, self.prev_action], axis=0).astype(np.float32)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        state = np.asarray(self.start_state, dtype=np.float64).reshape(-1)

        n = self.n_bodies
        expected = 7 * n + 6 * n
        if state.size != expected:
            raise ValueError(f"start_state len={state.size}, expected {expected} (=13*n with poses-first layout).")

        poses = state[: 7 * n].copy()      # [pos3, quat_xyzw4] repeated
        vels  = state[7 * n :].copy()      # [lin3, ang3] repeated

        # Convert poses quats xyzw -> wxyz for MuJoCo qpos
        for i in range(n):
            p0 = 7 * i
            qx, qy, qz, qw = poses[p0 + 3 : p0 + 7]
            poses[p0 + 3 : p0 + 7] = [qw, qx, qy, qz]

        # Now poses is exactly MuJoCo qpos layout for free joints: [pos3, quat(wxyz)] * n
        self.data.qpos[:] = poses
        self.data.qvel[:] = vels

        mujoco.mj_forward(self.model, self.data)

        self.prev_action[:] = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    # def reset(self, *, seed=None, options=None):
    #     super().reset(seed=seed)
    #     rng = self.np_random

    #     mujoco.mj_resetData(self.model, self.data)

    #     # --- sample positions ---
    #     payload_pos, quad_pos = sample_payload_and_quads(
    #     payload_bounds=self.payload_bounds,  # (low, high)
    #     n_quads=self.n_quads,
    #     cable_min=0.49,
    #     cable_max=0.5,
    #     rng=rng,
    #     quad_radius=0.1,
    # )

    #     # --- sample orientations ---
    #     # IMPORTANT: MuJoCo qpos wants quats as wxyz
    #     payload_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    #     # quad_quats_xyzw = np.stack([sample_bounded_quat(0.1, rng) for _ in range(self.n_quads)], axis=0)
    #     quad_quats_xyzw = np.stack([np.array([0,0,0,1]) for _ in range(self.n_quads)], axis=0)
    #     quad_quats_wxyz = np.zeros_like(quad_quats_xyzw, dtype=np.float64)
    #     for i in range(self.n_quads):
    #         qx, qy, qz, qw = quad_quats_xyzw[i]
    #         quad_quats_wxyz[i] = np.array([qw, qx, qy, qz], dtype=np.float64)

    #     # --- sample velocities ---
    #     # p_lin,p_ang: (3,), (3,)
    #     # q_lin,q_ang: (n_quads,3), (n_quads,3)
    #     p_lin, p_ang, q_lin, q_ang = sample_velocities(self.n_quads, rng=rng)

    #     # --- build qpos block for ALL bodies first ---
    #     qpos = np.zeros((7 * self.n_bodies,), dtype=np.float64)
    #     qvel = np.zeros((6 * self.n_bodies,), dtype=np.float64)

    #     # body 0 = payload
    #     qpos[0:3] = payload_pos
    #     qpos[3:7] = payload_quat_wxyz
    #     qvel[0:3] = p_lin
    #     qvel[3:6] = p_ang

    #     # bodies 1..n_quads = quads
    #     for i in range(self.n_quads):
    #         b = 1 + i
    #         qp = 7 * b
    #         qv = 6 * b

    #         qpos[qp:qp+3] = quad_pos[i]
    #         qpos[qp+3:qp+7] = quad_quats_wxyz[i]

    #         qvel[qv:qv+3] = q_lin[i]
    #         qvel[qv+3:qv+6] = q_ang[i]

    #     # write into MuJoCo
    #     self.data.qpos[:] = qpos
    #     self.data.qvel[:] = qvel
    #     mujoco.mj_forward(self.model, self.data)

    #     self.prev_action[:] = 0.0
    #     self.step_count = 0
    #     return self._get_obs(), {}


    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[: self.action_dim] = action
        mujoco.mj_step(self.model, self.data)

        self.prev_action = action.copy()
        self.step_count += 1

        obs = self._get_obs()

        # Minimal reward: negative payload position distance to goal
        state = obs[: self.state_dim]
        payload_pos = state[:3]
        goal_pos = self.goal[:3]
        dist_goal = float(np.linalg.norm(payload_pos - goal_pos))
        reward = -dist_goal

        terminated = dist_goal < 0.02
        truncated = self.step_count >= self.max_steps
        info = {"dist_goal": dist_goal}

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    import numpy as np

    from expert_pcdbcbs import PcDbCBSExpert, PcDbCBSPaths
    from videos_from_log import VideoConfig, render_from_actions  # same as your PD test

    xml_path = "/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    template_yaml = (
        "/home/khaledwahba94/inria/imitation-agile-payload-manipulation/"
        "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/"
        "mujocoquadspayload_empty2.yaml"
    )

    env = PayloadGymEnv(
        xml_path=xml_path,
        template_yaml_path=template_yaml,
        max_steps=500,
    )

    print("\n--- Testing with PcDbCBSExpert ---\n")

    # actuator bounds
    ctrlrange = env.model.actuator_ctrlrange.copy()
    act_low = ctrlrange[:, 0].astype(np.float32)
    act_high = ctrlrange[:, 1].astype(np.float32)

    paths = PcDbCBSPaths(
        bindings_path="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
        input_yaml=template_yaml,
        pc_dbcbs_cfg_yaml="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
        opt_cfg_yaml="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt.yaml",
        dynobench_base="/home/khaledwahba94/inria/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
        motion_primitives_base="/home/khaledwahba94/inria/pc-dbCBS/motion_primitives/",
        work_dir_root="runs/_tmp_pcdbcbs",
        keep_files=False,
        warmstart_optimization=True,
    )

    expert = PcDbCBSExpert(
        paths=paths,
        act_low=act_low,
        act_high=act_high,
        replan_every_k=100,
    )

    expert.reset_episode()

    # ---- rollout ----
    obs, _ = env.reset()
    plot_payload_and_quads_state(
        state=obs[:env.state_dim],
        n_bodies=env.n_bodies,
        radius=0.1,
        save_path="pos.pdf",
    )
    print("Initial payload and quads positions saved to pos.pdf")
    print("start state: ", obs[:env.state_dim])
    print("goal state:  ", env.goal)
    print("start rollout...")
    # exit()
    # capture init state for rendering
    qpos0 = env.data.qpos.copy()
    qvel0 = env.data.qvel.copy()

    u_traj = []
    x_traj = []
    replan_steps = []

    T_total = 500
    for t in range(T_total):
        action = expert.act(obs)
        if expert.just_replanned:
            print(f"[replan] t={t}")
            replan_steps.append(t)

        obs, reward, terminated, truncated, info = env.step(action)

        # store trajectories
        u_traj.append(action.copy())
        x_traj.append(np.concatenate([env.data.qpos.copy(), env.data.qvel.copy()]))

        state = obs[: env.state_dim]
        payload_pos = state[:3]
        print(
            f"t={t:03d} payload_pos=({payload_pos[0]:+.2f},{payload_pos[1]:+.2f},{payload_pos[2]:+.2f}) "
            f"reward={reward:+.3f}"
        )

        if terminated:
            print("Expert reached goal.")
            break
        if truncated:
            print("Episode truncated.")
            break

    print("Replans at:", replan_steps)

    u_traj = np.asarray(u_traj, dtype=np.float32)
    x_traj = np.asarray(x_traj, dtype=np.float32)
    print("Simulation done. u_traj shape:", u_traj.shape)

    # ---- render ----
    cfg = VideoConfig(
        out_dir="videos/test_pcdbcbs_payload",
        fps=50,
        views=["side", "top"],
        env_min=[-2.0, -2.0, 0],
        env_max=[+2.0, +2.0, 2],
    )

    render_from_actions(
        xml_path=xml_path,
        init_and_actions_or_npz=(qpos0, qvel0, u_traj),
        cfg=cfg,
    )

    print("PayloadGymEnv + expert test finished. Videos in:", cfg.out_dir)
