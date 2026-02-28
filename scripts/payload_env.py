# payload_gym_env.py
import os
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
    Observation (expert-compatible + learner features):
        obs = [raw_state, learner_features]
    where raw_state is preserved as a prefix so the expert wrappers that read
    obs[:state_dim] continue to work unchanged.

    Action:
        u_t in R^{nu}, normalized to [-1, 1]
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
        self.terminate_on_success = True
        # Success is declared only after staying near the goal for several consecutive steps.
        self.success_dist = 0.02
        self.success_hold_steps_required = 10
        self.success_hold_count = 0
        self.debug_done_reasons = os.environ.get("PAYLOAD_ENV_DEBUG_DONE_REASONS", "0") == "1"

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
        # if self.model.nu == self.action_dim and self.model.actuator_ctrlrange is not None:
        #     ctrlrange = self.model.actuator_ctrlrange.astype(np.float32)
        #     self.action_space = spaces.Box(
        #         low=ctrlrange[:, 0],
        #         high=ctrlrange[:, 1],
        #         shape=(self.action_dim,),
        #         dtype=np.float32,
        #     )
        # else:
        # fallback (adjust later if needed)
        self.u_nominal = 0.034*9.81/4  # nominal per-rotor thrust to hover one quad, TODO: fix this from cf mass from mujoco model if possible
        self.action_mujoco = np.zeros((self.action_dim,), dtype=np.float32)  # for logging / rendering convenience

        # Planner action range: the planner outputs in [planner_act_low, planner_act_high]
        # Policy space is [-1, 1]. The env maps [-1, 1] back to planner range, then to Newtons.
        self.planner_act_low = np.zeros(self.action_dim, dtype=np.float32)
        self.planner_act_high = 1.4 * np.ones(self.action_dim, dtype=np.float32)
        self._act_mid = 0.5 * (self.planner_act_low + self.planner_act_high)
        self._act_half = 0.5 * (self.planner_act_high - self.planner_act_low)
        # self.action_space = spaces.Box(
        #     low=np.zeros(self.action_dim, dtype=np.float32),
        #     high=np.ones(self.action_dim, dtype=np.float32) * 1.4,
        #     dtype=np.float32,
        # )
        self.action_space = spaces.Box(low=-np.ones(self.action_dim, np.float32),
                                    high=np.ones(self.action_dim, np.float32),
                                    dtype=np.float32)

        # Obs space: [raw_state, learner_features]
        # learner_features = payload_pos_err(3) + payload_vel(3)
        #                  + n_quads * (rel_pos_err(3) + rel_vel_err(3) + quat_err_vec(3) + ang_vel_err(3))
        #                  + prev_action(action_dim)
        self.learner_obs_dim = 6 + self.n_quads * 12 + self.action_dim
        obs_dim = self.state_dim + self.learner_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.prev_action = np.zeros((self.action_dim,), dtype=np.float32)
        # Match planner workspace bounds from the dynobench env YAML (environment.min/max).
        # pc-dbCBS passes these bounds as problem.p_lb/p_ub and the MujocoQuadsPayload model
        # applies them to both payload and quad positions in x_lb/x_ub.
        env_min = np.asarray(cfg["environment"]["min"], dtype=np.float32).reshape(-1)
        env_max = np.asarray(cfg["environment"]["max"], dtype=np.float32).reshape(-1)
        if env_min.shape != (3,) or env_max.shape != (3,):
            raise ValueError(f"environment min/max must be 3D, got {env_min.shape} / {env_max.shape}")
        self.workspace_bounds = (env_min, env_max)
        self.payload_bounds = (env_min.copy(), env_max.copy())
        self.quad_bounds = (env_min.copy(), env_max.copy())
    def _get_state(self) -> np.ndarray:
        # EXACT same mapping you used before
        return get_obs_from_qpos_qvel(self.data, self.n_bodies, quat_out="xyzw")

    def _get_obs(self) -> np.ndarray:
        state = self._get_state()
        feats = self._get_learner_features(state)
        return np.concatenate([state, feats], axis=0).astype(np.float32)

    def _payload_pos(self) -> np.ndarray:
        # state layout in your obs: payload pose starts at index 0
        state = self._get_state()
        return state[:3].astype(np.float32)

    @staticmethod
    def _quat_xyzw_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float32)

    @staticmethod
    def _quat_xyzw_inv(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        n2 = float(np.dot(q, q))
        if n2 <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([-x, -y, -z, w], dtype=np.float32) / n2

    def _quad_quat_error_vec(self, q_des_xyzw: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
        q_err = self._quat_xyzw_mul(self._quat_xyzw_inv(q_des_xyzw), q_xyzw)
        # Canonicalize sign so q and -q represent the same attitude consistently.
        if q_err[3] < 0:
            q_err = -q_err
        return q_err[:3].astype(np.float32)

    def _get_learner_features(self, state: np.ndarray) -> np.ndarray:
        """Learner features while assuming payload is a point mass (ignore payload quat/ang vel)."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        goal = self.goal.astype(np.float32)
        n = self.n_bodies

        poses = state[: 7 * n]
        vels = state[7 * n :]
        goal_poses = goal[: 7 * n]
        # By design for this setup, desired payload/quad linear+angular velocities are treated as zero.
        goal_vels = np.zeros_like(vels, dtype=np.float32)

        # Payload position error and payload linear velocity
        pL = poses[0:3]
        pL_goal = goal_poses[0:3]
        vL = vels[0:3]
        feat_parts = [pL - pL_goal, vL]

        # Per-quad blocks: relative pos error, relative vel error, quat error vec, ang vel error
        for qi in range(self.n_quads):
            pose_base = 7 * (1 + qi)
            vel_base = 6 * (1 + qi)

            p_i = poses[pose_base : pose_base + 3]
            q_i = poses[pose_base + 3 : pose_base + 7]
            v_i = vels[vel_base : vel_base + 3]
            w_i = vels[vel_base + 3 : vel_base + 6]

            p_i_goal = goal_poses[pose_base : pose_base + 3]
            q_i_goal = goal_poses[pose_base + 3 : pose_base + 7]
            v_i_goal = goal_vels[vel_base : vel_base + 3]
            w_i_goal = goal_vels[vel_base + 3 : vel_base + 6]

            e_p_rel = (p_i - pL) - (p_i_goal - pL_goal)
            e_v_rel = (v_i - vL) - (v_i_goal - goal_vels[0:3])
            e_q = self._quad_quat_error_vec(q_i_goal, q_i)
            e_w = w_i - w_i_goal

            feat_parts.extend([e_p_rel, e_v_rel, e_q, e_w])

        feat_parts.append(self.prev_action.astype(np.float32))
        return np.concatenate(feat_parts, axis=0).astype(np.float32)

    def _is_out_of_bounds(self) -> bool:
        p = self._payload_pos()
        low, high = self.payload_bounds
        return bool(np.any(p < low) or np.any(p > high))

    def _quad_positions(self) -> np.ndarray:
        state = self._get_state()
        if self.n_quads <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        positions = []
        for qi in range(self.n_quads):
            # Layout is poses-first then velocities:
            # [payload pose(7), quad1 pose(7), ..., payload vel(6), quad1 vel(6), ...]
            # so quad positions live in the poses block with stride 7.
            base = 7 * (1 + qi)
            positions.append(state[base: base + 3])
        return np.asarray(positions, dtype=np.float32)

    def _is_any_quad_out_of_bounds(self) -> bool:
        if self.n_quads <= 0:
            return False
        qpos = self._quad_positions()
        low, high = self.quad_bounds
        return bool(np.any(qpos < low) or np.any(qpos > high))

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
        self.success_hold_count = 0
        return self._get_obs(), {}

    # def reset(self, *, seed=None, options=None):
    #     super().reset(seed=seed)
    #     rng = self.np_random

    #     mujoco.mj_resetData(self.model, self.data)

    #     # --- sample positions ---
    #     payload_pos, quad_pos = sample_payload_and_quads(
    #     payload_bounds=self.payload_bounds,  # (low, high)
    #     n_quads=self.n_quads,
    #     cable_min=0.15,
    #     cable_max=0.5,
    #     rng=rng,
    #     quad_radius=0.1,
    # )

    #     # --- sample orientations ---
    #     # IMPORTANT: MuJoCo qpos wants quats as wxyz
    #     payload_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    #     quad_quats_xyzw = np.stack([sample_bounded_quat(20, rng) for _ in range(self.n_quads)], axis=0)
    #     # quad_quats_xyzw = np.stack([np.array([0,0,0,1]) for _ in range(self.n_quads)], axis=0)
    #     quad_quats_wxyz = np.zeros_like(quad_quats_xyzw, dtype=np.float64)
    #     for i in range(self.n_quads):
    #         qx, qy, qz, qw = quad_quats_xyzw[i]
    #         quad_quats_wxyz[i] = np.array([qw, qx, qy, qz], dtype=np.float64)

    #     # --- sample velocities ---
    #     # p_lin,p_ang: (3,), (3,)
    #     # q_lin,q_ang: (n_quads,3), (n_quads,3)
    #     p_lin, p_ang, q_lin, q_ang = sample_velocities(self.n_quads, lin_vel_max=1.0, ang_vel_max=0.1, rng=rng)

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
        # --- guard: if somehow called again after already OOB, end immediately ---
        payload_oob_pre = self._is_out_of_bounds()
        quad_oob_pre = self._is_any_quad_out_of_bounds()
        if payload_oob_pre or quad_oob_pre:
            obs = self._get_obs()
            info = {
                "out_of_bounds": bool(payload_oob_pre or quad_oob_pre),
                "payload_out_of_bounds": bool(payload_oob_pre),
                "quad_out_of_bounds": bool(quad_oob_pre),
                "dist_goal": float("inf"),
            }
            if self.debug_done_reasons:
                print(
                    "[PAYLOAD_ENV_DONE] "
                    f"step={self.step_count} terminated={True} "
                    f"truncated={False} "
                    "dist=inf "
                    f"payload_oob={bool(payload_oob_pre)} quad_oob={bool(quad_oob_pre)} "
                    f"in_goal={False} hold={int(self.success_hold_count)}/{int(self.success_hold_steps_required)} "
                    "(pre-step-guard)",
                    flush=True,
                )
            return obs, -100.0, True, False, info

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(action, -1.0, 1.0)
        action_planner = a * self._act_half + self._act_mid   # maps [-1,1] -> [act_low, act_high]
        self.action_mujoco = action_planner * self.u_nominal
        self.data.ctrl[: self.action_dim] = self.action_mujoco
        mujoco.mj_step(self.model, self.data)

        self.prev_action = action.copy()
        self.step_count += 1

        obs = self._get_obs()

        # reward as before
        state = obs[: self.state_dim]
        payload_pos = state[:3]
        goal_pos = self.goal[:3]
        dist_goal = float(np.linalg.norm(payload_pos - goal_pos))
        reward = -dist_goal

        # --- NEW: out-of-bounds termination after stepping ---
        payload_oob = self._is_out_of_bounds()
        quad_oob = self._is_any_quad_out_of_bounds()
        out_of_bounds = bool(payload_oob or quad_oob)
        if out_of_bounds:
            # strong penalty to teach the learner to avoid leaving workspace
            reward -= 100.0
        in_goal_region = dist_goal < float(self.success_dist)
        if in_goal_region and not out_of_bounds:
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0
        stable_success = self.success_hold_count >= int(self.success_hold_steps_required)
        oob_terminated = bool(payload_oob or quad_oob)
        terminated = (stable_success and self.terminate_on_success) or oob_terminated
        truncated = (self.step_count >= self.max_steps)

        info = {
            "dist_goal": dist_goal,
            "out_of_bounds": out_of_bounds,
            "payload_out_of_bounds": bool(payload_oob),
            "quad_out_of_bounds": bool(quad_oob),
            "success_dist": float(self.success_dist),
            "is_in_goal_region": bool(in_goal_region and (not out_of_bounds)),
            "success_hold_count": int(self.success_hold_count),
            "success_hold_steps_required": int(self.success_hold_steps_required),
            "is_success": bool(stable_success and (not out_of_bounds)),
        }
        if self.debug_done_reasons and (terminated or truncated):
            print(
                "[PAYLOAD_ENV_DONE] "
                f"step={self.step_count} terminated={bool(terminated)} truncated={bool(truncated)} "
                f"dist={dist_goal:.4f} payload_oob={bool(payload_oob)} quad_oob={bool(quad_oob)} "
                f"in_goal={bool(info['is_in_goal_region'])} hold={int(self.success_hold_count)}/{int(self.success_hold_steps_required)}",
                flush=True,
            )
        return obs, reward, terminated, truncated, info



if __name__ == "__main__":

    import numpy as np
    import torch as th

    from expert_pcdbcbs import PcDbCBSExpert, PcDbCBSPaths
    # Optional: if you want to load a saved learner
    # from stable_baselines3.common.policies import ActorCriticPolicy  # not needed if you already have a policy object

    # -----------------------
    # FLAGS: turn tests on/off
    # -----------------------
    DO_ENV_SANITY = True
    DO_ACTION_MAPPING = True
    DO_EXPERT_ROLLOUT = True
    DO_LEARNER_ROLLOUT = False          # set True if you load a learner policy below
    DO_EXPERT_VS_LEARNER_MAE = False    # set True if you load a learner policy below

    # -----------------------
    # Paths
    # -----------------------
    xml_path = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    template_yaml = (
        "/home/khaledwahba94/imitation-agile-payload-manipulation/"
        "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/"
        "mujocoquadspayload_zerogoal.yaml"
    )

    env = PayloadGymEnv(xml_path=xml_path, template_yaml_path=template_yaml, max_steps=500)

    def dist_goal_from_obs(obs: np.ndarray) -> float:
        state = obs[: env.state_dim]
        payload_pos = state[:3]
        goal_pos = env.goal[:3]
        return float(np.linalg.norm(payload_pos - goal_pos))

    def rollout_policy(policy_fn, name: str, T: int = 300, verbose_every: int = 10):
        """
        policy_fn: callable(obs)->action in policy space ([-1,1] per your env)
        """
        obs, _ = env.reset()
        total_return = 0.0
        oob = False
        reached = False

        acts = []
        ctrls = []
        dists = []

        for t in range(T):
            act = np.asarray(policy_fn(obs), dtype=np.float32).reshape(-1)
            obs, rew, terminated, truncated, info = env.step(act)

            total_return += float(rew)
            acts.append(act.copy())
            ctrls.append(env.action_mujoco.copy())
            dists.append(float(info.get("dist_goal", dist_goal_from_obs(obs))))

            if (t % verbose_every) == 0:
                print(f"[{name}] t={t:03d} dist={dists[-1]:.4f} rew={rew:+.3f} oob={info.get('out_of_bounds', False)}")

            if terminated or truncated:
                oob = bool(info.get("out_of_bounds", False))
                reached = (dists[-1] < 0.002) and (not oob)
                return {
                    "name": name,
                    "steps": t + 1,
                    "return": total_return,
                    "final_dist": dists[-1],
                    "out_of_bounds": oob,
                    "reached_goal": reached,
                    "acts": np.asarray(acts, dtype=np.float32),
                    "ctrls": np.asarray(ctrls, dtype=np.float32),
                    "dists": np.asarray(dists, dtype=np.float32),
                }

        # if horizon ends without termination
        return {
            "name": name,
            "steps": T,
            "return": total_return,
            "final_dist": dists[-1] if len(dists) else dist_goal_from_obs(obs),
            "out_of_bounds": False,
            "reached_goal": False,
            "acts": np.asarray(acts, dtype=np.float32),
            "ctrls": np.asarray(ctrls, dtype=np.float32),
            "dists": np.asarray(dists, dtype=np.float32),
        }

    # -----------------------
    # 1) ENV / OBS SANITY
    # -----------------------
    if DO_ENV_SANITY:
        obs, _ = env.reset()
        print("\n=== ENV SANITY ===")
        print("n_quads:", env.n_quads, "n_bodies:", env.n_bodies)
        print("state_dim:", env.state_dim, "action_dim:", env.action_dim)
        print("obs.shape:", obs.shape, "obs_dim expected:", env.state_dim + env.state_dim + env.action_dim)
        print("obs finite:", np.all(np.isfinite(obs)))
        print("initial dist_goal:", dist_goal_from_obs(obs))
        print("action_space:", env.action_space)

    # -----------------------
    # 2) ACTION MAPPING SANITY
    # -----------------------
    if DO_ACTION_MAPPING:
        print("\n=== ACTION MAPPING SANITY ===")
        obs, _ = env.reset()

        # test 3 representative actions: -1, 0, +1
        test_actions = [
            -np.ones(env.action_dim, np.float32),
            np.zeros(env.action_dim, np.float32),
            +np.ones(env.action_dim, np.float32),
        ]
        for i, a in enumerate(test_actions):
            obs2, rew, term, trunc, info = env.step(a)
            # env.action_mujoco is what got applied
            print(f"[map test {i}] act(min/mean/max)={a.min():+.3f}/{a.mean():+.3f}/{a.max():+.3f}")
            print(f"           ctrl(min/mean/max)={env.action_mujoco.min():+.6f}/{env.action_mujoco.mean():+.6f}/{env.action_mujoco.max():+.6f}")
            print(f"           data.ctrl(min/mean/max)={env.data.ctrl[:env.action_dim].min():+.6f}/{env.data.ctrl[:env.action_dim].mean():+.6f}/{env.data.ctrl[:env.action_dim].max():+.6f}")
            assert np.allclose(env.action_mujoco, env.data.ctrl[:env.action_dim], atol=1e-7), "ctrl mismatch!"
            assert np.all(env.action_mujoco >= -1e-9), "negative ctrl found!"

    # -----------------------
    # Build EXPERT
    # -----------------------
    ctrlrange = env.model.actuator_ctrlrange.copy()
    nu = env.action_dim
    act_low  = 0.0 * np.ones(nu, dtype=np.float32)
    act_high = 1.4 * np.ones(nu, dtype=np.float32)

    paths = PcDbCBSPaths(
        bindings_path="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
        input_yaml=template_yaml,
        pc_dbcbs_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
        opt_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt_training.yaml",
        dynobench_base="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
        motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
        work_dir_root="runs/_tmp_pcdbcbs",
        keep_files=True,
        warmstart_optimization=True,
        N_opt=100,
    )

    expert = PcDbCBSExpert(paths=paths, act_low=act_low, act_high=act_high, replan_every_k=100)
    expert.reset_episode()

    def expert_fn(obs):
        return expert.act(obs)

    # -----------------------
    # 3) EXPERT ROLLOUT SUMMARY
    # -----------------------
    expert_roll = None
    if DO_EXPERT_ROLLOUT:
        print("\n=== EXPERT ROLLOUT ===")
        expert.reset_episode()
        expert_roll = rollout_policy(expert_fn, "EXPERT", T=300, verbose_every=20)

        A = expert_roll["acts"]
        C = expert_roll["ctrls"]
        print("\n[EXPERT SUMMARY]")
        print("steps:", expert_roll["steps"])
        print("return:", expert_roll["return"])
        print("final_dist:", expert_roll["final_dist"])
        print("out_of_bounds:", expert_roll["out_of_bounds"])
        print("reached_goal:", expert_roll["reached_goal"])
        print("act min/mean/max:", A.min(), A.mean(), A.max())
        print("ctrl min/mean/max:", C.min(), C.mean(), C.max())

    # -----------------------
    # Load LEARNER (YOU FILL THIS)
    # -----------------------
    learner_policy = None
    # Example idea:
    #   from my_train_script import load_policy_somehow
    #   learner_policy = ...
    #
    # Then define:
    def learner_fn(obs):
        # expects obs np array -> act in [-1,1]
        # must return shape (action_dim,)
        act, _ = learner_policy.predict(obs, deterministic=True)
        return act

    # -----------------------
    # 4) EXPERT vs LEARNER on SAME STATES (MAE)
    # -----------------------
    if DO_EXPERT_VS_LEARNER_MAE:
        assert learner_policy is not None, "Set learner_policy first."
        assert expert_roll is not None, "Run expert rollout first."
        print("\n=== EXPERT vs LEARNER ACTION ERROR ===")

        # reconstruct expert obs sequence by re-rolling expert but logging obs
        obs, _ = env.reset()
        obs_seq = []
        exp_act_seq = []

        expert.reset_episode()
        for t in range(expert_roll["steps"]):
            obs_seq.append(obs.copy())
            a_exp = expert.act(obs).astype(np.float32)
            exp_act_seq.append(a_exp.copy())
            obs, _, term, trunc, _ = env.step(a_exp)
            if term or trunc:
                break

        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        exp_act_seq = np.asarray(exp_act_seq, dtype=np.float32)

        # learner acts on same obs
        learner_acts = []
        for o in obs_seq:
            a_learn = learner_fn(o)
            learner_acts.append(np.asarray(a_learn, dtype=np.float32).reshape(-1))
        learner_acts = np.asarray(learner_acts, dtype=np.float32)

        mae = float(np.mean(np.abs(learner_acts - exp_act_seq)))
        mse = float(np.mean((learner_acts - exp_act_seq) ** 2))
        print("MAE:", mae, "MSE:", mse)
        print("expert act min/mean/max:", exp_act_seq.min(), exp_act_seq.mean(), exp_act_seq.max())
        print("learnr act min/mean/max:", learner_acts.min(), learner_acts.mean(), learner_acts.max())

    # -----------------------
    # 5) LEARNER ROLLOUT SUMMARY
    # -----------------------
    if DO_LEARNER_ROLLOUT:
        assert learner_policy is not None, "Set learner_policy first."
        print("\n=== LEARNER ROLLOUT ===")
        learn_roll = rollout_policy(learner_fn, "LEARNER", T=300, verbose_every=20)

        A = learn_roll["acts"]
        C = learn_roll["ctrls"]
        print("\n[LEARNER SUMMARY]")
        print("steps:", learn_roll["steps"])
        print("return:", learn_roll["return"])
        print("final_dist:", learn_roll["final_dist"])
        print("out_of_bounds:", learn_roll["out_of_bounds"])
        print("reached_goal:", learn_roll["reached_goal"])
        print("act min/mean/max:", A.min(), A.mean(), A.max())
        print("ctrl min/mean/max:", C.min(), C.mean(), C.max())











    # import numpy as np

    # from expert_pcdbcbs import PcDbCBSExpert, PcDbCBSPaths
    # from videos_from_log import VideoConfig, render_from_actions  # same as your PD test

    # xml_path = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    # template_yaml = (
    #     "/home/khaledwahba94/imitation-agile-payload-manipulation/"
    #     "deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/"
    #     "mujocoquadspayload_zerogoal.yaml"
    # )

    # env = PayloadGymEnv(
    #     xml_path=xml_path,
    #     template_yaml_path=template_yaml,
    #     max_steps=500,
    # )

    # print("\n--- Testing with PcDbCBSExpert ---\n")

    # # actuator bounds
    # ctrlrange = env.model.actuator_ctrlrange.copy()
    # nu = env.action_dim
    # act_low  =  0.0 * np.ones(nu)  #ctrlrange[:, 0].astype(np.float32)
    # act_high =  1.4 * np.ones(nu)  #ctrlrange[:, 1].astype(np.float32)

    # paths = PcDbCBSPaths(
    #     bindings_path="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
    #     input_yaml=template_yaml,
    #     pc_dbcbs_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
    #     opt_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt_training.yaml",
    #     dynobench_base="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
    #     motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
    #     work_dir_root="runs/_tmp_pcdbcbs",
    #     keep_files=True,
    #     warmstart_optimization=True,
    #     N_opt=100,                        # <== number of optimization steps
    # )

    # expert = PcDbCBSExpert(
    #     paths=paths,
    #     act_low=act_low,
    #     act_high=act_high,
    #     replan_every_k=100,  # replan every K steps (K=0 means replan every step, which is slow but good for testing)
    # )

    # expert.reset_episode()

    # # ---- rollout ----
    # obs, _ = env.reset()
    # # plot_payload_and_quads_state(
    # #     state=obs[:env.state_dim],
    # #     n_bodies=env.n_bodies,
    # #     radius=0.1,
    # #     save_path="pos.pdf",
    # # )
    # print("Initial payload and quads positions saved to pos.pdf")
    # print("start state: ", obs[:env.state_dim])
    # print("goal state:  ", env.goal)
    # print("start rollout...")
    # # exit()
    # # capture init state for rendering
    # qpos0 = env.data.qpos.copy()
    # qvel0 = env.data.qvel.copy()

    # u_traj = []
    # x_traj = []
    # replan_steps = []

    # T_total = 300
    # for t in range(T_total):
    #     action = expert.act(obs)
    #     if expert.just_replanned:
    #         print(f"[replan] t={t}")
    #         replan_steps.append(t)

    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # store trajectories
    #     u_traj.append(env.action_mujoco.copy())
    #     x_traj.append(np.concatenate([env.data.qpos.copy(), env.data.qvel.copy()]))

    #     state = obs[: env.state_dim]
    #     payload_pos = state[:3]
    #     print(
    #         f"t={t:03d} payload_pos=({payload_pos[0]:+.2f},{payload_pos[1]:+.2f},{payload_pos[2]:+.2f}) "
    #         f"reward={reward:+.3f}"
    #     )

    #     if terminated:
    #         print("Expert reached goal.")
    #         break
    #     if truncated:
    #         print("Episode truncated.")
    #         break

    # print("Replans at:", replan_steps)

    # u_traj = np.asarray(u_traj, dtype=np.float32)
    # x_traj = np.asarray(x_traj, dtype=np.float32)
    # print("Simulation done. u_traj shape:", u_traj.shape)

    # # ---- render ----
    # cfg = VideoConfig(
    #     out_dir="videos/test_pcdbcbs_payload_env",
    #     fps=50,
    #     views=["diag"],
    #     env_min=[-3.0, -3.0, 0],
    #     env_max=[+3.0, +3.0, 1.5],
    # )

    # render_from_actions(
    #     xml_path=xml_path,
    #     init_and_actions_or_npz=(qpos0, qvel0, u_traj),
    #     cfg=cfg,
    # )

    # print("PayloadGymEnv + expert test finished. Videos in:", cfg.out_dir)
