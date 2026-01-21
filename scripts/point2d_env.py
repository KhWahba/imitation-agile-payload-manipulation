import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class Point2DEnv(gym.Env):
    """
    Observation (9,):
      [x, y, xdot, ydot, gx, gy, d1, d2, d3]

    d1..d3 are the 3 smallest contact distances between the robot geom and obstacle geoms:
      - con.dist < 0   penetration (signed)
      - con.dist = 0   touching
      - con.dist > 0   separating distance (only if MuJoCo reports it in contacts)

    If no relevant contacts exist, pads with pad_dist (large positive).
    """

    def __init__(
        self,
        xml_path: str,
        goal=(0.8, -0.2),
        max_steps: int = 500,
        k_obs: int = 3,
        pad_dist: float = 10.0,
        robot_geom_name: str = "ball",
        obstacle_geom_prefix: str = "obs",
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.goal = np.array(goal, dtype=np.float32)
        self.max_steps = int(max_steps)
        self.step_count = 0

        self.k_obs = int(k_obs)
        self.pad_dist = float(pad_dist)

        # Action space from ctrlrange
        ctrlrange = self.model.actuator_ctrlrange
        self.action_space = spaces.Box(
            low=ctrlrange[:, 0].astype(np.float32),
            high=ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        self.goal_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.goal_high = np.array([ 1.0,  1.0], dtype=np.float32)

        # Observation space: 6 base + k_obs distances
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 + self.k_obs,), dtype=np.float32
        )

        # Joint IDs (expects slide joints named jx, jy)
        self.jx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "jx")
        self.jy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "jy")
        if self.jx_id < 0 or self.jy_id < 0:
            raise ValueError("Expected joints named 'jx' and 'jy' in the XML.")

        # --- NEW: cache geom IDs for robot and obstacles ---
        self.robot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_name)
        if self.robot_geom_id < 0:
            raise ValueError(f"Robot geom '{robot_geom_name}' not found in XML.")

        # collect all obstacle geom ids by prefix
        self.obstacle_geom_ids = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name is not None and name.startswith(obstacle_geom_prefix):
                self.obstacle_geom_ids.append(gid)

        # if len(self.obstacle_geom_ids) == 0:
        #     raise ValueError(
        #         f"No obstacle geoms found with prefix '{obstacle_geom_prefix}'. "
        #         f"Rename your obstacle geoms or change obstacle_geom_prefix."
        #     )

        self.obstacle_geom_ids = set(self.obstacle_geom_ids)  # fast membership

    def _get_contact_dists(self) -> np.ndarray:
        """
        Return k_obs smallest contact distances between (robot geom, any obstacle geom).
        Pads with pad_dist if fewer contacts.
        """

        if len(self.obstacle_geom_ids) == 0:
            return np.full((self.k_obs,), self.pad_dist, dtype=np.float32)

        dists = []

        # data.contact is length data.ncon
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)

            # check if this contact involves robot geom and an obstacle geom
            if g1 == self.robot_geom_id and g2 in self.obstacle_geom_ids:
                dists.append(float(con.dist))
            elif g2 == self.robot_geom_id and g1 in self.obstacle_geom_ids:
                dists.append(float(con.dist))

        if len(dists) == 0:
            return np.full((self.k_obs,), self.pad_dist, dtype=np.float32)

        dists = np.sort(np.asarray(dists, dtype=np.float32))  # smallest first
        if dists.shape[0] >= self.k_obs:
            return dists[: self.k_obs]

        pad = np.full((self.k_obs - dists.shape[0],), self.pad_dist, dtype=np.float32)
        return np.concatenate([dists, pad], axis=0)

    def _get_obs(self) -> np.ndarray:
        qadr_x = self.model.jnt_qposadr[self.jx_id]
        qadr_y = self.model.jnt_qposadr[self.jy_id]
        dadr_x = self.model.jnt_dofadr[self.jx_id]
        dadr_y = self.model.jnt_dofadr[self.jy_id]

        x = float(self.data.qpos[qadr_x])
        y = float(self.data.qpos[qadr_y])
        xdot = float(self.data.qvel[dadr_x])
        ydot = float(self.data.qvel[dadr_y])

        dists = self._get_contact_dists()  # (k_obs,)

        base = np.array([x, y, xdot, ydot, self.goal[0], self.goal[1]], dtype=np.float32)
        return np.concatenate([base, dists], axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.model.jnt_qposadr[self.jx_id]] = self.np_random.uniform(-0.5, 0.5)
        self.data.qpos[self.model.jnt_qposadr[self.jy_id]] = self.np_random.uniform(-0.5, 0.5)
        self.data.qvel[:] = 0.0

        self.goal = self.np_random.uniform(
                low=self.goal_low,
                high=self.goal_high,
            ).astype(np.float32)


        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()

        # reward: go to goal, plus collision penalty based on negative contact dists
        pos = obs[:2]
        dist_goal = float(np.linalg.norm(pos - self.goal))
        contact_dists = obs[6:]
        collision_pen = float(np.sum(np.minimum(contact_dists, 0.0) ** 2))

        reward = -dist_goal - 10.0 * collision_pen

        terminated = dist_goal < 0.05
        truncated = self.step_count >= self.max_steps
        info = {"dist_goal": dist_goal, "contact_dists": contact_dists.copy()}
        return obs, reward, terminated, truncated, info




if __name__ == "__main__":
    xml_path = "../envs/point2d.xml"   # adjust if needed

    env = Point2DEnv(
        xml_path=xml_path,
        goal=(0.8, -0.2),
        max_steps=200,
        k_obs=3,
        pad_dist=10.0,
        robot_geom_name="ball",        # MUST match XML
        obstacle_geom_prefix="obs",    # MUST match XML
    )

    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Obs dim:", obs.shape)

    for t in range(50):
        # random action inside bounds
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        dists = obs[6:]
        print(
            f"t={t:03d}  pos=({obs[0]:+.2f},{obs[1]:+.2f})  "
            f"dists={dists}  reward={reward:+.3f} "
            f"action={action}"
        )

        if terminated:
            print("Reached goal.")
            break
        if truncated:
            print("Episode truncated.")
            break
    print("Contact-based env test finished.")






