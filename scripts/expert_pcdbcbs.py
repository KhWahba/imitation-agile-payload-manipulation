# expert_pcdbcbs.py
import os
import sys
import tempfile
import shutil
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy


@dataclass
class PcDbCBSPaths:
    bindings_path: str
    input_yaml: str
    pc_dbcbs_cfg_yaml: str
    opt_cfg_yaml: str
    dynobench_base: str
    motion_primitives_base: str
    time_limit: float = 350000.0
    work_dir_root: Optional[str] = None
    keep_files: bool = False
    warmstart_optimization: bool = True


class PcDbCBSExpert:
    def __init__(
        self,
        paths: PcDbCBSPaths,
        act_low: np.ndarray,
        act_high: np.ndarray,
        replan_every_k: int = 0,
        u_nominal: float = 0.034 * 9.81 / 4,
    ):
        self.paths = paths
        self.act_low = np.asarray(act_low, dtype=np.float32)
        self.act_high = np.asarray(act_high, dtype=np.float32)

        self._template_input_yaml = paths.input_yaml

        if paths.bindings_path not in sys.path:
            sys.path.append(paths.bindings_path)

        try:
            import pcdbcbs  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Could not import pcdbcbs. "
                f"Check bindings_path={paths.bindings_path}\n"
                f"Original error: {e}"
            )

        import pcdbcbs
        self.pcdbcbs = pcdbcbs

        self._U: Optional[np.ndarray] = None
        self._t: int = 0
        self._global_step: int = 0

        self.u_nominal = float(u_nominal)
        self.replan_every_k = int(replan_every_k)

        self.size_u: Optional[int] = None
        self.just_replanned = False

        # Load goal once from template YAML (also defines state_dim)
        with open(self._template_input_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        goal = np.asarray(cfg["joint_robot"][0]["goal"], dtype=np.float32).reshape(-1)

        self._state_dim = int(goal.shape[0])         # <-- IMPORTANT for slicing
        self._payload_goal_pos = goal[:3].copy()

        self.goal_switch_dist = 0.2  # meters

    @property
    def planned(self) -> bool:
        return self._U is not None

    def reset_episode(self):
        self._U = None
        self._t = 0
        self._global_step = 0

    def _make_work_dir(self) -> str:
        root = self.paths.work_dir_root
        if root is not None:
            os.makedirs(root, exist_ok=True)

        if self.paths.keep_files:
            return tempfile.mkdtemp(dir=root, prefix="pcdbcbs_")
        return ""

    def _plan(self):
        opt = self.pcdbcbs.Options()
        opt.override_visualize_mujoco = False
        opt.visualize_mujoco = False

        opt.input_yaml = self.paths.input_yaml
        opt.pc_dbcbs_cfg_yaml = self.paths.pc_dbcbs_cfg_yaml
        opt.opt_cfg_yaml = self.paths.opt_cfg_yaml
        opt.time_limit = float(self.paths.time_limit)

        opt.dynobench_base = self.paths.dynobench_base
        opt.motion_primitives_base = self.paths.motion_primitives_base
        opt.warmstart_optimization = self.paths.warmstart_optimization

        if self.paths.keep_files:
            work_dir = self._make_work_dir()
            opt.output_yaml = os.path.join(work_dir, "result_dbcbs.yaml")
            opt.optimization_yaml = os.path.join(work_dir, "result_dbcbs_opt.yaml")
            res = self.pcdbcbs.run(opt)
        else:
            if self.paths.work_dir_root is not None:
                os.makedirs(self.paths.work_dir_root, exist_ok=True)

            with tempfile.TemporaryDirectory(dir=self.paths.work_dir_root, prefix="pcdbcbs_") as work_dir:
                opt.output_yaml = os.path.join(work_dir, "result_dbcbs.yaml")
                opt.optimization_yaml = os.path.join(work_dir, "result_dbcbs_opt.yaml")
                res = self.pcdbcbs.run(opt)

        if not getattr(res, "feasible", False) or getattr(res, "U", np.array([])).size == 0:
            raise RuntimeError(
                f"pc-dbCBS failed or returned empty U. "
                f"feasible={getattr(res,'feasible',None)} solved_db={getattr(res,'solved_db',None)} "
                f"solved_opt={getattr(res,'solved_opt',None)}"
            )

        U = np.asarray(res.U, dtype=np.float32)
        if U.ndim != 2:
            raise RuntimeError(f"Expected res.U shape (T, nu), got {U.shape}")

        U = U * self.u_nominal
        U = np.clip(U, self.act_low[None, :], self.act_high[None, :])

        self._U = U
        self._t = 0
        self.size_u = int(U.shape[1])

    def _write_patched_input_yaml(self, state: np.ndarray) -> Tuple[str, str]:
        """
        state is the physical state only (13*n):
          [payload_pose(7), quad1_pose(7), ..., quadM_pose(7),
           payload_vel(6),  quad1_vel(6),  ..., quadM_vel(6)]
        """
        with open(self._template_input_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        state = np.asarray(state, dtype=float).reshape(-1)

        if state.size % 13 != 0:
            raise ValueError(f"state length {state.size} is not 13*n")

        # ---- existing behavior: patch joint_robot start ----
        cfg["joint_robot"][0]["start"] = state.tolist()

        # ---- MINIMAL FIX: also patch robots[i].start from joint_robot state ----
        # n_bodies = 1(payload) + n_quads
        n_bodies = state.size // 13
        n_quads = n_bodies - 1

        # If robots list exists and lengths match, set each robot start = [quad_pose(7), quad_vel(6)]
        if "robots" in cfg and isinstance(cfg["robots"], list) and len(cfg["robots"]) >= n_quads:
            # poses block length = 7*n_bodies
            pose_block = state[: 7 * n_bodies]
            vel_block = state[7 * n_bodies : 7 * n_bodies + 6 * n_bodies]

            for i in range(n_quads):
                quad_pose = pose_block[7 * (1 + i) : 7 * (1 + i + 1)]   # skip payload pose
                quad_vel = vel_block[6 * (1 + i) : 6 * (1 + i + 1)]     # skip payload vel
                cfg["robots"][i]["start"] = np.concatenate([quad_pose, quad_vel]).tolist()

        if self.paths.work_dir_root is not None:
            os.makedirs(self.paths.work_dir_root, exist_ok=True)

        tmp_dir = tempfile.mkdtemp(dir=self.paths.work_dir_root, prefix="pcdbcbs_input_")
        patched_yaml = os.path.join(tmp_dir, "input.yaml")

        with open(patched_yaml, "w") as f:
            yaml.safe_dump(cfg, f)

        return patched_yaml, tmp_dir

    def _plan_from_state(self, state: np.ndarray):
        patched_yaml, tmp_dir = self._write_patched_input_yaml(state)
        old = self.paths.input_yaml
        try:
            self.paths.input_yaml = patched_yaml
            self._plan()
        finally:
            self.paths.input_yaml = old
            if not self.paths.keep_files:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.just_replanned = False

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        state = obs[: self._state_dim]   # <-- slice state only

        # warmstart switching based on payload position (state[:3])
        payload_pos = state[:3]
        dist_to_goal = float(np.linalg.norm(payload_pos - self._payload_goal_pos))
        use_warmstart = dist_to_goal > self.goal_switch_dist

        need_first_plan = (self._U is None)
        plan_exhausted = (self._U is not None and self._t >= self._U.shape[0])
        need_periodic_replan = (self.replan_every_k > 0 and self._global_step % self.replan_every_k == 0)

        if need_first_plan or plan_exhausted or need_periodic_replan:
            old_ws = self.paths.warmstart_optimization
            try:
                self.paths.warmstart_optimization = bool(use_warmstart)
                self._plan_from_state(state)
            finally:
                self.paths.warmstart_optimization = old_ws

            self._t = 0
            self.just_replanned = True

        u = self._U[self._t].copy()
        self._t += 1
        self._global_step += 1
        return u.astype(np.float32)


class ExpertPolicySB3(BasePolicy):
    def __init__(self, observation_space, action_space, expert: PcDbCBSExpert, device="cpu"):
        super().__init__(observation_space, action_space)
        self.expert = expert
        self._device = th.device(device)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        obs_np = observation.detach().cpu().numpy()
        acts = [self.expert.act(o) for o in obs_np]
        acts_np = np.stack(acts, axis=0).astype(np.float32)
        return th.as_tensor(acts_np, dtype=th.float32, device=observation.device)
