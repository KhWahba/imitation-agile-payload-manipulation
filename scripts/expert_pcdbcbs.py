# expert_pcdbcbs.py
import os
import sys
import tempfile
import shutil
import yaml
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence, Union

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
    N_opt: int = 50 


class PcDbCBSExpert:
    def __init__(
        self,
        paths: PcDbCBSPaths,
        act_low: np.ndarray,
        act_high: np.ndarray,
        replan_every_k: int = 0,
        u_nominal: float = 0.034 * 9.81 / 4,
        goal_switch_dist: float = 0.2,  # meters
        disable_exhausted_replan: bool = False,
        exhausted_hold_steps: int = 0,
        max_replans_per_episode: int = 0,
    ):
        self.paths = paths
        self.act_low = np.asarray(act_low, dtype=np.float32).reshape(-1)
        self.act_high = np.asarray(act_high, dtype=np.float32).reshape(-1)

        # Derived scaling: planner outputs in [act_low, act_high]
        # Policy space is [-1, 1]
        # Linear map: policy = (planner - mid) / half_range
        #             planner = policy * half_range + mid
        self._act_mid = 0.5 * (self.act_low + self.act_high)    # midpoint
        self._act_half = 0.5 * (self.act_high - self.act_low)   # half-range
        if np.any(self._act_half <= 0):
            raise ValueError(f"act_high must be > act_low everywhere. "
                             f"Got act_low={self.act_low}, act_high={self.act_high}")

        self._template_input_yaml = paths.input_yaml

        # Force a predictable work root (so results/input/meta stay together)
        if self.paths.work_dir_root is None:
            self.paths.work_dir_root = "runs/_tmp_pcdbcbs"
        os.makedirs(self.paths.work_dir_root, exist_ok=True)

        if paths.bindings_path not in sys.path:
            sys.path.append(paths.bindings_path)

        try:
            import pcdbcbs  # type: ignore # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Could not import pcdbcbs. "
                f"Check bindings_path={paths.bindings_path}\n"
                f"Original error: {e}"
            )

        import pcdbcbs # pyright: ignore[reportMissingImports]
        self.pcdbcbs = pcdbcbs

        self._U: Optional[np.ndarray] = None
        self._t: int = 0
        self._global_step: int = 0
        self._episode_id: int = 0
        self.last_replan_reason: Optional[str] = None

        self.u_nominal = float(u_nominal)
        self.replan_every_k = int(replan_every_k)
        self.goal_switch_dist = float(goal_switch_dist)
        self.disable_exhausted_replan = bool(disable_exhausted_replan)
        self.exhausted_hold_steps = max(0, int(exhausted_hold_steps))
        self.max_replans_per_episode = max(0, int(max_replans_per_episode))

        self.size_u: Optional[int] = None
        self.just_replanned = False
        self._last_u: Optional[np.ndarray] = None
        self._exhausted_hold_count: int = 0
        self._replans_in_episode: int = 0

        # Load goal once from template YAML (also defines state_dim)
        with open(self._template_input_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        goal = np.asarray(cfg["joint_robot"][0]["goal"], dtype=np.float32).reshape(-1)

        self._state_dim = int(goal.shape[0])
        self._payload_goal_pos = goal[:3].copy()

    @property
    def planned(self) -> bool:
        return self._U is not None

    def reset_episode(self):
        self._episode_id += 1
        self._U = None
        self._t = 0
        self._global_step = 0
        self.last_replan_reason = None
        self._last_u = None
        self._exhausted_hold_count = 0
        self._replans_in_episode = 0

    # ----------------------------
    # Directory + bookkeeping
    # ----------------------------
    def _make_plan_dir(self, *, warmstart: bool, reason: str) -> str:
        """
        Create a per-plan directory that will contain:
          input.yaml, meta.yaml, result_dbcbs.yaml, result_dbcbs_opt.yaml, ...
        """
        root = self.paths.work_dir_root
        os.makedirs(root, exist_ok=True)

        # human-readable directory name for debugging
        prefix = (
            f"pcdbcbs_ep{self._episode_id:04d}_"
            f"gs{self._global_step:07d}_"
            f"ws{int(bool(warmstart))}_"
            f"{reason}_"
        )
        return tempfile.mkdtemp(dir=root, prefix=prefix)

    def _write_meta_yaml(self, plan_dir: str, *, reason: str, dist_to_goal: float, warmstart: bool):
        meta = {
            "episode_id": int(self._episode_id),
            "global_step": int(self._global_step),
            "reason": str(reason),
            "dist_to_goal": float(dist_to_goal),
            "warmstart_optimization": bool(warmstart),
            "t_in_plan_before": int(self._t),
        }
        with open(os.path.join(plan_dir, "meta.yaml"), "w") as f:
            yaml.safe_dump(meta, f)

    # ----------------------------
    # YAML patching
    # ----------------------------
    def _write_patched_input_yaml(self, state: np.ndarray, plan_dir: str) -> str:
        """
        Writes plan_dir/input.yaml based on template yaml, but patches start states.

        state is (13*n_bodies):
          [pose_all (7*n), vel_all (6*n)]
        pose_i = [pos3, quat_xyzw4]
        vel_i  = [lin3, ang3]
        """
        with open(self._template_input_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size % 13 != 0:
            raise ValueError(f"state length {state.size} is not 13*n")

        # Patch joint_robot[0].start (your mujoco "joint" state)
        cfg["joint_robot"][0]["start"] = state.tolist()

        # Also patch cfg["robots"][i]["start"] if present (quad-only states)
        n_bodies = state.size // 13
        n_quads = n_bodies - 1

        if "robots" in cfg and isinstance(cfg["robots"], list) and len(cfg["robots"]) >= n_quads:
            pose_block = state[: 7 * n_bodies]
            vel_block = state[7 * n_bodies : 7 * n_bodies + 6 * n_bodies]

            for i in range(n_quads):
                quad_pose = pose_block[7 * (1 + i) : 7 * (2 + i)]   # skip payload
                quad_vel = vel_block[6 * (1 + i) : 6 * (2 + i)]     # skip payload
                cfg["robots"][i]["start"] = np.concatenate([quad_pose, quad_vel]).tolist()

        os.makedirs(plan_dir, exist_ok=True)
        patched_yaml = os.path.join(plan_dir, "input.yaml")
        with open(patched_yaml, "w") as f:
            yaml.safe_dump(cfg, f)

        return patched_yaml

    # ----------------------------
    # Planning
    # ----------------------------
    def _plan(self, input_yaml: str, plan_dir: str, warmstart: bool):
        opt = self.pcdbcbs.Options()
        opt.override_visualize_mujoco = False
        opt.visualize_mujoco = False

        opt.input_yaml = input_yaml
        opt.pc_dbcbs_cfg_yaml = self.paths.pc_dbcbs_cfg_yaml
        opt.opt_cfg_yaml = self.paths.opt_cfg_yaml
        opt.time_limit = float(self.paths.time_limit)

        opt.dynobench_base = self.paths.dynobench_base
        opt.motion_primitives_base = self.paths.motion_primitives_base
        opt.warmstart_optimization = bool(warmstart)
        opt.output_yaml = os.path.join(plan_dir, "result_dbcbs.yaml")
        opt.optimization_yaml = os.path.join(plan_dir, "result_dbcbs_opt.yaml")
        opt.N_opt = self.paths.N_opt
        res = self.pcdbcbs.run(opt)

        if  getattr(res, "U", np.array([])).size == 0:
        # if not getattr(res, "feasible", False) or getattr(res, "U", np.array([])).size == 0:
            raise RuntimeError(
                f"pc-dbCBS failed or returned empty U. "
                f"feasible={getattr(res,'feasible',None)} "
                f"solved_db={getattr(res,'solved_db',None)} "
                f"solved_opt={getattr(res,'solved_opt',None)}"
            )

        U = np.asarray(res.U, dtype=np.float32)
        if U.ndim != 2:
            raise RuntimeError(f"Expected res.U shape (T, nu), got {U.shape}")

        # scale + clamp
        # U = U * self.u_nominal
        # U = np.clip(U, self.act_low[None, :], self.act_high[None, :])

        self._U = U
        self._t = 0
        self.size_u = int(U.shape[1])

    def _plan_from_state(self, state: np.ndarray, plan_dir: str, warmstart: bool):
        input_yaml = self._write_patched_input_yaml(state, plan_dir)
        self._plan(input_yaml=input_yaml, plan_dir=plan_dir, warmstart=warmstart)

    # ----------------------------
    # Acting
    # ----------------------------
    def act(self, obs: np.ndarray) -> np.ndarray:
        self.just_replanned = False
        self.last_replan_reason = None

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        state = obs[: self._state_dim]

        payload_pos = state[:3]
        dist_to_goal = float(np.linalg.norm(payload_pos - self._payload_goal_pos))
        use_warmstart = dist_to_goal > self.goal_switch_dist
        use_warmstart = use_warmstart and self.paths.warmstart_optimization
        need_first_plan = (self._U is None)
        plan_exhausted = (self._U is not None and self._t >= self._U.shape[0])
        need_periodic_replan = (
            self.replan_every_k > 0
            and self._global_step > 0
            and (self._global_step % self.replan_every_k == 0)
        )

        force_hold = False
        if plan_exhausted and (not need_first_plan) and (not need_periodic_replan):
            if self.disable_exhausted_replan and self._last_u is not None:
                force_hold = True
            elif self.exhausted_hold_steps > 0 and self._last_u is not None:
                if self._exhausted_hold_count < self.exhausted_hold_steps:
                    force_hold = True
                    self._exhausted_hold_count += 1

        need_replan_now = need_first_plan or need_periodic_replan or (plan_exhausted and not force_hold)
        if need_replan_now:
            if (
                self.max_replans_per_episode > 0
                and self._replans_in_episode >= self.max_replans_per_episode
                and self._last_u is not None
            ):
                force_hold = True
            else:
                reason = "first" if need_first_plan else ("exhausted" if plan_exhausted else "periodic")
                self.last_replan_reason = reason
                plan_dir = self._make_plan_dir(warmstart=use_warmstart, reason=reason)
                self._write_meta_yaml(plan_dir, reason=reason, dist_to_goal=dist_to_goal, warmstart=use_warmstart)

                try:
                    self._plan_from_state(state, plan_dir=plan_dir, warmstart=use_warmstart)
                finally:
                    if not self.paths.keep_files:
                        shutil.rmtree(plan_dir, ignore_errors=True)

                self._t = 0
                self.just_replanned = True
                self._replans_in_episode += 1
                self._exhausted_hold_count = 0

        if force_hold and self._last_u is not None:
            u = self._last_u.copy()
        else:
            if self._U is None:
                raise RuntimeError("No plan available and no previous action to hold.")
            idx = min(self._t, self._U.shape[0] - 1)
            u = self._U[idx].copy()
            self._t += 1
        self._global_step += 1
        self._last_u = u.copy()

        # Map planner output [act_low, act_high] -> policy space [-1, 1]
        u_centered = (u - self._act_mid) / self._act_half
        u_centered = np.clip(u_centered, -1.0, 1.0)
        return u_centered.astype(np.float32)


class ExpertPolicySB3(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        expert: Union[PcDbCBSExpert, Sequence[PcDbCBSExpert]],
        device="cpu",
        max_workers: int = 1,
    ):
        super().__init__(observation_space, action_space)
        if isinstance(expert, (list, tuple)):
            if len(expert) == 0:
                raise ValueError("expert list must be non-empty")
            self.experts = list(expert)
        else:
            self.experts = [expert]
        self._device = th.device(device)
        self.max_workers = max(1, int(max_workers))

    def _act_single(self, idx: int, obs_1d: np.ndarray) -> np.ndarray:
        if len(self.experts) == 1:
            ex = self.experts[0]
        else:
            if idx >= len(self.experts):
                raise RuntimeError(
                    f"Received batch with {idx+1} envs but only {len(self.experts)} experts."
                )
            ex = self.experts[idx]
        return ex.act(obs_1d)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        obs_np = observation.detach().cpu().numpy()
        batch_n = obs_np.shape[0]
        if self.max_workers > 1 and batch_n > 1:
            workers = min(self.max_workers, batch_n)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                acts = list(pool.map(lambda io: self._act_single(io[0], io[1]), enumerate(obs_np)))
        else:
            acts = [self._act_single(i, o) for i, o in enumerate(obs_np)]
        acts_np = np.stack(acts, axis=0).astype(np.float32)
        return th.as_tensor(acts_np, dtype=th.float32, device=observation.device)
