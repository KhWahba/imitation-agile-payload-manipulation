# train_dagger_payload.py
#
# DAgger training for cooperative aerial payload transport.
#
# Key design decisions:
#   1. Pre-train BC on seed expert trajectories BEFORE DAgger rounds
#      (so the learner survives long enough to collect useful on-policy data).
#   2. Expert replans every K steps during DAgger labeling (reactive expert).
#   3. Warmstart optimization enabled for planner diversity.
#   4. Entropy regularization to prevent variance collapse.

import os
import pickle
import time
from pathlib import Path
from typing import Any, Tuple, List, Optional

import gymnasium as gym
import numpy as np
import torch as th
import yaml

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from imitation.algorithms import bc
from imitation.algorithms.dagger import LinearBetaSchedule, SimpleDAggerTrainer
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout

# ---- your env + expert ----
from payload_env import PayloadGymEnv
from expert_pcdbcbs import PcDbCBSExpert, PcDbCBSPaths, ExpertPolicySB3

# ---- optional: video rendering ----
try:
    from videos_from_log import VideoConfig, render_from_actions
    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False
    print("[WARN] videos_from_log not available — skipping video rendering")


# =========================================================================
# Helpers
# =========================================================================

def _unwrap_env(e: Any) -> Any:
    """Unwrap nested wrappers until we reach the base PayloadGymEnv."""
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


class ExpertResetOnEnvResetWrapper(gym.Wrapper):
    """Calls expert.reset_episode() whenever the wrapped env resets."""

    def __init__(self, env, expert: PcDbCBSExpert):
        super().__init__(env)
        self._expert = expert

    def reset(self, *args, **kwargs):
        self._expert.reset_episode()
        return super().reset(*args, **kwargs)


def collect_expert_trajs(
    env_fn,
    expert: PcDbCBSExpert,
    n_episodes: int,
    *,
    xml_path: str = "",
    logs_dir: str = "",
    video_root_dir: str = "",
    render_videos: bool = True,
    seed_delta0_randomize: bool = False,
    seed_delta0_low: float = 0.7,
    seed_delta0_high: float = 0.9,
    seed_delta0_rng: Optional[np.random.Generator] = None,
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
    logs_path = Path(logs_dir) if logs_dir else None
    if logs_path:
        logs_path.mkdir(parents=True, exist_ok=True)

    video_root_path = Path(video_root_dir) if video_root_dir else None
    if video_root_path:
        video_root_path.mkdir(parents=True, exist_ok=True)

    trajs: List[Trajectory] = []
    npz_paths: List[Path] = []
    original_pc_cfg_path = expert.paths.pc_dbcbs_cfg_yaml
    seed_cfg_dir = None
    if seed_delta0_randomize:
        if seed_delta0_rng is None:
            seed_delta0_rng = np.random.default_rng(123)
        if not (seed_delta0_high > seed_delta0_low):
            raise ValueError(f"Invalid seed delta_0 range: [{seed_delta0_low}, {seed_delta0_high}]")
        if logs_path is not None:
            seed_cfg_dir = logs_path.parent / "pcdbcbs_seed_cfgs"
        else:
            seed_cfg_dir = Path("runs/_tmp_pcdbcbs_seed_cfgs")
        seed_cfg_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(n_episodes):
        if seed_delta0_randomize:
            delta0 = float(seed_delta0_rng.uniform(seed_delta0_low, seed_delta0_high))
            with open(original_pc_cfg_path, "r", encoding="utf-8") as f:
                pc_cfg = yaml.safe_load(f)
            pc_cfg.setdefault("pc-dbcbs", {}).setdefault("default", {})["delta_0"] = delta0
            patched_pc_cfg = seed_cfg_dir / f"pc_dbcbs_seed_ep_{ep:03d}.yaml"
            with open(patched_pc_cfg, "w", encoding="utf-8") as f:
                yaml.safe_dump(pc_cfg, f, sort_keys=False)
            expert.paths.pc_dbcbs_cfg_yaml = str(patched_pc_cfg)
            print(f"[seed] ep={ep:03d} randomized delta_0={delta0:.3f}")

        env = env_fn()
        base_env = _unwrap_env(env)

        expert.reset_episode()
        obs, info = env.reset()

        # capture init mujoco state for replay
        qpos0 = base_env.data.qpos.copy()
        qvel0 = base_env.data.qvel.copy()

        obs_list = [np.array(obs, dtype=np.float32)]
        act_list: List[np.ndarray] = []
        u_traj_mujoco: List[np.ndarray] = []
        terminated = False
        truncated = False
        info_list: List[dict] = []

        while not (terminated or truncated):
            act = expert.act(obs).astype(np.float32)
            next_obs, rew, terminated, truncated, step_info = env.step(act)

            act_list.append(act.copy())
            u_traj_mujoco.append(base_env.action_mujoco.copy())
            info_list.append(dict(step_info) if isinstance(step_info, dict) else {})

            obs = next_obs
            obs_list.append(np.array(obs, dtype=np.float32))

        # --- save NPZ log ---
        npz_path = None
        if logs_path:
            u_arr = np.asarray(u_traj_mujoco, dtype=np.float32)
            npz_path = logs_path / f"expert_ep_{ep:03d}.npz"
            np.savez_compressed(npz_path, qpos0=qpos0, qvel0=qvel0, u_traj=u_arr)
            npz_paths.append(npz_path)
            print(f"[seed] ep={ep:03d}  T={len(act_list)}  "
                  f"terminated={terminated}  truncated={truncated}")

        # --- render videos (one directory per episode) ---
        if render_videos and HAS_VIDEO and npz_path is not None and video_root_path is not None and xml_path:
            ep_video_dir = video_root_path / f"expert_ep_{ep:03d}"
            cfg = VideoConfig(
                out_dir=str(ep_video_dir),
                fps=50,
                views=["diag"],
                env_min=np.array([-2.0, -2.0, 0.0], dtype=float),
                env_max=np.array([+2.0, +2.0, 2.0], dtype=float),
            )
            written = render_from_actions(
                xml_path=xml_path,
                init_and_actions_or_npz=str(npz_path),
                cfg=cfg,
            )
            print(f"[seed] rendered: {written}")

        # --- build imitation Trajectory ---
        trajs.append(
            Trajectory(
                obs=np.stack(obs_list, axis=0),
                acts=np.stack(act_list, axis=0),
                infos=info_list,
                terminal=bool(terminated),
            )
        )

        try:
            env.close()
        except Exception:
            pass

    # Restore original pc-dbCBS config path for safety.
    expert.paths.pc_dbcbs_cfg_yaml = original_pc_cfg_path

    return trajs, npz_paths


# =========================================================================
# Main
# =========================================================================

def main():
    t0_total = time.perf_counter()
    rng = np.random.default_rng(42)
    th.manual_seed(42)

    # =====================================================================
    # 1. PATHS — update these to match your machine
    # =====================================================================
    xml_path = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
    template_yaml = "/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"

    paths = PcDbCBSPaths(
        bindings_path="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/build",
        input_yaml=template_yaml,
        pc_dbcbs_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml",
        opt_cfg_yaml="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/configs/opt_training.yaml",
        dynobench_base="/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/dynoplan/dynobench/",
        motion_primitives_base="/home/khaledwahba94/pc-dbCBS/motion_primitives/",
        time_limit=50000.0,
        work_dir_root="runs/_tmp_pcdbcbs",
        keep_files=False,            # save disk during DAgger
        warmstart_optimization=True, # planner diversity
        N_opt=30,
    )

    out_root = Path(os.environ.get("DAGGER_OUT_ROOT", "runs/dagger_2000steps"))
    out_root.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # 2. ENV FACTORY
    # =====================================================================
    MAX_STEPS = 200

    def make_env(max_steps: int = MAX_STEPS):
        env = PayloadGymEnv(
            xml_path=xml_path,
            template_yaml_path=template_yaml,
            max_steps=max_steps,
        )
        env.terminate_on_success = True  # end episodes on success (goal reached)
        env = Monitor(env)
        env = RolloutInfoWrapper(env)
        return env

    # =====================================================================
    # 3. EXPERT
    # =====================================================================
    tmp_env = PayloadGymEnv(xml_path=xml_path, template_yaml_path=template_yaml, max_steps=MAX_STEPS)
    nu = tmp_env.action_dim
    act_low  = 0.0 * np.ones(nu, dtype=np.float32)
    act_high = 1.4 * np.ones(nu, dtype=np.float32)

    # --- Expert for seed data collection (replan_every_k=0 → replan on exhaustion only) ---
    expert_seed = PcDbCBSExpert(
        paths=PcDbCBSPaths(
            bindings_path=paths.bindings_path,
            input_yaml=paths.input_yaml,
            pc_dbcbs_cfg_yaml=paths.pc_dbcbs_cfg_yaml,
            opt_cfg_yaml=paths.opt_cfg_yaml,
            dynobench_base=paths.dynobench_base,
            motion_primitives_base=paths.motion_primitives_base,
            time_limit=paths.time_limit,
            work_dir_root=paths.work_dir_root,
            keep_files=False,
            warmstart_optimization=True,
            N_opt=30,
        ),
        act_low=act_low,
        act_high=act_high,
        replan_every_k=0,  # let it play out the full plan
    )

    dagger_replan_k = int(os.environ.get("DAGGER_EXPERT_REPLAN_K", "10"))
    disable_exhausted_replan = os.environ.get("DAGGER_DISABLE_EXHAUSTED_REPLAN", "0") == "1"
    exhausted_hold_steps = int(os.environ.get("DAGGER_EXHAUSTED_HOLD_STEPS", "0"))
    max_replans_per_episode = int(os.environ.get("DAGGER_MAX_REPLANS_PER_EPISODE", "0"))
    n_envs = max(1, int(os.environ.get("DAGGER_NUM_ENVS", "1")))
    expert_threads = max(1, int(os.environ.get("DAGGER_EXPERT_THREADS", str(n_envs))))

    # --- Experts for DAgger labeling (one expert per env slot) ---
    expert_daggers = [
        PcDbCBSExpert(
            paths=PcDbCBSPaths(
                bindings_path=paths.bindings_path,
                input_yaml=paths.input_yaml,
                pc_dbcbs_cfg_yaml=paths.pc_dbcbs_cfg_yaml,
                opt_cfg_yaml=paths.opt_cfg_yaml,
                dynobench_base=paths.dynobench_base,
                motion_primitives_base=paths.motion_primitives_base,
                time_limit=paths.time_limit,
                work_dir_root=paths.work_dir_root,
                keep_files=paths.keep_files,
                warmstart_optimization=paths.warmstart_optimization,
                N_opt=paths.N_opt,
            ),
            act_low=act_low,
            act_high=act_high,
            replan_every_k=dagger_replan_k,
            disable_exhausted_replan=disable_exhausted_replan,
            exhausted_hold_steps=exhausted_hold_steps,
            max_replans_per_episode=max_replans_per_episode,
        )
        for _ in range(n_envs)
    ]

    def _make_wrapped_env_with_expert(expert_obj: PcDbCBSExpert):
        return ExpertResetOnEnvResetWrapper(make_env(), expert_obj)

    venv = DummyVecEnv(
        [lambda e=expert_daggers[i]: _make_wrapped_env_with_expert(e) for i in range(n_envs)]
    )

    expert_policy = ExpertPolicySB3(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        expert=expert_daggers,
        device="cpu",
        max_workers=expert_threads,
    )

    # =====================================================================
    # 4. COLLECT SEED EXPERT TRAJECTORIES (round-0)
    # =====================================================================
    # Observation/layout + seed randomization policy changed; use a versioned seed cache path.
    seed_delta0_randomize = os.environ.get("SEED_RANDOMIZE_DELTA0", "1") == "1"
    seed_delta0_low = float(os.environ.get("SEED_RANDOMIZE_DELTA0_LOW", "0.7"))
    seed_delta0_high = float(os.environ.get("SEED_RANDOMIZE_DELTA0_HIGH", "0.9"))
    seed_cache_tag = (
        f"round0_cache_obs_payload_rel_v2_seeddelta_{seed_delta0_low:.2f}_{seed_delta0_high:.2f}"
        if seed_delta0_randomize else
        "round0_cache_obs_payload_rel_v2_noseeddelta"
    )
    round0_dir = out_root / seed_cache_tag
    round0_dir.mkdir(parents=True, exist_ok=True)
    trajs_pkl = round0_dir / "expert_trajs.pkl"

    N_SEED_EPISODES = int(os.environ.get("DAGGER_N_SEED_EPISODES", "20"))

    if trajs_pkl.exists():
        t0_seed = time.perf_counter()
        with open(trajs_pkl, "rb") as f:
            expert_trajs = pickle.load(f)
        print(f"Loaded cached seed trajs: {trajs_pkl} (n={len(expert_trajs)})")
        print(f"Seed cache load time: {time.perf_counter() - t0_seed:.1f}s")
    else:
        t0_seed = time.perf_counter()
        render_seed_videos = os.environ.get("SEED_RENDER_VIDEOS", "0") == "1"
        expert_trajs, _ = collect_expert_trajs(
            env_fn=lambda: make_env(max_steps=MAX_STEPS),
            expert=expert_seed,
            n_episodes=N_SEED_EPISODES,
            xml_path=xml_path,
            logs_dir=str(round0_dir / "npz_logs"),
            video_root_dir=str(out_root / "videos_round0"),
            render_videos=render_seed_videos,
            seed_delta0_randomize=seed_delta0_randomize,
            seed_delta0_low=seed_delta0_low,
            seed_delta0_high=seed_delta0_high,
            seed_delta0_rng=rng,
        )
        with open(trajs_pkl, "wb") as f:
            pickle.dump(expert_trajs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved seed trajs: {trajs_pkl} (n={len(expert_trajs)})")
        print(f"Seed collection time: {time.perf_counter() - t0_seed:.1f}s")

    # Print seed data statistics
    total_steps = sum(t.acts.shape[0] for t in expert_trajs)
    print(f"\nSeed data: {len(expert_trajs)} episodes, {total_steps} transitions")
    all_acts = np.concatenate([t.acts for t in expert_trajs], axis=0)
    print(f"Action stats: min={all_acts.min():.3f} mean={all_acts.mean():.3f} max={all_acts.max():.3f}")

    # =====================================================================
    # 5. BC PRE-TRAINING (critical for DAgger to work)
    # =====================================================================
    bc_trim_last_steps = int(os.environ.get("DAGGER_BC_TRIM_LAST_STEPS", "0"))
    bc_trim_mode = os.environ.get("DAGGER_BC_TRIM_MODE", "failed_only").strip().lower()
    if bc_trim_mode not in {"failed_only", "all", "none"}:
        raise ValueError(f"Unsupported DAGGER_BC_TRIM_MODE={bc_trim_mode}. Use failed_only|all|none.")
    bc_trajs = list(expert_trajs)
    if bc_trim_last_steps > 0 and bc_trim_mode != "none":
        trimmed: List[Trajectory] = []
        dropped = 0
        unknown_fail_info = 0
        for tr in expert_trajs:
            T = int(tr.acts.shape[0])

            should_trim = False
            if bc_trim_mode == "all":
                should_trim = True
            elif bc_trim_mode == "failed_only":
                if tr.infos is None or len(tr.infos) == 0:
                    unknown_fail_info += 1
                    should_trim = False
                else:
                    last_info = tr.infos[-1]
                    if isinstance(last_info, dict):
                        should_trim = bool(
                            last_info.get("out_of_bounds", False)
                            or last_info.get("payload_out_of_bounds", False)
                            or last_info.get("quad_out_of_bounds", False)
                        )
            keep_T = max(1, T - bc_trim_last_steps) if should_trim else T
            if keep_T < T:
                dropped += (T - keep_T)
            trimmed.append(
                Trajectory(
                    obs=tr.obs[: keep_T + 1].copy(),
                    acts=tr.acts[:keep_T].copy(),
                    infos=None if tr.infos is None else tr.infos[:keep_T],
                    terminal=bool(tr.terminal and (keep_T == T)),
                )
            )
        bc_trajs = trimmed
        print(
            f"BC tail trim: dropped {dropped} transitions "
            f"(trim_last_steps={bc_trim_last_steps}, mode={bc_trim_mode}, "
            f"unknown_fail_info={unknown_fail_info})"
        )

    transitions = rollout.flatten_trajectories(bc_trajs)
    print(f"\nBC pre-training on {len(transitions)} transitions ...")

    bc_batch_size = min(256, len(transitions))
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=bc_batch_size,
        optimizer_kwargs=dict(lr=1e-3),
        ent_weight=1e-3,  # prevent variance collapse
    )

    BC_PRETRAIN_EPOCHS = int(os.environ.get("DAGGER_BC_PRETRAIN_EPOCHS", "400"))
    t0_bc = time.perf_counter()
    bc_trainer.train(n_epochs=BC_PRETRAIN_EPOCHS)
    print(f"BC pre-training done ({BC_PRETRAIN_EPOCHS} epochs)")
    print(f"BC wall time: {time.perf_counter() - t0_bc:.1f}s")

    # Quick sanity check
    obs = venv.reset()
    for i in range(3):
        act, _ = bc_trainer.policy.predict(obs, deterministic=True)
        print(f"  BC sanity check step {i}: act min={act.min():.3f} mean={act.mean():.3f} max={act.max():.3f}")
        obs, _, _, _ = venv.step(act)

    # =====================================================================
    # 6. DAgger TRAINING
    # =====================================================================
    scratch_dir = out_root / os.environ.get("DAGGER_SCRATCH_DIRNAME", "scratch_dagger")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    beta_mode = os.environ.get("DAGGER_BETA_MODE", "linear").strip().lower()
    beta_rampdown_rounds = int(os.environ.get("DAGGER_BETA_RAMPDOWN_ROUNDS", "15"))
    beta_constant = float(os.environ.get("DAGGER_BETA_CONSTANT", "0.5"))
    if beta_mode == "linear":
        beta_schedule = LinearBetaSchedule(max(1, beta_rampdown_rounds))
    elif beta_mode == "constant":
        beta_schedule = (lambda _r: float(np.clip(beta_constant, 0.0, 1.0)))
    else:
        raise ValueError(f"Unsupported DAGGER_BETA_MODE={beta_mode}. Use linear|constant.")

    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=scratch_dir,
        expert_policy=expert_policy,
        bc_trainer=bc_trainer,
        rng=rng,
        expert_trajs=expert_trajs,  # seed the dataset
        beta_schedule=beta_schedule,
    )

    TOTAL_TIMESTEPS = int(os.environ.get("DAGGER_TOTAL_TIMESTEPS", "2000"))
    ROLLOUT_MIN_EPISODES = int(os.environ.get("DAGGER_ROLLOUT_MIN_EPISODES", "1"))
    ROLLOUT_MIN_TIMESTEPS = MAX_STEPS * n_envs

    print(f"\n{'='*60}")
    print(f"Starting DAgger training")
    print(f"  total_timesteps:    {TOTAL_TIMESTEPS}")
    print(f"  min_episodes/round: {ROLLOUT_MIN_EPISODES}")
    print(f"  min_steps/round:    {ROLLOUT_MIN_TIMESTEPS}")
    print(f"  n_envs:                   {n_envs}")
    print(f"  expert_threads:           {expert_threads}")
    print(f"  expert replan_k:          {expert_daggers[0].replan_every_k}")
    print(f"  disable_exhausted_replan: {disable_exhausted_replan}")
    print(f"  exhausted_hold_steps:     {exhausted_hold_steps}")
    print(f"  max_replans/episode:      {max_replans_per_episode}")
    print(f"  beta_mode:                {beta_mode}")
    if beta_mode == "linear":
        print(f"  beta_rampdown_rounds:     {beta_rampdown_rounds}")
    else:
        print(f"  beta_constant:            {beta_constant}")
    print(f"  BC trim tail steps:       {bc_trim_last_steps}")
    print(f"  BC trim mode:             {bc_trim_mode}")
    print(f"  expert N_opt:       {paths.N_opt}")
    print(f"  terminate_on_success: {True}")
    print(f"{'='*60}\n")

    t0_dagger = time.perf_counter()
    dagger_trainer.train(
        total_timesteps=TOTAL_TIMESTEPS,
        rollout_round_min_episodes=ROLLOUT_MIN_EPISODES,
        rollout_round_min_timesteps=ROLLOUT_MIN_TIMESTEPS,
    )
    print(f"DAgger wall time: {time.perf_counter() - t0_dagger:.1f}s")

    # =====================================================================
    # 7. EVALUATION
    # =====================================================================
    print(f"\n{'='*60}")
    print("POST-TRAINING EVALUATION")
    print(f"{'='*60}")

    obs = venv.reset()
    print("\nLearner action samples:")
    for i in range(5):
        act, _ = dagger_trainer.policy.predict(obs, deterministic=True)
        print(f"  step {i}: act min={act.min():.3f} mean={act.mean():.3f} max={act.max():.3f}")
        obs, _, _, _ = venv.step(act)

    n_eval = 5
    mean_return, std_return = evaluate_policy(
        dagger_trainer.policy, venv, n_eval_episodes=n_eval, deterministic=True
    )
    print(f"\nDAgger policy: mean_return={mean_return:.2f} ± {std_return:.2f} (n={n_eval})")

    expert_seed.replan_every_k = 0
    expert_eval_policy = ExpertPolicySB3(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        expert=expert_seed,
        device="cpu",
    )
    expert_return, expert_std = evaluate_policy(
        expert_eval_policy, venv, n_eval_episodes=n_eval, deterministic=True
    )
    print(f"Expert policy: mean_return={expert_return:.2f} ± {expert_std:.2f} (n={n_eval})")

    # =====================================================================
    # 8. SAVE
    # =====================================================================
    policy_path = out_root / "dagger_policy.pt"
    th.save(dagger_trainer.policy.state_dict(), policy_path)
    print(f"\nSaved DAgger policy to: {policy_path}")
    print(f"Total wall time: {time.perf_counter() - t0_total:.1f}s")
    print(f"Done! Output directory: {out_root}")


if __name__ == "__main__":
    main()
