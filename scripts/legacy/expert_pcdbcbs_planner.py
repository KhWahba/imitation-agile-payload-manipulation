import os
import sys
from typing import Dict

import numpy as np

from expert_pcdbcbs import ExpertPolicySB3, PcDbCBSExpert as _BaseExpert, PcDbCBSPaths


def _planner_stats_to_dict(stats) -> Dict[str, object]:
    return {
        "run_calls": int(getattr(stats, "run_calls", 0)),
        "reset_calls": int(getattr(stats, "reset_calls", 0)),
        "cumulative_run_wall_sec": float(getattr(stats, "cumulative_run_wall_sec", 0.0)),
        "last_run_wall_sec": float(getattr(stats, "last_run_wall_sec", 0.0)),
        "last_rss_before_kb": int(getattr(stats, "last_rss_before_kb", -1)),
        "last_rss_after_kb": int(getattr(stats, "last_rss_after_kb", -1)),
        "last_rss_delta_kb": int(getattr(stats, "last_rss_delta_kb", -1)),
        "max_rss_after_kb": int(getattr(stats, "max_rss_after_kb", -1)),
        "last_primitive_cache_entries": int(getattr(stats, "last_primitive_cache_entries", 0)),
        "last_primitive_cache_entries_before_reset": int(
            getattr(stats, "last_primitive_cache_entries_before_reset", 0)
        ),
        "last_primitive_cache_entries_after_reset": int(
            getattr(stats, "last_primitive_cache_entries_after_reset", 0)
        ),
        "closed": bool(getattr(stats, "closed", False)),
    }


class PcDbCBSExpert(_BaseExpert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.paths.bindings_path not in sys.path:
            sys.path.append(self.paths.bindings_path)
        import pcdbcbs_planner as pcdbcbs_planner  # pyright: ignore[reportMissingImports]

        self._planner_mod = pcdbcbs_planner
        self._planner_per_call = os.environ.get("PCDBCBS_EXPERT_PLANNER_PER_CALL") == "1"
        self._planner = None if self._planner_per_call else pcdbcbs_planner.Planner()
        self._planner_stats_snapshot: Dict[str, object] = {}
        self._planner_reset_after_plan = os.environ.get("PCDBCBS_EXPERT_PLANNER_RESET_AFTER_PLAN") == "1"
        self._planner_trim_on_reset = os.environ.get("PCDBCBS_EXPERT_PLANNER_TRIM_ON_RESET") == "1"

    def get_planner_stats_dict(self) -> Dict[str, object]:
        out = dict(self._planner_stats_snapshot)
        out["per_call"] = bool(self._planner_per_call)
        return out

    def close(self) -> None:
        if self._planner is None:
            return
        try:
            self._planner.close(self._planner_trim_on_reset)
            self._planner_stats_snapshot = _planner_stats_to_dict(self._planner.stats())
        except Exception:
            pass

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

        if self._planner_per_call:
            planner = self._planner_mod.Planner()
            try:
                res = planner.run(opt)
                self._planner_stats_snapshot = _planner_stats_to_dict(planner.stats())
                if self._planner_reset_after_plan:
                    planner.reset(self._planner_trim_on_reset)
                    self._planner_stats_snapshot = _planner_stats_to_dict(planner.stats())
                planner.close(self._planner_trim_on_reset)
                self._planner_stats_snapshot = _planner_stats_to_dict(planner.stats())
            finally:
                del planner
        else:
            assert self._planner is not None
            res = self._planner.run(opt)
            self._planner_stats_snapshot = _planner_stats_to_dict(self._planner.stats())
            if self._planner_reset_after_plan:
                self._planner.reset(self._planner_trim_on_reset)
                self._planner_stats_snapshot = _planner_stats_to_dict(self._planner.stats())

        if getattr(res, "U", np.array([])).size == 0:
            raise RuntimeError(
                f"pc-dbCBS failed or returned empty U. "
                f"feasible={getattr(res,'feasible',None)} "
                f"solved_db={getattr(res,'solved_db',None)} "
                f"solved_opt={getattr(res,'solved_opt',None)}"
            )

        U = np.asarray(res.U, dtype=np.float32)
        if U.ndim != 2:
            raise RuntimeError(f"Expected res.U shape (T, nu), got {U.shape}")

        self._U = U
        self._t = 0
        self.size_u = int(U.shape[1])
