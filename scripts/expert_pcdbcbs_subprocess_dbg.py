import os
import subprocess
import sys
from typing import Optional

import numpy as np
import yaml

from expert_pcdbcbs_dbg import ExpertPolicySB3, PcDbCBSExpert as _BaseExpert, PcDbCBSPaths


class PcDbCBSExpert(_BaseExpert):
    """Debug expert wrapper that runs pc-dbCBS in a subprocess executable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reuse all base planning/act logic, but replace in-process pybind run() calls.
        self._subprocess_bin = os.environ.get(
            "PCDBCBS_EXPERT_SUBPROCESS_BIN",
            os.path.join(self.paths.bindings_path, "pc_dbcbs_expert_dbg"),
        )
        self._subprocess_timeout_sec: Optional[float] = None
        timeout_env = os.environ.get("PCDBCBS_EXPERT_SUBPROCESS_TIMEOUT_SEC")
        if timeout_env:
            self._subprocess_timeout_sec = float(timeout_env)
        self._subprocess_stream_logs = os.environ.get("PCDBCBS_EXPERT_SUBPROCESS_STREAM_LOGS") == "1"

    def _plan(self, input_yaml: str, plan_dir: str, warmstart: bool):
        output_yaml = os.path.join(plan_dir, "result_dbcbs.yaml")
        optimization_yaml = os.path.join(plan_dir, "result_dbcbs_opt.yaml")
        result_yaml = os.path.join(plan_dir, "result_subprocess_api.yaml")

        cmd = [
            self._subprocess_bin,
            "--input_yaml", input_yaml,
            "--output_yaml", output_yaml,
            "--optimization_yaml", optimization_yaml,
            "--pc_dbcbs_cfg_yaml", self.paths.pc_dbcbs_cfg_yaml,
            "--opt_cfg_yaml", self.paths.opt_cfg_yaml,
            "--time_limit", str(float(self.paths.time_limit)),
            "--dynobench_base", self.paths.dynobench_base,
            "--motion_primitives_base", self.paths.motion_primitives_base,
            "--warmstart_optimization", "1" if bool(warmstart) else "0",
            "--override_visualize_mujoco", "1",
            "--visualize_mujoco", "0",
            "--N_opt", str(int(self.paths.N_opt)),
            "--result_yaml", result_yaml,
        ]

        run_kwargs = dict(
            text=True,
            timeout=self._subprocess_timeout_sec,
            check=False,
        )
        if self._subprocess_stream_logs:
            run_kwargs["stdout"] = None
            run_kwargs["stderr"] = None
        else:
            run_kwargs["stdout"] = subprocess.PIPE
            run_kwargs["stderr"] = subprocess.PIPE

        try:
            proc = subprocess.run(cmd, **run_kwargs)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"pc-dbCBS subprocess timed out after {self._subprocess_timeout_sec}s: {self._subprocess_bin}"
            ) from e

        if not os.path.exists(result_yaml):
            raise RuntimeError(
                f"pc-dbCBS subprocess did not produce result YAML: {result_yaml}\n"
                f"returncode={proc.returncode}\n"
                f"stdout:\n{proc.stdout if proc.stdout is not None else '<streamed>'}\n"
                f"stderr:\n{proc.stderr if proc.stderr is not None else '<streamed>'}"
            )

        with open(result_yaml, "r", encoding="utf-8") as f:
            result = yaml.safe_load(f) or {}

        if proc.returncode != 0 or not bool(result.get("ok", False)):
            stdout_tail = proc.stdout[-1000:] if isinstance(proc.stdout, str) else "<streamed>"
            stderr_tail = proc.stderr[-1000:] if isinstance(proc.stderr, str) else "<streamed>"
            raise RuntimeError(
                "pc-dbCBS subprocess failed. "
                f"returncode={proc.returncode} "
                f"error={result.get('error')} "
                f"stdout_tail={stdout_tail} "
                f"stderr_tail={stderr_tail}"
            )

        U_rows = result.get("U", [])
        U = np.asarray(U_rows, dtype=np.float32)
        if U.size == 0:
            raise RuntimeError(
                "pc-dbCBS subprocess returned empty U. "
                f"solved_db={result.get('solved_db')} "
                f"solved_opt={result.get('solved_opt')} "
                f"feasible={result.get('feasible')} "
                f"info={result.get('info')}"
            )
        if U.ndim != 2:
            raise RuntimeError(f"Expected subprocess U shape (T, nu), got {U.shape}")

        self._U = U
        self._t = 0
        self.size_u = int(U.shape[1])
