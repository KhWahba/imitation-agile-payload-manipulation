#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODE="track_goal"
CASE_ROOT="$REPO_ROOT/deps/agile-payload-transport/envs/benchmark_zero_goal_2robots"
PLANNER_REF="$REPO_ROOT/deps/agile-payload-transport/envs/zero_goal/planner/result_dbcbs_opt.yaml"
POLICY_ONNX=""
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
RUNNER="$REPO_ROOT/scripts/benchmarks/run_nmpc_binding.py"
WORK_DIR="$REPO_ROOT/runs/nmpc_cli_tmp"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --mode {track_goal|track_reference_nmpc_refwarm|track_reference_policy}
  --case-root PATH
  --planner-ref PATH      (used for refwarm mode)
  --policy-onnx PATH      (optional override for policy mode)
  --python-bin PATH
  --runner PATH
  --work-dir PATH         (temp patched cfg/opt files)
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --case-root) CASE_ROOT="$2"; shift 2 ;;
    --planner-ref) PLANNER_REF="$2"; shift 2 ;;
    --policy-onnx) POLICY_ONNX="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

case "$MODE" in
  track_goal|track_reference_nmpc_refwarm|track_reference_policy) ;;
  *) echo "Invalid mode: $MODE"; usage; exit 2 ;;
esac

CASE_DIR="$CASE_ROOT/$MODE"
CFG_IN="$CASE_DIR/cfg.yaml"
OPT_IN="$CASE_DIR/opt_cfg_file.yaml"

if [[ ! -f "$CFG_IN" || ! -f "$OPT_IN" ]]; then
  echo "Missing cfg/opt in case dir: $CASE_DIR" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
CFG_TMP="$WORK_DIR/${MODE}_cfg.yaml"
OPT_TMP="$WORK_DIR/${MODE}_opt.yaml"

"$PYTHON_BIN" - <<PY
import pathlib, yaml
cfg_in = pathlib.Path(r"$CFG_IN")
opt_in = pathlib.Path(r"$OPT_IN")
cfg = yaml.safe_load(cfg_in.read_text())
opt = yaml.safe_load(opt_in.read_text())
mode = r"$MODE"
planner_ref = r"$PLANNER_REF"
policy_onnx = r"$POLICY_ONNX"
case_dir = cfg_in.parent

def abs_from_case(v: str) -> str:
    p = pathlib.Path(v)
    if p.is_absolute():
        return str(p)
    return str((case_dir / p).resolve())

for key in ("env_file", "init_file", "ref_file", "models_dir", "results_path", "video_prefix"):
    v = cfg.get(key, "")
    if isinstance(v, str) and v.strip():
        cfg[key] = abs_from_case(v)

if mode == "track_reference_nmpc_refwarm":
    cfg["init_file"] = str(pathlib.Path(planner_ref).resolve())
    cfg["ref_file"] = str(pathlib.Path(planner_ref).resolve())

if mode == "track_reference_policy" and policy_onnx:
    opt["policy_onnx_path"] = str(pathlib.Path(policy_onnx).resolve())

pathlib.Path(r"$CFG_TMP").write_text(yaml.safe_dump(cfg, sort_keys=False))
pathlib.Path(r"$OPT_TMP").write_text(yaml.safe_dump(opt, sort_keys=False))
print("Wrote:", r"$CFG_TMP")
print("Wrote:", r"$OPT_TMP")
PY

echo "[run_nmpc_case] Running mode=$MODE"
"$PYTHON_BIN" "$RUNNER" "$CFG_TMP" "$OPT_TMP"

echo "[run_nmpc_case] Done. Result path in cfg:"
"$PYTHON_BIN" - <<PY
import yaml
cfg = yaml.safe_load(open(r"$CFG_TMP", "r", encoding="utf-8"))
print(cfg.get("results_path", "<none>"))
PY
