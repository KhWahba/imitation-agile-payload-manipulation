#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT_YAML="$REPO_ROOT/deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
OUT_DIR="$REPO_ROOT/deps/agile-payload-transport/envs/zero_goal/planner"
PC_CFG="$REPO_ROOT/deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml"
OPT_CFG="$REPO_ROOT/deps/pc-dbCBS/configs/opt_training.yaml"
DYNOBENCH_BASE="$REPO_ROOT/deps/pc-dbCBS/deps/dynoplan/dynobench/"
MOTION_PRIMS_BASE="/home/khaledwahba94/pc-dbCBS/motion_primitives/"
TIME_LIMIT="50000"
N_OPT="30"
DETERMINISTIC="1"
WARMSTART_OPT="1"
EXPERT_BIN="$REPO_ROOT/deps/pc-dbCBS/build/pc_dbcbs_expert"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --input-yaml PATH
  --out-dir PATH
  --pc-cfg PATH
  --opt-cfg PATH
  --dynobench-base PATH
  --motion-prims-base PATH
  --time-limit FLOAT
  --n-opt INT
  --deterministic {0|1}
  --warmstart-opt {0|1}
  --expert-bin PATH
  -h, --help

Outputs written under out-dir:
  result_dbcbs.yaml
  result_dbcbs_opt.yaml
  init_guess_mujoco.yaml
  result_subprocess_api.yaml
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-yaml) INPUT_YAML="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --pc-cfg) PC_CFG="$2"; shift 2 ;;
    --opt-cfg) OPT_CFG="$2"; shift 2 ;;
    --dynobench-base) DYNOBENCH_BASE="$2"; shift 2 ;;
    --motion-prims-base) MOTION_PRIMS_BASE="$2"; shift 2 ;;
    --time-limit) TIME_LIMIT="$2"; shift 2 ;;
    --n-opt) N_OPT="$2"; shift 2 ;;
    --deterministic) DETERMINISTIC="$2"; shift 2 ;;
    --warmstart-opt) WARMSTART_OPT="$2"; shift 2 ;;
    --expert-bin) EXPERT_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

OUTPUT_YAML="$OUT_DIR/result_dbcbs.yaml"
OPTIMIZATION_YAML="$OUT_DIR/result_dbcbs_opt.yaml"
RESULT_YAML="$OUT_DIR/result_subprocess_api.yaml"

CMD=(
  "$EXPERT_BIN"
  --input_yaml "$INPUT_YAML"
  --output_yaml "$OUTPUT_YAML"
  --optimization_yaml "$OPTIMIZATION_YAML"
  --pc_dbcbs_cfg_yaml "$PC_CFG"
  --opt_cfg_yaml "$OPT_CFG"
  --time_limit "$TIME_LIMIT"
  --dynobench_base "$DYNOBENCH_BASE"
  --motion_primitives_base "$MOTION_PRIMS_BASE"
  --warmstart_optimization "$WARMSTART_OPT"
  --override_visualize_mujoco 1
  --visualize_mujoco 0
  --N_opt "$N_OPT"
  --result_yaml "$RESULT_YAML"
)

echo "[run_pcdbcbs_reference] Running:" 
printf '  %q' "${CMD[@]}"; echo

if [[ "$DETERMINISTIC" == "1" ]]; then
  PCDBCBS_DETERMINISTIC=1 "${CMD[@]}"
else
  "${CMD[@]}"
fi

echo "[run_pcdbcbs_reference] Done. Files:"
ls -lh "$OUT_DIR"/result_dbcbs*.yaml "$OUT_DIR"/init_guess_mujoco.yaml "$OUT_DIR"/result_subprocess_api.yaml
