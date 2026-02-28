#!/usr/bin/env bash
set -euo pipefail

# Run the standalone pc-dbCBS expert executable under valgrind memcheck.
# Pass all pc_dbcbs_expert[_dbg] CLI arguments after `--`.
#
# Example:
#   scripts/run_pcdbcbs_expert_valgrind.sh --dbg -- \
#     --input_yaml ... --output_yaml ... --optimization_yaml ... \
#     --pc_dbcbs_cfg_yaml ... --opt_cfg_yaml ... \
#     --dynobench_base ... --motion_primitives_base ... \
#     --time_limit 50000 --warmstart_optimization 1 --N_opt 100 \
#     --result_yaml /tmp/pcdbcbs_result.yaml

MODE="dbg"
if [[ "${1:-}" == "--dbg" ]]; then
  MODE="dbg"
  shift
elif [[ "${1:-}" == "--release" ]]; then
  MODE="release"
  shift
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PCDBCBS_BUILD_DIR:-$ROOT_DIR/deps/pc-dbCBS/build}"
if [[ "$MODE" == "dbg" ]]; then
  BIN="${PCDBCBS_EXPERT_BIN:-$BUILD_DIR/pc_dbcbs_expert_dbg}"
else
  BIN="${PCDBCBS_EXPERT_BIN:-$BUILD_DIR/pc_dbcbs_expert}"
fi

LOG_DIR="${PCDBCBS_VALGRIND_LOG_DIR:-$ROOT_DIR/scripts/valgrind_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%s)"
LOG_FILE="$LOG_DIR/valgrind_${MODE}_${STAMP}.log"

VALGRIND_ARGS=(
  --tool=memcheck
  --leak-check=full
  --show-leak-kinds=all
  --track-origins=yes
  --num-callers=30
  --log-file="$LOG_FILE"
)

echo "[valgrind] mode=$MODE"
echo "[valgrind] bin=$BIN"
echo "[valgrind] log=$LOG_FILE"

exec valgrind "${VALGRIND_ARGS[@]}" "$BIN" "$@"
