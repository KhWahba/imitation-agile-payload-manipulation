#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLANNER_SCRIPT="$REPO_ROOT/scripts/benchmarks/run_pcdbcbs_reference.sh"
NMPC_SCRIPT="$REPO_ROOT/scripts/benchmarks/run_nmpc_case.sh"

POLICY_ONNX="$REPO_ROOT/runs/bc_chunk_h10_features_only_e400/bc_chunk_best.onnx"
PLANNER_OUT="$REPO_ROOT/deps/agile-payload-transport/envs/zero_goal/planner"

DO_PLANNER=1
DO_TRACK_GOAL=1
DO_TRACK_REFWARM=1
DO_TRACK_POLICY=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --skip-planner
  --skip-track-goal
  --skip-track-refwarm
  --run-track-policy
  --policy-onnx PATH
  --planner-out PATH
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-planner) DO_PLANNER=0; shift ;;
    --skip-track-goal) DO_TRACK_GOAL=0; shift ;;
    --skip-track-refwarm) DO_TRACK_REFWARM=0; shift ;;
    --run-track-policy) DO_TRACK_POLICY=1; shift ;;
    --policy-onnx) POLICY_ONNX="$2"; shift 2 ;;
    --planner-out) PLANNER_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ "$DO_PLANNER" == "1" ]]; then
  "$PLANNER_SCRIPT" --out-dir "$PLANNER_OUT" --deterministic 1
fi

REF_PATH="$PLANNER_OUT/result_dbcbs_opt.yaml"

if [[ "$DO_TRACK_GOAL" == "1" ]]; then
  "$NMPC_SCRIPT" --mode track_goal
fi

if [[ "$DO_TRACK_REFWARM" == "1" ]]; then
  "$NMPC_SCRIPT" --mode track_reference_nmpc_refwarm --planner-ref "$REF_PATH"
fi

if [[ "$DO_TRACK_POLICY" == "1" ]]; then
  "$NMPC_SCRIPT" --mode track_reference_policy --policy-onnx "$POLICY_ONNX"
fi

echo "[run_zero_goal_pipeline] completed"
