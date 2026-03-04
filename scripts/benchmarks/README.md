# Benchmark CLI Wrappers

This folder contains small shell wrappers to avoid long manual command lines.

## 1) Generate planner reference (pc-dbCBS)

```bash
scripts/benchmarks/run_pcdbcbs_reference.sh
```

Key options:
- `--input-yaml`
- `--out-dir`
- `--deterministic {0|1}`
- `--n-opt`
- `--time-limit`

Default output dir:
- `deps/agile-payload-transport/envs/zero_goal/planner`

## 2) Run one NMPC benchmark case

```bash
scripts/benchmarks/run_nmpc_case.sh --mode track_goal
scripts/benchmarks/run_nmpc_case.sh --mode track_reference_nmpc_refwarm \
  --planner-ref deps/agile-payload-transport/envs/zero_goal/planner/result_dbcbs_opt.yaml
scripts/benchmarks/run_nmpc_case.sh --mode track_reference_policy \
  --policy-onnx runs/bc_chunk_h10_features_only_e400/bc_chunk_best.onnx
```

Notes:
- Script auto-patches temporary cfg/opt files under `runs/nmpc_cli_tmp/`.
- Relative paths in original case cfg are resolved to absolute paths safely.

## 3) End-to-end pipeline

```bash
scripts/benchmarks/run_zero_goal_pipeline.sh
```

Default behavior:
- runs planner
- runs `track_goal`
- runs `track_reference_nmpc_refwarm`

To include policy mode:

```bash
scripts/benchmarks/run_zero_goal_pipeline.sh --run-track-policy \
  --policy-onnx runs/bc_chunk_h10_features_only_e400/bc_chunk_best.onnx
```

## Help

```bash
scripts/benchmarks/run_pcdbcbs_reference.sh --help
scripts/benchmarks/run_nmpc_case.sh --help
scripts/benchmarks/run_zero_goal_pipeline.sh --help
```
