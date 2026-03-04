# Scripts Directory Guide

This folder contains active training/evaluation tooling plus a `legacy/` archive.

## Primary Entry Points
- `train_dagger_payload.py`: main DAgger training pipeline (supports env-var tuning).
- `train_dagger_payload_memprobe_subprocess.py`: wrapped launcher with memory/profiling logs.
- `bc_chunk.py`: chunked-action BC training and evaluation pipeline.
- `train_dagger_chunk.py`: chunked-policy bootstrap for DAgger (init BC over chunk labels + ONNX export path).
- `benchmarks/benchmark_nmpc_bindings.py`: mode-sweep benchmark launcher using NMPC Python bindings.
- `benchmarks/run_nmpc_binding.py`: minimal Python-bindings runner (`prob_file`, `cfg_file`).
- `eval_policy.py`: rollout-based policy evaluation + statistical plots.
- `play_rounds.py`: inspect DAgger rounds, reward/state plotting, demo rendering.

## Chunked BC Workflow
- Config template: `scripts/configs/bc_chunk.example.yaml`
- Train:
  - `.venv/bin/python scripts/bc_chunk.py train --config scripts/configs/bc_chunk.example.yaml`
- Closed-loop eval (buffered chunk execution with `replan_k`):
  - `.venv/bin/python scripts/bc_chunk.py eval --config scripts/configs/bc_chunk.example.yaml`
- Offline action-chunk eval (same obs, predicted chunk vs expert chunk):
  - `.venv/bin/python scripts/bc_chunk.py eval-offline --config scripts/configs/bc_chunk.example.yaml`
- Horizon observation eval (roll out predicted chunk and compare resulting observations vs expert over H steps):
  - `.venv/bin/python scripts/bc_chunk.py eval-horizon --config scripts/configs/bc_chunk.example.yaml`

## Expert Wrappers
- `expert_pcdbcbs.py`: in-process pc-dbCBS expert API.
- `expert_pcdbcbs_subprocess.py`: subprocess pc-dbCBS expert API.
- `expert_pcdbcbs_dbg.py`: debug in-process expert API.
- `expert_pcdbcbs_subprocess_dbg.py`: debug subprocess expert API.

## Analysis / Diagnostics
- Keep generated runtime logs in this folder:
  - `train_memprobe_*.log`
  - `expert_memprobe*.jsonl`

## Training Variants / Utilities
- `train_bc.py`
- `videos_from_log.py`
- `utils.py`
- `helpers.py`

## Legacy
Moved historical scripts to:
- `scripts/legacy/`

These are retained for reference and are not part of the current recommended pipeline.
