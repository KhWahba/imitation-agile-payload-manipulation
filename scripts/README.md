# Scripts Directory Guide

This folder contains active training/evaluation tooling plus a `legacy/` archive.

## Primary Entry Points
- `train_dagger_payload.py`: main DAgger training pipeline (supports env-var tuning).
- `train_dagger_payload_memprobe_subprocess.py`: wrapped launcher with memory/profiling logs.
- `eval_policy.py`: rollout-based policy evaluation + statistical plots.
- `play_rounds.py`: inspect DAgger rounds, reward/state plotting, demo rendering.

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
