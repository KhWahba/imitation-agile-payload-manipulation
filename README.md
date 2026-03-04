# Imitation Learning for Agile Payload Transport

DAgger/BC training setup for a cooperative multi-quadrotor payload task in MuJoCo, with a `pc-dbCBS` expert planner (in-process or subprocess).

## Repository Structure

- `scripts/`: active training, expert wrappers, evaluation, and analysis tools.
- `deps/pc-dbCBS/`: planner stack (with nested `dynoplan` and `dynobench` submodules).
- `runs/`: training/evaluation outputs (ignored by git).
- `envs/`: environment-related assets/configs.

See also: `scripts/README.md` for script-level entrypoints.
For NMPC + chunk-policy status handoff, see:
- `docs/FRESH_SESSION_NMPC_DAGGER_HANDOFF.md`
- `docs/nmpc_policy_chunk_action_tracker.md`

## Setup

1. Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd imitation-agile-payload-manipulation
```

2. Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3. Build planner binaries/bindings:

```bash
cmake -S deps/pc-dbCBS -B deps/pc-dbCBS/build
cmake --build deps/pc-dbCBS/build -j
```

## Main Training Entry Point

`scripts/train_dagger_payload.py`

Typical run:

```bash
.venv/bin/python scripts/train_dagger_payload.py
```

Common runtime controls are exposed via environment variables, for example:

- `DAGGER_OUT_ROOT`
- `DAGGER_TOTAL_TIMESTEPS`
- `DAGGER_NUM_ENVS`
- `DAGGER_EXPERT_REPLAN_K`
- `DAGGER_BETA_MODE`, `DAGGER_BETA_RAMPDOWN_ROUNDS`
- `DAGGER_N_SEED_EPISODES`
- `PCDBCBS_DETERMINISTIC`
- `SEED_RANDOMIZE_DELTA0`, `SEED_RANDOMIZE_DELTA0_LOW`, `SEED_RANDOMIZE_DELTA0_HIGH`

## Memory-Probed / Subprocess Expert Run

Use `scripts/train_dagger_payload_memprobe_subprocess.py` when diagnosing memory behavior around expert calls:

```bash
MEMPROBE_USE_DBG_EXPERT=1 \
PCDBCBS_EXPERT_SUBPROCESS_BIN=deps/pc-dbCBS/build/pc_dbcbs_expert_dbg \
.venv/bin/python scripts/train_dagger_payload_memprobe_subprocess.py
```

Outputs are written to:

- `scripts/expert_memprobe*.jsonl`
- `scripts/train_memprobe*.log`

## Evaluation and Analysis

- Policy rollout evaluation + plots:

```bash
.venv/bin/python scripts/eval_policy.py \
  --policy-state-dict <path/to/policy_state_dict.pt> \
  --expert-trajs-pkl <path/to/expert_trajs.pkl> \
  --out-dir runs/policy_eval
```

- DAgger round inspection / reward-state plotting / replay tools:

```bash
.venv/bin/python scripts/play_rounds.py summary --round 0
.venv/bin/python scripts/play_rounds.py plot-rewards-progress --demos-root <round_dir_parent>
```

## Chunked BC (Standalone, Pre-DAgger)

Use `scripts/bc_chunk.py` to train a policy that predicts `H` future actions from one observation, then evaluate it in three ways:

- `train`: trains BC on chunked labels built from cached expert trajectories.
- `eval`: closed-loop rollout in MuJoCo using an action buffer + `replan_k`.
- `eval-offline`: direct chunk prediction error on expert observations.
- `eval-horizon`: rolls out each predicted chunk for `H` steps and compares predicted observation trajectory against expert observation trajectory.

Config-driven usage:

```bash
# Edit settings (paths, horizon, epochs, eval params)
vim scripts/configs/bc_chunk.example.yaml

# Train
.venv/bin/python scripts/bc_chunk.py train --config scripts/configs/bc_chunk.example.yaml

# Closed-loop eval
.venv/bin/python scripts/bc_chunk.py eval --config scripts/configs/bc_chunk.example.yaml

# Offline chunk-action eval
.venv/bin/python scripts/bc_chunk.py eval-offline --config scripts/configs/bc_chunk.example.yaml

# Horizon observation eval
.venv/bin/python scripts/bc_chunk.py eval-horizon --config scripts/configs/bc_chunk.example.yaml
```

## Notes on Branching and PR Split

This project currently uses split PRs across root + submodules.

- Plan reference: `PR_SPLIT_PLAN.md`
- Submodule merge order: `dynobench` -> `dynoplan` -> `pc-dbCBS` -> root pointer bump

## Current Ignore Policy

Runtime artifacts are ignored by default:

- `runs/`
- `*.log`
- `*.jsonl`
- `.venv/`
