# Imitation Learning for Agile Payload Transport

DAgger/BC training setup for a cooperative multi-quadrotor payload task in MuJoCo, with a `pc-dbCBS` expert planner (in-process or subprocess).

## Repository Structure

- `scripts/`: active training, expert wrappers, evaluation, and analysis tools.
- `deps/pc-dbCBS/`: planner stack (with nested `dynoplan` and `dynobench` submodules).
- `runs/`: training/evaluation outputs (ignored by git).
- `envs/`: environment-related assets/configs.

See also: `scripts/README.md` for script-level entrypoints.

## Setup

1. Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd imitation-agile-payload-manipulation
```

2. Python environment (uv):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
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
