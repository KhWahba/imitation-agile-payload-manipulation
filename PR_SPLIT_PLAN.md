# PR Split Plan (Root + Submodules)

This repository has changes across:
- root repo (`imitation-agile-payload-manipulation`)
- submodule `deps/pc-dbCBS`
- nested submodule `deps/pc-dbCBS/deps/dynoplan`
- nested submodule `deps/pc-dbCBS/deps/dynoplan/dynobench`

To keep PRs clean, split by ownership and dependency direction (inside-out).

## 0) Safety Snapshot
```bash
git switch update-training
git branch wip/all-current
```

## 1) Submodule Branch Recovery (detached HEAD -> branch)

### pc-dbCBS
```bash
git -C deps/pc-dbCBS fetch origin
git -C deps/pc-dbCBS switch -c dagger-bindings --track origin/dagger-bindings || git -C deps/pc-dbCBS switch dagger-bindings
git -C deps/pc-dbCBS switch -c feat/subprocess-planner-api
```

### dynoplan
```bash
git -C deps/pc-dbCBS/deps/dynoplan fetch origin
git -C deps/pc-dbCBS/deps/dynoplan switch -c dagger-mujoco-opt --track origin/dagger-mujoco-opt || git -C deps/pc-dbCBS/deps/dynoplan switch dagger-mujoco-opt
git -C deps/pc-dbCBS/deps/dynoplan switch -c feat/opt-stability-costs
```

### dynobench
```bash
git -C deps/pc-dbCBS/deps/dynoplan/dynobench fetch origin
git -C deps/pc-dbCBS/deps/dynoplan/dynobench switch -c dagger-pcdbcbs --track origin/dagger-pcdbcbs || git -C deps/pc-dbCBS/deps/dynoplan/dynobench switch dagger-pcdbcbs
git -C deps/pc-dbCBS/deps/dynoplan/dynobench switch -c feat/payload-bounds-stability
```

Commit and PR order:
1. `dynobench` PR
2. `dynoplan` PR (includes updated dynobench pointer)
3. `pc-dbCBS` PR (includes updated dynoplan pointer)
4. root repo PR(s) (includes updated pc-dbCBS pointer)

## 2) Root Repo PR Buckets

Recommended root PR split:

1. **PR-A: Eval + Analysis tooling**
   - `scripts/eval_policy.py`
   - `scripts/play_rounds.py`

2. **PR-B: DAgger training pipeline changes**
   - `scripts/train_dagger_payload.py`
   - `scripts/payload_env.py`
   - `scripts/expert_pcdbcbs.py`
   - `scripts/expert_pcdbcbs_dbg.py`
   - `scripts/expert_pcdbcbs_subprocess.py`
   - `scripts/expert_pcdbcbs_subprocess_dbg.py`
   - `scripts/train_dagger_payload_memprobe_subprocess.py`

3. **PR-C: Script structure cleanup (legacy migration)**
   - `scripts/README.md`
   - moved files under `scripts/legacy/`
   - removal of old script-side artifacts (`scripts/runs`, `scripts/videos`)

4. **PR-D: Tests / small compatibility updates**
   - `scripts/tests/*` updates
   - optional utility updates (`scripts/utils.py`)

5. **PR-E: Repo hygiene**
   - `.gitignore` adjustments
   - no runtime logs/jsonl in commit

## 3) Practical Commit Method (No interactive pain)

Use path-based commits from current tree:
```bash
git add <files-for-pr-a>
git commit -m "feat(eval): add rollout statistical policy evaluation"

git add <files-for-pr-b>
git commit -m "feat(training): add multi-env expert + beta/replan controls"

git add <files-for-pr-c>
git commit -m "chore(scripts): move legacy scripts and document active pipeline"
```

## 4) Keep Runtime Artifacts Out

Before pushing:
```bash
git restore --staged scripts/*.log scripts/*.jsonl || true
git status --short
```

## 5) Push Strategy

Push each repo/PR branch independently:
```bash
# submodule repos first
git -C deps/pc-dbCBS/deps/dynoplan/dynobench push -u origin feat/payload-bounds-stability
git -C deps/pc-dbCBS/deps/dynoplan push -u origin feat/opt-stability-costs
git -C deps/pc-dbCBS push -u origin feat/subprocess-planner-api

# root
git push -u origin <root-pr-branch>
```

Then open PRs in the same inside-out order.
