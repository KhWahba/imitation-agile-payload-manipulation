# Full-Stack pc-dbCBS + NMPC + Imitation Learning Handoff

This document is a fresh handoff written from the current codebase, not from older notes. It is meant to be a high-signal technical map of the repository for continued work, especially by another coding agent such as Claude Opus 4.6.

## 0. Repository Structure

Top-level layout relevant to the payload transport pipeline:

- `deps/pc-dbCBS/`
  - Multi-robot planner and expert interface.
  - Contains:
    - the `pc_dbcbs_expert` CLI wrapper,
    - the in-process API (`pcdbcbs.run(opt)`),
    - the high-level pc-dbCBS loop,
    - low-level single-robot search over motion primitives,
    - initial-guess generation for optimization,
    - trajectory optimization through Dynoplan + Crocoddyl.
- `deps/agile-payload-transport/`
  - Stateful NMPC controller for the MuJoCo payload system.
  - Contains:
    - Crocoddyl action/cost/dynamics wrappers,
    - ONNX policy wrapper,
    - stateful NMPC controller,
    - `nmpc_mujoco` CLI,
    - `nmpc_controller_py` Python bindings.
- `scripts/`
  - Imitation-learning pipeline.
  - Contains:
    - MuJoCo environment wrapper for learning (`payload_env.py`),
    - expert wrappers (direct bindings and subprocess),
    - DAgger-H training,
    - memprobe wrapper,
    - evaluation and plotting scripts,
    - ONNX export scripts.
- `runs/`
  - All generated training, evaluation, sweep, and debug artifacts.
- `docs/`
  - Handoff documents and project context.

## 1. Concrete Example Used Throughout

The main example discussed here is:

- Environment YAML:
  - `deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml`
- Model YAML:
  - `deps/pc-dbCBS/deps/dynoplan/dynobench/models/mujocoquadspayload_empty2.yaml`
- MuJoCo XML used by the model:
  - `deps/pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml`

This example is a 2-quadrotor payload system with state layout:

$$
\mathbf{x} = [\text{payload pose},\; \text{quad 1 pose},\; \text{quad 2 pose},\; \text{payload vel},\; \text{quad 1 vel},\; \text{quad 2 vel}] \in \mathbb{R}^{39}
$$

where each body contributes:

- pose: $3$ position + $4$ quaternion,
- velocity: $3$ linear + $3$ angular.

So for $n_b = 3$ bodies:

$$
39 = 3 \times 13
$$

and for $n_q = 2$ quadrotors, the control dimension is:

$$
\nu = 4 n_q = 8
$$

## 2. Layer 1: pc-dbCBS Expert Stack

### 2.1 Main files

The planner/expert stack spans these files:

- CLI expert wrapper:
  - `deps/pc-dbCBS/src/pc_dbcbs_expert.cpp`
- Core API:
  - `deps/pc-dbCBS/src/pc_dbcbs_api.cpp`
- Planner utilities and conflict logic:
  - `deps/pc-dbCBS/src/pc_dbcbs_utils.hpp`
- Low-level search:
  - `deps/pc-dbCBS/deps/dynoplan/src/tdbastar/tdbastar.cpp`
- Initial-guess conversion for MuJoCo payload optimization:
  - `deps/pc-dbCBS/src/init_guess_mujoco.cpp`
- OCP generation:
  - `deps/pc-dbCBS/deps/dynoplan/src/optimization/generate_ocp.cpp`
- OCP solving:
  - `deps/pc-dbCBS/deps/dynoplan/src/optimization/ocp.cpp`

### 2.2 How to build pc-dbCBS

From repository root:

```bash
cmake -S deps/pc-dbCBS -B deps/pc-dbCBS/build -DCMAKE_PREFIX_PATH=/opt/openrobots
cmake --build deps/pc-dbCBS/build -j --target pc_dbcbs_expert pcdbcbs
```

Important environment assumptions:

- MuJoCo root is found via `MJ_ROOT` or defaults to `deps/pc-dbCBS/deps/mujoco-3.3.6`.
- Dynoplan / Crocoddyl optimization code in this repo is tied to the older Crocoddyl API style. In practice this stack is expected to work with the OpenRobots / Crocoddyl 2.1.x line. Crocoddyl 3.x style APIs will break older `boost::shared_ptr`-based code.

### 2.3 How to run the expert command

The CLI expert is the cleanest external interface:

```bash
PCDBCBS_DETERMINISTIC=1 \
deps/pc-dbCBS/build/pc_dbcbs_expert \
  --input_yaml deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml \
  --output_yaml runs/tmp/result_dbcbs.yaml \
  --optimization_yaml runs/tmp/result_dbcbs_opt.yaml \
  --pc_dbcbs_cfg_yaml deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml \
  --opt_cfg_yaml deps/pc-dbCBS/configs/opt_training.yaml \
  --motion_primitives_base /home/khaledwahba94/pc-dbCBS/motion_primitives/ \
  --dynobench_base deps/pc-dbCBS/deps/dynoplan/dynobench/ \
  --time_limit 50000 \
  --warmstart_optimization 1 \
  --override_visualize_mujoco 1 \
  --visualize_mujoco 0 \
  --N_opt 30 \
  --result_yaml runs/tmp/result_subprocess_api.yaml
```

Outputs:

- `result_dbcbs.yaml`
  - discrete multi-robot planner result.
- `result_dbcbs_opt.yaml`
  - optimized joint trajectory.
- `result_subprocess_api.yaml`
  - compact API summary with:
    - `ok`, `error`, `solved_db`, `solved_opt`,
    - durations,
    - `X`, `U` matrices.

### 2.4 CLI entrypoint: what happens in `pc_dbcbs_expert.cpp`

`pc_dbcbs_expert.cpp` is intentionally thin:

1. parse CLI arguments,
2. fill `pcdbcbs::Options`,
3. call `pcdbcbs::run(opt)`,
4. optionally write a compact result YAML with dense `X` and `U` matrices,
5. print a short status line.

So the real behavior is in `pc_dbcbs_api.cpp`.

### 2.5 Call graph: planner layer by layer

The effective call chain for expert use is:

```text
pc_dbcbs_expert (CLI)
  -> pcdbcbs::run(opt)                      [pc_dbcbs_api.cpp]
       -> load env + models + configs
       -> load or cache motion primitives
       -> build per-robot primitive libraries
       -> run low-level tdbA* for each robot
       -> run high-level conflict-based search
       -> if conflict-free discrete solution found:
            -> export discrete result
            -> generate joint MuJoCo init guess
            -> run trajectory optimization
            -> return optimized X/U
       -> else / timeout / trivial near-goal:
            -> optimization-only fallback
```

### 2.6 What `pcdbcbs::run(opt)` actually does

In `deps/pc-dbCBS/src/pc_dbcbs_api.cpp`, `pcdbcbs::run(opt)` does the following.

#### Step A: validate options and load config

It loads:

- the environment YAML,
- the pc-dbCBS config YAML,
- the optimization YAML,
- the motion primitive base directory,
- the model files via Dynobench.

For the main config, `pc_dbcbs_empty.yaml` sets parameters such as:

- initial search radius `delta_0`,
- primitive counts,
- duplicate filtering,
- the `payload.solve_p0` flag,
- payload pseudo-center optimization weights in `payload_opt`.

For optimization, `opt_training.yaml` sets:

- `solver_id`,
- `max_iter`,
- `weight_goal`,
- `collision_weight`,
- `states_reg`,
- MuJoCo payload regularization overrides.

#### Step B: optimization-only fallback support

Before entering the normal planner path, the API defines an internal optimization-only fallback.

That fallback:

1. converts `joint_robot` env YAML into a plain `robots:` env YAML,
2. writes a dummy init guess with repeated start state and constant actions,
3. runs MuJoCo trajectory optimization directly via `execute_optILMujoco(...)`.

This fallback is used when:

- `warmstart_optimization` is disabled at the API level,
- the start is already within `delta` of the goal,
- the overall planner time limit is exceeded,
- or the normal discrete stage becomes structurally unusable for this query.

### 2.7 Motion primitives and low-level search

For each robot type, `pcdbcbs::run()`:

1. loads motion primitives from the precomputed primitive file,
2. converts those trajectories to `Motion` objects,
3. optionally shuffles / filters / truncates them,
4. builds `sub_motions`, a reduced active primitive set for search,
5. calls `tdbastar(...)` for each robot.

The motion file for the MuJoCo quad payload case is currently mapped to the quad3d primitive set:

```text
quad3d_max_tilt_angle_40/quad3d.bin.im.bin.sp.bin
```

### 2.8 Low-level search: what `tdbastar.cpp` is doing

`tdbastar(...)` is a single-robot search over motion primitives with nearest-neighbor duplicate detection and optional reverse-search heuristic support.

Conceptually it maintains nodes of the form:

$$
n = (x, g, h, f, \text{arrivals}, \text{reachesGoal})
$$

with:

$$
f = g + h
$$

where:

- $x$ is the robot state,
- $g$ is the accumulated trajectory cost so far,
- $h$ is a heuristic lower bound to the goal,
- arrivals store different ways to reach nearly the same state.

#### Main ideas in the implementation

- Motion primitives are stored in a nearest-neighbor structure.
- Nodes are stored in another nearest-neighbor structure for duplicate detection / rewiring.
- A node is expanded by lazily transforming nearby motion primitives from the node state.
- A new state is considered novel only if no existing node lies within a radius of roughly:

$$
(1 - \alpha) \delta
$$

while primitive applicability is controlled by a companion radius of roughly:

$$
\alpha \delta
$$

This is why `alpha` and `delta` appear everywhere in the search.

#### Low-level search pseudocode

```text
function TDBASTAR(problem, robot_id, constraints, motions, delta, alpha, upper_bound):
    build robot model and load environment
    build NN index over motions
    build NN index over explored nodes

    start_node.x = start state of robot_id
    start_node.g = 0
    start_node.h = heuristic(start)
    start_node.f = start_node.g + start_node.h
    push start_node to open queue
    add start_node to node NN structure

    while open queue not empty and limits not exceeded:
        best = pop node with smallest f, then largest g tie-break

        if best is within goal tolerance and does not violate future constraints:
            reconstruct trajectory and return solved

        lazy_trajs = expand primitives around best.x

        for each lazy_traj in lazy_trajs:
            if primitive rollout is invalid under dynamics / env / constraints:
                continue

            x_new = terminal state of transformed primitive
            g_new = best.g + primitive_cost + jump_cost
            h_new = heuristic(x_new)
            if g_new > upper_bound:
                continue

            neighbors = nodes within novelty radius around x_new

            if no neighbors:
                create new node, set arrival from best, push to open
            else if rewire enabled:
                update neighbors if this arrival gives lower g

    if goal not reached:
        return nearest-to-goal partial solution, marked unsolved
```

### 2.9 High-level pc-dbCBS loop

The high-level part in `pc_dbcbs_api.cpp` is a CBS-like search over conflicts.

Each `HighLevelNode` contains:

- one low-level trajectory per robot,
- a set of constraints per robot,
- an accumulated cost.

Algorithmically:

1. solve low-level path for each robot without inter-robot constraints,
2. detect earliest inter-robot conflict,
3. if no conflict exists, convert the discrete solution to a joint initial guess and optimize,
4. otherwise create child nodes with extra constraints for the conflicting robots,
5. replan only the constrained robot(s),
6. continue until conflict-free.

#### High-level pseudocode

```text
function PCDBCBS(problem):
    start_node.solution = []
    start_node.constraints = empty per robot

    for each robot i:
        start_node.solution[i] = low_level_plan(robot=i, constraints=[])
        if any low-level plan fails:
            restart / relax primitive settings

    open = priority queue ordered by sum of low-level costs
    push start_node

    while open not empty and global time limit not exceeded:
        P = pop cheapest node
        conflict = earliest conflict among robot trajectories in P

        if no conflict:
            export discrete solution
            init_guess = generate_init_guess_mujoco(...)
            opt_solution = execute_optILMujoco(init_guess)
            if optimization succeeds:
                return opt_solution
            else:
                augment primitive library with the broken optimized trajectory
                restart high-level loop

        child_constraints = constraints created from conflict
        for each affected robot r:
            child = copy(P)
            child.constraints[r] += new constraints for robot r
            child.solution[r] = low_level_plan(robot=r, constraints=child.constraints[r])
            if replanning succeeds:
                update child cost and push child
```

### 2.10 Conflict detection and payload pseudo-center optimization

The main conflict function is `getEarliestConflict(...)` in `pc_dbcbs_utils.hpp`.

It checks two things:

1. ordinary robot-robot collisions using FCL,
2. optional payload pseudo-center / cable consistency checks when `solve_p0` is enabled.

#### Ordinary conflicts

For each time index $t$:

- each robot trajectory is sampled at $t$,
- the robot collision geometries are transformed,
- the FCL broadphase manager is updated,
- if a collision exists, the earliest collision is returned as the conflict.

#### Payload pseudo-center optimization (`solve_p0`)

When `solve_p0` is on, the code also solves a small payload-position subproblem to estimate a payload center $p_0$ consistent with the quad positions and cable lengths.

This is done with NLopt COBYLA in `optimizePayload(...)`.

The cost being minimized is effectively:

$$
J_{p_0}(p_0) = J_{\text{stretch}} + J_{\text{bounds}} + J_{\text{coll}} + J_{\text{min-z}} + J_{\text{reg}} + J_{\text{goal}}
$$

with:

$$
J_{\text{stretch}} = w_{\text{stretch}} \sum_i \max\left(0, \lVert p_0 - p_i \rVert - l_i\right)^2
$$

$$
J_{\text{bounds}} = w_{\text{bounds}} \sum_{k=1}^3 \text{boundViolation}_k(p_0)^2
$$

$$
J_{\text{coll}} = w_{\text{coll}} \max\left(0, d_{\text{safe}} - d_{\text{env}}(p_0)\right)^2
$$

$$
J_{\text{min-z}} = \lambda \max\left(0, z_0 - z_{\text{vert}}\right)^2
$$

where $z_{\text{vert}}$ is computed from the lowest quad and cable length heuristic, and:

$$
J_{\text{reg}} = \mu \lVert p_0 - p_0^{\text{prev}} \rVert^2
$$

and, when enabled in code:

$$
J_{\text{goal}} = w_{\text{goal}} \lVert p_0 - p_{\text{goal}} \rVert^2
$$

Important practical detail:

- `w_goal` is currently set in code during conflict checking (`data.w_goal = 0.8`), not loaded from the YAML.

This $p_0$ optimizer is not the main trajectory optimizer. It is a small geometric consistency optimizer used during conflict detection / payload consistency reasoning.

### 2.11 Initial guess generation for MuJoCo optimization

The discrete planner returns per-robot trajectories. The optimizer expects a joint MuJoCo payload trajectory. That conversion happens in:

- `deps/pc-dbCBS/src/init_guess_mujoco.cpp`

What it does:

1. read per-robot states/actions from `result_dbcbs.yaml`,
2. pad trajectories to a common horizon,
3. concatenate robot actions into a joint control matrix,
4. build a joint state matrix in the MuJoCo payload state layout,
5. write:
   - `env.yaml` with `robots:` format,
   - `env_traj_checker.yaml`,
   - `init_guess_mujoco.yaml`.

Important implementation detail:

- for the MuJoCo payload case, the converter currently seeds:
  - payload quaternion as identity,
  - quad quaternions as identity,
  - payload linear/angular velocities as zero,
  - quad linear/angular velocities as zero.

So the state seed is kinematically stitched, but not fully dynamically consistent.

This is a real limitation of the current pipeline and should be remembered when diagnosing optimizer behavior.

### 2.12 Trajectory optimization cost in Dynoplan

The actual trajectory OCP is generated in:

- `generate_ocp.cpp`

and solved in:

- `ocp.cpp`

The running cost is assembled as a sum of feature residuals, and each feature is used either as a least-squares cost or a linear cost. For the least-squares terms, Crocoddyl sees:

$$
\ell_t(x_t, u_t) = \frac{1}{2} \sum_j \lVert r_j(x_t, u_t) \rVert^2
$$

The terminal cost is similarly:

$$
\ell_N(x_N) = \frac{1}{2} \lVert r_N(x_N) \rVert^2
$$

For the MuJoCo quad-payload case with `opt_training.yaml`, the main active pieces are typically:

- stage-$0$ start anchoring cost,
- collision cost,
- state regularization cost,
- acceleration regularization cost,
- cable-direction regularization cost if enabled,
- state bounds,
- terminal goal cost.

#### Typical residuals

State tracking residual:

$$
r_x(x) = W_x \odot (x \ominus x_{\text{ref}})
$$

Control tracking / regularization residual:

$$
r_u(u) = W_u \odot (u - u_{\text{ref}})
$$

Collision residual from `Col_cost`:

$$
r_{\text{col}}(x) = \min\left(w_{\text{col}} (d(x) - m), 0\right)
$$

where:

- $d(x)$ is the signed clearance returned by the collision model,
- $m$ is the safety margin.

So collision cost becomes active only when the distance drops below the margin.

Terminal goal residual in `generate_ocp.cpp`:

$$
r_N(x_N) = W_N \odot (x_N \ominus x_g)
$$

where the difference is computed with the robot's manifold-aware state operations via `State_cost_model`.

#### Free-time behavior in `opt_training.yaml`

`solver_id = 1` maps to:

$$
\texttt{SOLVER::traj\_opt\_free\_time}
$$

In `ocp.cpp`, that means:

1. run a proxy free-time solve first,
2. then run a fixed-time repair / final solve.

So the optimization is effectively two-stage:

```text
free-time proxy solve -> resample / repair -> fixed-time solve
```

### 2.13 Crocoddyl wrappers used by Dynoplan optimization

Dynoplan's optimization stack uses the cost and action model wrappers in:

- `deps/agile-payload-transport/src/croco_models.cpp`
- `deps/agile-payload-transport/include/croco_models.hpp`

These provide:

- `Dynamics`
  - wraps `Model_robot::step(...)` and `stepDiff(...)`.
- `ActionModelDyno`
  - wraps one stage of dynamics + running costs.
- `State_cost`, `State_cost_model`, `Control_cost`, `Col_cost`, etc.
  - analytic residual and derivative implementations.

Important distinction:

- `State_cost` uses Euclidean state difference,
- `State_cost_model` uses the robot model manifold difference:

$$
\delta x = x_{\text{ref}} \ominus x
$$

via `model_robot->state_diff(...)`.

That is the correct choice for quaternion-containing states.

## 3. Layer 2: agile-payload-transport NMPC Stack

### 3.1 Main files

- NMPC controller:
  - `deps/agile-payload-transport/src/nmpc_controller.cpp`
  - `deps/agile-payload-transport/include/nmpc_controller.hpp`
- Crocoddyl action/cost/dynamics wrappers:
  - `deps/agile-payload-transport/src/croco_models.cpp`
  - `deps/agile-payload-transport/include/croco_models.hpp`
- ONNX wrapper:
  - `deps/agile-payload-transport/src/policy_onnx.cpp`
- Python binding:
  - `deps/agile-payload-transport/src/py_nmpc_controller.cpp`
- CLI entrypoint:
  - `deps/agile-payload-transport/src/nmpc_mujoco.cpp`
- NMPC options:
  - `deps/agile-payload-transport/include/options.hpp`

### 3.2 How to build agile-payload-transport

```bash
cmake -S deps/agile-payload-transport -B deps/agile-payload-transport/build \
  -DCMAKE_PREFIX_PATH=/opt/openrobots \
  -DMJ_ROOT=/home/khaledwahba94/imitation-agile-payload-manipulation/deps/pc-dbCBS/deps/mujoco-3.3.6 \
  -DONNXRUNTIME_ROOT=/home/khaledwahba94/imitation-agile-payload-manipulation/deps/agile-payload-transport/deps/onnxruntime

cmake --build deps/agile-payload-transport/build -j --target nmpc_mujoco nmpc_controller_py
```

Targets of interest:

- `nmpc_mujoco`
  - C++ CLI.
- `nmpc_controller_py`
  - Python binding.
- `policy_onnx`
  - ONNX Runtime wrapper.

### 3.3 How to run the NMPC from C++

```bash
deps/agile-payload-transport/build/nmpc_mujoco \
  --prob_file deps/agile-payload-transport/envs/empty2/cfg.yaml \
  --cfg_file deps/agile-payload-transport/envs/empty2/opt_cfg_file.yaml
```

The CLI does only three things:

1. parse `prob_file` and `cfg_file`,
2. build `NmpcController`,
3. call `run()` and then `maybe_visualize()`.

### 3.4 How to run the NMPC from Python bindings

```python
import sys
sys.path.insert(0, "deps/agile-payload-transport/build")
import nmpc_controller_py

ctrl = nmpc_controller_py.Controller(prob_path, cfg_path)
ctrl.run(
    mode="track_reference_policy",
    out_yaml="runs/tmp/result.yaml",
    out_timing_json="runs/tmp/result_timing.json",
    visualize=False,
)
```

Binding API:

- constructor:
  - `Controller(prob_path, cfg_path)`
- methods:
  - `run(mode="", out_yaml="", out_timing_json="", visualize=False)`
  - `maybe_visualize()`

### 3.5 Current controller modes

The controller currently supports these explicit modes:

- `track_goal`
- `track_reference_nmpc_standard`
- `track_reference_nmpc_refwarm`
- `track_reference_policy`
- `track_policy_warmstart_goal`
- `track_linear_hover`

Semantics:

1. `track_goal`
   - warm start from previous solved window / init guess,
   - running state reference is the goal,
   - control reference is hover control.

2. `track_reference_nmpc_standard`
   - warm start from previous solved window,
   - running state and control references come from planner trajectory.

3. `track_reference_nmpc_refwarm`
   - warm start directly from planner reference trajectory,
   - running state and control references also come from planner trajectory.

4. `track_reference_policy`
   - warm start generated from policy rollout,
   - running state and control references also come from policy rollout.

5. `track_policy_warmstart_goal`
   - warm start generated from policy rollout,
   - running state reference is the goal,
   - control reference is hover control.

6. `track_linear_hover`
   - placeholder / minimal mode, currently grouped with goal-style behavior in the switch.

Important practical note:

- `policy_rollout_as_ref` is a legacy compatibility knob.
- In the current controller:
  - if `nmpc_mode=track_reference_policy` and `policy_rollout_as_ref=false`,
  - the controller maps that request to `track_policy_warmstart_goal`.
- New experiments should set `nmpc_mode` explicitly and not rely on this flag.

### 3.6 Current NMPC cost function

`NmpcController::build_problem_once()` creates a fixed Crocoddyl shooting problem once and later updates only references and warm-start values.

The running cost at each stage is structurally:

$$
\ell_t(x_t, u_t) = \frac{1}{2}\lVert W_x (x_t - x_t^{\text{ref}}) \rVert^2 + \frac{1}{2}\lVert W_u (u_t - u_t^{\text{ref}}) \rVert^2 + \ell_{\text{collision}} + \ell_{\text{state-reg}} + \ell_{\text{acc}} + \ell_{\text{bounds}}
$$

depending on options.

The terminal cost is:

$$
\ell_N(x_N) = \frac{1}{2}\lVert W_N (x_N - x_g) \rVert^2
$$

More precisely:

- running state tracking is implemented with `State_cost`,
- running control tracking / regularization is implemented with `Control_cost`,
- terminal goal cost is another `State_cost`,
- collisions use `Col_cost`,
- state regularization uses model-provided `state_weights` and `state_ref`,
- acceleration regularization uses `mujoco_quads_payload_acc`,
- state bounds use `State_bounds`.

Then `update_problem_references()` changes:

- `x_ref`,
- `u_ref`,
- and corresponding weights,

without rebuilding the problem graph.

### 3.7 Current ONNX policy contract in NMPC

The ONNX wrapper is `PolicyOnnx`.

It supports two forms:

1. single-input chunk model:
   - input: feature vector only,
   - output: flat `H * nu` action chunk.
2. legacy two-input autoregressive model:
   - input: `(x, u_prev)`,
   - output: one control,
   - chunk constructed by repeated rollout of the same observation with updated `u_prev` only.

Current NMPC policy path expects the new chunk contract.

### 3.8 What features the NMPC policy actually receives

`NmpcController::build_policy_features(...)` intentionally matches `scripts/payload_env.py::_get_learner_features()`.

The learner feature vector is:

$$
\phi(x_t, u_{t-1}) = [e_{p_L}, v_L, e_{p,1}^{rel}, e_{v,1}^{rel}, e_{q,1}, e_{\omega,1}, \dots, e_{p,n_q}^{rel}, e_{v,n_q}^{rel}, e_{q,n_q}, e_{\omega,n_q}, u_{t-1}]
$$

with:

- payload position error:
  - $e_{p_L} = p_L - p_L^{goal}$,
- payload linear velocity:
  - $v_L$,
- for each quad $i$:
  - relative position error,
  - relative velocity error,
  - quaternion error vector,
  - angular velocity error,
- previous normalized action $u_{t-1}$.

For $n_q = 2$ and $\nu = 8$:

$$
\dim \phi = 6 + 2 \times 12 + 8 = 38
$$

This is the deployed policy input contract. It is not the full raw state.

### 3.9 How the controller uses the policy

For policy modes, `prepare_window_for_step(k)` does:

1. compute learner features from current measured state,
2. call `PolicyOnnx::predict_chunk(features, prev_action, N, nu)`,
3. interpret the output as a flat `H * nu` chunk,
4. clip it to `[-1, 1]`,
5. map it to planner control space:

$$
u_{planner} = a \odot \text{half} + \text{mid}
$$

6. clip to `policy_u_clip_min/max`,
7. roll the dynamics forward to obtain warm-start states.

Then:

- `track_reference_policy` sets the policy rollout as both warm start and reference,
- `track_policy_warmstart_goal` uses the policy rollout only as warm start and keeps the goal as the optimization reference.

### 3.10 Jacobians in the NMPC stack: how and why

This is important.

The current NMPC controller path uses **analytic derivatives**, not finite differences.

#### Dynamics derivatives

`Dynamics::calcDiff(...)` ultimately calls:

- `robot_model->stepDiff(...)`, or
- `robot_model->stepDiff_with_v(...)`

depending on the control mode.

So the state transition Jacobians are:

$$
F_x = \frac{\partial f}{\partial x}, \qquad F_u = \frac{\partial f}{\partial u}
$$

computed by the Dynobench / MuJoCo model wrappers.

#### Cost derivatives

Each cost implements `calcDiff(...)` analytically, for example:

- `State_cost`
- `Control_cost`
- `Col_cost`
- `State_cost_model`
- `mujoco_quads_payload_acc`

So Crocoddyl receives analytic gradients/Hessians such as:

$$
L_x,\; L_u,\; L_{xx},\; L_{uu},\; L_{xu}
$$

per stage.

#### Why analytic derivatives are used here

Because this is a receding-horizon controller. In NMPC you pay this derivative cost at each solve. Analytic derivatives are materially faster and more stable than wrapping the action model in finite differences.

The generic Dynoplan OCP builder supports finite-difference wrapping (`ActionModelNumDiff`) when `use_finite_diff = true`, but the current agile `NmpcController` path does not use that wrapper.

This is the right choice for controller throughput.

### 3.11 Where the NMPC runtime cost is currently coming from

The main controller bottlenecks are not only the DDP iterations. Current likely hot spots are:

1. `ddp_solver_->solve(...)`
2. `rollout_warm_start_states_from_actions()`
3. `build_policy_features()`
4. repeated trajectory copying in `prepare_common_warm_start()` and `shift_and_pad()`
5. policy debug JSONL writes in `write_policy_debug_step()` when enabled

Two controller details matter for benchmark interpretation:

- control noise is zero-mean and bounded in the current code path,
- policy debug JSONL writes are reduced to solve steps or terminal events.

So when benchmarking controller speed, keep:

- `debug_policy_loop = false`
- `control_noise = 0`

## 4. Layer 3: Imitation Learning Stack

### 4.1 Main files

- learning environment:
  - `scripts/payload_env.py`
- direct binding expert:
  - `scripts/expert_pcdbcbs.py`
- subprocess expert:
  - `scripts/expert_pcdbcbs_subprocess.py`
- DAgger-H training:
  - `scripts/train_dagger_payload_horizon.py`
- memprobe wrapper:
  - `scripts/train_dagger_payload_horizon_memprobe_subprocess.py`
- evaluation:
  - `scripts/eval_dagger_payload_horizon.py`

### 4.2 Observation and action contract for learning

This must be stated precisely.

The environment observation is:

$$
\text{obs} = [\text{raw\_state},\; \phi(\text{state}, u_{t-1})]
$$

where:

- `raw_state` is the full 39-D MuJoCo payload system state,
- `phi(...)` is the 38-D learner feature vector.

So the full environment observation has dimension:

$$
39 + 38 = 77
$$

But the **learner policy does not use the raw state**. The trainer slices it away using `FeatureSliceExtractor`, so the learner actually consumes only:

$$
\phi \in \mathbb{R}^{38}
$$

This is the key observation contract. Any exported ONNX model intended for NMPC must use this feature-only input.

Action contract in learning:

- policy outputs normalized action chunk in `[-1, 1]`,
- environment maps it to planner space `[0, 1.4]`,
- then to MuJoCo thrust via:

$$
u_{mujoco} = \nu_{planner} \cdot u_{nominal}
$$

with:

$$
u_{nominal} = 0.034 \cdot 9.81 / 4
$$

### 4.3 Direct binding expert vs subprocess expert

#### Direct binding expert: `expert_pcdbcbs.py`

The original expert wrapper imports the `pcdbcbs` Python module from the C++ pybind module and calls:

```python
opt = self.pcdbcbs.Options()
res = self.pcdbcbs.run(opt)
```

This means planner + optimizer run in-process inside the Python trainer.

#### Subprocess expert: `expert_pcdbcbs_subprocess.py`

The newer expert wrapper subclasses the direct expert and overrides `_plan(...)`.

Instead of calling `pcdbcbs.run(opt)` directly, it launches:

```text
pc_dbcbs_expert
```

as a subprocess, then reads `result_subprocess_api.yaml`.

Why this change was made:

- isolate repeated planner calls from the Python process,
- diagnose memory/runtime behavior more cleanly,
- get simpler failure handling and timeouts,
- keep training robust when expert calls are expensive or unstable.

Important nuance:

- bindings were not abandoned in general,
- subprocesses were introduced specifically for the expert path used during training.

### 4.4 What happens when the expert is queried during DAgger

In the current horizon DAgger implementation, the expert policy `ExpertChunkPolicySB3` does this per queried observation:

1. take the full env observation,
2. pass the full observation to `_chunk_for_index(...)`,
3. inside that function:
   - call `expert.reset_episode()`,
   - call `expert.act(full_obs)`,
4. `expert.act(...)` extracts `obs[:state_dim]` as raw state,
5. write patched `input.yaml` with that state as the new start state,
6. run `pc_dbcbs_expert` subprocess,
7. obtain `U` from the returned optimized plan,
8. take the first `H` actions from `U`,
9. normalize them to `[-1,1]`,
10. flatten to length `H * nu`.

So each DAgger expert label is generated by **fresh replanning from the learner-visited state**.

### 4.5 DAgger-H training pipeline

`train_dagger_payload_horizon.py` uses:

- `imitation.algorithms.bc.BC`
- `imitation.algorithms.dagger.SimpleDAggerTrainer`

Current architecture:

1. build vectorized learning environments,
2. build one expert instance per env,
3. collect seed expert trajectories,
4. pretrain policy with BC on those expert trajectories,
5. hand BC trainer + expert policy + env to `SimpleDAggerTrainer`,
6. let DAgger aggregate more expert-labeled rollouts.

#### DAgger-H pseudocode

```text
build env(s)
build subprocess expert(s)

seed_trajs = rollout expert policy for seed_episodes
bc_trainer = BC(seed_trajs)
bc_trainer.train(n_epochs=bc_pretrain_epochs)

dagger_trainer = SimpleDAggerTrainer(
    venv,
    expert_policy,
    bc_trainer,
    expert_trajs=seed_trajs,
    beta_schedule=LinearBetaSchedule(...)
)

dagger_trainer.train(total_timesteps, rollout_round_min_episodes, rollout_round_min_timesteps)

save checkpoint + seed trajectories + summary
```

### 4.6 BC initialization

The BC initialization is explicit.

Seed stage:

```python
expert_trajs = _collect_seed_trajs(expert_policy, venv, n_episodes=args.seed_episodes)
transitions = rollout.flatten_trajectories(expert_trajs)
```

Then BC pretraining:

```python
bc_trainer = bc.BC(... demonstrations=transitions ...)
bc_trainer.train(n_epochs=args.bc_pretrain_epochs)
```

So the initial learner is not random. It is first fitted to seed expert demonstrations, then refined by DAgger aggregation.

### 4.7 Horizon and execution stride

The learner outputs a flattened chunk of size:

$$
H \cdot \nu
$$

The env wrapper `ChunkActionWrapper` then executes only the first $k$ actions before replanning.

This means inference is closed-loop in the receding-horizon sense:

1. predict chunk,
2. execute first $k$ actions,
3. observe new state,
4. predict again.

If `H = 1`, then this reduces to one-step closed loop.

### 4.8 Why the memprobe wrapper exists

`train_dagger_payload_horizon_memprobe_subprocess.py` is not a different trainer. It is a thin instrumentation wrapper around `train_dagger_payload_horizon.py`.

It replaces `PcDbCBSExpert` with an instrumented subclass that logs:

- RSS / HWM memory,
- expert call duration,
- plan length,
- episode id,
- call counters,
- distance-to-goal at the queried state,
- replan reason.

It writes JSONL logs to:

- `scripts/expert_memprobe_horizon_subprocess.jsonl`

Environment variable:

- `MEMPROBE_USE_DBG_EXPERT=1`
  - switches to the debug subprocess expert binary.

### 4.9 How to run DAgger-H training

Example deterministic subprocess run:

```bash
.venv/bin/python scripts/train_dagger_payload_horizon.py \
  --out-dir runs/dagger_h20_k5_10000_det \
  --horizon 20 \
  --k-mode fixed \
  --k 5 \
  --seed-episodes 20 \
  --total-timesteps 10000 \
  --bc-pretrain-epochs 300 \
  --batch-size 256 \
  --n-envs 10 \
  --expert-threads 10 \
  --vec-env subproc \
  --subproc-start-method fork \
  --pcdbcbs-deterministic \
  --motion-primitives-base /home/khaledwahba94/pc-dbCBS/motion_primitives
```

Memprobe version:

```bash
.venv/bin/python scripts/train_dagger_payload_horizon_memprobe_subprocess.py \
  --out-dir runs/dagger_h20_k5_10000_det \
  --horizon 20 \
  --k-mode fixed \
  --k 5 \
  --seed-episodes 20 \
  --total-timesteps 10000 \
  --bc-pretrain-epochs 300 \
  --batch-size 256 \
  --n-envs 10 \
  --expert-threads 10 \
  --vec-env subproc \
  --subproc-start-method fork \
  --pcdbcbs-deterministic \
  --motion-primitives-base /home/khaledwahba94/pc-dbCBS/motion_primitives
```

### 4.10 Training options summary

`train_dagger_payload_horizon.py` options:

- paths:
  - `--out-dir`
  - `--xml-path`
  - `--env-yaml`
  - `--bindings-path`
  - `--pc-dbcbs-cfg`
  - `--opt-cfg`
  - `--dynobench-base`
  - `--motion-primitives-base`
- horizon / env:
  - `--horizon`
  - `--max-steps`
  - `--seed`
- network:
  - `--hidden`
  - `--activation`
  - `--lr`
- BC / DAgger:
  - `--seed-episodes`
  - `--total-timesteps`
  - `--rollout-min-episodes`
  - `--rollout-min-timesteps`
  - `--bc-pretrain-epochs`
  - `--batch-size`
  - `--beta-rampdown-rounds`
- chunk execution:
  - `--k-mode`
  - `--k`
  - `--k-far`
  - `--adaptive-near-goal-dist`
- expert / parallelism:
  - `--expert-time-limit`
  - `--expert-n-opt`
  - `--keep-expert-files`
  - `--n-envs`
  - `--expert-threads`
  - `--vec-env`
  - `--subproc-start-method`
  - `--pcdbcbs-deterministic`

### 4.11 Evaluation script: what it does

`eval_dagger_payload_horizon.py`:

1. load checkpoint,
2. rebuild the SB3 policy,
3. reset the environment,
4. at each replan point, feed only learner features to the policy,
5. execute first $k$ actions of predicted chunk,
6. save per-episode NPZ logs,
7. optionally render videos,
8. write `episodes.json` and `summary.json`.

It evaluates policy-only closed-loop behavior, not NMPC-assisted behavior.

### 4.12 Evaluation options summary

- `--checkpoint`
- `--out-dir`
- `--n-episodes`
- `--max-steps`
- `--seed`
- `--xml-path`
- `--env-yaml`
- `--k-mode`
- `--k`
- `--k-far`
- `--adaptive-near-goal-dist`
- `--video-fps`
- `--no-render`

Example:

```bash
.venv/bin/python scripts/eval_dagger_payload_horizon.py \
  --checkpoint runs/dagger_h20_k5_10000_det/dagger_h_policy.pt \
  --out-dir runs/dagger_h20_k5_10000_det_eval \
  --n-episodes 10 \
  --k-mode fixed \
  --k 5 \
  --max-steps 200
```

## 5. Critical Implementation Notes and Current Limitations

### 5.1 The planner and optimizer are already coupled inside pc-dbCBS

The expert is not just the discrete planner. For training, the expert label comes from the full pc-dbCBS pipeline:

$$
\text{expert} = \text{discrete pc-dbCBS} + \text{init guess generation} + \text{trajectory optimization}
$$

So when saying “planner trajectory”, in this codebase that usually means the optimized output trajectory unless explicitly stated otherwise.

### 5.2 The learning policy must use feature-only input

The environment observation is 77-D, but the deployed policy input is 38-D learner features. Any mismatch here breaks ONNX deployment into NMPC.

### 5.3 `generate_init_guess_mujoco` is structurally useful but physically crude

It is good enough to bootstrap the optimizer, but it does not seed realistic quaternions/velocities. If optimization quality is poor, this is one of the first places to revisit.

### 5.4 `track_reference_policy` and `track_policy_warmstart_goal` are different experiments

They are not equivalent.

- `track_reference_policy`
  - changes both initialization and the running objective.
- `track_policy_warmstart_goal`
  - changes initialization only and keeps the optimization objective goal-centric.

If the policy is weak, the second mode is the safer benchmark.

Also note the backward-compatibility mapping:

- `track_reference_policy + policy_rollout_as_ref=false`
  behaves as `track_policy_warmstart_goal`.

### 5.5 Current NMPC controller is build-once / update-many

This is a good architectural choice. The problem graph is created once in `build_problem_once()`, then the controller only updates:

- current state,
- running references,
- warm start.

That is the right shape for real NMPC.

## 6. Minimal Command Cookbook

### 6.1 Run expert once on zerogoal

```bash
PCDBCBS_DETERMINISTIC=1 \
deps/pc-dbCBS/build/pc_dbcbs_expert \
  --input_yaml deps/pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml \
  --output_yaml runs/tmp/result_dbcbs.yaml \
  --optimization_yaml runs/tmp/result_dbcbs_opt.yaml \
  --pc_dbcbs_cfg_yaml deps/pc-dbCBS/configs/pc_dbcbs_empty.yaml \
  --opt_cfg_yaml deps/pc-dbCBS/configs/opt_training.yaml \
  --motion_primitives_base /home/khaledwahba94/pc-dbCBS/motion_primitives/ \
  --dynobench_base deps/pc-dbCBS/deps/dynoplan/dynobench/ \
  --time_limit 50000 \
  --warmstart_optimization 1 \
  --override_visualize_mujoco 1 \
  --visualize_mujoco 0 \
  --N_opt 30 \
  --result_yaml runs/tmp/result_subprocess_api.yaml
```

### 6.2 Run NMPC from Python bindings

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, 'deps/agile-payload-transport/build')
import nmpc_controller_py

ctrl = nmpc_controller_py.Controller(
    'runs/nmpc_policy_sweep_zerogoal_empty2/bc_chunk_h10_flat/track_reference_policy/iter_1/prob.yaml',
    'runs/nmpc_policy_sweep_zerogoal_empty2/bc_chunk_h10_flat/track_reference_policy/iter_1/opt.yaml',
)
ctrl.run(
    mode='track_reference_policy',
    out_yaml='runs/tmp/nmpc_result.yaml',
    out_timing_json='runs/tmp/nmpc_timing.json',
    visualize=False,
)
PY
```

### 6.3 Train DAgger-H with subprocess expert

```bash
.venv/bin/python scripts/train_dagger_payload_horizon.py \
  --out-dir runs/dagger_h20_k5_10000_det \
  --horizon 20 \
  --k-mode fixed --k 5 \
  --seed-episodes 20 \
  --total-timesteps 10000 \
  --bc-pretrain-epochs 300 \
  --batch-size 256 \
  --n-envs 10 --expert-threads 10 \
  --vec-env subproc --subproc-start-method fork \
  --pcdbcbs-deterministic \
  --motion-primitives-base /home/khaledwahba94/pc-dbCBS/motion_primitives
```

### 6.4 Evaluate a trained DAgger-H checkpoint

```bash
.venv/bin/python scripts/eval_dagger_payload_horizon.py \
  --checkpoint runs/dagger_h20_k5_10000_det/dagger_h_policy.pt \
  --out-dir runs/dagger_h20_k5_10000_det_eval \
  --n-episodes 10 \
  --k-mode fixed --k 5 \
  --max-steps 200
```

## 7. NMPC Cost Structure Analysis

### 7.1 Detailed cost terms as built in `build_problem_once()`

The NMPC problem is assembled once with these cost terms per running stage $t \in [0, N-1]$:

$$
\ell_t(x_t, u_t) = \underbrace{\frac{1}{2}\lVert W_{x} \odot (x_t - x_t^{\text{ref}}) \rVert^2}_{\text{State\_cost}} + \underbrace{\frac{1}{2}\lVert W_{u} \odot (u_t - u_t^{\text{ref}}) \rVert^2}_{\text{Control\_cost}} + \ell_{\text{col}} + \ell_{\text{state-reg}} + \ell_{\text{acc}} + \ell_{\text{bounds}}
$$

And the terminal cost:

$$
\ell_N(x_N) = \frac{1}{2}\lVert W_N \odot (x_N - x_g) \rVert^2
$$

#### Exact weight vectors and their default values

| Cost term | Weight vector | Default value | Notes |
|-----------|---------------|---------------|-------|
| Running state (`State_cost`) | $W_x$ | `Constant(nx, ref_state_tracking_weight)` = `Constant(39, 100.0)` | **Uniform across all 39 dims** |
| Running control (`Control_cost`) | $W_u$ | `Ones(nu)` initially; updated per mode | 20.0 (ref modes) or 50.0 (goal mode) |
| Terminal goal (`State_cost`) | $W_N$ | `weight_goal * goal_weight` | goal_weight zeros out payload quat (dims 3-6) and payload ang_vel (dims 24-26) |
| Collision (`Col_cost`) | scalar | `collision_weight` = 350.0 | Active only when clearance < margin |
| State regularization | `state_weights` | Mostly zero; 0.001 for quad quats, payload/quad velocities | Only when `states_reg=true` |
| Acceleration (`k_acc`) | scalar | 0.005 | Only when `states_reg=true` |

#### Critical issue: uniform running state weights

The running state cost uses `Eigen::VectorXd::Constant(nx, w)` — every dimension of the 39-dim state gets the same weight. This means:

- Payload position (3 dims): weight $w$ per dim
- **Payload quaternion (4 dims): weight $w$ per dim** — but terminal cost correctly zeros these
- Quad positions (6 dims): weight $w$ per dim
- **Quad quaternions (8 dims): weight $w$ per dim** — stiff to track, wastes solver effort
- Payload velocities (6 dims): weight $w$ per dim
- Quad velocities (12 dims): weight $w$ per dim

The terminal cost correctly uses `goal_weight` which zeros out payload quaternion and payload angular velocity. But the running cost does not — it penalizes payload quaternion deviations with the same weight as position deviations.

This creates a mismatch: the terminal cost doesn't care about payload quaternion, but the running cost forces the solver to track it. This wastes solver iterations on dimensions that don't contribute to the actual objective.

#### Effective cost magnitudes

For a single state dimension with weight $w$, the per-dim cost is $\frac{1}{2} w^2 (x_i - \text{ref}_i)^2$.

With the sweep-winning config ($w_{\text{run}} = 20$, $w_{\text{goal}} = 2000$):

| | Running cost per dim | Terminal cost per dim (active) |
|---|---|---|
| Per-dim contribution | $\frac{1}{2} \cdot 20^2 = 200$ | $\frac{1}{2} \cdot 2000^2 = 2{,}000{,}000$ |
| Terminal/running ratio | | **10,000 : 1** |

The terminal cost completely dominates for active dimensions. This explains why `track_goal` and `track_reference_nmpc_standard` achieve similar results — the running reference barely matters.

### 7.2 Mode differences in cost and warm-start

| Aspect | `track_goal` | `track_reference_nmpc_standard` | `track_reference_nmpc_refwarm` |
|--------|-------------|-------------------------------|------------------------------|
| **Warm-start** | Shift previous solution | Shift previous solution | Direct copy from reference |
| **Running state ref** | Goal | Reference trajectory | Reference trajectory |
| **Running state weight** | $w$ (uniform) | $w$ (uniform) | $w$ (uniform) |
| **Control ref** | `u_hover` | Reference controls | Reference controls |
| **Control weight** | 50.0 (`goal_control_regularization_weight`) | 20.0 (`planner_ref_control_tracking_weight`) | 20.0 |
| **Terminal** | `weight_goal * goal_weight` | `weight_goal * goal_weight` | `weight_goal * goal_weight` |

Key observations:

1. **`track_goal` and `track_reference_nmpc_standard` use the same warm-start** — both call `prepare_common_warm_start()` which shifts the previous solution.

2. **`track_reference_nmpc_refwarm` uses a fundamentally different warm-start** — it copies the reference trajectory directly (`warm_start_N_ = ref_traj_N_`). This means it does NOT benefit from the previous solve.

3. After the reference trajectory is exhausted (~111 steps for zerogoal), `track_reference_nmpc_standard` pads with goal/hover — becoming effectively identical to `track_goal`.

### 7.3 NMPC sweep results and root-cause analysis

Results on the zerogoal problem (start [-1.5, 0, 0.567] → goal [0, 0, 0.567]):

#### Why `track_reference_nmpc_refwarm` diverges

| Config | goal_dist | Hz | Status |
|--------|-----------|-----|--------|
| refwarm, N=25, iter=10, wg=2000, run_w=20 | 10.23 | 6.3 | **DIVERGED** |
| refwarm, N=25, iter=20, wg=2000, run_w=20 | 0.025 | 6.9 | OK but 2× slower |
| standard, N=25, iter=10, wg=2000, run_w=20 | 0.025 | 11.6 | OK |

Root cause: `refwarm` copies the reference trajectory as warm-start. The reference ends at step ~111, so for a window that extends past it, the warm-start has a discontinuity (reference states then goal-padded states). With $w_g = 2000$ creating enormous terminal cost, the solver must make large corrections from this ill-conditioned starting point. 10 iterations is not enough; 20 barely suffices but halves throughput.

`standard` mode shifts the previous solution, which already incorporates the terminal cost from the last solve. Each successive warm-start is close to the new optimum → fewer iterations needed.

**Conclusion: `track_reference_nmpc_refwarm` should not be used with high terminal weights.** Use `track_reference_nmpc_standard` instead.

#### Relationship between `track_goal` and `track_reference_nmpc_standard`

At high iteration counts (iter=10), both modes achieve similar results because the terminal cost dominates the running cost by ~10,000:1 per active dimension. However, **when iterations are limited, `standard` mode is more robust** because the reference trajectory provides guidance the solver can exploit in fewer iterations.

Evidence from the final sweep (N=25, run_w=20):

| Config | goal_dist | Hz |
|--------|-----------|-----|
| standard, iter=5, wg=2000, reg=10.0 | 0.058 | 15.6 |
| goal, iter=5, wg=2000, reg=10.0 | 0.866 | 16.4 |

At iter=5/wg=2000/reg=10, standard converges while goal diverges. The reference trajectory matters when the solver has limited iterations.

#### The real bottleneck: `init_reg`, not running cost shape

Extensive experiments tested two hypotheses for why iter=5 diverges:

**Hypothesis 1 (disproved): Uniform running weights on quaternion dimensions waste solver effort.**

A `running_cost_goal_weight_mask` option was implemented and tested. It multiplies the running state weight by `goal_weight`, zeroing payload quaternion and angular velocity. Result: **no measurable effect** — identical goal_dist and Hz with and without the mask at all iteration counts. The 7 masked dimensions (of 39) change only 18% of the Hessian, and the terminal cost dominates regardless.

**Hypothesis 2 (confirmed): `init_reg=0.1` is too low for the BoxFDDP solver to converge in <10 iterations with `wg=2000`.**

| Config | Iter | WG | init_reg | goal_dist | Hz |
|--------|------|----|----------|-----------|-----|
| goal, iter=5, wg=2000, reg=0.1 | 5 | 2000 | 0.1 | **DIVERGES** | 20 |
| goal, iter=5, wg=2000, reg=1.0 | 5 | 2000 | 1.0 | **DIVERGES** | 18 |
| goal, iter=5, wg=2000, reg=10.0 | 5 | 2000 | 10.0 | 0.87 | 17 |
| goal, iter=5, wg=1000, reg=1.0 | 5 | 1000 | 1.0 | **0.043** | **16.7** |
| goal, iter=7, wg=2000, reg=0.1 | 7 | 2000 | 0.1 | 0.060 | 14.4 |
| goal, iter=7, wg=2000, reg=1.0 | 7 | 2000 | 1.0 | **0.024** | **14.5** |
| goal, iter=10, wg=2000, reg=0.1 | 10 | 2000 | 0.1 | 0.023 | 12.0 |

With `wg=2000`, the terminal cost gradient is extremely steep ($2000^2 = 4M$ per dim). At `init_reg=0.1`, the DDP backward pass must gradually increase regularization to handle this, consuming several of the allowed iterations for regularization adaptation before the solver can make progress on the trajectory. At `init_reg=1.0`, the solver starts with sufficient regularization and converges in fewer iterations.

### 7.4 Winning NMPC configurations from sweeps

**Best quality (recommended for deployment):**
```yaml
nmpc_mode: track_reference_nmpc_standard
N: 25
max_iter: 7
weight_goal: 2000
ref_state_tracking_weight: 20.0
planner_ref_control_tracking_weight: 20.0
goal_control_regularization_weight: 40.0
init_reg: 1.0
th_stop: 0.0001
# Result: goal_dist=0.027, 13.7 Hz (standard), 0.024 @ 14.6 Hz (goal)
```

**Best speed (when sub-5cm accuracy is acceptable):**
```yaml
nmpc_mode: track_reference_nmpc_standard
N: 25
max_iter: 5
weight_goal: 1000
ref_state_tracking_weight: 20.0
planner_ref_control_tracking_weight: 20.0
goal_control_regularization_weight: 40.0
init_reg: 1.0
th_stop: 0.0001
# Result: goal_dist=0.045, 16.7 Hz (standard), 0.043 @ 16.7 Hz (goal)
```

**Previous baseline (for reference):**
```yaml
# iter=10, wg=2000, init_reg=0.1 -> 0.025 @ 11.6 Hz
```

The key tuning insight: **`init_reg` must be scaled with `weight_goal`.** Higher terminal weights need higher initial regularization for the BoxFDDP solver to converge efficiently. The relationship is roughly `init_reg ≈ 1.0` for `wg=1000-2000`.

#### `running_cost_goal_weight_mask` option

Implemented in `options.hpp` / `nmpc_controller.cpp`. When `true`, multiplies the running state weight vector by `goal_weight`, zeroing payload quaternion and angular velocity dimensions. This is conceptually correct (standard quadrotor NMPC practice) but has no measurable effect on convergence with the current cost magnitudes. Available for future use if the terminal-to-running cost ratio changes.

## 8. Policy Observation Normalization

### 8.1 The problem: feature scale mismatch

The 38-dim learner feature vector has components spanning orders of magnitude:

| Feature | Dims | Typical range | Scale |
|---------|------|---------------|-------|
| payload_pos_err | 3 | [-1.5, 1.5] m | ~1 |
| payload_vel | 3 | [-2, 2] m/s | ~1 |
| per-quad rel_pos_err | 3×2 | [-0.5, 0.5] m | ~0.1 |
| per-quad rel_vel_err | 3×2 | [-2, 2] m/s | ~1 |
| per-quad quat_err_vec | 3×2 | [-0.01, 0.01] | ~0.01 |
| per-quad ang_vel_err | 3×2 | [-1, 1] rad/s | ~0.5 |
| prev_action | 8 | [-1, 1] | ~0.5 |

Without normalization, the MLP's gradient flow is dominated by large-scale features (position errors ~1). Quaternion error features (~0.01) are effectively invisible — 100× smaller contribution to the loss. The network cannot learn orientation-dependent behavior.

### 8.2 Running normalization (Welford online mean/var)

The `FeatureSliceExtractor` in `train_dagger_payload_horizon.py` implements online standardization:

1. During training, track per-feature running mean and variance using Welford's algorithm.
2. Normalize: $(x - \mu) / \sigma$, clamped to $[-10, 10]$.
3. Statistics are stored as registered buffers in `state_dict` → survive checkpoint save/load.

Benefits:
- Gradient balance across all 38 features
- Learning rate effective for all features equally
- Quaternion errors become first-class signal to the network

### 8.3 Policy I/O contract is unchanged

- **Input**: 38-dim learner features (always)
- **Output**: $H \times \nu$ flat actions in $[-1, 1]$ (always)

The normalization is an internal preprocessing step. For ONNX deployment, the running mean/std are baked into the ONNX graph as constant operations before the MLP. The C++ NMPC side computes 38-dim features via `build_policy_features()` and passes them unchanged to the ONNX model. No C++ changes needed.

### 8.4 ONNX export with baked normalization

`scripts/export_dagger_policy_onnx.py` handles both cases:

- **`normalize=False` checkpoint**: ONNX graph is `features → MLP → clamp`
- **`normalize=True` checkpoint**: ONNX graph is `features → (x-μ)/σ → clamp → MLP → clamp`

In both cases, the ONNX input is `obs_features` (38-dim) and the output is `u_flat` (H×nu).

The `NormalizedDeterministicPolicyHead` extracts `_running_mean` and `_running_var` from the trained feature extractor and registers them as constant buffers. With `do_constant_folding=True`, PyTorch's ONNX export bakes these into the graph as fixed constants.

## 9. Recommended Questions for the Next Iteration

1. Does dimension-aware running cost (using `goal_weight` shape) allow iter=3-5 convergence at ~25+ Hz?
2. With normalization, does the policy learn orientation-dependent behavior (visible in quaternion feature gradients)?
3. Is `solve_every_k_steps=2` sufficient for tracking quality with the current dt=0.02s?
4. Should the running cost use the model's `state_weights` vector instead of uniform weights?
5. Should signed distance / obstacle features be added to the learner observation before obstacle-rich training?

## 10. Claude Opus 4.6 Handover Prompt

```text
You are continuing work in the repository `imitation-agile-payload-manipulation`.

Read this document first:
- docs/FULL_STACK_PCDBCBS_NMPC_IL_HANDOFF.md

Then use the code directly from these locations:
- Planner / expert:
  - deps/pc-dbCBS/src/pc_dbcbs_expert.cpp
  - deps/pc-dbCBS/src/pc_dbcbs_api.cpp
  - deps/pc-dbCBS/src/pc_dbcbs_utils.hpp
  - deps/pc-dbCBS/src/init_guess_mujoco.cpp
  - deps/pc-dbCBS/deps/dynoplan/src/tdbastar/tdbastar.cpp
  - deps/pc-dbCBS/deps/dynoplan/src/optimization/generate_ocp.cpp
  - deps/pc-dbCBS/deps/dynoplan/src/optimization/ocp.cpp
- NMPC / controller:
  - deps/agile-payload-transport/src/nmpc_controller.cpp
  - deps/agile-payload-transport/include/nmpc_controller.hpp
  - deps/agile-payload-transport/src/croco_models.cpp
  - deps/agile-payload-transport/include/croco_models.hpp
  - deps/agile-payload-transport/src/policy_onnx.cpp
  - deps/agile-payload-transport/src/py_nmpc_controller.cpp
- Imitation learning:
  - scripts/payload_env.py
  - scripts/expert_pcdbcbs.py
  - scripts/expert_pcdbcbs_subprocess.py
  - scripts/train_dagger_payload_horizon.py
  - scripts/train_dagger_payload_horizon_memprobe_subprocess.py
  - scripts/eval_dagger_payload_horizon.py

Current facts that must be respected:
1. The expert used for learning is the full pc-dbCBS pipeline, including trajectory optimization.
2. The learner input contract is feature-only, not the full raw 77-D observation.
3. The environment observation is `[raw_state, learner_features]`, but the learner consumes only `obs[state_dim:]`.
4. The deployed ONNX contract for NMPC should remain `input = 38-dim learner features`, `output = flat(H*nu)`. When normalization is used, mean/std are baked into the ONNX graph as constants — the C++ side does not change.
5. In the NMPC controller, `track_reference_policy` means policy warm-start + policy running reference, while `track_policy_warmstart_goal` means policy warm-start + goal reference.
6. The current NMPC controller uses analytic derivatives from `stepDiff` and cost `calcDiff`, not finite differences.
7. Dynoplan/agile optimization code assumes the older Crocoddyl API line; do not casually migrate it to Crocoddyl 3.x semantics.
8. The NMPC running state cost uses uniform weights on all 39 dims. A `running_cost_goal_weight_mask` option was added to zero irrelevant dims but has no measurable effect because the terminal cost dominates by ~10,000:1.
9. `track_reference_nmpc_refwarm` diverges with high terminal weights — use `track_reference_nmpc_standard` instead.
10. **`init_reg` must be scaled with `weight_goal`**: `init_reg=1.0` for `wg=1000-2000`. This is the key tuning insight — it enables iter=7 convergence (14.5 Hz) instead of requiring iter=10 (12 Hz).
11. `track_reference_nmpc_standard` is more robust than `track_goal` at low iteration counts because the reference trajectory provides solver guidance. Use `standard` for deployment.

When proposing changes, separate these three questions cleanly:
- planner quality,
- policy quality,
- controller integration quality.

Do not conflate them.

If you modify training or deployment, keep all three contracts aligned:
- env feature computation,
- training input/output shapes,
- NMPC ONNX input/output shapes.
```
