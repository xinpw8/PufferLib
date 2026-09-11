# GEAR-SONIC vector reference environment

`gear_sonic_vector_env.py` provides a correctness-first Python vector runtime
for the public GEAR-SONIC family candidate. It does not establish REK parity.
The runtime shares one MuJoCo `MjModel`, one encoder session, one decoder
session, and one `GearSonicController`. Every environment owns a distinct
`MjData`, history, previous action, policy tick, heading alignment, and command
filter state.

## Schedule and control boundary

One call to `GearSonicVectorEnv.step()` performs one 50 Hz policy decision for
every environment. Each decision advances every `MjData` through ten 0.002 s
MuJoCo steps, giving 500 Hz physics and 0.02 s of simulated time. The native
scheduler receives a nominal 200 Hz command-filter work rate. Its ties-to-even
interval calculation resolves to two physics steps, so the effective update
rate is 250 Hz and there are five updates per policy decision. Target scaling,
joint-order conversion, native position-actuator configuration, joint-range
clipping, torque prediction, state history, observation construction, and
quaternion handling all call the functions in `gear_sonic_candidate.py`. The
scalar candidate runner uses the same vector kernel with `num_envs=1`.

## Construction

```python
from pathlib import Path

from gear_sonic_vector_env import GearSonicVectorEnv

environment = GearSonicVectorEnv.from_artifacts(
    bundle=Path("C:/path/to/pinned-public-bundle"),
    assets_dir=Path("C:/path/to/extracted-runtime-assets"),
    motion_role="idle",
    manifest=Path("C:/path/to/g1_runtime_assets.v1.json"),
    xml=Path("C:/path/to/g1_29dof.recovered.xml"),
    arena=Path("C:/path/to/g1_arena_physics_contract.v1.json"),
    num_envs=8,
    control_boundary="native-position-actuator",
    frame_mode="loop",
    force_limit_source="public-model-config",
    physics_workers=1,
)

steps = environment.step()
assert len(steps) == environment.num_envs
assert all(step.physics_steps == 10 for step in steps)
```

`reset()` can reset all members or an explicit set of unique indices. A
selected reset replaces only that member's MuJoCo and controller state. A
custom `reference_resolver` must implement `reset(indices)` before environment
reset is allowed, because the runtime cannot determine whether an arbitrary
callable has hidden cursor state.

If any resolver, inference, or physics operation raises during `step()`, the
environment enters a failed state. Further steps fail closed. A complete reset
is required because some environments may already contain staged history or
completed physics steps. A partial reset is rejected in this state.

## Physics worker lifecycle

`physics_workers` is a positive integer and defaults to 1. Values above 1
create one persistent `ThreadPoolExecutor`. Observation construction and shared
policy inference finish first. The per-environment physics operation then runs
through `executor.map`, which preserves environment result order while each
worker mutates only its assigned `MjData`. Call `close()` when finished or use
the environment as a context manager:

```python
with environment:
    steps = environment.step()
```

Set `physics_workers=16` in the preceding construction call to opt into that
worker count.

A worker exception shuts down the executor and waits for all submitted work
before returning the error. The vector remains poisoned until a complete reset
reinitializes every member and creates a fresh executor.

An isolated Spark probe used one shared MuJoCo model, 64 distinct `MjData`
instances, and ten physics substeps. The resulting `qpos` bytes matched for 1,
2, 4, 8, and 16 workers. Throughput increased from 663 policy samples/s with
one worker to 4,893 policy samples/s with 16 workers. Evidence ID:
`spark-g1-mujoco-threadpool-scaling-probe-1`. Those measurements cover only the
physics kernel. They exclude observation construction, ONNX inference, motion
composition, and end-to-end environment overhead.

## ONNX batching

The pinned public graphs declare `obs_dict` shapes `[1, 1762]` and `[1, 994]`.
For `N > 1`, the controller explicitly calls those fixed-batch graphs once per
environment while reusing the same session objects. It never pads a batch or
changes the graph.

The controller also accepts a compatible session whose declared input batch is
dynamic or exactly `N`. Such a session receives one `[N, width]` call. Output
shape, `float32` dtype, and finite values are checked in both paths. The current
mode is available as `controller.encoder_batch_mode` and
`controller.decoder_batch_mode`.

`from_artifacts(..., batch_models_dir=path)` can select an explicit batch-N
bundle that passes `gear_sonic_batch_model.py` source, recipe, graph-shape, and
output-hash validation for the requested environment count. Omitting that
argument preserves the pinned public graphs and their per-environment fallback.

## Reference cursor injection

`reference_resolver` is an optional callable with this contract:

```python
def resolve(env_index: int, policy_tick: int) -> tuple[int, Mapping[str, object]]:
    ...
```

The selected frame feeds the existing observation builder. Resolver metadata
is copied to `GearSonicStep.reference_metadata`. Tick zero must resolve frame
zero because reset heading alignment uses the first reference frame. The scalar
runner installs the exact-contract `MotionComposerCursorProvider`, whose pinned
clips advance by exactly one whole frame per 50 Hz policy tick. That scalar path
does not invoke environment reset.

The vector API does not convert held W, S, A, D, Q, E state into motion roles.
The current Python semantic envelope validates a caller-supplied `motion_role`
and explicitly records that REK runtime clip selection has not been recovered.
Adding an automatic binding would invent behavior.

## Verification and limits

Run the focused unit suite from `ocean/rek_g1`:

```powershell
python -m unittest -v test_gear_sonic_vector_env.py test_gear_sonic_candidate.py
```

The vector tests cover fixed-batch session reuse, compatible batched sessions,
ten-step scheduling, serial and parallel physics equivalence, deterministic
parallel result order, worker validation and lifecycle, selected reset
isolation, fail-stop recovery, resolver lifecycle checks, distinct `MjData`
enforcement, reference metadata propagation, and equivalence between an
independent vector member and an N=1 environment. They use a deterministic fake
MuJoCo boundary, so execution does not require external model assets.

The current server-only Sonic runtime config was not inspected or inferred by
this reference. Remaining gaps include packaged REK model byte identity,
authoritative held-input clip selection, fractional composer interpolation,
fight contacts, damage, rewards, opponent logic, and comparison with a
held-out REK trajectory.
