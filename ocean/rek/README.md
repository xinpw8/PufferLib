# ocean/rek

This directory contains native PufferLib MuJoCo plant diagnostics. The sibling
`ocean/rek_fight` environment is the high-level training sim: keyboard-level
actions on the recovered two-T800 plant. `ocean/rek_sandbox` is a balance
curriculum only and is not the fight policy.

What was here before was a fighting-game move state machine over a scalar
balance model: frame envelopes, circular hit volumes, hand-picked mass
exponents, invented friction and get-up timing. It was fast and internally
consistent, and it was a new game inspired by REK's controls rather than a
reconstruction of REK's transition function. A policy trained against it has no
reason to transfer. It is quarantined on the `rek_proxy` branch and is not a
training target.

For RL purposes a clone has to reproduce

    (s_{t+1}, r_t, d_t, e_t) = F(s_t, a_t^0, a_t^1, seed)

matching REK on action semantics, control and physics tick rates, input latency
and buffering, articulated dynamics, contact detection, balance and recovery,
hit/score/fall/round events, per-robot parameters, and reset and randomisation
distributions. The renderer does not need parity; the transition function does.

## Implemented plant profiles

- `rek` is the serialized client T800 FactoryPolicy plant associated with the
  private Bot 1 evidence. It has 25 motors, 37 robot geoms, 17 recovered static
  arena geoms, `nq=32`, `nv=31`, and a 63-float raw state observation.
- `rek_g1` is the G1 29-DoF plant intended for later physical-robot work. It
  has 29 motors, `nq=36`, `nv=35`, and a 71-float raw state observation.
- `rek_match` is a two-agent T800 diagnostic. One physical environment contains
  two independently actuated T800 copies and the measured static arena. Each
  agent receives a 126-float ego-first view of both raw `qpos + qvel` blocks and
  controls only its own 25-actuator block. Both agents receive the same terminal
  boundary and exactly zero reward.
- `rek_sandbox` exposes one learner and one deterministic internal dummy per
  physical match. It adds a provisional actuator-only root stabilizer, balance
  reward, and episode boundary so the plant can be trained without REK, a game
  window, desktop input, or a network service.

The single-robot profiles load a derived MJCF from `REK_MJCF_PATH`, enforce the
recovered model dimensions, MuJoCo settings, actuator names, actuator order,
serialized control ranges, and the `ctrllimited` flags, then share one immutable
`mjModel` across vector slots with one `mjData` per slot. Policy actions are
unbounded latent floats mapped through `tanh` to the measured motor control
ranges. Rewards are exactly zero. The configuration files set
`total_timesteps = 0` so this diagnostic cannot be mistaken for a useful
learning task.

The harness does not contain the REK velocity controller, move controller,
opponent, contact event rules, damage, score, fall/recovery logic, rewards, or
reset distribution. The T800 profile contains the identical static collider
geometry measured in all three shipped arena levels. Its initial root transform
is still serialized prefab state, not a measured match spawn. The recovered
reports explicitly set `control_equivalent: false`. Raw `qpos + qvel` and direct
motor controls are diagnostic interfaces, not a claim about REK policy
observations or actions.
For G1, the recovered `ctrllimited` flag is false; its serialized ranges are
used only as explicit action scales. T800 has `ctrllimited` enabled.

The `rek_match` reset is one hash-bound client-observed first-active frame from
a remote-authoritative private Bot 1 round. Its 50 hinge coordinates were fitted
from the recorded named-bone poses and reproduce that visual frame within the
reported kinematic residual. This is one pose, not a measured reset
distribution. Joint velocities default to zero because they were not recorded.
Both uncontrolled actors fall, and no result from this profile establishes Bot
1 behavior, authoritative server physics, controller equivalence, or parity.

The installed client contains no runnable T800 ONNX, active policy JSON, or
T800 move trajectory. `rek_sandbox` therefore keeps every unmeasured choice
explicitly provisional. It uses the measured PdStand gains and force limits as
implicit position servos around the observed keyframe, then applies a fixed
finite-difference root-acceleration feedback layer through the 12 leg targets.
The stabilizer held both fighters above the probe fall gate for 15,000 steps,
or 300 s, on Spark. That is a curriculum result, not controller parity.

The deterministic sources are:

- `rek.h`: plant selection, model validation, reset, observation, and step
- `binding.c`: Puffer vector contract and shared-model lifecycle
- `../rek_g1/binding.c`: explicit G1 build profile
- `test_rek.c`: strict standalone checks for every actuator, non-zero action
  effect, timeout/reset behavior, finite state, and buffer guards
- `../rek_match/rek_match.h`: two-agent match state, isolated action blocks,
  ego-first observations, keyframe reset, and shared terminal handling
- `../rek_match/binding.c`: two-agent Puffer vector layout and shared-model
  lifecycle
- `../rek_match/test_rek_match.c`: strict standalone match contract checks
- `../rek_sandbox/rek_sandbox.h`: one-policy curriculum, internal dummy,
  provisional stabilizer, reward, and episode lifecycle
- `../rek_sandbox/test_rek_sandbox.c`: strict standalone controller, terminal,
  deterministic-rollout, shared-model, and 300 s stability checks
- `evidence/mujoco_plant.py`: measured-asset to MJCF derivation
- `evidence/mujoco_arena.py`: native-code-backed static arena composition
- `evidence/mujoco_match.py`: hash-bound two-T800 composition and pose fitting
- `evidence/mujoco_validate.py`: independent MuJoCo loader/step validation

## Build boundary

Use an official native MuJoCo 3.7.0 distribution for the target architecture.
The Windows game DLL is never copied or linked on Spark.

```bash
export MUJOCO_HOME=/absolute/path/to/the/mujoco/python/package
export MUJOCO_LIB="$MUJOCO_HOME/libmujoco.so.3.7.0"
export REK_MJCF_PATH=/absolute/path/to/t800_factory.recovered.xml
./build.sh rek --cpu
./build.sh rek
```

For G1, build `rek_g1` and point `REK_MJCF_PATH` at
`g1_29dof.recovered.xml`.

For the two-agent diagnostic, build `rek_match` and point `REK_MJCF_PATH` at
`t800_t800_factory_arena.diagnostic.xml`. Its non-training configuration is
`config/rek_match_diagnostic.ini`.

For fight training, build `rek_fight` with the same two-fighter MJCF and
`config/rek_fight.ini`. See `ocean/rek_fight/README.md`.

For the server-free balance curriculum, build `rek_sandbox`, use the same
two-fighter MJCF, set `REK_CONTROLLER_PATH` to the hash-bound
`controller_path.json`, and set `REK_ROOT_CONTROLLER_PATH` to the hash-bound
root-controller search JSON. The loader enforces MuJoCo 3.7.0 at runtime. Its
training configuration is `config/rek_sandbox.ini`.

The Spark verification artifacts record successful ARM64 standalone, CPU
Puffer, CUDA Puffer, CPU vector-step, CUDA action-bridge, deterministic reset,
and shared-model close tests for both profiles. These are execution and
integration results only.

The same checks now pass for `rek_match`. With eight policy-agent rows, the
recorded 2,000-step vector smoke completed 16,000 agent steps at approximately
39,740 agent-steps/s. That measures the diagnostic direct-motor harness only;
it is not a throughput estimate for a recovered REK transition function.

With the provisional stabilizer enabled, the recorded 64-environment Spark
vector smoke completed 128,000 steps at approximately 23,057 agent-steps/s.
A real 16,384-step CUDA Puffer optimizer smoke ran at approximately 29,500
agent-steps/s and wrote four hashed checkpoints. Standalone network tracing
recorded zero network syscalls. These results establish a local training path,
not game-mechanics or trajectory parity.

Behavioral work still starts and ends with evidence. See
[`evidence/README.md`](evidence/README.md) for the gated sequence and its held-out
trajectory/event acceptance criterion. No plant result weakens that gate.

`tools/arm64/` and `tools/spark_train.sh` predate the plant harness. They must
not be used for `rek`, `rek_g1`, `rek_match`, or `rek_sandbox` until they
propagate the exact MuJoCo library and MJCF paths and run the current native
tests.
