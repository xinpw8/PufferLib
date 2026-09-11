# G1 Sonic-family diagnostic candidate

`sonic_candidate.py` runs the official NVlabs ProtoMotions G1 unified tracker
against build-pinned REK motion NPZ files and the recovered 29-DoF G1 XML.
Every output is classified `diagnostic_candidate` with
`rek_parity_claim: false`. A successful gate means that artifact identity,
schemas, mappings, and runtime health passed. It does not mean REK trajectory
parity passed.

The ONNX model, YAML, and extracted REK payloads stay outside this repository.
The runner accepts paths to those files and validates them before inference.

## Pinned evidence

| Artifact | SHA-256 |
| --- | --- |
| `unified_pipeline.onnx` | `a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c` |
| `unified_pipeline.yaml` | `9b7896f3355a9d9d5e7d3139b83924eeb2e45c62c30bfda44afe996cfc6cf01c` |
| `g1_29dof.recovered.xml` | `811fdc1e5bee74026b780974207cbcd628cdd83a249d3f76b75a668d71aad835` |
| `g1_arena_physics_contract.v1.json` | `67128d45f8b5995d57b5ca925a2db7b2d613ede15bfa19f8df46c4e01c60e3e8` |
| canonical G1 asset manifest | `09973b2793f5b3e546a4f32cbf6128a13100c2332e3ed18c7e3eb46398618367` |

The source XML contains one free joint, 29 hinge joints, and 29 direct motor
actuators with scalar gear 1. Its actuator control and force limit flags are
disabled. The runner computes torque as
`kp * (target - position) - kd * velocity` and manually clips each torque to
the actuator's recovered `ctrlrange`. It recomputes this torque from the live
joint state on every 1 ms physics step. Recovered passive joint damping and
friction loss remain enabled. Because the source actuator has both limit flags
disabled, treating its declared `ctrlrange` as a torque cap is explicitly a
candidate safety assumption, not a recovered enforcement rule.

The ONNX boundary follows the official ProtoMotions deployment implementation
at commit `607ca7a0bb92e261120bcab8d9f97f28b3130ffc`:

- [`deployment/test_tracker_mujoco.py` lines 341-379](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/test_tracker_mujoco.py#L341-L379)
  establishes the MuJoCo state slices, wxyz to xyzw reorder, torso anchor, and
  direct use of the already-local free-joint angular velocity.
- [`deployment/tracker_inputs.py` lines 156-200](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/tracker_inputs.py#L156-L200)
  maps those values and future references to the semantic ONNX inputs.
- [`deployment/test_tracker_mujoco.py` lines 720-775](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/test_tracker_mujoco.py#L720-L775)
  shows that `historical.processed_actions` receives the previous processed PD
  position targets, not the previous raw `actions` output.
- [`deployment/state_utils.py` lines 296-350](https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/state_utils.py#L296-L350)
  defines the fixed yaw-only start alignment applied to future rotations.

No quaternion sign canonicalization is added. MuJoCo quaternions are reordered
from wxyz to xyzw at the ONNX boundary. The pinned policy disables target
acceleration clamping and sets EMA alpha to 1, so the processed target fed back
to history is the returned `joint_pos_targets` value.

The source XML timestep is the recovered `2822399/141120000` s and it has no
floor. Executable runs require exactly one environment input. The preferred
input is the hash-pinned arena contract, which composes one floor, eight
pillars, and eight walls with their recovered box poses and contact settings.
`--diagnostic-floor-z` remains available for isolated synthetic diagnostics.
The runner records the timestep change to the pinned policy's 0.001 s physics
step. The ONNX policy runs every 20 physics steps, which gives 50 Hz control.

## Validate without simulation

```bash
python ocean/rek_g1/sonic_candidate.py \
  --onnx /external/compiled_models/unified_pipeline.onnx \
  --yaml /external/compiled_models/unified_pipeline.yaml \
  --assets-dir /external/g1-runtime-assets/f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659 \
  --motion-role idle \
  --arena-contract ocean/rek/evidence/evidence_out/g1_arena_physics_contract.v1.json \
  --validate-only
```

Validation loads the real ONNX session on the CPU provider, checks every
input and output, compiles the recovered XML, constructs the 29-joint named
mapping, validates the selected motion and extraction inventory, and derives
all reference torso rotations through recovered-plant forward kinematics. If
an arena contract is supplied, validation also composes and verifies all 17
named boxes in the compiled model.

## Run one held-duration segment

```bash
python ocean/rek_g1/sonic_candidate.py \
  --onnx /external/compiled_models/unified_pipeline.onnx \
  --yaml /external/compiled_models/unified_pipeline.yaml \
  --assets-dir /external/g1-runtime-assets/f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659 \
  --motion-role idle \
  --steps 20 \
  --arena-contract ocean/rek/evidence/evidence_out/g1_arena_physics_contract.v1.json \
  --metrics-out /external/results/idle-20.json \
  --trace /external/results/idle-20.jsonl
```

`--motion-role` and `--steps` form one held-duration diagnostic segment. The
maximum duration is the motion frame count minus eight, so every control tick
has complete future references at offsets `[1, 2, 4, 8]`. With no `--steps`,
the runner uses that complete valid span.

For a synthetic contact-only diagnostic, replace `--arena-contract ...` with
`--diagnostic-floor-z 0`. Reports label that mode
`synthetic_diagnostic_floor` and never treat it as recovered arena evidence.

## Semantic adapter envelope

`--segment-json` accepts a machine-readable envelope whose
`semantic_command` fields match `RekG1SemanticCommand` in `semantic_action.h`.
It replaces `--motion-role` and `--steps`.

```json
{
  "schema": "rek.g1_sonic_candidate.segment.v1",
  "classification": "diagnostic_candidate",
  "motion_role": "kick_left_front",
  "duration_ticks": 40,
  "semantic_command": {
    "kind": 1,
    "held_code": 9,
    "duration_ticks": 40,
    "kick_registry_index": 2
  }
}
```

Kinds use the C enum values: `0` for locomotion and `1` for kick. Locomotion
uses the `UINT16_MAX` kick sentinel `65535`. A kick held code may contain yaw
and cannot contain translation. This is the direct semantic segment envelope.
Puffer adapter-table kick templates are instead normalized to neutral and
inherit the adapter's current desired yaw when dispatched. The envelope
supplies the motion-role binding. That binding is recorded as caller supplied
because the available evidence does not recover REK runtime clip selection.

## Explicit derivations and remaining gaps

- The NPZ files contain joint position, pelvis position, pelvis rotation, and
  frame rate. Joint velocity uses REK's recovered forward finite difference
  with a zero terminal frame.
- Future torso orientation uses MuJoCo forward kinematics from each NPZ pelvis
  pose and 29-joint pose, followed by the official yaw-only start alignment.
  This torso reconstruction is classified as an inference because the NPZ
  schema omits the per-body rotations consumed by the official deployment
  motion interface.
- Initial joint positions are clipped to recovered joint ranges, with every
  affected joint and maximum correction recorded.
- Historical processed action input receives the prior ONNX joint-position
  target, matching the public ProtoMotions deployment runner.
- The arena contract establishes static geometry, contacts, timestep, and two
  spawn anchors for the pinned client build. It does not establish the
  replacement free-joint pelvis height. The candidate therefore initializes
  from the motion reference and records that no REK spawn rebase was applied.
- The synthetic plane option is a diagnostic contact surface. It does not
  imply recovered arena geometry or contact material.
- REK uses a separate Sonic encoder, decoder, configuration, motion composer,
  and runner state machine. Their payloads and complete runtime traces remain
  unavailable. Trajectory acceptance remains false in all reports.
- The candidate preserves recovered direct-motor, passive damping, and joint
  friction semantics. The official ProtoMotions runner instead rewrites its
  training plant to implicit PD and zeros passive terms. This deliberate plant
  difference remains a trajectory-compatibility unknown.

## Tests

```powershell
python -m unittest -v ocean\rek_g1\test_sonic_candidate.py
```

The tests use schema fixtures, the hash-pinned recovered XML, and the pinned
arena contract. They verify the arena hash, schema, timestep, 17 boxes, and
injected contact attributes. They do not require or create copies of the ONNX
model or extracted game payloads.
