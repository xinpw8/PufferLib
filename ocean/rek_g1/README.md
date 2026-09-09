# REK Unitree G1 semantic environment

This directory contains the native, state-based Unitree G1 duel candidate for
PufferLib. It is executable on the DGX Spark and no longer exposes direct
actuator actions. It is not yet accepted as a control-equivalent REK clone.
Acceptance requires held-out REK action sequences whose trajectory and event
errors are no greater than REK's repeated-run variance.

## Runtime boundary

One arena contains two physical G1 robots in one MuJoCo model. Puffer rows are
arena-major, player then opponent. Every 50 Hz outer action advances ten 2 ms
physics steps. Both fighters' controls are staged before the shared-contact
step, so neither row receives an ordering advantage.

The native observation ABI is schema 3 with 223 binary32 values per robot:

* 86 values for self state;
* 86 values for opponent state;
* 12 command, heading, route, and composer values;
* 39 score, fall, referee, round, and tick-event values.

The one-head categorical action ABI has 20 categories:

* continue the currently scheduled command;
* neutral and the validated W, S, A, D, Q, and E held combinations;
* the four recovered G1 kick routes for runtime moves 6 through 9.

Held translation and yaw are represented every controller tick. Q or E can be
held with a translation input. Translation blocks a new kick until the
locomotion transition settles. An accepted kick suppresses effective yaw while
the desired yaw hold and ramp state continue. No F binding is present because
the installed keyboard asset has no identified F locomotion field.

## Implemented REK semantics

The runtime contains native motion composition, batched ONNX policy execution,
shared-arena MuJoCo contacts, build-pinned fall measurement, hit filtering,
strike attribution, scoring, referee counts, round timing, paired terminal
events, and arena-local episode reset. Its score-delta reward is an explicit
training contract, not a recovered REK reward.

Pinned static evidence and recovered code establish these fall and reset facts:

* the selected serialized G1 prefab has null prone and supine recovery clips;
* a committed fall suspends Sonic policy and motion progression while MuJoCo
  continues under the last joint targets with exact binary32 0.1 retention of
  the live proportional gain, damping gain, and effort limit;
* `ResetBothToSpawn` gives 2.0 s of fall-detection grace, applies the root spawn
  poses immediately, and completes joint and controller reset one 2 ms fixed
  update later;
* local `ResetAfterFall` instead gives 0.5 s of fall-detection grace.

Runtime clip injection has not been excluded, so current REK `CanGetUp` remains
unknown. This candidate provisionally selects `CanGetUp=false` and its 3.0 s
referee branch. That selection is not a current-build parity fact. Staged
actuator gains are inputs to the 0.1 retention rule. They are not represented
as measurements of the current service's live gain table.

## Asset and authority boundary

`g1_semantic_assets.c` accepts only the generated semantic bundle whose model,
clip, idle-reference, size, and SHA-256 identities match its compiled manifest.
The ONNX encoder and decoder are supplied at process start and must pass the
compiled identity gate used for this candidate family. Proprietary game
binaries, extracted NPZ archives, and ONNX payloads are not linked into or
committed with the Puffer extension.

The exact encoder and decoder weights used by the current REK service are not
known. The executable candidate therefore remains classified as the validated
public GEAR-SONIC family unless a separately captured, hash-pinned current-build
identity proves otherwise. Shape compatibility alone is not identity.

## Evidence boundary

The Windows recorder is the authority for actual REK behavior. Spark is the
target for the stripped-down native environment and Puffer training. The
current held-input replay is diagnostic because its source fight lacks a
matched initial physical state, the opponent action timeline, action
acknowledgements, and a repeated-run variance envelope. It cannot accept or
reject parity.

Useful entry points are:

* `binding.c` for the Puffer extension and required process inputs;
* `semantic_duel_runtime.c` for the 50 Hz semantic and 500 Hz combat loop;
* `gear_sonic_native_duel.c` for batched policy and shared MuJoCo execution;
* `native_puffer_extension_smoke.py` for deterministic ABI and runtime checks;
* `held_trace_candidate_replay.py` for the measured Windows trace replay;
* `POLICY_ADAPTER_DESIGN.md` and `NATIVE_PUFFER_BINDING.md` for detailed
  contracts and remaining gates.
