# Native live policy worker

`live_policy_worker.cu` performs frozen sampled BF16 inference on CUDA. It does
not instantiate an environment, run physics, encode observations, or send game
input. JSON parsing, checkpoint I/O, and protocol validation run on the host.
All neural computation and recurrent state run on the GPU.

Build with `bash build_live_policy_worker.sh EXACT_NATIVE_POLICY_BUILD NEW_BUILD`.
The build first compiles and tests the same protocol parser without CUDA. This
test executable can only validate requests and never emits actions. The native
executable links the exact existing `native_policy.o` and `cJSON.o`, CUDA BLAS,
CUDA random, and OpenSSL. Build commands and hashes are saved in `NEW_BUILD`.

Run `NEW_BUILD/live-policy-worker CHECKPOINT CHECKPOINT_SHA256 73`. Checkpoint
hash is mandatory. The pinned network is 223 observations, 33 actions, 256 hidden
units, two recurrent layers, ABI 2. Startup emits a JSON `ready` response with
identity, dimensions, precision, sampling mode, and device.

## JSONL protocol

Each request is one JSON object followed by a newline, at most 65,536 bytes.
Only the listed fields are accepted. Duplicate fields are rejected.

```
{"type":"step","seq":1,"round_id":"<64 lowercase hex characters>",
 "observation_schema":"rek.native5.scaled_polar_xy.v1",
 "observation":[223 finite encoded floats],"mask":[33 booleans or 0/1],
 "terminal":false}
```

The array placeholders above are schema notation, not a runnable request.
Missing, nonfinite, wrong-sized, or out-of-float32-range observations fail.
The adapter must produce all fields from its explicit mapping; this worker
does not replace unknown values. A nonterminal request needs a nonempty mask.
An action response echoes `seq`, `round_id`, checkpoint hash, and schema. It
contains integer `action` in `[0,32]`, `selection:"sampled"`, `precision:"bf16"`,
`recurrent_reset`, `round_changed`, `decision_index`, `latency_ms`, and `gpu_ms`.

`seq` is a globally increasing integer for this worker process, bounded by
JavaScript's exact integer range. Failed requests do not consume a sequence.
The relay should restart the worker when the source telemetry stream restarts
its sequence. Round IDs are the authentic Unity `TrialRoundIdentity` SHA256,
never an inferred candidate episode counter.

Changing measured round ID clears recurrent state before inference. An explicit
`{"type":"reset","seq":2,"round_id":"<hash>"}` clears recurrent state without
emitting an action. `terminal:true` validates the full observation and mask,
clears recurrent state, and returns a terminal acknowledgment without inference
or action. It permits an all-zero mask. All resets preserve sampler progression.
The following action reports that recurrence was reset. `{"type":"close",
"seq":3}` acknowledges close and exits. EOF also exits.

Invalid requests return an error code and `action_available:false`, with no
action. CUDA or native inference failure terminates the worker with exit code 2.
An invalid result is never replaced with a neutral action.

## Timing and live safety boundary

`latency_ms` covers accepted-request recurrent reset, upload, inference, result
download and status verification. It excludes time queued on stdin and JSON
parsing. `gpu_ms` is a CUDA event interval around upload/inference/download;
it excludes the preceding recurrent reset. Neither is an end-to-end game delay.

The relay must independently enforce the current round hash, original Unity
observation sequence, the 250 ms Unity-QPC freshness limit, private-versus-AI
state, and release of its own held inputs when stale or rejected. Worker output
is never authority to control a public match. No Windows input is part of this
process. A measured live-to-223 adapter remains a separate prerequisite.

## Tests

`node live_policy_worker.test.cjs protocol NEW_BUILD/live-policy-protocol-test`
tests request validation and state transitions without GPU inference.

`node live_policy_worker.test.cjs gpu NEW_BUILD/live-policy-worker CHECKPOINT SHA`
runs a bounded GPU protocol test with synthetic zero observations and one-hot
masks. It proves checkpoint loading, legal-action sampling, reset protocol, and
CUDA execution. It does not prove authentic-game observation equivalence,
policy quality, or sim-to-sim parity. No proprietary observations or checkpoint
weights are written by the tests.
