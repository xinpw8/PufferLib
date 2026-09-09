# G1 kick timing comparison

`g1_kick_timing_compare.py` reduces one or more
`rek.g1_schedule_pose_response.v1` reports into the machine-readable
`rek.g1_kick_timing_comparison.v1` contract.

The input report paths are labels, so partial and complete captures can be kept
separate:

```text
python g1_kick_timing_compare.py \
  --run partial=partial-pose-response.json \
  --run repeat-1=repeat-1-pose-response.json \
  --run repeat-2=repeat-2-pose-response.json \
  --coverage repeat-1=repeat-1-held-motion-coverage.json \
  --out g1-kick-timing-comparison.v1.json
```

The output path must not already exist. The analyzer verifies every embedded
transcript and raw-recorder SHA-256 before using it. `--transcript` and `--raw`
accept matching `LABEL=PATH` overrides when the verified source has moved.

## Measurement boundaries

Each probe reports these independent observations:

1. `local_arm_to_send_prefix` measures fixed substeps from the bridge's local
   `ExecuteMoveByIndex` edge to its intercepted `SendMoveEvent` prefix. This is
   local visual-client dispatch timing.
2. `outbound_request_projection` verifies the same request in the independent
   recorder hook and reports signed QPC observer skew. This is a request
   projection, not server delivery or acceptance.
3. `received_pose_candidate` reconstructs the conservative consecutive-packet
   change points from `g1_schedule_pose_response.py`, then joins selected raw
   bone packet sequences to their `Time.realtimeSinceStartupAsDouble` receipt
   timestamps. Its onset and duration intervals exist only in the local-client
   receipt domain.
4. `canonical_input_to_physical_response_latency`, `canonical_move_duration`,
   and `server_action_acceptance` remain `unknown`.

For a bounded received-pose departure, the lower duration bound is the elapsed
receipt time from the first departed packet to the last confirmed departed
packet. The upper bound spans the preceding non-departed packet to the first
returned packet. A missing return produces a right-censored lower bound.

`yaw_preempted` probes are identified as
`neutral_after_yaw_preemption` only after the transcript proves their effective
controller vector was `[0, 0, 0]` at the request edge. They remain distinct
from translation-held probes and are not treated as settled, contact-free idle
trials.

## Trace and NPZ limits

An optional held-trace coverage artifact is verified together with its trace.
Its bone `source_age` means elapsed time from local packet receipt to the 50 Hz
resampling grid. It is not network age, server age, or response latency.

Requested NPZ hashes and recovered controller-tick counts are retained only as
asset metadata. The analyzer does not align NPZ rows or convert frame count to
duration because the captures do not establish executed asset identity, server
acceptance, a calibrated NPZ-joint to received-bone transform, or an
opponent/contact-free response. Repeat interval hulls and intersections are
descriptive diagnostics, not timing estimators or parity evidence.
