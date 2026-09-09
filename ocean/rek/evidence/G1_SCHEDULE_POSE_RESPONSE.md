# G1 schedule pose-response correlation

`g1_schedule_pose_response.py` joins one completed
`RekUiPipeClient g1-held` transcript to the completed
`rek.private_ai.protocol.v7` capture that overlapped it.

```text
python g1_schedule_pose_response.py \
  --transcript g1-held-run.jsonl \
  --raw rek-private-ai-root-motion-....jsonl \
  --out g1-held-pose-response.json
```

The output path must not already exist. The analyzer validates the sealed v2
schedule identity, monotonically maps every 50 Hz schedule tick onto the
recorder's 500 Hz root stream using their shared Unity fixed time, and rejects
clock drift. It also requires the named-pipe server PID to equal the recorder's
REK PID. Each held W/S/A/D/Q/E condition receives a measured root path,
displacement, yaw change, outbound input-request coverage, and a schedule-grid
trajectory. No keyboard events or controller axes are reconstructed.

Translation kick probes also retain the runtime's explicit release event,
base-velocity sample, transition-settle threshold, first settled substep, and
classification of the move-send boundary relative to release and settle. These
are visual-only client diagnostics. They do not establish the server's movement
or attack gate.

For every kick edge, a sent request is reported only when the schedule's
`SendMoveEvent` prefix anchor matches a recorder `REK_Move` projection with the
same requested index on the shared QPC clock. This proves two observations of
the local request method. It does not prove delivery, acknowledgement,
acceptance, or execution.

The pose detector compares each received post-anchor local-slot bone pose to
the nearest pose in that probe's 0.8 s pre-anchor envelope. Quaternion
geodesic RMS must exceed the measured envelope plus a robust margin, one joint
must exceed the configured angular floor, and two consecutive received packets
must qualify. A single packet spike does not satisfy the detector. Onset and return are
reported only as brackets in the client receipt domain. `REK_Bones` is
unreliable and has no source timestamp, server tick, move identity, or
acknowledgement, so executed move identity, causal attribution, server
acceptance, and canonical input delay or duration remain `unknown`.

The same detector can observe departures in windows with no `REK_Move` because
locomotion transitions, Bot motion, contact, and yaw preemption remain present.
An observed departure is therefore an uncalibrated candidate pose signal only. It is not
causal action evidence.

An empirical smoke check used completed recorder-v7 capture SHA-256
`b371f073d9e97ad4d2dc13b704cf2da5066c00a1fe5e79f443ac67957652253a`.
The detector reported departures at 1.224 to 1.284 s and 0.522 to 0.556 s
after recorder-tick anchors 2000 and 3500, respectively. Both following 4 s
windows contained no `REK_Move`. The
window anchored at the move-6 request at recorder tick 6139 contained a
departure bracket of 0.132 to 0.166 s. These are client receipt-domain
observations, not measured action delays. The no-request results are retained
as evidence that latency and pose departure alone cannot identify or certify a
kick.

The detector is deliberately a provisional conservative departure detector,
not a trained action classifier. Repeated clean captures are required to learn
per-condition idle and locomotion envelopes and to establish REK's own
repeated-run variance before pose trajectories can be used for parity
acceptance.
