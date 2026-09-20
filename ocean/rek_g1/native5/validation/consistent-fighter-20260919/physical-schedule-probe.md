# Native physical schedule comparison prepared 2026-09-20

The native MuJoCo CUDA + SONIC probe is compiled and CPU-checked. GPU execution
is pending the live-evaluation resource window. This is a candidate behavior
check from the candidate's own reset. It does not replay authentic state and
does not establish physical parity or fighting strength.

## Why this comparison

The [busy-root measurement](busy-root-motion.md) found substantial observed
planar movement during move-3/move-10 request windows outside compact root-body
overlap. `fast_runtime.cu` suppresses logical planar integration while attacking.
`fast_assets.cpp` additionally requires the largest source clip root XY span to
be below `1e-4` and records `max_source_root_xy_span_m`. `FastFrame` deliberately
omits planar displacement. Restoring a dropped nonzero clip translation therefore
cannot supply the measured movement from this existing asset bundle.

The prior physical probe did demonstrate articulated root movement, but its
action-19 lane was mislabeled move 3. It actually selected move 9/right knee.
The [historical report](../quality-20260916/PHYSICAL_QUALITY_PROBE.md) now corrects
that interpretation and identifies the original request ticks. Executed source,
raw labels, executable and evidence remain unchanged.

## Bounded schedule

Four arenas retain the preserved eight-row SONIC controller. Three independent
native resets test desired yaw `-1`, `0`, and `+1`. Each condition lasts 259
decision ticks at 50 Hz, with ten 0.002 s physical substeps per decision.

| Lane | Actor intervention | Paired control |
| --- | --- | --- |
| 0 | Native move 3, category 23 | Lane 1: no attack |
| 2 | Native move 10, category 26 | Lane 3: no attack |

Every opponent receives category 1/neutral throughout. All actors receive 50
neutral warmup ticks, followed by 25 ticks of the condition's held yaw. At tick
76 each attack lane requests its move once. Category 0 then retains the prior
desired yaw through the remaining nominal segment. The controls keep sending
the same yaw category. Neutral category 1 releases at tick 121 for move 3 and
tick 210 for move 10. The run continues through tick 259.

The actual registry order is `6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16`.
Move 3 has 45 decision ticks (0.90 s); move 10 has 134 (2.68 s). These are the
existing native semantic durations, with the initial request consuming one tick.
Every supplied action must pass the current native mask. A masked action aborts
the fixed experiment; no replacement, retry or mask bypass occurs.

The probe logs per-tick roots/quaternions, model-order root qvel, physical and
logical headings, effective yaw, actual route and busy state, root separation,
foot/non-foot floor contacts, tilt, fall phase, points, falls, failure bits,
terminal state, referee and reset signals. Summaries measure net/path XY and
unwrapped physical yaw from pre-action tick 75 through ticks 120/209. These
are nominal duration windows; recorded busy state remains available separately.
Unexpected terminal/reset events are explicitly flagged. Paired qpos/qvel
differences at tick 75 are reported. Any nonzero difference limits claims about
exactly identical initial states; material differences prevent causal paired
interpretation. Contact or fall differences must likewise remain visible.

## Reproduction and current evidence

Sources are [physical_schedule_probe.cpp](../../validation-quality/physical_schedule_probe.cpp),
[build_physical_schedule_probe.sh](../../validation-quality/build_physical_schedule_probe.sh),
and [run_physical_schedule_probe.sh](../../validation-quality/run_physical_schedule_probe.sh).
The build links the preserved feature-enabled native runtime objects, excluding
`pufferl.o`, against their preserved ABI. It does not rebuild physics or policies.
CPU-call wrappers abort `mj_step`, `mj_forward`, and `mj_kinematics`.

After staging those three files together on Spark, build into a fresh directory:

```sh
bash SOURCE/build_physical_schedule_probe.sh NEW_BUILD
```

Building runs only a separately compiled CPU schedule executable: 3,760 checks
passed, and 777 scheduled tick records plus their identity were emitted.
CUDA compilation/linking succeeded with empty compiler stderr. No GPU probe was
executed during preparation. After explicit GPU availability, execute once into
a fresh directory:

```sh
bash NEW_BUILD/run_physical_schedule_probe.sh NEW_BUILD NEW_RUN
```

The runner saves its full six-input command, UTC times, stdout/stderr, exit code,
process timing and hashes. Asset manifests, XML/export, both ONNX controllers,
kernel catalog/PTX and all linked object hashes are recorded by the build/run
pair. Both scripts refuse an existing output directory. Timeout is 120 s. Based
on the earlier 1,000-tick diagnostic, approximately 10 to 20 s is a planning
estimate for this 777-tick schedule, not a measured runtime.

The compiled private preparation currently uses the earlier short harness names:

```sh
bash /home/spark-advantage/rek-training/physical-schedule-probe-20260920-r1/build-r2/run.sh /home/spark-advantage/rek-training/physical-schedule-probe-20260920-r1/build-r2 /home/spark-advantage/rek-training/physical-schedule-probe-20260920-r1/run-r1
```

```text
executable SHA256 adf666e5b68f837e984cca65ceecf4b396d99eb42f923bdfec1ac72b69ffedc8
C++ SHA256        7cd67c58b9958488ecc5b6c7142f5194646d23bc16b5bb7e834894b38b1a5a9f
```

`build-r1` is preserved. `build-r2` adds explicit terminal/reset reporting and
manifest hashes. Its C++ bytes equal the repository source. The local private
preparation is `C:\rekagent\work\consistent-fighter-20260919-r1\physical-schedule-probe-r1`.

## Minimum missing inputs for authentic state replay

The preserved runtime ABI supports native reset, action input and read-only
qpos/qvel snapshots. It has no complete-state restoration API. Its reset sets
qvel to zero and initializes idle motion/controller state. The existing probe's
CLI accepts six asset paths, without an initial-state or action-history input.

The selected authentic captures contain visual poses, local bone rotations,
request QPC timestamps, and owned desired-action history. Their visual-only
root velocities are zero and joint velocities unavailable. Encoder derivatives
are finite differences of replicated poses; they are not authoritative full
MuJoCo qvel. Native clip phase, SONIC/controller history, server acceptance and
execution timing, and opponent action history are also unavailable. Those
values cannot be inferred exactly from a pose snapshot. The preparation does
not fill them with zero or use the visual finite differences as true simulator
initial velocity. Exact state/history replay would require those inputs and a
validated restoration boundary, which is outside this bounded comparison.
