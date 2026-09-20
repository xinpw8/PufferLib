# Left-front kick candidate mechanics, 2026-09-20

The existing native MuJoCo CUDA + SONIC candidate executed native move 7,
`LEFT_FRONT`, without a fall in three declared neutral/signed-yaw contexts.
This establishes limited candidate executability. It does not establish actual
REK kick value, reliable scoring, opponent trips, or the reason a learned policy
rarely selects it. No policy, reward, model, or physics implementation changed.

## Exact comparison

This is a separate source and run from the [move-3/move-10 probe](physical-schedule-probe.md).
The recovered action table maps move 7 to category 17 and route 8; move 3,
`RIGHT_HOOK`, maps to category 23 and route 14. The configured semantic durations
are respectively 145 ticks (2.90 s) and 45 ticks (0.90 s). Clip frame counts and
semantic decision durations are distinct; the schedule uses the pinned semantic
durations.

Four arenas retain the fixed eight-row SONIC controller. Each of three fresh
candidate resets uses desired yaw -1, 0, or +1. Actor lanes are kick/control and
hook/control. Opponents remain neutral. After 50 neutral warmup ticks and 25
held-yaw lead ticks, one attack is requested at tick 76. Category 0 retains the
prior yaw until the segment ends; category 1 releases at tick 221 for the kick
and tick 121 for the hook. Controls retain the same yaw until the matching
release. Each reset runs through tick 270 at 50 Hz, ten 0.002 s physical
substeps per decision. Every action must pass the native mask; there is no
substitution or retry.

## Measured result

One approved execution passed, exit 0, from 03:39:39.091844470 to
03:39:52.806725330 UTC, 13.715 s. GPU execution ended before subsequent work.
All 3,252 per-arena samples parsed. Actual selected routes were 8 for every one
of the kick's 145 busy samples and 14 for all 45 hook busy samples. Effective
yaw was zero during both attacks, although desired yaw remained held. Control
routes/effective yaw were 6/-1, 0/0, and 5/+1.

| Desired yaw | Kick net XY / control, m | Kick path XY, m | Kick maximum tilt, degrees | Hook net XY / control, m |
| --- | --- | --- | --- | --- |
| -1 | 0.236221 / 0.035532 | 1.841978 | 37.1287 | 0.032920 / 0.060798 |
| 0 | 0.219761 / 0.040345 | 1.840888 | 36.1730 | 0.054365 / 0.039974 |
| +1 | 0.305859 / 0.141863 | 1.914552 | 37.5644 | 0.074662 / 0.032406 |

These windows start at pre-action tick 75 and end at tick 220 for the kick or
120 for the hook. Kick minimum root height was 0.6570 to 0.6594 m; hook maximum
tilt was 13.83 to 16.08 degrees. Across both fighters and all twelve lanes,
there were no non-foot floor-contact samples, final falls, terminal/reset
events, or runtime failure bits. Foot contacts were observed; every final
point count was 0:0. The log does not expose all inter-fighter contact pairs,
so zero points is not evidence of zero physical contact.

Paired pre-action states were not identical, despite sharing the declared
reset/action history:

| Yaw | Move | Maximum qpos component difference | Maximum qvel component difference |
| --- | --- | --- | --- |
| -1 | 7 | 0.002588242 | 0.067186939 |
| -1 | 3 | 0.029908508 | 0.249321699 |
| 0 | 7 | 0.006612398 | 0.047980964 |
| 0 | 3 | 0.004132211 | 0.095797151 |
| +1 | 7 | 0.003603756 | 0.148143917 |
| +1 | 3 | 0.005743235 | 0.368266746 |

These maxima combine different model coordinate types, not uniform SI units.
They limit exact causal interpretation of attack/control differences. The kick
completed its longer commitment without falling here, with substantial root
path movement and higher tilt than the hook. Three from-reset contexts against
an idle opponent provide no reliability rate under combat disturbances and no
evidence of successful opponent trips. Authentic full velocity/controller state
and server execution timing were not restored.

## Sources, tests, and reproducibility

Sources: [physical_left_front_probe.cpp](../../validation-quality/physical_left_front_probe.cpp),
[build_physical_left_front_probe.sh](../../validation-quality/build_physical_left_front_probe.sh),
[run_physical_left_front_probe.sh](../../validation-quality/run_physical_left_front_probe.sh).
The C++ bytes match the executed source. The public shell filenames are expanded;
the private preparation used `build.sh` and `run.sh`. The harness links the same
preserved native runtime objects/assets as the original physical probe and
retains CPU-physics abort wrappers. Build logs include object/source hashes;
run logs include all six asset inputs, manifest/controller/kernel hashes,
commands, UTC times, stdout/stderr, process timing and exit status.

CPU-only schedule tests passed 3,892 checks. A separate JSON parse verified all
810 decision ticks, move identities, and request/release boundaries. Both shell
scripts passed `bash -n`; native compilation/linking completed with empty
compiler stderr. Reproduce from staged sources, using fresh output directories:

```sh
bash SOURCE/build_physical_left_front_probe.sh NEW_BUILD
# Run only after confirming the GPU resource window is available.
bash NEW_BUILD/run_physical_left_front_probe.sh NEW_BUILD NEW_RUN
```

The run is bounded by a 120 s process timeout. Existing directories are refused.
Original run evidence is preserved at
`/home/spark-advantage/rek-training/physical-left-front-probe-20260920-r1/run-r1`;
the build is the sibling `build-r1`. Verified local evidence is
`C:\rekagent\work\consistent-fighter-20260919-r1\physical-left-front-probe-r1\run-r1`.
All eight manifest file hashes matched after copying.

```text
executable SHA256 6a482b6232db907c2241e298c613471dd54112086845279ca2b7b1884c21c589
C++ SHA256        5a48dd5a5acfe3046a7a27cea38f426a646ce1ac805df2a886c26496b98f2033
stdout SHA256     ef122a716cafb26d1ae1c905892556c2977a43c506516a6c3fb8a3a35745b451
stderr SHA256     cb328adb3c8f5c71147f95c552c7b32b7466b3f32363ab44465d2c81b153cfc9
```
