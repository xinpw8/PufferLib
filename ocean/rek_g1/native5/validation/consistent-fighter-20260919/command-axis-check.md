# Recorded command and motion signs

A file-only check of the two existing human rounds found no identifiable
forward-axis or yaw-sign mismatch. It does not establish complete movement
parity. The observations do not justify changing heading by 90 or 180 degrees.

Commands and independently captured root poses were aligned by source order
and the shared 500 Hz fixed-tick counter. Each forward interval below contains
only exact `[forward=1, strafe=0, yaw=0]` commands and no move requests.
Positions and gaps are captured coordinate units; no physical-unit calibration
is added here.

| Human interval | Commands | Local forward / lateral displacement | Minimum opponent gap |
| --- | ---: | --- | ---: |
| R2, 0.30 to 1.20 s | 54 | +0.456 / -0.068 | 1.56 |
| R1, 8.50 to 9.10 s | 37 | +0.315 / -0.101 | 1.64 |
| R2, 14.40 to 15.40 s | 60 | +0.240 / -0.041 | 1.67 |

Rendered pelvis headings were approximately 0, -70 and 1 degrees. Positive
forward displacement therefore holds at substantially different world
headings. It also holds after trimming 0.1 or 0.2 s from both ends. Prior
move requests were absent or at least 4.17 s earlier. Native busy/acceptance/
execution receipts are unavailable for these historical human intervals,
so this does not prove a settled authoritative server state.

Two pure-yaw intervals agree with the compact training sign:

- R2, 41.00 to 41.60 s: 37 positive-yaw commands, +44.51 degree heading change.
- R2, 43.36 to 43.66 s: 19 negative-yaw commands, -10.33 degree heading change.

Only one separated pure-strafe interval qualified, lasting 0.24 s. Positive
strafe accompanied +0.039 lateral and -0.014 forward displacement. Its short
duration and preceding motion do not establish strafe parity. No qualifying
backward interval was found.

Forward residual angles of -8.5 to -17.8 degrees must not be reinterpreted
as a serialized orientation offset. Pelvis animation, physical drift and
unknown command-to-observation latency remain confounded. The compact
observation also composes logical heading with clip-root rotation in
[fast_assets.cpp](../../fast_assets.cpp). Logical and rendered pelvis heading
need not be identical at every animation frame.

## Evidence

Private dataset: `C:/rekagent/work/imitation-20260919-r1/dataset-r1/manifest.json`.
Dataset SHA256:
`71238da150db6e926170efdc27662f4426145aac7f69a7c93f619ece54d6cf1d`.

Source-line intervals: forward R2 77-301, R1 2128-2278, R2 3609-3858;
yaw R2 10278-10428 and 10867-10944. Original recordings and ledgers remain
unchanged. No GPU, new gameplay, or runtime modification was used for this
check. Current forward/yaw signs are retained.
