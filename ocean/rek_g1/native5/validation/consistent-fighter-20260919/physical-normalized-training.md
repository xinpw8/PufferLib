# Physical PPO fall exposure, 2026-09-21

Native MuJoCo CUDA + SONIC PPO completed 4,194,304 learner transitions and
16 updates with normalized point/fall rewards. The runtime recorded 208 learner
falls and 1,222 opponent falls. Exit status, runtime failure bits, invalid-action
metrics, and reward saturation count were all zero.

This is a physical fine-tuning baseline. The compact warm-start checkpoint has
the same 223-input tensor shape but different physical observation semantics.
It is not a schema-compatible initialization, and this checkpoint has not been
validated for live deployment. The current live encoder does not reproduce all
physical support, tilt, fall, and referee fields used here.

## Executed configuration

The run used 512 arenas, horizon 512, 120 s rounds, minibatch 8,192, 50 Hz
decisions and ten 0.002 s physics substeps per decision. It retained the pinned
SONIC encoder/decoder with 1,024 controller rows. The network was hidden size
256 with two layers. Both seeds were 419. Learning rate started at 0.0001 with
the existing annealing setting; entropy coefficient was 0.01, gamma
0.9998844821426083, and GAE lambda 0.9978673240629938.

PufferLib source pin `773f923d80e73bdc255a2ba730c918b28e416aa1` was freshly
compiled with the current action-mask, initial-model, and temporal-credit
patches. `compact_training_autoreset=0`; the physical binding invokes
`rek_native5_step`. Corrected contact measurements and the normalized physical
runtime were linked to the preserved CUDA physics/controller objects. Abort
wrappers cover CPU `mj_step`, `mj_step1`, `mj_step2`, `mj_forward`, and
`mj_kinematics`. Startup reported zero CPU physics and zero Python runtime.

The opponent is `CandidateApproachDummy` in
[runtime.cu](../../runtime.cu), function `choose_actions`. It approaches above
1.25 m, retreats below 0.72 m, turns outside a 0.16 rad bearing tolerance,
cycles attack categories 16 through 31, and selects neutral while falling.
It obeys the legal-action mask. This is the physical runtime's scripted
opponent, not authentic Bot1 or the compact recovered-Bot1 implementation.

## Measured results

| Measure | Result |
| --- | --- |
| Learner transitions / PPO updates | 4,194,304 / 16 |
| Simulated time per arena | 163.84 s |
| Training-loop time / throughput | 632.414002 s / 6,632.212 transitions/s |
| Process wall time / throughput | 642.65 s / 6,526.576 transitions/s |
| Completed rounds | 512: 374 wins, 105 losses, 33 ties |
| Completed-round points, learner/opponent | 5,267 / 3,272 |
| Confirmed falls, learner/opponent | 208 / 1,222 |
| Awarded points across all runtime transitions | 6,715 / 4,375 |
| Reward saturations / runtime failure bits | 0 / 0 |

The 512 completed rounds represent one full cohort. Training outcomes include
the changing policy and are not held-out strength measurements. The cumulative
fall/point counters explicitly cover `all_executed_runtime_transitions`, which
includes unfinished subsequent rounds and any startup transitions. They must
not be substituted for completed-round totals. No final per-arena pose snapshot
was taken during this PPO run.

At update 12, 509 rounds completed: logged mean own falls 0.314, score 10.277,
episode return 0.036, wins 0.729, losses 0.206, and draws 0.065. At update 13,
the remaining three rounds completed: mean falls 0, score 12, return 0.08, and
wins 1. These printed values are rounded. Both windows reported zero invalid
actions and finite failures. All 16 update panels, including policy/value loss,
entropy, KL, and clipping metrics, are retained in stdout. The native INI's
metrics are downsampled and include a repeated final bin; its bins are not
additional optimizer updates or independent rounds.

Reward is native awarded-point difference minus one point on the own
`BECAME_FALLEN` event, divided by 100, with safety bounds [-1,1]. No additional
opponent-fall or terminal reward is added. The preceding
[physical contact probe](physical-fall-exposure.md) separately verified 20,000
fighter rewards and 10,000 learner-buffer values, including both confirmed
falls, an ordinary score, and a bilateral countout/reset.

## Checkpoints, reproduction, and receipts

Initial checkpoint SHA256:
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`.
The step-zero readback matched byte-for-byte. The 16 subsequent checkpoint
files all have distinct hashes; all 17 copied files were verified.

Final checkpoint, at step 4,194,304:
`a71f017f80d8e85a3dae520ba9c8e91410a6add059e5a29ddd9d83e6bcdd9f4a`.

The exact executed harnesses are preserved without generalization:
[build](../../validation-quality/build_physical_training.sh),
[run](../../validation-quality/run_physical_training.sh), and
[CPU guards](../../validation-quality/cpu_physics_guard.cpp).
Their paths identify the executed private stage and preserved source/object
inputs. These are exact experiment receipts with pinned output paths. A new
execution requires new stage/output paths so the recorded evidence is preserved.

Remote stage:
`/home/spark-advantage/rek-training/physical-fall-exposure-20260921-r1`.
Build: `build-training-r1`; run: `train-physical-normalized-r1`.
Start/end UTC: `2026-09-21T08:11:39.020627892Z` /
`2026-09-21T08:22:21.678500408Z`.
The private stage retains all commands, source snapshots, object hashes,
guarded executable, checkpoints, controller/model hashes, process timing,
stdout/stderr, and the native training INI.

```text
guarded executable cdbaddc2f30fe542f0bed031389de1f52529a96f921e76a5f70eb460b5717b87
physical runtime   b57235df70c3d10e0677d1617ba479c8d795cef27cdea8c8d98513760ed18dc6
training INI       ccb8af174ef49087d963264fc4dfbb8996b4fba51e00024572b9f21ddff8e30e
evidence archive   7afd5e3bf83505d3f9d97fb4969845f001443178bc5d2f371257b8749ede6603
```

The 75,761,181-byte archive was copied to the verified existing REK evidence
project on the physical file server, into a new directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-fall-exposure-r1\physical-fall-exposure-20260921-r1-evidence.tar.gz`.
Its SHA256 matches Spark and the local private copy. Existing server files were
preserved. Local private stage:
`C:\rekagent\work\consistent-fighter-20260919-r1\physical-fall-exposure-r1`.
