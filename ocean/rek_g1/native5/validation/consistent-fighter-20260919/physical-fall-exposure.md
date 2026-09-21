# Physical fall exposure and normalized reward, 2026-09-21

The corrected MuJoCo CUDA + SONIC runtime produced confirmed falls for both
fighters, a native double countout, and bilateral spawn reset during a bounded
contact schedule. Every checked normalized reward reached the learner buffer.
This establishes action-driven physical fall exposure. It is not a trained
policy result or evidence of authentic client physics parity.

The run used the native spawn and categorical actions only. No pose was forced,
external impulse applied, state restored, or CPU physics executed. Four arenas
ran 2,500 decisions at 50 Hz, ten 0.002 s physical substeps per decision, with a
60 s round limit. The schedule follows the earlier physical-quality probe:
approach between 0.65 and 1.05 m, yaw tolerance 0.16 rad, legal-action masking,
and native move cycling. Requested, selected, and recorded actions are logged.

| Schedule | Attack requests, self/opponent | Confirmed falls | Final points | Attributed/scored contacts |
| --- | --- | --- | --- | --- |
| Neutral both | 0/0 | 0/0 | 0:0 | 0/0 |
| Stationary attack cycle versus neutral | 29/0 | 0/0 | 0:0 | 0/0 |
| Approach and repeated move 9/right knee versus neutral | 13/0 | 0/0 | 0:0 | 2/0 |
| Approach and attack cycles, both fighters | 17/16 | 1/1 | 5:6 | 3/1 |

The repeated-knee category is 19, correctly mapped to native move 9. The cycle
uses native registry order `6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16`.

## Observed physical and reward transitions

| Decision tick | Both-attacking arena event | Reward, self/opponent |
| --- | --- | --- |
| 147/148 | Opponent/self enter falling and subsequently recover | 0/0 |
| 853 | Attributed contact, measured struck speed 4.362681 m/s | 0/0 |
| 861 | Self begins falling | 0/0 |
| 869 | Opponent begins falling | 0/0 |
| 891 | Self becomes fallen; native slip call; 3 s count starts | -0.01/0 |
| 897 | Opponent becomes fallen; double-down call; count restarts | 0/-0.01 |
| 1047 | Double countout, native award +5/+5, bilateral spawn reset | 0/0 |
| 1693 | Ordinary scored contact awards opponent one point | -0.01/+0.01 |

The native referee classified these falls as slips. Contact and instability
were present, but this run does not establish that a scored opponent strike
caused either fall. Minimum root heights were 0.130062/0.132685 m; maximum tilts
were 106.556/102.905 degrees. There were 178/174 samples with non-foot floor
contact. The neutral control had zero non-foot samples and maximum tilt
8.46984 degrees. The 50 s observation window ended before any round terminal.

The reward mode was `normalized_points_falls_v1`: awarded score difference
minus one point for the own fighter's `BECAME_FALLEN` event, divided by 100,
with safety bounds [-1,1]. All 20,000 post-step fighter reward comparisons and
10,000 learner-buffer equality checks passed. Holding a fallen state did not
repeat the penalty. The simultaneous +5/+5 referee award correctly canceled
in point difference. No terminal bonus is involved.

Exit status was 0 after 41.16 s wall time. All runtime failure bits were zero.
Reported reward saturation count was zero. Abort wrappers covered `mj_step`,
`mj_step1`, `mj_step2`, `mj_forward`, and `mj_kinematics`. Startup confirmed
`mujoco_cuda`, zero CPU physics, zero Python runtime, eight SONIC rows,
2,048 pooled contacts and 512 constraint rows per arena. This probe performed
zero PPO updates.

## Reproduction and retained evidence

Sources: [probe](../../validation-quality/physical_fall_exposure_probe.cpp),
[build](../../validation-quality/build_physical_fall_exposure_probe.sh), and
[runner](../../validation-quality/run_physical_fall_exposure_probe.sh).
The build copies the preserved corrected physical source, adds the current
normalized reward runtime/helper, recompiles `runtime.o`, and links the corrected
`measurement.o` plus the preserved native physics/controller objects. It never
links a trainer object. Build and run directories must be new.

Remote stage: `/home/spark-advantage/rek-training/physical-fall-exposure-20260921-r1`.
Executed build: `build-r2`; run: `run-r1`. The initial `build-r1` was compiled
against the previous reward runtime and was never executed. The complete run
records qpos/qvel, decisions, route/busy state, contact attribution, fall events,
referee state, native points, and rewards at every decision.

Private local evidence:
`C:\rekagent\work\consistent-fighter-20260919-r1\physical-fall-exposure-r1`.
All eight run artifact hashes matched after retrieval.

```text
executable 149f0ac2c37661e05c72c443d2dfed3d20f53505a0f5c4cd6773ae970ea7849e
probe C++  7ba8ce229f2c44d112fd51b8c2cbfb9e6f4a7c5812c7b7fdba1a6776983cf477
runtime    022c0ebdc684e5fb45a03c6fb52a1f12f72942dc11c0d971027ea9e42f2b134f
reward     14cc963f8b07230da3c0390d0c0ff9e6d671bba3d762ee157056211d95745981
stdout     cd02cb2c05a580874eb121c2e07c03654c7d2b698b8d4c3f62a919e72f1eba89
```

The existing physical training binding passes PufferLib's device reward buffer
into `RekNative5Buffers`, and noncompact `puf_step` invokes `rek_native5_step`.
The probe validates that runtime-to-learner-buffer boundary. The subsequent
[physical PPO run](physical-normalized-training.md) also recorded confirmed
falls during learning rollouts; its performance and deployment limits are
reported separately.
