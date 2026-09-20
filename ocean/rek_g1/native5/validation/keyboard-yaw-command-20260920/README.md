# Keyboard-reset yaw command experiment

`REK_FAST_YAW_COMMAND=keyboard_reset_v1` is an opt-in compact-training command
model. Unset or `legacy_velocity_slew_v1` retains the previous dynamics;
unknown values fail. Both modes completed a matched native CUDA training run.
Authentic evaluation is separate and pending in this report.

## Command mismatch and bounded change

The previous compact calculation slews physical angular velocity toward the
desired target: `omega += clamp(yaw*yaw_speed - omega, -delta, delta)`, where
`delta = yaw_speed*0.02/yaw_ramp`. Release leaves residual rotation and reversal
can continue rotating in the previous direction.

The recovered Windows `G1HeldInputScheduleContract.AdvanceKeyboardYaw` uses
separate command state `(ramp, sign)`. Zero raw yaw clears both immediately.
A changed nonzero sign starts ramp at zero, then each update adds `dt/rampTime`,
capped at one. Output is `ramp*sign*yawSpeed`. The nonpositive-ramp branch clears
state and returns `rawYaw*yawSpeed`; zero dt is permitted. The shared native
[helper](../../keyboard_yaw.h) reproduces that finite-input expression.

The opt-in uses 0.02 s per existing environment tick and a 0.5 s command ramp.
The latter is the recovered schedule contract's expected parameter, not an
independently measured actuator time constant or a verified current live
RobotConfig measurement. `FastAssets.yaw_ramp_seconds` was a hardcoded candidate
initializer; it did not establish actuator lag. Normalized command is multiplied
directly by the unchanged candidate `yaw_speed=1.8 rad/s`. No second lag is added.

Starting from saturated positive yaw, the first 20 ms release step changes
legacy omega from 1.8 to 1.728 rad/s, versus zero in the new mode. A reversal
produces +1.728 versus -0.072 rad/s. Sustained same-sign commands retain ramp;
hold category0 retains the desired category and continues advancing the command
at 50 Hz. Release1 clears it. Busy command state is reset on every canned-attack
tick, including the final tick; retained yaw then restarts at normalized 0.04.
Existing pose/round resets clear state. Stride5 hold-only rows would still advance
the ramp at 50 Hz, but both training arms here use stride1.

Scope is non-Bot policy or fallback scripted controllers. Recovered Bot1's
continuous-command template retains the original velocity expression. No bridge,
live encoder, observation feature, policy architecture, reward, canned duration,
translation constant, root-lock rule, or training discount was changed. Internal
command ramp is controller state, inferred from recurrent history without adding
an observation column. Mode and physical-response approximation are logged, and
the build manifest includes the helper hash.

Windows resets using actual local `_moveInFlight`, `IsPunching` or `IsRecovering`.
Compact training uses its existing `attack_duration` proxy. Captured evidence
already shows those intervals can differ. The tests establish the conditional
command formula and its integration, not identical busy lifecycle, rendered
update timing, server execution, physical turning or balance. Compact planar
root motion remains locked during canned attacks.

## Verification

[CPU tests](../../test_keyboard_yaw.cpp) passed 20,077 checks: sustained hold,
release, sign reversal, repress, zero dt, saturation, direct-ramp branch, unknown
mode rejection, busy reset, hold0, cadence and randomized comparison with the
independently transcribed C# expression.

The [direct CUDA fixture](../../test_keyboard_yaw_runtime.cu) passed 651 checks
over 120 ticks. It checks 45 busy ticks, busy-held yaw followed by hold0 and a
fresh ramp after completion, release/reversal counterexamples, pose/round resets,
unused command state under legacy mode, and byte-identical Bot1 arena state.

The existing [real-asset ABI fixture](../../test_action_cadence.cu), linked against
the archived cadence runtime and this new runtime, reproduced all 2,335,662
default bytes across 666 snapshots. Observation, pose, action, masks, rewards and
round fields match, including terminal/autoreset cases. SHA256:
`5c14c01516fadd773c273822df9de7fa11b5bd12340b4114a71aa4fa7a96028e`.
The existing stride5 comparator also passed 65,934 mask checks. Under the new
keyboard mode, 120 policy-controlled snapshots change while all 222 snapshots
with both fighters controlled by recovered Bot1 remain byte-identical.

The single GPU fixture reservation was 2026-09-20 22:46:24.765199365 through
22:46:26.752222269 UTC, bounded by 60 s. All checks passed without a retry.
The compiled trainer is
`/home/spark-advantage/rek-training/keyboard-yaw-command-20260920-r1/fast-build-r1/puffer-rek-native5`,
SHA256 `c3e5e2325174eb0c49a91a5980e3632f367515ae05cc6352ed2588ae3357e450`.
The exact source snapshot is sibling `source/ocean/rek_g1/native5`.

## Matched full-training pair

Both arms independently start from f3 checkpoint
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`.
They use the same new binary, byte-identical benchmark INIs and stride1.
Settings: 33,554,432 learner transitions at 50 Hz, 512 arenas, horizon512,
minibatch8192, H256/two MinGRU layers/BF16/CUDA graphs, learning rate0.0001 with
existing annealing, entropy0.01, vf0.5, policy/value clip0.2, replay ratio1,
gamma0.9998844821426083, lambda0.9978673240629938, environment seed419 and native
base seed73. Recovered Bot1, rendered-pose observations, eight primitive-contact
substeps, round-outcome reward, 120 s rounds, random gaps0.55-2.5 m and heading
spread pi are unchanged. No feature mask, frozen opponent or extra shaping.

| Measurement | Legacy velocity slew | Keyboard reset |
| --- | ---: | ---: |
| Native exit / failure bits | 0 / 0 | 0 / 0 |
| Learner transitions | 33,554,432 | 33,554,432 |
| Trainer-loop seconds | 36.40170836 | 36.12033391 |
| Full-training transitions/s | 921,781.79 | 928,962.40 |
| Process wall seconds | 37.11 | 36.82 |
| Startup-inclusive transitions/s | 904,188.41 | 911,309.94 |
| Rollout CUDA seconds | 28.14935818 | 27.80967410 |
| Training model / misc seconds | 7.51447475 / 0.16038515 | 7.63634539 / 0.16100698 |
| Unattributed loop residual seconds | 0.57749028 | 0.51330744 |
| Changing-policy round W/L/T | 4856 / 238 / 26 | 4527 / 565 / 28 |
| Changing-policy awarded points | 379145:216258 | 352810:214656 |

Training ran from 22:46:47.108782153 through 22:48:01.292890240 UTC on
2026-09-20. Both native runs and the outer wrapper exited0. No failed attempt or
training retry occurred. The single ordered pair's +0.779% SPS difference does
not establish a speed improvement. Timer aggregation uses the existing 64-bin,
43-unique-mean calculation with first-bin weight2 and excludes repeated final
bins; graph-mode rollout inference/environment/copy subdivision is unavailable.

The legacy final checkpoint exactly matches the previous stride1 benchmark,
providing a full-training reproducibility check of unchanged default behavior.
Lower changing-policy compact wins for the new mode do not determine authentic
strength; these totals span optimization and different command dynamics, with
no held-out evaluation or promotion claim.

Under each run's `checkpoints/rek_native5/RUN/0000000033554432.bin`:

- `train-legacy_velocity_slew_v1-r1`:
  `93180341b685c99549d9f55693a56b8cf42dfc45857790d77d43920d35219ba3`.
- `train-keyboard_reset_v1-r1`:
  `85808a6731f4f40f5756800a9edf93faa5c312aa7d5f7468cafdd4ae3a5c381f`.

## Reproduction and retained artifacts

Compile the helper tests and actual pinned native trainer on Spark:

```sh
native=/path/to/repo/ocean/rek_g1/native5
g++ -std=c++17 -O3 "$native/test_keyboard_yaw.cpp" -o NEW_CPU_TEST
./NEW_CPU_TEST
bash "$native/build_fast.sh" NEW_NATIVE_BUILD
mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -I"$native" \
  "$native/test_keyboard_yaw_runtime.cu" NEW_NATIVE_BUILD/fast_assets.o \
  NEW_NATIVE_BUILD/cJSON.o -L"$mujoco" -Xlinker=-rpath -Xlinker="$mujoco" \
  -l:libmujoco.so.3.7.0 -lcrypto -o NEW_GPU_TEST
timeout 60s ./NEW_GPU_TEST
```

Exact training settings, with fresh output paths:

```sh
initial=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin
unset REK_POLICY_FEATURE_MASK REK_FAST_CONTACT_POTENTIAL REK_TRAIN_OPPONENT_CHECKPOINT REK_FROZEN_OPPONENT_FRACTION
unset REK_FAST_SHAPING_GAMMA REK_FAST_SHAPING_TARGET REK_FAST_SHAPING_BEARING_WEIGHT REK_FAST_REWARD_GAMMA
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1 REK_POLICY_ACTION_STRIDE=1
export REK_FAST_REWARD=round_outcome_v1 REK_FAST_SCORING=recovered_hit_rules_v2
export REK_FAST_GEOMETRY=primitive_samples_v1 REK_FAST_CONTACT_SUBSTEPS=8
export REK_FAST_OPPONENT=recovered_bot1_v1 REK_FAST_OBSERVATION=rendered_pose_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1 REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5
export REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265 REK_FAST_SHAPING_WEIGHT=0
export REK_TRAIN_SEED=419 REK_TRAIN_LEARNING_RATE=.0001 REK_TRAIN_MINIBATCH=8192 REK_TRAIN_ENTROPY=.01 REK_TRAIN_TIMEOUT=180
export REK_TRAIN_GAMMA=.9998844821426083 REK_TRAIN_GAE_LAMBDA=.9978673240629938
REK_FAST_YAW_COMMAND=legacy_velocity_slew_v1 bash "$native/run_diverse_training.sh" NEW_NATIVE_BUILD NEW_LEGACY_OUTPUT 33554432 512 512 120 "$initial"
REK_FAST_YAW_COMMAND=keyboard_reset_v1 bash "$native/run_diverse_training.sh" NEW_NATIVE_BUILD NEW_KEYBOARD_OUTPUT 33554432 512 512 120 "$initial"
```

No command-mode flag is added to live inference. Private configs
`authentic-yaw-command-legacy-trained-v1.json` and
`authentic-yaw-command-keyboard-trained-v1.json` are verified clones of the f3
legacy-v1 configuration, changing only name/checkpoint path/hash. Encoder, worker,
sampled selection, seed73 and no-feature-mask behavior are identical.

The selected completed build, tests, both runs, scripts, commands, stdout/stderr,
timings and hashes are mirrored under
`C:\rekagent\work\consistent-fighter-20260919-r1\keyboard-yaw-command-r1\spark-results`:
96 files, 34,787,971 bytes. Every remote/local SHA256 matches, and a second remote
hash pass confirms the selected sources remained unchanged during copying.
Mirror manifest SHA256:
`1b0ac672fa718f1145f63145d7b877b680080da15307e69e61dd8d06eae0da6b`.
Private `build-and-test.sh`, `verify-runtime.sh`, `run-training-pair.sh` and
`training-wrapper-result.json` preserve the executed sequence. No game capture,
proprietary asset payload or credential is added to the repository.

## Subsequent authentic evaluation and archival

The [six-round development comparison](../consistent-fighter-20260919/yaw-command-development-r37-r42.md)
finished legacy 0W/3L, 26:55 points, and keyboard reset 2W/1L, 46:36 points.
All six passed the existing strict checks. The [command analysis](authentic-yaw-mechanism.md)
found no broad increase in sustained turning, and both arms received 16
non-five-point points. These results do not establish causality or consistent
winning. The [separate frozen evaluation](../consistent-fighter-20260919/keyboard-yaw-frozen-evaluation.md)
does not count these development fights.

Build/training artifacts, mechanism measurements, derived development results,
frozen configurations and helper were copied to the evidence server under
`pufferlib/rek-evidence/2026-09-19/consistent-fighter-r1/keyboard-yaw-command-r1`:
151 files, 35,072,202 bytes. Every copied SHA256 matched and source files stayed
unchanged. Archive-manifest SHA256:
`4b0c84bedb63aa6479cddef6c09423a4ef73f7fb8069107313bdf79033c77565`.
