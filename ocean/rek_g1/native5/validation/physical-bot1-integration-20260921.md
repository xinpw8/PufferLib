# Recovered Bot 1 controller in the physical CUDA runtime

The opt-in `REK_PHYSICAL_OPPONENT=recovered_bot1_g1_v1` path compiled and passed
the GPU integration probe on `spark-4ae3`, 2026-09-21. The default remains
`candidate_approach_dummy`. This connects the recovered high-level Bot 1
controller to continuous native locomotion and assigned move dispatch in the
existing MuJoCo CUDA and SONIC runtime. It does not establish authentic REK
physics or server-controller parity.

## Changes

- Both fighters' current physical root transforms determine the bot's planar
  distance and signed facing angle. The 17 assigned moves obtain their primary
  limb from the existing motion/impact route catalog.
- Direct velocity magnitudes and move indices use native locomotion/composer
  logic. They do not pass through the learner's categorical held-input segments.
  Actual local acceptance or rejection determines the bot's next tactical state
  and the same-tick velocity command.
- CancelPunch preserves its native conditions. Motor suspension permits logical
  command dispatch while reference rebuilding and playback advancement remain
  suspended. Unsupported active direct/categorical handoffs return status 401.
- Confirmed falls drive the source-grounded G1 dampen/Straighten path. Current
  G1 get-up clips are null and Straighten returns false. Existing physical fall
  detection, scoring, countdown, and counted body reset are retained. Counted
  resets preserve tactical/RNG/recovery state; round reactivation restores the
  initial delay. See [recovery evidence](native-bot1-g1-recovery-20260921.md).
- The learner's observation schema and normalized reward are unchanged. The
  bot's categorical inspection slot is zero because its actual commands are
  continuous/direct; zero must not be interpreted as its selected action.

Candidate choices remain explicit: Update then dispatch then FixedUpdate once
per 50 Hz tick, and candidate-private xorshift32 RNG. Authoritative server
cadence and Unity RNG are unknown. Motor shutdown hold is not modeled. No
unmeasured get-up animation or physical fall rule was introduced.

## Verification

CPU tests passed 21,841 direct-scheduler checks, including 1,200 disabled-path
ticks byte-identical to the pre-change scheduler bodies pinned at `4abf6a9e`.
Fixtures exercise continuous commands, assigned move mapping, ordinary
rejection feedback, cancellation, suspension, and lifecycle boundaries. They
use synthetic clips and do not measure dynamics. The unchanged Bot 1 tactical
suite passed 20,042 checks, its new recovery suite passed 198 checks in both
GCC and Clang sanitizers, and planar geometry passed 52 checks.

The GPU probe includes production `runtime.cu` and links the new scheduler and
motion objects. Five CPU MuJoCo stepping/kinematics entry points are wrapped
with aborts. It uses the existing physical model, motion assets, and batch-eight
native SONIC controller. No Python or PPO is used by this probe.

First, 70 explicitly synthetic binding checks exercised assigned limb mapping,
fall dampening, retained tactical state at counted reset, move acceptance while
motors are suspended, learner/direct separation, and round activation. A full
runtime reset separates these fixtures from the subsequent physical execution.

Then four arenas ran 1,200 CUDA graph replays each, with a masked-neutral
learner, autonomous recovered opponent, and diagnostic 10 s rounds:

| Physical integration measurement | Result |
| --- | ---: |
| Move attempts / accepted / rejected | 22 / 22 / 0 |
| Nonzero bot velocity-command ticks | 2,300 |
| Aggregate terminal transitions | 8 |
| Counted-reset arena ticks | 2 |
| Confirmed falls, learner : bot | 0 : 2 |
| Awarded points, learner : bot | 10 : 2 |
| Runtime failures / reward clipping | 0 / 0 |
| CPU physics calls / PPO updates | 0 / 0 |

These are implementation checks with real physical execution, not a trained
policy evaluation. A velocity-command tick does not itself prove displacement.
The terminal check is aggregate across arenas. The 19.47 s process duration
includes initialization, test fixtures, and repeated host inspection; it is
not a training-SPS benchmark. The test makes no contact-causality or authentic
parity claim beyond its measured fields.

## Reproduction and provenance

Private stage:
`/home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1`.

The fresh `build-r1` recompiled runtime, motion assets, and scheduler. The
original trainer object and 31 other object pins were verified against the
previous physical build. Trainer configuration and runtime API are unchanged.
Both trainer and probe compiled and linked without compiler/linker diagnostics.

```sh
bash run_physical_bot1_runtime_probe.sh \
  /home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1/probe-build-r1 \
  /home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1/run-r2
```

The output directory must be fresh. The retained `run-r2` already exists and
must not be overwritten. The initial `run-r1` wrapper failed while recording
the command, before executing the probe. Its output is preserved; the shell
continuations were corrected before `run-r2`, which exited zero.

| Artifact | SHA256 |
| --- | --- |
| Runtime source | `a0c35585f70bb65a57da7e4278037b421c28a587b4bab436251032a49050df43` |
| Probe source | `f5a8e15e044a519fe1a3e1ed737b9ee76531705cf112ba49558053561f4008ec` |
| Probe executable | `80ef158eb4951ab6569cbce6cffe428308d0ab35a28ac4a23f76668efc053b91` |
| New trainer executable | `61dea9fd819e24f132ace54eaeff7cfb4520a3d79fd6b791f4cb79cc010b8aa4` |

Commands, output, source/object hashes and private artifact receipts are kept
with the stage. Proprietary assets, executable binaries, and checkpoints are
not included in this repository.

The 32,801,738-byte private build/probe archive was verified at
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-bot1-integration-r1`.
Archive SHA256:
`a1c98077fd903f5d92fc20f7fc0d5980e3ad59ea33081c25a55432f9486d7e1b`;
receipt SHA256:
`f899a1939bd037dcdbb92f492a22a70ff3d41ec8e6e592846f65d1b8a8a15b9b`.

Following the passing probe, a controlled 4,194,304-transition physical training
run was started from original checkpoint `390007e2...`, with LR 0.015 and the
same normalization/horizon/seeds as the prior dummy-opponent arm. Its complete
training and authentic evaluation results are separate from this probe.
