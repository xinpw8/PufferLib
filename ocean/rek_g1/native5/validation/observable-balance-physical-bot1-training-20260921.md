# Physical observable-balance training with recovered G1 Bot 1

The opt-in recovered Bot 1 arm completed 4,194,304 learner transitions and 16
rollout/training iterations on 2026-09-21. Native process and wrapper receipt
exit codes were zero. Final checkpoint:
`7c34eaa9f00c1ee97f8cbd8abf894d773d8a838547a10e496bcb5980de821c84`.
The full run produced 208 wins, 287 losses and 17 draws across 512 completed
training rounds, with zero runtime failure bits and zero reward saturations.
These are changing-policy physical simulator results. They do not establish
frozen-policy strength, authentic REK parity or a promotion decision.

## Controlled configuration

This arm and the [previous dummy-opponent LR 0.015 arm](observable-balance-physical-lr015-20260921.md)
start independently from the same original physical checkpoint:
`390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310`.
Initialization is weights-only with fresh optimizer, RNG, recurrent state and
step count. The copied input and native step-zero readback are byte-identical
to that original. The new final checkpoint differs; no critic transformation
was applied.

The recorded command and environment were compared mechanically with the
previous LR 0.015 run. Only the native executable, run ID, input/output paths
and the additional `REK_PHYSICAL_OPPONENT=recovered_bot1_g1_v1` selection differ.
The trainer configuration tree is identical and the original trainer object
is reused. Runtime, motion assets and scheduler code were rebuilt for the
Bot 1 integration. See the [integration report](physical-bot1-integration-20260921.md)
for its source-grounded implementation and separately preserved GPU probe.

Configuration: `mujoco_cuda`, native SONIC inference, CPU physics disabled,
no Python runtime; `rek.native5.observable_balance.v1`;
`normalized_points_falls_v1`; 512 arenas / 1,024 fighters; horizon 512;
minibatch 8,192; replay ratio 1; 120 s rounds; 50 Hz actions with stride 1;
223 observations / 33 actions; two 256-wide recurrent layers;
base/environment seeds 419; initial learning rate 0.015 with cosine annealing;
entropy coefficient 0.001; gamma 0.9998844821426083;
GAE lambda 0.9978673240629938. Each arena executes 8,192 learner steps,
or 163.84 simulated seconds. The 1,200 s timeout was not reached.

The opponent uses source-derived G1 Bot 1 tactics, continuous native commands
and assigned move dispatch with local acceptance feedback. Physical fall
detection, countdown and counted body reset remain active. Counted resets
preserve tactical/RNG/recovery state; round resets deactivate and reactivate
the controller while preserving RNG and recovery fields. This differs from
the previous deterministic approach/backoff/turn and cycling-attack dummy.

Explicit candidate assumptions remain: Update, dispatch and FixedUpdate once
per 50 Hz tick; candidate-private xorshift32 RNG; current G1 null get-up clips;
unmodeled motor shutdown hold. Authentic server cadence, Unity RNG and full
physical parity are not established. The bot's categorical inspection slot
is zero because commands are direct; it does not identify its selected action.
Shared joint pose/rate mapping remains unavailable.

## Complete-run comparison

Both arms completed 4,194,304 new transitions and 512 rounds from original
`390007e2...` with LR 0.015. Learner values precede opponent values in pairs.

| Measurement | Previous approach dummy | Recovered G1 Bot 1 |
| --- | ---: | ---: |
| Wins / losses / draws | 259 / 226 / 27 | 208 / 287 / 17 |
| Completed-round points | 5,891 : 5,638 | 8,339 : 9,067 |
| Lifetime confirmed falls | 982 : 1,165 | 1,892 : 1,530 |
| Lifetime awarded points | 7,529 : 7,571 | 11,206 : 12,503 |
| Runtime failure bits / reward saturations | 0 / 0 | 0 / 0 |
| Native trainer uptime | 709.207 s | 714.890 s |
| Complete-training native SPS | 5,914.08 | 5,867.06 |
| Whole-process elapsed time | 718.23 s | 724.03 s |
| Whole-process SPS | 5,839.78 | 5,793.00 |

The Bot 1 arm has zero redos and zero unclassified outcomes. Its learner has
fewer training wins and more confirmed falls than the dummy arm, while both
fighters receive more points. Changing the opponent changes the training
distribution, so these totals are not a same-opponent comparison of learned
strength. This single-seed run alone cannot identify whether the final policy
improved against authentic Sparring Bot 1. Frozen authentic evaluation is a
separate cohort in the [evaluation report](observable-balance-authentic-evaluation-20260921.md).

Completed-round points exclude unfinished rounds. Lifetime counters cover all
executed transitions, including unfinished rounds. Confirmed falls are native
`BECAME_FALLEN` events. Reward retains fixed scale 0.01, own confirmed-fall
penalty -0.01, bounds [-1,1], no terminal bonus and zero saturation events.

Native uptime includes all rollouts, PPO updates and checkpoint writes, but
excludes initial construction and CUDA graph capture. Whole-process time
includes initialization and shutdown. Display resolution is 1 ms and 0.01 s
respectively. SPS is the entire 4,194,304-transition budget divided by those
times; neither value is a selected dashboard window or physics-only benchmark.

## Verification and preservation

All 17 checkpoint files, from step zero through 4,194,304 in increments of
262,144, are 1,836,032 bytes and contain only finite FP32 values. Each output
sidecar's checkpoint hash, observation shape/schema, original initialization,
LR, reward, opponent, candidate cadence/RNG and trainer binary hash matched.
All 823 entries across the selected provenance, source-tree, build-artifact
and checkpoint hash manifests were rehashed successfully. Required paths and
nonempty manifests were checked. The run's completed outcomes reconcile to
512 and the recorded reward/observation/opponent modes match the command.

Spark stage:
`/home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1`.

Executed wrapper command:

```sh
bash /home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1/run-balance-physical-bot1-lr015.sh lr015-r1
```

Output: `train-bot1-lr015-r1`.
Final checkpoint relative path:
`train-bot1-lr015-r1/checkpoints/rek_native5/bot1-lr015-r1/0000000004194304.bin`.
The run's `command.txt` preserves exact native arguments and environment.

| Artifact | SHA256 |
| --- | --- |
| Executed runner | `02f483834936e7eec803ea948589052645fbe3333b3070f4f6fd4b99ab52babf` |
| Native trainer executable | `61dea9fd819e24f132ace54eaeff7cfb4520a3d79fd6b791f4cb79cc010b8aa4` |
| Reused trainer object | `254363eac4668a25ff8e3cc1cd4c3226b4216ef2046a0a9bb1ce3651caa26959` |
| Final checkpoint | `7c34eaa9f00c1ee97f8cbd8abf894d773d8a838547a10e496bcb5980de821c84` |
| Training archive | `bcc8b5d8c86bd24a5e99c8d646719a8eabd72eb25a544df0a81fd378b84740a9` |
| Aggregate result | `e4134b3d0dda64a30e62841dd8a6cc03e081c2af6d00de4f8047e3c76833f556` |
| NAS receipt | `76da256f02e495bfe7cf7f08c303aa0a8c344e2dd96b1ab36c9fe8b37aa71e12` |

Fresh private NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-bot1-training-r1`.

The 33,036,937-byte archive contains 81 selected source/build/run files plus
aggregate and inventory sidecars. These include initial weights, all 17 output
checkpoints and schema sidecars, exact scripts, raw logs, full-run timing,
selected implementation source, the new trainer binary, reused trainer object
and build/configuration provenance. Proprietary game, model and motion assets
are hash-referenced and excluded. The separately preserved integration probe
is stage `run-r2`; its archive is not duplicated here.

Source hashes before/after packaging and tar comparison passed. Spark, local
and NAS archive hashes agree. Completion and both zero exit codes were checked
before issuing the NAS receipt. Both the original training archive and prior
dummy LR 0.015 archive were reverified. All destinations were fresh; no
existing evidence was overwritten. Collection and preservation launched no
trainer, inference or game process. No proprietary assets, binary artifacts,
checkpoints or raw captures are published in this repository.

## Next controlled continuation

After four completed authentic rounds (r114..r117) produced two wins and two
losses, a weights-only continuation from the frozen `7c34eaa9...` checkpoint
was started. Output is `train-bot1-continue8m-r1` under the same Spark stage.
Its new-transition budget is 8,388,608, with the same native binary, opponent,
reward, schema, seeds, horizon, minibatch, replay ratio, initial LR and entropy.
Optimizer, RNG, recurrent state and step count restart as declared by the runner.

This tests additional training exposure. The earlier run supplied only one
completed round per arena and 16 rollout updates. The continuation supplies
32 rollout updates and 327.68 simulated seconds per arena; cosine LR decay
also stretches across that longer budget. It is not a duration-only control
with an identical per-update LR sequence. No result or promotion is claimed
before completion and new authentic evaluation.

Runner: `run-balance-physical-bot1-continue8m.sh`.
SHA256: `49adff962e2473863f7eeb5cecf5b45bfaccc05c9cc97ddd8857bcd2264fcea4`.
Syntax and no-execution input/schema/hash checks passed before launch.

The continuation subsequently completed with zero exit codes. Its full-run
results, normalized comparisons, timing and preservation receipts are in the
[8M continuation report](observable-balance-physical-bot1-continuation8m-20260921.md).
That training completion does not establish authentic fighting strength.
