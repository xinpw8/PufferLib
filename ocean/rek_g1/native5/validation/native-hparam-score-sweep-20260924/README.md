# Native CUDA score-first hyperparameter sweep

Executed on `spark-4ae3`, 2026-09-24. The selected configuration improved the independent simulator holdout. Authentic REK match improvement remains unestablished.

## Result

25 configurations received 8,388,608 additional training transitions each. Four configurations were then independently restarted for 33,554,432 transitions at each of two training seeds. All 33 training runs exited successfully with finite checkpoints and zero reported runtime failure bits. This totals 478,150,656 training transitions, plus a separate 1,048,576-transition smoke test.

The winning tested configuration is `arm-02`: Muon LR `0.000055`, entropy `0.00017`, horizon `512`, gamma `0.9998844821426083`, GAE lambda `0.9978673240629938`, PPO clip `0.2`, value coefficient `0.5`. Fixed settings: 512 arenas, minibatch 8192, replay ratio 1, hidden width 256, two MinGRU layers, cosine LR annealing, carried recurrent state. A 512-step rollout spans 10.24 s at 50 Hz; discount and advantage-trace half-lives are 120 s and 6.16 s. This is the best configuration tested here, not a global optimum.

Confirmation results average two independently trained checkpoints evaluated on the same three evaluation seeds. They are descriptive averages, not additional independent evaluation fixtures.

| Configuration | Own points / round | Conceded / round | Margin / round | Round wins |
|---|---:|---:|---:|---:|
| Unchanged checkpoint | 32.18 | 9.10 | 23.08 | 91.93% |
| Inherited control, further training | 33.27 | 9.56 | 23.71 | 92.97% |
| arm-02, LR .000055, horizon 512 | 34.86 | 9.11 | 25.74 | 94.53% |
| arm-05, LR .0003, horizon 256 | 31.63 | 9.14 | 22.49 | 94.79% |
| arm-10, LR .001, horizon 512 | 21.24 | 6.86 | 14.39 | 85.55% |

Win-rate-only selection would favor arm-05 despite its reduced points and margin. The score-first objective selected arm-02. Its seed-947 checkpoint was fixed before a new ten-seed holdout, with 1,280 policy-side-0 rounds per checkpoint:

| Independent holdout | Unchanged | Selected |
|---|---:|---:|
| Own points / round | 32.1883 | 34.2430 |
| Conceded points / round | 9.1547 | 8.7477 |
| Margin / round | 23.0336 | 25.4953 |
| Wins / losses / draws | 1170 / 104 / 6 | 1210 / 60 / 10 |
| Round win rate | 91.4063% | 94.5313% |

Paired ten-seed cluster-bootstrap 95% intervals: own-score improvement `[1.030, 3.190]` points, margin improvement `[1.375, 3.648]` points, and round-win improvement `[1.6, 4.8]` percentage points. These intervals describe this candidate environment and sampled fixtures. They cannot establish real-match performance or physical parity.

## Throughput and native reference ranges

Selected training run: **947,865 learner transitions/s**, or **928,714/s** including process startup, graph setup, and checkpoint I/O. Confirmation-run median: **944,107/s**. The full screen ranged from 921,623 to 1,019,892/s. These are end-to-end headless training rates, not arena-only stepping or rendered FPS. Environment stepping, inference, recurrent learning and Muon updates use native CUDA. Node handles experiment orchestration and results; no Python training runtime was used.

Inspected native PufferLib defaults and actual GPU robot-arm/Breakout configurations before choosing ranges. Default Muon LR is .015, robot-arm .0003, and Breakout approximately .0627. Higher values are inappropriate for this warm-started policy: matched .0075/.015 arms exhibited large KL/clipping excursions and collapsed scoring. See `LR_DIAGNOSTIC.md` for the measured update and loss evidence. The full executed ranges and optimizer/time-scale interpretation are in `HYPERPARAMETERS.md`.

## What stayed fixed

Every training run starts from checkpoint `7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96` with fresh optimizer/RNG/recurrent state. Larger budgets are independent restarts, not exact resumed optimizer trajectories. The trainer and evaluator share the corrected `f7-action-id-fix-20260924-r1` runtime objects. Observation contract is `rek.native5.scaled_polar_xy.v1`, 223 floats. Equal-width `balance8_v1` checkpoints are incompatible.

Reward remains normalized own-minus-opponent point delta, with .15 per accepted learner attack divided by 100. Calibration has corrected action IDs, coherent hit rejection, bot-award probability .25, and no unmeasured kick-fall prior. Rewards, opponent, action frequency and contact settings were not swept. This compact runtime still does not model full contact-driven balance, falls or multi-round matches.

The native evaluator executes both sides. Side 1 is retained privately but excluded from ranking because calibration is asymmetric. Its files called matches actually contain completed rounds; no synthetic grouping is reported as a match win.

## Checkpoint and evidence

Selected checkpoint on Spark:

```
/home/spark-advantage/rek-training/native-hparam-score-sweep-20260924-r1/confirm-s947/arm-02/checkpoints/rek_native5/arm-02-s947/0000000033554432.bin
SHA256 fa6f760a5823d904b9f636df6421426e21344c03767ffa8ed0fe776c400b75ad
```

Complete commands, environment overrides, stdout, stderr, process times, checkpoint readbacks, frozen round records and native exit receipts are under:

```
/home/spark-advantage/rek-training/native-hparam-score-sweep-20260924-r1
```

The public `PUBLIC_SUMMARY.json` covers screening and confirmation. `HOLDOUT_SUMMARY.json` covers the independent final holdout. Scripts and synthetic tests accompany both. No checkpoint, game binary, credential or raw proprietary game recording is committed here.

Constellation uses its existing native `cache_data`/`seethestars` binaries. Original training INIs remain byte-identical in `constellation/datasets/import-20260924T164925795Z-jcdF4f`. Separately labeled, derived frozen evaluations are in `constellation/frozen-datasets/import-20260924T170008789Z-Ycz9OE`. Score, conceded points, margin and round-win fraction remain distinct metrics. The viewer was not launched over the live REK display.

The completed training/holdout snapshot is stored on the physical server:

```
\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\native-hparam-score-sweep-20260924-r1\native-hparam-score-sweep-20260924-r1-snapshot-20260924T170200Z.tar.gz
Bytes: 161798201
SHA256: e9a1d3dd53d0b1b2bdac6d186af57be350722598cfd0fd7fbd0f4e7e2ff1cc0a
```

## Authentic-client validation

The first selected-policy attempt reached a logged-in Free Play screen and requested private practice. `SkipIntro` was accepted 22 ms after acquiring the native control lease. No arena arrived within 45 s. The trial recorded zero observations, predictions and applied actions, then stopped its stream and released control cleanly. The existing client remained running. This is an entry failure, not a policy win, loss or successful transfer test. Details are in [LIVE_ENTRY_FAILURE.md](LIVE_ENTRY_FAILURE.md).

The later native log explicitly reported: `Solo find ended: No practice arena is free right now. Try again in a few minutes, or drop into an open arena.` No public-arena fallback was used. Source: `/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-native-hparam-baseline-20260924-r1/unity.log`, line 14824. This attempt therefore does not provide any real opponent score or match outcome.

A retry using the same running client successfully entered private G1 sparring against Bot 1. Round 1 completed with a **12:19 loss**, 2,991 applied policy actions and 59 armed attack requests. Round 2 began automatically in that client, then ended incompletely at **1:7 with 69.30366 s remaining** when the game exited at 17:11:44 UTC with code 5. Native logs report SIGSEGV and a CoreCLR access violation. The underlying crash cause is unknown. The controller did not kill the client or restart it between these rounds. There was no restart after the crash.

**Completed matches: zero. Consistently beating authentic REK AI remains unachieved.** The incomplete round is excluded from completed-round aggregates. One completed loss and no live baseline block cannot establish a transfer improvement. [LIVE_RESULTS.md](LIVE_RESULTS.md) and [LIVE_RESULTS.json](LIVE_RESULTS.json) contain exact receipts, action counts, point increments, video hashes and crash evidence.

Live evidence and two validated MP4 files are saved on the physical server:

```
\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\native-hparam-live-20260924-r1
Archive: native-hparam-live-evidence-20260924-r1-snapshot-20260924T171310Z.tar.gz
Bytes: 59333557
SHA256: dcb04f290d0919423796ee0afbaadd33cfc82b13fa0d97c9e74184dc350c3c05
round-1.mp4: 10882438 bytes, completed round
round-2-incomplete.mp4: 4792699 bytes, partial round before crash
```

Both videos are below 20,000,000 bytes. The archive retains the entry attempt, retry, runtime exit/crash receipts and recovery evidence. No proprietary game binary or credentials are published to Git.
