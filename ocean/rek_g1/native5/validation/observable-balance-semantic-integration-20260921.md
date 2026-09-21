# Observable balance: semantic pretraining adapter

Date: 2026-09-21. Status: CPU fixtures, serial CUDA compile/link, native GPU
numeric/default-path checks and the three-arm full-training performance
benchmark passed. The root agent ran the GPU checks after build review. No
authentic evaluation or checkpoint promotion is claimed here. Existing compact
defaults remain selected unless the new schema is requested. These results pin
the base adapter in commit `c35bed7d9b918ec060f7ce5bb1f3fd6d0cb44501`; they do
not validate subsequent policy-owned previous-action augmentation.

## Scope and mapping

Select `REK_PHYSICS_BACKEND=semantic_cuda` and
`REK_OBSERVATION_SCHEMA=rek.native5.observable_balance.v1`. The native trainer
binding now accepts this explicit combination, while continuing to reject a
schema-unverified frozen opponent. Other backends and old-schema behavior are
unchanged. No checkpoint conversion or relabeling is provided.

`fast_observable_balance.h` creates the existing shared
`rek_observable_balance::Snapshot` and calls its unchanged `project()` for both
fighter perspectives. Each row has the same 223-feature contract and structural
availability as the physical/live adapters. Absolute fighter slots remain
absolute in the snapshots; `actor_slot` alone determines row perspective.

| Shared input | Existing semantic producer |
| --- | --- |
| Root X/Y | Candidate slider `Fighter.x/y`, also used by raw/qpos export |
| Root Z | Current source frame `FastFrame.root_z` |
| Root WXYZ | Existing `root_quaternion`: logical yaw composed with current clip orientation |
| Sample time | Current arena tick multiplied by 0.02 s |
| Round identity | `RekNative5RoundResult.round_number` |
| Duration, remaining time | Existing parameters and round result |
| Points | Absolute fighter point counters |
| Active, terminal | Existing round phase and terminal flag |
| Joint pose | Explicitly unavailable; existing clip joint values are not a verified client joint mapping |
| Referee/count state | Explicitly unavailable; `referee_available=0`, `count_mask=0` |

The shared projection computes root-origin finite differences, quaternion tilt,
heading, bearing and point deltas. It does not reuse legacy compact tilt/down
proxies, route-entry histories, internal velocities, contact labels or action
state. Columns 202, 204 and 205 are zero because referee state is unavailable,
not because the adapter observed absence of falls or counts. Joint pose/rate
columns and their availability flags are zero. All structural padding follows
the unchanged shared header.

The optional allocation stores one prior absolute snapshot and two projected
rows per arena. Projection advances once during each exported observation;
repeated `rek_native5_encode_fighter_observations` reads use the cached rows and
do not consume history. Explicit reset clears the history. A changed round key
masks derivatives and point deltas at genuine round boundaries. Same-round
body/route changes retain history. Existing training autoreset still exports the
terminal result followed by the next round's first observation. Invalid shared
input clears both projected rows, invalidates history, and raises sticky runtime
failure bit 2048.

The raw inspection buffer, qpos/qvel, movement, opponent logic, contact/scoring
rules, reward and action masks were not changed. The startup record identifies
the candidate root producer, absent joints/referee state, unavailable compact
fall-event source, incompatible old weights and unestablished authentic parity.
Checkpoint sidecars must retain this exact schema and record the semantic
pretraining producer; equal tensor size does not make legacy `scaled_polar_v1`
or `owned_yaw_v1` weights compatible.

## Checks completed

- The actual CPU/CUDA adapter helper passes 2,602,487 CPU checks with GCC 11.4
  and undefined-behavior sanitization, and with Clang 14 and address plus
  undefined-behavior sanitization. All 446 values match independently assembled
  equivalent shared snapshots exactly. Fixtures cover both perspectives,
  availability, first/history observations, body displacement, terminal/new
  rounds, explicit reset, score deltas/regression, invalid input, time gaps and
  quaternion sign aliases.
- The unchanged shared suite passes 1,172,058 assertions. Its maximum equivalent
  native/Unity input discrepancy remains `1.49011612e-08`.
- A normalized-line-ending comparison of the existing source block from
  `clampf` through the end of `scaled_value` is exact over 47,276 characters.
  This checks unchanged source, not compiled default behavior or throughput.
- A fresh serial `nvcc` compile/link for `sm_121` passes. The first attempt
  stopped at a missing staged compact header and is retained as `compile-r1`.
  Dependency-complete `compile-r2` exits zero. It uses `--threads 1`, a fresh
  trainer object from the original pinned patched PufferLib sources, the updated
  `puffer_env.cu`, and `REK_NATIVE5_COMPACT_AUTORESET=1`. No old trainer object is
  reused. The linked binary has no `libpython` or `libtorch` dependency.

The current `ptxas -v` output reports:

| Kernel | Registers | Stack bytes | Shared bytes | Spill loads/stores |
| --- | ---: | ---: | ---: | ---: |
| `fast_step_warp` | 210 | 1088 | 4032 | 0 / 0 |
| `fast_step` | 219 | 1024 | 0 | 0 / 0 |
| `fast_reset` | 166 | 784 | 0 | 0 / 0 |
| `encode_rows` | 33 | 32 | 0 | 0 / 0 |

The separately compiled immutable pre-change runtime reports 211 registers and
416 stack bytes for `fast_step_warp`, 219 registers and 400 stack bytes for
`fast_step`, and 46 registers and 48 stack bytes for `fast_reset`, with no spills.
Main step register counts therefore do not increase, while stack usage rises by
672 and 624 bytes. Reset register and stack usage increase. The baseline object
is `fast-observable-balance-20260921-r1/build-r1/baseline_runtime.o`, SHA-256
`b722e6f49ae32553ad68b4f96065d350e5acfdd71ceb2346752c83828809ebc5`.

Lane-zero projection uses the shared double-precision math and snapshot storage.
The optional runtime branch does not prove zero default-path resource cost.
The complete-process measurements below include this implementation. No
optimization is included here.

## Native GPU validation

`fast-observable-balance-20260921-r1/run-r1` exits zero. Four arenas and 902 ticks
per runtime exercise deterministic legal actions, six-second rounds, standalone
round rollover, explicit reset and training autoreset. The current pair passes
2,109,928 checks; shared-projection maximum absolute error is
`9.53674316e-07`. Coverage includes 7,192 history rows, 40 first observations,
12 round resets, 24 terminal rows and 16 nonzero point-delta rows. Repeated
encoded reads remain byte-identical and do not consume observation history.

Current default and observable opt-in have byte-identical raw observations,
qpos/qvel, action masks, selected actions, rewards, terminals, round state and
logs. The separately linked immutable pre-change default runtime and current
default runtime produce identical 19,229,888-byte traces, both SHA-256
`294acb38612a72315a3e00c6a06e9e7d59462be72fe54b8e43107c7a6670a0e8`.
The probe uses no synthetic state writes, policy inference, training or CPU
physics stepping. Its correctness checks do not establish authentic dynamics
or policy parity.

## Complete-training performance benchmark

`semantic-observable-balance-20260921-r1/benchmark-paired-r1` contains one
sequential, order-fixed measurement per arm: old default, new default, new
observable. Each completes 4,194,304 transitions and 16 updates with 512 arenas,
horizon 512, minibatch 8192, replay 1, LR 0.015, entropy 0.001, gamma
`0.9998844821426083`, GAE lambda `0.9978673240629938`, 120-second rounds and
fresh seed 419. No checkpoint is loaded. All arms use the same newly compiled
patched trainer object and exact pinned configuration; only the runtime object
and requested observation schema distinguish the arms. Fresh step-zero byte
identity is unmeasured because this unchanged trainer does not export it.

| Arm | Whole process seconds | Whole process transitions/s | Native loop seconds | Native loop transitions/s |
| --- | ---: | ---: | ---: | ---: |
| Old default | 5.56 | 754,371 | 4.805 | 872,904 |
| New default | 5.65 | 742,355 | 4.880 | 859,489 |
| New observable | 7.46 | 562,239 | 6.687 | 627,233 |

Whole-process timing includes startup, warmup and checkpoint I/O; native timing
uses the final trainer uptime. The new default is 1.59% lower in this single
sample, which is insufficient to establish a default-path regression. The
observable arm is 25.47% lower than old default in this measurement. No kernel
timing is substituted for complete training throughput.

All three native processes exit zero, reach final epoch 16, and save 16 finite
checkpoints with schema sidecars. Each reports 512 completed rounds, reconciled
outcomes, zero failure bits and zero reward saturations. The reward retains
`fall_event_source=unavailable_in_compact`. Old and new default both report
24 wins, 484 losses, four draws and completed points 5,283:13,879. Observable
reports 55 wins, 453 losses, four draws and 7,219:16,227. These short fresh-seed
performance arms do not measure learned fighting strength.

Old and new default final checkpoints are byte-identical, SHA-256
`06fded95823930349b7ad209d6f53a5f3a961f94823f77ba1ef7fd9a17b55e68`.
The observable final checkpoint is
`50edeeb4910000af2b699d6751cd854a49d12ba10a9c68d218332eacf63fbecc`.
None is promoted. The initial byte-identity limitation does not weaken the
observed final byte identity, but limits claims about unrecorded initialization.

## Observable batch-scale performance

The separate, root-run `batch-scale-r1` measures the same observable binary with
1,024, 2,048 and 4,096 arenas, sequentially. Each arm completes 16,777,216
transitions. Horizon 512, minibatch 8192, replay 1, fresh seed 419, LR 0.015,
entropy 0.001, gamma/lambda, reward, 120-second rounds and semantic settings
remain as above. This is another single-sample performance comparison, with
startup and warmup included in whole-process timing. It is not a comparison of
learned policy quality, and its larger step budget differs from the three-arm
benchmark.

| Arenas | Native epochs | Whole process seconds | Whole process transitions/s | Native loop seconds | Native loop transitions/s | Completed rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 32 | 22.42 | 748,315 | 21.588 | 777,155 | 2,048 |
| 2,048 | 16 | 20.88 | 803,507 | 19.981 | 839,658 | 2,048 |
| 4,096 | 8 | 19.80 | 847,334 | 18.693 | 897,513 | 0 |

All processes exit zero, report zero failure bits and reward saturations, and
save finite final weights. Checkpoint counts are 32, 16 and eight. Round outcome
counters reconcile without assuming a fixed number of completed rounds. At
4,096 arenas, 4,096 transitions per arena span only 81.92 simulated seconds,
shorter than a 120-second round. Zero completed episodes and zero completed
point/outcome totals are therefore expected, and provide no win/loss evidence.
The larger batch is fastest here, but changes collection length per arena and
the number of rollout updates at fixed total transitions.

Final checkpoint SHA-256 values, retained as performance artifacts only:

| Arenas | SHA-256 |
| ---: | --- |
| 1,024 | `ab041fa788b8cd7de09a4fd2dd4cb4abbb6b11074f7158d055d98652a5ac241a` |
| 2,048 | `fd105f8dfe0aa5798c4322e49837626c86def80ae4b69d6c326ff7d67290f28a` |
| 4,096 | `5c0c78c0c59057129ef1572e0d56d3cfba07fd0145d897f74867bb82ac7ef46e` |

## Evidence and pins

Private build root on Spark:
`/home/spark-advantage/rek-training/semantic-observable-balance-20260921-r1`.
The selected 44-file public source tree is `source-r2`; no private game assets
were copied. `compile-r2` contains `build-script.sh`, `commands.txt`, per-unit
stdout/stderr, `source-hashes.sha256`, `artifact-hashes.sha256`, `validation.txt`
and `exit-code.txt`. Both manifests were reverified after linking. The authored
Windows script is
`C:\rekagent\work\semantic-observable-balance-20260921-r1\compile-only-r2.sh`.

SHA-256 pins for the exact staged source bytes and build evidence:

| Item | SHA-256 |
| --- | --- |
| `fast_runtime.cu` | `cda6c04441e40c8d5e97014e21eaa48041234a8b27c5878d8854026c7a1ae5c5` |
| `puffer_env.cu` | `fbadcdcd2d1d814cf47ff406d3699473ee6431496d8394bdcaf30ba823228376` |
| `fast_observable_balance.h` | `75f47b21282f15fcea3260caea84e30117fa653efea2b88b5f635511d9ea60ce` |
| `fast_observable_balance_test.cpp` | `971e4df893bf373735a41f86c5fa89480236e6d5a3af7b8b346820a4dd81cbad` |
| Unchanged `observable_balance.h` | `4b651a2f02b7a335cb781bb886b048d83025acc88dca4a522f62bc4cea7a0d3a` |
| `build-script.sh` | `919c73f5a9829cb8619425d0cb62537fe6f20c2d4e124c9124c886ad35522af9` |
| `source-hashes.sha256` | `f2c384371467fb7a6a8d00b41792480ae4965eb35d3017d12e3cbefcd8fb0d16` |
| `artifact-hashes.sha256` | `bf32d1d7f9368246fcfdadf726cadaa82b387e6344cda421f84115715f9e5f5a` |
| `fast_runtime.o` | `5bd1cd242666211ed8f512ddecf335ba66b57d0a6fb37e67258379aea9b1a774` |
| Fresh `pufferl.o` | `80a92c67f21d4ddf773fe7feb098e96be1c4445fa74cb1370a8026e1222d3e09` |
| `puffer-rek-fast-observable` | `ed1aa0c0107630cb012473460300990e12ac4eb9a84f7ec1d66fd1197e4fcb25` |

The trainer snapshot comes from
`physical-observable-balance-20260921-r1/trainer-build-r1/trainer`. Its pinned
patched `pufferl.cu` is
`ae71826468701bf19691548555c1a2d354f8795065bb3bc7fb6e5e2e2b0eb378` and
`algo.cu` is
`8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92`.
The prior trainer manifest is retained beside the fresh build. Production
PufferLib source pin remains `773f923d80e73bdc255a2ba730c918b28e416aa1`.

The baseline full trainer, linked with the same fresh `pufferl.o` and immutable
old runtime object, is SHA-256
`ba9b5b4b6e0e308b4e431350e98c4f329c47de1c0837643fe25d09c664ff7b4d`.
The benchmark collector verifies commands/configuration/pins, final native
epoch, finite checkpoint bytes, round reconciliation and runtime/reward status.
Raw stdout, timing and reward records independently support the totals above.

Verified NAS archive:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\semantic-observable-balance-r1`.
It preserves the selected public source, original failed and successful build
evidence, baseline/current probe builds and complete numeric traces, all three
benchmark logs/configurations/checkpoints/sidecars, and preservation scripts.
No private game assets, unfinished scale benchmark or subsequent augmentation
work is included. Source hashes were verified before and after archiving, tar
contents were compared with their sources, and Spark/local/NAS copy hashes
match. Existing NAS files were not overwritten.

| Archive item | Bytes | SHA-256 |
| --- | ---: | --- |
| `semantic-observable-validation-r1.tar.gz` | 95,858,575 | `cfec3c0f8994754bb1e49e82428cc2a531407c60b5578acec18d5d4e3dd44613` |
| `source-hashes.sha256` | 45,723 | `df9317fe43a24d5637aefcf8ec7b71b9414c46c3954818aab6659dba0241a995` |
| `benchmark-summary.json` | 5,126 | `e732bcd987359b02848df5257691ef10cbeffeacf106f155b5617420c1049a69` |
| `archive-receipt.json` | 1,688 | `db5ab91e964faa633bc82e72c579d42c406192faed62778f276f571f0b8fc4c9` |

The local preservation scripts are in
`C:\rekagent\work\semantic-observable-balance-20260921-r1`; the resulting
local evidence directory is `archive-validation-r1`.

Batch-scale evidence is preserved separately in the new NAS package
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\semantic-observable-batch-scale-r1`.
It retains all three runs, 56 checkpoints and their sidecars, exact commands,
configuration/pins, runner/collector, raw logs/timings and summary. The earlier
base package remains unchanged and supplies the pinned binary/build/source
provenance without duplicating those artifacts. Before/after source checks, tar
comparison and Spark/local/NAS hashes all pass.

| Batch archive item | Bytes | SHA-256 |
| --- | ---: | --- |
| `semantic-observable-batch-scale-r1.tar.gz` | 90,901,661 | `a0d1b4ca4165e4ec44c8681abac468f6b61a1cadd44830cee21b44e6f3c25bc1` |
| `summary.json` | 3,165 | `07353c8c8a4305eb5fa4242c5169bcb758f78c14ff1ad4bcef9022cd65b7fbfb` |
| `archive-receipt.json` | 1,415 | `fbc544b0dfffcd5ccdbf80130d8734debf38bfdd5058797c01bdecba0ed36b1a` |

## Interpretation and remaining work

This is an ordinary-move pretraining path with common observation definitions.
Candidate slider/root clips do not model integrated balance, falling, floor
contacts, physical recovery or referee count/reset dynamics. The compact
translation-settled move acceptance and attack movement behavior also differ
from the recovered physical opponent dispatch. Compact contact geometry and
body-velocity modes remain kinematic approximations. Shared encoding removes
observation-definition differences; it does not remove these dynamics or
opponent-distribution differences.

The checks above support this adapter's numeric and default-path behavior under
the tested workloads. Broader performance characterization and ordinary-move
pretraining remain separate work. Any policy produced through this path requires
physical fall-aware fine-tuning and independent frozen authentic Bot1 rounds.
No compact-only result qualifies a checkpoint for promotion.
