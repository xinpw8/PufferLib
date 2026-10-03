# Lite training: compact speed with modeled falls

The compact `semantic_cuda` runtime trains at roughly 1M SPS but has no balance,
so it has no falls. In REK every counted fall costs the faller 5 points, and in
the recovered authentic data 88 of 105 falls were slips. The physical MuJoCo +
SONIC runtime has real falls but trains at about 6k SPS. Lite training keeps
the compact kernel and adds:

1. a fall model, fitted to the physical runtime, that decides when a fighter
   starts to fall from its current situation, and
2. the recovered REK referee, applied exactly to those falls.

Falls are not drawn uniformly. The onset hazard is a logistic function of the
fighter's own move and its phase, the opponent's move and phase, distance,
closing speed and whether a scored hit just landed, including own-move and
opponent-move interactions with distance. The physical runtime is itself not
reproducible (identical inputs diverge from tick 1, with tilt differing by up
to 13.5 degrees), so a conditional probability is the honest target.

## Runtime switches

| Variable | Effect |
| --- | --- |
| `REK_LITE_FALLS=model.json` | Enables lite falls with a `rek.lite_falls.v1` model. Unset: unchanged behavior. |
| `REK_FAST_REWARD=move_start_v1` | Diagnostic reward: a fixed value on each accepted start of one native move, nothing else. |
| `REK_FAST_REWARD_MOVE=7` | Native move index for `move_start_v1` (7 is the HH left front kick, action category 17). Required. |
| `REK_FAST_REWARD_MOVE_VALUE=0.01` | Reward per accepted start. Default 0.01. |

With both features unset, the CPU emulation below reproduces the previous
`fast_runtime.cu` bit for bit on every exported buffer. The step kernels keep
their resource use: the warp kernel uses 229 registers (233 before) and the
scalar kernel 234 (233 before), with the same 1088/1312-byte stacks and no
spills (nvcc 12.9, `sm_121`). The added per-fighter work is a few table lookups, one
`expf` and three counter-based hashes per tick.

## Fall and referee semantics

From `ocean/rek/evidence/G1_FIGHT_SEMANTICS_CURRENT_BUILD.md`, with
`CanGetUp` false for G1:

- **Onset.** An upright fighter outside the 2 s post-spawn grace starts
  falling with the model's hazard. It stops acting from the next tick: its move
  is dropped and only "continue" is unmasked.
- **Resolution.** A falling fighter recovers, becomes fallen quickly, or stays
  down uncounted ("stuck") before becoming fallen. Outcome probabilities and
  delay quantiles come from the model.
- **Counting.** `BECAME_FALLEN` increments `falls`, raises Slip or Knockdown
  (recovered strike window) and starts the 3 s count. A second fallen fighter
  makes it a double knockdown and restarts the count.
- **Expiry.** The opponent receives 5 points (5 each on a double count). With
  time remaining, `ResetBothToSpawn` teleports both fighters, clears every fall
  and starts the 2 s grace. Your edge case follows directly: a fighter that is
  down but uncounted when the opponent's count expires is cleared without
  conceding, and only it scores 5.
- **Hits.** Hits score only while both fighters were upright at the start of
  the tick (`requireBothUpright`). The gate acts before any cooldown or dedup
  update.
- **Round end.** The round does not end during a count. Expiry at zero time
  awards the points and ends it.
- **Recovered Bot1.** Bot1 receives `opponent_down` when the other fighter is
  fallen. It is not updated while its own fighter is down, because compact Bot1
  has no recovery behavior.

When enabled, observations publish the physical runtime's fall block in raw
fields 71..85 (tilt, height ratio, contact proxies, phase, timers, grace,
events) and the referee fields 184+18..25, 32, 35 and 36. The rendered root
pitches over and drops while falling or fallen. Joints keep the clip pose.

## The HH sanity probe

```sh
bash ocean/rek_g1/native5/build_fast.sh /abs/new-fast-build
bash ocean/rek_g1/native5/run_lite_kick_probe.sh /abs/new-fast-build /abs/new-probe-output
```

The probe uses the current compact configuration (recovered Bot1, rendered pose
observations, primitive geometry, geom-pair contacts, body cvel, random resets)
plus `move_start_v1` for native move 7 at 0.01. It trains a fresh policy on
4096 arenas with horizon 64 for 512 updates, about 134M steps by default, with
entropy 0.001. The pinned trainer uses raw advantages, so the entropy weight must
stay well below the reward.

It then builds the evaluator from the same objects and evaluates greedy BF16 on
both sides, 256 arenas with 4 rounds each. The policy passes when, on each side,
at least 95% of its accepted move starts are HH and it uses at least 90% of its
HH opportunities (ticks where category 17 was legal). `verdict.json` records
both numbers, the per-category start histogram, falls and the final training
SPS.

`LITE=smoke` (the default) runs with the synthetic smoke model below, so falls,
counts and resets happen during the probe. `LITE=none` disables falls, and
`LITE=/path/model.json` uses a fitted model. Under any of them the correct policy
is still "repeat HH": falls carry no penalty here, only lost time.

`fast_policy_eval` now reports `learner_move_starts_by_category`,
`watched_share_of_starts` and `watched_opportunity_use` (watched category
`REK_EVAL_WATCH_ACTION`, default 17). Runtime JSON may set
`fast.lite_falls_model`.

## Phase 0: fit a real fall model

`lite_fall_dataset.cu` runs any `runtime_api.h` runtime, normally the
physical backend. A mixed GPU behavior policy (aggressive, random, kicker,
passive, fixed per row) drives both fighters. On the GPU it aggregates, for
every fighter tick, upright exposure and fall onsets per model cell, plus
outcome delays per move class. Falls cleared by a reset or a round end are
recorded as censored. The output size does not depend on run length.

```sh
bash ocean/rek_g1/native5/build_lite_fall_dataset.sh /abs/new-dataset-build   # physical objects of 2026-09-21
/abs/new-dataset-build/lite-fall-dataset lite_fall_dataset.physical.example.json /abs/train.json
# second run with another seed for a holdout
python3 ocean/rek_g1/native5/fit_lite_falls.py fit /abs/train.json /abs/lite-falls.json --holdout /abs/holdout.json
REK_LITE_FALLS=/abs/lite-falls.json bash ocean/rek_g1/native5/run_lite_kick_probe.sh ...
```

The example config uses 512 arenas, matching the batch-1024 SONIC controller
used for physical training, and 60,000 ticks (30.7M arena ticks). At that
runtime's ~10k arena steps/s, an estimate rather than a measurement, this is
about an hour. The fitter is a weighted, L2-regularized logistic regression on
the aggregated cells: the same likelihood as row-level data. Outcomes use
smoothed counts. Delay quantiles use Kaplan-Meier for the censored stuck group.
The fitter reports held-out log loss and calibration deciles.

Logger features are derived only from raw fields that both runtimes publish
with the same meaning: routes 179/182, commands 176..178, roots, fall block 79,
83 and 85, and score delta 184+34. The lite runtime computes its hazard
features from the same quantities.

## Verification in this repository

None of this has run on a GPU yet. Each item lists what each test establishes:

- `test_lite_falls.cpp` (ASan/UBSan, 498 checks): binning, sampling, hazard,
  fall paths, single and double counts, the uncounted-down edge case, and
  loader validation and rejections.
- `test_fast_runtime_cpu.sh` (60 checks plus closed-loop recovery) compiles the
  device half of `fast_runtime.cu` as host C++ and steps it on synthetic assets.
  - **Bit identity.** With `REK_FAST_ORIGINAL` set, disabled features are
    bit-identical to the original runtime for 4 policies.
  - **Exact reward.** `move_start_v1` pays exactly on accepted move-7 starts
    and never otherwise. Repeat-HH reaches the kick-rate limit.
  - **Referee invariants with falls.** No hits unless both fighters are
    upright, downed fighters masked to "continue", no round end during a
    count, and 5 points per counted fighter. Counts, knockouts, double
    knockouts, spawn resets and stuck downs cleared by a reset all occur.
  - **Closed loop.** Data generated under the known smoke model, aggregated by
    the logger's own `lite_fall_dataset_observe.h` and fitted, recovers that
    model on held-out data. Log loss came within 0.11% of the true model, onset
    calibration within 1.5%, the HH mid-kick hazard within 3%, kick outcome
    probabilities within 0.03 and delay medians within 3%.
  - `REK_CPU_TEST_CUDA_INCLUDE` must point at CUDA headers. No GPU is needed.
- `fast_mode_config_test.cpp`: the eval config switch.
- All changed CUDA units compile with nvcc 12.9 for `sm_121`.

## Limits

- **No fitted model yet.** `lite_falls_smoke_v1.json` is hand-written and only
  exercises the pipeline. Phase 0 must run on Spark before lite training means
  anything about REK falls.
- **Features exclude joint state.** If physical falls depend on it, held-out
  log loss will show the gap. The heavier fallback is a learned balance-state
  model.
- **Shrinkage in low-data cells.** They move toward pooled rates. In the
  recovery test, idle hazard was overestimated 5x at about 1e-5 per tick.
- **Model drift is not automated.** A policy optimized against the model can
  drift to states the logger never saw. Refit on physical rollouts of the
  current policy and compare falls per round between lite and physical before
  trusting results.
- **Proxies.** The fall pose and contact counts are presentation and
  observation proxies, not physics.
