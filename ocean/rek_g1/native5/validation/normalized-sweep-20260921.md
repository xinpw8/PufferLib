# Normalized-reward sweep and authentic screening

## Executed scope

Fourteen native C++/CUDA PufferLib arms ran on Spark, each for 8,388,608 learner
transitions. All started with identical actor and recurrent parameters from
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`.
Only the 256 value-head weights differed between critic initialization controls.
All training, simulator inference, environment stepping and PPO were native;
Node only aggregated completed artifacts.

The grid varied reward objective, entropy coefficient `{0.0001,0.001,0.01}` and
learning rate `{0.00003,0.0001}`. Normalized point/fall arms reset the critic to
zero. Terminal-outcome arms retained the original critic, which was trained on
that objective. Two additional normalized controls retained the original critic
or scaled it by 0.01. That scaling is an initialization experiment, not a valid
conversion between these different reward objectives.

Fixed configuration: 512 environments, horizon512, minibatch8192, two256-wide
MinGRU layers, 120 s rounds, 50 Hz decisions, environment seed419, base seed73,
gamma0.9998844821426083 and lambda0.9978673240629938. Discount half-life is120 s;
the gamma-lambda trace half-life is6.16 s; each rollout is10.24 s. Contact modes
are `body_cvel_v1` and `geom_pair_v1`, yaw `keyboard_reset_v1`, observation
`rendered_pose_v1`, opponent `recovered_bot1_v1`, eight contact substeps,
random initial gap0.55..2.5 and full heading spread. Additional shaping is off.

Normalized reward remains `(own points - opponent points - confirmed own fall)
/100`, safely clipped to[-1,1]. The compact runtime supplies no fall events;
this sweep cannot measure a fall penalty's behavioral effect. The separate
physical runtime produces actual detector events. Outcome reward uses terminal
win/loss plus discounted bounded point-margin potential. Its per-step bound is
[-2,2]; it is an experimental alternative, not the normalized point/fall mode.

The actual pinned trainer uses raw GAE advantages. Therefore fixed entropy and
critic coefficients are sensitive to reward scaling. See the corrected
[previous report](normalized-falls-training-20260921.md) and
[critic initialization tests](critic-fixed-scale-20260921.md).

## Frozen simulation screening

Each checkpoint played512 complete rounds:256 arenas on each fighter side,
seed10001, sampled BF16, 120 s each, same runtime and opponent. All records had
the expected checkpoint hash,6000 ticks, distinct side/arena keys, no diagnostic
feature override and zero failure bits. No physical falls occurred or were
invented. These fixtures are development screening, not authentic REK wins.

| Arm | Reward | Critic scale | Entropy | LR | W/L/D | Mean point margin |
| --- | --- | ---: | ---: | ---: | --- | ---: |
|01|normalized|0|0.0001|0.00003|507/5/0|97.961|
|02|normalized|0|0.0001|0.0001|505/6/1|102.588|
|03|normalized|0|0.001|0.00003|507/5/0|99.775|
|04|normalized|0|0.001|0.0001|508/3/1|94.594|
|05|normalized|0|0.01|0.00003|503/8/1|92.369|
|06|normalized|0|0.01|0.0001|483/28/1|63.109|
|07|outcome|1|0.0001|0.00003|506/5/1|58.910|
|08|outcome|1|0.0001|0.0001|508/4/0|56.805|
|09|outcome|1|0.001|0.00003|507/4/1|65.236|
|10|outcome|1|0.001|0.0001|509/2/1|85.762|
|11|outcome|1|0.01|0.00003|501/11/0|71.689|
|12|outcome|1|0.01|0.0001|506/6/0|56.928|
|13|normalized|1|0.01|0.0001|482/26/4|64.596|
|14|normalized|0.01|0.01|0.0001|477/33/2|60.717|

Original f3 baseline:506/6/0, points54,873:16,080, margin75.768 per round.
Previous normalized d4 baseline:215/292/5, points25,746:18,645. The previous
longer normalized training regressed substantially in this fixed test.

Arm10 ranks first by win fraction, then margin. Arm04 leads the normalized
objective. The differences from f3 are just3 and2 additional wins, respectively;
one training seed and this saturated simulator do not establish a robust or
global optimum. Arm02 has the highest points and margin, but fewer wins than
arm10. Thus selecting maximum points/reward is demonstrably different from
selecting maximum win fraction in this screen.

At fixed LR0.0001 and zero critic, normalized entropy0.001 gives508 wins versus
483 with entropy0.01. At entropy0.01, resetting/scaling the critic alone gives
483/477 wins versus482 for the unchanged critic. This supports lowering entropy
in this bounded experiment; it does not support claiming critic scaling alone
fixed fighting.

The exported common reward is the undiscounted sum reconstructed from points
and fall events. It is not the exact discounted PPO return and does not make
returns from different reward formulas comparable.

## Performance and Puffer Constellation

Complete training-loop throughput was approximately929,795..957,494 learner
SPS. The final native console uptime has1 ms display resolution. Complete
process timing, including initialization and checkpoint I/O, gives863,026..
885,809 SPS. These are training measurements, not physics-only or evaluator
throughput. All fourteen runs exited0.

`run_diverse_training.sh` now accepts `REK_TRAIN_METRIC`,
`REK_TRAIN_DOWNSAMPLE` and `REK_TRAIN_BASE_SEED`, retaining existing defaults.
This sweep used metric`wins`. The old `perf/train` metric measures training
seconds, not strength or SPS. Selecting it also caused the trainer to choose
log columns before any complete round, dropping later score/win columns.
With metric`wins`, history contains completed-round metrics, but its last row
may precede training completion. The summarizer therefore uses final console
uptime and independent whole-process timing, not incomplete history duration.

The actual pinned native C Constellation viewer and C cache converter were
built from source hash
`e7d3a78955358a416c91c2c415cd3189de090f80ceca129c3ca717ba630a441e`.
All14 original INIs were imported unchanged into two reward groups using
`cache_data --full`. All28 score/win values survived cache serialization within
its printed precision. No win-rate-to-score remapping was used. The viewer's
`env/score` means changing-policy own awarded points, not frozen win rate.
Its axes were set to entropy, own score and learning rate on isolated Spark
display`:98`. The viewer is available through the existing loopback noVNC
forward. No Windows keyboard/mouse input was used.

One inherited evaluator metadata limitation: its final summary hardcodes
`environment_randomizes_seed:false`. The executed wrapper explicitly enables
`REK_FAST_RANDOM_RESETS=1`. Commands/environment and raw records take precedence
over that stale summary field; no fixed-fixture determinism claim is made.

## Authentic development screening

Both candidates used the same existing native worker, v1 live encoder,
sampled seed73 and isolated Windows desktop. Each fought private G1 Sparring
Bot1, difficulty0, on account moogleod. Both owned clients closed after testing.

| Candidate | Attempt | Result | Ordinary awards | Five-point awards |
| --- | --- | --- | --- | --- |
|arm04 normalized|r64|loss10:20|5:10|5:10|
|arm10 outcome|r65|win18:6|8:6|10:0|

For each round, exact owned-PID native capture, complete policy coverage,
complete native capture, consistent terminal evidence, full awarded-point
reconciliation and referee verification all passed. No server-confirmed causal
attack attribution is inferred from point packets. Neither candidate requested
straight-left/front category17 during its screening round.

These are one-round development results per candidate. Arm04's508/512 simulated
wins did not transfer to an authentic win. Arm10 is the best-tested candidate
from this sweep so far, not an accepted consistently winning policy.

Arm10 checkpoint SHA256:
`cba28ce1a7cb4e4eecf60e4bc70d564c27add0bf180ef6e566b6ce9068b81c4b`.
It is frozen for a prospective20-round authentic cohort, attempts r66..r85,
requiring18 wins and positive aggregate margin. Ties count as non-wins. The
screening round r65 is excluded. A third non-win stops testing because acceptance
has become impossible. No parameters change during that cohort.

## Reproduction and artifacts

Scripts: `run_normalized_sweep.sh`, `run_normalized_sweep_eval.sh`,
`summarize_normalized_sweep.cjs`; runtime input:
`validation/normalized-sweep-runtime.json`, staged as`scripts/sweep-runtime.json`.
The training build is the preserved normalized-falls build. Critic variants and
the frozen evaluator were built in new directories. Source snapshots, exact
commands, stdout/stderr, config/environment, hashes, original logs, checkpoints
and per-match records are private under:

- Spark:`/home/spark-advantage/rek-training/normalized-sweep-20260921-r1`.
- Windows:`C:/rekagent/work/normalized-sweep-20260921-r1`.
- Authentic raw trials:`C:/rekagent/work/consistent-fighter-20260919-r1/live-point_difference_v1-r64`
  and`live-round_outcome_v1-r65`. The first path is a legacy harness label; its
  candidate was trained with normalized rewards.

No proprietary binaries, controller assets, checkpoints, credentials or raw
game captures are included in the public repository.
