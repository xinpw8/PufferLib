# Keyboard-reset yaw with five-observation cadence

One native CUDA training run combines the existing
[keyboard-reset command mode](../keyboard-yaw-command-20260920/README.md) with
[stride5 action masking](../action-cadence-20260920/README.md). It completed
successfully without rebuilding, changing source, adding a loss, or retrying.
Authentic evaluation is separate and pending in this training report.

## Matched treatment

Relative to the run that produced `85808a67...`, the sole training treatment is
`REK_POLICY_ACTION_STRIDE=5` instead of1. Both independently start from f3
`f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`, with a fresh
optimizer. This is not continued training from858 and does not reintroduce the
legacy yaw-command model.

The same existing native binary has SHA256
`c3e5e2325174eb0c49a91a5980e3632f367515ae05cc6352ed2588ae3357e450`:
`/home/spark-advantage/rek-training/keyboard-yaw-command-20260920-r1/fast-build-r1/puffer-rek-native5`.
Settings remain 33,554,432 learner transitions, 512 arenas, horizon512,
minibatch8192, H256/two MinGRU layers/BF16/CUDA graphs, learning rate0.0001 with
existing annealing, entropy0.01, vf0.5, policy/value clip0.2, replay ratio1,
gamma0.9998844821426083, lambda0.9978673240629938, environment seed419 and native
base seed73. Recovered Bot1, 120 s rounds, randomized resets, legacy223 rendered
observations, eight primitive-contact substeps and round-outcome reward are
unchanged. There is no feature mask, frozen opponent or extra shaping.

The options act at separate boundaries: cadence intersects learner action masks;
the keyboard command state still advances or resets every 20 ms. Hold0 preserves
desired input without repeating an attack, and busy ticks reset the command ramp.
Bot1 remains excluded from both changes. Recurrent inference, value-learning
samples, physics, canned durations, rewards and discounts retain the 50 Hz clock.
Existing tested helpers are reused; no additional validation harness was added.

Live cadence counts successfully ready nonterminal worker-input rows, beginning
at phase0. Warmup/unavailable rows do not consume phase; derivative-only resets
preserve it; terminal/new round/explicit reset clear it. Five observations are
nominally 100 ms, with actual duration determined by source QPC and jitter.
Existing busy projection and approximate physical-response limitations remain.

## Recorded result

The single run lasted from 2026-09-20 23:40:50.778969444 to
23:41:27.491768252 UTC. Native and outer exits were0; failure bits0. Before launch,
the GPU process query showed no REK workload and one unrelated seven-day-old
Python process using5,096 MiB. Sampled GPU/memory utilization was0%/0%; that process
was left untouched. This snapshot does not establish its activity throughout.

| Measurement | Existing keyboard stride1 | New keyboard stride5 |
| --- | ---: | ---: |
| Learner transitions | 33,554,432 | 33,554,432 |
| Trainer-loop seconds | 36.12033391 | 35.87901592 |
| Full-training learner transitions/s | 928,962.40 | 935,210.49 |
| Nominal full-action opportunities/s | 928,962.40 | 187,042.10 |
| Process wall seconds | 36.82 | 36.59 |
| Startup-inclusive learner transitions/s | 911,309.94 | 917,038.32 |
| Changing-policy compact W/L/T | 4527 / 565 / 28 | 4865 / 217 / 38 |
| Changing-policy compact awarded points | 352810:214656 | 306953:188043 |

The new run's recorded CUDA times are rollout27.69479883 s, model7.50115567 s
and training-misc0.16089926 s, with loop residual0.52216216 s. These use the
existing 64-bin/43-unique-mean aggregation, first-bin weight2, excluding repeated
final bins. Graph-mode inference/environment/copy subdivision is unavailable.

Stride5 retains every learner row. Only every fifth nonterminal row permits the
full existing legal action set; the intervening rows force hold0. Thus SPS/5 is
a nominal opportunity rate, not an exact count of attacks or successful commands.
The +0.673% learner-SPS difference between these two ordered runs is not a
speedup claim, and certainly not fivefold throughput. Changing-policy compact
scores are training diagnostics, not held-out authentic fighting strength.

Final checkpoint:
`/home/spark-advantage/rek-training/keyboard-yaw-cadence-20260920-r1/train-keyboard_reset_v1-cadence5-r1/checkpoints/rek_native5/train-keyboard_reset_v1-cadence5-r1/0000000033554432.bin`.
SHA256: `dafda776f7898bd26f5b99f313168656a61436ada999b700e0799f75f5d01bf8`.
The step-zero readback equals the original f3 bytes. The halfway checkpoint is
retained with SHA256 `d7e892de008bfc50ee5b8e130e6ddb5b15a6c10bc4c185383b6f9626caecbd89`.

## Reproduction and live handoff

The executed private `run-training.sh` SHA256 is
`386fe44225085120c678e640f2250093d3a331628f3dc0a89590bb6e92bfff1c`.
It pins the existing binary and f3 hashes, preserves stdout/stderr, native command,
timings, exits and checkpoint hashes, and invokes the unchanged native wrapper:

```sh
reference=/home/spark-advantage/rek-training/keyboard-yaw-command-20260920-r1
native="$reference/source/ocean/rek_g1/native5"
initial=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin
unset REK_POLICY_FEATURE_MASK REK_FAST_CONTACT_POTENTIAL REK_TRAIN_OPPONENT_CHECKPOINT REK_FROZEN_OPPONENT_FRACTION
unset REK_FAST_SHAPING_GAMMA REK_FAST_SHAPING_TARGET REK_FAST_SHAPING_BEARING_WEIGHT REK_FAST_REWARD_GAMMA
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1 REK_POLICY_ACTION_STRIDE=5 REK_FAST_YAW_COMMAND=keyboard_reset_v1
export REK_FAST_REWARD=round_outcome_v1 REK_FAST_SCORING=recovered_hit_rules_v2
export REK_FAST_GEOMETRY=primitive_samples_v1 REK_FAST_CONTACT_SUBSTEPS=8
export REK_FAST_OPPONENT=recovered_bot1_v1 REK_FAST_OBSERVATION=rendered_pose_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1 REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5
export REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265 REK_FAST_SHAPING_WEIGHT=0
export REK_TRAIN_SEED=419 REK_TRAIN_LEARNING_RATE=.0001 REK_TRAIN_MINIBATCH=8192 REK_TRAIN_ENTROPY=.01 REK_TRAIN_TIMEOUT=180
export REK_TRAIN_GAMMA=.9998844821426083 REK_TRAIN_GAE_LAMBDA=.9978673240629938
bash "$native/run_diverse_training.sh" "$reference/fast-build-r1" NEW_OUTPUT 33554432 512 512 120 "$initial"
```

Private final config
`C:\rekagent\work\consistent-fighter-20260919-r1\authentic-yaw-command-keyboard-cadence5-trained-v1.json`
has SHA256 `914d9d539fea0ac849e89241ee61e971369ec292cc70702f0c7b1e02cac74638`.
Relative to858 it changes checkpoint/name and selects the already tested stride5
encoder command. Worker binary, legacy schema, sampled selection, seed73 and
no-feature-mask configuration remain identical. Encoder SHA256 is
`dfac88ecde9a9825b80479ab60f6d9029a61f8f236783878b890d92765fa96d5`, with
`--action-stride 5`. No compact yaw environment flag is added to live inference.
The pending template is retained separately and is explicitly invalid for use.

Remote results were mirrored to
`C:\rekagent\work\consistent-fighter-20260919-r1\keyboard-yaw-cadence-r1\spark-results`:
24 files, 5,621,766 bytes, all remote/local SHA256 equal and remote hashes unchanged
on readback. Mirror manifest SHA256:
`b6e1eeb6e265978505c1d3d5753eac49f32467777217298f09e036ab6fb08adb`.
The final config is also copied byte-exact into the local private stage. Earlier
training, source, captures and frozen actors remain unchanged; no GPU work follows
this run during live evaluation.
