# Native outcome PPO after left-front exploration

One authorized native PPO epoch trained on completed authentic development
rounds r31-r33 from the frozen left-front exploration checkpoint. These are
recorded real-client trajectories, not transitions from the compact simulator
or the separate cadence-throughput benchmark. No environment was stepped,
trainer source changed, additional value scaling applied, or hyperparameter
search performed. Authentic evaluation of the resulting actor was pending at
training handoff; the diagnostics below do not establish fighting improvement.

## Data and frozen behavior

The existing strict contact/referee validation passed before export. The
default legacy-v1 exporter preserved 16,138 actual worker decisions across
three fresh-worker sequences: r31 5,625, r32 4,815 and r33 5,698. There are
16,136 locally applied requests and two terminal-race requests with actor
weight zero and value weight one. Failed startup r30 is excluded. These three
episodes are development training, not a held-out cohort.

Inputs remain the exact saved 223 unmasked features and 33-action legal masks,
with original recurrent order and fresh seed 73 per worker. Pacer-skipped
sources are not inserted or resampled. Rewards use observed awarded points
and the actual closed-round outcome: `outcome + gamma_t*Phi(next)-Phi(current)`,
where `Phi=d/(5+abs(d))` and terminal potential is zero. Reference gamma and
lambda are 0.9998844821426083 and 0.9978673240629938 per 0.02 s, exponentiated
using actual source-to-next-source QPC intervals. Receipt/observed timing is
not authoritative server execution timing.

Exact native BF16 replay matched all 16,138 sampled actions, with zero
mismatches. Its frozen FP32 values/logprobs are newly computed from the actual
behavior checkpoint `6e410064...`; no f3 replay was reused. CPU byte comparison
confirmed the exploration intervention changed only decoder category 17.
The entire parent f3 value function is unchanged for identical histories.
Earlier value calibration is already embedded in those weights and was not
applied again. This does not assert calibration on newly visited states.

| Artifact | SHA256 |
| --- | --- |
| Initial exploration checkpoint | `6e4100648c17c06142d8fe4f997b6e1e7fd958dd2f0d3a90a7de367f370ab245` |
| Actual trajectory dataset | `f6eda2eac86765e5caa07f39f86ba6f8140d09a686f7df90b4590b997baf490b` |
| Exact frozen replay | `d1f02a12c8e567237196bf61a4a85ae09b3e0fb2c6843155048e021bf8cc724a` |
| Final and epoch1 checkpoint | `add8d59367caf2487a550308dcf535409da9dbdbb4fe52e57cd6b4aba2ce168f` |
| Reused replay executable | `b44ee7d100bf9719eb2543f665b5ccbc592fced2427af374064542016b2b59ec` |
| Reused PPO executable | `0b6fcacccffa951375d0ef7cd9cc520e22f3cbdfa4b77009ee2fe25e598145c6` |
| Exact live-worker-linked native policy object | `4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c` |

## Existing update and measured diagnostics

The update reused the last f3 control's settings: one epoch, learning rate
1e-5, horizon 128, policy/value clips 0.2, value coefficient 0.5, entropy 0.001,
frozen-value GAE, native Puffer backprop/Muon and full-prefix recurrent burn-in
before each chunk. There is no MC-target flag or v2-schema flag. All 127
optimizer updates completed. Whole-round order remains fixed and there is
no held-out loss claim. CUDA produced the GAE targets; its comparison to the
independent CPU reference had maximum error zero. No extra advantage
normalization was introduced.

The existing explicit bounded-BF16 mode accepted the initial comparison:
native one-step teacher logits/value were exact; batch-forward maximum chosen
ratio error was 0.0140260525, below the unchanged 0.02 limit, with zero initial
clipping. Mean initial legal KL was 8.7457181e-8. Exact batch parity is not claimed.

| Full-dataset training-forward diagnostic | Before update | After update |
| --- | ---: | ---: |
| Mean legal KL from frozen behavior | 0.000000087457 | 0.000229849322 |
| Maximum legal KL | 0.000180651314 | 0.006293297406 |
| Chosen ratio minimum / maximum | 0.985973948 / 1.011525743 | 0.871810505 / 1.212285313 |
| Fraction outside 20% ratio band | 0 | 0.00006196555 |
| Mean value | 0.082709996 | 0.048868763 |
| Value MSE to fixed GAE target | 0.033462222 | 0.030628405 |

Frozen replay mean value was 0.082710873; mean fixed GAE target was 0.043163971.
GAE advantage mean/standard deviation were -0.0395469024/0.1786008339, with
range [-0.650390625, 0.539596498]. Progressive epoch policy loss was 0.039526877
and value MSE 0.0329655091. The post-update table recomputes the complete
dataset after optimization, rather than averaging progressively changing
parameters. One of 16,138 final ratios exceeds the clipping band: PPO clipping
is an objective term, not a hard constraint on final parameter changes.
Lower training-target value error is not held-out critic accuracy.

The GPU reservation command span was
`2026-09-20T22:28:31.811198215Z` to `22:28:36.734245153Z`, 4.923047 s.
`/usr/bin/time` measured replay 2.79 s and PPO 2.08 s, including their host work
and diagnostics. These are process times, not isolated GPU-kernel timings.
Replay, trainer and outer wrapper all exited 0. There was no failed training
attempt, numerical-threshold relaxation, or optimizer retry. The initial
checkpoint and dataset hashes were rechecked after completion.

## Commands and handoff

Windows file-only export, after all three existing strict analyses completed:

```powershell
$repo='C:\Users\Daniel\codex-rek-puffysics-training-profile'
$root='C:\rekagent\work\consistent-fighter-20260919-r1'
node "$repo\ocean\rek_g1\native5\authentic_trajectory_data.cjs" `
  $root "$root\authentic-left-front-cohort-r1" `
  0.9998844821426083 0.9978673240629938 `
  live-round_outcome_v1-r31 live-round_outcome_v1-r32 live-round_outcome_v1-r33
```

The fresh output was copied to Spark `STAGE/data`. With fresh replay/train
directories, the executed native commands were:

```bash
stage=/home/spark-advantage/rek-training/authentic-left-front-cohort-20260920-r1
initial=/home/spark-advantage/rek-training/left-front-exploration-20260920-r1/candidate-r1/left-front-exploration.bin
sha=6e4100648c17c06142d8fe4f997b6e1e7fd958dd2f0d3a90a7de367f370ab245
replay=/home/spark-advantage/rek-training/authentic-trajectory-20260919-r1/build-r1/replay-authentic-behavior
trainer=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/ppo-build-r2/authentic-ppo
mkdir "$stage/replay-r1" "$stage/train-gae-r1"
timeout 60s "$replay" "$stage/data/authentic-trajectories.bin" \
  "$initial" "$sha" "$stage/replay-r1/behavior-replay.bin"
timeout 60s "$trainer" "$stage/data/authentic-trajectories.bin" \
  "$stage/replay-r1/behavior-replay.bin" "$initial" "$sha" \
  "$stage/train-gae-r1/ppo.bin" 1 1e-5 128 .2 .2 .5 .001 \
  --allow-bounded-bf16-batch
```

The preserved `commands.sh` additionally pins executable/input hashes,
records each command, stdout/stderr, native exit code, UTC start/end and
`/usr/bin/time -v`, and verifies final/epoch checkpoint equality. It was
uploaded as a file and invoked remotely, avoiding the previous unrelated
PowerShell-stdin trailing-CR issue. Existing outputs must not be overwritten.

Candidate config:
`C:\rekagent\work\consistent-fighter-20260919-r1\authentic-left-front-outcome-ppo-v1.json`.
SHA256: `50397e8dca228b57df677aed142223a1243860d450f5b78093608a14a144620c`.
An exact structural comparison confirms only name, checkpoint path and hash
differ from the exploration config. Encoder, worker, legacy v1, sampled seed 73,
unmasked features and nominal 20 ms publication cadence remain unchanged. The checkpoint is
`STAGE/train-gae-r1/ppo.bin`. No live client was touched by this training task.

All 28 Spark files, 30,745,655 bytes, were mirrored and SHA256-readback verified
under `C:\rekagent\work\consistent-fighter-20260919-r1\left-front-outcome-ppo-r1\spark-results-r2`.
The source hashes were identical before and after copy. Mirror manifest SHA256:
`bd6f34adad0a86d08c4f1c426dd63bf5975e69809f50723384e144b0e3a02bca`.
The first mirror-only attempt rejected SCP's `remote/.` spelling before copying
any file. Its script, error and empty destination are preserved. Explicit
source children were copied into the fresh r2 mirror; no replay or training
was repeated. Export scripts/logs and the candidate config are preserved in
the private parent stage. No raw observations, weights or proprietary assets
are included in this public report.

The finalized local training/export artifacts and completed-round derived
summaries were also copied to the physical evidence server:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\left-front-outcome-ppo-r1`.
All 68 files, 55,451,574 bytes, passed source-before/source-after/NAS SHA256
comparison. Existing files were preserved. Archive manifest SHA256:
`291f4c3bde621bc5d22412c573b59775a85765c94baf274d5460eaf0938816c6`.

## Subsequent authentic evaluation

The unchanged updated actor completed r34-r36 against private Sparring Bot 1:
3:15 loss, 7:10 loss, 30:22 win. All three strict checks passed. Aggregate
1W/2L and 40:47 points do not establish improvement over its exploration
parent's 1W/2L and 41:34 points. These remain small development cohorts.
See the [fresh-round report](left-front-outcome-development-r34-r36.md).
