# Round outcome rewards and unresolved balance parity

The work includes simulator parity, observation parity, reward correctness,
temporal credit assignment, native CUDA throughput and authentic private-AI
evaluation. Launching the client is only a prerequisite.

## Parity comes first

The compact runtime still has no balance dynamics. Four recent native Windows
rounds contain received referee counts that the old visual fall fields miss.
The [external held-out validation](../balance-transfer-windows-20260919/README.md)
rejects the existing root predictor as a replacement: all six frozen models
worsen 0.5 s and 1 s planar prediction, with particularly large drift during
an observed local count. It remains outside the runtime. No hit-count-to-fall
rule or invented support contact was added.

The bridge now supplies received referee state, rather than `referee:null`.
Its [measurement contract](../../../../rek/evidence/windows/RekUiBridgeAgent/G1_RECEIVED_REFEREE.md)
binds copied packets to successful native application, client mirrors and the
current lifecycle. This adds measurements. The 223-feature policy projection
and its checkpoint meaning have not changed. The policy still does not receive
the new referee fields as features; a compatible dynamics/observation revision
and retraining remain necessary.

## Explicit reward alternatives

`point_difference_v1` remains the default: newly awarded own scoreboard points
minus newly awarded opponent points. This rewards scoring margin. It includes
referee awards when the dynamics supplies them; the compact runtime currently
cannot generate authentic countouts.

The opt-in `round_outcome_v1` targets completed-round results: win +1, loss -1,
draw 0. A bounded potential supplies intermediate feedback from the score
difference `d`: `Phi = d / (5 + abs(d))`. Five points equals the recovered G1
countout award, but this scale affects credit distribution only. Every step
adds `gamma * Phi(next) - Phi(previous)`, with terminal Phi exactly zero.
There is no separate raw score reward in this mode.

For T transitions, its discounted return is
`gamma^(T-1) * terminal_outcome - Phi(initial)`, up to floating-point error.
Thus a large winning margin cannot outweigh more wins merely by farming points
in equal-duration rounds. The same identity removes any positive reward for
returning repeatedly to a high-potential score state. Variable-duration rounds
retain the learner's preference for earlier wins and later losses; this is not
an undiscounted win-probability guarantee in that setting.

A countout score/reset is nonterminal. It must neither generate a terminal
win/loss bonus nor clear RNN memory. The reward function consumes scoreboard
points and an explicit round result; it cannot label visual contact as a score,
infer an attack cause, or create missing physics.

`run_diverse_training.sh` binds reward gamma and learner gamma to the same
`REK_TRAIN_GAMMA` value and rejects a contradictory explicit reward gamma.
Direct executable callers must provide that same contract themselves.
The outcome mode rejects spatial/contact-potential shaping. Reward choice
does not change the observation schema, simulator points or action masks.

## Tests and controlled native training

CPU: 286,630 checks, including 512 complete synthetic 120 s trajectories and
countout/reset/terminal cases. Maximum telescoping error is 0.00006903.
CUDA: 16,384 host/device comparisons across four graph replays, including wins,
losses and draws, with zero maximum difference. The actual `fast_step`
autoreset test passes 592 device checks across twelve graph replays, preserving
the previous 48 checks. It exercises all three terminal outcomes under both
reward modes, initial observations after reset, cumulative metrics and the
absence of duplicate reward/done on the next step. These are software tests,
not physical parity tests.

Both Spark GB10 runs start from checkpoint
`5bdad2893c5e97e682fdb33ab48a298e2cd2d9c03df724fa2e629cc1c9882246`.
Each executes 33,554,432 learner transitions with 512 arenas, horizon 512,
minibatch 8,192, seed 419, replay ratio 1, learning rate 0.0001 and entropy 0.01.
The native 50 Hz simulator, BF16 policy inference and PPO updates are C++/CUDA;
no Python or Torch training is used. Spatial shaping is disabled.

The existing task-time profile is retained: gamma 0.9998844821426083,
lambda 0.9978673240629938, 10.24 s rollout, 120 s reward half-life and 6.16 s
gamma-lambda trace half-life. Recurrent state persists across rollout boundaries.
Those choices are task-derived hypotheses, not empirically optimal settings.
Entropy remains inherited and untuned; no intrinsic exploration bonus is added.

| Reward | Full-training SPS | Startup-inclusive SPS | Training wins / completed rounds |
| --- | ---: | ---: | ---: |
| Point difference | 964,615 | 945,728 | 5,087 / 5,120 |
| Round outcome | 961,310 | 942,276 | 5,092 / 5,120 |

The 0.34% measured throughput difference is from one run per condition, in fixed
order. These are training results against the compact opponent, with a changing
policy. They do not establish authentic strength or a population win rate.
Full statistics are in `training-comparison.json`.

Frozen final checkpoint SHA-256 values:

- Point difference: `1dc17dab7f36856e058b1f32dab07f70d1edaadef58969ae80b4fc8898336b70`.
- Round outcome: `61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f`.

Private build, source snapshot, commands, test logs, training logs and checkpoints:
`/home/spark-advantage/rek-training/reward-objective-20260919-r1`.
Windows deployment and live evidence:
`C:/rekagent/work/reward-objective-20260919-r1`.

## Evaluation lifecycle

The runner now waits for both encoder calibration and CUDA worker readiness
before connecting its gameplay relay. An optional machine-coordinated readiness
handoff permits a fresh isolated client launch before arena entry. It does not
ask the user for another approval. Trials reject first observations more than
one second after round start. Redo and unknown results are incomplete.
This fixes the fresh-trial route; persistent multi-round worker reuse is still
unimplemented. The full offline control-coverage check remains required.

### Authentic private Bot 1 evaluation

Four complete native Windows rounds ran on D21 as `moogleod`, with inference
on Spark GB10. Every opponent was Sparring Bot 1 / difficulty 0 in the
verified private-AI route. The order was point, outcome, outcome, point.
Each round used a fresh isolated client, the same 223-feature projection,
BF16 sampled inference, and worker seed 73. The server RNG was not pinned.

| Reward / trial | Native score, ours:AI | Result | Dispatched attack requests | Maximum applied-action gap |
| --- | ---: | --- | ---: | ---: |
| Point / r1 | 10:6 | Win | 101 | 0.1063 s |
| Outcome / r2 | 12:9 | Win | 95 | 0.2160 s |
| Outcome / r4 | 20:15 | Win | 114 | 0.0733 s |
| Point / r2 | 15:19 | Loss | 101 | 0.1083 s |

Each trial passes the strict first-second, continuous local-control and
terminal-coverage checks. Received native score packets reconcile exactly
with the terminal scoreboards. Adjacent `*-contact-summary.json` files retain
the full checks, hashes and limits. Attack requests are not verified server
execution or attributed hits; no request was assigned a fabricated hit/miss.

The outcome checkpoint won its two rounds; the point checkpoint won one of
two. This small frozen-checkpoint comparison does not establish a population
win rate, a statistically reliable improvement, or superhuman performance.
The objective remains opt-in. Neither reward mode is parity-complete.

All 20,353 live referee payloads matched 4,772 exact native packet receipts,
with zero unavailable/ambiguous matches. Seventeen distinct latched call IDs
were deduplicated. The outcome r4 round included one local and two opponent
countouts; the point r2 round included a bilateral `DoubleKnockout` resolution.
Both rounds continued to `WonByPoints`. Individual count episodes and call
awards remain in the `*-referee-validation.json` reports; a bilateral call
must not be converted into two assumed five-point awards.

This validates the received-state observation path. It also confirms that the
missing balance/countout dynamics affect real evaluation. The compact runtime
cannot teach those transitions yet. The old visual fall counters remained zero
in the round summaries even while the referee counted fighters down.

No global keyboard/mouse input or foreground activation was used. Task-owned
isolated clients were closed after each trial. The original recorder config
was restored byte-for-byte; no REK process remains. No video was captured.

### Reproducing a fresh native trial

Use the existing isolated-desktop launcher and prepared encoder, worker and
relay configuration. Resolve deployment paths locally and pin the checkpoint,
bridge and executable hashes. Every trial requires a new output directory.

Add `startup_gate` to the runner configuration:

```json
{
  "ready_path": "<absolute-new-readiness-file>",
  "release_path": "<absolute-new-release-file>",
  "timeout_ms": 120000
}
```

The coordinator runs these commands in sequence, keeping the driver and
launcher processes hidden and preserving stdout, stderr and arguments:

1. Start `node <native5>/live_transfer_run.cjs <prepared-config>`.
2. Wait for a complete readiness JSON object verifying encoder and worker
   readiness, the checkpoint hash and `relay_connected:false`.
3. Launch with `RekIsolatedDesktopLauncher.exe --launch-rek --exe <installed-REK>
   --sha256 <executable-SHA256> --output-directory <new-launch-directory>`.
4. Read complete `launch.json`, record the actual client PID, executable and
   UTC creation time, then wait for its bridge pipe.
5. Run `node <native5>/windows_authenticated_continue.cjs <bridge-SHA256>
   <new-continuation-log> --approved-authenticated-continuation`.
6. After verified authenticated Home, atomically publish release JSON containing
   `readiness_id` and `checkpoint_sha256` copied from readiness.
7. After the driver exits, obtain fresh passive state. Close only the recorded
   client after rechecking process identity, native isolation, account and
   released control. Preserve the client if any check is unavailable.

Two outcome-policy launch attempts exposed publication races: the JSON path
existed while its contents were still empty or exclusively writer-locked.
The private coordinator now waits
for a complete JSON object within the original deadline. It retries file reads
only, including Win32 sharing/lock errors 32/33; other I/O errors, malformed
JSON, identity mismatches and producer exit fail immediately. All 1,160
publication tests pass, including real exclusive-file and byte-range locks.
Both failed attempts and exact helper sources are retained in the private
archive. No gameplay occurred in either attempt.
The checked-in driver and launcher remain the execution components; this
subsection documents their coordination, not a new launcher.

## Preserved artifacts and next boundary

The physical evidence archive is
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\reward-objective-r1`.
Its `spark` subtree contains the immutable source snapshot, build, native tests,
training commands, logs and both checkpoints. The 264,332,620-byte archive has
SHA-256 `fb36653e172beb80f18c5f841b88b04d10213ff26a4fca954edb374c42881ef8`,
verified identically on Spark, Windows staging and the NAS. The `windows`
subtree retains successful and failed attempts, raw native captures, deployed
bridge provenance and private helper sources, with per-file copied hashes.
Its 259 files total 1,666,998,295 bytes; `snapshot-hashes.json` SHA-256 is
`0b8b4d4d18ece59667e4663741b7b9a7675e89da25f3802ba1af78df488bd5ee`.
No proprietary game assets or account-page snapshots are committed to Git.

Final local regression results: 113 Node tests and 860 pure received-referee
checks pass. The native reward and actual CUDA runtime tests are described
above. Tests and game measurements have distinct scopes.

The next dynamics work must supply a validated balance/contact producer before
feeding the recovered fall/referee rules. The
[controller asset and producer investigation](balance-producer-boundary.md)
locates the actual game paths and distinguishes the current Steam build from
an older asset-bearing build. A further temporal/exploration sweep is deferred
until this material parity limitation is addressed; the current horizon,
discount and entropy settings have not been established as optimal.
