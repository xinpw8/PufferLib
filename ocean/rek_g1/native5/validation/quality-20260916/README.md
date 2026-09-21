# September 16 policy-quality investigation

Latest follow-up: [any private REK AI evaluation](ANY_AI_EVALUATION.md) removes
the Bot1-only policy restriction while preserving private/no-human scope.
It records authentic 18:10 and 5:17 round results, an incomplete follow-up,
and a reproduced/fixed pipe-writer failure mode. A reliable win rate and parity
remain unproven.

Started from fetched GitHub branch `codex/puffysics-training-profile`, commit
`82c40510b136e05b6e883b428c1cab4f9cbf17d2`. The earlier throughput and stationary
target failures were already repaired. The selected V4 checkpoint trains at
approximately 2.23 million aggregate training SPS and remains native CUDA.

Its high compact win rate did not carry into authentic private Sparring Bot 1.
The two earlier authentic rounds lost 15:32 and 6:16 clean hits. The latter is
one trial with a video and duplicate report, not two independent trials.

## Verified live action-mask defect and repair

Visual-client telemetry sometimes reports a transient zero native velocity
while the bridge still retains held W/S/A/D. The source mask offered attacks
in that state. The next callback restored translation, and dispatch rejected
the requested attack. Compact training checks retained held translation too.

The bridge and native observation encoder now also check retained translation.
The encoder only removes mask entries; it does not relax source restrictions,
queue attacks, change observations, or add a facing/range heuristic.

Offline replay of the exact saved source streams found:

| Trial | Rejected attacks | Rejected with translation held | Previously applied attacks blocked by correction | Observation vectors changed |
| --- | ---: | ---: | ---: | ---: |
| Original r4 | 208 | 208 | 0 | 0 / 3686 |
| Original video-r2 | 155 | 155 | 0 | 0 / 3631 |

These counts establish the action-contract defect. Offline replay cannot
establish which new actions would be selected or predict a new round result.

Regression first failed on held translation with zero native velocity under
the old encoder. Updated native tests pass 145 hinge cases and 890 encoder
assertions. The bridge contract passes 149 assertions with no game or global
input invocation. The passive report analyzer also passes its two Node tests.

## Executed one-variable authentic trial

`live-mask-r3` used the unchanged checkpoint, CUDA worker, bridge binary,
observation schema, sampling seed and busy-duration projection. Only the
encoder executable changed, plus output path and maximum recording duration.
The bridge source fix was not deployed and the running game was not restarted.

- Host: `spark-4ae3`, isolated Wine/X11 `:98`, proven private Sparring Bot 1.
- Round completed at 2026-09-16 22:54:03 UTC.
- 25 attack requests, 25 local acknowledgments and native send returns,
  zero attack-gate rejections.
- Still lost: 0:5 replicated clean hits, no falls, Bot 1 won by points.
- 3614 policy decisions, 30.14 decisions/s; worker median 0.290 ms.
- Frontal alignment within 0.16 rad: 9.32% of sampled observations.
- Owned velocity neutralized and neutral-send returned; lease released.
- No Windows input or public/human opponent interaction.

This confirms the command-mask repair in a live round. It does not show
improved fighting strength. Local send returns still do not prove server
acceptance of every move. One nonattack request reached the terminal boundary
after the policy stream closed and was rejected as `policy_stream_not_owned`.

Earlier attempts are preserved: `live-mask-r1` failed CUDA device startup with
out-of-memory before taking control. A minimal 1 MiB probe failed at
`cudaSetDevice`, before allocation, then later succeeded without stopping other
jobs. `live-mask-r2` requested private entry but hit the driver's 45 s deadline.
Passive inspection subsequently proved the arena had loaded. No authentication
failure was demonstrated. The third attempt attached to that private arena.

Full private traces:
`/home/spark-advantage/rek-training/policy-quality-20260916-r1/`.
Only aggregate reports, code, hashes and test outcomes are published here.

## Training target defect

The [native contact audit](../../validation-quality/CONTACT_AUDIT_20260916.md)
replayed the unchanged V4 policy and reconstructed every awarded contact with
zero accounting mismatches. Against the scripted opponent, 37.22% of rewarded
contacts failed the recovered strike-apex gate. Applying proxy relative speed,
apex, cooldown and deduplication in sequence rejected 47.09% of the original
rewarded contacts. This is a partial acceptance audit on proxy geometry, not
an authentic score prediction.

The selected curriculum also devoted approximately 83.3% of transitions to
nonattacking opponents in expectation. Held-out evaluation varied starting
geometry while retaining the same opponent programs and compact dynamics.
Remaining transfer differences include command-heading versus rendered-root
heading, 50 Hz training versus approximately 30 Hz live recurrence, requested
move timing versus unavailable server acceptance/completion, and omitted
knockdown/contact-response dynamics. More training in unchanged V4 cannot
validate those differences.

## Executed native CUDA retraining

The optional `recovered_hit_rules_v1` mode is described and tested in
[RECOVERED_SCORING_20260916.md](../../validation-quality/RECOVERED_SCORING_20260916.md).
Legacy `v4_spheres` remains the default, with bitwise preservation verified.
Neither mode models authentic contact response or knockdowns.

Both runs used the pinned native PufferLib 5 trainer, 512 arenas, horizon 128,
minibatch 8192, replay ratio 1, 120 s rounds, learning rate 0.0001 annealed to
zero, gamma 0.999, GAE 0.995 and entropy coefficient 0.01. Resets randomize
distance 0.55 to 2.5 m and full heading. Shaping is zero. Both opponents attack:
the frozen old V4 policy and the compact scripted attacker. This script is not
the authentic Sparring Bot 1 implementation. Policy weights warm-start exactly;
optimizer state restarts. No Python interpreter or CPU physics ran in training.

| Run | Additional transitions | Frozen / scripted arenas | Training uptime | Aggregate training SPS |
| --- | ---: | ---: | ---: | ---: |
| recovered-r1 | 67,108,864 | 384 / 128 | 41.823468 s | 1,604,574 |
| recovered-r2, initialized from r1 | 268,435,456 | 256 / 256 | 167.126178 s | 1,606,184 |

SPS is learner transitions divided by the native trainer's complete training
uptime, including rollout and optimizer work. It is not environment-only SPS.
GPU jobs already running on Spark were left untouched. These timings are not a
controlled performance comparison against the earlier V4 throughput result.

Closed-loop evaluation used the same explicit recovered scoring mode, BF16
sampled inference, 120 s rounds, 256 arenas on each side and reset seed 200003,
which was not used for training. There is no shaping in evaluation.

| Frozen evaluated policy | Versus compact script: W / L / D | Versus old V4: W / L / D |
| --- | ---: | ---: |
| Old V4 | 492 / 19 / 1 | 256 / 210 / 46 |
| recovered-r1 | 374 / 134 / 4 | 278 / 231 / 3 |
| recovered-r2 | 447 / 63 / 2 | 297 / 160 / 55 |

The final candidate improves the measured matchup with the old policy but
regresses against the compact script. The same evaluation seed informed the
second stage, so these are validation comparisons, not an untouched final
test set. No checkpoint is promoted on these numbers alone.

Candidate SHA-256:
`af84c4ec92e953e72c6bf4cde9ccb8cf80258a91dbf01947312e622dd492876e`.

## Authentic trial scheduling

The first candidate captures attached partway through already-active rounds.
They are not full-round evaluations. The driver also revealed separate
`BetweenRounds`, `FightOver` and transient `RoundActive` readiness races. It
now waits through native transitions and only sends `StartRound` when the
client explicitly reports Idle, excluding the separately handled losing
post-fight prompt. The bridge itself validates its native idle-ready path.
Its `space_gate_would_allow` diagnostic describes the post-win prompt only and
must not block ordinary idle ready requests. Node protocol and
passive-analysis tests pass 13 cases. The bridge contract passes 149 assertions
with zero native-game or global-input invocations.

The authentic recovered state machine has five-second inter-round and
fight-over transitions. Neither trial commands nor inspected local controllers
accounted for automatic progression. The exact server-side new-fight trigger
is unobserved. `fight_epoch` increments during fighter despawn, so it is not
used as a reliable count of matches.

All rows below use the new checkpoint. Clean hits are client-replicated
counters, not a reconstructed weighted scoreboard. Each listed attack request
received a local send acknowledgment; no server-acceptance field exists.

| Trial | Initial time remaining | Initial clean hits | Last clean hits | Classification |
| --- | ---: | ---: | ---: | --- |
| r1 | 13.958 s | 0:7 | 7:11 | Late attachment; Bot 1 won |
| r2 | 87.175 s | 0:2 | 9:21 | Late attachment; Bot 1 won |
| r4 | 55.628 s | 0:6 | 3:17 | Late attachment; Bot 1 won |
| r6 | 31.330 s | 5:14 | 7:14 | Late attachment; Bot 1 won |
| r8 | 47.652 s | 5:9 | 14:17 | Late attachment; Bot 1 won |
| r9 | 119.967 s | 0:0 | 16:18 | Client exited with 13.772 s remaining; result unknown |
| r13 | 119.683 s | 0:0 | 9:16 | Complete timed round; Bot 1 won |

Trials r3, r5 and r7 issued no policy actions because start requests hit the
transitional states described above. Trial r10 could not connect after the
authentic client exited. The planned r11 was not executed. The two early
partial captures and every subsequent failure are retained, not silently
discarded as successful full rounds.

The r9 exit exposed an additional cleanup error: a failed relay write could
leave its already-created response promise to reject unhandled. The driver
now consumes that rejection while preserving the original transport failure;
the regression test exercises both failures. No kernel OOM entry or fatal
game-stderr diagnosis established why the authentic process disappeared. Its
exit must not be labeled a policy defeat, deliberate user close, or a GPU OOM
without further evidence.

The isolated Spark client was relaunched once at 23:19:37 UTC, retaining its
existing login. No Windows application was manipulated. Attempt r12 then
exposed the overly restrictive post-win space-flag check described above and
issued no gameplay input. After correcting it against the actual native
bridge source, r13's `StartRound` returned `remote_ready_request_issued` and the
round completed. All 78 attacks were locally accepted, with zero rejected
attacks. The policy still lost 9:16 clean hits. Its stream ended normally,
owned velocity was neutralized, and the control lease was released.

**Result: the action-mask defect and several controller lifecycle failures are
repaired, and the recovered scoring candidate has been trained and measured.
Reliable victory over authentic Bot 1 has not been demonstrated.** Neither the
new checkpoint nor the approximation is promoted as a parity-validated solver.
The complete trial still ran at 30.113 policy decisions/s rather than the
trainer's 50 Hz, alongside the remaining heading/pose/timing/model differences.

Private source traces and commands remain under the task's Spark artifact
root. Public copies of the two training command lines and bridge-test stdout
normalize trailing whitespace only; original logs remain unchanged on Spark
or in the captured command output. No game binary, clip array or checkpoint
payload is included in this directory.

## Publication cadence repair and second complete round

The old publication deadline restarted from every emitted frame. With a
roughly 60 FPS client, its nominal 20 ms interval commonly emitted every
second frame. The repaired deadline retains its phase, drops missed-frame
debt, and publishes at most one real observation per Unity frame. It never
fabricates intermediate observations. Offline clock/relay tests pass 4,473
assertions: 500 publications in ten seconds at 60 and 100 FPS, and 300 at
30 FPS. Sixteen Node transport/configuration tests also pass.

The new bridge DLL has SHA-256
`40109bdcb2b84fabb2d66b8c995855fee1253a2c7ca943382aff8dbd7dc4eb99`.
At 23:39 UTC, only the verified task-owned Spark `:98` game was stopped. Its
previous DLL was backed up outside the plugin directory, the installed hash
was verified, and the same isolated launcher restarted it. Windows input,
the container, Wine services, and unrelated Spark jobs were untouched. The
trial config explicitly pins the new bridge hash and the unchanged r2
checkpoint. The deployed DLL also includes the already-tested held-input
source-mask correction; previously the encoder alone enforced that repair.

| Complete trial | Measured decisions/s | Accepted attacks | Final native counter | Winner |
| --- | ---: | ---: | ---: | --- |
| recovered-r13, old deadline | 30.113 | 78/78 | 9:16 | Bot 1 |
| cadence-r1, repaired deadline | 48.343 | 70/70 | 13:24 | Bot 1 |

`cadence-comparison.json` records the actual QPC windows and frame rates.
The new round began at 119.667 s remaining, 0:0, and ended normally at zero.
One terminal-boundary nonattack was rejected. Control was neutralized and
released. These are independent sampled rounds, so their scores do not
establish that cadence increased or decreased strength. They establish that
the repaired cadence runs substantially closer to the training frequency
and does not by itself make the candidate beat Bot 1.

### Native score terminology correction

The replicated field is named `RoundState.CleanHits`, but recovered
`PointTracker.RecordHit` adds the limb's point award into it: hand 1,
foot/shin 2. The native counters above are therefore weighted point counters
according to that recovered implementation, not necessarily numbers of
individual contacts. Earlier tables retain the literal source-field name.
The recovered training scorer already uses these weights. A separate
observation mismatch remains: live features 221/222 use the weighted counter
delta, while the original compact runtime uses an unweighted contact count.

The source-grounded opponent audit is in [NATIVE_BOT1_GAP.md](NATIVE_BOT1_GAP.md).
The existing compact script is not the recovered native Bot 1 controller.

## Recovered Bot1 training target and rendered-pose observations

Three explicit optional modes now address identified training mismatches:
`recovered_bot1_v1`, `rendered_pose_v1`, and `recovered_hit_rules_v2`.
The [implementation report](../../validation-quality/NATIVE_BOT1_IMPLEMENTATION_20260916.md)
documents the recovered state machine, all 17 move pools, rendered-heading
and derivative features, and catalog-derived limb eligibility. These modes
retain the existing approximate slider/sphere dynamics. Native server physics,
knockdowns, controller blending and authoritative move timing remain unmodeled.
Legacy modes stay selectable; no existing viewer checkpoint is replaced.

Training ran on Spark's NVIDIA GB10 using native PufferLib 5 CUDA, 512 arenas,
horizon 128, minibatch 8192, BF16 sampled minGRU inference, seed 197 and 120 s
rounds. All opponent rows use the recovered Bot1 candidate. There are no frozen,
neutral, retreat-only or strafe-only opponents in this run. Shaping is zero.
The weights warm-started bitwise from recovered-r2, with a fresh optimizer,
learning rate 0.0001 annealed to zero, gamma 0.999, GAE 0.995, entropy 0.01 and
replay ratio 1. Starting gaps span 0.55 to 2.5 m with randomized full headings.

- Additional transitions: **536,870,912**.
- Complete native training uptime: **335.270462513 s**.
- Aggregate training throughput: **1,601,307 learner transitions/s**.
- Training process exit: **0**; runtime failure bits: **0**.
- Final checkpoint SHA-256:
  `f87dae69a777e4ac28782bdee89b30d7434208d773be75bf97f56fba4a52b07e`.

Throughput includes rollout and optimization. No Python interpreter or CPU
physics stepping ran. The native startup asset loader still uses MuJoCo
kinematics to bake clip poses; linked NCCL happens to reside in a Python package
directory, which does not imply Python execution. Trainer display labels with
zero-millisecond environment time are not a verified physics time breakdown.

Each frozen evaluation comprises 256 randomized arenas on each side, sampled
BF16 inference and 120 s rounds, with all three explicit new modes enabled.

| Checkpoint | Evaluation seed | Wins / losses / draws | Win rate | Weighted points for / against |
| --- | ---: | ---: | ---: | ---: |
| recovered-r2 before this run | 200009 | 360 / 151 / 1 | 70.31% | 32,225 / 23,376 |
| Bot1-rendered-r1 | 200009 | 499 / 11 / 2 | 97.46% | 77,961 / 25,045 |
| Bot1-rendered-r1, fresh seed | 200011 | 500 / 12 / 0 | 97.66% | 79,505 / 25,085 |

A separate fixed-start candidate check with seed 200013 scored 507 wins,
5 losses and no draws across 512 matches (99.02%), with 68,619 to 24,855
weighted points. It retains sampled policy actions and randomized Bot1 move
selection. Fixed candidate starts are not certified authentic reset states.

All 1,024 post-training games scored at least one hit. Mean first-hit time by
side was 3.20 to 3.78 s. Facing within 0.16 rad covered 21.1% to 24.0% of ticks,
and the busy fraction was approximately 96.8% to 96.9%. The two evaluation
seeds were not training seeds. These are victories over the recovered
controller inside the candidate dynamics, not evidence of authentic wins.

The first authentic trial, `live-bot1-rendered-r1`, issued no fighting actions.
The requested private route loaded an inactive solo AI session reporting
Sparring Bot 3, difficulty 2, round 3. Exact-Bot1 validation correctly prevented
control, but the driver reported the nonspecific entry timeout. Its stream
cleanup and lease release succeeded. This is a session-selection failure,
not a loss and not an authentic evaluation result.

Commands, aggregate evaluations, trainer metrics, hashes and test stdout/stderr
are saved in the adjacent `train-bot1-rendered-r1`, `eval-*-bot1-*` and
`build-bot1` directories. Full raw traces and checkpoint payloads remain under
`/home/spark-advantage/rek-training/policy-quality-20260916-r1` on Spark.
Public text copies normalize trailing whitespace; no game binaries or weights
are published.

## Completed authentic test of the new checkpoint

After the failed idle-session attempts, the isolated Spark client was restarted
with the tested private-ready bridge. The previous DLL was backed up outside the
plugin directory. Only the verified owned game PID 3412 was stopped. No Windows
input, container-wide restart, Wine service termination or unrelated job stop
was performed. Existing account authentication was retained.

`live-bot1-rendered-r3` then entered a fresh private session already reporting
Bot1, requested the ordinary `StartRound`, and verified an active exact G1/Bot1
pair before starting policy input. The new pre-start ready branch was therefore
not exercised in this round. The policy ran from 119.700 s remaining, 0:0, to
the timed round end at 2026-09-17 00:33:03 UTC.

**Authentic result: lost 4:14 to Bot1.** All 78 attack requests were locally
accepted. There were 5,820 policy decisions and one terminal-boundary nonattack
rejection. Facing within 0.16 rad covered 11.72% of observations; projected busy
fraction was 64.26%. Local send returns still do not prove authoritative attack
acceptance. Owned velocity was neutralized and the lease released normally.

This result does not support promoting the checkpoint as a reliable authentic
fighter. The large candidate-simulator improvement did not transfer in this
complete round. Additional training on unchanged candidate dynamics cannot
establish the missing mechanics.

### Knockout target gap

The [native KO audit](NATIVE_KO_TARGET_GAP.md) identifies a major omitted loss
mechanism in the previous 13:24 authentic trial. Four opponent +5 score updates
occurred with the actor low and heavily tilted, each followed by both fighters
resetting to spawn. Recovered native code awards five knockout points through
the same score counter, then continues the round after resetting both robots.
The best-supported decomposition is actor 13 ordinary points versus opponent
4 ordinary points plus 20 knockout points. Authoritative per-event reasons are
not available, so causal hit attribution remains unknown.

The earlier zero client fall counters and terminal `knockout=false` did not
establish an absence of continuing-round knockouts. That interpretation is
withdrawn. The fast trainer explicitly omits contact response, balance and
knockdowns. Recovering the opponent and ordinary-hit rules does not repair this
missing outcome model. No synthetic hit-count or guessed tilt rule was added.

### Private-session startup correction

Idle difficulty is a replicated pre-start value. Native AI ownership/difficulty
selection occurs during active possession, so a nonzero Idle value does not
prove which opponent will be spawned. The one exit/reentry attempt in r2
completed but retained that pre-start value and issued no gameplay actions.

The new `ReadyPrivateAiSession` only permits one native ready request in a
verified isolated, private, no-human, inactive Idle session with no visual
fighter pair. It does not modify difficulty, score, pose or physics. The existing
exact active-Bot1 and G1 policy checks remain unchanged. A spawned wrong bot
stops the driver before any policy input. The runtime-route preflight is
nonbinding; ordinary active-session binding and invalidation remain intact.
Tests pass 25 Node cases, 4,492 relay assertions and 275 protocol/pipe cases.

Deployed bridge SHA-256:
`2edf75c65e6693db28eb7e87c45d88522668f549c7b457cdf0aa108582667952`.
Relay SHA-256:
`0438528bbda9ad736b10bd68a1d2dbfee8f1f555f74e1e6b73b4a61799d8a674`.

## Existing physical CUDA backend verified

The [completed physical probe](PHYSICAL_QUALITY_PROBE.md) links the preserved
native MuJoCo CUDA implementation and the public GEAR-SONIC family controller
candidate. Identity with the current REK service's controller weights remains
unknown; see the [asset and authority boundary](../../../README.md#asset-and-authority-boundary).
Four arenas each advanced 20 simulated seconds with no PPO updates, no learned
policy loaded and zero CPU physics calls or runtime failure flags. Idle robots
remained upright. The two-sided scripted attack lane produced a measured fall,
a three-second referee count, a five-point knockout award and bilateral reset.
These physical/referee capabilities already exist; they were bypassed by the
fast runtime. The diagnostic's 15.487 s wall time includes per-tick host action
selection, synchronization and logging and must not be called training SPS.

The [detailed authentic audit](PRIVATE_READY_AND_LIVE_R3.md) also shows that
the new 4:14 loss had ordinary score increments and no KO-like pose/reset
sequence. It therefore cannot be explained by knockouts alone. Recorded right-
hook request windows contain root movement absent from the frozen-root fast
attack branch. The physical diagnostic likewise moves during attack windows,
but its combined hook/yaw inputs prevent treating those as matched authentic
motion measurements. Neither backend nor the new checkpoint is promoted as
parity-validated. The evidence establishes concrete outcome and movement gaps,
and a runnable physical reference; a reliable authentic fighter remains unmet.
