# Authentic KO scoring and compact-training target gap

Read-only audit of `live-cadence-r1`, performed 2026-09-17 UTC. No training,
game input, GPU job, or production-code change was performed for this audit.

## Finding

The authentic 13:24 loss contains four opponent +5 score updates. Each occurs
while the policy fighter is low and heavily tilted, then both fighters return
to their spawn positions within 0.220 to 0.276 s. Recovered native code explicitly
awards 5 KO points through the same `CleanHits` counter and resets both fighters
while continuing a round when the fallen fighter cannot get up.

The best-supported decomposition is policy **13 ordinary hit points**, opponent
**4 ordinary hit points plus 20 KO points**. This is supported by the native rule
and four measured down-pose/reset sequences, not by treating every +5 update as
proof of a KO. The captured stream does not contain authoritative `ScoringEvent`
reasons. It cannot exclude an unobserved same-update scoring coincidence or
identify which attack caused each fall.

Any earlier interpretation of this round's zero `falls` or false `knockout`
fields as evidence that no knockouts occurred is incorrect. Those client fields
must not be used for that conclusion.

## Recovered source

Source directory: `C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp`.
Native disassembly was checked alongside the recovered field/method labels.

- `FightCoordinator.txt:75615`: constructor sets `koPoints = 5f`.
- `FightCoordinator.txt:40096`, `ResolveCountExpiry`: single-fighter count-out
  calls `PointTracker.RecordRefereeAward` with `this.koPoints` (native call
  `0x18239CA10`, award value loaded from `[this+0xF0]`). It emits the format
  `Fighter {0} counted out -- KO, +{1:F0} to fighter {2}.` at line 40733.
- The non-terminal branch calls `ContinueRoundAfterKnockout` at line 40752.
  That method, starting at line 41146, logs `Faller cannot get up -- KO points
  stand, fighters reset, round continues.` and calls `ResetBothToSpawn` if time
  remains. `ResetBothToSpawn` starts at line 74657 and resets both robot slots.
- `PointTracker.txt:1362`, `RecordRefereeAward`: adds integer award points to the
  same round score array used by `RecordHit`. `CleanHits` is consequently not an
  exclusively hit-derived counter. Ordinary hit awards are 1 or 2 points.
- `ResolveCountExpiry` sets `RoundState.KnockoutOccurred` in its terminal KO
  branch, after the continuation branch has already returned. A continuing
  KO/reset round can finish `WonByPoints` with `knockout=false`.
- The bridge exports `round.Falls` and visual-only `Robot` getter values directly
  (`Plugin.G1PolicyStream.cs:376`, fighter export near line 395). The inspected
  snapshot score-update block writes score and terminal-result fields. An
  authoritative fall-count assignment was not identified there. The precise
  reason the client fall counters remain zero is not established by this audit.

Source SHA-256:

```text
FightCoordinator.txt 9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5
PointTracker.txt     68e85604eb0bca59f9aef28f29df41f055ae2f8474fa8d6a1ada617afdce189a
```

## Exact recorded events

Private evidence is
`/home/spark-advantage/rek-training/policy-quality-20260916-r1/live-cadence-r1/trial/relay.stdout.jsonl`,
SHA-256 `ed963bce8bd851ebde4d56263e9b2d3eab449409c529f7bbf6ae97e01a8119ae`.
There are 5,781 state rows. QPC frequency is 10,000,000 Hz. Fighter 0 is the policy.
The checkpoint is `af84c4ec92e953e72c6bf4cde9ccb8cf80258a91dbf01947312e622dd492876e`.
Public run context: [summary](live-cadence-r1/summary.json) and
[provenance](live-cadence-r1/provenance.json).

| State sequence | Score-update QPC | Remaining s | Score before → after | Actor pelvis height m / tilt degrees | First bilateral spawn-return QPC | Lag s |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 2580 | 2900129156491 | 65.546364 | 10:3 → 10:8 | 0.18621682 / 83.86857 | 2900131736204 | 0.2579713 |
| 3615 | 2900342445001 | 44.240185 | 10:9 → 10:14 | 0.21611737 / 89.490585 | 2900344643682 | 0.2198681 |
| 4218 | 2900465964134 | 31.938114 | 10:14 → 10:19 | 0.17181066 / 90.69612 | 2900468719890 | 0.2755756 |
| 4707 | 2900565848005 | 21.935570 | 12:19 → 12:24 | 0.13862583 / 117.92358 | 2900568482212 | 0.2634207 |

Spawn return is a descriptive measurement: both roots within 0.025 m of
Unity X coordinates (-0.9, +0.9), |Z| < 0.005 m, pelvis Y > 0.75 m, and tilt
< 5 degrees. This criterion selects the first returned sample after each score
event; it is not a proposed knockdown detector. The reset samples are sequences
2592, 3626, 4231, and 4721. Their root heights are approximately 0.796 to 0.803 m.
The largest single observed reset displacement is 2.496 m in 0.0342 s. The
multi-sample pose interpolation is consistent with a replicated reset, not a
normal get-up trajectory.

| Score sequence | Most recent locally sent move index | Dispatch QPC | Dispatch age at award s |
| --- | ---: | ---: | ---: |
| 2580 | 3 | 2900125355771 | 0.3800720 |
| 3615 | 15 | 2900332179443 | 1.0265558 |
| 4218 | 6 | 2900446332710 | 1.9631424 |
| 4707 | 3 | 2900561451466 | 0.4396539 |

These are actual `g1_policy_dispatch` records with `send_method_returned=true`.
They establish local dispatch only. Native execution, completion, and server
acceptance are recorded as unknown. The most recent request can occur after a
fall has begun and is not causal attribution.

Across this complete round, actor score increments are five +1 and four +2;
opponent increments are two +1, one +2, and the four +5 above. Counters never
decrease. Both fighters' `falling`, `fallen`, and `resetting` getters remain
false, `round.falls` remains [0,0], and `referee` remains null even through these
visually evident resets. Both robots are `visual_only=true` throughout.

The exact owned-process log was also checked inside Docker `codexrook-xserver`:
`/opt/codexrook/wineprefix/drive_c/rekagent/live-transfer-20260916T233902Z-unity.log`.
Case-insensitive searches for `counted`, `knockout`, and `KO,` returned no lines.
Thus there is no captured native count-out log entry to attach. The source log
format above must not be represented as an observed runtime message.

## Training implication and bounded next measurement

`fast_runtime.cu:420` explicitly omits balance/fall dynamics. Its reward at line
436 is point delta, and its provenance declares constant upright state and
unmodeled knockdowns. Recovering Bot1's high-level choices and ordinary hit gates
does not introduce the authentic failure mode demonstrated here. A policy can
win the ordinary-hit exchange and still lose the authentic round through falls
and KO awards. The compact 152:49 score therefore cannot establish this missing
competence. Do not restore a synthetic "N hits implies knockdown" rule or invent
a tilt threshold as a training fix.

First measure fall/reset exposure and ordinary-point versus inferred-KO-point
decomposition on the already captured full rounds, retaining the evidence limits
above. A physical stability model requires a separately validated dynamics or
measured transition contract.

Exact live accepted-contact replay is also currently unavailable: both native
move identities are null, both motion-frame fields stay zero, and hit attribution
is null. Actual bone world positions, local rotations, and QPC timestamps are
available for a narrower geometry-only replay. That can test compact enclosing
spheres against recovered primitive geometry on observed poses, but it cannot
supply the missing native active-clip/apex phase or authoritative contact-enter
history.
