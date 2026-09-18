# Native awarded-point counter semantics

`RoundState.CleanHits[fighter]` is the cumulative integer awarded-point total
for that round. It includes clean-strike awards and referee awards. The source
names `round.clean_hits` and packet `new_hit_count` remain unchanged for
compatibility; neither name establishes a count of strike events.

`summarize_live_transfer.cjs` adds `awarded_points.initial_by_slot` and
`awarded_points.final_by_slot`, copied from the corresponding summary rounds.
When a summary round is absent, the first or last observed stream round supplies
the value. An unavailable or non-integer pair becomes `null`, without changing
the raw field. No across-round difference or strike-event count is inferred.

## Recovered implementation evidence

The pinned build's `PointTracker.RecordHit` and `RecordRefereeAward` both add
the point award converted to an integer by truncation toward zero to
`currentRound.CleanHits[fighter]`. The sender obtains this total with
`GetCleanHits` and serializes it as `ScorePacket.newHitCount`. The client receiver
assigns the packet total to the corresponding `CleanHits` element; its following
`FireNetworkScore` call publishes the point event without adding to the total
again. Thus totals `2, 4, 6, 11` are consistent with awards `2, 2, 2, 5`, and do
not mean eleven clean strikes. Aggregate counters alone do not establish the
cause of a particular award.

Evidence locations are private local files; no disassembly or binary payload
is included in this repository:

| Source | SHA-256 | Relevant locations |
| --- | --- | --- |
| `C:/rekagent/work/controller-audit-isil/IsilDump/REKApp/REKApp/PointTracker.txt` | `68e85604eb0bca59f9aef28f29df41f055ae2f8474fa8d6a1ada617afdce189a` | `RecordHit` lines 795-802; `RecordRefereeAward` 1403-1413; `GetCleanHits` 2567-2584; notification-only `FireNetworkScore` 2464-2509. |
| `C:/rekagent/work/controller-audit-isil/IsilDump/REKApp/REKApp/FightCoordinator.txt` | `9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5` | Sender 24454-24460; receiver assignment 36058-36070; notification 36100-36103. |
| `C:/rekagent/work/rek-current-interop-b20ca0d/cpp2il-dummy/REKApp.dll` | `efc1921e4727adbd6930a7c2c61d0301f6ce6faecb3bb388114aada5d9acd7b4` | Recovered type/field metadata used to resolve the offsets below. |

Verified offsets: `PointTracker.currentRound` is `0x20`,
`RoundState.CleanHits` is `0x20`, and `FightCoordinator.currentRound` is `0x318`.
The packed score packet contains `fighterIndex` (`Byte`) at `0x0`,
`newHitCount` (`Int16`) at `0x1`, and `pointsAwarded` (`Single`) at `0x3`.
The in-memory round total is an `Int32`; this note does not infer overflow
behavior beyond the recovered packet-width conversion.

## Compatibility limit

This correction changes summary labels and adds explicit score metadata. It
does not change the encoder binary, observation values, action selection or
live evaluation contract. Existing encoder manifests describe these counters
as clean hits and say the scoreboard formula is unknown. That metadata is stale
and remains unchanged pending a metadata-only rebuild.
Encoder counter-increase hit proxies remain proxies; referee awards and multiple
events can contribute to an increase.

Run the focused regression tests with
`node --test ocean/rek_g1/native5/summarize_live_transfer.test.cjs`.
