# Received referee observation contract

Schema `rek.g1_received_referee.v1`, added 2026-09-19. This is a read-only
observation extension to the existing 33-action G1 policy stream. Desktop,
account, authenticated-continuation, private-session, and gameplay guards are
unchanged. Policy encoders and compact simulation are not changed.

## Source and availability

`FightCoordinator.ApplyFightStateSnapshot` prefix copies exactly 33 bytes from
`FastBufferReader.GetUnsafePtrAtCurrentPosition` with `Marshal.Copy`. The reader
is never advanced or written. This is the same byte copy used by the existing
recorder. The native method runs unchanged. Postfix accepts the copy only if the
same private policy coordinator/network remains bound, its applied round/redo
matches the packet, and `clientRefSnapshotSeen`, `clientRefCallSeq`,
`clientRefCountMask`, and `clientRefCountSeconds` match native application.

The cache is bound to coordinator, network, fight, and round object identities,
fight epoch, round number, and redo state. Network lifecycle, native referee
replay reset, changed native mirrors, changed identity, malformed body, failed
native application/probe, or invalid clocks invalidate it. New applied packets
may establish a new lifecycle. Client mirrors alone never establish availability.

The freshness cap is **0.5 s of client receipt age**, five nominal 0.1 s snapshots.
This is an explicit engineering observation budget, not a recovered game rule.
At greater age, `available=false`, `reason=referee_receipt_stale`, measured fields
become null, and old data cannot be revived by a reversed clock.

## Exposed fields

| Fields | Meaning |
| --- | --- |
| `available`, `reason`, `observation_hooks_verified` | Receipt validity and diagnostics. |
| `source`, `provenance`, `authority_scope` | Received server-authored packet observed on client, not server-current hidden state. |
| `maximum_receipt_age_seconds`, `receipt_age_seconds` | Configured client freshness budget and measured age. |
| `receipt_sequence`, `lifecycle` | Bridge-local receipt and lifecycle identities; not server IDs. |
| `receipt_qpc_ticks`, `receipt_qpc_frequency_hz`, `receipt_unity_frame`, `receipt_unity_time`, `receipt_unity_unscaled_time` | Actual prefix receipt clocks. |
| `wire_body_sha256`, `wire_body_base64` | Exact copied 33-byte receipt for independent decoding. |
| `count_mask`, `count_seconds` | Unsigned wire bytes at offsets 25 and 26. Zero is exposed only if measured. |
| `slot0_count_active`, `slot1_count_active` | Bits 0 and 1 of measured count mask. |
| `call_available`, `call_sequence` | Sequence byte at offset 27; zero explicitly means no call. |
| `call_type`, `call_name`, `call_faller`, `call_points` | Bytes 28, signed 29, and 30. All null if sequence zero or receipt unavailable. Unknown numeric type retains raw value with null name. |
| `call_observation_sequence`, `call_sequence_transition`, `call_history_censored` | Client deduplication and censoring diagnostics. Repeated latched payload retains the same observation identity. |
| `packet_phase`, `packet_round_number`, `packet_round_active`, `packet_round_redo`, `packet_round_knockout_occurred`, `packet_round_result` | Round context from that same wire body, not a later local read. |
| `server_tick`, `server_time`, `server_fight_epoch` | Always null: packet contains none. |

Call names in native numeric order: Slip, SlipEStop, Knockdown, BeatCount,
Knockout, DoubleKnockdown, DoubleKnockout. Sequence zero is never a Slip.
Call type/faller/points stay latched across many packets. They are not a new event
per policy observation. Initial nonzero snapshots and snapshots after receipt
gaps or runtime resets are left-censored. Forward jumps greater than one,
decreasing sequences except 255 to 1, and changed payload without changed sequence
are explicitly censored. A 255 to 1
wrap is recognized without treating zero as an event. Client observation IDs do
not prove that no server calls were skipped between received snapshots.

Countout calls in G1 can award points and reset while the round continues.
`Knockout`/`DoubleKnockout` call names do not replace `packet_round_knockout_occurred`
or round/fight results. No attacker, hit, active clip, command acceptance, physical
event timestamp, or action causality is inferred. A late packet has no server
fight epoch; the contract proves its current client application binding only.

## Recorder coexistence and validation boundary

The bridge prefix runs at Harmony First priority and only copies bytes; the
recorder's own prefix sees the same unchanged reader. Bridge postfix runs at Last
priority and reads client mirrors. Its finalizer only invalidates bridge cache on
an existing exception and does not suppress or replace that exception. The reset
hook does not skip or modify the native reset. All observer paths catch their own
read/decoding errors. Native getters and Harmony IL2CPP runtime compatibility
still require an actual isolated startup receipt check; compilation and pure
tests alone cannot establish that runtime result.

Pure test project: `Tests/G1ReceivedRefereeTests/G1ReceivedRefereeTests.csproj`.
860 checks pass for exact offsets and bytes, signed faller, all byte-valued enum
inputs, hash/body identity, malformed snapshots, applied mirror checks, missing
receipt, clock regression, freshness boundary, reset and lifecycle invalidation,
sequence zero, repetition, wrapping, gaps, and uncertain history. The decoder
uses the recorder's audited offsets and recovered client replay semantics.

Pinned GameAssembly SHA256:
`6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`.
Pinned metadata SHA256:
`e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd`.
Interop REKApp SHA256:
`faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2`.
Plugin remains version 0.4.9; deployment must pin the new exact DLL hash.
No deployment, game interaction, or changes to generated bin/obj are included in
the implementation handoff.

Final build artifact:
`C:\rekagent\tmp\referee-bridge-20260919-r1\bridge\RekUiBridgeAgent.dll`.
SHA256: `ea8511a87b9e456547f13ffdb2f9af5e941b8b04f97bf40419d38e1c9b31ff3e`.
This build includes the forward-gap censoring fix and its tests. Build completed
with zero warnings/errors. Existing isolation/authenticated-continuation 5,930,
relay 6,447, and protocol 441 checks plus local pipe roundtrip also passed.
Generated bin/obj files were not staged; no commit was made by this subtask.
