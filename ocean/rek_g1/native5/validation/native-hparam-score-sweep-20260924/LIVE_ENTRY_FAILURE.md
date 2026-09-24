# Actual REK entry attempt, 2026-09-24

At 17:06:38 UTC, read-only inspection confirmed that the retained old-Box64 client had reported a native private-practice capacity failure. The selected policy never received a gameplay observation and emitted no gameplay action. This attempt supplies no live policy-strength result.

## Frozen attempt identity

- Stage: `/home/spark-advantage/rek-training/native-hparam-live-candidate-20260924-r1`.
- Attempt: `hparam-s1901`, one permitted attempt; no campaign relaunches.
- Candidate checkpoint SHA256: `fa6f760a5823d904b9f636df6421426e21344c03767ffa8ed0fe776c400b75ad`.
- Observation schema: `rek.native5.scaled_polar_xy.v1`; native CUDA BF16 sampled inference; all-ones feature mask SHA256 `59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b`.
- Isolated old-Box64 REK process: PID `3162378`, Linux start time `95712930`. It remained alive at the final inspection.
- Runtime directory: `/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-native-hparam-baseline-20260924-r1`.

## Recorded controller timings, UTC

| Time | Recorded event |
| --- | --- |
| 16:59:59.658 | First attempt started. |
| 17:00:00.698 | Matching v1 encoder/worker inference ready. |
| 17:00:00.782 | Exclusive control acquired. |
| 17:00:00.804 | Native observed intro skip accepted. |
| 17:00:01.039 | Existing logged-in state confirmed; Home observed. |
| 17:00:01.232 | FreePlay navigation accepted. |
| 17:00:01.447 | EnterSolo accepted with reason `private_practice_reservation_requested`. |
| 17:00:45.862 | Driver stopped with `private-practice entry timeout`. |
| 17:00:45.880 | StopG1PolicyStream accepted. |
| 17:00:45.900 | ReleaseExclusiveControl accepted. |
| 17:00:46.769 | Encoder, worker and relay exited cleanly, with no signals. Transport receipt confirms stream stopped and lease released. |
| 17:00:51.812 | Controller ended at its fixed attempt limit. |

The frozen summary records `source_count=0`, `predictions=0`, `applied=0`, no observed opponent, and no initial or final gameplay round. These are measurements from the completed attempt, not a claim about any later session.

## Native capacity evidence

File: `unity.log` in the runtime directory above, line **14824**:

```text
[KothScreenController] Solo find ended: No practice arena is free right now. Try again in a few minutes, or drop into an open arena.
```

Lines 14828 and 14829 identify the originating native path:

```text
REKApp.KothScreenController:EndSoloFind(String)
REKApp.<SoloFindLoop>d__93:MoveNext()
```

This message was present by the 17:05:16 UTC inspection and was reread at 17:06:38 UTC. The Unity line does not include its own timestamp, so the exact capacity-error time is unknown. Later log content includes native analytics heartbeat and flush activity. At 17:06:38 UTC, the log was 1,137,199 bytes with modification time 17:06:01.127 UTC.

The accepted EnterSolo response did not establish an allocated arena. The later native message identifies why private entry did not complete. No public/open-arena fallback was taken or proposed.

## Evidence hashes

Paths below are relative to `hparam-s1901/trial` in the attempt stage:

| File | SHA256 |
| --- | --- |
| `summary.json` | `6b4f99149196ad0851385e5aeb64dd742c48e228f0a2858a0c2f40869a6ca6ed` |
| `orchestrator.jsonl` | `50084169cd411d6018d6afee714f45d03498dd6e1624c6fc8dec0e555f5ebcbe` |

## Interpretation and bounded next step

The client remained responsive through cleanup. This differs from the previous upstream-Box64 process, which became unresponsive after an ArenaBootstrap load and could not confirm lease release. A sampled `pipe_read` wait alone is not evidence of a hang.

A single later same-client private retry has a concrete basis after the native completed-search error, provided the root's fresh native state check still shows isolated Lobby/FreePlay, no control lease and no policy stream. Preserve this failed attempt and use fresh output paths and the same checkpoint/seed. Let the existing EnterSolo command enforce its pending-search check. Do not write native search flags, change authentication, restart the client, or substitute public matchmaking. This diagnosis issued no relay command, UI action, capture, signal or GPU workload.
