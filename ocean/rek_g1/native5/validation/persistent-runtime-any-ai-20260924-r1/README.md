# Closed any-AI runtime accounting

The original ledger contains **two completed Bot1 rounds: 1W1L, 15:19**. A separately verified native terminal packet recovers the next Bot1 round as **19:9**, completing that native match **2:1 with 34:28 points**. The final owned child subsequently completed **Bot3 round 1, 8:7** while its parent was paused. Original plan, ledger and summaries remain unchanged.

| Segment | Opponent | Native round | Result | Evidence category | Predictions / applied |
| --- | --- | ---: | --- | --- | ---: |
| any-s1801-retry4 | Bot1 | 1 | 8:13 loss | Original ledger complete | 3213 / 3212 |
| any-s1802 | Bot1 | 2 | 7:6 win | Original ledger complete | 3255 / 3254 |
| any-s1803 | Bot1 | 3 | 19:9 win | Separate native terminal reconstruction | 3201 / 3200 |
| any-s1803-retry5 | Bot3 | 1 | 8:7 win | Owned child plus parent-retirement proof | 5956 / 5956 |

There were **10 executed attempts**: these four controlled segments and exactly **six zero-action timeouts**, comprising two private-AI-ready bootstrap timeouts and four active-round timeouts. All six have zero predictions, applied actions and rejections. Their summaries and orchestrators contain no explicit unsupported-pairing error, so their causes are not relabelled. The prepared `any-s1804` directory has no executed attempt. The original ledger has nine entries because it omits the final paused-parent child.

## Terminal transition and retirement

For `any-s1803`, the last policy source remained active at 19:9 with 0.0741855 seconds left. The next native received packet and applied postfix record inactive round 3 at zero seconds, `WonByPoints` for slot 0, and match `WonByRounds` at 2:1. The same packet advances the displayed opponent to Bot2. The driver then reports `policy_opponent_identity_changed`, followed by its generic `InvalidDataException` stream-end reason. Driver and wrapper exit 2 remain historical facts. The exact preceding-body join, native file hash, packet hash and line indices are preserved in `recovered-terminal.json`; no packet bytes are included.

The parent was retired at 13:41:42.939 UTC after the final child completed. Encoder, worker and relay had closed at 13:40:43.661 UTC, all exit 0 without signals. Wrapper kernel exit was 0. Only the paused parent received SIGTERM followed by SIGCONT; the game and children were not signalled. The parent stdout tail may be unread. Child-owned summary, media and retirement evidence independently preserve the final result.

## Limited stability observation

Saved process receipts identify the same REK PID 3122697 and start ticks 94319512 at startup and the 13:42:36 cutoff. The four controlled segments span **479.1626 seconds** of observed active policy-source intervals. A prior, separate Bot2 segment adds 119.6345 seconds, yielding 598.7970 seconds of cumulative observed intervals. Process lifetime is excluded from controlled-exposure claims. The saved cutoff reports no original signal, access violation, null-dispatch, CoreCLR fatal or fail-fast event. This exposure does not establish a crash fix.

The three Bot1 source rates were 26.82, 27.17 and 26.74 Hz; the later Bot3 segment was 49.72 Hz. These measured cadence differences and opponent differences prohibit treating the segments as a uniform policy-performance cohort. No policy promotion is claimed.

## Media and archive

All four MP4s passed the existing recording/decode checks and remain below 20,000,000 bytes: **10,879,187; 10,903,303; 10,930,745; 10,919,462**. Exact video hashes are in `runtime-results.json` and `archive-receipt.json`.

The verified private NAS archive is `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\anyAI-retired-20260924T134142Z`. It contains 255 unchanged campaign files plus the exact closed native round-3 capture in a separate archive. Full NAS hash readback and exact archive member checks passed. Active game and EventPipe files were excluded. This compact report includes only scalar results, hashes and saved process receipts; raw captures, videos and binaries remain private.
