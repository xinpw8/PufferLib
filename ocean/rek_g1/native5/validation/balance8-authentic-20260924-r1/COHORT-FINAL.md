# Balance8 live cohort: failed consistency criterion

The campaign ended at **2026-09-24 07:50:49 UTC** with **2 wins, 3 losses, 42:58 points** against private Sparring Bot 1 / difficulty 0. It stopped at the prespecified third nonwin, so 18 wins among 20 fixed labels was no longer possible. It did not beat the bot consistently.

| Seed | Completed attempt | Score, moogleod:bot | Result |
| --- | --- | --- | --- |
| 901 | balance8-s901-retry4 | 5:13 | Loss |
| 902 | balance8-s902-retry3 | 9:23 | Loss |
| 903 | balance8-s903-retry3 | 13:3 | Win |
| 904 | balance8-s904-retry3 | 10:9 | Win |
| 905 | balance8-s905-retry3 | 5:10 | Loss |

All five counted rounds were completed, fair-start, 120-second non-redo rounds under checkpoint `9a875c347b512bb46dde25481be75d276ea1b4c0799886b30aaf03be6ae1c5ce`. Eleven other attempts were outside that criterion; they remain in the ledger and archive. The private client was closed by its scoped lifecycle helper at campaign end.

Source stage: `/home/spark-advantage/rek-training/balance8-live-20260924-r1`. Ledger SHA256: `d086e522c50875d307852d9673f26a9056f2320324ac446f143a3eeb4db1bde9`.

Complete closed campaign archive:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\balance8-live-20260924-r1-closed\evidence.tar.gz`

183,920,094 bytes; SHA256 `93e6a12c69179451dc5aa5184b6b0eb9e3d3bb8f3dbf7bad39bfeaa30b641304`. Transfer exit 0, matching NAS readback, and separate tar listing exit 0 with 506 entries. Original files remain on Spark. This archive covers the campaign directory; external native recorder files retain their separately recorded Spark paths.

The first-three-round receipt in `LIVE-EVAL.md` remains an explicitly earlier cutoff. No later checkpoint's result is included here.
