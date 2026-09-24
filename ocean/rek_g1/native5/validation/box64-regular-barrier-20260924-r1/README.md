# Box64 regular-barrier experiment, 2026-09-24

Setting `BOX64_DYNAREC_WEAKBARRIER=0`, with `STRONGMEM=2` unchanged, completed one usable 120 s policy round in launch r35. That attempt produced 3,157 predictions and 3,156 applied actions, with a 6:22 loss. No CoreCLR crash signature was found in r35 stderr. The campaign then deliberately recycled the client. Its exit code 137 is therefore not evidence of a spontaneous crash.

## Observed timing

| Measurement | r34 / s702, weak barriers 1 | r35 / s703 retry2, regular barriers 0 |
| --- | ---: | ---: |
| Forwarded source observations | 5,726 | 3,158 |
| Source QPC span | 119.386726 s | 119.589756 s |
| Forwarded source rate | 47.953 Hz | 26.399 Hz |
| Unity frame counter rate | 54.755 Hz | 26.399 Hz |
| Ready observations | 5,724 | 3,157 |
| Ready observations/s | 47.945 | 26.399 |
| Applied actions | 5,723 | 3,156 |
| Applied actions/s | 47.937 | 26.390 |
| Median observation gap | 18.641 ms | 36.030 ms |
| p95 observation gap | 34.594 ms | 51.396 ms |
| p99 observation gap | 37.983 ms | 59.605 ms |
| Maximum observation gap | 315.669 ms | 152.339 ms |
| Gaps greater than 250 ms | 1, during startup | 0 |
| Final 30 s source rate | 48.441 Hz | 26.973 Hz |
| Final 30 s Unity frame rate | 55.843 Hz | 26.973 Hz |
| Completed score, policy:bot | 2:11 | 6:22 |

The warmed r35 trial was usable and had lower observed cadence than the preceding baseline. Source rate was 44.95% lower and Unity frame counter rate was 51.79% lower in this pair. This is a sequential observation: policy seeds differed (702 and 703), and the completed trials used round 1 in a fresh process versus round 2 in an already-running process. These differences prevent a clean causal performance estimate.

Unity frame counter gaps in the baseline indicate frames without a forwarded policy observation. They do not establish dropped render frames or transport loss.

## Initial r35 attempt

The first r35 attempt produced three source observations, zero ready observations, and zero actions. Its first observation required derivative warmup. The following QPC intervals were 448.710 ms and 321.550 ms, both beyond the encoder's 250 ms limit. The action watchdog ended the stream at 05:14:09.5676468 UTC.

The later retry completed with no interval above 250 ms. The cause of the initial delay is unknown; these records do not identify JIT compilation or establish a persistent startup defect.

## Runtime scope and result

The installed Box64 commit was `2544543b3ace019bf2c9073a3afdb49b68d28bb1`, with Wine 11.13 and CoreCLR 6.0.7. The experiment added one Docker environment variable for the isolated X98 client. The installed Box64 documentation describes setting 0 as the regular safe barrier path; its source substitutes full `DMB ISH` barriers for selected weaker barriers.

The campaign recycled r35 at 05:18:23-05:18:24 UTC after the completed round. Launch r36 at 05:18:27 used the restored default weak-barrier setting 1. No manual client or controller termination was needed.

One surviving process and one completed round do not demonstrate that the repeated CoreCLR access violation is fixed.

## Measurement and evidence

`measurements.json` contains exact statistics, counts, source paths, runtime settings, and limitations. Cadence uses `clock.qpc_ticks` at 10 MHz from all complete records in each trial's `encoder.stdin.jsonl`. Ready counts come from `encoder.stdout.jsonl`; applied counts and results come from `summary.json`. Quantiles select index `floor(q * (n - 1))` in the sorted adjacent-interval list.

Records remain on Spark under:
`/home/spark-advantage/rek-training/f7-joint-mask-live-20260924-r1/`

Launch evidence remains under:
`/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-r34/`
and the corresponding `r35` directory.

This record contains no credentials, raw game state, game binaries, or global input.
