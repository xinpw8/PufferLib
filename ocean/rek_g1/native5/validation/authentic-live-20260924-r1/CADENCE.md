# Recorded source and recurrent-policy cadence

Read-only CPU log analysis, 2026-09-24. No GPU, game, bridge connection, physics test, or runtime changes. Existing committed reports and receipts are unchanged.

## Measurement

The five authentic PPO training rounds contain 28,730 source observations and 28,715 encoder-ready/worker decisions. Across their five nominal 120-second rounds, those counts correspond to **47.883 source observations/s and 47.858 decisions/s**. This explains the earlier approximate 47.8 Hz description.

For precise QPC cadence, the table below uses `(N-1)/(last timestamp-first timestamp)` within each observed series. Startup/terminal coverage differences mean these rates differ slightly from `N/120`. A worker decision is matched to its source observation by round identity and sequence; its timestamp is that source's QPC time, not an invented GPU completion timestamp.

| Round | Source N | Source Hz | Decision N | Decision Hz | Decision mean interval, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| noprior-s601 | 5,731 | 47.968 | 5,728 | 48.197 | 20.748 |
| noprior-s602 | 5,793 | 48.346 | 5,790 | 48.372 | 20.673 |
| noprior-s603-retry2 | 5,727 | 47.927 | 5,724 | 48.141 | 20.772 |
| noprior-s604-retry4 | 5,730 | 47.922 | 5,727 | 48.098 | 20.791 |
| noprior-s605-retry3 | 5,749 | 48.076 | 5,746 | 48.282 | 20.712 |
| authentic-s801-retry9, S2W0 | 3,104 | 25.866 | 3,102 | 25.887 | 38.630 |
| authentic-s802-retry2, S2W0 | 2,943 | 24.536 | 2,941 | 24.572 | 40.696 |

All seven rounds have an exact 1:1 ordered match between encoder-ready observations and worker action responses. Worker decision indices are consecutive from 1. Encoder provenance QPC equals the matching source QPC for every ready row.

Pooled results combine only within-round intervals, excluding gaps between rounds:

| Decision interval distribution | Original five | s801-retry9 | s802-retry2 | Current two pooled |
| --- | ---: | ---: | ---: | ---: |
| Hz | 48.218 | 25.887 | 24.572 | 25.230 |
| Mean, ms | 20.739 | 38.630 | 40.696 | 39.636 |
| Median, ms | 18.527 | 36.951 | 38.627 | 37.668 |
| 95th percentile, ms | 34.376 | 52.426 | 55.775 | 54.536 |
| 99th percentile, ms | 38.027 | 60.740 | 63.653 | 62.239 |
| Maximum, ms | 152.604 | 90.638 | 119.185 | 119.185 |
| Intervals above 40 ms | 0.735% | 26.217% | 39.966% | 32.908% |
| Intervals above 50 ms | 0.331% | 7.159% | 11.259% | 9.154% |

Pooled source cadence is 48.048 versus 25.201 Hz. Independently, local applied-action return QPC gives 48.257 versus 25.240 Hz. Thus the shift appears in source sampling, ready/decision timing, and local return timing. These are not three independent experiments.

Four original rounds also contain a startup source-interval rejection above 250 ms; all seven contain derivative warmup. Those startup outliers do not explain the steady distribution shift: the current ready-observation median approximately doubles, even though its maximum is lower than the original pooled maximum.

## Interpretation and comparison limit

Current recurrent update rate is **52.32%** of the original data's rate: 47.68% fewer updates per second. Mean elapsed time per decision is **1.911 times** greater. The worker advances the recurrent policy once per nonterminal step request; it does not automatically insert missing intermediate updates when source sampling slows. See `ocean/rek_g1/native5/live_policy_worker.cu:130-145,182-194` for recurrent reset, one captured policy step, and one inference/decision increment per request.

Consequently, a given number of recurrent steps now spans almost twice as much elapsed time. Physical pose changes, action opportunities, repeated held-action decisions, and elapsed motion between recurrent updates have a different time-step distribution from the training trajectories. QPC-based velocity estimation or busy-duration aging does not itself normalize the recurrent-state update cadence. These observations identify a train-to-deployment timing distribution shift; they do not quantify its causal effect on score or prove it caused the observed attack bearings.

These measurements concern received observations and policy decisions, **not the game's physics frequency**, which this analysis does not measure. They also do not isolate the barrier setting's causal performance effect from every other difference between historical and current runs.

The balance8 candidate reuses the same five original training rounds with eight additional features. Adding those features does not resample the trajectories or correct their approximately 48 Hz recurrent training cadence. Comparing its current S2W0 results to historical approximately 48 Hz results cannot identify an eight-feature effect separately from the changed runtime cadence and other changed conditions. A contemporaneous baseline/candidate comparison under the same S2W0 runtime, with each round's actual cadence reported, controls that runtime difference more closely; the training-to-deployment cadence mismatch remains for both.

## Inputs and calculation

Original trial logs are reached through `/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/evidence/<round>/trial/`, whose symlinks resolve to `/home/spark-advantage/rek-training/f7-no-kick-prior-live-20260924-r1/<round>/trial/`.

Current logs: `/home/spark-advantage/rek-training/authentic-ppo-live-20260924-r1/<round>/trial/`.

For each of the seven rounds, the scan read `relay.stdout.jsonl`, `encoder.stdout.jsonl`, and `worker.stdout.jsonl`. File size, modification time, and inode were unchanged across each read; SHA256 was calculated. No full raw logs are added to Git. Local read-only scan source: `C:\rekagent\work\authentic-live-failure-analysis-20260924-r1\cadence_probe.py`, executed through SSH stdin without creating a remote script.

Source timestamps are `clock.qpc_ticks / clock.qpc_frequency_hz`; ready timestamps are `provenance.source_qpc_ticks / provenance.source_qpc_frequency_hz`. Positive adjacent deltas yield interval statistics. Pooled rate is total interval count divided by total within-round duration. Percentiles use sorted index `floor((n-1)*p)`; medians use the ordinary midpoint average for even counts.

Relay source SHA256 identities:

```text
noprior-s601         b5ccd0689153e5ce159edb0ffd0b160298970a30c79d48a4e209496d3fbaccb4
noprior-s602         b397f1f6e48638287d2218e6c7969f8f07fed608df0ac59ee55182241f9da5b4
noprior-s603-retry2  ce28320536a6ad0e3a3ad64c24822930000f7afd291ed4c4634ef91f28712450
noprior-s604-retry4  ae41d1d32e45dd6d93665ef787171ca90a73412e4b0030ab0c6e5b91b5de4e9f
noprior-s605-retry3  2fbb3d359f37d21980bc9ac2f58cdcf733a5fe29d33de8eeb1d621571db55d06
authentic-s801-retry9 25533ebd37e4b823cf51b5933345523d5fcbd60ca55bab6a77b80ef40ee31cfc
authentic-s802-retry2 a7bb79c6d1090bb9ac7d8954d92c3bea8f46a6bf5b7318c4ab4ea376be6c6538
```
