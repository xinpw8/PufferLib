# f147 live cohort: intentional runtime-repair stop

The old cohort finished **1 win, 1 loss and 1 tie, 34:50 points**, across three completed native 120-second rounds. It was intentionally stopped at the user's request to remove per-round process kills and repair runtime behavior. The 18/20 criterion was neither reached nor failed. This is not policy-based early selection, checkpoint acceptance or evidence of improvement.

Behavior checkpoint: `f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84`. Original worker, full 223-column balance8 observations, all-ones mask and all 17 permitted attacks were retained. Native round duration is 120 seconds; actual controlled starts were approximately 118.70 to 118.78 seconds remaining.

| Closed attempt | Result | Score | Ordinary points (awards below 5) | +5 award counts | Applied / predicted |
| --- | --- | ---: | ---: | ---: | ---: |
| refresh-s1501 | Loss | 5:22 | 0:7 | 1:3 | 2875 / 2876 |
| refresh-s1502-retry3 | Win | 16:15 | 11:5 | 1:2 | 2923 / 2924 |
| refresh-s1503-retry2 | Tie | 13:13 | 8:13 | 1:0 | 3015 / 3016 |
| Total | 1W 1L 1T | 34:50 | 19:25 | 3:5 | 8813 / 8816 |

Strict validation passed for all three completed rounds: 8,835 received-referee observations and 40 independently decoded native score receipts matched exact bridge scorer/counter increments and terminal totals. The three terminal request rejections remain recorded. There were 184 locally accepted attack requests; category 17 had zero. Local acceptance or an `ExecuteMove` return is not proof of server execution, contact or causal scoring.

Three incomplete attempts remain in the record: two unsupported G1/T800 pairings, and `refresh-s1503`, interrupted by the reported CoreCLR crash at 12:12 with 35.628323 seconds remaining. Their partial results are excluded from completed-round totals. Completion-selection bias remains.

## Recovered tie and controller retirement

The recorder child for `refresh-s1503-retry2` completed successfully while its outer controller was stopped. Before retirement, `/proc` recorded child PID 3081568 as a zombie of parent PID 3051691 with exit status 0. The child's own summary reports the completed 13:13 tie at 11:53:01.472 UTC.

Only the exact stopped outer controller was terminated, at 11:59:46.846 UTC. The game and process group were untouched. The third ledger record was recovered at 12:01:04.405 UTC from the saved process exit evidence, summary and attempt provenance. Parent wrapper stdout may have an unread pipe tail; child-owned summary, media and referee captures remain preserved. The archive includes the original and recovered ledger, retirement source and evidence. New persistent-runtime trials are a separate stage and are not included here.

## Analysis accounting correction

The first analysis stopped at `action_summary_mismatch`. Its source and preparation receipt are retained unchanged. The old analyzer compared the summary action histogram with accepted-only requests. In the winning round, action 21 has 34 accepted ACKs plus one terminal `policy_stream_not_owned` rejection, yielding the recorded summary count of 35.

The isolated R2 derivative checks every attack category's total recorded ACKs against the summary and preserves accepted-only and rejected statistics separately. No native packet, referee, recipient, counter or terminal-total guard was removed. The raw cohort was unchanged. [Compact results](receipts/live-outcome.json) include hashes and the exact accounting correction.

## Recoverable archive

NAS directory: `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\timing500-ppo-refresh-live-20260924-r1-retired-20260924T115946Z`.

- `evidence.tar.gz`: all 199 closed-cohort files, including scripts, configurations, logs, resource samples, videos and retirement/recovery provenance. SHA256 `87bc1a47213a783a4f3d43c72a8464e871577f06f4cd6788da518cb7d3b718e6`.
- `native-captures.tar.gz`: exactly three strictly selected external native captures. SHA256 `405a98ad0e621e10a0286cbce1d059880e6c8f39d41f96565e5372b197606c3a`.
- `analysis.tar.gz`: nine exact files covering failed R1 and completed R2 analyses plus their native packet/referee modules. SHA256 `128063ed00be8510ac5779471d973d2f45f70833003c6ab54a7a04c901cf86f9`.

All archive listings and NAS readback hashes passed. Every source, analysis and native-capture hash matched before and after archiving. The three native files total 616,840,334 uncompressed bytes. [Archive receipt](receipts/live-archive-receipt.json) records exact members, hashes, sizes and commands. Source files were neither removed nor modified. No raw captures, videos, weights or binaries are committed in this follow-up.
