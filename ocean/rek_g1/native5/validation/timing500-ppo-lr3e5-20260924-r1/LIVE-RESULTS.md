# Closed 7001 live outcome

The LR3e-5 candidate closed at 2026-09-24T11:25:21.524Z with **2 wins, 3 losses, 47:60 points**. It failed the predefined 18/20 criterion and stopped after its third completed nonwin. The checkpoint remains `7001cee36887ef5e168726576e0d5f07282628eb2c80a87d12e9e9f87e05ff5c`; this result does not establish an improvement over the LR1e-5 sibling's 2W3L, 49:55 outcome.

Eight attempts produced five completed 120-second rounds. All completed native captures passed received-referee validation, independently decoded native score-packet checks, exact scorer/counter joins to bridge increments, and terminal-total reconciliation. The audit covers 15,057 referee observations. [live-outcome.json](live-outcome.json) retains all eight attempt records, per-round source identities, point awards and request counts.

| Completed attempt | Score, own:opponent | Own +1/+2/+5 award counts | Opponent +1/+2/+5 award counts | Accepted attack requests |
| --- | ---: | ---: | ---: | ---: |
| lr3e5-s1401-retry3 | 15:9 | 5/0/2 | 3/3/0 | 90 |
| lr3e5-s1402 | 11:8 | 4/1/1 | 6/1/0 | 81 |
| lr3e5-s1403-retry2 | 12:13 | 5/1/1 | 1/1/2 | 88 |
| lr3e5-s1404 | 4:11 | 4/0/0 | 6/0/1 | 65 |
| lr3e5-s1405 | 5:19 | 5/0/0 | 6/4/1 | 79 |
| Total | 47:60 | 23/2/4 | 22/9/4 | 403 |

Ordinary +1/+2 awards account for 27:40 points; four +5 awards per side account for 20:20. These are received scoring events, with no inferred fall labels or attribution to an executed attack.

## Incomplete attempts

| Attempt | Recorded end reason | Last partial score | Native round time remaining |
| --- | --- | ---: | ---: |
| lr3e5-s1401 | Unsupported local G1/opponent T800 pairing | No admitted round | Unavailable |
| lr3e5-s1401-retry2 | Source stream missing; REK CoreCLR crash confirmed in runtime r85 | 5:0 | 59.428318 s |
| lr3e5-s1403 | Child exit; REK CoreCLR crash confirmed in runtime r88 | 10:12 | 35.626118 s |

Partial scores remain recorded but are excluded from completed-match totals. Both crashes happened during a round. Completed-only performance therefore has completion-selection bias. These five completed development rounds do not support a causal learning-rate claim or a generalization claim.

## Local requests

The completed rounds contain 15,026 predictions, 15,021 locally applied acknowledgments and five rejected acknowledgments. All 403 attack acknowledgments were locally accepted; all corresponding local `ExecuteMove` calls returned true. Neither fact proves server execution, contact, or that a particular request caused a later scoring event.

Policy action 21 accounts for 168 attack requests; action 23 accounts for 115. The remaining counts by action are: 16=7, 17=1, 18=7, 19=0, 20=2, 22=0, 24=12, 25=17, 26=36, 27=1, 28=1, 29=1, 30=35, 31=0, 32=0. Action 17/native move 7 is Left Front Kick, requested once; action 16/native move 6 is Left Side Kick, requested seven times. The JSON retains each action's native move mapping per round.

## Private evidence and publication limits

The closed cohort's 247 files, including MP4s, and exactly five validated external native captures are preserved under:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\timing500-ppo-lr3e5-live-20260924-r1-closed-20260924T112521Z`

`evidence.tar.gz` is 213,101,993 bytes, SHA256 `720ecfff13d5d4db1040b9536f2735c6a9d758b8b2c1084055ec65ef04e811e6`. `native-captures.tar.gz` is 151,699,439 bytes, SHA256 `29e28d3bba951b8815472f1f43c4165242db04610a977b18cc99e7a809a7cd5e`. Source/native hashes remained unchanged before and after archival; both NAS readbacks matched. The separate strict-analysis report is referenced by path and hash, not embedded as raw observations. See [the live archive receipt](receipts/live-closed-archive-receipt.json).

This publication contains derived counts and provenance only. Raw captures, register/stack logs, checkpoints and proprietary binaries remain private. The subsequent nonscoring Box64 diagnostic is a separate experiment and supplies no outcome or training data here. Earlier README statements describe their stated preparation/publication cutoff; this file records the final live outcome.
