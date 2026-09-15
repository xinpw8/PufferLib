# Why the frozen policy fails against an AFK opponent

Both final 33M checkpoints fail to approach and score against a neutral opponent under the original 20-second training duration. The high scripted-opponent win rate does not establish competent general fighting behavior.

## Controlled native GPU results

Each row contains 1,024 matches: 128 arenas, four rounds per arena on each fighter side, frozen BF16 sampled inference, action RNG seed 10001. Initial geometry is fixed. Only the opponent action source changes for the neutral intervention. No runtime equations, checkpoint weights, or observations were modified.

| Runtime / policy | Configured round | Opponent | Wins / losses / draws | Policy points | Zero-scoring games |
| --- | ---: | --- | ---: | ---: | ---: |
| V2 | 20 s | Neutral | 0 / 0 / 1,024 | 0 | 1,024 |
| V2 | 20 s | Scripted | 1,007 / 4 / 13 | 16,001 | 0 |
| V3 | 20 s | Neutral | 0 / 0 / 1,024 | 0 | 1,024 |
| V3 | 20 s | Scripted | 1,021 / 0 / 3 | 4,826 | 2 |
| V2 | 300 s | Neutral | 7 / 0 / 1,017 | 45 | 1,017 |
| V2 | 300 s | Scripted | 1,024 / 0 / 0 | 21,545 | 0 |
| V3 | 300 s | Neutral | 21 / 0 / 1,003 | 304 | 1,003 |
| V3 | 300 s | Scripted | 1,023 / 1 / 0 | 85,737 | 0 |

All eight cases returned exit 0 and zero failure bits. Both 20-second scripted controls reproduced the previously measured W/L/D exactly. Neutral actions use external override byte 1 with action category 1 on the opponent row after each inference call. A device assertion checks those rows every tick; policy output rows remain untouched. Recurrent state resets on the same native terminal signals as the original evaluator.

V2 awarded synthetic knockdown bonuses and could end after three downs. V3 is points-only. Their score totals are not equivalent, and a configured 300-second V2 match can end early. The summary records actual duration ranges separately. Changing 20 to 300 seconds changes both episode length and timer inputs. This experiment does not isolate those two effects; AFK failure at 20 seconds already rules out the duration change as its sole cause.

## Direct mechanism observed through the human worker

The separate native worker trace used the exact human seed 73, fighter 0, and neutral opponent. In 20 seconds:

| Checkpoint | Attack entries | Ticks locked in moves | Root path length | Minimum root-to-root gap | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 | 8 | 96.6% | 0.017455 m | 1.797572 m | 0:0 |
| V3 | 9 | 98.4% | 0.003972 m | 1.798372 m | 0:0 |

The policies repeatedly enter canned attacks while still far from the target, spending almost the entire round locked in those moves. They barely translate toward the opponent. When the scripted opponent approaches, that behavior can score. Neutral opponents expose the missing approach behavior. These traces support dependence on the scripted opponent's approach, not a claim that every possible failure mechanism has been excluded. The 300-second worker traces also scored 0:0, with 108 and 109 attack entries respectively.

## Evidence and boundaries

- `summary.json`: complete eight-condition derived results, both-side totals, zero-score counts, actual duration ranges, checkpoint hashes, and private-record hashes.
- `worker-trace/summary.json`: independent native human-worker measurements.
- `worker-trace/rek-afk-worker-trace.cjs`: source of that worker diagnostic.
- `afk_policy_eval.cu`: isolated copy of the frozen evaluator with neutral-row injection and verification. It does not replace the production evaluator.
- `run.sh`: exact build/run matrix, linked against preserved V2 and V3 native objects.
- `reduce.cjs`: offline reduction of saved match records, without physics or model execution.

All model inference and simulation execution were native CUDA on `spark-4ae3`. No Python, CPU physics, production input, deployment, or retraining occurred. Raw private match records and checkpoints stay on Spark. Timings are recorded for provenance only; overlapping diagnostic work means no throughput claim is made here. Both compact runtimes remain approximations, without authentic REK parity certification.
