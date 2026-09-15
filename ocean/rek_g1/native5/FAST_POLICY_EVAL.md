# Native batched compact-policy evaluation

`fast_policy_eval.cu` runs a frozen native Puffer checkpoint against the compact
runtime's GPU scripted opponent. Both fighter assignments are evaluated. The
executable links the exact `fast_runtime.o`, `fast_assets.o`, `native_policy.o`,
and `cJSON.o` from the training build. It requests no rendered frames.

## Contract

The checkpoint is a flat FP32 storage array in encoder, decoder, MinGRU order.
Storage precision is distinct from inference precision. The trainer built
without `PRECISION_FLOAT` uses BF16 inference. A 223-input, 33-action plus
value-output, hidden-size-256, two-layer model has 459,008 stored floats:
1,836,032 bytes. The native loader verifies the explicit dimensions, exact
byte count, finite coefficients, and supplied SHA-256.

The evaluator uses `scaled_polar_xy` observations, the runtime's actual
33-category legal-action masks, and the native policy's recurrent-state reset
on terminal inputs. The action space has no quit, disconnect, reset, or UI
action. The opponent uses external override value 2, which invokes the same
scripted CUDA function on either side. Policy inference remains CUDA.

Runtime JSON settings override ambient compact-model environment variables.
Checkpoint hidden size and layer count default to 256 and 2 and can be supplied
as the final executable arguments. Sampling must explicitly be `sampled` or
`greedy`; inference must explicitly be `bf16` or `fp32`.

The current compact runtime does not randomize starting state from its seed.
Sampled evaluation varies the policy's Philox draws across batch rows and over
time. It does not test diverse environments. Repeated greedy trials duplicate
the fixed initial fixture and must not be counted as independent evidence.

## Execution

Building does not execute a GPU workload:

```sh
bash ocean/rek_g1/native5/build_fast_policy_eval.sh \
  /private/exact-training-build /private/new-evaluation-build
```

The runner creates a new private output directory, records executable/object,
configuration, and checkpoint hashes, and imposes a five-minute watchdog:

```sh
bash ocean/rek_g1/native5/run_fast_policy_eval.sh \
  /private/evaluation-build /private/runtime-config.json \
  /private/checkpoint.bin CHECKPOINT_SHA256 /private/new-results \
  128 4 10001 sampled bf16
```

This evaluates four completed rounds per arena on each side, for 1,024
recorded matches. Rounds end on the runtime's official terminal, including an
earlier knockout. Per-round records retain both scores, falls, winner, result,
side, RNG seed, arena stream, episode, and exact elapsed ticks. Device-side
collection occurs after every simulation tick, even inside captured graphs.
An arena that reaches its quota may continue executing while others finish;
subsequent rounds are excluded from that arena's recorded quota. Each side's
reported executed ticks includes this bounded surplus.

Only final records and aggregate progress are transferred to the host. The
measured execution wall time is frozen-policy evaluation time and is never
reported as training SPS. Any runtime, policy, mask, terminal, or collection
failure fails the run rather than producing successful strength statistics.

## Initial executed results

On Spark, both BF16 sampled runs completed 1,024 matches without failure:

| Checkpoint training transitions | Wins | Losses | Draws | Win rate |
| --- | ---: | ---: | ---: | ---: |
| 1,048,576 | 148 | 595 | 281 | 14.4531% |
| 33,554,432 | 642 | 137 | 245 | 62.6953% |

The 33.6M result was similar by side: 324/66/122 on fighter 0 and 318/71/123
on fighter 1. One greedy fixture per side instead won 21 to 1, with a knockout
at exactly 686 ticks, or 13.72 s. These are separate action-selection protocols.
Neither result establishes authentic REK parity or superiority to humans.

The native four-arena sampled probe's arena-0 scores matched the Node worker's
same-seed paired fixture on both sides: 1 to 1 draw, then 1 to 8 with the policy
on fighter 1. The initial league round robin completed all 12 matches with no
forfeit or invalid trial. Its eight games per policy are provisional, and
both checkpoint policies drew all four direct head-to-head games 0 to 0.

These frozen results should not be substituted for training-window win rates.
The initial runtime exposed cumulative round number as observation 186. That
value kept increasing during training but started at 1 in a fresh evaluation.
A controlled GPU intervention held all other inputs constant and replaced that
one encoded feature with either 1 or 64. Each run used 128 arenas, two rounds
per side, the same checkpoint, BF16 sampled inference, and seed 10001:

| Diagnostic observation 186 | Wins | Losses | Draws | Win rate |
| --- | ---: | ---: | ---: | ---: |
| Constant 1 | 294 | 67 | 151 | 57.4219% |
| Constant 64 | 510 | 0 | 2 | 99.6094% |

The intervention proves substantial sensitivity to the leaked cumulative
counter. It explains why a late training window can look much stronger than
a fresh evaluation. The correct repair is a stationary episode observation
and retraining. The stronger diagnostic is not a policy-ranking result.

`REK_EVAL_ROUND_FEATURE=64` enables this diagnostic GPU override after normal
observation encoding. It changes no physics, action mask, opponent, or policy
weight. Both match records and final output explicitly include `diagnostic`
and `round_feature_override`. Do not set it for production evaluations or
ingest its results into a league. The runner records the override in the
command for new runs; the initial two runs predated that command-recording
addition, and their exact invocation is retained with the aggregate evidence.

See [executed aggregate evidence](validation/compact-policy-20260914/README.md).

## Stationary v2 result

After making observation 186 episode-local and retraining 33,554,432
transitions, the frozen BF16 sampled checkpoint won 1,007 of 1,024 matches
(98.3398%), with four losses and thirteen draws. Side 0 produced 504/3/5 and
side 1 produced 503/1/8. No diagnostic override was enabled. Each side's one
greedy fixture won 21 to 1. All runs exited 0 with failure bits 0.

The same limitations remain: fixed arena starts, a single scripted opponent,
and compact approximate physics. High wins against this opponent do not
establish superhuman play, authentic REK parity, or generalization to new
opponents. See [v2 evidence](validation/compact-policy-v2-20260914/README.md).

The separate v2 league completed 60 side-paired matches with no invalids or
forfeits. The 33M sampled checkpoint ranked first with 19 wins, zero losses,
and 21 draws; the scripted opponent ranked second, and the 1M checkpoint
third. Each played 40 games. All 20 trained-versus-trained games ended 0 to 0.
The trained policies therefore still have an important pursuit/generalization
weakness against opponents that do not approach like the scripted bot.

All full match records and checkpoints remain under the private Spark directory
`/home/spark-advantage/rek-training/semantic-fast-20260914-v1`. Aggregate evidence
can be published without including model or checkpoint payloads.
