# Stationary compact v2 frozen evaluation

After the causal observation-counter diagnosis, compact runtime v2 presents
round 1 in policy observation 186 for each independent round episode. The
cumulative round counter remains available separately for diagnostics. The
policy was retrained for 33,554,432 transitions before these frozen tests.

Checkpoint SHA-256:
`0fa325324083023d69f6e5b489792880e423b5b7c7bd06387eaf2c6007018fe2`.
Weights remain private. Inference is BF16, 256 hidden units, two recurrent
layers, 223 observations, 33 masked legal actions. The action space offers no
quit, disconnect, reset, or other escape from a match.

| Protocol | Side | Wins | Losses | Draws | Points for/against |
| --- | --- | ---: | ---: | ---: | --- |
| Sampled | Fighter 0 | 504 | 3 | 5 | 8184/588 |
| Sampled | Fighter 1 | 503 | 1 | 8 | 7817/577 |
| Sampled total | Both | 1007 | 4 | 13 | 16001/1165 |
| Greedy | Fighter 0 | 1 | 0 | 0 | 21/1 |
| Greedy | Fighter 1 | 1 | 0 | 0 | 21/1 |

The sampled result is 98.3398% wins over 1,024 completed matches. There were
128 arenas, four rounds per arena per fighter assignment, and Philox seed
10001. All results use fresh runtime resets, actual terminal transitions,
terminal recurrent-state reset, actual action masks, and the identical
runtime GPU scripted opponent on the other side. No diagnostic observation
override was enabled. Both runs exited 0 with failure bits 0.

Each greedy result is one fixed fixture; repeating it would not add
independent evidence. Environment seed does not randomize arena start state.
Sampled runs vary action draws, not environment geometry or opponent variety.
These wins do not establish authentic REK parity, superhuman ability, or
generalization beyond this opponent and configuration.

Execution occurred on Spark's NVIDIA GB10. The standalone C++/CUDA evaluator
links the exact v2 training `fast_runtime.o`, `fast_assets.o`, `native_policy.o`,
and `cJSON.o`. Policy inference, simulation, masks, and per-tick terminal
collection execute on GPU. Asset loading is offline CPU work, with no CPU
physics stepping and no Python runtime. Nothing is rendered in these tests.

Aggregate output, exact command, configuration/checkpoint/binary/object
hashes, exit code, and process timing are preserved here. Private full
per-match records and weights remain under Spark's
`/home/spark-advantage/rek-training/semantic-fast-20260914-v1` staging directory.
The staging directory's v1 suffix is historical: these runs explicitly link
`build-v2`, as recorded in their hashes.

`execution_wall_seconds` is frozen evaluation graph time, not training SPS.
It excludes loading and graph capture; `training_sps` is explicitly null.
Unrelated GPU services were left running. No speed claim should be derived
from these frozen evaluation timings.

## Within-backend paired league

The native interactive worker completed 60 matches using seeds 20001 through
20010, both fighter assignments for each pair of the sampled 33M checkpoint,
sampled 1M checkpoint, and scripted opponent. Every match reached an official
terminal. There were no forfeits or invalid trials. The v2 configuration hash
is `ef31abc5889635d7386a07a436596bf8553d7e2dd09b964b1541dfb1fc624f76`.

| Empirical rank | Opponent | Wins | Losses | Draws | Completed games |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 33M BF16 sampled | 19 | 0 | 21 | 40 |
| 2 | Scripted | 10 | 23 | 7 | 40 |
| 3 | 1M BF16 sampled | 4 | 10 | 26 | 40 |

Ranks use the 95% Wilson lower bound of wins over completed side-reversed
pairs. Their confidence intervals overlap. They are empirical ordering within
this small opponent pool, not a claim of statistically certain superiority.
The separate greedy variants remain unranked because they did not participate.

Against scripted alone, 33M won 19, drew 1, and lost 0; 1M won 4, drew 6, and
lost 10. All 20 checkpoint-versus-checkpoint games drew 0 to 0. This exposes
limited generality: the checkpoint policies exploit the approaching scripted
opponent and do not demonstrate effective pursuit against another similarly
trained policy. High scripted win rate does not imply strong self-play.

`league-summary.json` retains exact fixture seeds, sides, points, duration,
terminal disposition, head-to-head counts, and ranking statistics. Checkpoint
paths were removed from its policy entries; hashes and architecture remain.
`league-command.sh` records the executed command. The early 1M checkpoint hash
is `7c4507537df7a5b82d06caa72c5d1987f231706b94fdca38dca9f9e99916067e`.
