# V3 points-only frozen evaluation

Both runs executed on Spark using an evaluator linked to the exact V3
training objects. Checkpoint SHA-256:
`e209edd0d1301170c5253f622f5f11aabccb57daf7fd4117983942bccb989fc6`.
Inference is BF16, 223 observations, 33 masked actions, hidden size 256,
two recurrent layers. The full checkpoint remains private.

| Action selection | Side | Wins | Losses | Draws | Points for/against |
| --- | --- | ---: | ---: | ---: | --- |
| Sampled | Fighter 0 | 511 | 0 | 1 | 2440/526 |
| Sampled | Fighter 1 | 510 | 0 | 2 | 2386/526 |
| Sampled total | Both | 1021 | 0 | 3 | 4826/1052 |
| Greedy | Fighter 0 | 1 | 0 | 0 | 7/1 |
| Greedy | Fighter 1 | 1 | 0 | 0 | 7/1 |

The sampled policy wins 99.70703125% of 1,024 completed matches. Each side
uses 128 arenas and four rounds per arena. Policy seed is 10001. Inference
state resets at terminal inputs, legal-action masks are applied, and the
opponent is the same GPU scripted implementation used during training.
No diagnostic override is enabled. Both runs exited 0 with failure bits 0.

V3 removed the synthetic cumulative-damage knockdown/reset mechanism. Hits
score points without resetting positions. Physical knockdowns are not modeled,
so zero reported falls are a consequence of that limitation. Points determine
timed-round winners. These are fresh V3 results, not V2 result reuse.

Environment seeds do not randomize starts or geometry. Sampled action streams
vary across arenas and time; repeated greedy runs would duplicate the same
fixed fixture. Neither result establishes authentic REK parity, human-level
skill, or strength against other trained opponents. The new V3 league is
empty and no rank is inferred from these scripted-opponent tests.

Aggregated stdout, command, executable/object/checkpoint/configuration hashes,
process timing, and exit status are preserved here. Full per-match records,
weights, and prepared assets stay private on Spark. There is no Python runtime
or CPU physics stepping in this evaluator. Asset preparation is offline and
rendering is disabled.

`execution_wall_seconds` measures frozen evaluation graph execution, not
training SPS. It excludes loading and graph construction. `training_sps` is
explicitly null. Unrelated services were preserved; these evaluation timings
are not isolated hardware benchmarks.
