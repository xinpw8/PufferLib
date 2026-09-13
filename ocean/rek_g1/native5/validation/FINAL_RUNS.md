# Final native build execution

Executed on `spark-4ae3`, NVIDIA GB10, 2026-09-13 UTC. Native trainer source:
public commit `773f923d80e73bdc255a2ba730c918b28e416aa1`, with the checked-in
action-mask and pre-PPO rollout-status hooks. No Python, Torch, or ORT is linked
by or invoked from the training executable. Native MuJoCo is a startup model
loader; the simulation backend is CUDA Puffysics.

Executable on Spark:

```
/home/spark-advantage/rek-training/native5-rek-20260913-v1/build-final-v3/puffer-rek-native5
SHA256 c5c328a0c70b54415b91772a75aac2a46229cb347343b655052d3b4868cecde6
```

| Run | Exit code | Completed PPO epochs | Completed training transitions | Final interval SPS |
| --- | ---: | ---: | ---: | ---: |
| `final-512` | 0 | 3 | 24,576 | 5,461.06 |
| `final-4096` | 0 | 3 | 196,608 | 12,032.67 |
| `final-extended` | 134 | 4 | 32,768 | Invalid fifth rollout rejected |

The first two runs contain finite printed losses, real environment transitions,
and native PPO/Muon calls. Each completes the final status check and writes its
checkpoint privately on Spark. No completed rounds occur (`env/n=0`), so no
winning percentage or effective learned fighting behavior is established.

The extended run requests 163,840 transitions. After four finite-loss epochs,
the fifth rollout fails physics/observation/combat/scheduler checks. The final
hook aborts before the fifth PPO update. Its stdout contains four epochs and
no NaN loss; its stderr explicitly says `check completed rollout before training`.
The prior `train-extended-v1` output shows the old ordering reaching a fifth
NaN-loss epoch before aborting. This is an executed regression check of the
pre-PPO validation boundary, not evidence that the physics failure is fixed.

The final runs used `sweep.downsample=3`. Native history also appends its final
snapshot, so the first serialized metric bin averages epochs one and two and
the last snapshot is repeated. The final interval shown above is exact; the
first averaged bin is not an actual epoch or a separate trial. The reproduction
script uses five bins to retain the three short-run epochs individually, as in
the initial warm-throughput measurements. Changing this reporting option does
not change the compiled executable or training computation.

All commands, exit codes, stdout/stderr, configuration/metrics, executable
dependencies, source equivalence evidence, and build hashes are adjacent.
Raw model files, checkpoints, core dumps, and profiler databases are excluded.
