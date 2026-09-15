# Compact v1 frozen-policy strength and observation diagnosis

These runs executed on `spark-4ae3`, NVIDIA GB10, with native C++/CUDA inference
and compact GPU simulation. No Python or CPU physics ran in this evaluator.
Private checkpoints, prepared model assets, and complete per-match records
remain on Spark. This directory contains aggregate stdout, commands, hashes,
exit status, and process provenance only.

All runs linked the exact v1 training runtime object. Evaluator build v2 adds
the diagnostic feature override; that build name does not mean compact physics
v2. Inference is BF16, while checkpoint coefficients are stored as FP32.

| Fixture | Matches | W/L/D | Win rate |
| --- | ---: | --- | ---: |
| 1M checkpoint, sampled | 1024 | 148/595/281 | 14.4531% |
| 33M checkpoint, sampled | 1024 | 642/137/245 | 62.6953% |
| 33M checkpoint, greedy | 2 | 2/0/0 | Two fixed fixtures only |
| Diagnostic 33M, sampled, observation 186 = 1 | 512 | 294/67/151 | 57.4219% |
| Diagnostic 33M, sampled, observation 186 = 64 | 512 | 510/0/2 | 99.6094% |

All five runs exited 0 with failure bits 0. Results include both fighter
assignments, scored only at the runtime's official terminal. A loss, draw,
and knockout are distinguished. The action space has no disconnect, quit, or
reset action. Greedy fixtures both ended in knockout at 686 ticks, 13.72 s,
with score 21 to 1 for the policy. Repeating them would duplicate fixtures.

Sampled batch streams vary Philox action draws. The current environment seed
does not randomize starts, so these trials are not diverse initial-state
tests. They do not establish authentic REK parity or human superiority.

## Diagnostic interpretation

The two interventions differ only in the constant placed in encoded policy
observation 186 after normal GPU encoding. The v1 runtime otherwise supplies
the cumulative round number, which increases throughout training. Both
interventions use the same frozen 33M checkpoint, 128 arenas, two rounds per
arena per side, sampled BF16 inference, seed 10001, actual masks, and the same
runtime GPU scripted opponent. There were no physics or checkpoint changes.

Changing the feature from 1 to 64 raises wins from 57.42% to 99.61%. This is
evidence of a training/evaluation distribution leak. Diagnostic outcomes
must never populate league rankings. The appropriate correction is stationary
episode observations and retraining, not selecting the flattering override.

The original diagnostic `command.txt` files contain the executable invocation
but predate explicit environment-prefix recording. `diagnostic-invocation.sh`
preserves the exact enclosing shell loop that set the two values. The final
JSON and every private match record identify the override explicitly. The
runner has subsequently been amended to record it in future command files.

## Measurement denominator

`execution_wall_seconds` measures frozen evaluation graph execution. It
excludes asset loading, checkpoint loading, graph construction, and process
startup. It is not training throughput. `training_sps` is explicitly null.
The process timing file separately reports whole-process execution. Other
unrelated GPU services were not stopped, so these timings are not isolated
hardware benchmarks.

Runtime source, prepared configuration, binary, checkpoint, and linked-object
hashes are recorded per run. Model and checkpoint paths are provenance, not
included payloads. No result from the parent's invalid `train --load_model_path`
attempt is included: that trainer command did not load the checkpoint.
