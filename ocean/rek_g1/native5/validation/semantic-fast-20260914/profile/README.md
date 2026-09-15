# Native reduced-state training profile

Actual host: `spark-4ae3`, NVIDIA GB10. Native PufferLib 5.0 trained the
`semantic_cuda_v1` candidate headlessly using 512 arenas, horizon 16,
minibatch 8,192 and 1,048,576 requested learner transitions. Training and
Nsight profiling exited 0. No Python or CPU physics ran.

The first complete PPO epoch is excluded. The remaining 127 epochs contain
1,040,384 learner transitions over 0.542016256 s on the captured timeline:
**1,919,470 profiled training SPS**. This short instrumented interval includes
environment execution, inference, PPO and between-epoch gaps. It is not an
exclusive-GPU or maximum-throughput benchmark. Use the separate unprofiled
runs for throughput comparisons.

The trace records exactly **2,032 learner action-sampling kernels and 2,032
fused environment kernels**, one environment kernel per batch. There are no
MuJoCo/Puffysics substeps or Newton iterations in this candidate.

| GPU kernel category | Summed time | Share of summed kernel time |
| --- | ---: | ---: |
| PPO training, including backward auxiliary stream | 199.408 ms | 57.125% |
| Learner inference and rollout I/O kernels | 96.167 ms | 27.549% |
| Fused environment step | 53.491 ms | 15.324% |
| Unclassified, one kernel | 0.007 ms | 0.002% |

Each fused environment kernel averaged 26.325 microseconds for 512 arenas.
It includes both fighters, input scheduling, slider motion, contacts, points,
knockdowns, resets and observation/pose export. This is an environment timing,
not another training-SPS denominator.

Classification uses verified disjoint streams: `fast_step` and `sample_logits`
identify the rollout stream, while `ppo_loss_compute` and
`mingru_scan_backward` identify the PPO and auxiliary backward streams.
The classifier rejects overlapping training/rollout stream ownership. The
exact environment-kernel duration is subtracted from the rollout stream.
This includes NVIDIA `nvjet` GEMMs missed by name-only GEMM matching.

Kernel union covers 63.344% of the measured wall interval. This describes the
target process's kernel timeline, not SM utilization. GPU kernel durations
can overlap across streams and must not be added to host API durations.

The native training wrapper, final round counters, logs, parameters, source
hashes, checkpoint hashes and ELF dependencies are under `run/`. Raw profiler
files, policies and models remain on Spark. `summary-final.json` uses the final
verified stream classifier; `final-analysis-hashes.txt` binds it to the
original trace and analyzer. The earlier capture command's initial analysis
was superseded by this CPU-only reanalysis, without rerunning training.

This candidate deliberately replaces articulated dynamics with an explicitly
approximate action-level model. Its speed does not establish parity or a
superhuman policy. With environment work at 15.3% of GPU kernel time, further
end-to-end optimization now requires attention to the learner and rollout
pipeline as well as the environment.
