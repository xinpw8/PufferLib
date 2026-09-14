# Frozen native policy evaluation

`native_policy.cu` loads frozen flat FP32 checkpoint files and executes the
recorded inference precision on CUDA. It does not use Python, Torch, an
optimizer, or CPU policy inference. The caller must explicitly choose the
observation encoding, precision, architecture, and legacy hidden activation.
Checkpoint SHA256 is checked when supplied and always available to the caller.

## Executed inference checks

Executed on `spark-4ae3`, NVIDIA GB10, 2026-09-14. Private results are under
`/home/spark-advantage/rek-training/native5-eval-policy-20260914-v1/validation-v3`.
No environment training or Windows input was performed by these tests.

| Oracle and tested mode | Compared logits | Maximum absolute error | Other checks |
| --- | ---: | ---: | --- |
| Actual PufferLib `773f923d` `arch_forward`, BF16 | 1,632 | 0 | Masked argmax and seeded Philox actions exact; terminal reset, explicit reset, graph capture, selected-row isolation, invalid mask rejection pass |
| Actual PufferLib `773f923d` `arch_forward`, FP32 | 1,632 | 0 | Same checks pass |
| Original legacy `models.cu` `policy_forward`, corrected checkpoint, FP32 | 9,520 | 0 | Greedy actions exact; recurrent reset at tick 64 passes |
| Original legacy `models.cu` `policy_forward`, combined checkpoint, FP32 | 9,520 | 0 | Same checks pass |

The first two tests use four selected rows in interleaved paired-fighter
buffers for 12 ticks. The legacy tests use four rows for 70 ticks. Tests load
the actual private checkpoints; no coefficients are printed or published.
These are inference tests, not environment parity, tournament results, or
evidence of training competence. Different GEMM batch sizes may select different
algorithms; the zero-error claim applies to these executed batch-four tests.

The legacy kernel uses a polynomial `fast_sigmoid` for negative hidden values.
Native5 uses an exponential sigmoid there. Set `legacy_fast_hidden=1` only for
the recorded legacy policies. The legacy source used by the oracle has hashes:

```
models.cu  0c4b5b27f80e41517eb0fa6b4372846e467965aeae7cdfd55abb60586cb0613f
kernels.cu 5efa435c4716ab6d5426dc08a8fb03ab84a7936919c66b2a0c4845e668a5d231
tensor.h   7b365738b17d5232f83ef5b0b8d017db2c991cbe1d3481b82e8c35dd0dd6f501
```

Run `bash test_native_policy.sh NEW_OUTPUT_DIRECTORY PRIVATE_CHECKPOINT` for
both native5 precision tests. `REK_NATIVE5_REFERENCE_SRC` selects the pinned
trainer's `src` directory. `test_native_policy_legacy.cu` must compile with
`-DPRECISION_FLOAT` and the original legacy source include directory. Link its
object with `native_policy.o`, `-lcublas -lcurand -lcrypto`. Both test sources
invoke the original implementation rather than a separate numerical rewrite.

## Verified existing policy provenance

The following files exist privately on Spark. Both legacy hashes were checked
against the actual file, its manifest, and the recorded evaluation input.

| Policy | Recorded training | Observation / precision / hidden activation | Historical frozen evaluation |
| --- | --- | --- | --- |
| `combined3276800-r1/0000000003276800.bin` | 3,276,800 MuJoCo-GPU transitions, 100 epochs | raw 223 / FP32 / legacy polynomial | 82 wins, 35 losses, 11 ties in 128 rounds |
| `corrected3276800-r1/0000000003276800.bin` | 3,276,800 further MuJoCo-GPU transitions, 100 epochs, initialized from combined checkpoint | raw 223 / FP32 / legacy polynomial | 82 wins, 38 losses, 8 ties in 128 rounds |
| Native5 `final-512/.../0000000000024576.bin` | 24,576 Puffysics transitions, 3 PPO epochs | scaled_polar_xy / BF16 / native5 sigmoid | No completed rounds measured |
| Native5 `final-4096/.../0000000000196608.bin` | 196,608 Puffysics transitions, 3 PPO epochs | scaled_polar_xy / BF16 / native5 sigmoid | No completed rounds measured |

All four policies have hidden size 256, two MinGRU layers, 33 categorical
actions, and 1,836,032 checkpoint bytes. Legacy paths are beneath
`/home/spark-advantage/rek-training/training-opt-20260911/training/`.

```
combined3276800 SHA256 7575182d6b4c2bf0fa7fa558e9210cea2d0c7a8832fd4d279335eb646dfffd77
corrected3276800 SHA256 87d9ef33f2893ad8b369783e8fecbd95afc4a4793cce8d701adf0423b933a544
```

Legacy evaluations are recorded beneath the same experiment's `evaluations/`
directory, in `eval-trained256-ticks6400-v1/report.json` and
`eval-corrected256-ticks6400-v1/report.json`. They ran 6,400 control ticks per
arena against the semantic candidate dummy, with stochastic actions and
recurrent state reset every 64 ticks. They are not new league rankings or
human/authentic REK evaluations. Both policies were trained as fighter 0;
role-swapped evaluation is a distribution shift and should be reported by role.

For historical horizon behavior, use `rek_native_policy_reset_recurrent` every
64 ticks. This preserves sampler RNG. `rek_native_policy_reset` additionally
restarts RNG and is appropriate for explicitly restarting a match.

## Runtime and trainer integration

`runtime_api.h` ABI 2 exposes borrowed device views, exact external action
overrides for either fighter, both raw and explicitly encoded observations,
and one-arena inspection snapshots. Device stepping remains GPU-resident.
Snapshots synchronize only on the explicit inspection call.

Round counters classify points/KO wins, ties, redos, and unclassified results
separately. Invalid physics/observation/scheduler states do not become wins.
The optional `round_seconds` configuration is zero by default, preserving the
recovered 120-second normal and 30-second redo durations. A positive override
changes clocks only for newly prepared rounds; it must be part of match/training
configuration identity. Disconnect handling and league attribution belong to
the evaluation server, not the simulation reset function.

The native trainer optionally accepts a frozen opponent through:

```
--env.opponent_checkpoint=PRIVATE_PATH
--env.opponent_sha256=VERIFIED_SHA256
--env.opponent_observation_encoding=raw|scaled_polar_xy
--env.opponent_precision=0|1
--env.opponent_legacy_fast_hidden=0|1
--env.opponent_hidden_size=256
--env.opponent_num_layers=2
--env.opponent_deterministic=0|1
```

Precision 0 is BF16; 1 is FP32. The default remains the original scripted
opponent. CPU viewing backend selectors are rejected by the trainer even in
eager mode. The updated runtime and trainer wrapper compile on Spark; full
simulation/UI validation is separate from these frozen-policy oracle tests.
