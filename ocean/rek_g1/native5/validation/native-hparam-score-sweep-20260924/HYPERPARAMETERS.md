# Native REK hyperparameter screen

This records the actual plan in `plan.cjs` and execution in `run.cjs`, not a proposed sweep. The reward scale, initial actor and critic, opponent, observation contract and evaluation seeds are shared across each comparison. Selection uses held-out policy-side-0 own points per completed round, with nonregressing margin and round win rate against the unchanged checkpoint. Full-match results are unavailable from these records.

The screen executes 25 configurations: one inherited control plus 24 arms crossing learning rate `{.000055,.0003,.001,.003,.0075,.015}`, entropy `{.00017,.002}`, and the two temporal profiles below. The inherited control uses LR `.000055`, entropy `.00017`, policy clip `.13`, value coefficient `1.02`, horizon `128`, gamma `.9998844821426083` and lambda `.9978673240629938`. The other 24 arms use policy clip `.2` and value coefficient `.5`, so their comparison with control changes these coefficients as well as any LR/entropy/temporal changes.

| Profile | Horizon | Gamma | Lambda | Discount / trace half-life |
|---|---:|---:|---:|---|
| A | 256 | .9997689776295918 | .9957391964326402 | 60 s / 3.08 s |
| B | 512 | .9998844821426083 | .9978673240629938 | 120 s / 6.16 s |

All runs use 512 learner arenas, minibatch 8192, replay ratio 1, value clip `.2`, gradient clip `.5`, momentum `.95`, MinGRU hidden size 256 and two layers. LR uses cosine annealing to zero over each run; entropy is constant. The original checkpoint SHA256 is `7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96`. Every training run starts from these same weights with fresh optimizer and recurrent state.

The screen trains all 25 configurations for 8,388,608 transitions using training seed 73. Frozen evaluation uses seeds 10001 and 10003, 64 arenas per seed, one completed 120 s round per arena. Only policy side 0 enters selection, giving 128 evaluated rounds per checkpoint; validated side-1 records are excluded because the calibration is asymmetric. The unchanged checkpoint is evaluated on the same fixtures.

Selection takes up to four configurations ordered by own score, then round win rate, then margin, from those passing the point-estimate guards. If none passes, the four highest own-score configurations are selected provisionally to examine tradeoffs. This fallback does not establish improvement. Confirmation always includes the inherited control plus selected non-control configurations, each independently restarted for 33,554,432 transitions at training seeds 73 and 947. Confirmation uses new evaluation seeds 20011, 20021 and 20031, 128 arenas per seed, one round per arena, giving 384 policy-side-0 rounds per checkpoint. There is no 134-million-transition stage in the executed plan.

The fixed reward is normalized own-minus-opponent point delta with an accepted-attack cost of `.15`, divided by 100; the point term is clipped to [-1,1]. The preserved calibration uses corrected action IDs, bot-award probability `.25`, and zero unmeasured kick-fall prior. These runtime choices are shared across configurations. They do not establish authentic REK parity.

Compare each checkpoint with both the unchanged checkpoint and the matched training-seed control. Common seed/arena/episode pairs support paired differences; policy-dependent RNG consumption can still differ. Only two screening and three confirmation evaluation seed clusters are available, so the report gives seed-specific effects and observed ranges instead of claiming a calibrated seed-level 95% interval. Repeating training seeds on the same evaluation seeds does not create additional independent evaluation clusters.

At 50 Hz, compute gamma as `2^(-.02/discountHalfLife)` and lambda as `2^(-.02/traceHalfLife)/gamma`. Action holds do not change the reward or discount clock. Stock native sweep lambda maximum .995 excludes both profiles, requiring explicit overrides.

Evidence paths on Spark use `N=/home/spark-advantage/rek-training/native5-rek-20260913-v1/pufferlib5` and `U=/home/spark-advantage/pufferlib-5.0-wr64`. Local paths use `R=C:/Users/Daniel/codex-rek-puffysics-training-profile/ocean/rek_g1/native5`.

- Native Muon default LR .015, gamma .995, lambda .90, entropy .001, value coefficient 2: `N/config/default.ini:67`. Committed GPU robot-arm config uses LR .0003, horizon 128, minibatch 32768, replay 1, entropy .002: `U/config/robot_arm.ini:22`. Committed GPU Breakout config uses LR .0627130643, horizon 32, minibatch 65536: `U/config/breakout.ini:33`. These config defaults do not establish REK performance.
- Actual Muon normalization and update semantics: `N/src/algo.cu:1051` and `:1144`. Raw advantages enter PPO at `:1500`; entropy/value coefficients depend on reward units. REK unclipped rewards: `R/puffer_env.cu:5`.
- Updates per rollout are `floor(replay_ratio*arenas*horizon/minibatch)`, with contiguous recurrent rows: `N/src/pufferl.cu:1527`. There is no independent PPO-epochs knob. Use integer replay ratios. Minibatch and horizon constraints: `:3488`. Carry state persists across horizons: `:1516`; training chunks still truncate gradients.
- Executed REK task-time profile B and its limitations: `R/validation/task-credit-20260919/README.md:40`. Measured compact throughput with 512 arenas, horizon 512, minibatch 8192 and LR .0001: `R/validation/native-training-throughput-20260920/README.md:9` and `:49`. Throughput is not policy quality.
- Native MuJoCo GPU is a separate REK integration, measured at 6,020 transitions/s with horizon 16: `R/mujoco_gpu/validation/training-20260914/README.md:9`. No generic MuJoCo config exists in the inspected U tree.
- The exact 20 ms transition clock is `R/fast_runtime.cu:32`; holds preserve 50 Hz inference/reward/discount: `R/action_cadence.h:7`.
- Required final-transition bootstrap patch: `R/pufferlib5_temporal_credit.patch:75`. Keep synchronous execution, one buffer, one policy, and `reset_every_horizon=0`.
- Checkpoints restore weights only, with fresh optimizer, RNG, RNN and step count: `R/pufferlib5_initial_model.patch:10`. Cosine LR uses each invocation's requested total budget: `N/src/pufferl.cu:1605`. Rerun promoted settings from the original initialization for each larger budget, or explicitly describe optimizer-reset continuation. It is not an exact resumed trajectory.

Native source/config blobs in N were checked against U commit `84a89728fafd4034ae3c386a9e0665d62310780b` and match. The local build pins equivalent public source commit `773f923d80e73bdc255a2ba730c918b28e416aa1`: `R/build_native.sh:34`.

The source references explain why the executed ranges were considered. They do not establish that any setting is optimal. Arena-count, replay-ratio and architecture sweeps were not executed in this plan.
