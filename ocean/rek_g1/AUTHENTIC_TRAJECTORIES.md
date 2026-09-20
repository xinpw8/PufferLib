# Authentic worker trajectories, version 1

`native5/authentic_trajectory_data.cjs` exports explicitly named completed development rounds. It reads files only. It does not collect gameplay, run physics, infer hit attribution, or infer server execution.

```text
node native5/authentic_trajectory_data.cjs ROOT NEW_OUTPUT GAMMA_PER_20MS LAMBDA_PER_20MS TRIAL_ID [TRIAL_ID ...]
```

The initial export uses the four sampled `live-round_outcome_v1-r1` through `r4` rounds under the private `consistent-fighter-20260919-r1` evidence directory. All four belong to development training. They are not part of a future frozen holdout.

## Verified CPU export

The four rounds contain 22,585 exact worker input, sampled-action, sent-request, and acknowledgement joins. There are 22,582 locally applied requests and three final terminal-race rejections. Each round starts a fresh worker with seed 73, BF16 precision, 223 observations, 33 actions, hidden size 256 and two recurrent layers. The frozen checkpoint SHA256 is `61f97b0b0a4504c6bdd0ee16d369ad4c1915e3cdf73d6358bab01ce64c8fde3f`.

All 68 native score award events reconcile with the observed scoreboard, totaling 69 local and 57 opponent points. Every round has a terminal relay observation. Only round 4 delivered a terminal request to the worker; rounds 1 through 3 ended while a final action was awaiting acknowledgement.

The current dataset is export r2, SHA256 `f2b67a599600bcca878085a942a6ae9b174e4d054aff0bd3cddf3d883c6d3b04`. It preserves the original task-time profile: gamma `0.9998844821426083` and lambda `0.9978673240629938` per 0.02 seconds, as recorded in `native5/validation/reward-objective-20260919/README.md`. Its 223-byte all-ones feature mask SHA256 is `59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b`. This preserves the original unmasked behavior. The human BC feature mask must not be substituted when recovering this behavior policy.

Ten Node tests and 15 native CPU validation checks pass. The native loader also validates the complete actual export.

The exact original native inference object reproduced all 22,585 sampled actions with zero mismatches in 4.42 seconds on Spark. Current r2 replay SHA256: `06c14f853cba325fd276fa9e97102f3917a623c6c9969722d43aca2c161cb171`. Recovered float32 log probabilities range from -8.57739258 to -0.0000972747803. No optimizer update was performed. Training-forward parity remains a separate check.

Private artifact directory: `C:\rekagent\work\consistent-fighter-20260919-r1\authentic-trajectory-r2`. It contains the dataset, manifest, transition ledger, exact feature mask, `behavior-replay-r2.bin`, and its verification JSON. The Spark working copy is `/home/spark-advantage/rek-training/authentic-trajectory-20260919-r1/data-r2`. The replay links the retained original `semantic-fast-20260914-v1/build-v4/native_policy.o`.

Export r1 remains preserved in the sibling `authentic-trajectory-r1` directory as a superseded discount experiment with references 0.999 and 0.995. Its dataset SHA256 is `31e6f59c25fa834fa2c05cb12c60ca390186366b335abc5656e78ab4adf8a02e`, and its replay SHA256 is `9b9c22f7e86ca98469ed1d16be7489d7761000eeb1c612ad92746b1f389f2133`. It was not used for an optimizer update. Between r1 and r2, every action, observation, mask, time, point counter and acknowledgement flag is byte-identical. Only discounts and shaped rewards changed in the dataset. All replay rows, including all decoder outputs and log probabilities, are byte-identical; only the dataset binding in the replay header changed.

## Causal and temporal contract

- A row contains the exact float32 conversion of one saved worker observation and its actual legal-action mask. Rewards, future observations, future scores, acknowledgements and terminal labels are not model inputs.
- Sequence order is the worker's actual inference order. Pacer-omitted relay observations do not advance the RNN. No missing 50 Hz decisions are fabricated, and no resampling is performed.
- The first worker decision resets recurrent state. Each new round was a fresh worker, including fresh sampler RNG. A recurrent reset alone must not reset RNG.
- The next state is the next actual worker source, or the captured terminal source for the final decision. Time is the difference between these client QPC observations. It is not server execution time.
- With reference interval 0.02 seconds, `gamma_t = gamma_reference ** (dt / 0.02)` and `lambda_t = lambda_reference ** (dt / 0.02)`, stored as float32. Current reference values are 0.9998844821426083 and 0.9978673240629938. The CLI requires explicit values and has no implicit discount defaults.
- The reward follows `round_reward.h`: `outcome + gamma_t * Phi(next_points) - Phi(current_points)`, where `Phi = difference / (5 + abs(difference))`, terminal `Phi` is zero, and terminal outcome is +1 or -1 for the observed winner. The same `gamma_t` must be used for return calculation. No causal strike or fall labels are required.
- Terminal-race rejected requests keep their originally selected action, have actor weight zero and value weight one, and remain part of the recorded recurrent history. They are not recoded as neutral or as executed attacks.
- The final transition has a true observed terminal and zero next-state bootstrap. Unobserved opening gameplay is reported and is not reconstructed.

## Dataset binary: REKRL001

All fields are little-endian. Header size is 256 bytes.

| Header offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | ASCII `REKRL001` |
| 8 | uint32 | Version 1 |
| 12, 16 | uint32 each | Observation count 223, action count 33 |
| 20, 24, 28 | uint32 each | Row count, row bytes 1128, round count |
| 32 | 223 uint8 | Fixed input feature mask, all ones |
| 255 | uint8 | Reserved zero |

Every row is 1128 bytes. Its initial observation and mask offsets match the BC row layout, but its support-mask semantics are actual behavior legality.

| Row offset | Type | Meaning |
| --- | --- | --- |
| 0, 4, 8 | uint32 each | Split 0, sequence ID, reset before observation |
| 12 | int32 | Originally chosen action |
| 16, 20 | float32, uint32 | Actor loss weight, reserved zero |
| 24 | float64 | Source QPC seconds relative to first worker source |
| 32 | 223 float32 | Exact preceding observation |
| 924 | 33 float32 | Actual saved legal-action mask |
| 1056, 1064 | float64 each | Next source relative seconds, actual dt |
| 1072, 1076, 1080, 1084 | float32 each | Gamma, lambda, shaped reward, terminal outcome |
| 1088, 1092 | uint32 each | Source and next-source observation sequence |
| 1096, 1100, 1104, 1108 | int32 each | Own, opponent, next-own, next-opponent awarded points |
| 1112, 1116 | uint32 each | Terminal after this decision, locally applied flag |
| 1120, 1124 | float32, uint32 | Value loss weight, reserved zero |

The shared C++ API is `rek_authentic::load(path)` in `native5/authentic_trajectory.h`. The output includes a manifest with input SHA256 bindings and a transition ledger containing decision and acknowledgement identities, clocks, and explicit rejection reasons. It does not copy account state.

## Frozen behavior replay: REKBR001

Build with the exact object used by the original native live worker:

```text
bash native5/build_authentic_behavior_replay.sh EXACT_NATIVE_POLICY_OBJECT NEW_BUILD_DIRECTORY
replay-authentic-behavior DATA CHECKPOINT CHECKPOINT_SHA256 NEW_REPLAY_BINARY
```

The utility invokes `rek_native_policy_step_rows` in saved order, reads its 34 diagnostic decoder values through `rek_native_policy_logits`, and computes the chosen-action log probability with the original native sampler's float32 reduction order and intrinsics. It compares every replayed sampled action to the recorded action. A mismatch prevents publication. It does not perform an optimizer update or connect to a game.

Header size is 128 bytes: magic `REKBR001` at 0, uint32 version 1 at 8, row count at 12, row bytes 152 at 16, reserved zero at 20, raw 32-byte dataset SHA256 at 24, raw 32-byte checkpoint SHA256 at 56, uint64 seed 73 at 88, and zero reserved bytes 96 through 127.

Each 152-byte row contains uint32 index at 0, uint32 reproduced action at 4, float32 old log probability at 8, float32 old value at 12, and all 34 float32 decoder outputs at 16. The last decoder output must equal the stored old value. `load_replay(path, dataset)` validates the exact dataset binding, dimensions, actions, finite values, and seed.

Behavior log probabilities and values are immutable teacher outputs. Before any PPO update, compare the full-prefix training forward pass at unchanged weights to this sequential replay. Existing synthetic native rollout tests and BC gradient tests do not establish long-sequence parallel-scan parity on these actual trajectories. A changed feature mask, checkpoint, recurrent prefix, or action mask changes the policy and must not silently replace the recorded behavior distribution.

## Statistical limits

There are four completed episodes, including one loss. Their 22,585 temporally correlated decisions do not constitute 22,585 independent fighting outcomes. Training on this fixed batch can be one bounded PPO improvement attempt from the original behavior checkpoint. Reusing it after substantial policy updates is offline/off-policy training and cannot be described as fresh on-policy experience. Authentic performance must be measured on subsequent rounds that were not used to choose the update.
