# Owned pending-yaw migration and matched authentic GAE update

Completed 2026-09-20 UTC. This is a representation and learning-control experiment. It establishes checkpoint/input compatibility and records two matched native updates. It does not establish improved fighting or repair the compact simulator's contact, balance, or server execution dynamics.

## Input contract and migration

The frozen f8 actor's completed development rounds r21, r22 and r23 supplied 17,558 decision rows: 5,768, 5,934 and 5,856 respectively. Outcomes were wins 22:5 and 11:9, then a loss 9:14. All three terminal-race requests remain in the data with actor weight zero and value weight one, leaving 17,555 actor-weighted rows. These are three training episodes, with no held-out claim.

`owned_yaw_trajectory_data.cjs` upgraded the immutable original export, without re-encoding poses. Column 187 was positive zero in every original row. The new value is the already-owned desired yaw while the saved busy projection is true, otherwise zero. The same decision's relay source is bound to its encoder input by full parsed-source hash and to its saved worker input by sequence, clock, observations and mask. Desired category must be known, integer and within 1..15. Category zero is not a valid owned desired state. The new sampled action is never used to reconstruct its preceding observation.

7,793 rows have nonzero upgraded column 187. All data bytes outside the explicit format header and that column are unchanged, including feature 178, masks, action labels, rewards, elapsed times, discounts, recurrence and loss weights. `terminal_after` describes the next state; it does not erase the active last decision's pending yaw, including terminal-race decisions.

The explicit v2 dataset/replay formats are `REKRL002`/`REKBR002`, both version 2. The legacy formats reject them. `authentic_ppo.cu` accepts them only with `--observation-schema=rek.native5.scaled_polar_xy.owned_yaw_v2`. Its default loader and training math are unchanged. The legacy JSONL exporter now requires `rek.native5.scaled_polar_xy.v1` on ready and every worker request, rejecting missing or v2 schema. Future actual-v2 exports require an explicit compatible exporter; the historical upgrader does not silently relabel them.

`owned_yaw_migration.cu` obtains the encoder's registered `[256,223]` layout from the native architecture. It creates a new checkpoint with only the 256 FP32 weights at encoder input column 187 zeroed. Whole-vector comparison proves that all other 458,752 parameters, including shared recurrent parameters and the decoder/value output, remain bitwise unchanged. The original f8 checkpoint is preserved.

## Verification

- Seven historical-upgrade Node tests and twelve legacy-exporter tests passed, including unknown desired intent, source/busy mismatch, nonzero legacy column, pre-action causality, terminal-after semantics and schema rejection.
- Native CPU migration self-test passed with no CUDA calls. The trainer build passed 24 GAE checks, 14 numerical-acceptance checks and two pinned-kernel adapter tests.
- Native dual BF16 recurrent replay, seed 73 per fresh sequence, matched all 17,558 recorded actions. All 34 outputs, including value, plus native-order chosen log probabilities and sampled actions were bitwise equal between original f8/v1 inputs and migrated 9c0/v2 inputs. No input feature mask was applied. This ran 02:56:26.117105575Z to 02:56:33.429536692Z.
- The original replay utility separately reproduced all 17,558 original actions. Four native CLI negative checks rejected v2 without opt-in, v1 with v2 opt-in, and both mixed replay formats before optimization.
- Epoch-zero runs of the prior pinned trainer and new trainer on v1 produced byte-identical stdout and unchanged f8 checkpoint bytes. A v2 epoch-zero run preserved the migrated checkpoint. Both arms had identical target and initial numerical diagnostics.
- CUDA GAE versus the independent CPU reference had maximum absolute error zero. The Puffer sequential teacher matched native replay logits and values exactly. Initial BF16 batch-forward maximum chosen ratio error was 0.0168448606123, with zero clipping and mean legal KL 1.80712900259e-7. The existing explicit bounded-BF16 acceptance mode passed; exact batch parity is not claimed.

## Matched one-epoch update

Both arms used fresh optimizers, horizon 128, learning rate 1e-5, policy/value clips 0.2, value coefficient 0.5, entropy coefficient 0.001, and frozen-value GAE. There was no MC target flag, sweep or simulator stepping. Each arm completed 139 updates on the same three full-round sequences in the same order, with the existing full-prefix burn-in before each chunk. This order is a known limitation; no episode shuffling was added.

Actual-time discounts were preserved byte-for-byte, with reference gamma 0.9998844821426083 and lambda 0.9978673240629938 per 20 ms. Both arms' initial advantages had mean -0.0039738728638, standard deviation 0.205704732583 and range [-0.722899854183, 1.36139512062]. The initial actor and value outputs and the resulting GAE targets were equal before optimization.

| Measurement | Control: unchanged v1 | Treatment: owned-yaw v2 |
|---|---:|---:|
| Process wall time, including diagnostics | 2.42 s | 2.60 s |
| Epoch policy loss | 0.00399340785 | 0.00401735808 |
| Post-update mean legal KL | 0.000627730156069 | 0.000627609023898 |
| Post-update maximum legal KL | 0.0254013506941 | 0.0245080206733 |
| Post-update chosen ratio range | 0.755465789534 to 1.39776047085 | 0.746126567883 to 1.4047903799 |
| Post-update clipping fraction | 0.00148080646999 | 0.00142385237499 |
| Post-update value MSE to frozen GAE target | 0.0435313644047 | 0.0434990953069 |

Original value MSE to the same target was 0.0423302286958. Neither arm improved that diagnostic. The table describes training-batch forward evaluation, not authentic fighting efficacy or held-out critic prediction. PPO clipping is an objective term, not a hard bound on final probability ratios.

Control ran 03:07:05.192340592Z to 03:07:07.617869693Z. Treatment ran 03:07:07.623066426Z to 03:07:10.232792494Z. The full replay/check/train wrapper ran 03:06:55.556618678Z to 03:07:10.277545327Z, then released the GPU. No client or desktop interaction occurred.

## Reproduction and immutable artifacts

Private Spark root: `/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1`.
Private Windows root: `C:\rekagent\work\consistent-fighter-20260919-r1\owned-yaw-migration-r1`.

Exact commands, stdout, stderr, UTC stamps, process timing and hashes are in `matched-gae-r1/commands.sh`, `commands-and-stdout.txt`, `set-x-stderr.txt`, and each arm's directory. Migration replay commands and receipt are under `replay-r1`. Build provenance is under `build-r1/build-hashes.sha256` and `ppo-build-r2/build-provenance.txt`. The new trainer was built from the existing prepared BC `imitation-20260919-r1-native-bc/build-r5`. The first attempted PPO build used an already-prepared PPO directory and refused an existing generated kernel; that failed directory was retained and the successful build used a fresh `ppo-build-r2` destination.

The final native invocations share this argument tail:

```text
authentic-ppo DATA REPLAY INITIAL INITIAL_SHA NEW_OUTPUT 1 1e-5 128 .2 .2 .5 .001 --allow-bounded-bf16-batch
```

Only the treatment additionally supplies `--observation-schema=rek.native5.scaled_polar_xy.owned_yaw_v2`. The exact paths and pre/post input hash checks are saved in the command script. The original replay and initial checkpoints remain separate from the trained candidates.

| Artifact, relative to Spark root unless stated | SHA256 |
|---|---|
| Original f8 checkpoint, `critic-calibration-20260919-r1/train-scaled-gae-r1/ppo-one-epoch.bin` under sibling stage | `f8bcd3f3d6ef3d691209823d5a0d452ca16ff16b03c7ed1867715510986ea483` |
| `migration-r1/owned-yaw-v2-initial.bin` | `9c0cefb9776c6e30b120ae9feeae94756833c653eb9de1fd508f25c554064f55` |
| `data-v1-r1/authentic-trajectories.bin` | `847d73fb0b2d19e035ac81e318ca23d352ff7acaff8116f511f83dd322ded328` |
| `data-v2-r1/authentic-trajectories-owned-yaw-v2.bin` | `a58502e1a36b1b7a730e572325cc9e58a7e55b33d1428221b4575dab12102d77` |
| `matched-gae-r1/replay-v1/behavior-replay.bin` | `d42e11fca690410b5dc6577d29cb2082e2ea5aa31c4166f46ac89247e3e19a72` |
| `replay-r1/owned-yaw-v2-replay.bin` | `1b935939d7c5f689c9e8bd592ada999a52e6c465e1577612e712d9e023655729` |
| `matched-gae-r1/train-control-v1/ppo.bin` | `f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4` |
| `matched-gae-r1/train-treatment-v2/ppo.bin` | `c59a9f9044c8975a7cd790c5d8b44b253e6394fc2de93f2727b7d99879c4981e` |
| `build-r1/owned-yaw-migration` | `a5d64a355c390456c60bbbc233d6ddb9ad50b0d8371455cd06dcea0ac012fd28` |
| `ppo-build-r2/authentic-ppo` | `0b6fcacccffa951375d0ef7cd9cc520e22f3cbdfa4b77009ee2fe25e598145c6` |
| `ppo-source-r1/authentic_ppo.cu` | `8b2ebee449dc2e6412a52120c5cec923fb36eda439e8a6a8ee175a45aa30430b` |
| `ppo-source-r1/owned_yaw_trajectory.h` | `2682a82ec8dcc56148650c035eee23cf1367583865cf6d88ed19ec5f6e2bc92f` |
| `ppo-source-r1/owned_yaw_observation.h` | `791b7ab142089147821e3a87dffa15106ebd8cfa0725773aa83a83b8b4493b76` |
| Historical JSONL upgrader source | `6da97c7f940c4f906a22b484a2d513943246d7ff02b9c3bbb7b56a89112d04d0` |
| Strict legacy exporter source | `436c0c25875e23f8dc51128fe9ecfe18fc9ead07cd7864b997cf459efc89e1bb` |

Completed build/source/checkpoint/replay/training artifacts were copied to the Windows root's `spark-results`: all 114 manifest-listed files, totaling 25,271,820 bytes, passed independent SHA256 verification after extraction. The private compressed copy is `completed-artifacts-r1.tar.gz`, 18,136,737 bytes, SHA256 `c10122f812ed0c80de0a53bba3ca25dfc0093ef3d922048423eff29d90602619`; its contained source manifest SHA256 is `296b830f78b0e0050fe55770dee22cd954e28c24903cdb834e683f999eb7cf20`. The original/v2 datasets remain in their separate private directories with the hashes above. No raw records, weights or private source snapshots are included in this public report.
