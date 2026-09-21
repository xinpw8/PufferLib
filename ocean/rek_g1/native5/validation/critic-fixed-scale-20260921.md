# Fixed value-head initialization, 2026-09-21

`critic_calibration.cu --fixed-scale CHECKPOINT SHA SCALE NEW_CHECKPOINT`
creates an exclusive new checkpoint, verifies its input SHA and architecture,
scales only the bias-free value row, verifies all other parameter bytes, and
reads back the complete output. This mode calls no CUDA APIs. Negative and
nonfinite scales, malformed or mismatched SHA values, nonfinite weights, and
existing outputs are rejected. Zero is explicitly supported as a cold critic.
SHA arguments use 64 lowercase hexadecimal characters.

The actual pinned architecture registration yields 459008 float parameters.
Only indices `[65536,65792)` are scaled using
`float(double(original_weight) * scale)`. The other 458752 parameters remain
byte-identical. Actor parameters and recurrent weights are unchanged; further
PPO training can subsequently change both actor and critic. BF16 inference is
not claimed to be an exact scalar multiple of old value predictions.

## Input reward verification

Source checkpoint:
`/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin`

SHA256: `f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`.

Its saved `stdout.jsonl:4` binds dataset
`847d73fb0b2d19e035ac81e318ca23d352ff7acaff8116f511f83dd322ded328`;
lines 12 and 13 bind the resulting checkpoint hash. The matching
`owned-yaw-migration-20260920-r1/data-v1-r1/manifest.json:19` specifies:

```text
terminal_outcome + gamma_t*Phi(next_observed_points) - Phi(current_observed_points)
terminal_Phi=0; Phi=d/(5+abs(d))
```

Thus 0.01 scaling is a critic-initialization experiment for a changed objective.
It is not a verified conversion from raw point-difference reward units.

## Reproduction and outputs

Fresh private stage:
`/home/spark-advantage/rek-training/normalized-sweep-20260921-r1/critic-units`.
The public sources were copied into its `source/` directory. Commands run there:

```bash
stage=/home/spark-advantage/rek-training/normalized-sweep-20260921-r1/critic-units
input=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin
digest=f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4
bash "$stage/source/build_critic_calibration.sh" \
  /home/spark-advantage/rek-training/authentic-ppo-20260919-r1/build-r5 \
  "$stage/build-r1"
"$stage/build-r1/critic-calibration" --fixed-scale "$input" "$digest" .01 \
  "$stage/f3-value-scale-0p01.bin"
"$stage/build-r1/critic-calibration" --fixed-scale "$input" "$digest" 0 \
  "$stage/f3-value-scale-zero.bin"
bash "$stage/source/test_critic_fixed_scale.sh" \
  "$stage/build-r1/critic-calibration" "$input" "$digest" "$stage/cli-test-r1"
```

| Artifact | SHA256 |
|---|---|
| `f3-value-scale-0p01.bin` | `4f858ff5e504c95cb96f08eaca495b133505eabea9a25be73e53f3f0a5c529cd` |
| `f3-value-scale-zero.bin` | `e9255d8638bc9aee9db2cf2729de5fa93a66ea7aea46ca4b3688e31e37ffdb10` |
| `build-r1/critic-calibration` | `94076614bec5dca7fefdbb86f308138ef94b79a9cae98bc1d8b6692f36413403` |
| `source/critic_calibration.cu` | `0cab15ddbc6b032b36f66d757d868e02ff6d15822cdabf63fb24503d8215eb22` |
| `source/test_critic_fixed_scale.sh` | `5a290e7c8074c880ea80e095d867535688e8256f5f08dc659863b782c756d073` |

Build, native CPU self-test, both requested transformations, and CLI suite exited
0. The CPU test covers scales 0, 0.01, 1, and 2; value-row multiplication;
byte-identical untouched parameters including signed zero; non-value mutation
rejection; invalid scale/SHA rejection; nonfinite input and scaled overflow.
The CLI suite independently compares both untouched byte ranges with `cmp`,
checks whole-checkpoint identity at scale 1, and verifies nine rejection cases:
negative, NaN, positive/negative infinity, malformed/overflow scale, malformed
SHA, wrong SHA, and existing output. Each rejected call exits 1. Failed new-output
requests leave no file; the existing-output rejection preserves its old bytes.

Raw receipts remain in private `build.log`, `scale-0p01.json`, `scale-zero.json`,
`cli-test.log`, `cli-test-r1/`, and `build-r1/build-hashes.sha256`. The original
checkpoint was not modified. No GPU test, environment stepping, game launch, or
authentic policy-performance claim is part of this change.
