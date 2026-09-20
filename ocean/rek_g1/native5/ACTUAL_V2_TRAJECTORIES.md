# Exporting actual owned-yaw-v2 trajectories

The existing exporter defaults to strict legacy v1. Actual recorded v2 fights require this exact opt-in:

```text
node authentic_trajectory_data.cjs ROOT NEW_OUTPUT 0.9998844821426083 0.9978673240629938 live-SELECTED_ROUND --observation-schema=rek.native5.scaled_polar_xy.owned_yaw_v2
```

Use only completed rounds with finalized strict contact/referee analyses and a single behavior checkpoint. No inference, optimizer, client interaction or GPU is involved in export.

The shared exporter keeps existing acknowledgement, score, reward, elapsed-time and recurrent-history checks. V2 additionally verifies ready/action/encoder schema, full-hash equality of the same pre-action relay and encoder input, recorded worker/encoder arrays, source QPC, saved busy provenance and active owned desired category 1..15. It checks recorded column 187 against that evidence; it does not replace the column or derive intent from the newly sampled action. Inactive terminal-only inputs require explicit schema and zero column 187. The last active decision retains its yaw even when `terminal_after` is true or its actor weight is zero.

Output is `authentic-trajectories-owned-yaw-v2.bin`, magic `REKRL002`, version 2, with the existing 256-byte header and 1,128-byte rows. Its manifest declares actual recorded v2 origin and the actual behavior checkpoint. No migration is claimed. Native replay bound to this dataset/checkpoint is still required before the explicit-v2 authentic PPO loader may optimize it. Legacy export remains strict and rejects v2; unknown or duplicate CLI options reject.

CPU regression coverage includes 24 exporter/upgrader tests. Refactoring into `owned_yaw_export_evidence.cjs` preserved the full historical r21/r22/r23 upgrade: 17,558 rows, 7,793 nonzero column-187 values, SHA256 `a58502e1a36b1b7a730e572325cc9e58a7e55b33d1428221b4575dab12102d77`, identical to the pre-refactor binary. The fresh private verification is under `C:\rekagent\work\consistent-fighter-20260919-r1\actual-v2-export-cpu-r1\historical-regression`.

Real-data CPU validation also passed. The current and immutable `65fc8393` legacy exporters produced byte-identical r24 datasets: 5,758 rows, SHA256 `724aec47b76bf86adeb54d68257347a9f665192235449bb870ea8007ac65ddb9`. The first actual-v2 export, r25, contains 5,717 rows, 5,716 applied actions and one terminal race; SHA256 `92589ff39205a478380732aeb8902a62ea3b18e5a3081d09c38529342888bb76`. The existing C++ v2 loader accepted its full reward/time/recurrence contract and rejected mixed schemas. Column 187 counts are 1,405 negative, 3,417 zero and 895 positive, with 4,146 busy rows. The last active decision retains -1 despite `terminal_after=1`. No observations were replaced.

The four focused Node suites passed 63 tests, including exporter/upgrader, runner and pinned-kernel adapter tests. The private `actual-v2-export-cpu-r1` directory contains the r24 pinned/current and r25 actual-v2 exports, test output, `verify_export.cpp`, its CPU-only build/run script, loader receipt and source/binary hashes. These checks perform no policy replay, optimization or GPU calls. V2 actor behavior probabilities remain a separate required native replay step before training.

## Native frozen behavior replay

The existing native replay utility now supports the same explicit schema selection:

```text
replay-authentic-behavior DATA_V2 CHECKPOINT CHECKPOINT_SHA NEW_REPLAY --observation-schema=rek.native5.scaled_polar_xy.owned_yaw_v2
```

V2 uses the existing v2 dataset/replay validators and emits `REKBR002`, version 2, with explicit v2 report identity and the actual dataset/checkpoint hashes. Legacy invocation still uses v1. Sampling, per-sequence RNG resets, recurrent state handling and native-order chosen log probabilities are unchanged. This path supports trained v2 checkpoints with nonzero column-187 weights; it does not use the initial-migration dual-replay comparison.

Build-only receipt: `/home/spark-advantage/rek-training/actual-v2-behavior-replay-20260920-r1/build-r1/replay-authentic-behavior`, SHA256 `52bdec853309d02b68288aae33acdb2086a5e837eb8478852aeb035f6928e0c1`. Its source SHA256 is `76ba9357bc9d519dea0cda43629277bb5c0faa167afb89999bee58c1a8c2bb8d`. It links the same native policy object as the actual v2 worker, SHA256 `4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c`. Fifteen existing dataset CPU checks and six native CLI checks passed before CUDA initialization: mixed schemas, unknown/duplicate options and both valid dataset loaders reaching the existing-output guard. Actual GPU replay of the new v2 fights remains pending an assigned GPU window; compilation alone does not establish sampled-action parity.

The actual configured worker is `/home/spark-advantage/rek-training/owned-yaw-observation-20260920-r1/worker-build/live-policy-worker`, SHA256 `321a24d6871799c724f8652b00efebba5e35017e4962304ab61c8c19b06ef1a1`. Its `build-command.txt` and `build-hashes.txt` bind `/home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o`, the same literal object file linked into the new replay executable, with the SHA256 above. There is no separate `worker-build/native_policy.o` copy. This comparison uses actual `worker-build` provenance, not only the separately built `worker-build-r2`.
