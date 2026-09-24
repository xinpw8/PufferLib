# Measured balance8 authentic-PPO candidate

Private stage only. Existing repository, frozen encoder, active campaign, original dataset, and recorded checkpoints were not modified.

The preceding checkpoint 056818f3 cohort had two counted losses and zero wins at preparation time and had not met the acceptance criterion. Balance8 has no completed live evaluation yet. Incomplete infrastructure attempts are excluded from policy results.

Root launched the frozen balance8 cohort at 2026-09-24 07:13:30 UTC, controller PID 2758535, with attempt and unplanned-relaunch budgets both 10. Status at launch: running, no completed result. Subsequent outcomes belong in a separate result update; no efficacy claim is made in this frozen preparation bundle.

## Observation contract

Explicit schema: `rek.native5.scaled_polar_xy.balance8_v1`.

| Cells | Meaning |
| --- | --- |
| 9, 95 | Actor/opponent vertical root finite difference, m/s, using actual preceding valid source QPC interval |
| 72, 158 | Actor/opponent normalized root-up tilt divided by pi, with no fall threshold |
| 202 | Validated fresh lifecycle-bound received referee state available |
| 203 | Same-round, same-perspective source history with positive interval at most 250 ms available |
| 204, 205 | Received actor/opponent count-active bits; padding when 202 is zero |

The other 215 cells, source action mask, request-duration projection, all 33 categories, and all 17 attack categories are unchanged. Unknown derivatives/count bits have explicit availability flags. Invalid geometry or an available receipt failing strict wire/hash/freshness checks makes the observation unavailable. No fall hazard, threshold, transition probability, guessed measurement, reward bonus, or extra action mask was added. A valid received packet is client-observed server-authored telemetry, not proof of current server state.

The encoder wraps unchanged copied legacy and observable encoders in separate translation units. It overlays only eight cells on legacy output. Actual five-round export compared every protected FP32 cell and every other trajectory row byte against the frozen worker input.

## Recorded identity and derived teacher

Recorded behavior checkpoint: `7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96`.

Derived migrated teacher: `4906e39d0afa5e92334c5303b8c15cbadf48455537944c5a3d4c1f161d8b5263`.

Migration derives the actual registered native encoder layout, zeroes exactly 2,048 weights, and preserves the remaining 456,960 FP32 weights bitwise. The eight original behavior inputs were zero on every exported decision. The migrated teacher was never represented as the recorded worker.

`REKRL004` retains the original worker/checkpoint/native-object identity and per-round seeds. Its extended header separately binds the migrated teacher, original dataset, and original replay hashes. `REKBR004` is published only after all sampled actions and all 34 logits/value/logprob bytes match the original replay. Its row payload is copied from that original replay unchanged. PPO keeps the true original FP32 behavior logprob denominator.

Original native-policy object SHA: `4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c`.

## Completed checks and update

- Five original completed no-prior rounds, seeds 601 through 605: 28,715 decisions, 28,710 applied policy rows.
- 6,173,725 protected observation cells bitwise equal; every other row byte preserved. Eight columns changed 173,306 cells.
- CPU helper tests: 12 cases; encoder edge tests: 8 cases; dataset mutation rejections: 4; original v3 regression checks: 8; worker protocol: 112 assertions; schema runner tests: 4.
- Native migration: 8,704 CPU encoder dot products equal; exactly 2,048 zeroed entries checked, including protected-byte and negative-zero rejection.
- Root-executed GPU replay: 28,715/28,715 sampled actions exact, all logits/value/logprob bitwise exact; 5.07 s full process.
- Root-executed native CUDA PPO: one epoch, 226 updates, 5.40 s full process. No environment stepping or Python runtime.
- Trained output SHA: `9a875c347b512bb46dde25481be75d276ea1b4c0799886b30aaf03be6ae1c5ce`.
- Actual encoder plus trained native worker smoke: six recorded raw snapshots, three ready decisions; schema/checkpoint/mask identity and recurrent reset checked. No game connection.

Training is the existing complete-MC zero-baseline objective with value coefficient zero, learning rate 1e-5, horizon 128, clipping 0.2, entropy 0.001. The existing explicit BF16 distributional acceptance passed with exact sequential teacher parity, zero initial clipping, mean legal KL 3.54386e-7, maximum KL 0.000488247, relative absolute-advantage-weighted surrogate perturbation 3.86362e-5. This is not exact batched parity.

Post-update mean legal KL is 0.000492405, maximum KL 0.0210613, and clipped fraction 0.000557200. These are training diagnostics, not live efficacy evidence.

## Reproduction and deployment

Spark stage: `/home/spark-advantage/rek-training/balance8-authentic-20260924-r1`.

Native sources: `source/ocean/rek_g1/native5`; encoder sources under its `live_transfer` directory. Local mirrors are in `source/`, and compact receipts are in `evidence/`.

This validation bundle is a private variant, not an installation into production native sources. Its `baseline/` files and three `*.patch` files document exact reader/PPO/worker changes. The compact semantic-fast runtime and its production configuration are unchanged. Proprietary game/model assets and binaries are excluded. Compilation requires the existing CUDA/NCCL/OpenSSL/MuJoCo development dependencies and pinned prepared Puffer core identified in the build scripts. The registered core hashes are algo.cu `8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92` and puffer5_bc_core.cuh `d3e07e6c5f376584543cdba457d582d6674183a0efb29c0755f236adcf284454`.

Build/CPU export sequence for a fresh stage: `build.sh`, `export_balance8.cjs`, `finish_build.sh`. Build scripts compile CUDA code but execute only CPU tests. Export runs the native encoder and the CPU-only native migration.

GPU reproduction commands are `run_balance8.sh replay --run` and then `run_balance8.sh train --run`. They were already executed by root; fresh output paths are required for another run. `--check` prints the pinned command without GPU execution. All artifacts remain in the existing execution directories.

Live encoder binary: `build/encode-balance8`, explicit `--observation-schema rek.native5.scaled_polar_xy.balance8_v1` together with the unchanged model/projection/busy options.

Live worker binary: `build/live-policy-worker-balance8`. Arguments: checkpoint, exact SHA, frozen seed, `sampled`, all-ones mask path, `--observation-schema=rek.native5.scaled_polar_xy.balance8_v1`. Worker and runner check this schema explicitly.

`prepare_live.cjs` creates a fresh cohort only. It copies the authoritative s802 settings: same referee bridge, all attacks, all-ones mask, strong memory 2 / weak barrier 0, existing capture and isolated lifecycle. It never starts the controller. Optional explicit driver/controller paths select the separately tested pairing variant.

Prepared trained cohort: `/home/spark-advantage/rek-training/balance8-live-20260924-r1`, seeds 901 through 920, checkpoint 9a875c34. The driver explicitly binds the balance8 schema, checkpoint, mask, encoder requests, and returned actions. Approved pairing recovery identifies a measured private G1-versus-T800 mismatch, excludes it from policy results, verifies lease/stream release, and counts isolated client recovery against the existing unplanned restart budget. It does not alter cold-start grace, action semantics, or attempt accounting. Ten focused pairing tests passed. Original runtime files were preserved. Driver SHA: `0666e0567173fb389ff5cce6a50cc93bab12560fd5325418969691b54d5196a1`; controller template SHA: `a2ab5d6b585e59a428a40cca47825928fd829732d80be61b2eab4cdb66c18b97`. No controller was started by the candidate builder.

## Interpretation limits

The measured eight inputs enable conditional policy learning from actual balance/count observations. They do not establish fall causality or introduce a physical fall model. Actual finite differences include body displacement during native recovery/reset; observed maxima were approximately 20.24 m/s actor and 24.84 m/s opponent, with no invented clipping.

Complete-episode potential shaping telescopes to discounted terminal outcome minus the current score potential, with terminal potential zero. With the zero baseline, this supplies no independent intermediate-hit optimization in expectation beyond the terminal objective. Five episodes remain a high-variance sample. Rewards and actual timing were retained to isolate the observation change. New live evidence is needed to assess wins and credit adequacy; no live validation of this checkpoint is claimed here.
