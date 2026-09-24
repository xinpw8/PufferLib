# Timing500 PPO: one-epoch LR3e-5 sibling

Native CUDA training completed successfully: one epoch, 192 optimizer updates, exit 0, 2.57 seconds full process wall time. The selected final checkpoint is:

`7001cee36887ef5e168726576e0d5f07282628eb2c80a87d12e9e9f87e05ff5c`

This is a controlled learning-rate variant of [timing500-onpolicy r2](../timing500-onpolicy-20260924-r2/README.md). The sole learning change is LR 1e-5 to LR 3e-5. Both runs start with a fresh optimizer from the actual recorded behavior checkpoint `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`. This run does not continue from the earlier c2c4987 checkpoint.

## Unchanged data and objective

The same eight completed parent rounds, including all three losses, supply 24,010 actual decisions: 24,002 locally applied actions and eight terminal-race rejections with actor weight 0. Native/referee checks reconcile 104:93 points. All 223 inputs, masks, actions, actual seeds, timestamps, recurrent order and rejected rows remain unchanged. No sampling-rate conversion, feature migration or new teacher is introduced. The original parent batch was 5W3L and failed its predefined 18/20 criterion. Eight episodes remain a small development training set.

Reward remains received own-minus-opponent point delta divided by 5, with actual QPC interval discount `2^(-dt/5)` and complete discounted Monte Carlo returns. There is no terminal win bonus, score potential, guessed fall hazard or reward clipping. Horizon 128, PPO clip 0.2, VF clip 0.2, VF coefficient 0, entropy 0.001 and no advantage normalization are unchanged. H128 limits backpropagation length; returns span the complete episode, and every chunk replays its full preceding history using current weights.

The byte-identical parent replay is reused. Its native worker reproduced 24,010/24,010 original samples exactly. FP32 behavior log probabilities are fixed throughout optimization. Dataset SHA256 is `a2ab10a34404cebea926cb2cec0fdea1386e0032c5333328acfa1c8fe0b0ba21`; replay SHA256 is `d8707ff84cee844f7ad43853941886472b56c16c766ae523b4a6efde6ad5526a`. Full parent identities, source provenance, controlled-start coverage and archive dependencies are in the prior package and this folder's exact `parent-run-plan.json` copy.

## Executed native result

Training ran from 2026-09-24T11:00:45.086Z to 11:00:47.661Z. The runner verified the frozen parent plan, replay and replay receipt plus every checkpoint, dataset, identity, selection, worker, native object, feature mask and executable hash. It then exclusively created `train-one-epoch/` and executed the unchanged native CUDA trainer. No environment stepping or Python training occurred.

CUDA complete-MC targets matched the CPU reference with maximum error 0. Initial sequential teacher logit/value error was 0. Initial batched BF16 evaluation used the existing explicit distributional acceptance: mean legal KL 1.81612e-7, maximum 0.000478677, zero initial clipping, and relative absolute surrogate perturbation 3.35920e-5. Exact batched parity and exact batched gradients are not claimed.

| Frozen-parent post-update diagnostic | LR1e-5 sibling | LR3e-5 sibling |
| --- | ---: | ---: |
| Mean legal KL | 0.000679761137 | 0.002055485552 |
| Maximum legal KL | 0.010287117911 | 0.033029600376 |
| Clipped fraction | 0.000166597251 | 0.005456059975 |
| Optimizer updates | 192 | 192 |
| Full process wall time, s | 2.58 | 2.57 |

These are recorded offline optimization diagnostics. Timing is not environment-training SPS. Increased policy change does not establish improved Bot1 play. Final epoch 1 was selected prospectively without development-based checkpoint selection. The earlier LR 1e-5 candidate subsequently closed 2W3L, 49:55 points, and failed its live criterion; its analysis is separate from the new candidate's still-pending outcome.

Exact command, stdout, stderr, UTC bounds, process time, exit status and checkpoint receipt are in `receipts/train-one-epoch/`. The saved checkpoint SHA was independently read back from Spark; weights are excluded from this publication.

## Reproduction and source scope

Eight executable source/config files are unchanged copies from the private stage:
`/home/spark-advantage/rek-training/timing500-ppo-lr3e5-20260924-r1`.

`SOURCE-MANIFEST.json` records their hashes. This publication README replaces the private CPU-preparation description with executed results; the original README and CPU-ready receipt remain verbatim under `receipts/CPU-PREP-README.md` and `receipts/CPU-READY.json`. Their `gpu_executed:false` fields describe the earlier CPU-preparation cutoff, preceding the completed native training. The private originals were not edited.

Five tests passed on Windows and Spark before training, and again from this repository package. `receipts/PUBLICATION-CPU-TESTS.txt` records the latter run. Local imports require only Node.js:

```sh
node --test run_candidate.test.cjs prepare_live.test.cjs
```

Executed artifact checks and training, from the private stage:

```sh
node run_candidate.cjs train --check
node run_candidate.cjs diagnose --check
node run_candidate.cjs train --run
```

The optional `diagnose --run` mode was not executed. It requests zero optimizer epochs and a separate output directory. It is not an added prerequisite. Re-execution needs a fresh isolated destination in `experiment.json`; existing outputs are refused. Native source/build dependencies remain those in [scorecredit5s](../scorecredit5s-20260924-r2/README.md). This variant changes no CUDA code or production file and introduces no new loss implementation.

The unchanged `capture_resources.cjs` helper is included for reproducibility. It checks an exact controller PID/argv, reads GPU/resource statistics and writes an exclusive passive log while that controller exists. Resource recording started at approximately 2026-09-24T11:01:55Z for the active cohort. The separate `snapshot_controls.cjs` helper reads a fixed-size snapshot of the relay log and reports local applied/rejected actions and QPC source-to-ACK age; those measurements do not prove server execution. Neither helper is part of the policy or trainer, and neither changes game input or policy actions.

No datasets, checkpoints, executable binaries, raw captures, credentials or proprietary game assets are included. Publication does not modify production source or active runtime artifacts.

## Prospective live cohort

The candidate was prepared using `prepare_live.cjs CHECKPOINT_PATH CHECKPOINT_SHA256`; the controller launched at 2026-09-24T11:01:45.214Z, PID 3006797. The new stage is `timing500-ppo-lr3e5-live-20260924-r1`, labels `lr3e5-s1401` through `lr3e5-s1420`. Target remains 18/20, stopping after the third completed nonwin, with 10-attempt and 10-relaunch budgets.

The frozen timing500 driver/controller, 500/750 ms bridge, balance8 encoder, native worker, all-ones mask, all 17 attacks and sampled selection are retained. Only checkpoint, seed and output paths change in per-round configurations. Planned-round receipt SHA256 is `0ec7a4e77ccf5c60c5ab1a9dd561a0ef8ce4dbe509c7f8a1fa55af5fa7e505f6`; its exact copy is in `receipts/planned-rounds.json`. The preparation receipt's `controller_started:false` describes preparation before the separately reported launch. No live win or improvement claim is made at this cutoff.
