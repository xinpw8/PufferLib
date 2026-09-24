# Executed fixed state-baseline PPO update

The planned native CUDA update completed with exit 0 and saved the prospectively selected final epoch-1 checkpoint:

`35263bb752e049c40e2d65ea80d58b72ba1bd8c98042d0a51ddbf175ce8b9680`

Private output: `/home/spark-advantage/rek-training/timing500-state-baseline-20260924-r1/ppo-variant/train-fixed-r1/policy.bin`. The epoch-1 copy has the same hash. This checkpoint is not deployed or live-validated. Runtime control remains frozen f147.

## Execution and unchanged contract

The existing `run_once.sh` ran on 2026-09-24. Its saved start and completion timestamps are 12:44:42 and 12:44:45 UTC. Their one-second resolution gives a coarse 3 s interval, not a precise full-process benchmark. No live policy worker/control was active during this execution; the existing game and unrelated GPU workload were left untouched.

The update started from actual C2 on all five completed development episodes: 14,957 original-order rows, including 14,952 actor-eligible rows and five terminal-rejected rows. It performed one epoch, 120 updates, LR 3e-5, H128, clip 0.2, VF 0 and entropy 0.001. Rewards remain actual received own-minus-opponent point changes divided by 5, discounted using recorded QPC intervals with a 5 s half-life. There is no terminal bonus or score potential.

The frozen leave-one-episode-out state prediction is subtracted only from complete-MC advantages. Returns, true behavior log probabilities, actions, masks and recurrent history remain unchanged. The executed synthetic CUDA self-test passed, including zero-mode identity and propagation of terminal reward from an actor-excluded row. The real 14,957-row CUDA return and subtraction reference checks both reported maximum absolute error 0 and unchanged returns. No environment stepping occurred.

## Recorded numerical results

| Measure | Value |
| --- | ---: |
| Initial sequential logit/value error | 0 / 0 |
| Initial mean / maximum legal KL | 9.84230e-8 / 0.000120369 |
| Initial clipped fraction | 0 |
| Relative absolute initial surrogate perturbation | 0.00002215385 |
| Post-update mean / maximum legal KL | 0.0042795853 / 0.0587343005 |
| Post-update chosen approximate KL | 0.0042754833 |
| Post-update clipped fraction | 0.0430567627 |
| Eligible advantage MSE after baseline | 0.0337223340 |

The existing explicit distributional BF16 acceptance passed. Initial batched outputs were not bitwise equal: maximum chosen-ratio error was about 0.0208234. Exact sequential teacher outputs and the true behavior denominator were retained. These diagnostics do not assert exact batched parity.

Post-update measurements use the training observations, not held-out live outcomes. They establish an executed actor update, not improved gameplay. Five correlated episodes, the adverse held-out fold and finite-sample cross-fitting remain limitations.

## Evidence

`receipts/training-stdout.jsonl` is the exact scalar-only native stdout, SHA256 `e5770a0548cee1f3de760fe1419b35c5f938fa663dbefc920c55e2ad6a4d5ddf`. `receipts/training-execution.json` binds source receipts, input hashes, frozen launch plan, both checkpoint hashes, executed settings and diagnostics. Stderr was empty. Input pin checks ran before and after training; final policy and epoch-1 policy compared equal.

The original plan and preparation receipts retain their earlier unexecuted status as history. This result supersedes that status for execution only. Source files, optimizer code and frozen protocol were not changed by this publication update. No binary, baseline weights, dataset, replay logits or raw capture is published.
