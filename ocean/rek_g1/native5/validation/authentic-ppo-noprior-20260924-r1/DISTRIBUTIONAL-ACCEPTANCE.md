# Explicit BF16 distributional acceptance

Original failed execution remains in `ppo-execution`. No gate or output was overwritten. Root subsequently approved the explicit mode and completed one epoch in `distributional-candidate`, with 226 optimizer updates. No live validation of the resulting checkpoint has been completed.

## Evidence

Exact native behavior replay reproduced 28,715 of 28,715 sampled actions. The training engine's sequential teacher logits and values also match exactly. Batched BF16 arithmetic differs.

At horizon 128, 376 chosen ratios differ from one, 25 exceed 1% error, 10 exceed 2%, three exceed 4%, and none reaches the 20% clip boundary. There are 710 changed legal logits among 195,792, across 424 rows. Nine of the ten >2% rows have maximum legal-logit distance one BF16 ULP; the other has 11. Overall maximum ULP distance is 284, so this is not a claim that all differences are one ULP. The worst chosen-action logit changes from 8.5625 to 8.625, exactly one BF16 ULP.

- Mean full legal-distribution KL: `3.54386016849e-7`.
- Maximum state KL: `0.000488246700609`.
- Mean total variation: `1.92234380689e-5`; maximum: `0.0156223748078`.
- Mean absolute chosen-ratio error: `3.55765873768e-5`.
- Mean absolute surrogate perturbation `mean(w*abs(A)*abs(r-1))`: `1.79604272016e-5`.
- Perturbation divided by `mean(w*abs(A))`: `3.86362154335e-5`, or 0.00386%.
- Mean signed policy-loss perturbation: `-1.2890140927e-6`.

Horizon 64 reproduces the same floor. Horizon 256 increases maximum ratio error to 0.15604 and maximum KL to 0.00726243. Both fail the unchanged original gate; all diagnostics perform zero updates. No additional horizon sweep was performed.

## Candidate contract

The original strict and max-ratio 2% modes remain available unchanged. The new explicit `--allow-distributional-bf16-batch` mode requires all of:

1. Exact sequential teacher logits and values.
2. Zero initial sampled-action clipping and maximum chosen ratio error strictly below the configured PPO clip.
3. Mean full legal-distribution KL at most `1e-5` and maximum state KL at most `1e-3`.
4. Absolute advantage-weighted surrogate perturbation divided by total absolute advantage at most `1e-3`.
5. Finite, nonnegative evidence throughout.

These are explicit engineering error budgets chosen after examining the diagnostic, not a claimed mathematical theorem, gradient-parity proof, or prospectively registered threshold. The normalized surrogate budget limits initial objective perturbation to 0.1% of its absolute-advantage scale. KL budgets bound both aggregate and individual-state distribution changes; they reject the observed horizon-256 discrepancy. The fixed candidate must still be evaluated on new live seeds. No affected row or action is omitted.

The true recorded behavior distribution remains the denominator. If recorded behavior is b and the batched training policy is q, the ratio remains q/b. Replacing old log probabilities with batched recomputation would change that estimator and is not done. PPO clipping, gradients, reward, targets, recurrence, optimizer and native kernels are unchanged. The new gate only permits an explicitly labelled approximation; it does not force initial ratios to one.

## Checks and execution

Sixteen new CPU gate tests pass, including exact sequential failure, clipping, KL limits, surrogate limits, nonfinite evidence, original-gate rejection and horizon-256 rejection. Existing GAE, parity and kernel preparation tests pass. Zero-epoch candidate execution passed in 3.61 s and saved the unchanged input SHA256 `7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96`.

Candidate trainer SHA256: `1ddff37719ad26670c8ebf8a19abddd83872882c893d4c120b6c863335bde680`.

Root approved and executed the explicit configuration:

```bash
bash /home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/run_distributional.sh --train
```

This uses the already selected one epoch, LR 1e-5, horizon 128, clip 0.2, value coefficient zero, entropy 0.001 and complete-MC zero-baseline targets. New output is `distributional-candidate/epoch-1/policy.bin`. The check-only option is `--check`.

The run completed with exit code zero and full native wall time 4.58 s. Final checkpoint SHA256 is `056818f3947c3e7efb1a8050521cfed72f6147c2444b21b02c7aa7ef0166b5d7`. Post-update mean legal-distribution KL is `0.000495983981592`, maximum KL `0.0246379835966`, final sampled-action clipped fraction `0.000800975100122`. These are post-training diagnostics, not evidence of live improvement. Initial acceptance budgets do not imply post-update policy guarantees. See `evidence/training.stdout.jsonl` and `evidence/training.time.txt`.
