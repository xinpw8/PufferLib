# Positive contact-geometry diagnostic

`contact-potential-model.json` fits only the 12 local, uniquely paired strike/score observations in round 1. `contact-potential-diagnostics.json` evaluates the 10 round-2 local strikes and reports training leave-one-out distances. AI events and five-point awards are excluded. There are no action-execution labels, negative examples, inferred misses, hit-probability estimates or accuracy claims.

Features are `[distance, cos(bearing), sin(bearing)]`. Distance is the numeric Unity-world-unit XZ pelvis separation at score receipt. Bearing uses native G1 pelvis-local +X, with `heading=atan2(world_x, world_z)` and `bearing=wrap(target_heading-local_heading)`. The simulator mapping **1 Unity unit = 1 candidate unit is an explicit experiment assumption**. It is not a measured metre calibration. Poses are independently received preceding packets, not synchronized authoritative contact poses.

Each feature scale is its population standard deviation across the 12 training positives only. Zero or nonfinite scales are rejected. For feature vector `f`, standardized nearest-anchor distance is `d = sqrt(min_a sum_j ((f_j-a_j)/scale_j)^2)`. Potential is `Phi=-d/(1+d)`, within `[-1,0]`. This expresses proximity to observed positive geometry. It does not estimate whether an attack will hit. Leave-one-out diagnostics omit the queried anchor but retain the full round-1 feature scale.

Source report SHA-256: `6cf6f73c75638b1364c710c49c221e6d9d3489973ca99392a026792def46cfe7`.
Model ID: `5ae0dd200cdf0fb6da4fceed477baf293fe388f97e627f3ca9eb026166ebab1b`.
The model ID hashes recursively key-sorted JSON of the model body excluding `model_id`. Source bytes remain unchanged. Only selected numeric geometry, event references, explicit assumptions and hashes are exported; private source paths, identities and raw poses are omitted.

From repository root:

```powershell
node --test ocean/rek_g1/native5/contact_potential_data.test.cjs
node ocean/rek_g1/native5/contact_potential_data.cjs "\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-17\human-gameplay-win10-0731\training-signal-analysis\contact-signal-report.json" NEW_OUTPUT_DIRECTORY
```

The exporter refuses to overwrite existing model/diagnostic outputs. All eight tests passed, covering holdout isolation, malformed data, duplicated events, excluded AI/referee awards, angular wrapping, scale estimation and the potential formula. These are offline diagnostics on two rounds from one session, not a population-level or authentic-physics validation.

## Native CUDA reward experiment

The optional `REK_FAST_CONTACT_POTENTIAL` path loads this numeric model once during native startup. Its evaluation executes inside the existing fused CUDA environment step. Python, host physics stepping and per-step host reward calculations are absent. The default remains unchanged when the option is absent.

The point objective remains `own_awarded_points - opponent_awarded_points`. The experimental addition is `weight * (gamma * Phi(next) - Phi(current))`, with terminal potential zero and the same discount as the learner. This gives credit for improving positioning without adding points to the scoreboard or treating proximity as a landed strike. The discounted shaping sum telescopes to an initial-state constant; repeatedly moving between two positions cannot create an additional discounted objective. Native tests include long cyclic sequences and the terminal correction.

The runtime uses the rendered pelvis heading, matching the evidence's measured pose basis. The signed bearing is negated because the candidate uses `atan2(Unity Z, Unity X)` while the evidence uses `atan2(Unity X, Unity Z)`. An independent check of all 186 original request geometries gave a maximum angular-conversion discrepancy of `1.6341095143701523e-15` radians.

Training comparisons start from the same private `dd335d696ab0f2ae891b448d174e3615d834cd5b8fd4cf8cffbae6c5126f4c46` checkpoint. Each continuation uses 33,554,432 learner transitions, 512 arenas, horizon 128, the nine-target primitive model with eight temporal samples, rendered-pose observations, the reconstructed Bot 1 controller, and 120-second rounds. The predeclared shaping weight is 1; it is not tuned on held-out scores. Frozen evaluation disables shaping and measures actual candidate points and round results. These are candidate-simulator tests, not authentic REK policy evaluations.

The separate small-network experiment uses observed future score receipts as targets. A zero target means no eligible score was recorded in a fully covered future interval. It does not mean the requested move was accepted, executed or mechanically missed. Issued-request history is input context, never an executed-action label. The second round is excluded from fitting and normalization.

The 18 September paired scabnft/moogleod session is video-only and is not included in these geometric or network training rows. The recorder's earlier scope expiry prevented native telemetry from that session. The nine exported videos remain useful visual evidence; they do not supply measured move IDs, distances or angles.
