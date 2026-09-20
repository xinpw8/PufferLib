# Frozen keyboard-reset policy evaluation

## Plan fixed before collection

The private plan was written and hashed before the first acceptance fight:
SHA256 `8471691b9dae0bda3e56cf5fec2ef008f63c351d2e04fad5e9cb4288947a3aeb`.
The six development trials r37-r42 are excluded. Their treatment finished
2 wins / 1 loss, 46:36 awarded points. Development outcomes selected this
candidate; they cannot establish held-out consistency.

Frozen checkpoint:
`85808a6731f4f40f5756800a9edf93faa5c312aa7d5f7468cafdd4ae3a5c381f`.
Configuration SHA256:
`8ab8e31eb3af3946df9420ec331be1088cbea9f2148bd76a6ab54c25a1a63eda`.
Existing isolated trial helper SHA256:
`469ec2192bf22266b73e74b8f3285e0030d0e01a9416117879c25e167972e5f2`.

Authentic private Sparring Bot 1, difficulty 0, 120-second rounds. Native BF16
sampled inference, seed 73, legacy-v1 observations and action stride 1 remain
unchanged. The Windows client runs on the isolated desktop; native inference
runs on Spark. No global desktop input or human opponent is permitted.

Starting at r43, the existing operational consistency target is at least
18 wins in 20 fresh full rounds, positive aggregate score margin and valid
full-round control/referee coverage. Ties are non-wins. Preserve every failed
attempt and distinguish incomplete infrastructure failures from fighting
outcomes. Stop this candidate early upon its third completed non-win because
18/20 then becomes impossible. Do not reset or retry a poor fighting result.
No checkpoint, reward, encoder or configuration tuning is allowed within this
cohort. Any later changed candidate requires a new evaluation.

This engineering target does not imply a guaranteed population win rate of
90 percent. Report ordinary and five-point awards separately, and do not infer
move-specific contact causality from local request acknowledgements.

## Results

| Attempt | Outcome | Policy:bot points | Policy non-five + five | Bot non-five + five | Existing strict checks |
| --- | --- | ---: | ---: | ---: | --- |
| r43 | Win | 18:14 | 8 + 10 | 14 + 0 | Pass |

Current cohort: one win, zero losses, 18:14 points. Evaluation remains in
progress; no acceptance claim. r43's exact owned client closed before offline
validation. Maximum applied-control gap was 0.059948 s; all 5,796 referee
sources were available. One terminal-race rejection and 41 in-flight source
skips remain in the record. All 68 attack requests have local dispatch evidence;
none selected left-front category 17. No attack-to-award causality is inferred.

Private outputs: `keyboard-yaw-frozen-evaluation-r1/r43` under
`C:\rekagent\work\consistent-fighter-20260919-r1`, with the existing referee and
contact outputs in `live-round_outcome_v1-r43`. Derived-summary SHA256:
`f8d591d653f0c8224871af2c724e4001b24efbe13f1467cc707c6667ef144589`.
Exact owned PID 337348 native-capture SHA256:
`130ff4f803f47f702e3379c5fc70ae3eee5c633d830dc5ddc09ddce5b9fc0d48`.
