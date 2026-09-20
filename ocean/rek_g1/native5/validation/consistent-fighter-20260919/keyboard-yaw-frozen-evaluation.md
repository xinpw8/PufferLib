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

No acceptance round has completed at this report's initial publication.
