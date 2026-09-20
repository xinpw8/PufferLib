# Frozen keyboard-reset policy evaluation

Acceptance failed: the candidate reached its third loss after seven full
rounds, finishing 4W/3L and 76:75 points. The predeclared early-stop rule ended
this cohort; the unplayed 13 rounds are not fabricated or classified as losses.
Even winning all 13 could yield only 17/20. This actor is not promoted as a
consistent fighter. Every completed round passed the existing strict checks.

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
| r44 | Loss | 6:13 | 6 + 0 | 8 + 5 | Pass |
| r45 | Win | 9:5 | 4 + 5 | 5 + 0 | Pass |
| r46 | Loss | 7:16 | 2 + 5 | 11 + 5 | Pass |
| r47 | Win | 21:4 | 6 + 15 | 4 + 0 | Pass |
| r48 | Win | 11:5 | 1 + 10 | 5 + 0 | Pass |
| r49 | Loss | 4:18 | 4 + 0 | 13 + 5 | Pass |

Final cohort: four wins, three losses, 76:75 points. Non-five-point awards are
31:60 and five-point awards 45:15. The narrow total-point lead does not establish
superior ordinary striking or satisfy the predeclared win target. The actor,
configuration and seed remained frozen throughout; none of these rounds was
used for training or tuning within this cohort.

All seven existing strict analyses passed after exact owned-client closure.
There were 40,806 predictions, 40,799 locally applied requests, 301 in-flight
source skips, and seven terminal-race rejections. All 41,114 source referee
payloads were available and verified. Maximum referee receipt age was
0.132494 s and maximum applied-control gap 0.072944 s. r46 retains one explicitly
censored referee-call gap. r47 retains the final five-point award after the
active timer first reached zero.

The contact analyzer recorded 566 attack requests, of which 565 were locally
applied with native dispatch evidence. The additional r49 category30 request
was rejected at the terminal race, with zero native dispatch or outbound
projection. Request categories include 173 right hooks and one left-front kick.
The left-front request in r49 was locally applied and dispatched at captured
gap 0.622662 and absolute rendered bearing 1.83135 degrees. These are request
contexts, not evidence of server playback, a successful contact or a trip.

r43's exact owned client closed before offline
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

| Additional derived summary | SHA256 |
| --- | --- |
| r44 | `422b62929762999e7edd964d4727ee0fac5959a47ee823b0e35c3e322528db39` |
| r45 | `bb6c828f02e77322ac45fec3fb79ed00ba02186b040849beaaf081f36f6b05b0` |
| r46 | `fb409cf517e591600eb9111ffc713f30a3ace0b59be979b7cd9dec5d671e7606` |
| r47 | `a06ed990c05e0bc65bb9947c2f48a08e5fbb98d5db6594d9e01ef3268fdb4f61` |
| r48 | `bca65141fba69a50a5fc9d738d43d37fdf57806d4c4e18f84b3bac538efd8cc5` |
| r49 | `b06d3963929202ce78a4384c47dfb21c76fc20f3b4c52771cbfa4cb517e592ea` |

## Archive

The seven complete trial trees, exact native recordings, derived analyses,
configuration and predeclared plan are preserved on the existing evidence
server under
`pufferlib/rek-evidence/2026-09-19/consistent-fighter-r1/windows-keyboard-frozen-evaluation-r1`.
All 341 copied files, totaling 2,793,870,705 bytes, passed matching source-before,
destination and source-after SHA256 checks. No source file was removed.
Archive-manifest SHA256:
`5313f166e971255567bc22d8dcfcded5407c3eff81120d14b743921a4f3c39cf`.
