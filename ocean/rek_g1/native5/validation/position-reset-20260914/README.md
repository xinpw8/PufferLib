# Position reset diagnosis

Isolated native CUDA evaluation on `spark-4ae3`. No policy was loaded. Orange
received external actions while blue received neutral category 1. The live
browser and Windows applications received no input from these tests. Physics
and state updates used the native GPU runtime; Node only drove the diagnostic
protocol. Full snapshots, private asset configuration, and model files remain
on Spark and are not included here.

## Measured V2 behavior

`baseline-full.json` contains 17 complete, single-edge move trials with orange
displaced 1.180000 m from spawn and blue 2.980000 m away. All 17 had exactly zero
planar displacement after settling, zero hits, zero falls, and zero spawn
resets. The same run tested U and I at eight ranges each. U produced no hits at
these fixtures. I produced one hit at each of the four closest fixtures and no
hits at the four farther fixtures. None of these single attacks produced
multiple hits or a knockdown.

`baseline-paired-i.json` then reproduces a different, accumulated-hit case:

| Event | Native tick | Orange score | Blue falls | Orange position change |
| --- | ---: | ---: | ---: | ---: |
| First I contact | 147 | 1 | 0 | 0 m |
| Second I contact | 547 | 7 | 1 | 0 m |
| Automatic pose reset | 572 | 7 | 1 | 1.159999877 m to spawn |

Each kick was entered exactly once, with 100 idle ticks between the two
300-tick attack observations. The second contact caused the old synthetic
damage threshold to award five additional points and reset both robots after
25 ticks (0.5 s). Orange was the attacker and had not fallen. This is a
reproduced bug in the candidate. It does not establish that this was the exact
event observed during the user's earlier live session.

`baseline-v2-asserted.json` repeats that case with explicit assertions, then
continues to the legitimate timed terminal at tick 15000 (300 s). The next
step resets the round at tick 15001. All assertions passed.

## Measured V3 correction

`v3-paired-i.json` repeats the same deterministic fixture against the V3
worker. The contacts still occur at ticks 147 and 547. Scores are now one and
two, falls remain zero, and maximum planar movement is exactly zero. Orange
remains 1.159999877 m from spawn throughout both attacks and subsequent idle.
The normal timed terminal still occurs at tick 15000, followed by the valid
round-two spawn reset at tick 15001. Every assertion passed, exit code zero.

`v3-full.json` repeats all 33 original no-contact/range fixtures. Their trial
results are exactly equal to the V2 results. This correction removes the
unmeasured damage-derived knockdown transition. The compact candidate does
not model genuine balance loss or knockdowns, and these tests establish no
parity with the authentic simulator.

## Independent pose discontinuity

`baseline-pose-transitions.json` measures rendered root pose across the same
17 no-contact trials. The largest action-to-idle transition was category 25:
root height changed by 0.065100014 m and root orientation by 40.300518 degrees
in one tick. Logical yaw and planar position did not change. This is separate
from the contact-triggered spawn reset. `analyze_pose.cjs` performs this
measurement using private snapshot logs and emits only derived pose metrics.
`v3-pose-transitions.json` confirms that all 17 pose-transition measurements
are unchanged in V3. The height/orientation discontinuity remains unresolved.

## Reproduction and evidence

Run from the repository's native5 directory using an isolated evaluator build
and the actual local asset configuration:

```sh
REK_EXPECT_NO_HIT_RESETS=0 bash fast_position_probe.sh \
  EVAL_BUILD_V2 BASE_ASSET_CONFIG NEW_OUTPUT paired_i
REK_EXPECT_NO_HIT_RESETS=1 bash fast_position_probe.sh \
  EVAL_BUILD_V3 BASE_ASSET_CONFIG NEW_OUTPUT paired_i
bash fast_position_probe.sh EVAL_BUILD BASE_ASSET_CONFIG NEW_OUTPUT all
```

The expectation flag only changes assertions. It does not change runtime
behavior. `0` asserts the old failure, while `1` requires two contact points,
zero falls, and zero planar movement from the two kicks. Both also verify the
300 s terminal and the next-round reset.

The `*-provenance.txt` files contain executed commands, host identity, exit
codes, and SHA-256 hashes for scripts, worker, runtime object, asset object,
and private configuration. `baseline-probe-source.cjs` and
`baseline-probe-source.sh` preserve the exact R2 diagnostic source; their
hashes are `8cc4639bb32f2d81c89d89ba8b2de5525be8446486e0549dccdffed5ffb1809f`
and `c4d396d6289d2744a874e07798d9dc0cbe42b7bb33829f5ce0132f5b219854b2`.
For replay, those preserved sources must be placed back at their original
native5 paths so the relative worker import resolves.
