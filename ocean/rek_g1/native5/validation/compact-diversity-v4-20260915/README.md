# Compact V4 diversity checks

The standalone native CUDA probe uses the same `fast_runtime.o` as V4
training. `fast_diversity_probe.sh` records its build/run command, host,
source/object/executable hashes, exit code and process timing.

The first completed 64-arena run used seed 73. Mixed opponent selection
produced 14 scripted, 17 neutral, 16 retreat and 17 strafe arenas. Explicit
reset replayed initial positions/headings bitwise. Advancing the round and
changing seed to 74 each produced different fixtures. All sampled gaps,
headings and interior-wall bounds passed. Each explicit opponent mode
generated its specified GPU action.

Potential shaping with weight 0.7, gamma 0.999, target 0.65 m and bearing
weight 0.25 matched a separate arithmetic check with maximum absolute reward
error 1.76395014e-8. Terminal potential was zero and discounted rewards
telescoped to negative initial potential. Shaped and unshaped runs had
bitwise-identical poses, raw observations and masks, with unchanged zero
points and falls. These are transition-contract checks, not policy-strength
claims.

The final `r2` run repeated those checks and rejected all ten invalid-parameter
cases, exit code zero. The current probe source matches its recorded hash.

`paired-i-summary.json` repeats the prior position-reset reproduction against
the V4 evaluator. The two single I inputs contact at ticks 147 and 547, score
one point each, and cause zero falls or planar movement. The legitimate timed
terminal remains tick 15000 and round-two reset tick 15001. Its command was:

```sh
REK_FAST_SHAPING_WEIGHT=0 REK_FAST_RANDOM_RESETS=0 \
REK_FAST_OPPONENT_MODE=scripted REK_EXPECT_NO_HIT_RESETS=1 \
bash SOURCE/native5/fast_position_probe.sh \
  EVAL_BUILD_V4 BASE_ASSET_CONFIG NEW_OUTPUT paired_i
```

The actual source, build and configuration paths and their hashes are in
`paired-i-provenance.txt`. The explicit `scripted` setting does not select an
opponent in this fixture: both fighters receive external diagnostic actions.

No training, Python runtime, CPU physics, graphical interaction or production
service changes were performed by this probe. CPU asset parsing and diagnostic
snapshot assertions are outside its native GPU state transitions.
