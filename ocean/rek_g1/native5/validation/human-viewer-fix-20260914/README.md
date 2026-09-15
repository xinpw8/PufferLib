# Human evaluator input, clock and score repair

The original human input whitelist rejected 49 of the 64 possible WASDQE
held-key sets, including W+A. The evaluator inherited 20-second benchmark
rounds, advanced before the human started, and discarded terminal score
visibility at the next native automatic reset.

Validation executed on Windows and Spark:

```sh
node --test ocean/rek_g1/league/*.test.cjs ocean/rek_g1/league/public/app.test.cjs
```

All 76 tests passed. Coverage includes all 64 key sets, 4,096 transitions,
HTTP acceptance, attack locking, private human-duration overrides, session
score preservation, pause/resume and checkpoint-specific training labels.

`native_smoke.cjs STAGE_DIRECTORY` was executed on `spark-4ae3` against an
isolated server on port 18770. `native-smoke.json` records the worker hash,
accepted inputs, actual movement/kick categories and native terminal score.
The original 20-second configuration and production ports were untouched by
that probe. These checks are evaluator functionality checks, not training SPS
or authentic REK parity measurements.

Browser checks on the isolated viewer confirmed Resume advances the clock,
Pause stops it, and loading a 120-second evaluation produces a paused native
snapshot with tick zero and 120 seconds remaining. The updated live 18769
viewer was then opened and read back: paused, 5:00, no input error, zero initial
round/session scores, and the recorded 2,508,920 training SPS visible above the
arena. The selected 1M sampled opponent and human blue side were preserved.

The native binary was unchanged. Human 120/300-second rounds are separately
labeled because timer values enter the policy observation. Ranked fixtures
remain on their original 20-second protocol; three knockdowns can terminate
a human round before its selected time limit.
