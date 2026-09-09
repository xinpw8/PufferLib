# REK T800 strategy router

This directory contains the fail-closed action router for the measured T800
strategy interface. It is not a stepping environment and has no Puffer binding.

The supported initial Puffer encoding is `MultiDiscrete{3,3,3,7}`:

- forward, strafe, yaw: categories `0,1,2` map to `-1,0,+1`
- move: category `0` emits no request; categories `1..6` map to REK move slots
  `2,3,4,5,9,10`

The three velocity categories are held input state, sampled again on every
environment step. Forward and strafe may overlap yaw. A nonzero move category
does not start while forward or strafe remains held; keeping that category held
starts the move only after translation is released and the executor reports its
measured local translation below the model-specific strict settle threshold.
Yaw alone does not block a move. An accepted move clears the physical forward,
strafe, and yaw command for its full execution while preserving the held input
state for observation and post-move resumption. Whether the real controller
buffers a brief attack tap that is released before translation settles remains
unmeasured; this router requires the attack category to remain held.

`F` is not an action head or alias in this interface. No verified G1 binding or
controller meaning for `F` is available in the current evidence.

The current Puffer training backends do not support a mixed continuous and
categorical policy head. This discrete encoding exactly covers the measured
keyboard endpoints. A future hybrid-policy backend can expose continuous
velocity without changing the move categories.

`strategy_router.h` refuses initialization unless locomotion, canned-move, and
`DriveRecovery` executors are all declared present. Recovery has priority over
learned locomotion and attack. Nonzero move categories are one-shot requests;
category zero rearms the same move after it has been emitted.

No executor is implemented here. In particular, the existing `rek_sandbox`
root stabilizer, dummy wave, fall reset, and 25 joint-action interface are not
substitutes for the missing T800 executors. The measured interface and remaining
unknowns are pinned in
`../rek/evidence/evidence_out/t800_strategy_contract_v2_20260903.json`.

Compile the pure router test with:

```bash
cc -std=c11 -Wall -Wextra -Werror -pedantic test_strategy_router.c -o test_strategy_router
./test_strategy_router
```
