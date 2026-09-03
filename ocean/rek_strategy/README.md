# REK T800 strategy router

This directory contains the fail-closed action router for the measured T800
strategy interface. It is not a stepping environment and has no Puffer binding.

The supported initial Puffer encoding is `MultiDiscrete{3,3,3,7}`:

- forward, strafe, yaw: categories `0,1,2` map to `-1,0,+1`
- move: category `0` emits no request; categories `1..6` map to REK move slots
  `2,3,4,5,9,10`

The current Puffer training backends do not support a mixed continuous and
categorical policy head. This discrete encoding exactly covers the measured
keyboard endpoints. A future hybrid-policy backend can expose continuous
velocity without changing the move categories.

`strategy_router.h` refuses initialization unless locomotion, canned-move, and
`DriveRecovery` executors are all declared present. Recovery has priority over
learned locomotion and attack. Nonzero move categories are one-shot requests;
category zero rearms the same move.

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

