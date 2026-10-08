# Claudiolo

A hand-built REK G1 fighter that plays through the keyboard channel only.
Goal: beat Sparring Bot 1 in a private room in the authentic `rek.exe`, and
beat the GPT-trained MinGRU policies. Plan and rationale:
"Claudiolo: plan of attack" (the project's plan doc).

Claudiolo is not a neural network. It exploits the recovered Sparring Bot 1
decision loop (`../native5/native_bot1.cuh`): Bot 1 stands still for a fixed
0.3 s before every swing and then sends zero velocity and zero yaw for the
whole canned move plus 0.258 s. Claudiolo strikes inside that settle delay,
steps out of live strikes, punishes the frozen windows, never pushes into
contact (any fall is +5 to the other side) and runs the clock when ahead.

## Files

| File | What it is |
| --- | --- |
| `core.cjs` | Perception, opponent tracker, tactics, online reach learner, human-legal governor |
| `bot1.cjs` | Bit-exact JS port of the recovered Bot 1 AI (20,000-step parity test vs the C++ header) |
| `moves.cjs` | Build-pinned move table: categories, keys, durations, impact windows, points |
| `surrogate.cjs` | Calibrated planar proxy of a round for logic tests (not a parity model) |
| `calibrate.cjs` | Fits the proxy's free hazards to the authentic passive-defender cohort |
| `worlds.cjs`, `tune.cjs` | Physics variants and a robust parameter search across them |
| `adapters/clone.cjs` | Drives the native clone worker (`rek-eval-worker` JSONL) as the human side |
| `adapters/league.cjs` | Claudiolo vs every registered GPT checkpoint, side-reversed, Wilson bounds |
| `adapters/live.cjs` | Authentic `rek.exe` through the RekUiBridgeAgent policy stream |

## Human-legal input

Every output passes `Governor`: only the 33 keyboard categories (release,
W/S/A/D/Q/E and translation+turn pairs, 17 bound moves), at least 60 ms
between key-state changes and at most 12 per second (Space chords and double
taps count as two presses), no strike while translation is held or the game's
mask forbids it, one move in flight, no quit/pause/menu/disconnect actions.
`reactionDelay` adds an artificial perception delay (tests run 150 ms).
Perception uses only what the screen shows: both robots' poses, the clock and
the score. Claudiolo never touches GPT checkpoints or their processes.

## Run

```sh
node --test test/*.test.cjs                     # parity, legality, adapter tests

# Native clone vs recovered Bot 1 (worker config selects the opponent)
node adapters/clone.cjs --worker /path/rek-eval-worker --config worker.json \
  --rounds 20 --out /new/dir

# Native clone vs every GPT checkpoint in a league backend
node adapters/league.cjs --server /private/run/server.json --backend mujoco \
  --out /new/dir --seeds 1,2,3,4,5 --round-seconds 120

# Authentic rek.exe, private room, Sparring Bot 1
node adapters/live.cjs trial.json   # {"relay":[...], "out":"/new/dir", "max_seconds":150, "enter_private":true}
```

## Evidence status

Surrogate results are development signals only. Acceptance follows the plan's
validation ladder: native clone vs Bot 1, native clone vs GPT checkpoints,
then 18 of 20 fresh authentic private-room rounds against Sparring Bot 1.
