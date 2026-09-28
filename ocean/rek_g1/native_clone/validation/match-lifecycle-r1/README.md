# Explicit native match lifecycle

The training entrypoint resets the entire combat arena after every `ROUND_ENDED`. That clears `rounds_won`, so repeated rounds cannot produce a full match. This isolated additive path preserves the existing fight state and calls its original transition functions. It does not infer a winner externally.

## Integration

Copy the three files in `source/` together. Replace the reused `g1_native_combat_cuda.o` with the compiled object in `cuda-build-r1/`; rebuild the runtime caller. Other fight/combat source and state layouts are unchanged: state size252 bytes, `reset_pending` offset248.

The caller opts in to both APIs:

1. `rek_g1_cuda_native_combat_begin_tick_match`: same arguments as `begin_tick`, with `uint8_t countdown_complete` appended **after** `cudaStream_t stream`.
2. `rek_g1_cuda_native_combat_post_step_deferred_match`: exactly the argument list of `post_step_deferred`.

Use both together. Original APIs retain their default training behavior. The parent runtime owns strict `REK_NATIVE_MATCH_MODE=0|1` parsing and selection.

Each begin call represents20ms. On an active round, it preserves combat state and clears only tick outputs and the legacy episode-reset request. For `BETWEEN_ROUNDS`/`OVER`, it calls `rek_g1_fight_advance_transition(...,0.02f,...)`. The existing state machine supplies the five-second delay, next round number,30s redo/120s normal duration and match winner. Its FP32 countdown is quantized to the controller tick, within one20ms tick of5s. Ending partway through a controller tick introduces up to a controller-tick boundary uncertainty.

`ROUND_PREPARED` emits exactly one `episode_reset=1` pulse while leaving the phase `ROUND_COUNTDOWN`. The caller must complete its existing physical/body/controller reset during that tick. Only a later begin call with `countdown_complete=1` activates the round. With0, countdown stays inactive. The clone chooses1, explicitly omitting the unknown Unity presentation Timeline. Initial `combat_arena_init` already activates the first round immediately; that existing cold-start countdown omission remains.

Inactive match phases skip contact assembly and fall/referee/score/round-clock processing. Deferred observation input validation is retained. Existing active substeps and the next-physics-step counted reset path remain unchanged. Match-end outcome and wins survive `OVER -> IDLE` and remain available until explicit cold reset. No automatic new match is created.

The caller must preserve redo duration: its configurable normal-round override should exclude `current_round_is_redo` in match mode. A separate UI wall-clock pause should not be added to the native between-round clock.

## Validation

`run_cpu.sh` executes exact copied CUDA kernel bodies with serial CPU launch indices and links the unchanged fight/fall/combat/hit state machines. This is host execution of the kernels, not a GPU result.

- 3,704 focused assertions:2wins; native5s transitions; tie ->30s redo ->120s normal; tie/loss/tie native0:1 match; body reset before activation; caller-held countdown; outcome retention; explicit cold reset; active-mode bit parity; counted-reset parity; rejected requests remain atomic; legacy training episode reset retained.
- 705 unchanged fight-state assertions and69 unchanged combat checks pass.
- CUDA compilation with the parent's prior sm_121 precision flags exits0 with empty stderr. Object SHA256 `3bd88b29ec2cf9d436ee5592ba6acf70243b95e80990dc52dfc0012f805c68ce`.
- No GPU execution, game launch, UI input or production edit was performed by this task.

`SOURCE-PINS.json`, `match-lifecycle.diff`, `cpu-run-r1/` and `cuda-build-r1/` preserve source and results. The kernel CPU copy is derived by retaining the exact CUDA source through the namespace closing boundary, before host launch wrappers. Re-run in a fresh output revision to preserve these receipts.

## Scope

This fixes a demonstrated adapter reset error using the already recovered state-machine functions. It does not establish authoritative-server timing, original subscription order, full physics parity or an exact Unity countdown duration. Root's integrated GPU test is still required before calling the runnable match path verified.
