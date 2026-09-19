# Recovered balance and referee boundary

`recovered_balance.cuh` composes the existing fall classifier and combat
coordinator without changing their rules. It accepts explicit dynamics,
contact-stream, recovery, early-gate and clock facts. Every required input
must be marked measured or modeled. Missing fields return `MissingInput`;
overlapping or unknown provenance bits return `InvalidInput`. Both errors
leave state and output unchanged. Modeled provenance records an assumption;
it does not establish agreement with a live game.

The game-build fingerprint declared by both recovered rule headers is
`f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659`.
This test work did not independently hash the currently running game's
assembly. Revalidate that fingerprint and the source hashes in the validation
report before claiming parity with another build.

## Integration

1. Allocate one `rek5_balance::State` per arena. `init_active` uses the existing
   immediate active-round episode boundary. It assumes neither an external
   countdown duration nor a physical spawn operation. Production reset and
   countdown lifecycle remains caller-owned.
2. Supply `Input` after each dynamics substep. The two `fixed_delta_seconds`
   values must equal the explicit shared `delta_seconds`. The round clock is
   separately supplied, clamped, finite, nonnegative and nonincreasing.
3. Derive tilt against the calibrated upright reference; derive pelvis-height
   ratio from height above the floor divided by calibrated standing pelvis
   height. The existing `g1_fall_mujoco` adapter provides that measurement
   contract. Count distinct non-foot rigid bodies touching the floor, rather
   than manifold points or renderer-sphere overlaps. `CanGetUp`, tracking and
   recovery must come from a known policy/runtime configuration or model.
4. Resolve `detector_tick_enabled` from the native early gates. A disabled
   detector tick preserves its timers and phase. Referee time still advances.
   Complete dynamics remain required and validated during a gated tick.
5. Certify `CompleteContactStream` only when all contact fields and nested
   strike intents are known, including a genuinely observed or explicitly
   modeled empty stream. The existing hit detector processes those contacts
   before fall callbacks, preserving attribution even when no score is given.
6. Consume `Result` only on `Ok`. `next_state.combat.fight.falls` and
   `next_state.fall[].phase` are the recovered outputs. Preserve the events,
   score deltas and referee calls for replay diagnostics.
7. On `physical_reset_pending`, perform the physical two-fighter spawn reset
   and clear any caller-owned dynamics/contact caches. Acknowledge completion
   with `acknowledge_spawn_reset(..., 1, ...)`. Further steps return
   `PhysicalResetPending` until then. Acknowledgement preserves scores, fall
   totals and time, clears native hit/strike history, and applies the native
   2 s detector grace to both fighters.

No root-height, orientation, dynamics, impulse-to-fall or hit-count-to-fall
model is supplied. A fixed root pose and absent floor/body contacts cannot
produce actual-fall parity merely by using this classifier. Get-up execution,
recovery settle/handoff and non-active round transitions are outside this
adapter. Recovery-capable fixtures verify the native 10/20 s count and KO
resolution; they do not simulate a completed get-up. The existing native APIs
remain the authority for those broader lifecycle transitions.

## Compilation

For host code include `recovered_balance.cuh` before other recovered headers.
Its native host declarations use C linkage. Compile the original
`g1_fall_state.c`, `g1_fight_state.c`, `g1_combat_tick.c`, and
`g1_hit_detector.c` as C11 and link those objects.

For a separate CUDA translation unit define `REK_G1_CUDA_DEVICE` before any
recovered header. Compile each original module through `g1_cuda_device.cu`
with `-dc -DREK_G1_CUDA_SOURCE=\"module.c\"`, then device-link its objects with
the adapter kernel. Do not include `recovered_contact_rules.cuh` in that same
translation unit: it independently overrides the global qualifier macros.
The executable may link both translation units. The test script demonstrates
the host/device linkage and uses precise floating-point flags.

Run `bash test_recovered_balance.sh NEW_BUILD_DIRECTORY` for CPU checks or
append `cuda` for the CPU/CUDA batch comparison. Build directories must be
new so existing artifacts are preserved.

## Verified scope

On 2026-09-19, the native C11 CPU oracle passed 2,048,033 checks under WSL
Ubuntu 22.04 and on Spark. Spark's NVIDIA GB10 with CUDA 13.0.88 passed
2,052,130 checks including 4,096 parallel arena comparisons. The GPU and CPU
produced identical per-step state/event digests. Fixtures cover strict tilt
and height thresholds, missing-contact rejection of fall qualification,
fast/slow contact dwell, untracked fallback, early gates, falling clearance,
attribution without scoring, estop slip, single/double fall counts,
3/10/20 s deadlines, +5 scoring, reset acknowledgement, bilateral reset and
the exact 2 s grace gate. Every required availability bit is also individually
removed and checked for atomic rejection. Fixtures are synthetic probes of
rules, not live trajectory validation.
