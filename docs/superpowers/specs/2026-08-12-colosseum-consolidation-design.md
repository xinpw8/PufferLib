# Colosseum Consolidation Design

## Goal

Make `valtteri/colo-5c-port` the single source of truth for the latest Colosseum environment and the shared OSRS runtime work. The consolidated environment must load `checkpoints/colo-w11-protein-best/run38_378535936.bin` with its `933`-observation and `451`-mask contract.

## Source Lines

The current 5c port copied the Colosseum tree from `0fc4b50c07cc4ec1435613fe4396b2b77ea8b647`. The missing committed behavior is the net delta through `8281518ea1cd238a59500aebb81937cc607588a0`. The missing uncommitted behavior was preserved in `3812fd454880a4605b2a9b39af41c72514e43a95`.

Port behavior, not branch structure. The current 5c architecture remains authoritative.

## Included Behavior

### Sol laser observations

Restore the honest multi-crystal laser observation pack and its episode diagnostics. Preserve `laser_obs_mode` because run38 records it and requires the enabled mode. The pack exposes the shared cooldown and per-crystal active, line-delta, and freeze state.

### Asset archive provenance

Pin the `osrs-assets-v22` release archive to `2f908da5b5ddf148c0cbef48ad6334c5253c04c6cd83782340bc2f1c5dffc7ad`. A fresh setup download produced this SHA. The prior manifest SHA rejects the published archive and prevents clean worktree builds.

### Manticore lifecycle

Represent the first attack separately from the repeating barrage:

1. Activate for seven ticks after spawn.
2. Wait until the player is in range and line of sight.
3. Start a ten-tick charge without firing.
4. Keep the three orbs hidden for the first three charge ticks.
5. Fire the first orb after the full charge.
6. Continue with the existing one-tick orb sequence, seven-tick lull, and two-Manticore stagger rules.

Use the shared finalized arena topology for line of sight.

### Thrall lifecycle

Remove the summon action. A thrall exists from reset and resummons automatically on expiry. Its lifetime equals the player's current Magic level in the beginner profile and twice that level in the speedrun profile. It acquires a target only after an executed player attack, retains that target for twelve ticks, attacks every four ticks, and never targets Sol Heredit.

### Volatility lifecycle

Do not explode on the lethal hit. Explode when the corpse leaves the simulation after its death linger. Queue one render event in `ColosseumContext` and emit the family-specific spot animation from the render bridge. Use the shared arena topology for blast geometry.

## Interface Contract

The canonical model contract is:

- observations: `933`
- action mask: `451`
- spell head: none or Death Charge
- snapshot version: `25`

Older `934`/`452` checkpoints are intentionally incompatible. No compatibility slots, aliases, or dual model path remain.

## Preserved Current Architecture

Keep these current 5c systems unchanged except where their public call shape must carry the new behavior:

- immutable `EncounterArenaTopology`
- shared route engine
- canonical `uint16_t` inventory content codes
- shared generated item metadata
- current encounter context ownership
- current Inferno, Zulrah, and NH PvP migrations

The WIP's old per-Colosseum item table, item registry accesses, and source-layout edits do not return.

## Excluded Changes

Do not port:

- the unrelated Puffer source-layout and Python migration
- `.cu` to `.inc` comment and include edits
- deleted or replaced per-Colosseum item metadata generators
- stale golden or sim-invariant hashes
- the WIP test's accidental nonstandard FNV seed
- canceled next-style ungating and pending-style schedule experiments

## Verification

Add the WIP behavior tests before implementation and confirm they fail against current 5c. Then require:

- Colosseum mechanics suite
- Colosseum golden suite
- Colosseum simulation-invariant suite
- Colosseum forecast and topology suites
- Colosseum snapshot round trip and stale-version rejection
- worker-count determinism
- shared inventory action, click, and item-effect suites
- focused Inferno mechanics and lab suites
- local Colosseum viewer build
- run38 model load with hidden size 512 and two layers
- a wave-11 viewer smoke run reporting the `933`/`451` contract

Update deterministic baselines only after the new behavior explains every changed digest.

## Integration

Implement on a temporary branch in `/private/tmp`. Commit the focused consolidation, fast-forward `valtteri/colo-5c-port` after verification, then remove the temporary worktree and branch. Leave local checkpoints, binaries, screenshots, and other untracked artifacts untouched.
