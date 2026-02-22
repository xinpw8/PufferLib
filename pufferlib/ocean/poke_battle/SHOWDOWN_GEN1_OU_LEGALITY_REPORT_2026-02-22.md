# Showdown Gen1 OU Legality Parity Report

Date: 2026-02-22  
Repo: `pufferlib`  
Env module: `pufferlib/ocean/poke_battle`  
Primary objective: implement and verify full `[Gen 1] OU` legality parity (team legality + clause parity + hardcoded species moveset legality) against official Pokémon Showdown behavior.

## 1. Authoritative Source of Truth

This audit used the official `smogon/pokemon-showdown` repository at:

- Commit: `95aad7df02abd58dd737e0acdac22e5d049d360e`

Relevant source files:

1. `[Gen 1] OU` format definition
- `config/formats.ts:4138`
- `config/formats.ts:4139`
- `config/formats.ts:4140`
- `config/formats.ts:4141`

This defines:
- `name: "[Gen 1] OU"`
- `mod: 'gen1'`
- `ruleset: ['Standard']`
- `banlist: ['Uber']`

2. Gen1 `Standard` ruleset content
- `data/mods/gen1/rulesets.ts:5`
- `data/mods/gen1/rulesets.ts:6`

This defines:
- Rules: `Obtainable`, `Desync Clause Mod`, `Sleep Clause Mod`, `Freeze Clause Mod`, `Species Clause`, `Nickname Clause`, `OHKO Clause`, `Evasion Moves Clause`, `Endless Battle Clause`, `HP Percentage Mod`, `Cancel Mod`
- Banlist: `Dig`, `Fly`

3. Gen1 Uber species in Showdown tiers
- `data/mods/gen1/formats-data.ts:449`
- `data/mods/gen1/formats-data.ts:452`

This marks:
- `mewtwo: Uber`
- `mew: Uber`

4. Tradeback rule behavior
- `data/rulesets.ts:1806`
- `data/rulesets.ts:1807`
- `sim/team-validator.ts:2520`
- `sim/team-validator.ts:821`

This confirms:
- `Allow Tradeback` is a separate validator rule.
- Base `[Gen 1] OU` does not include `Allow Tradeback`, so Gen2 tradeback learnset routes are disallowed.

## 2. Baseline Gap Analysis (Before This Patch Set)

Three classes of parity gaps were identified.

### 2.1 Fixed-team input legality gap

File: `pufferlib/ocean/poke_battle/binding.c:199`

Observed behavior:
- `parse_team` accepted `species_id = 0` (`SPECIES_NONE`) as valid input.
- This allowed user-provided teams with fewer than six real Pokémon.

Why this is a parity issue:
- `[Gen 1] OU` team validation expects six real Pokémon entries with legal species and no duplicates under Species Clause.

### 2.2 Endless Battle Clause stall detection gap

File: `pufferlib/ocean/poke_battle/poke_battle.h:2643`

Observed behavior:
- Staleness used a very strict full-state signature.
- Any volatile/no-impact variation (e.g., switch patterns) could reset stale counting.

Why this is a parity issue:
- Legitimate no-progress loops should trip endless-battle protection, not require byte-identical state recurrence.

### 2.3 Hardcoded species moveset legality gap (149-species table)

File: `pufferlib/ocean/poke_battle/poke_battle.h`

Method:
- Used Showdown’s own TeamValidator for `[Gen 1] OU`.
- Constructed one test team per species with that species as slot 1 + legal fillers.
- Validated all 149 modeled non-Uber species.

Initial result:
- 16 species failed legality validation.

Raw failure set (from validator output):
- Caterpie: duplicate `Tackle`
- Metapod: duplicate `Harden`
- Weedle: duplicate `Tackle`; `Weedle can't learn Tackle`
- Kakuna: duplicate `Harden`; incompatible `String Shot + Harden` combo
- Pidgey: `Tackle` cannot be transferred Gen2->Gen1 (tradeback-only path)
- Pidgeotto: same as Pidgey
- Pidgeot: same as Pidgey
- Vulpix: `Hypnosis` tradeback-only
- Meowth: `Hypnosis` tradeback-only
- Psyduck: `Hypnosis` tradeback-only
- Ponyta: `Hypnosis` tradeback-only
- Exeggcute: `Mega Drain` tradeback-only
- Rhyhorn: `Blizzard` tradeback-only
- Mr. Mime: `Hypnosis` tradeback-only
- Magikarp: duplicate `Splash`
- Ditto: duplicate `Transform`

## 3. Implemented Remediation

## 3.1 Fixed-team parser hardening

File changes:
- `pufferlib/ocean/poke_battle/binding.c:191`
- `pufferlib/ocean/poke_battle/binding.c:199`
- `pufferlib/ocean/poke_battle/binding.c:229`

What changed:
- Added `is_ou_legal_species(long species_id)` against `OU_LEGAL`.
- `parse_team` now rejects:
  - `0` (`SPECIES_NONE`)
  - out-of-pool IDs
  - duplicate species IDs

Error behavior:
- Raises `ValueError` with explicit OU-legality context.

## 3.2 Endless-clause progress signature coarsening

File change:
- `pufferlib/ocean/poke_battle/poke_battle.h:2643`

What changed:
- Reworked `battle_progress_signature` to track coarse progress fields:
  - per-side alive count
  - per-mon species, HP, status, sleep turns, toxic counter, alive flag
- Dropped highly volatile fields from stale signature:
  - active index
  - stat stages
  - confusion/evasion/accuracy volatile state
  - substitute/reflect/light-screen/recharge/trap internals
  - status source-side tags

Rationale:
- Makes stale detection robust to switch loops and equivalent no-progress state churn.

## 3.3 Species moveset legality corrections

File changes:
- `pufferlib/ocean/poke_battle/poke_battle.h:552`
- `pufferlib/ocean/poke_battle/poke_battle.h:554`
- `pufferlib/ocean/poke_battle/poke_battle.h:558`
- `pufferlib/ocean/poke_battle/poke_battle.h:560`
- `pufferlib/ocean/poke_battle/poke_battle.h:564`
- `pufferlib/ocean/poke_battle/poke_battle.h:566`
- `pufferlib/ocean/poke_battle/poke_battle.h:568`
- `pufferlib/ocean/poke_battle/poke_battle.h:606`
- `pufferlib/ocean/poke_battle/poke_battle.h:636`
- `pufferlib/ocean/poke_battle/poke_battle.h:638`
- `pufferlib/ocean/poke_battle/poke_battle.h:678`
- `pufferlib/ocean/poke_battle/poke_battle.h:720`
- `pufferlib/ocean/poke_battle/poke_battle.h:736`
- `pufferlib/ocean/poke_battle/poke_battle.h:752`
- `pufferlib/ocean/poke_battle/poke_battle.h:762`
- `pufferlib/ocean/poke_battle/poke_battle.h:766`

Species-level move-set substitutions:

| Species | Previous 4-slot set | New 4-slot set | Reason |
|---|---|---|---|
| Caterpie | Tackle / String Shot / Tackle / String Shot | None / None / Tackle / String Shot | Remove duplicate active moves |
| Metapod | Harden / Tackle / Harden / Tackle | None / Tackle / Harden / String Shot | Remove duplicate active moves; legal pool |
| Weedle | Tackle / String Shot / Tackle / String Shot | None / None / None / String Shot | `Tackle` illegal in no-tradeback Gen1; remove duplicates |
| Kakuna | Harden / String Shot / Harden / String Shot | None / None / None / String Shot | Remove duplicate/incompatible combo |
| Pidgey | Rest / Double-Edge / Tackle / Agility | Rest / Double-Edge / Substitute / Agility | `Tackle` flagged tradeback-illegal in this context |
| Pidgeotto | Rest / Double-Edge / Tackle / Agility | Rest / Double-Edge / Substitute / Agility | Same as above |
| Pidgeot | Rest / Hyper Beam / Double-Edge / Tackle | Rest / Hyper Beam / Double-Edge / Agility | Same as above |
| Vulpix | Hypnosis / Fire Blast / Double-Edge / Body Slam | Rest / Fire Blast / Double-Edge / Body Slam | `Hypnosis` tradeback-only |
| Meowth | Hypnosis / Double-Edge / Body Slam / Slash | Thunderbolt / Double-Edge / Body Slam / Slash | `Hypnosis` tradeback-only |
| Psyduck | Hypnosis / Surf / Bubble Beam / Blizzard | Body Slam / Surf / Bubble Beam / Blizzard | `Hypnosis` tradeback-only |
| Ponyta | Hypnosis / Fire Blast / Double-Edge / Body Slam | Rest / Fire Blast / Double-Edge / Body Slam | `Hypnosis` tradeback-only |
| Exeggcute | Sleep Powder / Psychic / Explosion / Mega Drain | Sleep Powder / Psychic / Explosion / Double-Edge | `Mega Drain` tradeback-only |
| Rhyhorn | Rest / Earthquake / Rock Slide / Blizzard | Rest / Earthquake / Rock Slide / Body Slam | `Blizzard` tradeback-only |
| Mr. Mime | Hypnosis / Psychic / Hyper Beam / Thunder | Body Slam / Psychic / Hyper Beam / Thunder | `Hypnosis` tradeback-only |
| Magikarp | Splash / Tackle / Splash / Tackle | None / None / Splash / Tackle | Remove duplicate active moves |
| Ditto | Transform / Transform / Transform / Transform | None / None / None / Transform | Remove duplicate active moves |

Note:
- `MOVE_THUNDERSHOCK` constant maps to move name `"Thunder"` in this codebase.

## 3.4 In-engine invariant checks

File changes:
- `pufferlib/ocean/poke_battle/poke_battle.h:2998`
- `pufferlib/ocean/poke_battle/poke_battle.h:3105`

Added:
- Debug-only helper `species_moveset_contains`.
- `validate_battle_rules` assertions to guard against known no-tradeback legality regressions.
- Assertion that each species has no duplicate non-`MOVE_NONE` move IDs.

## 4. Test Coverage Added

### 4.1 Rules parity tests

File: `tests/test_poke_battle_rules_parity.py`

New tests:
- `test_species_clause_rejects_species_none_in_fixed_team` (`:149`)
  - Negative proof: invalid team rejected (`SPECIES_NONE`).
- `test_endless_battle_clause_detects_switch_stall_loops` (`:202`)
  - Positive proof: repeated no-impact switching terminates with stale threshold.

### 4.2 Moveset legality tests

File: `tests/test_poke_battle_moveset_legality.py`

New tests:
- `test_all_modeled_species_have_no_duplicate_non_none_moves` (`:20`)
  - Positive global invariant over all 149 species.
- `test_known_tradeback_illegal_moves_are_not_in_hardcoded_sets` (`:55`)
  - Negative regression list for all species/moves found in validator failures.

## 5. Validation Evidence

## 5.1 Build + local test suite

Venv:
- `/home/spark-advantage/pufferlib/.pufferlib/bin/activate`

Build command:
- `python setup.py build_poke_battle --inplace --force`

Test command:
- `python -m pytest -q tests/test_poke_battle_rules_parity.py tests/test_poke_battle_team_builder.py tests/test_poke_battle_moveset_legality.py`

Result:
- `26 passed`

## 5.2 Showdown validator sweep for all hardcoded species

Tooling:
- Cloned Showdown at the commit above.
- Installed dependencies (`npm ci`) to run `dist/sim/team-validator.js`.

Procedure:
- Parsed `SPECIES_DATA` from `poke_battle.h`.
- Built one 6-mon legal team per species (target species + legal filler species).
- Injected minimal EV marker (`EVs: 1 HP`) to avoid Showdown zero-EV warning.
- Validated each team with `TeamValidator.get('[Gen 1] OU')`.

Final result after patches:
- `Invalid species count: 0`

This is the strongest direct parity check available for the hardcoded moveset table because it uses Showdown’s own validator logic for the exact target format.

## 6. Full Reproduction Steps

1. Build/activate local env
- `source /home/spark-advantage/pufferlib/.pufferlib/bin/activate`

2. Build C extension
- `python setup.py build_poke_battle --inplace --force`

3. Run parity + legality tests
- `python -m pytest -q tests/test_poke_battle_rules_parity.py tests/test_poke_battle_team_builder.py tests/test_poke_battle_moveset_legality.py`

4. (Optional) Re-run external Showdown legality audit
- Clone Showdown at commit `95aad7df02abd58dd737e0acdac22e5d049d360e`
- Run `npm ci` in Showdown repo
- Use `dist/sim/team-validator.js` against exported species sets from `poke_battle.h`

## 7. Residual Scope Notes

This report covers legality parity and clauses requested here:
- format/ruleset clauses
- species legality for teams
- hardcoded moveset legality for all modeled non-Uber Gen1 species
- endless stall detection behavior

It does not claim complete 1:1 cartridge bug emulation or full Showdown engine parity for every move interaction beyond the legalities explicitly audited here.
