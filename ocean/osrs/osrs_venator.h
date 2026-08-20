#ifndef OSRS_VENATOR_H
#define OSRS_VENATOR_H

#include <stdint.h>
#include <stdlib.h>

#ifndef OSRS_COMBAT_H
#error "Include osrs_combat.h instead of osrs_venator.h"
#endif

#define OSRS_VENATOR_BOUNCE_RADIUS 2
#define OSRS_VENATOR_MAX_CHAIN_HITS 3

typedef enum {
    OSRS_VENATOR_SIZE_ODD,
    OSRS_VENATOR_SIZE_2,
    OSRS_VENATOR_SIZE_4,
} OsrsVenatorSizeClass;

typedef enum {
    OSRS_VENATOR_MONSTER_DEAD = 0,
    OSRS_VENATOR_MONSTER_ALIVE = 1,
} OsrsVenatorMonsterLife;

typedef enum {
    OSRS_VENATOR_HIT_PRIMARY = 0,
    OSRS_VENATOR_HIT_BOUNCE = 1,
} OsrsVenatorHitKind;

typedef enum {
    OSRS_VENATOR_CHAIN_LENGTH_ONE = 1,
    OSRS_VENATOR_CHAIN_LENGTH_TWO = 2,
    OSRS_VENATOR_CHAIN_LENGTH_THREE = 3,
} OsrsVenatorChainLength;

typedef enum {
    OSRS_VENATOR_CANDIDATE_MISSING = 0,
    OSRS_VENATOR_CANDIDATE_FOUND = 1,
} OsrsVenatorCandidateSearchKind;

typedef enum {
    OSRS_VENATOR_PRIMARY_CANDIDATE_EXCLUDED = 0,
    OSRS_VENATOR_PRIMARY_CANDIDATE_INCLUDED = 1,
} OsrsVenatorPrimaryCandidateMode;

typedef enum {
    OSRS_VENATOR_DAMAGE_NOT_ROLLED = 0,
    OSRS_VENATOR_DAMAGE_ROLLED = 1,
} OsrsVenatorDamageRollState;

typedef enum {
    OSRS_VENATOR_ACCURACY_MISS = 0,
    OSRS_VENATOR_ACCURACY_HIT = 1,
} OsrsVenatorAccuracyOutcome;

typedef struct {
    int x;
    int y;
} OsrsVenatorTile;

typedef struct {
    int sw_x;
    int sw_y;
    int size;
} OsrsVenatorFootprint;

typedef struct {
    int slot;
    OsrsVenatorFootprint footprint;
    OsrsVenatorMonsterLife life;
} OsrsVenatorMonster;

typedef struct {
    int count;
    OsrsVenatorTile tiles[4];
} OsrsVenatorTileSet;

typedef struct {
    int count;
    OsrsVenatorTile tiles[2];
} OsrsVenatorRequiredTiles;

typedef struct {
    int slot;
    OsrsVenatorFootprint footprint;
    OsrsVenatorHitKind kind;
} OsrsVenatorChainHit;

typedef struct {
    OsrsVenatorChainLength length;
    OsrsVenatorChainHit hits[OSRS_VENATOR_MAX_CHAIN_HITS];
} OsrsVenatorChain;

typedef struct {
    OsrsVenatorCandidateSearchKind kind;
    OsrsVenatorMonster monster;
    int distance;
} OsrsVenatorCandidateSearch;

typedef struct {
    OsrsVenatorDamageRollState roll_state;
    OsrsVenatorAccuracyOutcome accuracy;
    int damage;
    int max_hit;
} OsrsVenatorResolvedDamageHit;

typedef struct {
    OsrsVenatorChainLength length;
    int total_damage;
    OsrsVenatorResolvedDamageHit hits[OSRS_VENATOR_MAX_CHAIN_HITS];
} OsrsVenatorDamageResult;

static inline OsrsVenatorTile osrs_venator_tile(int x, int y) {
    return (OsrsVenatorTile){ .x = x, .y = y };
}

static inline OsrsVenatorFootprint osrs_venator_footprint(
    int sw_x,
    int sw_y,
    int size
) {
    return (OsrsVenatorFootprint){ .sw_x = sw_x, .sw_y = sw_y, .size = size };
}

static inline OsrsVenatorMonster osrs_venator_monster(
    int slot,
    int sw_x,
    int sw_y,
    int size,
    OsrsVenatorMonsterLife life
) {
    return (OsrsVenatorMonster){
        .slot = slot,
        .footprint = osrs_venator_footprint(sw_x, sw_y, size),
        .life = life,
    };
}

static inline OsrsVenatorSizeClass osrs_venator_size_class(int size) {
    switch (size) {
        case 1:
        case 3:
        case 5:
            return OSRS_VENATOR_SIZE_ODD;
        case 2:
            return OSRS_VENATOR_SIZE_2;
        case 4:
            return OSRS_VENATOR_SIZE_4;
    }
    abort();
}

static inline OsrsVenatorTile osrs_venator_sw_tile(OsrsVenatorFootprint f) {
    return osrs_venator_tile(f.sw_x, f.sw_y);
}

static inline OsrsVenatorTile osrs_venator_centre_odd_tile(OsrsVenatorFootprint f) {
    if (osrs_venator_size_class(f.size) != OSRS_VENATOR_SIZE_ODD) abort();
    int offset = (f.size - 1) / 2;
    return osrs_venator_tile(f.sw_x + offset, f.sw_y + offset);
}

static inline OsrsVenatorTile osrs_venator_centre_reference_tile(
    OsrsVenatorFootprint f
) {
    OsrsVenatorSizeClass size_class = osrs_venator_size_class(f.size);
    if (size_class == OSRS_VENATOR_SIZE_ODD) {
        return osrs_venator_centre_odd_tile(f);
    }
    int offset = f.size / 2;
    return osrs_venator_tile(f.sw_x + offset, f.sw_y + offset);
}

static inline OsrsVenatorTile osrs_venator_centre_sw_reference_tile(
    OsrsVenatorFootprint f
) {
    switch (f.size) {
        case 1:
        case 2:
            return osrs_venator_sw_tile(f);
        case 3:
            return osrs_venator_centre_odd_tile(f);
        case 4:
        case 5:
            return osrs_venator_tile(f.sw_x + 1, f.sw_y + 1);
    }
    abort();
}

static inline OsrsVenatorTile osrs_venator_selection_anchor_tile(
    OsrsVenatorFootprint f
) {
    if (osrs_venator_size_class(f.size) == OSRS_VENATOR_SIZE_ODD) {
        return osrs_venator_centre_odd_tile(f);
    }
    return osrs_venator_sw_tile(f);
}

static inline void osrs_venator_required_tiles_push_unique(
    OsrsVenatorRequiredTiles* required,
    OsrsVenatorTile tile
) {
    for (int i = 0; i < required->count; i++) {
        if (required->tiles[i].x == tile.x && required->tiles[i].y == tile.y) {
            return;
        }
    }
    if (required->count >= 2) abort();
    required->tiles[required->count++] = tile;
}

static inline OsrsVenatorTileSet osrs_venator_sender_origin_tiles(
    OsrsVenatorFootprint sender
) {
    OsrsVenatorTileSet origins = {0};
    switch (osrs_venator_size_class(sender.size)) {
        case OSRS_VENATOR_SIZE_ODD:
            origins.tiles[origins.count++] = osrs_venator_centre_odd_tile(sender);
            return origins;
        case OSRS_VENATOR_SIZE_2:
            for (int dx = 0; dx < 2; dx++) {
                for (int dy = 0; dy < 2; dy++) {
                    origins.tiles[origins.count++] =
                        osrs_venator_tile(sender.sw_x + dx, sender.sw_y + dy);
                }
            }
            return origins;
        case OSRS_VENATOR_SIZE_4:
            for (int dx = 1; dx <= 2; dx++) {
                for (int dy = 1; dy <= 2; dy++) {
                    origins.tiles[origins.count++] =
                        osrs_venator_tile(sender.sw_x + dx, sender.sw_y + dy);
                }
            }
            return origins;
    }
    abort();
}

static inline OsrsVenatorRequiredTiles osrs_venator_accept_required_tiles(
    OsrsVenatorFootprint sender,
    OsrsVenatorFootprint target
) {
    OsrsVenatorRequiredTiles required = {0};
    OsrsVenatorSizeClass sender_class = osrs_venator_size_class(sender.size);

    switch (target.size) {
        case 1:
        case 2:
            osrs_venator_required_tiles_push_unique(
                &required, osrs_venator_sw_tile(target));
            return required;
        case 3:
            if (sender_class == OSRS_VENATOR_SIZE_ODD) {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_reference_tile(target));
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_sw_tile(target));
            } else if (sender_class == OSRS_VENATOR_SIZE_2) {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_reference_tile(target));
            } else {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_sw_tile(target));
            }
            return required;
        case 4:
        case 5:
            if (sender_class == OSRS_VENATOR_SIZE_ODD) {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_reference_tile(target));
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_sw_tile(target));
            } else if (sender_class == OSRS_VENATOR_SIZE_2) {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_reference_tile(target));
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_sw_reference_tile(target));
            } else {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_sw_tile(target));
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_sw_reference_tile(target));
            }
            return required;
    }
    abort();
}

static inline OsrsVenatorRequiredTiles osrs_venator_send_required_tiles(
    OsrsVenatorFootprint sender,
    OsrsVenatorFootprint target
) {
    OsrsVenatorRequiredTiles required = {0};
    switch (osrs_venator_size_class(sender.size)) {
        case OSRS_VENATOR_SIZE_ODD:
            osrs_venator_required_tiles_push_unique(
                &required, osrs_venator_sw_tile(target));
            osrs_venator_required_tiles_push_unique(
                &required, osrs_venator_centre_reference_tile(target));
            return required;
        case OSRS_VENATOR_SIZE_2:
            if (target.size <= 3) {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_reference_tile(target));
            } else {
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_reference_tile(target));
                osrs_venator_required_tiles_push_unique(
                    &required, osrs_venator_centre_sw_reference_tile(target));
            }
            return required;
        case OSRS_VENATOR_SIZE_4:
            osrs_venator_required_tiles_push_unique(
                &required, osrs_venator_sw_tile(target));
            osrs_venator_required_tiles_push_unique(
                &required, osrs_venator_centre_sw_reference_tile(target));
            return required;
    }
    abort();
}

static inline int osrs_venator_origins_find_required_tiles(
    OsrsVenatorTileSet origins,
    OsrsVenatorRequiredTiles required
) {
    for (int origin_idx = 0; origin_idx < origins.count; origin_idx++) {
        int found_all = 1;
        for (int tile_idx = 0; tile_idx < required.count; tile_idx++) {
            int dist = chebyshev_distance(
                origins.tiles[origin_idx].x,
                origins.tiles[origin_idx].y,
                required.tiles[tile_idx].x,
                required.tiles[tile_idx].y);
            if (dist > OSRS_VENATOR_BOUNCE_RADIUS) {
                found_all = 0;
                break;
            }
        }
        if (found_all) return 1;
    }
    return 0;
}

static inline int osrs_venator_accepts_bounce(
    OsrsVenatorFootprint sender,
    OsrsVenatorFootprint target
) {
    return osrs_venator_origins_find_required_tiles(
        osrs_venator_sender_origin_tiles(sender),
        osrs_venator_accept_required_tiles(sender, target));
}

static inline int osrs_venator_sends_bounce(
    OsrsVenatorFootprint sender,
    OsrsVenatorFootprint target
) {
    return osrs_venator_origins_find_required_tiles(
        osrs_venator_sender_origin_tiles(sender),
        osrs_venator_send_required_tiles(sender, target));
}

static inline int osrs_venator_can_bounce(
    OsrsVenatorFootprint sender,
    OsrsVenatorFootprint target
) {
    OsrsVenatorTileSet origins =
        osrs_venator_sender_origin_tiles(sender);
    return osrs_venator_origins_find_required_tiles(
            origins,
            osrs_venator_accept_required_tiles(sender, target)) &&
        osrs_venator_origins_find_required_tiles(
            origins,
            osrs_venator_send_required_tiles(sender, target));
}

static inline int osrs_venator_selection_distance(
    OsrsVenatorFootprint sender,
    OsrsVenatorFootprint candidate
) {
    OsrsVenatorTile sender_anchor = osrs_venator_selection_anchor_tile(sender);
    OsrsVenatorTile candidate_anchor = osrs_venator_selection_anchor_tile(candidate);
    return chebyshev_distance(
        sender_anchor.x,
        sender_anchor.y,
        candidate_anchor.x,
        candidate_anchor.y);
}

static inline int osrs_venator_is_better_candidate(
    int candidate_distance,
    int candidate_slot,
    int best_distance,
    int best_slot
) {
    if (candidate_distance < best_distance) return 1;
    return candidate_distance == best_distance && candidate_slot < best_slot;
}

static inline void osrs_venator_consider_candidate(
    OsrsVenatorCandidateSearch* search,
    OsrsVenatorFootprint range_sender,
    OsrsVenatorFootprint selection_sender,
    OsrsVenatorMonster candidate,
    int forbidden_slot
) {
    if (candidate.life != OSRS_VENATOR_MONSTER_ALIVE) return;
    if (candidate.slot == forbidden_slot) return;

    if (!osrs_venator_can_bounce(range_sender, candidate.footprint)) return;

    int distance = osrs_venator_selection_distance(selection_sender, candidate.footprint);
    if (search->kind == OSRS_VENATOR_CANDIDATE_MISSING ||
            osrs_venator_is_better_candidate(
                distance,
                candidate.slot,
                search->distance,
                search->monster.slot)) {
        search->kind = OSRS_VENATOR_CANDIDATE_FOUND;
        search->monster = candidate;
        search->distance = distance;
    }
}

static inline int osrs_venator_live_non_primary_count(
    const OsrsVenatorMonster* candidates,
    int candidate_count,
    int primary_slot
) {
    if (candidate_count < 0) abort();
    if (candidate_count > 0 && candidates == NULL) abort();

    int count = 0;
    for (int i = 0; i < candidate_count; i++) {
        if (candidates[i].life == OSRS_VENATOR_MONSTER_ALIVE &&
                candidates[i].slot != primary_slot) {
            count++;
        }
    }
    return count;
}

static inline OsrsVenatorCandidateSearch osrs_venator_find_next_candidate(
    OsrsVenatorFootprint range_sender,
    OsrsVenatorFootprint selection_sender,
    OsrsVenatorMonster primary,
    const OsrsVenatorMonster* candidates,
    int candidate_count,
    int forbidden_slot,
    OsrsVenatorPrimaryCandidateMode primary_candidate_mode
) {
    if (candidate_count < 0) abort();
    if (candidate_count > 0 && candidates == NULL) abort();

    OsrsVenatorCandidateSearch search = { OSRS_VENATOR_CANDIDATE_MISSING };
    if (primary_candidate_mode == OSRS_VENATOR_PRIMARY_CANDIDATE_INCLUDED &&
            primary.life == OSRS_VENATOR_MONSTER_ALIVE) {
        osrs_venator_consider_candidate(
            &search, range_sender, selection_sender, primary, forbidden_slot);
    }

    for (int i = 0; i < candidate_count; i++) {
        if (candidates[i].slot == primary.slot) continue;
        osrs_venator_consider_candidate(
            &search, range_sender, selection_sender, candidates[i], forbidden_slot);
    }

    return search;
}

static inline OsrsVenatorChain osrs_venator_resolve_chain(
    OsrsVenatorMonster primary,
    const OsrsVenatorMonster* candidates,
    int candidate_count
) {
    if (candidate_count < 0) abort();
    if (candidate_count > 0 && candidates == NULL) abort();

    OsrsVenatorChain chain = {
        .length = OSRS_VENATOR_CHAIN_LENGTH_ONE,
        .hits = {
            {
                .slot = primary.slot,
                .footprint = primary.footprint,
                .kind = OSRS_VENATOR_HIT_PRIMARY,
            },
        },
    };

    OsrsVenatorCandidateSearch hit2 = osrs_venator_find_next_candidate(
        primary.footprint,
        primary.footprint,
        primary,
        candidates,
        candidate_count,
        primary.slot,
        OSRS_VENATOR_PRIMARY_CANDIDATE_EXCLUDED);
    if (hit2.kind == OSRS_VENATOR_CANDIDATE_MISSING) {
        return chain;
    }

    chain.length = OSRS_VENATOR_CHAIN_LENGTH_TWO;
    chain.hits[1] = (OsrsVenatorChainHit){
        .slot = hit2.monster.slot,
        .footprint = hit2.monster.footprint,
        .kind = OSRS_VENATOR_HIT_BOUNCE,
    };

    if (osrs_venator_live_non_primary_count(
            candidates, candidate_count, primary.slot) < 2) {
        return chain;
    }

    OsrsVenatorCandidateSearch hit3 = osrs_venator_find_next_candidate(
        primary.footprint,
        hit2.monster.footprint,
        primary,
        candidates,
        candidate_count,
        hit2.monster.slot,
        OSRS_VENATOR_PRIMARY_CANDIDATE_INCLUDED);
    if (hit3.kind == OSRS_VENATOR_CANDIDATE_MISSING) {
        return chain;
    }

    chain.length = OSRS_VENATOR_CHAIN_LENGTH_THREE;
    chain.hits[2] = (OsrsVenatorChainHit){
        .slot = hit3.monster.slot,
        .footprint = hit3.monster.footprint,
        .kind = OSRS_VENATOR_HIT_BOUNCE,
    };
    return chain;
}

static inline int osrs_venator_bounce_max_hit(int original_max_hit) {
    if (original_max_hit < 0) abort();
    return original_max_hit * 2 / 3;
}

static inline void osrs_venator_validate_chain_length(OsrsVenatorChainLength length) {
    switch (length) {
        case OSRS_VENATOR_CHAIN_LENGTH_ONE:
        case OSRS_VENATOR_CHAIN_LENGTH_TWO:
        case OSRS_VENATOR_CHAIN_LENGTH_THREE:
            return;
    }
    abort();
}

static inline OsrsVenatorDamageResult osrs_venator_roll_chain_damage(
    const OsrsVenatorChain* chain,
    const int target_defence_rolls[OSRS_VENATOR_MAX_CHAIN_HITS],
    int attack_roll,
    int original_max_hit,
    uint32_t* rng_state
) {
    if (chain == NULL || target_defence_rolls == NULL || rng_state == NULL) abort();
    if (attack_roll < 0) abort();
    if (original_max_hit < 0) abort();
    osrs_venator_validate_chain_length(chain->length);

    OsrsVenatorDamageResult result = { .length = chain->length };
    int bounce_max_hit = osrs_venator_bounce_max_hit(original_max_hit);
    for (int i = 0; i < (int)chain->length; i++) {
        int max_hit = i == 0 ? original_max_hit : bounce_max_hit;
        int hit = encounter_roll_hit_chance(
            rng_state, attack_roll, target_defence_rolls[i]);
        int damage = hit ? encounter_rand_int(rng_state, max_hit + 1) : 0;
        result.hits[i] = (OsrsVenatorResolvedDamageHit){
            .roll_state = OSRS_VENATOR_DAMAGE_ROLLED,
            .accuracy = hit
                ? OSRS_VENATOR_ACCURACY_HIT
                : OSRS_VENATOR_ACCURACY_MISS,
            .damage = damage,
            .max_hit = max_hit,
        };
        result.total_damage += damage;
    }
    return result;
}

#endif
