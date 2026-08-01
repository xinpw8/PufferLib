// NetHack env, obs layout mirrored by ocean/nethack/nethack.cu
// One env per agent, each owning an nle_ctx_t and a private vardir on tmpfs.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <signal.h>
#include "fs.h"
#include "pufferenv.h"

// nletypes.h, not nle.h: nle.h's `settings` macro would rewrite env->settings
#include "nletypes.h"

#ifdef __cplusplus
extern "C" {
#endif
extern nle_ctx_t* nle_start(nle_obs*, FILE*, nle_settings*);
extern nle_ctx_t* nle_step(nle_ctx_t*, nle_obs*);
extern nle_ctx_t* nle_obs_refresh(nle_ctx_t*, nle_obs*);
extern void       nle_end(nle_ctx_t*);
#ifdef __cplusplus
}
#endif

#include "netlib.h"

#define OBS_SIZE NETHACK_OBS_SIZE
#define NUM_ATNS 19
#define ACT_SIZES {NETHACK_NUM_ACTIONS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, \
    NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS}
typedef unsigned char obs_t;

typedef Env Nethack;
struct Env {
    Log log;
    Agent agents[1];
    unsigned char* action_mask;
    int num_agents;
    int pending_reset;   // NLE's coroutine must reset on a stepping thread
    int tag;
    int boundary_reached;

    // engine handle
    nle_ctx_t* ctx;
    nle_obs obs;
    nle_settings settings;
    char vardir[1024];

    // NLE-written buffers
    short          glyphs[NH_GRID];
    long           blstats[NLE_BLSTATS_SIZE];
    unsigned char  chars[NH_GRID];
    unsigned char  message[NLE_MESSAGE_SIZE];
    int            misc[NLE_MISC_SIZE];
    int            internal[NLE_INTERNAL_SIZE];
    short          inv_glyphs[NLE_INVENTORY_SIZE];
    unsigned char  inv_letters[NLE_INVENTORY_SIZE];
    unsigned char  inv_oclasses[NLE_INVENTORY_SIZE];
    signed char    inv_state[NLE_INVENTORY_SIZE * NLE_INV_STATE_FIELDS];

    Stats stats;

    // reward-delta trackers
    int prev_action;
    int enh_ready;
    long prev_score;
    long prev_exp;
    long prev_gold;
    long start_gold;
    long prev_hp;
    long prev_hunger;
    long prev_time;
    int prev_depth;
    long prev_ac;
    int prev_bad_cond;
    // reward coefs
    float gold_coef;
    float exp_coef;
    float descent_coef;
    float floor_coef;
    float xp_coef;
    float scout_coef;
    float hp_coef;
    float hunger_coef;
    float illegal_penalty;
    float death_penalty;
    float ac_coef;
    float heal_coef;
    float status_coef;

    unsigned int rng;    // required by vecenv.h
    unsigned long seed;  // advanced each reset
};

#include "macros.h"   // keystroke utils

// init

// demo-only obs planes; NULL in training (fills skipped)
static unsigned char* nethack_color_sink;
static unsigned char* nethack_invstr_sink;

static void nethack_bind_obs(Nethack* env) {
    nle_obs* o = &env->obs;
    memset(o, 0, sizeof(*o));
    o->colors   = nethack_color_sink;
    o->inv_strs = nethack_invstr_sink;
    o->glyphs   = env->glyphs;
    o->blstats  = env->blstats;
    o->chars    = env->chars;
    o->message  = env->message;
    o->misc     = env->misc;
    o->internal = env->internal;
    o->inv_glyphs   = env->inv_glyphs;
    o->inv_letters  = env->inv_letters;
    o->inv_oclasses = env->inv_oclasses;
    o->inv_state    = env->inv_state;
    // partial fills
    o->partial = 1;
}

static void nethack_init_settings(Nethack* env) {
    memset(&env->settings, 0, sizeof(env->settings));
    const char* source = getenv("NETHACKDIR");
    if (source == NULL) source = "./vendor/fast-nle/build/dat";

    if (nethack_make_vardir(source, env->vardir, sizeof(env->vardir)) != 0) {
        fprintf(stderr, "nethack: failed to create vardir from source=%s\n", source);
        strncpy(env->settings.hackdir, source, sizeof(env->settings.hackdir) - 1);
    } else {
        strncpy(env->settings.hackdir, env->vardir, sizeof(env->settings.hackdir) - 1);
    }
    env->settings.spawn_monsters = 1;
    env->settings.underfoot_glyphs = 1;   // underfoot shows objects
    snprintf(env->settings.options, sizeof(env->settings.options), "@%s",
             nethack_rc_path(NETHACK_DEFAULT_OPTIONS));
    env->settings.fix_moon_phase = true;  // moon phase from seed
}

void init(Nethack* env) {
    env->seed = 0xCAFEBEEFUL + (unsigned long)env->rng;   // rng = env index
    // opt into consumed-head gating (env_head_consume_map below); =0 still disables
    setenv("PUFFER_HEAD_GATING", "1", 0);
    // nle_start deferred to first puf_reset
    nethack_init_settings(env);
}

// masking

static int nethack_slot_usable(const Nethack* env, const Verb* verb, int i) {
    if (!(verb->item_classes & (1u << env->inv_oclasses[i]))) return 0;
    int worn = env->inv_state[i * NLE_INV_STATE_FIELDS + 5] & 1;
    if (verb->wornreq == WORN_ONLY) return worn;
    if (verb->wornreq == UNWORN_ONLY) return !worn;
    return 1;
}

// visible target (monster/detected/warning glyph) on the ray within 8 tiles
static int nethack_ray_target(Nethack* env, int dx, int dy) {
    long hx = env->blstats[NLE_BL_X], hy = env->blstats[NLE_BL_Y];
    for (int k = 1; k <= 8; k++) {
        long x = hx + dx * k, y = hy + dy * k;
        if (x < 0 || x >= NH_COLS || y < 0 || y >= NH_ROWS) return 0;
        int gl = env->glyphs[y * NH_COLS + x];
        if ((gl >= 0 && gl < NETHACK_NUMMONS)
            || (gl >= 762 && gl < 1144)
            || (gl >= 5589 && gl < 5595)) return 1;
    }
    return 0;
}

static void nethack_compute_mask(Nethack* env) {
    unsigned char* mask = env->action_mask;
    memset(mask, 1, NETHACK_NUM_ACTIONS);
    if (env->blstats[NLE_BL_HUNGER] == 0) mask[NETHACK_ACT_EAT] = 0;   // choke gate

    // underfoot
    long hero_x = env->blstats[NLE_BL_X], hero_y = env->blstats[NLE_BL_Y];
    int underfoot = (hero_x >= 0 && hero_x < NH_COLS && hero_y >= 0 && hero_y < NH_ROWS)
           ? env->glyphs[hero_y * NH_COLS + hero_x] : -1;
    int on_object = (underfoot >= NETHACK_GLYPH_OBJ_LO && underfoot < NETHACK_GLYPH_OBJ_HI);
    int on_corpse = (underfoot >= NETHACK_GLYPH_BODY_OFF
                     && underfoot < NETHACK_GLYPH_BODY_OFF + NETHACK_NUMMONS);

    if (underfoot != NETHACK_GLYPH_DNSTAIR && underfoot != NETHACK_GLYPH_DNLADDER)
        mask[NETHACK_ACT_DOWN] = 0;
    if (underfoot != NETHACK_GLYPH_UPSTAIR && underfoot != NETHACK_GLYPH_UPLADDER)
        mask[NETHACK_ACT_UP] = 0;
    if (env->blstats[NLE_BL_DEPTH] <= 1) mask[NETHACK_ACT_UP] = 0;   // declined exit
    if (!on_object && !on_corpse) mask[NETHACK_ACT_PICKUP] = 0;

    // item slot heads
    for (int a = 0; a < NETHACK_NUM_ACTIONS; a++) {
        const Verb* verb = &NETHACK_VERBS[a];
        if (verb->head < 0) continue;   // direct verb, no item argument
        unsigned char* slots = mask + NETHACK_NUM_ACTIONS + verb->head * NETHACK_INV_SLOTS;
        memset(slots, 0, NETHACK_INV_SLOTS);
        int has_usable = 0;
        for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i]; i++) {
            if (env->inv_oclasses[i] >= NETHACK_NUM_OCLASSES) break;   // padded tail
            if (nethack_slot_usable(env, verb, i)) { slots[i] = 1; has_usable = 1; }
        }
        if (has_usable) continue;
        // no usable item
        slots[0] = 1;
        int floor_food = (a == NETHACK_ACT_EAT) && (on_object || on_corpse);
        if (!floor_food) mask[a] = 0;
    }

    // per-verb dir rows: wall legality; THROW/ZAP use target rays (all-1 fallback)
    unsigned char* dirs = mask + NETHACK_NUM_ACTIONS + 12 * NETHACK_INV_SLOTS;
    memset(dirs, 1, NETHACK_NUM_DIRS);
    int legal_dirs = 0;
    for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
        long col = hero_x + NETHACK_DIR_DX[d], row = hero_y + NETHACK_DIR_DY[d];
        if (row < 0 || row >= NH_ROWS || col < 0 || col >= NH_COLS) { dirs[d] = 0; continue; }
        int g = env->glyphs[row * NH_COLS + col];
        if (g >= NETHACK_WALL_GLYPH_LO && g <= NETHACK_WALL_GLYPH_HI) dirs[d] = 0;
        else legal_dirs++;
    }
    if (!legal_dirs) memset(dirs, 1, NETHACK_NUM_DIRS);
    for (int h = 1; h < NETHACK_DIR_HEADS; h++)
        memcpy(dirs + h * NETHACK_NUM_DIRS, dirs, NETHACK_NUM_DIRS);
    static const int ray_verbs[2] = {NETHACK_ACT_THROW, NETHACK_ACT_ZAP};
    for (int v = 0; v < 2; v++) {
        unsigned char* vdirs = dirs + nethack_dir_head(ray_verbs[v]) * NETHACK_NUM_DIRS;
        int any = 0;
        for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
            vdirs[d] = nethack_ray_target(env, NETHACK_DIR_DX[d], NETHACK_DIR_DY[d]);
            any |= vdirs[d];
        }
        if (!any) memset(vdirs, 1, NETHACK_NUM_DIRS);
    }
}

// observations

static void nethack_pack_obs(Nethack* env) {
    memcpy(((obs_t*)env->agents[0].observations) + NETHACK_OFF_GLYPHS, env->glyphs, sizeof(env->glyphs));
    unsigned char* bl = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_BLSTATS;
    for (int i = 0; i < NLE_BLSTATS_SIZE; i++) {
        uint32_t v = (uint32_t)(int32_t)env->blstats[i];
        bl[4*i + 0] = (unsigned char)(v & 0xffu);
        bl[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        bl[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        bl[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    int32_t extra[NETHACK_EXTRA_INTS] = {0};
    // engraving state 0/1/2
    extra[0] = env->internal[6];
    extra[1] = env->prev_action;
    for (int i = 0; i < NLE_INVENTORY_SIZE; i++) {
        int oc = env->inv_oclasses[i];
        if (oc >= NETHACK_NUM_OCLASSES) break;   // padded tail
        extra[2 + oc]++;
    }
    unsigned char* ex = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_EXTRA;
    for (int i = 0; i < NETHACK_EXTRA_INTS; i++) {
        uint32_t v = (uint32_t)extra[i];
        ex[4*i + 0] = (unsigned char)(v & 0xffu);
        ex[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        ex[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        ex[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    // slot glyphs
    unsigned char* iv = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_INV;
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        uint16_t g = env->inv_oclasses[i] < NETHACK_NUM_OCLASSES
                   ? (uint16_t)env->inv_glyphs[i] : (uint16_t)NETHACK_PAD_GLYPH;
        iv[2*i + 0] = (unsigned char)(g & 0xffu);
        iv[2*i + 1] = (unsigned char)((g >> 8) & 0xffu);
    }
    // item state
    memcpy(((obs_t*)env->agents[0].observations) + NETHACK_OFF_INVST, env->inv_state, sizeof(env->inv_state));
    // topline
    unsigned char* mv = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_MSG;
    size_t mlen = strnlen((const char*)env->message, NETHACK_MSG_LEN);
    memcpy(mv, env->message, mlen);
    if (mlen < (size_t)NETHACK_MSG_LEN) memset(mv + mlen, 0, NETHACK_MSG_LEN - mlen);

    if (env->action_mask != NULL) nethack_compute_mask(env);
}

// logging

static void nethack_add_log(Nethack* env, int how) {   // how: nle how_done, -1 = truncated
    for (int v = 0; v < NETHACK_NUM_ACTIONS; v++)
        env->log.verb_uses[v] += (float)env->stats.verb_uses[v];
    env->log.perf += (float)env->prev_score;
    env->log.score += (float)env->prev_score;
    env->log.valid_moves += (float)env->stats.valid_moves;
    env->log.illegal_actions += (float)env->stats.illegal_actions;
    env->log.new_tiles += (float)env->stats.new_tiles;
    env->log.max_depth += (float)env->stats.max_depth;
    env->log.floors += (float)env->stats.floors;
    env->log.enhances += (float)env->stats.enhances;
    env->log.prayers_low_hp += (float)env->stats.prayers_low_hp;
    env->log.prayers_starving += (float)env->stats.prayers_starving;
    env->log.floor_eats += (float)env->stats.floor_eats;
    env->log.damage_taken += (float)env->stats.damage;
    env->log.ac += env->stats.length > 0
        ? (float)env->stats.ac_sum / (float)env->stats.length : 0.0f;
    env->log.min_ac += (float)env->stats.min_ac;
    env->log.armor_swaps += (float)env->stats.armor_swaps;
    env->log.heal_hp += (float)env->stats.heal_hp;
    env->log.cures += (float)env->stats.cures;
    env->log.burdened_frac += env->stats.length > 0
        ? (float)env->stats.burdened_steps / (float)env->stats.length : 0.0f;
    env->log.game_time += (float)env->prev_time;
    env->log.max_xp_level += (float)env->stats.max_xp;
    env->log.episode_return += env->stats.ret;
    env->log.episode_length += env->stats.length;
    if (how == -1) env->log.truncated += 1.0f;
    else if (how == 0) env->log.death_combat += 1.0f;
    else if (how == 3) env->log.death_starved += 1.0f;
    else if (how == NLE_HOW_WRATH) env->log.death_smited += 1.0f;
    else env->log.death_other += 1.0f;
    if (how == 0) {
        env->log.death_mon_level += (float)env->internal[NETHACK_INTERNAL_KILLER_MLEV];
        env->log.death_adj_monsters += (float)env->stats.last_adj;
        env->log.death_maxhp += (float)env->stats.last_maxhp;
    }
    env->log.reach_mines += (env->stats.areas & NETHACK_AREA_MINES) ? 1.0f : 0.0f;
    env->log.reach_minetown += (env->stats.areas & NETHACK_AREA_MINETOWN) ? 1.0f : 0.0f;
    env->log.reach_deep_mines += (env->stats.areas & NETHACK_AREA_DEEP_MINES) ? 1.0f : 0.0f;
    env->log.reach_main_d5 += (env->stats.areas & NETHACK_AREA_MAIN_D5) ? 1.0f : 0.0f;
    env->log.reach_sokoban += (env->stats.areas & NETHACK_AREA_SOKOBAN) ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

// reset

static void nethack_do_reset(Nethack* env) {
    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }

    nethack_bind_obs(env);
    env->obs.how_done = -2;   // only really_done() sets it

    // seed advance
    env->seed = env->seed * 6364136223846793005UL + 1442695040888963407UL;
    env->settings.initial_seeds.seeds[0] = env->seed;
    env->settings.initial_seeds.seeds[1] = env->seed ^ 0x9E3779B97F4A7C15UL;
    env->settings.initial_seeds.use_init_seeds = true;
    env->settings.time_seed = env->seed;
    env->settings.time_seed_is_set = true;
    env->ctx = nle_start(&env->obs, NULL, &env->settings);

    nethack_drain_prompts(env);
    nle_obs_refresh(env->ctx, &env->obs);   // full fill: prev_* seeds read blstats

    env->prev_score = 0;
    env->prev_exp = env->blstats[NLE_BL_EXP];
    env->start_gold = env->blstats[NLE_BL_GOLD];
    env->prev_gold = 0;   // clamped net gold
    env->prev_hp = env->blstats[NLE_BL_HP];
    env->prev_hunger = env->blstats[NLE_BL_HUNGER];
    if (env->prev_hunger < 1) env->prev_hunger = 1;
    else if (env->prev_hunger > 6) env->prev_hunger = 6;
    env->prev_depth = (int)env->blstats[NLE_BL_DEPTH];
    env->prev_ac = env->blstats[NLE_BL_AC];
    env->prev_bad_cond = __builtin_popcount((unsigned)env->blstats[NLE_BL_CONDITION] & NETHACK_COND_BAD);
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_action = -1;
    env->enh_ready = 0;
    memset(&env->stats, 0, sizeof(env->stats));
    env->stats.max_depth = env->prev_depth;
    env->stats.max_xp = (int)env->blstats[NLE_BL_XP];
    env->stats.min_ac = (int)env->blstats[NLE_BL_AC];
    nethack_pack_obs(env);
}

void puf_reset(Nethack* env) {
    env->pending_reset = 1;
}

// reward

static void nethack_update_stats(Nethack* env) {
    int depth = (int)env->blstats[NLE_BL_DEPTH];
    long dnum = env->blstats[NLE_BL_DNUM];
    if (dnum == 2) {
        env->stats.areas |= NETHACK_AREA_MINES;
        long mlvl = env->blstats[NLE_BL_DLEVEL];
        if (mlvl >= 3) env->stats.areas |= NETHACK_AREA_MINETOWN;
        if (mlvl >= 5) env->stats.areas |= NETHACK_AREA_DEEP_MINES;
    }
    else if (dnum == 0 && depth >= 5) env->stats.areas |= NETHACK_AREA_MAIN_D5;
    else if (dnum == 4) env->stats.areas |= NETHACK_AREA_SOKOBAN;

    long hp = env->blstats[NLE_BL_HP];
    if (hp < env->prev_hp) env->stats.damage += env->prev_hp - hp;
    if (env->blstats[NLE_BL_CAP] > 0) env->stats.burdened_steps++;

    // death anatomy, read back at death
    env->stats.last_maxhp = env->blstats[NLE_BL_HPMAX];
    long hx = env->blstats[NLE_BL_X], hy = env->blstats[NLE_BL_Y];
    int adj = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            long r = hy + dy, c = hx + dx;
            if (r < 0 || r >= NH_ROWS || c < 0 || c >= NH_COLS) continue;
            int g = env->glyphs[r * NH_COLS + c];
            if (g >= 0 && g < NETHACK_NUMMONS) adj++;
        }
    env->stats.last_adj = adj;

    env->prev_score = env->blstats[NLE_BL_SCORE];
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_depth = depth;
}

static int nethack_first_visit(Nethack* env, int depth, long px, long py) {
    if (px < 0 || px >= NH_COLS || py < 0 || py >= NH_ROWS) return 0;
    int d = depth < 1 ? 0 : (depth > NETHACK_MAX_DEPTH ? NETHACK_MAX_DEPTH - 1 : depth - 1);
    int bit = (int)py * NH_COLS + (int)px;
    unsigned char mask = (unsigned char)(1 << (bit & 7));
    if (env->stats.visited[d][bit >> 3] & mask) return 0;
    env->stats.visited[d][bit >> 3] |= mask;
    return 1;
}

static float nethack_reward(Nethack* env, int illegal) {
    // death payout
    if (env->obs.done)
        return env->death_penalty - env->hp_coef * (float)env->prev_hp;
    nethack_update_stats(env);

    int depth = (int)env->blstats[NLE_BL_DEPTH];

    // exp, gains only
    long exp = env->blstats[NLE_BL_EXP];
    float r = exp > env->prev_exp ? env->exp_coef * (float)(exp - env->prev_exp) : 0.0f;
    env->prev_exp = exp;

    // gold, net of start
    long g = env->blstats[NLE_BL_GOLD] - env->start_gold;
    if (g < 0) g = 0;
    r += env->gold_coef * (float)(g - env->prev_gold);
    env->prev_gold = g;

    // descent_coef pays per max-depth delta, floor_coef per new (dnum, dlevel) floor
    { long dn = env->blstats[NLE_BL_DNUM], dl = env->blstats[NLE_BL_DLEVEL];
      if (dn >= 0 && dn < 16 && dl >= 1 && dl <= 64) {
          unsigned long long fb = 1ULL << (dl - 1);
          if (!(env->stats.floors_bits[dn] & fb)) {
              env->stats.floors_bits[dn] |= fb;
              env->stats.floors++;
              r += env->floor_coef;
          }
      } }
    if (depth > env->stats.max_depth) {
        r += env->descent_coef * (float)(depth - env->stats.max_depth);
        env->stats.max_depth = depth;
    }

    // hp potential
    long hp = env->blstats[NLE_BL_HP];
    long hp_delta = hp - env->prev_hp;
    r += env->hp_coef * (float)hp_delta;
    // gain-only heal credit
    if (hp_delta > 0 && (env->prev_action == NETHACK_ACT_QUAFF
                      || env->prev_action == NETHACK_ACT_PRAY)) {
        r += env->heal_coef * (float)hp_delta;
        env->stats.heal_hp += hp_delta;
    }
    env->prev_hp = hp;

    // ac potential
    long ac = env->blstats[NLE_BL_AC];
    r += env->ac_coef * (float)(env->prev_ac - ac);
    env->prev_ac = ac;
    env->stats.ac_sum += ac;
    if ((int)ac < env->stats.min_ac) env->stats.min_ac = (int)ac;

    // status potential
    int bad_cond = __builtin_popcount((unsigned)env->blstats[NLE_BL_CONDITION] & NETHACK_COND_BAD);
    r += env->status_coef * (float)(env->prev_bad_cond - bad_cond);
    if (bad_cond < env->prev_bad_cond) env->stats.cures += env->prev_bad_cond - bad_cond;
    env->prev_bad_cond = bad_cond;

    // hunger potential
    long hunger = env->blstats[NLE_BL_HUNGER];
    if (hunger < 1) hunger = 1;
    else if (hunger > 6) hunger = 6;
    r += env->hunger_coef * (float)(env->prev_hunger - hunger);
    env->prev_hunger = hunger;

    // xp level, max only
    int xp = (int)env->blstats[NLE_BL_XP];
    if (xp > env->stats.max_xp) {
        r += env->xp_coef * (float)(xp - env->stats.max_xp);
        env->stats.max_xp = xp;
    }

    // scout
    if (nethack_first_visit(env, depth, env->blstats[NLE_BL_X], env->blstats[NLE_BL_Y])) {
        r += env->scout_coef;
        env->stats.new_tiles++;
    }

    if (illegal) r += env->illegal_penalty;
    return r;
}

// stepping

static void nethack_execute(Nethack* env, int verb, int slot, int dirkey, int* bad_pick) {
    Stats* st = &env->stats;
    switch (verb) {
    case NETHACK_ACT_MOVE:
        nethack_send_key(env, dirkey);
        break;
    case NETHACK_ACT_RUN:
        nethack_send_key(env, dirkey - 32);   // uppercase = run
        break;
    case NETHACK_ACT_DOWN:
        nethack_send_key(env, '>');
        break;
    case NETHACK_ACT_UP:
        nethack_send_key(env, '<');
        break;
    case NETHACK_ACT_KICK:
        nethack_send_key(env, 4);   // ^D
        nethack_answer_direction(env, dirkey);
        break;
    case NETHACK_ACT_SEARCH:
        st->verb_uses[verb]++;
        nethack_send_key(env, 's');
        break;
    case NETHACK_ACT_ELBERETH:
        st->verb_uses[verb]++;
        nethack_do_elbereth(env);
        break;
    case NETHACK_ACT_SEARCH20:
        st->verb_uses[verb]++;
        nethack_send_key(env, '2');
        nethack_send_key(env, '0');
        nethack_send_key(env, 's');
        break;
    case NETHACK_ACT_PICKUP:
        st->verb_uses[verb]++;
        nethack_send_key(env, ',');
        nethack_answer_menu(env);
        break;
    case NETHACK_ACT_PRAY:
        if (4 * env->blstats[NLE_BL_HP] <= env->blstats[NLE_BL_HPMAX]) st->prayers_low_hp++;
        if (env->blstats[NLE_BL_HUNGER] >= NETHACK_HUNGER_WEAK) st->prayers_starving++;
        st->verb_uses[verb]++;
        nethack_send_key(env, 0x80 | 'p');
        break;
    case NETHACK_ACT_WEAR:
        nethack_wear_takeoff_conflict(env, slot);
        nethack_item_use(env, 'W', "want to wear", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_EAT:
        nethack_item_use(env, 'e', "want to eat", "eat it", slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_QUAFF:
        nethack_item_use(env, 'q', "want to drink", "rink from the", slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_THROW:
        if (nethack_item_use(env, 't', "want to throw", NULL, slot, &st->verb_uses[verb], bad_pick))
            nethack_answer_direction(env, dirkey);
        break;
    case NETHACK_ACT_ZAP:
        if (nethack_item_use(env, 'z', "want to zap", NULL, slot, &st->verb_uses[verb], bad_pick))
            nethack_answer_direction(env, dirkey);
        break;
    case NETHACK_ACT_TAKEOFF:
        nethack_item_use(env, 'T', "take off", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_PUTON:
        nethack_item_use(env, 'P', "put on", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_REMOVE:
        nethack_item_use(env, 'R', "remove", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_WIELD:
        nethack_verb_wield(env, slot, bad_pick);
        break;
    case NETHACK_ACT_APPLY:
        if (nethack_item_use(env, 'a', "apply", NULL, slot, &st->verb_uses[verb], bad_pick)) {
            // diggers dig down
            int otyp = env->inv_glyphs[slot] - NH_GLYPH_OBJ_OFF;
            nethack_answer_direction(env,
                (otyp == 234 /* PICK_AXE */ || otyp == 50 /* MATTOCK */) ? '>' : dirkey);
        }
        break;
    case NETHACK_ACT_READ:
        if (nethack_item_use(env, 'r', "read", NULL, slot, &st->verb_uses[verb], bad_pick))
            nethack_answer_menu(env);
        break;
    case NETHACK_ACT_DROP:
        nethack_item_use(env, 'd', "drop", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    }
}

// compute_mask writes through this flat alias of agents[0].action_mask
static void nethack_sync_buffers(Nethack* env) {
    env->action_mask = env->agents[0].action_mask;
}

void puf_step(Nethack* env) {
    nethack_sync_buffers(env);   // agent pointers are re-dealt between epochs
    if (env->pending_reset) {
        env->pending_reset = 0;
        nethack_do_reset(env);
    }

    int verb = (int)env->agents[0].actions[0];
    int head = NETHACK_VERBS[verb].head;
    int slot = (head >= 0) ? (int)env->agents[0].actions[1 + head] : 0;
    int dh = nethack_dir_head(verb);
    int dirkey = NETHACK_DIR_KEYS[dh >= 0 ? (int)env->agents[0].actions[13 + dh] : 0];

    long time_before = env->blstats[NLE_BL_TIME];
    int bad_pick = 0;
    nethack_execute(env, verb, slot, dirkey, &bad_pick);

    env->prev_action = verb;
    int illegal = nethack_handle_prompts(env);
    if (!env->obs.done) nle_obs_refresh(env->ctx, &env->obs);
    nethack_auto_enhance(env);

    if (bad_pick) { illegal = 1; env->stats.illegal_actions++; }
    if (env->blstats[NLE_BL_TIME] > time_before) env->stats.valid_moves++;
    env->stats.length++;

    float reward = nethack_reward(env, illegal);
    env->agents[0].rewards[0] = reward;
    env->stats.ret += reward;

    int done = env->obs.done || env->stats.length >= NETHACK_MAX_EPISODE_STEPS;
    env->agents[0].terminals[0] = done ? 1.0f : 0.0f;   // truncation reported as terminal too
    if (done) {
        nethack_add_log(env, env->obs.done ? env->obs.how_done : -1);
        // eager same-thread reset: the terminal step returns the fresh obs
        nethack_do_reset(env);
    } else {
        nethack_pack_obs(env);
    }
}

void puf_close(Nethack* env) {
    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }
    nethack_rm_vardir(env->vardir);
    env->vardir[0] = '\0';
}

void puf_render(Nethack* env) {
    printf("\x1b[H\x1b[2J");
    for (int r = 0; r < NH_ROWS; r++) {
        for (int c = 0; c < NH_COLS; c++) {
            unsigned char ch = env->chars[r * NH_COLS + c];
            putchar(ch ? ch : ' ');
        }
        putchar('\n');
    }
    printf("HP %ld/%ld  AC %ld  Dlvl %ld  Score %ld  T %ld\n",
           env->blstats[NLE_BL_HP], env->blstats[NLE_BL_HPMAX],
           env->blstats[NLE_BL_AC], env->blstats[NLE_BL_DEPTH],
           env->blstats[NLE_BL_SCORE], env->blstats[NLE_BL_TIME]);
    printf("Msg: %.*s\n", NLE_MESSAGE_SIZE, env->message);
    fflush(stdout);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    init(env);
    env->gold_coef = dict_get(kwargs, "gold_coef");
    env->exp_coef = dict_get(kwargs, "exp_coef");
    env->descent_coef = dict_get(kwargs, "descent_coef");
    env->floor_coef = dict_get(kwargs, "floor_coef");
    env->scout_coef = dict_get(kwargs, "scout_coef");
    env->xp_coef = dict_get(kwargs, "xp_coef");
    env->hp_coef = dict_get(kwargs, "hp_coef");
    env->hunger_coef = dict_get(kwargs, "hunger_coef");
    env->illegal_penalty = dict_get(kwargs, "illegal_penalty");
    env->death_penalty = dict_get(kwargs, "death_penalty");
    env->ac_coef = dict_get(kwargs, "ac_coef");
    env->heal_coef = dict_get(kwargs, "heal_coef");
    env->status_coef = dict_get(kwargs, "status_coef");
}

// Export order: outcomes first (score/depth/reaches/deaths), then action
// stats, then per-step diagnostics. Dict insertion order is what the
// dashboard/CSV see.
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_depth", log->max_depth);
    dict_set(out, "min_ac", log->min_ac);
    dict_set(out, "reach_mines", log->reach_mines);
    dict_set(out, "reach_minetown", log->reach_minetown);
    dict_set(out, "reach_deep_mines", log->reach_deep_mines);
    dict_set(out, "reach_main_d5", log->reach_main_d5);
    dict_set(out, "reach_sokoban", log->reach_sokoban);
    dict_set(out, "death_combat", log->death_combat);
    dict_set(out, "death_starved", log->death_starved);
    dict_set(out, "death_smited", log->death_smited);
    dict_set(out, "death_other", log->death_other);
    dict_set(out, "death_mon_level", log->death_mon_level);
    dict_set(out, "death_adj_monsters", log->death_adj_monsters);
    dict_set(out, "death_maxhp", log->death_maxhp);
    for (int v = 0; v < NETHACK_NUM_ACTIONS; v++) {
        if (NETHACK_VERB_STAT[v])
            dict_set(out, NETHACK_VERB_STAT[v], log->verb_uses[v]);
    }
    dict_set(out, "valid_moves", log->valid_moves);
    dict_set(out, "illegal_actions", log->illegal_actions);
    dict_set(out, "new_tiles", log->new_tiles);
    dict_set(out, "enhances", log->enhances);
    dict_set(out, "floor_eats", log->floor_eats);
    dict_set(out, "prayers_low_hp", log->prayers_low_hp);
    dict_set(out, "prayers_starving", log->prayers_starving);
    dict_set(out, "burdened_frac", log->burdened_frac);
    dict_set(out, "damage_taken", log->damage_taken);
    dict_set(out, "ac", log->ac);
    dict_set(out, "armor_swaps", log->armor_swaps);
    dict_set(out, "heal_hp", log->heal_hp);
    dict_set(out, "cures", log->cures);
    dict_set(out, "game_time", log->game_time);
    dict_set(out, "max_xp_level", log->max_xp_level);
    dict_set(out, "floors", log->floors);
    dict_set(out, "truncated", log->truncated);
}

// Per-(verb,head) consumption map for PPO consumed-head gating (weak symbol
// read by src/algo.cu). heads: [0]=verb, [1..12]=slot heads 0..11,
// [13..18]=per-verb dir heads. A head is "consumed" iff the sampled verb
// actually uses it.
#define PUFFER_PROVIDES_HEAD_CONSUME_MAP 1
const signed char* env_head_consume_map(int* n_verbs, int* n_atns) {
    static signed char map[NETHACK_NUM_ACTIONS * NUM_ATNS];
    static int built = 0;
    if (!built) {
        memset(map, 0, sizeof(map));
        for (int v = 0; v < NETHACK_NUM_ACTIONS; v++) {
            signed char* row = map + v * NUM_ATNS;
            row[0] = 1;                                   // verb head: always
            int sh = NETHACK_VERBS[v].head;               // slot head 0..11 or -1
            if (sh >= 0) row[1 + sh] = 1;
            int dh = nethack_dir_head(v);
            if (dh >= 0) row[13 + dh] = 1;
        }
        built = 1;
    }
    *n_verbs = NETHACK_NUM_ACTIONS;
    *n_atns = NUM_ATNS;
    return map;
}
