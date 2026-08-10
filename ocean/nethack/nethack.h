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
#include <math.h>
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
extern int        nle_path_drain(nle_ctx_t*, short*, int);
extern long       nle_shop_price(nle_ctx_t*);
extern int        nle_terrain_underfoot(nle_ctx_t*);
extern int        nle_inside_shop(nle_ctx_t*);
extern int        nle_container_at(nle_ctx_t*);
extern int        nle_food_underfoot(nle_ctx_t*);
extern int        nle_discoveries(nle_ctx_t*);
extern int        nle_peaceful_at(nle_ctx_t*, int, int);
extern int        nle_spellprot(nle_ctx_t*);
extern void       nle_weight(nle_ctx_t*, int*, int*);
extern int        nle_spells(nle_ctx_t*, short*, signed char*, signed char*, int);
extern int        nle_spells2(nle_ctx_t*, short*, signed char*, signed char*, int*, int);
extern void       nle_end(nle_ctx_t*);
#ifdef __cplusplus
}
#endif

#include "netlib.h"

#define OBS_SIZE NETHACK_OBS_SIZE
#define NUM_ATNS 20
#define ACT_SIZES {NETHACK_NUM_ACTIONS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, \
    NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, \
    NETHACK_SPELL_SLOTS}
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
    short          inv_true[NLE_INVENTORY_SIZE];
    unsigned char  inv_letters[NLE_INVENTORY_SIZE];
    unsigned char  inv_oclasses[NLE_INVENTORY_SIZE];
    signed char    inv_state[NLE_INVENTORY_SIZE * NLE_INV_STATE_FIELDS];
    short          spell_ids[8];
    signed char    spell_levs[8], spell_fails[8];
    int            spell_knows[8];   // retention turns, 0 = forgotten (slot-faithful)
    int            castlog_pending;  // sid awaiting its post-cast topline (castlog)
    long           stall_prev_turn;  // stall watchdog: last seen game turn
    int            stall_ctr;        // consecutive same-turn steps
    char           stall_lastmsg[96];// last non-empty topline (stall forensics)
    int            n_spells;

    Stats stats;

    // reward-delta trackers
    int prev_action;
    int enh_ready;
    long prev_score;
    long prev_exp;
    long prev_gold;
    long start_gold;
    long prev_hp;
    float prev_ac_led;         // ac-delta ledger: last ledgered AC (durable-weighted)
    float ac_account;          // accrued unpaid ac-delta reward (carries)
    long prev_time;
    int prev_depth;
    int prev_bad_cond;
    unsigned prev_floor;       // dnum << 8 | dlevel at last reward; guards path attribution
    int disc0;                 // discoveries count at reset (episode delta = types learned)
    unsigned long long engid_tested;   // letters engrave-tested this episode (one test = all info)
    int prev_disc;             // discoveries count at last reward (discovery_coef delta)
    // reward coefs
    float discovery_coef;
    float gold_coef;
    float exp_coef;
    float descent_coef;
    float floor_coef;
    float xp_coef;
    float scout_coef;
    float ac_coef;
    float heal_coef;
    float depth_alive;         // 0 = off; else every live step pays depth_alive * dlvl (occupancy)
    float descend_gate;        // 0 = off; 1 = xp>=depth (parity); NEGATIVE = slack: -2 allows 3 under
    float bank_floor;          // zone banking: max free depth while below bank_xp
    float bank_xp;             // 0 = off; xp level that unlocks descent past bank_floor
    float mask_cast;           // 1 = CAST always masked (no-spellcasting ablation)
    float max_episode_steps;   // 0 = default NETHACK_MAX_EPISODE_STEPS (10000)
    float scout_ready;         // 0 = off; else scout only claimable at xp_level >= depth + (ready-1)
    float descent_depth;       // 0 = flat descent; >0 = pay descent_depth * depth per level (deeper pays more)
    float scout_depth;         // 0 = flat scout; >0 = a fresh tile on depth d pays scout_depth * d (deep novelty pays more)
    float exp_log;             // 0 = linear exp reward; >0 = exp_coef * log1p(delta)
    float exp_sqrt;            // 0 = off; else pay exp_sqrt * sqrt(delta) ~= coef * MONSTER LEVEL
    float exp_rel;             // 0 = off; else exp_rel * sqrt(delta)/u_level ~= coef * (m_lev/u_lev): challenge ratio
    float exp_rel_floor;       // subtracted from the ratio first; below it a kill pays ZERO (0 = plain ratio)
    float exp_depth;           // 0 = off; >0 = exp pays exp_depth * delta * depth (same kill pays more deeper)
    float exp_frontier;        // <0 = off; else multiplier on exp for kills behind the frontier (0 = frontier-only pay)
    float exp_site_rel;        // 0 = off; else exp scaled by min(dlvl/u_level, knob): deep-for-your-level kills pay more
    float ac_nospell;          // 0 = off; 1 = AC ledger excludes protection-spell AC (durable AC only)
    float death_penalty;

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
    o->inv_glyphs      = env->inv_glyphs;
    o->inv_true_glyphs = env->inv_true;
    o->inv_letters     = env->inv_letters;
    o->inv_oclasses    = env->inv_oclasses;
    o->inv_state       = env->inv_state;
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
    if (verb->wornreq == UNWORN_ONLY) {
        if (worn) return 0;
        // WEAR (the only armor-only UNWORN verb): mirror the engine's layering
        // rules and known-cursed same-slot swaps (56% of measured prompt aborts)
        if (verb->item_classes == (1u << 3)) {
            int gn = env->inv_glyphs[i] - NH_GLYPH_OBJ_OFF;
            int cat = (gn >= 0 && gn < NH_NUM_OBJECTS) ? nh_obj_armcat[gn] : -1;
            if (cat >= 0) {
                for (int j = 0; j < NETHACK_INV_SLOTS && env->inv_letters[j]; j++) {
                    if (!(env->inv_state[j * NLE_INV_STATE_FIELDS + 5] & 1)) continue;
                    int gj = env->inv_glyphs[j] - NH_GLYPH_OBJ_OFF;
                    int cj = (gj >= 0 && gj < NH_NUM_OBJECTS) ? nh_obj_armcat[gj] : -1;
                    if ((cat == 0 || cat == 6) && (cj == 5 || (cat == 6 && cj == 0))) return 0;
                    { static int abl = -1;
                      if (abl < 0) abl = getenv("NH_ABL_CSWAP") != NULL;
                      if (!abl && cj == cat
                          && env->inv_state[j * NLE_INV_STATE_FIELDS + 0] == 1) return 0; }
                }
            }
        }
        return 1;
    }
    return 1;
}

// visible HOSTILE target (monster/detected/warning glyph) on the ray within
// 8 tiles. A peaceful in the line blocks the shot: ranged attacks have no
// "really attack?" prompt, so aiming at (or through) a watchman starts a war
static int nethack_ray_target(Nethack* env, int dx, int dy) {
    long hx = env->blstats[NLE_BL_X], hy = env->blstats[NLE_BL_Y];
    for (int k = 1; k <= 8; k++) {
        long x = hx + dx * k, y = hy + dy * k;
        if (x < 0 || x >= NH_COLS || y < 0 || y >= NH_ROWS) return 0;
        int gl = env->glyphs[y * NH_COLS + x];
        static int ablv1 = -1;
        if (ablv1 < 0) ablv1 = getenv("NH_ABL_RAYV2") != NULL;
        if (ablv1) {   // ablation: v1 accept-set (no warning band, no peaceful block)
            if ((gl >= 0 && gl < NETHACK_NUMMONS) || (gl >= 762 && gl < 1144)) return 1;
            continue;
        }
        if ((gl >= 0 && gl < NETHACK_NUMMONS)
            || (gl >= 762 && gl < 1144)
            || (gl >= 5589 && gl < 5595))
            return !nle_peaceful_at(env->ctx, (int)x + 1, (int)y);
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

    // terrain, not the map glyph: an object on the tile occludes the stairs
    // (underfoot_glyphs shows the top item), which used to mask DOWN off and
    // strand the agent on a littered staircase
    int terrain = nle_terrain_underfoot(env->ctx);
    if (terrain != NETHACK_GLYPH_DNSTAIR && terrain != NETHACK_GLYPH_DNLADDER)
        mask[NETHACK_ACT_DOWN] = 0;
    if (terrain != NETHACK_GLYPH_UPSTAIR && terrain != NETHACK_GLYPH_UPLADDER)
        mask[NETHACK_ACT_UP] = 0;
    if (env->blstats[NLE_BL_DEPTH] <= 1) mask[NETHACK_ACT_UP] = 0;   // declined exit
    // descend gate: hold the agent on the level until its experience level
    // matches the depth it's about to enter. 0 disables. This is state-dependent
    // STRATEGY masking (not action-space design like SEARCH20) -- an experiment
    // to price the xp/depth ordering, not a shipped default.
    // ZONE BANKING (autoascend-style two-phase, generalized): roam freely
    // within depth <= bank_floor, but no descent past it until xp >= bank_xp;
    // unconstrained forever after. Enforces the grind-then-descend sequencing
    // the ~400-step credit horizon can't learn. bank_floor=1/bank_xp=8 is the
    // strict autoascend rule (stalled: dlvl-1 spawn trickle too slow for our
    // episode budget); 4/7 banks where the upper Mines supply real exp.
    if (env->bank_xp > 0.0f) {
        if (env->blstats[NLE_BL_DEPTH] >= (long)env->bank_floor
            && env->blstats[NLE_BL_XP] < (long)env->bank_xp) {
            if (mask[NETHACK_ACT_DOWN]) env->stats.descend_blocked++;
            mask[NETHACK_ACT_DOWN] = 0;
        }
    }
    if (env->descend_gate != 0.0f
        && (float)env->blstats[NLE_BL_XP]
             < (float)env->blstats[NLE_BL_DEPTH] + env->descend_gate - 1.0f) {
        if (mask[NETHACK_ACT_DOWN]) env->stats.descend_blocked++;   // counts only real binds
        mask[NETHACK_ACT_DOWN] = 0;
    }
    if (!on_object && !on_corpse) mask[NETHACK_ACT_PICKUP] = 0;
    if (terrain != NETHACK_GLYPH_ALTAR) mask[NETHACK_ACT_ALTAR_ID] = 0;
    // presence is public (the container renders); locked/empty is learnable
    if (!nle_container_at(env->ctx)) mask[NETHACK_ACT_TIP] = 0;
    // engrave-test needs an unidentified wand not yet tested this episode:
    // only 3 types formally identify, the rest just print their tell once —
    // re-testing is informationless and drains charges
    { int unid_wand = 0;
      for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i]; i++) {
          if (env->inv_oclasses[i] != 11
              || env->inv_state[i * NLE_INV_STATE_FIELDS + 6] != 0) continue;
          int lb = nethack_letter_bit(env->inv_letters[i]);
          if (lb >= 0 && (env->engid_tested & (1ULL << lb))) continue;
          unid_wand = 1; break;
      }
      if (!unid_wand) mask[NETHACK_ACT_ENGRAVE_ID] = 0;
      { static int abl = -1;
        if (abl < 0) abl = getenv("NH_ABL_ENGID") != NULL;
        if (abl) mask[NETHACK_ACT_ENGRAVE_ID] = 0; } }
    // ELBERETH: unengravable terrain (fountain/water/lava/air/cloud) refuses
    // for free -- measured census residue.
    { static int abl = -1;
      if (abl < 0) abl = getenv("NH_ABL_ELBTER") != NULL;
      int tg = nle_terrain_underfoot(env->ctx) - 2359;
      if (!abl && (tg == 31 || tg == 32 || tg == 34 || tg == 39 || tg == 40 || tg == 41))
          mask[NETHACK_ACT_ELBERETH] = 0; }
    // ELBERETH: the engine refuses engraving while levitating or engulfed
    // ("can't reach the floor", a free no-op) -- 61% of attempts in the wild.
    if ((env->blstats[NLE_BL_CONDITION] & 0x400L) || (env->internal[6] & 4))
        mask[NETHACK_ACT_ELBERETH] = 0;
    // CAST + spell-slot head: a slot is castable iff still known (retention
    // > 0) and Pw covers 5 * level. fail%% is deliberately NOT masked -- it's
    // in the obs; the trade is learnable. hunger gate mirrors spell.c:953
    // (uhunger <= 10 refuses for free; census caught a 5.6M-step spam loop).
    // NH_SPELL=0: slot 0 only (legacy single-spell behavior); default on.
    { static int spf = -1;
      if (spf < 0) { const char* e = getenv("NH_SPELL"); spf = e ? (e[0] && e[0] != '0') : 1; }
      unsigned char* sp = mask + NETHACK_NUM_ACTIONS + 12 * NETHACK_INV_SLOTS
                          + NETHACK_DIR_HEADS * NETHACK_NUM_DIRS;
      memset(sp, 0, NETHACK_SPELL_SLOTS);
      int lim = spf ? env->n_spells : (env->n_spells > 0 ? 1 : 0);
      int any = 0;
      for (int s = 0; s < lim && s < NETHACK_SPELL_SLOTS; s++) {
          if (env->spell_ids[s] > 0 && env->spell_knows[s] > 0
              && env->blstats[NLE_BL_ENE] >= 5L * (long)env->spell_levs[s]) {
              sp[s] = 1; any = 1;
          }
      }
      if (!any) sp[0] = 1;   // unconsumed head still needs a legal entry
      if (env->mask_cast != 0.0f || !any || env->internal[7] <= 10)
          mask[NETHACK_ACT_CAST] = 0;
    }
    // shop goods we can't pay for: picking them up incurs a bill the agent
    // has no way to settle, so gate on affordability (price is quoted to the
    // player on arrival, so this is public information)
    long shop_price = nle_shop_price(env->ctx);
    if (shop_price > env->blstats[NLE_BL_GOLD]) mask[NETHACK_ACT_PICKUP] = 0;

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
        // EAT: any-object underfoot kept EAT legal on inedible piles (own
        // drops!) -- a measured 1M-step "nothing to eat" refusal loop. Gate
        // on actual floor food (pile walk, corpses included).
        int floor_food = (a == NETHACK_ACT_EAT) && (on_object || on_corpse)
                         && nle_food_underfoot(env->ctx);
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
    { // NH_MASK_TRICE: eval-only ceiling probe -- forbid stepping into
      // cockatrice family (melee touch = stoning for a bare-handed Monk).
      static int mt = -1;
      if (mt < 0) mt = getenv("NH_MASK_TRICE") != NULL;
      if (mt) for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
          long col = hero_x + NETHACK_DIR_DX[d], row = hero_y + NETHACK_DIR_DY[d];
          if (row < 0 || row >= NH_ROWS || col < 0 || col >= NH_COLS) continue;
          int g = env->glyphs[row * NH_COLS + col] % 381;   // pm idx mod across mon-derived ranges
          int gr = env->glyphs[row * NH_COLS + col];
          if (gr < 1906 && (g == 9 || g == 10)) dirs[d] = 0;   // chickatrice(9)/cockatrice(10)
      } }
    if (!legal_dirs) memset(dirs, 1, NETHACK_NUM_DIRS);
    for (int h = 1; h < NETHACK_DIR_HEADS; h++)
        memcpy(dirs + h * NETHACK_NUM_DIRS, dirs, NETHACK_NUM_DIRS);
    // MOVE-head-only refinements: void-action mirrors
    { unsigned char keep[NETHACK_NUM_DIRS];
      memcpy(keep, dirs, NETHACK_NUM_DIRS);
      int under = nle_terrain_underfoot(env->ctx) - 2359;
      int on_door = (under >= 12 && under <= 14);
      for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
          if (!dirs[d]) continue;
          long col = hero_x + NETHACK_DIR_DX[d], row = hero_y + NETHACK_DIR_DY[d];
          if (row < 0 || row >= NH_ROWS || col < 0 || col >= NH_COLS) continue;
          int g = env->glyphs[row * NH_COLS + col];
          { static int ablp = -1;
            if (ablp < 0) ablp = getenv("NH_ABL_PEACE") != NULL;
            if (!ablp && g >= 0 && g < 381
                && nle_peaceful_at(env->ctx, (int)col + 1, (int)row))
                { dirs[d] = 0; continue; } }
          { static int abld = -1;
            if (abld < 0) abld = getenv("NH_ABL_DOOR") != NULL;
            if (!abld && d >= 4 && (on_door || (g >= 2371 && g <= 2373))) dirs[d] = 0; }
      }
      { int any = 0;
        for (int d = 0; d < NETHACK_NUM_DIRS; d++) any |= dirs[d];
        if (!any) memcpy(dirs, keep, NETHACK_NUM_DIRS); } }   // cornered: restore
    // RUN head: a run aimed at an adjacent hostile is void with p=1.0
    // (census: 70000/70000 zero-clock). Directional only -- escape dirs stay.
    { static int ablr = -1;
      if (ablr < 0) ablr = getenv("NH_ABL_DIRRUN") != NULL;
      unsigned char* rdirs = dirs + 1 * NETHACK_NUM_DIRS;   // RUN dir head
      unsigned char rkeep[NETHACK_NUM_DIRS];
      memcpy(rkeep, rdirs, NETHACK_NUM_DIRS);
      for (int d = 0; ablr == 0 && d < NETHACK_NUM_DIRS; d++) {
          if (!rdirs[d]) continue;
          long col = hero_x + NETHACK_DIR_DX[d], row = hero_y + NETHACK_DIR_DY[d];
          if (row < 0 || row >= NH_ROWS || col < 0 || col >= NH_COLS) continue;
          int g = env->glyphs[row * NH_COLS + col];
          if (g >= 0 && g < 381 && !nle_peaceful_at(env->ctx, (int)col + 1, (int)row))
              rdirs[d] = 0;
      }
      { int any = 0;
        for (int d = 0; d < NETHACK_NUM_DIRS; d++) any |= rdirs[d];
        if (!any) memcpy(rdirs, rkeep, NETHACK_NUM_DIRS); } }
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
    // engraving state 0/1/2 (bit 2 = engulfed, mask-only -- keep it out of obs)
    extra[0] = env->internal[6] & 3;
    extra[1] = env->prev_action;
    for (int i = 0; i < NLE_INVENTORY_SIZE; i++) {
        int oc = env->inv_oclasses[i];
        if (oc >= NETHACK_NUM_OCLASSES) break;   // padded tail
        extra[2 + oc]++;
    }
    env->n_spells = nle_spells2(env->ctx, env->spell_ids, env->spell_levs,
                                env->spell_fails, env->spell_knows,
                                NETHACK_SPELL_SLOTS);
    { // full 8-slot channel with retention (default); NH_SPELL=0 = first-slot
      // info only, retention hidden (info-parity control for the new arch)
      static int spf = -1;
      if (spf < 0) { const char* e = getenv("NH_SPELL"); spf = e ? (e[0] && e[0] != '0') : 1; }
      extra[NETHACK_EXTRA_SPELL] = env->n_spells;
      for (int s = 0; s < NETHACK_SPELL_SLOTS; s++) {
          int* q = extra + NETHACK_EXTRA_SPELL + 1 + 4 * s;
          int known = s < env->n_spells && env->spell_ids[s] > 0 && (spf || s == 0);
          q[0] = known ? env->spell_ids[s] : 0;
          q[1] = known ? env->spell_levs[s] : 0;
          q[2] = known ? env->spell_fails[s] : 0;
          q[3] = known && spf ? env->spell_knows[s] : 0;
      } }
    { int wt, wcap;
      nle_weight(env->ctx, &wt, &wcap);
      if (wcap < 1) wcap = 1;
      extra[NETHACK_EXTRA_WEIGHT + 0] = (int)(100L * wt / wcap);
      extra[NETHACK_EXTRA_WEIGHT + 1] = wcap; }
    { long sp = nle_shop_price(env->ctx);
      extra[NETHACK_EXTRA_SHOP] = nle_inside_shop(env->ctx);
      // gold/price as a percent, capped at 100 (0 = no purchase available here)
      long g = env->blstats[NLE_BL_GOLD];
      extra[NETHACK_EXTRA_SHOP + 1] = (sp > 0)
          ? (int32_t)(g >= sp ? 100 : (g * 100) / sp) : 0; }
    unsigned char* ex = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_EXTRA;
    for (int i = 0; i < NETHACK_EXTRA_INTS; i++) {
        uint32_t v = (uint32_t)extra[i];
        ex[4*i + 0] = (unsigned char)(v & 0xffu);
        ex[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        ex[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        ex[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    // slot glyphs; NH_DISC_SWAP=1: discovered identity REPLACES the appearance
    // glyph (stable rep once known; the add-channel below goes silent)
    static int dsw = -1;
    if (dsw < 0) { const char* e = getenv("NH_DISC_SWAP"); dsw = e && e[0] && e[0] != '0'; }
    unsigned char* iv = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_INV;
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        uint16_t g = env->inv_oclasses[i] < NETHACK_NUM_OCLASSES
                   ? (dsw && env->inv_true[i] != NETHACK_PAD_GLYPH
                      ? (uint16_t)env->inv_true[i] : (uint16_t)env->inv_glyphs[i])
                   : (uint16_t)NETHACK_PAD_GLYPH;
        iv[2*i + 0] = (unsigned char)(g & 0xffu);
        iv[2*i + 1] = (unsigned char)((g >> 8) & 0xffu);
    }
    // item state
    memcpy(((obs_t*)env->agents[0].observations) + NETHACK_OFF_INVST, env->inv_state, sizeof(env->inv_state));
    // discovered-type glyphs (engine pads with NO_GLYPH == NETHACK_PAD_GLYPH)
    unsigned char* it = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_INVTRUE;
    static int abldisc = -1;
    if (abldisc < 0) abldisc = getenv("NH_ABL_DISC") != NULL;
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        uint16_t g = !abldisc && !dsw && env->inv_oclasses[i] < NETHACK_NUM_OCLASSES
                   ? (uint16_t)env->inv_true[i] : (uint16_t)NETHACK_PAD_GLYPH;
        it[2*i + 0] = (unsigned char)(g & 0xffu);
        it[2*i + 1] = (unsigned char)((g >> 8) & 0xffu);
    }
    { // stall watchdog: a key-eating modal (getpos-class) can freeze the game
      // turn while swallowing every action. At 96 frozen steps, fire an
      // ESC burst to exit the modal (ESC is a harmless no-op at the main
      // prompt, so wall-bump refusal loops are unaffected). NH_STALLLOG dumps.
      static int sl = -1;
      if (sl < 0) sl = getenv("NH_STALLLOG") != NULL;
      if (sl && env->message[0]) {
          strncpy(env->stall_lastmsg, (const char*)env->message,
                  sizeof(env->stall_lastmsg) - 1);
          env->stall_lastmsg[sizeof(env->stall_lastmsg) - 1] = 0;
      }
      long turn = env->blstats[NLE_BL_TIME];
      if (turn == env->stall_prev_turn && !env->obs.done) {
          // burst only when a modal is actually indicated: behavioral
          // free-action loops (misc all zero) are the policy's own business
          if (++env->stall_ctr == 96
              && (env->misc[0] || env->misc[1] || env->misc[2])) {
              if (sl)
                  fprintf(stderr, "STALL env=%p t=%ld misc=%d,%d,%d "
                          "lastmsg=%.90s -> ESC burst\n", (void*)env, turn,
                          env->misc[0], env->misc[1], env->misc[2],
                          env->stall_lastmsg);
              for (int k = 0; k < 8 && !env->obs.done; k++) {
                  env->obs.action = 27;
                  env->ctx = nle_step(env->ctx, &env->obs);
              }
              env->stall_ctr = 0;   // re-arm; recovery shows as turn advance
          } else if (env->stall_ctr >= 96) env->stall_ctr = 0;
      } else { env->stall_ctr = 0; env->stall_prev_turn = turn; }
    }
    // topline
    unsigned char* mv = ((obs_t*)env->agents[0].observations) + NETHACK_OFF_MSG;
    size_t mlen = strnlen((const char*)env->message, NETHACK_MSG_LEN);
    memcpy(mv, env->message, mlen);
    if (mlen < (size_t)NETHACK_MSG_LEN) memset(mv + mlen, 0, NETHACK_MSG_LEN - mlen);

    if (env->action_mask != NULL) nethack_compute_mask(env);
}

// logging

static void nethack_add_log(Nethack* env, int how) {   // how: nle how_done, -1 = truncated
    { static int wlog = -1;
      if (wlog < 0) wlog = getenv("NH_WASTELOG") != NULL;
      if (wlog) {
          fprintf(stderr, "WASTE len=%d", env->stats.length);
          for (int v = 0; v < NETHACK_NUM_ACTIONS; v++)
              fprintf(stderr, " %d", env->stats.waste[v]);
          fprintf(stderr, "\n");
      } }
    { // NH_EPSCORE: per-episode score dump for distribution reads (median etc.)
      static int epscore = -1;
      if (epscore < 0) epscore = getenv("NH_EPSCORE") != NULL;
      if (epscore) {
          FILE* f = fopen(getenv("NH_EPSCORE"), "a");
          if (f) { fprintf(f, "%ld %d %d %d %d %d\n", env->prev_score,
                           env->stats.max_depth, how, env->stats.saw_stone,
                           env->prev_action, env->stats.min_ac); fclose(f); }
      } }
    for (int v = 0; v < NETHACK_NUM_ACTIONS; v++)
        env->log.verb_uses[v] += (float)env->stats.verb_uses[v];
    env->log.perf += (float)env->prev_score;
    env->log.score += (float)env->prev_score;
    env->log.valid_moves += (float)env->stats.valid_moves;
    env->log.illegal_actions += (float)env->stats.illegal_actions;
    env->log.new_tiles += (float)env->stats.new_tiles;
    env->log.max_depth += (float)env->stats.max_depth;
    env->log.floors += (float)env->stats.floors;
    env->log.reach_d10 += env->stats.max_depth >= 10 ? 1.0f : 0.0f;
    env->log.descend_blocked += (float)env->stats.descend_blocked;
    env->log.scout_held += (float)env->stats.scout_held;
    for (int i = 0; i < 9; i++) {
        env->log.r_raw[i] += env->stats.r_raw[i];
        env->log.r_clip[i] += env->stats.r_clip[i];
    }
    env->log.r_death += env->stats.r_death;
    env->log.enhances += (float)env->stats.enhances;
    env->log.prayers_low_hp += (float)env->stats.prayers_low_hp;
    env->log.prayers_starving += (float)env->stats.prayers_starving;
    env->log.floor_eats += (float)env->stats.floor_eats;
    env->log.sells += (float)env->stats.sells;
    env->log.buys += (float)env->stats.buys;
    env->log.sale_gold += (float)env->stats.sale_gold;
    env->log.drop_value += (float)env->stats.drop_value;
    env->log.discoveries += (float)(nle_discoveries(env->ctx) - env->disc0);
    env->log.trouble_frac += (float)env->stats.trouble_steps / (float)env->stats.length;
    env->log.prayers_fed += (float)env->stats.prayers_fed;
    env->log.altar_steps += (float)env->stats.altar_steps;
    env->log.wear_blind += (float)env->stats.wear_blind;
    env->log.cursed_worn_frac += (float)env->stats.cursed_worn_steps / (float)env->stats.length;
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
    if (how >= 0 && env->stats.last_hunger >= NETHACK_HUNGER_WEAK)
        env->log.death_weak += 1.0f;
    if (how == 0) {
        env->log.death_mon_level += (float)env->internal[NETHACK_INTERNAL_KILLER_MLEV];
        env->log.death_adj_monsters += (float)env->stats.last_adj;
        { static int sl = -1;
          if (sl < 0) sl = getenv("NH_STONELOG") != NULL;
          if (sl && how == 8)
              fprintf(stderr, "STONELOG sawcond=%d score=%ld\n",
                      env->stats.saw_stone, env->prev_score); }
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

    // discovered types render as true glyphs on the map (engine opt-in)
    if (!getenv("NH_ABL_DISC")) setenv("NLE_TRUE_GLYPHS", "1", 1);

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
    env->prev_depth = (int)env->blstats[NLE_BL_DEPTH];
    env->prev_bad_cond = __builtin_popcount((unsigned)env->blstats[NLE_BL_CONDITION] & NETHACK_COND_BAD);
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_action = -1;
    { short scratch[2 * NETHACK_PATH_MAX];   // discard boot-walk residue
      nle_path_drain(env->ctx, scratch, NETHACK_PATH_MAX); }
    env->prev_floor = (unsigned)(env->blstats[NLE_BL_DNUM] << 8 | env->blstats[NLE_BL_DLEVEL]);
    env->disc0 = nle_discoveries(env->ctx);
    env->prev_disc = env->disc0;
    env->engid_tested = 0;
    env->enh_ready = 0;
    memset(&env->stats, 0, sizeof(env->stats));
    env->stats.max_depth = env->prev_depth;
    env->stats.max_xp = (int)env->blstats[NLE_BL_XP];
    env->stats.min_ac = (int)env->blstats[NLE_BL_AC];
    env->prev_ac_led = (float)env->blstats[NLE_BL_AC];
    env->ac_account = 0.0f;
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

    // prayer demand vs supply: "trouble" is roughly what pray.c will fix
    long hunger = env->blstats[NLE_BL_HUNGER], maxhp = env->blstats[NLE_BL_HPMAX];
    if (hunger >= NETHACK_HUNGER_WEAK || (maxhp > 0 && hp * 7 <= maxhp))
        env->stats.trouble_steps++;
    if (env->prev_action == NETHACK_ACT_PRAY && hunger < env->stats.last_hunger)
        env->stats.prayers_fed++;
    env->stats.last_hunger = (int)hunger;

    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        const signed char* s = &env->inv_state[i * NLE_INV_STATE_FIELDS];
        if ((s[5] & 1) && s[0] == 1) { env->stats.cursed_worn_steps++; break; }
    }

    // altar exposure: terrain, so loot on the altar doesn't hide it
    if (nle_terrain_underfoot(env->ctx) == NETHACK_GLYPH_ALTAR)
        env->stats.altar_steps++;

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
    if (env->blstats[NLE_BL_CONDITION] & 0x1L) env->stats.saw_stone = 1;

    env->prev_score = env->blstats[NLE_BL_SCORE];
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_depth = depth;
}

// bitmaps are keyed by (dnum, dlevel), slots allocated on first entry to a
// floor; depth-only keying aliased branches (mines tiles vs main at same depth)
// Fractional scout claim. A tile on depth D is worth its full scout_coef only
// once the hero has reached the required experience level; below that it pays
// pro-rata and the REMAINDER STAYS ON THE BOARD for a later, stronger visit.
// Returns the newly-claimable fraction in [0,1]. Total over all visits to a
// tile is capped at 1.0, so revisiting cannot farm it.
//   required = max(1, (int)(depth * scout_ready + 0.5))  (integers throughout:
//   a fractional stored level would let truncation re-claim the same sliver)
// scout_ready <= 0 restores plain first-visit semantics (claim once, in full).
static float nethack_tile_claim(Nethack* env, long dn, long dl, long px, long py) {
    if (px < 0 || px >= NH_COLS || py < 0 || py >= NH_ROWS) return 0.0f;
    if (dn < 0 || dn > 15 || dl < 1 || dl > 64) return 0.0f;
    unsigned short key = (unsigned short)(dn << 8 | dl);
    int d = -1;
    for (int i = 0; i < env->stats.n_visited_floors; i++)
        if (env->stats.visited_key[i] == key) { d = i; break; }
    if (d < 0) {
        if (env->stats.n_visited_floors >= NETHACK_MAX_DEPTH) return 0.0f;
        d = env->stats.n_visited_floors++;
        env->stats.visited_key[d] = key;
    }
    int idx = (int)py * NH_COLS + (int)px;
    unsigned char prev = env->stats.visited[d][idx];
    if (env->scout_ready <= 0.0f) {          // plain first-visit
        if (prev) return 0.0f;
        env->stats.visited[d][idx] = 1;
        return 1.0f;
    }
    int depth = (int)env->blstats[NLE_BL_DEPTH];
    if (depth < 1) depth = 1;
    int req = (int)((float)depth * env->scout_ready + 0.5f);
    if (req < 1) req = 1;
    if (req > 255) req = 255;
    int cap = env->stats.max_xp < req ? env->stats.max_xp : req;   // max_xp is monotonic
    if (cap <= (int)prev) return 0.0f;
    env->stats.visited[d][idx] = (unsigned char)cap;
    return (float)(cap - (int)prev) / (float)req;
}


static float nethack_reward(Nethack* env, int illegal) {
    // death payout
    if (env->obs.done) {
        env->stats.r_death += env->death_penalty;
        return env->death_penalty;
    }
    nethack_update_stats(env);

    int depth = (int)env->blstats[NLE_BL_DEPTH];

    // per-term ledger: t[0..8] = exp gold descent floor xp scout ac heal occupancy
    float t[9] = {0};

    // exp, gains only
    long exp = env->blstats[NLE_BL_EXP];
    if (exp > env->prev_exp) {
        float d = (float)(exp - env->prev_exp);
        // linear saturates the [-1,1] clamp at 1/exp_coef points, flattening
        // every kill above it; log1p keeps deep, high-value kills ordered.
        // exp_depth multiplies by dungeon depth: the engine pays exp for
        // MONSTER level, not depth (exper.c m_lev^2), and spawns track
        // (zlevel + u.ulevel)/2 -- so leveling in place half-feeds itself.
        // Pricing depth here breaks the farm-in-place loop.
        // exp_sqrt: engine exp ~ 1 + m_lev^2, so sqrt(delta) recovers the
        // MONSTER's level -- pay fighting up, wherever it happens. The spawn
        // cap (dlvl+ulevel)/2 makes big payouts deep-only, so the depth pull
        // is mediated by fights: tagging depth or camping earns nothing.
        // exp_rel divides by the PLAYER's level: the challenge ratio
        // m_lev/u_lev. Yesterday's prey decays toward zero pay as you level,
        // so income maintenance requires punching up -- and bigger monsters
        // live deeper. A treadmill that renews at every stage.
        t[0] = env->exp_rel > 0.0f   ? env->exp_rel
                                       * fmaxf(0.0f, sqrtf(d)
                                           / fmaxf((float)env->blstats[NLE_BL_XP], 1.0f)
                                           - env->exp_rel_floor)
             : env->exp_sqrt > 0.0f  ? env->exp_sqrt * sqrtf(d)
             : env->exp_depth > 0.0f ? env->exp_depth * d * (float)depth
             : env->exp_log > 0.0f   ? env->exp_coef * log1pf(d)
                                     : env->exp_coef * d;
        // kill-SITE weighting, composable with any exp shape above:
        // exp_frontier: kills behind the frontier (dlvl < max_depth) scale by
        // it (0 = frontier-only pay). exp_site_rel: scale by dlvl/u_level
        // capped at the knob -- deep-for-your-level kills pay more.
        if (env->exp_frontier >= 0.0f && (float)depth < (float)env->stats.max_depth)
            t[0] *= env->exp_frontier;
        if (env->exp_site_rel > 0.0f) {
            float sr = (float)depth / fmaxf((float)env->blstats[NLE_BL_XP], 1.0f);
            t[0] *= sr < env->exp_site_rel ? sr : env->exp_site_rel;
        }
    }
    env->prev_exp = exp;

    // gold, net of start
    long g = env->blstats[NLE_BL_GOLD] - env->start_gold;
    if (g < 0) g = 0;
    t[1] = env->gold_coef * (float)(g - env->prev_gold);
    env->prev_gold = g;

    // descent_coef pays per max-depth delta, floor_coef per new (dnum, dlevel) floor
    { long dn = env->blstats[NLE_BL_DNUM], dl = env->blstats[NLE_BL_DLEVEL];
      if (dn >= 0 && dn < 16 && dl >= 1 && dl <= 64) {
          unsigned long long fb = 1ULL << (dl - 1);
          if (!(env->stats.floors_bits[dn] & fb)) {
              env->stats.floors_bits[dn] |= fb;
              env->stats.floors++;
              t[3] = env->floor_coef;
          }
      } }
    if (depth > env->stats.max_depth) {
        // depth-scaled: deeper levels pay proportionally more, coefficient
        // sized so the deepest relevant descent (~25) just reaches the clamp
        t[2] = env->descent_depth > 0.0f
             ? env->descent_depth * (float)(depth - env->stats.max_depth) * (float)depth
             : env->descent_coef * (float)(depth - env->stats.max_depth);
        env->stats.max_depth = depth;
    }
    // additions in the pre-ledger order (bit-exact float sums)
    float r = t[0];
    r += t[1];
    r += t[3];
    r += t[2];

    // discovery_coef pays per newly identified object type (any ID route);
    // oc_name_known is monotonic per game, so this cannot be farmed
    if (env->discovery_coef != 0.0f) {
        int dc = nle_discoveries(env->ctx);
        r += env->discovery_coef * (float)(dc - env->prev_disc);
        env->prev_disc = dc;
    }

    // heal/cure stats (no reward); ac restored 2026-08-05 (see r_raw/ac)
    long hp = env->blstats[NLE_BL_HP];
    long hp_delta = hp - env->prev_hp;
    if (hp_delta > 0 && (env->prev_action == NETHACK_ACT_QUAFF
                      || env->prev_action == NETHACK_ACT_PRAY
                      || env->prev_action == NETHACK_ACT_CAST)) {
        // gain-only, heal-action-gated (regen never pays). CAST included by
        // design: the healing spell needs an income path to be learned.
        // Pw is renewable, so watch r_raw/heal for a scratch-and-cast farm.
        t[7] = env->heal_coef * (float)hp_delta;
        env->stats.heal_hp += hp_delta;
    }
    env->prev_hp = hp;
    long ac = env->blstats[NLE_BL_AC];
    // ac: DELTA with a conservation ledger (2026-08-07, replaces monotone-best).
    // Each AC change accrues into an account; at most +-ac_coef pays out per
    // step and the remainder CARRIES. Restores exact telescoping under the
    // per-step clamp: the castv2 arbitrage (losses hidden on clipped steps)
    // dies because unpaid debt persists until settled; churn nets zero for
    // real turn cost. Transient spell AC self-cancels over its cycle (pay on
    // stack, charge on decay), so no armor-only carve-out is needed; only
    // durable AC held to episode end accumulates net reward.
    env->stats.ac_sum += ac;
    if ((int)ac < env->stats.min_ac) env->stats.min_ac = (int)ac;   // stat only
    // ac_nospell: ledger runs on durable AC (displayed + spellprot), so the
    // protection spell's transient AC never enters the account -- kills the
    // die-with-debt and discount arbitrage on cast cycles (cast-spam driver)
    // ac_nospell is the UNPAID fraction of spell AC: 0 = full pay (legacy),
    // 1 = spell AC never enters the ledger, 0.75 = spells pay out 25%
    float ac_led = env->ac_nospell != 0.0f
        ? (float)ac + env->ac_nospell * (float)nle_spellprot(env->ctx) : (float)ac;
    env->ac_account += env->ac_coef * (env->prev_ac_led - ac_led);
    env->prev_ac_led = ac_led;
    { float cap = env->ac_coef;
      float pay = env->ac_account > cap ? cap
                : (env->ac_account < -cap ? -cap : env->ac_account);
      t[6] = pay;
      env->ac_account -= pay; }
    int bad_cond = __builtin_popcount((unsigned)env->blstats[NLE_BL_CONDITION] & NETHACK_COND_BAD);
    if (bad_cond < env->prev_bad_cond) env->stats.cures += env->prev_bad_cond - bad_cond;
    env->prev_bad_cond = bad_cond;

    // xp level, max only
    int xp = (int)env->blstats[NLE_BL_XP];
    if (xp > env->stats.max_xp) {
        t[4] = env->xp_coef * (float)(xp - env->stats.max_xp);
        env->stats.max_xp = xp;
    }

    // scout: pay every tile walked this step. A rush resolves many moves
    // inside one nle_step and turns corners, so drain the engine's path
    // rather than crediting only where the hero stopped.
    { long dn = env->blstats[NLE_BL_DNUM], dl = env->blstats[NLE_BL_DLEVEL];
      unsigned floor = (unsigned)(dn << 8 | dl);
      short path[2 * NETHACK_PATH_MAX];
      int n = nle_path_drain(env->ctx, path, NETHACK_PATH_MAX);
      // a mid-step level change (trapdoor, hole) leaves path coords from the
      // OLD floor; attributing them here would mark and pay the wrong tiles
      if (floor != env->prev_floor) n = 0;
      env->prev_floor = floor;
      // fractional claim: a deep tile pays pro-rata to the hero's level and
      // keeps its remainder for a later visit (see nethack_tile_claim)
      float fresh = 0.0f;
      int touched = 0;
      for (int i = 0; i < n; i++) {
          float c = nethack_tile_claim(env, dn, dl, path[2 * i], path[2 * i + 1]);
          fresh += c; touched += (c > 0.0f);
      }
      // no usable path (non-move verbs, never left the tile, or floor change)
      if (!n) {
          float c = nethack_tile_claim(env, dn, dl,
                        env->blstats[NLE_BL_X], env->blstats[NLE_BL_Y]);
          fresh += c; touched += (c > 0.0f);
      }
      // steps where a walked tile paid less than its full value
      if (n && fresh < (float)n - 1e-6f) env->stats.scout_held++;
      if (fresh > 0.0f) {
          // depth-scaled scout: deep fresh tiles pay more than shallow ones —
          // the novelty gradient that pushes descent, made to escalate with
          // depth instead of paying flat. Overrides scout_coef when set.
          t[5] = env->scout_depth > 0.0f
               ? env->scout_depth * (float)depth * fresh
               : env->scout_coef * fresh;
          env->stats.new_tiles += touched;
      } }
    r += t[4];
    // occupancy: alive at depth d pays depth_alive * d every step. Dense, tiny,
    // unclampable; its integral is depth-weighted lifespan -- pays OPERATING
    // deep, not touching it. Camping tell: game_time up with kills flat.
    if (env->depth_alive > 0.0f)
        t[8] = env->depth_alive * (float)depth;
    r += t[5];
    r += t[6];
    r += t[7];
    r += t[8];

    // ledger: raw, plus clip-attributed (terms scaled by clamp(r)/r; the
    // trainer clamps rewards to [-1,1] before training)
    float rc = r < -1.0f ? -1.0f : r > 1.0f ? 1.0f : r;
    float scale = r != 0.0f ? rc / r : 0.0f;
    for (int i = 0; i < 9; i++) {
        env->stats.r_raw[i] += t[i];
        env->stats.r_clip[i] += t[i] * scale;
    }
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
        st->verb_uses[verb]++;
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
    case NETHACK_ACT_ELBERETH: {
        static int elog = -1;
        if (elog < 0) elog = getenv("NH_ENGRLOG") != NULL;
        st->verb_uses[verb]++;
        nethack_do_elbereth(env);
        if (elog) fprintf(stderr, "ENGRLOG dl=%ld\n", (long)env->blstats[NLE_BL_DEPTH]);
        break; }
    case NETHACK_ACT_SEARCH20:
        st->verb_uses[verb]++;
        nethack_send_key(env, '2');
        nethack_send_key(env, '0');
        nethack_send_key(env, 's');
        break;
    case NETHACK_ACT_PICKUP: {
        int purchase = nle_shop_price(env->ctx) > 0;   // 0 = own/no-charge pile, nothing to pay
        st->verb_uses[verb]++;
        nethack_send_key(env, ',');
        nethack_answer_menu(env);
        // shop pickup bills you; settle it now (the mask guarantees we can)
        if (purchase && !env->obs.done) {
            nethack_send_key(env, 'p');
            st->buys++;
        }
        break; }
    case NETHACK_ACT_PRAY:
        if (4 * env->blstats[NLE_BL_HP] <= env->blstats[NLE_BL_HPMAX]) st->prayers_low_hp++;
        if (env->blstats[NLE_BL_HUNGER] >= NETHACK_HUNGER_WEAK) st->prayers_starving++;
        st->verb_uses[verb]++;
        nethack_send_key(env, 0x80 | 'p');
        break;
    case NETHACK_ACT_WEAR: {
        // wearing with BUC unknown is exactly what an altar drop would prevent
        int blind = env->inv_state[slot * NLE_INV_STATE_FIELDS] == 0;
        nethack_wear_takeoff_conflict(env, slot);
        if (nethack_item_use(env, 'W', "want to wear", NULL, slot,
                &st->verb_uses[verb], bad_pick) && blind) st->wear_blind++;
        break; }
    case NETHACK_ACT_EAT:
        nethack_item_use(env, 'e', "want to eat", "eat it", slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_QUAFF:
        nethack_item_use(env, 'q', "want to drink", "rink from the", slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_THROW: {
        // NH_THROWLOG: what gets thrown and with what result (quality forensic)
        static int tlog = -1;
        if (tlog < 0) tlog = getenv("NH_THROWLOG") != NULL;
        int t_otyp = env->inv_glyphs[slot] - NH_GLYPH_OBJ_OFF;
        int t_oc = env->inv_oclasses[slot];
        int ok = nethack_item_use(env, 't', "want to throw", NULL, slot, &st->verb_uses[verb], bad_pick);
        if (ok) nethack_answer_direction(env, dirkey);
        if (tlog)
            fprintf(stderr, "THROWLOG ok=%d otyp=%d oc=%d dl=%ld msg=%.48s\n",
                ok, t_otyp, t_oc, (long)env->blstats[NLE_BL_DEPTH],
                (const char*)env->message);
        break; }
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
    case NETHACK_ACT_ALTAR_ID: {
        // each drop onto an altar flashes the item's curse state (sets bknown),
        // so dump everything still unknown and take it straight back
        st->verb_uses[verb]++;
        for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i] && !env->obs.done; i++) {
            if (env->inv_oclasses[i] >= NETHACK_NUM_OCLASSES) break;
            if (env->inv_oclasses[i] == 12) continue;   // gold: no flash
            const signed char* st8 = &env->inv_state[i * NLE_INV_STATE_FIELDS];
            if (st8[0] != 0 || (st8[5] & 1)) continue;  // BUC known, or worn
            nethack_item_use(env, 'd', "drop", NULL, i, NULL, NULL);
        }
        if (!env->obs.done) {
            nethack_send_key(env, ',');
            nethack_answer_menu(env);
        }
        break; }
    case NETHACK_ACT_TIP:
        // M('T') = #tip; the "tip it? [ynq]" prompt auto-commits, spillage
        // lands underfoot for PICKUP. Locked -> "It's locked." (kick first).
        st->verb_uses[verb]++;
        nethack_send_key(env, 0x80 | 'T');
        break;
    case NETHACK_ACT_ENGRAVE_ID:
        st->verb_uses[verb]++;
        nethack_do_engrave_id(env);
        break;
    case NETHACK_ACT_CAST: {
        // Z, then 'a' = first known spell; directional spells then prompt and
        // take the (ZAP-masked) direction, self-spells resolve immediately
        st->verb_uses[verb]++;
        // NH_CASTLOG=1: per-cast forensic trace (demo/eval only; flag cached)
        static int castlog = -1;
        if (castlog < 0) castlog = getenv("NH_CASTLOG") != NULL;
        if (castlog)
            fprintf(stderr, "CASTLOG env=%p t=%ld slot=%d sid=%d fail=%d hp=%ld/%ld pw=%ld/%ld ac=%ld dl=%ld\n",
                (void*)env, (long)env->blstats[NLE_BL_TIME], slot,
                slot < env->n_spells ? env->spell_ids[slot] : -1,
                slot < env->n_spells ? (int)env->spell_fails[slot] : -1,
                (long)env->blstats[NLE_BL_HP], (long)env->blstats[NLE_BL_HP + 1],
                (long)env->blstats[NLE_BL_ENE], (long)env->blstats[NLE_BL_ENE + 1],
                (long)env->blstats[NLE_BL_AC], (long)env->blstats[NLE_BL_DEPTH]);
        { static int ctrace = -1;
          if (ctrace < 0) ctrace = getenv("NH_CASTTRACE") != NULL;
          nethack_send_key(env, 'Z');
          if (ctrace && slot > 0)
              fprintf(stderr, "CASTSTEP env=%p k=Z msg=%.80s\n", (void*)env, (const char*)env->message);
          if (!env->obs.done) {
              nethack_send_key(env, 'a' + (slot >= 0 && slot < NETHACK_SPELL_SLOTS ? slot : 0));
              if (ctrace && slot > 0)
                  fprintf(stderr, "CASTSTEP env=%p k=%c msg=%.80s\n", (void*)env,
                          'a' + slot, (const char*)env->message);
              nethack_answer_direction(env, dirkey);
              if (ctrace && slot > 0)
                  fprintf(stderr, "CASTSTEP env=%p k=dir msg=%.80s\n", (void*)env,
                          (const char*)env->message);
          } }
        if (castlog)
            fprintf(stderr, "CASTRES env=%p sid=%d msg=%.100s\n", (void*)env,
                slot < env->n_spells ? env->spell_ids[slot] : -1,
                (const char*)env->message);
        break; }
    case NETHACK_ACT_DROP: {
        // value of the discard stream = sellable stock the agent gives up for free
        int otyp = env->inv_glyphs[slot] - NH_GLYPH_OBJ_OFF;
        long quan = env->inv_state[slot * NLE_INV_STATE_FIELDS + 2];
        if (nethack_item_use(env, 'd', "drop", NULL, slot, &st->verb_uses[verb], bad_pick)
                && otyp >= 0 && otyp < NH_NUM_OBJECTS)
            st->drop_value += nh_obj_cost[otyp] * (quan > 0 ? quan : 1);
        break; }
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
    if (verb == NETHACK_ACT_CAST) slot = (int)env->agents[0].actions[19];
    int dh = nethack_dir_head(verb);
    int dirkey = NETHACK_DIR_KEYS[dh >= 0 ? (int)env->agents[0].actions[13 + dh] : 0];

    long time_before = env->blstats[NLE_BL_TIME];
    int bad_pick = 0;
    // NH_RUNCENSUS: outcome-determinism check for RUN aimed at an adjacent
    // hostile (the properly-scoped descendant of the reverted RUN mask)
    static int rcen = -1;
    if (rcen < 0) rcen = getenv("NH_RUNCENSUS") != NULL;
    // class 0: none-adjacent, 1: adjacent+aimed-away, 2: aimed-at-hostile
    int run_class = -1;
    if (rcen && verb == NETHACK_ACT_RUN) {
        int dd = (int)env->agents[0].actions[13 + 1];
        long tc = env->blstats[NLE_BL_X] + NETHACK_DIR_DX[dd];
        long tr = env->blstats[NLE_BL_Y] + NETHACK_DIR_DY[dd];
        int at = 0;
        if (tr >= 0 && tr < NH_ROWS && tc >= 0 && tc < NH_COLS) {
            int g = env->glyphs[tr * NH_COLS + tc];
            at = (g >= 0 && g < 381 && !nle_peaceful_at(env->ctx, (int)tc + 1, (int)tr));
        }
        run_class = at ? 2 : (env->stats.last_adj > 0 ? 1 : 0);
    }
    nethack_execute(env, verb, slot, dirkey, &bad_pick);

    env->prev_action = verb;
    int illegal = nethack_handle_prompts(env);
    if (!env->obs.done) nle_obs_refresh(env->ctx, &env->obs);
    nethack_auto_enhance(env);

    if (bad_pick) { illegal = 1; env->stats.illegal_actions++; }
    if (run_class >= 0) {   // post-refresh: clocks are now trustworthy
        static unsigned att[3], vd[3];   // racy: census only
        att[run_class]++;
        if (env->blstats[NLE_BL_TIME] == time_before) vd[run_class]++;
        static unsigned tick;
        if ((++tick % 20000) == 0)
            fprintf(stderr, "RUNCENSUS3 free=%u/%u away=%u/%u at=%u/%u\n",
                    vd[0], att[0], vd[1], att[1], vd[2], att[2]);
    }
    if (env->blstats[NLE_BL_TIME] > time_before) env->stats.valid_moves++;
    else { // NH_WASTELOG census: which verbs burn steps without game time
        env->stats.waste[verb]++;
        static int wlog = -1;
        if (wlog < 0) wlog = getenv("NH_WASTELOG") != NULL;
        static unsigned wn;   // benign race: sampling only
        if (wlog && (++wn % 97) == 0) {
            int dh2 = nethack_dir_head(verb);
            int tg = -1;
            if (dh2 >= 0) {
                int dd = (int)env->agents[0].actions[13 + dh2];
                long tc = env->blstats[NLE_BL_X] + NETHACK_DIR_DX[dd];
                long tr = env->blstats[NLE_BL_Y] + NETHACK_DIR_DY[dd];
                if (tr >= 0 && tr < NH_ROWS && tc >= 0 && tc < NH_COLS)
                    tg = env->glyphs[tr * NH_COLS + tc];
            }
            fprintf(stderr, "WMSG verb=%d tg=%d msg=%.48s\n", verb, tg,
                    (const char*)env->message);
        }
    }
    env->stats.length++;

    float reward = nethack_reward(env, illegal);
    env->agents[0].rewards[0] = reward;
    env->stats.ret += reward;

    int step_cap = env->max_episode_steps > 0.0f
        ? (int)env->max_episode_steps : NETHACK_MAX_EPISODE_STEPS;
    int done = env->obs.done || env->stats.length >= step_cap;
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
    if (env->n_spells > 0)
        printf("Sp: id%d L%d fail%d%%  x%d\n", env->spell_ids[0],
               env->spell_levs[0], env->spell_fails[0], env->n_spells);
    printf("Msg: %.*s\n", NLE_MESSAGE_SIZE, env->message);
    fflush(stdout);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    init(env);
    env->discovery_coef = dict_get(kwargs, "discovery_coef");
    env->gold_coef = dict_get(kwargs, "gold_coef");
    env->exp_coef = dict_get(kwargs, "exp_coef");
    env->descent_coef = dict_get(kwargs, "descent_coef");
    env->floor_coef = dict_get(kwargs, "floor_coef");
    env->scout_coef = dict_get(kwargs, "scout_coef");
    env->ac_coef = dict_get(kwargs, "ac_coef");
    env->heal_coef = dict_get(kwargs, "heal_coef");
    env->depth_alive = dict_get(kwargs, "depth_alive");
    env->descend_gate = dict_get(kwargs, "descend_gate");
    env->bank_floor = dict_get(kwargs, "bank_floor");
    env->bank_xp = dict_get(kwargs, "bank_xp");
    env->mask_cast = dict_get(kwargs, "mask_cast");
    env->max_episode_steps = dict_get(kwargs, "max_episode_steps");
    env->scout_ready = dict_get(kwargs, "scout_ready");
    env->descent_depth = dict_get(kwargs, "descent_depth");
    env->scout_depth = dict_get(kwargs, "scout_depth");
    env->exp_log = dict_get(kwargs, "exp_log");
    env->exp_sqrt = dict_get(kwargs, "exp_sqrt");
    env->exp_rel = dict_get(kwargs, "exp_rel");
    env->exp_rel_floor = dict_get(kwargs, "exp_rel_floor");
    env->exp_frontier = dict_get(kwargs, "exp_frontier");
    env->exp_site_rel = dict_get(kwargs, "exp_site_rel");
    env->ac_nospell = dict_get(kwargs, "ac_nospell");
    env->exp_depth = dict_get(kwargs, "exp_depth");
    env->xp_coef = dict_get(kwargs, "xp_coef");
    env->death_penalty = dict_get(kwargs, "death_penalty");
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
    dict_set(out, "sells", log->sells);
    dict_set(out, "buys", log->buys);
    dict_set(out, "discoveries", log->discoveries);
    dict_set(out, "sale_gold", log->sale_gold);
    dict_set(out, "drop_value", log->drop_value);
    dict_set(out, "trouble_frac", log->trouble_frac);
    dict_set(out, "prayers_fed", log->prayers_fed);
    dict_set(out, "altar_steps", log->altar_steps);
    dict_set(out, "wear_blind", log->wear_blind);
    dict_set(out, "cursed_worn_frac", log->cursed_worn_frac);
    dict_set(out, "death_combat", log->death_combat);
    dict_set(out, "death_weak", log->death_weak);
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
    dict_set(out, "reach_d10", log->reach_d10);
    dict_set(out, "descend_blocked", log->descend_blocked);
    dict_set(out, "scout_held", log->scout_held);
    { static const char* rn[9] = {"exp","gold","descent","floor","xp","scout","ac","heal","occ"};
      char key[32];
      for (int i = 0; i < 9; i++) {
          snprintf(key, sizeof(key), "r_raw/%s", rn[i]);
          dict_set(out, key, log->r_raw[i]);
          snprintf(key, sizeof(key), "r_clip/%s", rn[i]);
          dict_set(out, key, log->r_clip[i]);
      }
      dict_set(out, "r_raw/death", log->r_death); }
    dict_set(out, "truncated", log->truncated);
}

// Per-(verb,head) consumption map for PPO consumed-head gating (weak symbol
// read by src/algo.cu). heads: [0]=verb, [1..12]=slot heads 0..11,
// [13..18]=per-verb dir heads, [19]=spell-slot head (CAST). A head is
// "consumed" iff the sampled verb actually uses it.
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
            if (v == NETHACK_ACT_CAST) row[19] = 1;       // spell-slot head
        }
        built = 1;
    }
    *n_verbs = NETHACK_NUM_ACTIONS;
    *n_atns = NUM_ATNS;
    return map;
}
