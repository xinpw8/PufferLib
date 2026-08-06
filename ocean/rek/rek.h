// REK G1 combat env — two Unitree G1s in a timed hit-counting round.
//
// Rules, as they work in the shipped REK sim: the round is on a clock, the
// scoreboard counts clean hits, and most hits wins. Going down costs the
// falling robot a point, and 3 downs loses the match outright.
//
// Slot 0 / slot 1 are the two corners. Per-slot pointer arrays are required by
// the selfplay pool (MY_USES_PERM) so a frozen historical opponent can own
// slot 1 while the learner owns slot 0.

#pragma once

#include <stdlib.h>
#include <string.h>

#include "reklib.h"

typedef struct Client Client;
typedef struct Rek Rek;

struct Rek {
    Client* client;
    Log log;
    Log* logs;

    float* observations;
    float* actions;
    float* rewards;
    float* terminals;

    // Populated by my_setup_perm. With an identity perm these are just
    // base + slot*stride; under the selfplay pool they point wherever the
    // permutation put this slot.
    float* obs_ptr[2];
    float* action_ptr[2];
    float* reward_ptr[2];
    float* terminal_ptr[2];

    int num_agents;
    int num_bots;       // 0 or 1; a scripted opponent instead of a policy
    int bot_policy;

    Fighter fighters[2];
    int tick;
    int round_frames;   // round length in 30 Hz frames

    // Arena
    float arena_radius;
    float body_radius;

    // Movement / balance model. All are [env] kwargs so a sweep can tune them
    // and so extraction can pin them to REK's real values.
    float move_speed;
    float guard_speed_mult;
    float accel;
    float friction;
    float turn_rate;         // lock-on slew, rad/frame
    float balance_decay;     // balance shed per frame while standing
    float guard_balance_mult; // balance impact multiplier while guarding
    int   hitstun_frames;
    int   getup_frames;

    // Reward weights
    float reward_hit;
    float reward_hit_taken;
    float reward_down;
    float reward_down_dealt;
    float reward_win;
    float reward_guard;

    float dr;   // domain-randomisation scale, 0 disables

    // Per-episode randomised multipliers, resampled in c_reset when dr > 0.
    float dr_reach;
    float dr_speed;
    float dr_balance;
    int   dr_latency;        // frames of strike latency, 0..REK_MAX_LATENCY
    int   move_delay[2][REK_MAX_LATENCY];  // shift register feeding the latency

    // Selfplay-pool tagging. tag = 0 is pure selfplay; tag = 1..REK_MAX_BANKS
    // means slot 0 is the learner and slot 1 is a frozen bank opponent.
    int tag;
    int boundary_reached;
    // slot_for_corner[c] = which slot fights out of corner c. Randomised per
    // env so a matched pair is not scored from a fixed corner advantage.
    int slot_for_corner[2];

    uint32_t rng;
};

void c_reset(Rek* env);

static inline void rek_add_log(Rek* env) {
    for (int s = 0; s < env->num_agents; s++) {
        Fighter* f = &env->fighters[s];
        float denom = (env->tick > 0) ? (float)env->tick : 1.0f;
        env->logs[s].score        = (float)rek_score(f);
        env->logs[s].hits_landed  = (float)f->hits;
        env->logs[s].downs        = (float)f->downs;
        env->logs[s].guard_uptime = (float)f->guard_frames / denom;
        env->logs[s].whiff_rate   = (f->moves_started > 0)
            ? (float)f->moves_whiffed / (float)f->moves_started : 0.0f;

        env->log.perf           += env->logs[s].perf;
        env->log.score          += env->logs[s].score;
        env->log.episode_return += env->logs[s].episode_return;
        env->log.episode_length += env->logs[s].episode_length;
        env->log.hits_landed    += env->logs[s].hits_landed;
        env->log.hits_taken     += env->logs[s].hits_taken;
        env->log.downs          += env->logs[s].downs;
        env->log.knockouts      += env->logs[s].knockouts;
        env->log.guard_uptime   += env->logs[s].guard_uptime;
        env->log.whiff_rate     += env->logs[s].whiff_rate;
        env->log.n              += 1.0f;
    }
}

// outcome: +1 slot 0 won, -1 slot 0 lost, 0 draw. Mirrors robocode's
// end_episode so selfplay.py's pool accounting reads the same fields.
static inline void rek_end_episode(Rek* env, int outcome) {
    float s0 = (outcome > 0) ? 1.0f : (outcome < 0) ? 0.0f : 0.5f;
    env->log.slot_0_score += s0 * env->num_agents;
    env->log.slot_1_score += (1.0f - s0) * env->num_agents;
    if (outcome == 0) env->log.draw_rate += env->num_agents;

    if (env->tag > 0 && env->tag <= REK_MAX_BANKS) {
        int bank = env->tag - 1;
        env->log.hist_score_bank[bank] += s0;
        env->log.hist_n_bank[bank]     += 1.0f;
        env->log.hist_score            += s0;
        env->log.hist_n                += 1.0f;
        env->boundary_reached = 1;
    }

    // Terminal reward is the round result, which is what the scoreboard
    // actually pays out; shaped hit rewards only steer toward it.
    if (env->num_agents > 1) {
        *env->reward_ptr[0] += env->reward_win * (2.0f * s0 - 1.0f);
        *env->reward_ptr[1] += env->reward_win * (1.0f - 2.0f * s0);
        env->logs[0].episode_return += env->reward_win * (2.0f * s0 - 1.0f);
        env->logs[1].episode_return += env->reward_win * (1.0f - 2.0f * s0);
    }

    for (int s = 0; s < env->num_agents; s++) {
        // perf is the headline number on the dashboard: win = 1, draw = 0.5.
        env->logs[s].perf = (s == 0) ? s0 : 1.0f - s0;
        *env->terminal_ptr[s] = 1.0f;
    }

    rek_add_log(env);
    c_reset(env);
}

static inline void rek_reset_fighter(Fighter* f, float x, float z, float yaw) {
    memset(f, 0, sizeof(Fighter));
    f->x = x;
    f->z = z;
    f->yaw = yaw;
}

void c_reset(Rek* env) {
    env->tick = 0;
    // boundary_reached is owned by selfplay.py alignment; do not clear it here.

    // Corner assignment is randomised so neither policy learns a corner bias.
    int flip = (int)(rek_rand(&env->rng) & 1u);
    env->slot_for_corner[0] = flip;
    env->slot_for_corner[1] = 1 - flip;

    float half = env->arena_radius * 0.5f;
    float jitter = 0.25f * env->dr;
    rek_reset_fighter(&env->fighters[env->slot_for_corner[0]],
        -half + rek_uniform(&env->rng, -jitter, jitter), 0.0f, 0.0f);
    rek_reset_fighter(&env->fighters[env->slot_for_corner[1]],
        half + rek_uniform(&env->rng, -jitter, jitter), 0.0f, (float)M_PI);

    memset(env->move_delay, 0, sizeof(env->move_delay));

    // Domain randomisation. REK's exact reach, walk speed and balance
    // thresholds are unknown until tools/extract_rek.py runs, so training wide
    // here is what buys sim-to-sim robustness rather than a lucky guess.
    if (env->dr > 0.0f) {
        float d = env->dr;
        env->dr_reach   = rek_uniform(&env->rng, 1.0f - 0.15f * d, 1.0f + 0.15f * d);
        env->dr_speed   = rek_uniform(&env->rng, 1.0f - 0.20f * d, 1.0f + 0.20f * d);
        env->dr_balance = rek_uniform(&env->rng, 1.0f - 0.25f * d, 1.0f + 0.25f * d);
        env->dr_latency = (rek_randf(&env->rng) < 0.5f * d)
            ? 1 + (int)(rek_rand(&env->rng) % (uint32_t)REK_MAX_LATENCY) : 0;
    } else {
        env->dr_reach = 1.0f;
        env->dr_speed = 1.0f;
        env->dr_balance = 1.0f;
        env->dr_latency = 0;
    }

    // Rewards and terminals are owned by c_step, which zeroes them at the top
    // of every step. Clearing them here would wipe the terminal payout that
    // rek_end_episode just wrote before calling us.
    for (int s = 0; s < env->num_agents; s++) {
        memset(&env->logs[s], 0, sizeof(Log));
    }
}

// Scripted opponent for num_bots = 1. Not meant to be strong — it exists so the
// env is playable and debuggable before a policy exists, and as a sanity
// baseline that a learned policy must beat.
static inline void rek_bot_act(Rek* env, int slot, int* out_dir, int* out_move, int* out_guard) {
    Fighter* me = &env->fighters[slot];
    Fighter* opp = &env->fighters[1 - slot];
    float dx = opp->x - me->x;
    float dz = opp->z - me->z;
    float dist = sqrtf(dx * dx + dz * dz);

    float strike_range = 0.85f * env->dr_reach + env->body_radius;
    if (dist > strike_range) {
        *out_dir = 1;  // close the gap
        *out_move = 0;
        *out_guard = 0;
        return;
    }

    *out_dir = 0;
    // Guard when the opponent is committed to something, otherwise poke.
    if (rek_committed(opp) && rek_randf(&env->rng) < 0.6f) {
        *out_move = 0;
        *out_guard = 1;
        return;
    }
    *out_guard = 0;
    *out_move = (rek_randf(&env->rng) < 0.5f) ? 1 : 1 + (int)(rek_rand(&env->rng) % (uint32_t)(NUM_MOVE_DEFS - 1));
}

static inline void rek_apply_locomotion(Rek* env, Fighter* f, int dir_action) {
    float fwd, side;
    rek_move_dir(dir_action, &fwd, &side);

    float speed = env->move_speed * env->dr_speed;
    if (f->guard) speed *= env->guard_speed_mult;

    // Ego-relative: forward is along the facing the lock-on has slewed to.
    float cy = cosf(f->yaw);
    float sy = sinf(f->yaw);
    float tx = (fwd * cy - side * sy) * speed;
    float tz = (fwd * sy + side * cy) * speed;

    f->vx += (tx - f->vx) * env->accel;
    f->vz += (tz - f->vz) * env->accel;
}

static inline void rek_integrate(Rek* env, Fighter* f, float root_motion) {
    if (root_motion != 0.0f) {
        f->vx += cosf(f->yaw) * root_motion;
        f->vz += sinf(f->yaw) * root_motion;
    }

    f->x += f->vx * REK_DT;
    f->z += f->vz * REK_DT;
    f->vx *= env->friction;
    f->vz *= env->friction;

    // Keep both robots inside the ring. REK has no ring-out, so this clamps
    // rather than terminating.
    float d = sqrtf(f->x * f->x + f->z * f->z);
    float limit = env->arena_radius - env->body_radius;
    if (d > limit && d > 1e-6f) {
        float k = limit / d;
        f->x *= k;
        f->z *= k;
        f->vx *= 0.5f;
        f->vz *= 0.5f;
    }
}

// Push the two robots apart so they cannot occupy the same space.
static inline void rek_resolve_overlap(Rek* env) {
    Fighter* a = &env->fighters[0];
    Fighter* b = &env->fighters[1];
    float dx = b->x - a->x;
    float dz = b->z - a->z;
    float d2 = dx * dx + dz * dz;
    float min_d = 2.0f * env->body_radius;
    if (d2 >= min_d * min_d || d2 < 1e-8f) return;

    float d = sqrtf(d2);
    float push = 0.5f * (min_d - d) / d;
    a->x -= dx * push;
    a->z -= dz * push;
    b->x += dx * push;
    b->z += dz * push;
}

static inline void rek_go_down(Rek* env, int slot) {
    Fighter* f = &env->fighters[slot];
    f->downs += 1;
    f->balance = 0.0f;
    f->move = 0;
    f->frame = 0;
    f->stun = 0;
    f->down_timer = env->getup_frames;
    f->vx = 0.0f;
    f->vz = 0.0f;

    if (slot < env->num_agents) {
        *env->reward_ptr[slot] += env->reward_down;
        env->logs[slot].episode_return += env->reward_down;
    }
    int other = 1 - slot;
    if (other < env->num_agents) {
        *env->reward_ptr[other] += env->reward_down_dealt;
        env->logs[other].episode_return += env->reward_down_dealt;
    }
}

// Resolve the active frame of `slot`'s move against the opponent.
static inline void rek_resolve_hit(Rek* env, int slot) {
    Fighter* att = &env->fighters[slot];
    Fighter* def = &env->fighters[1 - slot];
    const MoveDef* m = &REK_MOVE_TABLE[att->move];

    float reach = m->reach * env->dr_reach;
    float hx = att->x + cosf(att->yaw) * reach;
    float hz = att->z + sinf(att->yaw) * reach;
    float dx = def->x - hx;
    float dz = def->z - hz;
    float r = m->radius + env->body_radius;
    if (dx * dx + dz * dz > r * r) return;

    att->move_connected = 1;

    bool guarded = def->guard && !m->guard_breaks && def->down_timer == 0;
    float impact = m->balance_impact * env->dr_balance;
    if (guarded) impact *= env->guard_balance_mult;

    // A guarded hit does not reach the scoreboard, but it still moves the
    // defender's balance — chip pressure is how guard gets broken.
    if (!guarded && m->damage > 0.0f) {
        att->hits += 1;
        def->stun = env->hitstun_frames;
        if (slot < env->num_agents) {
            *env->reward_ptr[slot] += env->reward_hit;
            env->logs[slot].episode_return += env->reward_hit;
        }
        int d = 1 - slot;
        if (d < env->num_agents) {
            *env->reward_ptr[d] += env->reward_hit_taken;
            env->logs[d].episode_return += env->reward_hit_taken;
            env->logs[d].hits_taken += 1.0f;
        }
    }

    def->balance += impact;
    // Knockback along the strike direction.
    float kb = impact * 2.0f;
    def->vx += cosf(att->yaw) * kb;
    def->vz += sinf(att->yaw) * kb;
}

static inline void rek_step_fighter(Rek* env, int slot, int dir_action, int move_action, int guard_action) {
    Fighter* f = &env->fighters[slot];

    if (f->down_timer > 0) {
        f->down_timer -= 1;
        f->guard = 0;
        rek_integrate(env, f, 0.0f);
        return;
    }

    f->guard = (guard_action != 0 && !rek_committed(f)) ? 1 : 0;
    if (f->guard) f->guard_frames += 1;

    if (f->stun > 0) {
        f->stun -= 1;
        rek_integrate(env, f, 0.0f);
        return;
    }

    // Lock-on: facing slews toward the opponent at a bounded rate, which is
    // what a WASD-only control scheme implies — the pilot never aims directly.
    Fighter* opp = &env->fighters[1 - slot];
    float want = atan2f(opp->z - f->z, opp->x - f->x);
    float dyaw = rek_angle_delta(f->yaw, want);
    float max_turn = env->turn_rate;
    if (dyaw > max_turn) dyaw = max_turn;
    if (dyaw < -max_turn) dyaw = -max_turn;
    f->yaw += dyaw;

    float root_motion = 0.0f;

    if (rek_committed(f)) {
        const MoveDef* m = &REK_MOVE_TABLE[f->move];
        int total = rek_move_total(f->move);
        // Root motion is spread across startup + active, not the recovery tail.
        int drive = m->startup + m->active;
        if (drive > 0 && f->frame < drive) root_motion = m->root_motion / (float)drive;

        if (f->frame >= m->startup && f->frame < m->startup + m->active && !f->move_connected) {
            rek_resolve_hit(env, slot);
        }

        f->frame += 1;
        if (f->frame >= total) {
            if (!f->move_connected) f->moves_whiffed += 1;
            f->move = 0;
            f->frame = 0;
            f->move_connected = 0;
        }
    } else {
        rek_apply_locomotion(env, f, dir_action);
        if (move_action > 0 && move_action < NUM_MOVE_DEFS) {
            f->move = move_action;
            f->frame = 0;
            f->move_connected = 0;
            f->moves_started += 1;
            f->balance += REK_MOVE_TABLE[move_action].balance_cost * env->dr_balance;
            f->guard = 0;
        }
    }

    rek_integrate(env, f, root_motion);

    if (env->reward_guard != 0.0f && f->guard && slot < env->num_agents) {
        *env->reward_ptr[slot] += env->reward_guard;
        env->logs[slot].episode_return += env->reward_guard;
    }
}

static inline void rek_write_fighter_obs(const Rek* env, float* obs, const Fighter* f) {
    float inv_r = 1.0f / env->arena_radius;
    obs[0]  = f->x * inv_r;
    obs[1]  = f->z * inv_r;
    obs[2]  = f->vx * 0.25f;
    obs[3]  = f->vz * 0.25f;
    obs[4]  = sinf(f->yaw);
    obs[5]  = cosf(f->yaw);
    obs[6]  = f->balance;
    obs[7]  = (float)f->guard;
    obs[8]  = (float)f->stun / (float)(env->hitstun_frames + 1);
    obs[9]  = (float)f->down_timer / (float)(env->getup_frames + 1);
    obs[10] = rek_committed(f) ? (float)f->frame / (float)rek_move_total(f->move) : 0.0f;
    obs[11] = (float)f->hits * 0.1f;
    obs[12] = (float)f->downs / (float)REK_DOWNS_TO_LOSE;
    obs[13] = (float)rek_score(f) * 0.1f;

    // Which move is out, one-hot. The animation is visible to a pilot, so
    // hiding it from the policy would model less information than REK gives.
    float* onehot = obs + REK_SCALARS_PER_FIGHTER;
    memset(onehot, 0, NUM_MOVE_DEFS * sizeof(float));
    onehot[f->move] = 1.0f;
}

static inline void rek_compute_obs(Rek* env, int slot) {
    float* obs = env->obs_ptr[slot];
    const Fighter* me = &env->fighters[slot];
    const Fighter* opp = &env->fighters[1 - slot];

    rek_write_fighter_obs(env, obs, me);
    rek_write_fighter_obs(env, obs + REK_FIGHTER_FEATURES, opp);

    float* rel = obs + 2 * REK_FIGHTER_FEATURES;
    float dx = opp->x - me->x;
    float dz = opp->z - me->z;
    float dist = sqrtf(dx * dx + dz * dz);
    float inv_r = 1.0f / env->arena_radius;
    rel[0] = dx * inv_r;
    rel[1] = dz * inv_r;
    rel[2] = dist * inv_r;
    float bearing = rek_angle_delta(me->yaw, atan2f(dz, dx));
    rel[3] = sinf(bearing);
    rel[4] = cosf(bearing);
    // How square the opponent is to us: +1 means they are facing us head on.
    rel[5] = cosf(rek_angle_delta(opp->yaw, atan2f(-dz, -dx)));
    // Closing speed, positive when the gap is shrinking.
    rel[6] = (dist > 1e-6f)
        ? -((opp->vx - me->vx) * dx + (opp->vz - me->vz) * dz) / (dist * 4.0f) : 0.0f;

    float* clock = rel + REK_RELATIVE_FEATURES;
    clock[0] = 1.0f - (float)env->tick / (float)env->round_frames;
}

void c_step(Rek* env) {
    for (int s = 0; s < env->num_agents; s++) {
        *env->reward_ptr[s] = 0.0f;
        *env->terminal_ptr[s] = 0.0f;
    }

    env->tick += 1;
    for (int s = 0; s < env->num_agents; s++) env->logs[s].episode_length += 1.0f;

    // Passive balance recovery for the frame that just passed, applied to both
    // fighters before either acts. Doing it here rather than inside
    // rek_step_fighter keeps the slots symmetric — otherwise slot 1 would shed
    // part of a hit slot 0 landed earlier in the same step — and leaves this
    // frame's impacts at full value for the knockdown check below. Guard state
    // is last frame's, which is what the recovery was earned under.
    for (int s = 0; s < 2; s++) {
        Fighter* f = &env->fighters[s];
        if (f->down_timer > 0) continue;
        f->balance -= env->balance_decay * (f->guard ? 1.5f : 1.0f);
        if (f->balance < 0.0f) f->balance = 0.0f;
    }

    int dir[2] = {0, 0};
    int move[2] = {0, 0};
    int guard[2] = {0, 0};

    for (int s = 0; s < 2; s++) {
        if (s < env->num_agents) {
            const float* a = env->action_ptr[s];
            dir[s] = (int)a[0];
            move[s] = (int)a[1];
            guard[s] = (int)a[2];
        } else {
            rek_bot_act(env, s, &dir[s], &move[s], &guard[s]);
        }

        // Strike latency, in frames. There is a render loop and a network hop
        // between a REK pilot's key press and the robot swinging; training
        // through a randomised delay keeps a policy from timing itself to our
        // exact frame budget. Applied to the strike only — locomotion latency
        // is already absorbed by the velocity filter in rek_apply_locomotion.
        int lat = env->dr_latency;
        if (lat > 0) {
            int emitted = env->move_delay[s][0];
            for (int i = 0; i < lat - 1; i++) {
                env->move_delay[s][i] = env->move_delay[s][i + 1];
            }
            env->move_delay[s][lat - 1] = move[s];
            move[s] = emitted;
        }
    }

    for (int s = 0; s < 2; s++) {
        rek_step_fighter(env, s, dir[s], move[s], guard[s]);
    }

    rek_resolve_overlap(env);

    // Balance past the threshold puts you on the floor. Checked after both
    // fighters have stepped so a double knockdown resolves as one.
    for (int s = 0; s < 2; s++) {
        if (env->fighters[s].balance >= 1.0f && env->fighters[s].down_timer == 0) {
            rek_go_down(env, s);
        }
    }

    // 3 downs loses the match outright, ahead of the clock.
    for (int s = 0; s < 2; s++) {
        if (env->fighters[s].downs >= REK_DOWNS_TO_LOSE) {
            if (s < env->num_agents) env->logs[s].knockouts += 1.0f;
            rek_end_episode(env, (s == 0) ? -1 : +1);
            for (int i = 0; i < env->num_agents; i++) rek_compute_obs(env, i);
            return;
        }
    }

    if (env->tick >= env->round_frames) {
        int s0 = rek_score(&env->fighters[0]);
        int s1 = rek_score(&env->fighters[1]);
        rek_end_episode(env, (s0 > s1) ? +1 : (s0 < s1) ? -1 : 0);
        for (int i = 0; i < env->num_agents; i++) rek_compute_obs(env, i);
        return;
    }

    for (int s = 0; s < env->num_agents; s++) rek_compute_obs(env, s);
}

void init(Rek* env) {
    env->logs = (Log*)calloc(env->num_agents > 0 ? env->num_agents : 1, sizeof(Log));
    // vecenv seeds envs with their index (0, 1, 2, ...). xorshift32 walks
    // small seeds through very similar early states, so scramble first —
    // otherwise the first few thousand envs open near-identical rounds.
    uint32_t s = env->rng * 0x9e3779b9u + 0x85ebca6bu;
    s ^= s >> 16;
    s *= 0xc2b2ae35u;
    s ^= s >> 15;
    env->rng = s ? s : 0x9e3779b9u;
    env->dr_reach = 1.0f;
    env->dr_speed = 1.0f;
    env->dr_balance = 1.0f;
}

// Standalone (non-vecenv) path used by rek.c. Under vecenv these buffers are
// owned by the StaticVec and the per-slot pointers come from my_setup_perm.
void allocate_env(Rek* env) {
    init(env);
    env->observations = (float*)calloc(REK_OBS_SIZE * env->num_agents, sizeof(float));
    env->actions = (float*)calloc(REK_NUM_ATNS * env->num_agents, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));
    for (int s = 0; s < env->num_agents; s++) {
        env->obs_ptr[s]      = env->observations + s * REK_OBS_SIZE;
        env->action_ptr[s]   = env->actions + s * REK_NUM_ATNS;
        env->reward_ptr[s]   = env->rewards + s;
        env->terminal_ptr[s] = env->terminals + s;
    }
}

void free_allocated(Rek* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void c_close(Rek* env) {
    free(env->logs);
    env->logs = NULL;
}
