/*
 * will_connect.h - "will connect" light for U / I bound attacks.
 *
 * For the clean-room REK-parity env (your MuJoCo/PufferLib clone). Not a hook
 * or overlay for REK's own client.
 *
 * Header-only C99, depends only on <math.h>. No allocation. Safe to call per
 * env per step: a broad phase rejects clear misses; otherwise the sweep is
 * 2 exact segments when not turning, ~10-25 steps when turning.
 *
 * Question answered: "if the pilot pressed U (or I) right now, and both
 * fighters kept their current velocities, would the glove land a clean hit?"
 *
 * Model
 *   - Attacker and defender roots move at constant velocity over the window.
 *   - Attacker turns at constant yaw_rate. Shoulder and aim turn with it.
 *   - Defender hurtboxes (head sphere, vertical torso capsule) don't depend on
 *     rotation, so defender yaw is ignored.
 *   - Fist center:
 *       F(t) = S(t) + e(t) * reach * aim(t)
 *     e goes linearly 0 -> 1 over `extend`, then holds at 1 for `hold`.
 *     t is measured from the key press.
 *   - Each time step is a swept test in the target's own frame, so a fast
 *     fist can't tunnel through between samples. The first-contact time is
 *     then refined by bisection, and fist speed at contact is analytic.
 *     Turning sweeps approximate the curved path with chords. Grazing contacts
 *     can be missed; an actual overlap is required before reporting contact.
 *   - Contacts count once the glove is WC_MIN_EXT extended; already touching
 *     at that point = smothered -> WEAK.
 *   - Optional guard gloves: if the fist meets one first -> BLOCKED.
 *   - Impact speed (relative velocity along the contact normal) below
 *     min_speed -> WEAK. Glancing blows land here.
 *   - auto_aim: yaw-only convergence of the punch onto the defender's
 *     centerline at launch, clamped to +-auto_aim.
 *
 * Lights
 *   GREEN  clean hit (WC_HIT)
 *   AMBER  weak hit (WC_WEAK), or a miss by less than cfg.amber_band
 *   RED    miss or blocked
 *   OFF    key is not U or I, or input is invalid
 *
 * Frames
 *   Default: right-handed, z-up (MuJoCo). For Unity-style logs, set
 *   cfg.up = (0,1,0) and cfg.left_handed = 1. "right" is cross(fwd, up) for a
 *   right-handed world and cross(up, fwd) for a left-handed one.
 *   yaw_rate follows the world's own rotation convention (Rodrigues about up).
 *
 * All numbers in WC_ATK_U / WC_ATK_I / WC_HURTBOX_G1 are PLACEHOLDERS. Fit
 * them from your parity measurements (reach, timings, heights).
 * Key mapping is also an assumption: U = lead (left) jab, I = rear (right) cross.
 *
 * Usage (env step or render):
 *   wc_result ru = wc_eval_key('U', &me, &opp, &WC_HURTBOX_G1, opp_gloves, 2, &WC_CFG_DEFAULT);
 *   wc_result ri = wc_eval_key('I', &me, &opp, &WC_HURTBOX_G1, opp_gloves, 2, &WC_CFG_DEFAULT);
 *   // ru.light / ri.light -> HUD (see will_connect_hud.h), wc_obs() -> RL observation
 */
#ifndef WILL_CONNECT_H
#define WILL_CONNECT_H

#include <math.h>

/* ------------------------------------------------------------------ types */

typedef struct { float x, y, z; } wc_v3;

typedef struct {
    wc_v3 pos;      /* root (pelvis) position, m */
    wc_v3 vel;      /* root linear velocity, m/s */
    wc_v3 fwd;      /* facing; projected onto the horizontal plane internally */
    float yaw_rate; /* rad/s about up (used for the attacker only) */
} wc_body;

typedef struct {
    wc_v3 pos, vel; /* guard glove center (world) and velocity */
    float radius;
} wc_guard;

typedef struct {
    float head_up, head_r;             /* head sphere: center height above root, radius */
    float torso_lo, torso_hi, torso_r; /* torso capsule: segment heights above root, radius */
} wc_hurtbox;

enum { WC_TGT_HEAD = 1, WC_TGT_TORSO = 2 };

typedef struct {
    char  key;        /* 'U' or 'I' */
    float startup;    /* s: key press -> fist starts moving (include measured input latency) */
    float extend;     /* s: fist travels 0 -> full reach */
    float hold;       /* s: time at full extension that can still land */
    float reach;      /* m: shoulder -> fist center at full extension */
    float fist_r;     /* m: glove radius */
    float sh_fwd, sh_right, sh_up; /* m: shoulder offset from root in body frame */
    float aim_pitch;  /* rad: + = upward */
    float auto_aim;   /* rad: max yaw correction at launch toward the defender's centerline (0 = off) */
    float min_speed;  /* m/s: min impact speed (closing speed along the contact normal) for a clean hit */
    int   targets;    /* WC_TGT_* mask */
} wc_attack;

typedef struct {
    wc_v3 up;         /* world up (unit) */
    int   left_handed;/* 0 = right-handed (MuJoCo), 1 = left-handed (Unity) */
    float max_dt;     /* s: max sweep step while the attacker is turning (exact otherwise) */
    float amber_band; /* m: a miss closer than this shows AMBER */
} wc_cfg;

#ifndef WC_FAR
#define WC_FAR 0.5f       /* m: clearance beyond which the broad phase skips the sweep */
#endif
#ifndef WC_MIN_EXT
#define WC_MIN_EXT 0.25f  /* contacts count once the fist is this fraction extended;
                             touching already at that point = smothered (WEAK) */
#endif
#ifndef WC_MAX_DYAW
#define WC_MAX_DYAW 0.02f /* rad: max attacker turn per sweep step (chord error ~r*th^2/8) */
#endif
#ifndef WC_MAX_GUARDS
#define WC_MAX_GUARDS 4   /* guard gloves beyond this are ignored */
#endif
#ifndef WC_MAX_SWEEP_STEPS
#define WC_MAX_SWEEP_STEPS 256 /* total per evaluation; larger workloads return INVALID */
#endif

typedef enum { WC_OFF = 0, WC_RED, WC_AMBER, WC_GREEN } wc_light;
typedef enum { WC_INVALID = 0, WC_MISS, WC_HIT, WC_WEAK, WC_BLOCKED } wc_outcome;

typedef struct {
    wc_outcome outcome;
    wc_light   light;
    float margin;     /* m: min signed clearance over the active window. Chord approximation
                         while turning: a negative margin need not be an actual contact.
                         Exact for straight sweeps; broad rejection returns a lower bound. */
    float t_contact;  /* s after key press of first target contact, -1 if none */
    float rel_speed;  /* m/s: impact speed at contact (relative velocity on the contact normal) */
    int   target;     /* WC_TGT_* that was hit first, 0 if none */
    wc_v3 p_contact;  /* world fist center at contact (valid if t_contact >= 0) */
} wc_result;

/* ---------------------------------------------------- defaults (PLACEHOLDER) */

/* G1-scale guesses, root = pelvis. Replace with fitted values. */
static const wc_hurtbox WC_HURTBOX_G1 = { 0.47f, 0.10f, 0.05f, 0.32f, 0.12f };

static const wc_attack WC_ATK_U = {
    'U', 0.05f, 0.16f, 0.04f, 0.42f, 0.06f,
    0.02f, -0.16f, 0.30f,           /* lead (left) shoulder */
    0.00f, 0.50f, 1.5f, WC_TGT_HEAD | WC_TGT_TORSO
};
static const wc_attack WC_ATK_I = {
    'I', 0.08f, 0.22f, 0.05f, 0.46f, 0.06f,
    0.02f, +0.16f, 0.30f,           /* rear (right) shoulder */
    0.00f, 0.50f, 1.5f, WC_TGT_HEAD | WC_TGT_TORSO
};

static const wc_cfg WC_CFG_DEFAULT = { { 0.0f, 0.0f, 1.0f }, 0, 1.0f / 120.0f, 0.05f };

/* ------------------------------------------------------------------ math */

static inline wc_v3 wc_v(float x, float y, float z) { wc_v3 r; r.x = x; r.y = y; r.z = z; return r; }
static inline int wc__finite_v(wc_v3 v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
static inline int wc__finite_body(const wc_body *b) {
    return b && wc__finite_v(b->pos) && wc__finite_v(b->vel) &&
           wc__finite_v(b->fwd) && isfinite(b->yaw_rate);
}
static inline wc_v3 wc__add(wc_v3 a, wc_v3 b) { return wc_v(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline wc_v3 wc__sub(wc_v3 a, wc_v3 b) { return wc_v(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline wc_v3 wc__mul(wc_v3 a, float s) { return wc_v(a.x * s, a.y * s, a.z * s); }
static inline float wc__dot(wc_v3 a, wc_v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline wc_v3 wc__cross(wc_v3 a, wc_v3 b) {
    return wc_v(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float wc__len(wc_v3 a) { return sqrtf(wc__dot(a, a)); }
static inline float wc__clamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline wc_v3 wc__norm(wc_v3 a) {
    float l = wc__len(a);
    return l > 1e-9f ? wc__mul(a, 1.0f / l) : wc_v(0.0f, 0.0f, 0.0f);
}
/* Rodrigues: rotate v about unit axis k by angle th. */
static inline wc_v3 wc__rot(wc_v3 v, wc_v3 k, float th) {
    float c, s;
    if (th == 0.0f) return v;                 /* common case: not turning */
    c = cosf(th); s = sinf(th);
    return wc__add(wc__add(wc__mul(v, c), wc__mul(wc__cross(k, v), s)),
                   wc__mul(k, wc__dot(k, v) * (1.0f - c)));
}

/* Segment-segment distance (Ericson, RTCD 5.1.9); *s_out = param on [p1,q1]. */
static inline float wc__seg_seg(wc_v3 p1, wc_v3 q1, wc_v3 p2, wc_v3 q2, float *s_out) {
    wc_v3 d1 = wc__sub(q1, p1), d2 = wc__sub(q2, p2), r = wc__sub(p1, p2);
    float a = wc__dot(d1, d1), e = wc__dot(d2, d2), f = wc__dot(d2, r);
    float s, t;
    const float EPS = 1e-12f;
    if (a <= EPS && e <= EPS) {
        s = t = 0.0f;
    } else if (a <= EPS) {
        s = 0.0f;
        t = wc__clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = wc__dot(d1, r);
        if (e <= EPS) {
            t = 0.0f;
            s = wc__clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = wc__dot(d1, d2), denom = a * e - b * b;
            s = denom > EPS ? wc__clamp((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f)      { t = 0.0f; s = wc__clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = wc__clamp((b - c) / a, 0.0f, 1.0f); }
        }
    }
    *s_out = s;
    return wc__len(wc__sub(wc__add(p1, wc__mul(d1, s)), wc__add(p2, wc__mul(d2, t))));
}

/* ------------------------------------------------------------- kinematics */

typedef struct {
    wc_v3 up, fwd0;    /* world up, attacker horizontal facing at t = 0 */
    wc_v3 aim0;        /* aim direction at launch (after auto-aim) */
    float t0;          /* launch time (startup, clamped >= 0) */
    int   lh;
    const wc_attack *k;
    const wc_body   *a;
} wc__ctx;

/* A capsule [a,b] + radius r (a == b -> sphere) riding on an owner that
 * translates: world segment at time t is [p0 + v t + a, p0 + v t + b]. */
typedef struct { wc_v3 p0, v, a, b; float r; } wc__shape;

static inline wc_v3 wc__right(wc_v3 fwd, wc_v3 up, int lh) {
    return lh ? wc__cross(up, fwd) : wc__cross(fwd, up);
}

static inline wc_v3 wc__shoulder(const wc__ctx *c, float t) {
    wc_v3 f = wc__rot(c->fwd0, c->up, c->a->yaw_rate * t);
    wc_v3 r = wc__right(f, c->up, c->lh);
    wc_v3 p = wc__add(c->a->pos, wc__mul(c->a->vel, t));
    p = wc__add(p, wc__mul(f, c->k->sh_fwd));
    p = wc__add(p, wc__mul(r, c->k->sh_right));
    return wc__add(p, wc__mul(c->up, c->k->sh_up));
}

static inline float wc__ext(const wc__ctx *c, float tau) {
    return c->k->extend > 0.0f ? wc__clamp(tau / c->k->extend, 0.0f, 1.0f) : 1.0f;
}

/* Fist center at time t (t >= launch). */
static inline wc_v3 wc__fist(const wc__ctx *c, float t) {
    float tau = t - c->t0;
    wc_v3 aim = wc__rot(c->aim0, c->up, c->a->yaw_rate * tau);
    return wc__add(wc__shoulder(c, t), wc__mul(aim, wc__ext(c, tau) * c->k->reach));
}

/* Exact fist velocity at time t (d/dt of wc__fist; left derivative at the
 * end-of-extension kink). Rotation about up: d/dt rot(v) = w * (up x rot(v)). */
static inline wc_v3 wc__fist_vel(const wc__ctx *c, float t) {
    float tau = t - c->t0, w = c->a->yaw_rate;
    wc_v3 f = wc__rot(c->fwd0, c->up, w * t);
    wc_v3 r = wc__right(f, c->up, c->lh);
    wc_v3 arm = wc__add(wc__mul(f, c->k->sh_fwd), wc__mul(r, c->k->sh_right));
    wc_v3 aim = wc__rot(c->aim0, c->up, w * tau);
    float de = (c->k->extend > 0.0f && tau >= 0.0f && tau <= c->k->extend) ? 1.0f / c->k->extend : 0.0f;
    wc_v3 v = wc__add(c->a->vel, wc__mul(wc__cross(c->up, arm), w));
    v = wc__add(v, wc__mul(aim, de * c->k->reach));
    return wc__add(v, wc__mul(wc__cross(c->up, aim), w * wc__ext(c, tau) * c->k->reach));
}

/* Signed surface gap fist <-> shape at time t (<= 0 means touching). */
static inline float wc__gap(const wc__ctx *c, const wc__shape *sh, float t) {
    float s;
    wc_v3 rel = wc__sub(wc__fist(c, t), wc__add(sh->p0, wc__mul(sh->v, t)));
    return wc__seg_seg(rel, rel, sh->a, sh->b, &s) - sh->r - c->k->fist_r;
}

/* Impact speed at time t: fist velocity relative to the shape, projected on
 * the contact normal (fist center -> closest point on the shape's axis). */
static inline float wc__closing(const wc__ctx *c, const wc__shape *sh, float t) {
    wc_v3 rel = wc__sub(wc__fist(c, t), wc__add(sh->p0, wc__mul(sh->v, t)));
    wc_v3 ab = wc__sub(sh->b, sh->a), q, n;
    float L = wc__dot(ab, ab);
    float u = L > 1e-12f ? wc__clamp(wc__dot(wc__sub(rel, sh->a), ab) / L, 0.0f, 1.0f) : 0.0f;
    wc_v3 vrel = wc__sub(wc__fist_vel(c, t), sh->v);
    q = wc__add(sh->a, wc__mul(ab, u));
    n = wc__sub(q, rel);
    if (wc__len(n) < 1e-6f) return wc__len(vrel);   /* fist center on the axis */
    return wc__dot(vrel, wc__norm(n));
}

/* Swept test over [ta, tb] in the shape's frame. Returns min gap on the swept
 * segment; if it touches and refine != 0, *t_hit = first-contact time (bisection). */
static inline float wc__sweep(const wc__ctx *c, const wc__shape *sh, float ta, float tb,
                              wc_v3 Fa, wc_v3 Fb, int refine, float *t_hit) {
    float s, lo, hi, it;
    wc_v3 A = wc__sub(Fa, wc__add(sh->p0, wc__mul(sh->v, ta)));
    wc_v3 B = wc__sub(Fb, wc__add(sh->p0, wc__mul(sh->v, tb)));
    float g = wc__seg_seg(A, B, sh->a, sh->b, &s) - sh->r - c->k->fist_r;
    *t_hit = -1.0f;
    if (g > 0.0f || !refine) return g;
    if (wc__gap(c, sh, ta) <= 0.0f) { *t_hit = ta; return g; }
    lo = ta; hi = ta + s * (tb - ta);
    if (wc__gap(c, sh, hi) > 0.0f) hi = tb;       /* curvature: fall back to step end */
    /* A chord can cut through a target that the curved fist path never reaches.
     * Without an actual inside endpoint there is no valid bisection bracket. */
    if (!(wc__gap(c, sh, hi) <= 0.0f)) return g;
    for (it = 0; it < 24.0f; it += 1.0f) {         /* ~1e-9 s resolution */
        float m = 0.5f * (lo + hi);
        if (wc__gap(c, sh, m) <= 0.0f) hi = m; else lo = m;
    }
    *t_hit = hi;
    return g;
}

/* ------------------------------------------------------------------- API */

static inline const wc_attack *wc_attack_for_key(char key) {
    if (key == 'U' || key == 'u') return &WC_ATK_U;
    if (key == 'I' || key == 'i') return &WC_ATK_I;
    return 0;
}

static inline wc_result wc__eval_impl(const wc_attack *k, const wc_body *a, const wc_body *d,
                                      const wc_hurtbox *hb, const wc_guard *guards, int n_guards,
                                      const wc_cfg *cfg, int broad) {
    wc_result R;
    wc__ctx c;
    wc__shape sh[2 + WC_MAX_GUARDS];
    int tgt_id[2 + WC_MAX_GUARDS];
    int n_sh = 0, n, i, j, hit_j = -1;
    float t0, t1, t_s, best_t = 1e9f, guard_t = 1e9f;
    wc_v3 prevF;

    R.outcome = WC_INVALID; R.light = WC_OFF; R.margin = 1e9f;
    R.t_contact = -1.0f; R.rel_speed = 0.0f; R.target = 0; R.p_contact = wc_v(0, 0, 0);
    if (!k || !a || !d || !hb || !cfg || !(k->targets & (WC_TGT_HEAD | WC_TGT_TORSO))) return R;
    if (!wc__finite_body(a) || !wc__finite_body(d) || !wc__finite_v(cfg->up) ||
        !isfinite(cfg->max_dt) || cfg->max_dt <= 0.0f ||
        !isfinite(cfg->amber_band) || cfg->amber_band < 0.0f ||
        !isfinite(k->startup) || k->startup < 0.0f ||
        !isfinite(k->extend) || k->extend < 0.0f ||
        !isfinite(k->hold) || k->hold < 0.0f ||
        !isfinite(k->reach) || k->reach < 0.0f ||
        !isfinite(k->fist_r) || k->fist_r < 0.0f ||
        !isfinite(k->sh_fwd) || !isfinite(k->sh_right) || !isfinite(k->sh_up) ||
        !isfinite(k->aim_pitch) || !isfinite(k->auto_aim) || k->auto_aim < 0.0f ||
        !isfinite(k->min_speed) || k->min_speed < 0.0f ||
        !isfinite(hb->head_up) || !isfinite(hb->head_r) || hb->head_r < 0.0f ||
        !isfinite(hb->torso_lo) || !isfinite(hb->torso_hi) || hb->torso_lo > hb->torso_hi ||
        !isfinite(hb->torso_r) || hb->torso_r < 0.0f ||
        n_guards < 0 || (n_guards > 0 && !guards)) return R;
    for (j = 0; j < n_guards && j < WC_MAX_GUARDS; ++j) {
        if (!wc__finite_v(guards[j].pos) || !wc__finite_v(guards[j].vel) ||
            !isfinite(guards[j].radius) || guards[j].radius < 0.0f) return R;
    }

    c.up = wc__norm(cfg->up);
    c.fwd0 = wc__norm(wc__sub(a->fwd, wc__mul(c.up, wc__dot(a->fwd, c.up))));
    if (wc__len(c.up) < 0.5f || wc__len(c.fwd0) < 0.5f) return R; /* degenerate facing/up */
    c.lh = cfg->left_handed; c.k = k; c.a = a;

    t0 = k->startup < 0.0f ? 0.0f : k->startup;
    t1 = t0 + (k->extend > 0.0f ? k->extend : 0.0f) + (k->hold > 0.0f ? k->hold : 0.0f);
    if (!isfinite(t1) || !isfinite(a->yaw_rate * t1)) return R;
    c.t0 = t0;

    /* Broad phase: fist stays within L_h + reach (horizontally) of the attacker
     * root, targets within r_max of the defender's vertical axis. If the closest
     * horizontal root approach leaves >= WC_FAR of clearance, it's a clear miss;
     * a rejected margin is a lower bound. Other margins retain the straight /
     * curved sweep accuracy documented on wc_result. */
    if (broad) {
        wc_v3 X = wc__sub(a->pos, d->pos), V = wc__sub(a->vel, d->vel);
        float VV, ts, lh_reach, rmax, lb;
        X = wc__sub(X, wc__mul(c.up, wc__dot(X, c.up)));
        V = wc__sub(V, wc__mul(c.up, wc__dot(V, c.up)));
        VV = wc__dot(V, V);
        ts = VV > 1e-12f ? wc__clamp(-wc__dot(X, V) / VV, t0, t1) : t0;
        lh_reach = sqrtf(k->sh_fwd * k->sh_fwd + k->sh_right * k->sh_right) + k->reach;
        rmax = hb->head_r > hb->torso_r ? hb->head_r : hb->torso_r;
        lb = wc__len(wc__add(X, wc__mul(V, ts))) - lh_reach - rmax - k->fist_r;
        if (!isfinite(lb)) return R;
        if (lb >= WC_FAR && lb >= cfg->amber_band) {
            R.outcome = WC_MISS; R.light = WC_RED; R.margin = lb; return R;
        }
    }

    /* Launch aim: horizontal facing, optionally yawed (<= auto_aim) toward the
     * defender's centerline as predicted at full extension, then pitched. */
    {
        wc_v3 fs = wc__rot(c.fwd0, c.up, a->yaw_rate * t0);
        if (k->auto_aim > 0.0f) {
            wc_v3 tgt = wc__add(d->pos, wc__mul(d->vel, t0 + k->extend));
            wc_v3 want = wc__sub(tgt, wc__shoulder(&c, t0));
            /* The shoulder translates during extension too. A common velocity
             * added to both fighters must not change their relative aim. */
            want = wc__sub(want, wc__mul(a->vel, k->extend));
            want = wc__sub(want, wc__mul(c.up, wc__dot(want, c.up)));
            if (wc__len(want) > 1e-6f) {
                float yaw = atan2f(wc__dot(c.up, wc__cross(fs, want)), wc__dot(fs, want));
                fs = wc__rot(fs, c.up, wc__clamp(yaw, -k->auto_aim, k->auto_aim));
            }
        }
        c.aim0 = wc__norm(wc__add(wc__mul(fs, cosf(k->aim_pitch)), wc__mul(c.up, sinf(k->aim_pitch))));
    }

    /* Shapes: defender hurtboxes (targets) + guard gloves (blockers). */
    if (k->targets & WC_TGT_HEAD) {
        sh[n_sh].p0 = d->pos; sh[n_sh].v = d->vel;
        sh[n_sh].a = sh[n_sh].b = wc__mul(c.up, hb->head_up);
        sh[n_sh].r = hb->head_r; tgt_id[n_sh++] = WC_TGT_HEAD;
    }
    if (k->targets & WC_TGT_TORSO) {
        sh[n_sh].p0 = d->pos; sh[n_sh].v = d->vel;
        sh[n_sh].a = wc__mul(c.up, hb->torso_lo); sh[n_sh].b = wc__mul(c.up, hb->torso_hi);
        sh[n_sh].r = hb->torso_r; tgt_id[n_sh++] = WC_TGT_TORSO;
    }
    for (j = 0; guards && j < n_guards && j < WC_MAX_GUARDS; ++j) {
        sh[n_sh].p0 = guards[j].pos; sh[n_sh].v = guards[j].vel;
        sh[n_sh].a = sh[n_sh].b = wc_v(0, 0, 0);
        sh[n_sh].r = guards[j].radius; tgt_id[n_sh++] = 0;
    }

    /* Sweep. Contacts count once the fist is WC_MIN_EXT out (t_s). Phases:
     * extension [t_s, t0+extend] and hold [.., t1]. Without yaw the fist path
     * relative to each (translating) shape is exactly linear in each phase, so
     * one swept segment per phase is exact. While turning, split so each step
     * turns <= WC_MAX_DYAW and lasts <= cfg->max_dt. */
    {
        float bnd[3], ext = k->extend > 0.0f ? k->extend : 0.0f, wabs = fabsf(a->yaw_rate);
        int nb = 0, p, counts[2], total_steps = 0;
        t_s = t0 + WC_MIN_EXT * ext;
        bnd[nb++] = t_s;
        if (ext > 0.0f) bnd[nb++] = t0 + ext;
        if (t1 > t0 + ext) bnd[nb++] = t1;
        if (nb == 1) bnd[nb++] = t_s;             /* zero-length window: point test */
        /* Validate all sweep sizes before any float-to-int conversion or work. */
        for (p = 0; p + 1 < nb; ++p) {
            float L = bnd[p + 1] - bnd[p], step = L, nf;
            if (wabs > 0.0f) {
                step = WC_MAX_DYAW / wabs;
                if (cfg->max_dt > 1e-5f && cfg->max_dt < step) step = cfg->max_dt;
            }
            nf = (L > 0.0f && step > 0.0f) ? ceilf(L / step) : 1.0f;
            if (!isfinite(nf) || nf > (float)WC_MAX_SWEEP_STEPS ||
                (L > 0.0f && step <= 0.0f)) return R;
            n = nf < 1.0f ? 1 : (int)nf;
            if (n > WC_MAX_SWEEP_STEPS - total_steps) return R;
            total_steps += n;
            counts[p] = n;
        }
        prevF = wc__fist(&c, t_s);
        if (!wc__finite_v(prevF)) return R;
        for (p = 0; p + 1 < nb; ++p) {
            float L = bnd[p + 1] - bnd[p];
            n = counts[p];
            for (i = 0; i < n; ++i) {
                float ta = bnd[p] + L * (float)i / (float)n, tb = bnd[p] + L * (float)(i + 1) / (float)n;
                wc_v3 F = wc__fist(&c, tb);
                if (!wc__finite_v(F)) return R;
                for (j = 0; j < n_sh; ++j) {
                    /* only refine while an earlier contact of that kind is still possible */
                    int refine = tgt_id[j] ? (best_t > ta) : (guard_t > ta);
                    float th, g = wc__sweep(&c, &sh[j], ta, tb, prevF, F, refine, &th);
                    if (tgt_id[j]) {
                        if (g < R.margin) R.margin = g;
                        if (th >= 0.0f && th < best_t) { best_t = th; hit_j = j; }
                    } else if (th >= 0.0f && th < guard_t) {
                        guard_t = th;
                    }
                }
                prevF = F;
            }
        }
    }

    if (hit_j >= 0) {
        R.t_contact = best_t;
        R.target = tgt_id[hit_j];
        R.p_contact = wc__fist(&c, best_t);
        R.rel_speed = wc__closing(&c, &sh[hit_j], best_t);
        if (guard_t <= best_t)               { R.outcome = WC_BLOCKED; R.light = WC_RED; }
        else if (best_t <= t_s)              { R.outcome = WC_WEAK;    R.light = WC_AMBER; } /* smothered */
        else if (R.rel_speed < k->min_speed) { R.outcome = WC_WEAK;    R.light = WC_AMBER; }
        else                                 { R.outcome = WC_HIT;     R.light = WC_GREEN; }
    } else {
        R.outcome = WC_MISS;
        R.light = R.margin < cfg->amber_band ? WC_AMBER : WC_RED;
    }
    return R;
}

static inline wc_result wc_eval(const wc_attack *k, const wc_body *a, const wc_body *d,
                                const wc_hurtbox *hb, const wc_guard *guards, int n_guards,
                                const wc_cfg *cfg) {
    return wc__eval_impl(k, a, d, hb, guards, n_guards, cfg, 1);
}

/* Only U and I get a light; any other key returns WC_OFF. */
static inline wc_result wc_eval_key(char key, const wc_body *a, const wc_body *d,
                                    const wc_hurtbox *hb, const wc_guard *guards, int n_guards,
                                    const wc_cfg *cfg) {
    return wc_eval(wc_attack_for_key(key), a, d, hb, guards, n_guards, cfg);
}

/* 3 floats for an RL observation: [clean hit, margin / 0.5 m clipped, contact time / window or -1]. */
static inline void wc_obs(const wc_result *r, const wc_attack *k, float out[3]) {
    float w = k->startup + k->extend + k->hold;
    out[0] = r->outcome == WC_HIT ? 1.0f : 0.0f;
    out[1] = wc__clamp(r->margin / 0.5f, -1.0f, 1.0f);
    out[2] = (r->t_contact >= 0.0f && w > 0.0f) ? r->t_contact / w : -1.0f;
}

#endif /* WILL_CONNECT_H */
