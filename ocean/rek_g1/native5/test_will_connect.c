/*
 * test_will_connect.c - geometry, invariance and brute-force tests for will_connect.h
 *   gcc -std=c99 -O2 -Wall -Wextra -pedantic test_will_connect.c -lm && ./a.out
 *   cl /O2 test_will_connect.c        (MSVC)
 *
 * Analytic cases use "straight" specs (auto_aim = 0) so expected numbers can be
 * derived by hand:
 *   U: shoulder (0.02, +0.16, 0.30), reach 0.42 over 0.16 s after 0.05 s -> 2.625 m/s,
 *      gate (25% out) at t = 0.09 s, fist x = 0.125
 *   I: shoulder (0.02, -0.16, 0.30), reach 0.46 over 0.22 s after 0.08 s -> 2.0909 m/s
 *   torso r 0.12 + glove r 0.06 -> contact at 0.18 m from the torso axis.
 * Frame: right-handed z-up, attacker at origin facing +x, so +y is the attacker's left.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "will_connect.h"

static int fails = 0, passes = 0;
#define CHECK(cond, msg) do { if (cond) { ++passes; } else { ++fails; \
    printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); } } while (0)

static const char *OUT[] = { "INVALID", "MISS", "HIT", "WEAK", "BLOCKED" };
static const char *LIT[] = { "OFF", "RED", "AMBER", "GREEN" };

static wc_body body(float x, float y, float vx, float vy, float fx, float fy) {
    wc_body b;
    b.pos = wc_v(x, y, 0); b.vel = wc_v(vx, vy, 0); b.fwd = wc_v(fx, fy, 0); b.yaw_rate = 0;
    return b;
}
static float fabsf_(float x) { return x < 0 ? -x : x; }
static float rnd(unsigned *s) { *s = *s * 1664525u + 1013904223u; return (float)(*s >> 8) / 16777216.0f; }

/* rotate about z (right-handed, z-up) */
static wc_v3 rz(wc_v3 v, float p) {
    return wc_v(cosf(p) * v.x - sinf(p) * v.y, sinf(p) * v.x + cosf(p) * v.y, v.z);
}
/* right-handed z-up (x fwd, y left) -> Unity left-handed y-up (x right, y up, z fwd) */
static wc_v3 to_unity(wc_v3 v) { return wc_v(-v.y, v.z, v.x); }

static void show(const char *name, wc_result r) {
    printf("  %-36s %-8s %-6s margin=%+.4f t=%.4f v=%.3f tgt=%d\n", name, OUT[r.outcome], LIT[r.light],
           (double)r.margin, (double)r.t_contact, (double)r.rel_speed, r.target);
}

int main(void) {
    const wc_cfg *C = &WC_CFG_DEFAULT;
    const wc_hurtbox *H = &WC_HURTBOX_G1;
    wc_attack SU = WC_ATK_U, SI = WC_ATK_I;       /* straight specs */
    wc_body me = body(0, 0, 0, 0, 1, 0), op;
    wc_result r, r2;
    SU.auto_aim = 0; SI.auto_aim = 0;

    printf("in-line punches (analytic)\n");
    op = body(0.50f, 0.16f, 0, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U in line d=0.50", r);
    /* contact x = 0.50 - 0.18 = 0.32 -> e = 0.30/0.42 -> t = 0.05 + 0.7143*0.16 = 0.16429 */
    CHECK(r.outcome == WC_HIT && r.light == WC_GREEN && r.target == WC_TGT_TORSO, "U in line -> green torso");
    CHECK(fabsf_(r.t_contact - 0.16429f) < 5e-4f && fabsf_(r.p_contact.x - 0.32f) < 1e-3f, "U contact time/point");
    CHECK(fabsf_(r.rel_speed - 2.625f) < 0.01f, "U head-on impact speed = reach/extend");
    CHECK(fabsf_(r.margin + 0.12f) < 1e-3f, "U margin = deepest penetration (0.50-0.44-0.18)");
    op = body(0.50f, -0.16f, 0, 0, -1, 0);
    r = wc_eval(&SI, &me, &op, H, 0, 0, C); show("I in line d=0.50", r);
    CHECK(r.outcome == WC_HIT && fabsf_(r.t_contact - 0.22348f) < 5e-4f && fabsf_(r.rel_speed - 2.0909f) < 0.01f,
          "I in line -> hit, t = 0.08 + (0.30/0.46)*0.22");
    op = body(1.2f, 0.16f, 0, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U far d=1.2", r);
    CHECK(r.outcome == WC_MISS && r.light == WC_RED && fabsf_(r.margin - 0.58f) < 1e-3f, "far -> red, margin 0.58");

    printf("range edges (in line: U <= 0.62, I <= 0.66)\n");
    op = body(0.61f, 0.16f, 0, 0, -1, 0); r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U d=0.61", r);
    CHECK(r.light == WC_GREEN, "U 0.61 green");
    op = body(0.63f, 0.16f, 0, 0, -1, 0); r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U d=0.63", r);
    CHECK(r.outcome == WC_MISS && r.light == WC_AMBER && fabsf_(r.margin - 0.01f) < 1e-3f, "U 0.63 near miss amber");
    op = body(0.65f, -0.16f, 0, 0, -1, 0); r = wc_eval(&SI, &me, &op, H, 0, 0, C); show("I d=0.65", r);
    CHECK(r.light == WC_GREEN, "I 0.65 green");
    op = body(0.67f, -0.16f, 0, 0, -1, 0); r = wc_eval(&SI, &me, &op, H, 0, 0, C); show("I d=0.67", r);
    CHECK(r.outcome == WC_MISS && fabsf_(r.margin - 0.01f) < 1e-3f, "I 0.67 miss");

    printf("impact speed / smother\n");
    op = body(0.45f, 0, 0, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U straight, centered opp (glancing)", r);
    /* contact x = 0.45 - sqrt(0.18^2 - 0.16^2) = 0.36754; normal (0.0825,-0.16)/0.18 -> 2.625*0.4581 */
    CHECK(r.outcome == WC_WEAK && fabsf_(r.rel_speed - 1.2025f) < 0.01f, "glancing blow -> weak, impact speed on normal");
    r = wc_eval_key('U', &me, &op, H, 0, 0, C); show("U default (auto-aim), centered opp", r);
    CHECK(r.outcome == WC_HIT && fabsf_(r.rel_speed - 2.625f) < 0.02f, "auto-aim converges -> head-on hit");
    op = body(0.20f, 0.16f, 0, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U opp inside chamber range", r);
    CHECK(r.outcome == WC_WEAK && fabsf_(r.t_contact - 0.09f) < 1e-4f, "already touching at 25% out -> smothered");
    { wc_body away = body(0, 0, 0, 0, -1, 0);
      op = body(0.45f, 0, 0, 0, -1, 0);
      r = wc_eval_key('U', &away, &op, H, 0, 0, C); show("U default, facing away", r);
      CHECK(r.outcome == WC_MISS && r.light == WC_RED, "facing away -> red");
      op = body(-0.05f, 0.45f, 0, 0, 1, 0);   /* beside/behind on the lead side, bodies not overlapping */
      r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U opp beside/behind lead shoulder", r);
      CHECK(r.outcome == WC_MISS, "target beside/behind -> miss"); }

    printf("motion\n");
    op = body(0.9f, 0.16f, 0, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U d=0.9 static", r);
    CHECK(r.outcome == WC_MISS && fabsf_(r.margin - 0.28f) < 1e-3f, "0.9 static miss");
    op = body(0.9f, 0.16f, -2.0f, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U d=0.9 opp stepping in 2 m/s", r);
    /* 1.01125 - 4.625 t = 0.18 -> t = 0.17973, impact 2.625 + 2 */
    CHECK(r.outcome == WC_HIT && fabsf_(r.t_contact - 0.17973f) < 5e-4f && fabsf_(r.rel_speed - 4.625f) < 0.02f,
          "step-in -> hit at closing speed");
    op = body(0.45f, 0.16f, 3.0f, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U opp retreating 3 m/s", r);
    CHECK(r.outcome == WC_MISS, "retreat -> miss");
    op = body(0.85f, 0.16f, -1.0f, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U opp walks into held fist", r);
    CHECK(r.outcome == WC_WEAK && fabsf_(r.t_contact - 0.23f) < 5e-4f && fabsf_(r.rel_speed - 1.0f) < 0.02f,
          "held fist, 1 m/s walk-in -> weak");

    printf("guard / clean-hit speed\n");
    { wc_guard g[2];
      g[0].pos = wc_v(0.30f, 0.16f, 0.30f);  g[0].vel = wc_v(0, 0, 0); g[0].radius = 0.06f;
      g[1].pos = wc_v(0.30f, -0.40f, 0.30f); g[1].vel = wc_v(0, 0, 0); g[1].radius = 0.06f;
      op = body(0.50f, 0.16f, 0, 0, -1, 0);
      r = wc_eval(&SU, &me, &op, H, g, 2, C); show("U glove in path", r);
      CHECK(r.outcome == WC_BLOCKED && r.light == WC_RED, "guard in path -> blocked");
      r = wc_eval(&SU, &me, &op, H, g + 1, 1, C); show("U glove off-line", r);
      CHECK(r.outcome == WC_HIT, "guard off-line -> hit"); }
    { wc_attack slow = SU; slow.min_speed = 10.0f;
      op = body(0.50f, 0.16f, 0, 0, -1, 0);
      r = wc_eval(&slow, &me, &op, H, 0, 0, C); show("U min_speed 10", r);
      CHECK(r.outcome == WC_WEAK && r.light == WC_AMBER, "below min impact speed -> weak amber"); }

    printf("keys\n");
    op = body(0.45f, 0, 0, 0, -1, 0);
    r = wc_eval_key('J', &me, &op, H, 0, 0, C);
    CHECK(r.light == WC_OFF && r.outcome == WC_INVALID, "non U/I key -> off");
    r = wc_eval_key('u', &me, &op, H, 0, 0, C);
    CHECK(r.outcome == WC_HIT, "lowercase u accepted");

    printf("laterality (open side)\n");
    op = body(0.50f, 0.10f, 0, 0, -1, 0);
    r = wc_eval(&SU, &me, &op, H, 0, 0, C); show("U straight, opp offset left", r);
    r2 = wc_eval(&SI, &me, &op, H, 0, 0, C); show("I straight, opp offset left", r2);
    /* U: lateral 0.06 -> normal cos = 0.1697/0.18 -> 2.475 m/s; I: lateral 0.26 > 0.18 */
    CHECK(r.outcome == WC_HIT && fabsf_(r.rel_speed - 2.475f) < 0.01f && r2.outcome == WC_MISS,
          "left offset: lead hand lands, rear misses");

    printf("auto-aim (yaw-only)\n");
    { wc_attack aa = SU;
      op = body(0.40f, -0.15f, 0, 0, -1, 0);
      r = wc_eval(&aa, &me, &op, H, 0, 0, C); show("U no auto-aim, opp off-line", r);
      CHECK(r.outcome == WC_MISS, "no auto-aim -> miss");
      aa.auto_aim = 0.8f;   /* needed yaw = atan(0.31/0.38) = 0.684 rad */
      r = wc_eval(&aa, &me, &op, H, 0, 0, C); show("U auto-aim 0.8 rad", r);
      CHECK(r.outcome == WC_HIT && fabsf_(r.rel_speed - 2.625f) < 0.02f, "auto-aim lock -> head-on hit");
      aa.auto_aim = 0.2f;
      r = wc_eval(&aa, &me, &op, H, 0, 0, C); show("U auto-aim 0.2 rad (cone too small)", r);
      CHECK(r.outcome != WC_HIT, "cone smaller than needed -> no clean hit"); }

    printf("invariance: world yaw rotation\n");
    { float phis[3] = { 0.7f, 2.1f, -1.3f };
      int n, bad = 0;
      wc_guard g; g.pos = wc_v(0.26f, 0.05f, 0.30f); g.vel = wc_v(0.3f, 0, 0); g.radius = 0.06f;
      for (n = 0; n < 3; ++n) {
          wc_body a = me, d = body(0.62f, 0.07f, -1.1f, 0.2f, -1, 0), a2, d2;
          wc_guard g2 = g;
          wc_attack k = WC_ATK_I; k.auto_aim = 0.3f;
          a.yaw_rate = 1.5f;
          r = wc_eval(&k, &a, &d, H, &g, 1, C);
          a2 = a; d2 = d;
          a2.pos = rz(a.pos, phis[n]); a2.vel = rz(a.vel, phis[n]); a2.fwd = rz(a.fwd, phis[n]);
          d2.pos = rz(d.pos, phis[n]); d2.vel = rz(d.vel, phis[n]);
          g2.pos = rz(g.pos, phis[n]); g2.vel = rz(g.vel, phis[n]);
          r2 = wc_eval(&k, &a2, &d2, H, &g2, 1, C);
          if (r.outcome != r2.outcome || fabsf_(r.margin - r2.margin) > 1e-4f ||
              fabsf_(r.t_contact - r2.t_contact) > 1e-4f) ++bad;
          if (n == 0) { show("base (yawing, moving, guard)", r); show("rotated 0.7 rad", r2); }
      }
      CHECK(bad == 0, "results invariant to world yaw"); }

    printf("invariance: Unity (left-handed, y-up) == MuJoCo (right-handed, z-up)\n");
    { wc_cfg U = WC_CFG_DEFAULT; int n, bad = 0; unsigned seed = 99u;
      U.up = wc_v(0, 1, 0); U.left_handed = 1;
      for (n = 0; n < 2000; ++n) {
          wc_body a = body(0, 0, rnd(&seed) - 0.5f, rnd(&seed) - 0.5f, 1, 0.4f * rnd(&seed) - 0.2f);
          wc_body d = body(0.2f + 0.7f * rnd(&seed), 0.6f * rnd(&seed) - 0.3f, 2 * rnd(&seed) - 1, 2 * rnd(&seed) - 1, -1, 0);
          wc_body a2, d2; char key = (n & 1) ? 'I' : 'U';
          a.yaw_rate = 4 * rnd(&seed) - 2;
          r = wc_eval_key(key, &a, &d, H, 0, 0, C);   /* default specs: auto-aim on */
          a2 = a; d2 = d;
          a2.pos = to_unity(a.pos); a2.vel = to_unity(a.vel); a2.fwd = to_unity(a.fwd);
          a2.yaw_rate = -a.yaw_rate;                  /* reflection flips rotation sense */
          d2.pos = to_unity(d.pos); d2.vel = to_unity(d.vel);
          r2 = wc_eval_key(key, &a2, &d2, H, 0, 0, &U);
          if (r.outcome != r2.outcome || fabsf_(r.margin - r2.margin) > 1e-4f ||
              fabsf_(r.t_contact - r2.t_contact) > 1e-4f || fabsf_(r.rel_speed - r2.rel_speed) > 1e-3f) ++bad;
      }
      CHECK(bad == 0, "same answers in Unity frame (2000 random cases, auto-aim + yaw)"); }

    printf("broad phase == full sweep (random scenarios)\n");
    { unsigned seed = 12345u; int n, bad = 0, lb_bad = 0, skipped = 0;
      for (n = 0; n < 20000; ++n) {
          wc_body a = body(0, 0, 3 * rnd(&seed) - 1.5f, 3 * rnd(&seed) - 1.5f, rnd(&seed) - 0.5f, rnd(&seed) - 0.5f);
          wc_body d = body(2.5f * rnd(&seed) - 1.25f, 2.5f * rnd(&seed) - 1.25f, 3 * rnd(&seed) - 1.5f, 3 * rnd(&seed) - 1.5f, 1, 0);
          wc_guard g; wc_attack k = (n & 1) ? WC_ATK_I : WC_ATK_U;
          g.pos = wc__add(d.pos, wc_v(0.4f * rnd(&seed) - 0.2f, 0.4f * rnd(&seed) - 0.2f, 0.3f + 0.2f * rnd(&seed)));
          g.vel = d.vel; g.radius = 0.06f;
          a.yaw_rate = 4 * rnd(&seed) - 2; k.auto_aim = (n & 2) ? 0.4f : 0.0f;
          r  = wc__eval_impl(&k, &a, &d, H, &g, 1, C, 1);
          r2 = wc__eval_impl(&k, &a, &d, H, &g, 1, C, 0);
          if (r.outcome != r2.outcome || r.light != r2.light) ++bad;
          if (r.margin < WC_FAR) { if (fabsf_(r.margin - r2.margin) > 1e-6f) ++bad; }
          else { ++skipped; if (r.margin > r2.margin + 1e-5f) ++lb_bad; }
      }
      printf("  %d scenarios, %d rejected by broad phase\n", 20000, skipped);
      CHECK(bad == 0, "broad phase never changes outcome/light; margin exact below WC_FAR");
      CHECK(lb_bad == 0, "broad-phase margin is a true lower bound"); }

    printf("fist velocity: analytic == finite difference\n");
    { unsigned seed = 777u; int n, bad = 0;
      for (n = 0; n < 5000; ++n) {
          wc__ctx c; wc_body a = body(rnd(&seed), rnd(&seed), 2 * rnd(&seed) - 1, 2 * rnd(&seed) - 1, rnd(&seed) - 0.5f, rnd(&seed) - 0.5f);
          wc_attack k = WC_ATK_I; float t, h = 1e-3f; wc_v3 fd, an;
          a.yaw_rate = 6 * rnd(&seed) - 3;
          c.up = wc_v(0, 0, 1); c.fwd0 = wc__norm(wc_v(a.fwd.x, a.fwd.y, 0)); c.lh = n & 1;
          c.k = &k; c.a = &a; c.t0 = k.startup;
          c.aim0 = wc__norm(wc_v(rnd(&seed) - 0.5f, rnd(&seed) - 0.5f, rnd(&seed) - 0.5f));
          t = c.t0 + 0.01f + (k.extend + k.hold - 0.02f) * rnd(&seed);
          if (fabsf_(t - (c.t0 + k.extend)) < 2 * h) continue;       /* skip the kink */
          fd = wc__mul(wc__sub(wc__fist(&c, t + h), wc__fist(&c, t - h)), 0.5f / h);
          an = wc__fist_vel(&c, t);
          if (wc__len(wc__sub(fd, an)) > 2e-3f * (1.0f + wc__len(an))) ++bad;
      }
      CHECK(bad == 0, "analytic fist velocity matches central difference"); }

    printf("sweep vs brute force (dense 0.1 ms sampling, yawing attackers)\n");
    { unsigned seed = 4242u; int n, bad = 0, tested = 0, mix[5] = { 0, 0, 0, 0, 0 };
      for (n = 0; n < 4000; ++n) {
          wc_body a = body(0, 0, 2 * rnd(&seed) - 1, 2 * rnd(&seed) - 1, 1, 0.3f * rnd(&seed) - 0.15f);
          wc_body d = body(0.3f + 0.5f * rnd(&seed), 0.5f * rnd(&seed) - 0.25f, 2 * rnd(&seed) - 1, 2 * rnd(&seed) - 1, -1, 0);
          wc_attack k = (n & 1) ? WC_ATK_I : WC_ATK_U;
          wc_guard g; wc__ctx c; wc__shape sh[3]; int tid[3] = { WC_TGT_HEAD, WC_TGT_TORSO, 0 };
          float tt, ts, t_hit = 1e9f, t_g = 1e9f, mmin = 1e9f, spd = 0; int j, hj = -1, exp_out;
          a.yaw_rate = 8 * rnd(&seed) - 4;                 /* up to 4 rad/s */
          k.aim_pitch = 0.3f * rnd(&seed); k.auto_aim = 0;
          g.pos = wc__add(d.pos, wc_v(-0.15f, 0.3f * rnd(&seed) - 0.15f, 0.25f + 0.2f * rnd(&seed)));
          g.vel = d.vel; g.radius = 0.06f;
          r = wc__eval_impl(&k, &a, &d, H, &g, 1, C, 0);
          c.up = wc_v(0, 0, 1); c.fwd0 = wc__norm(wc_v(a.fwd.x, a.fwd.y, 0)); c.lh = 0; c.k = &k; c.a = &a;
          c.t0 = k.startup;
          { wc_v3 fs = wc__rot(c.fwd0, c.up, a.yaw_rate * c.t0);
            c.aim0 = wc__norm(wc__add(wc__mul(fs, cosf(k.aim_pitch)), wc__mul(c.up, sinf(k.aim_pitch)))); }
          sh[0].p0 = d.pos; sh[0].v = d.vel; sh[0].a = sh[0].b = wc_v(0, 0, H->head_up); sh[0].r = H->head_r;
          sh[1].p0 = d.pos; sh[1].v = d.vel; sh[1].a = wc_v(0, 0, H->torso_lo); sh[1].b = wc_v(0, 0, H->torso_hi); sh[1].r = H->torso_r;
          sh[2].p0 = g.pos; sh[2].v = g.vel; sh[2].a = sh[2].b = wc_v(0, 0, 0); sh[2].r = g.radius;
          ts = c.t0 + WC_MIN_EXT * k.extend;
          for (tt = ts; tt <= c.t0 + k.extend + k.hold + 1e-6f; tt += 1e-4f)
              for (j = 0; j < 3; ++j) {
                  float gp = wc__gap(&c, &sh[j], tt);
                  if (tid[j] && gp < mmin) mmin = gp;
                  if (gp <= 0) {
                      if (tid[j] && tt < t_hit) { t_hit = tt; hj = j; }
                      if (!tid[j] && tt < t_g) t_g = tt;
                  }
              }
          if (hj >= 0) spd = wc__closing(&c, &sh[hj], t_hit);
          exp_out = hj < 0 ? WC_MISS : (t_g <= t_hit ? WC_BLOCKED :
                    (t_hit <= ts ? WC_WEAK : (spd < k.min_speed ? WC_WEAK : WC_HIT)));
          /* skip grazing / tie cases that sampling can't resolve */
          if (fabsf_(mmin) < 2e-3f || (hj >= 0 && t_g < 1e8f && fabsf_(t_g - t_hit) < 3e-4f) ||
              (hj >= 0 && fabsf_(spd - k.min_speed) < 0.05f) || (hj >= 0 && t_hit > ts && t_hit < ts + 3e-4f)) continue;
          ++tested; ++mix[exp_out];
          if ((int)r.outcome != exp_out || (hj >= 0 && fabsf_(r.t_contact - t_hit) > 3e-4f) ||
              fabsf_(r.margin - mmin) > 1e-3f) {
              if (bad < 5) printf("  mismatch n=%d: sweep %s t=%.5f m=%.5f | brute %s t=%.5f m=%.5f\n", n,
                                  OUT[r.outcome], (double)r.t_contact, (double)r.margin,
                                  OUT[exp_out], (double)(hj >= 0 ? t_hit : -1), (double)mmin);
              ++bad;
          }
      }
      printf("  %d scenarios compared (miss %d, hit %d, weak %d, blocked %d)\n",
             tested, mix[WC_MISS], mix[WC_HIT], mix[WC_WEAK], mix[WC_BLOCKED]);
      CHECK(bad == 0, "sweep matches dense brute force (outcome, contact time, margin)"); }

    printf("timing\n");
    { int n, N = 200000, pass; volatile float sink = 0;
      const float dist[2] = { 0.9f, 2.5f }; const char *nm[2] = { "near miss (full sweep)", "far (broad-phase reject)" };
      wc_guard g[2];
      g[0].pos = wc_v(0.3f, 0.3f, 0.3f); g[0].vel = wc_v(0, 0, 0); g[0].radius = 0.06f; g[1] = g[0];
      for (pass = 0; pass < 2; ++pass) {
          clock_t t0 = clock();
          for (n = 0; n < N; ++n) {
              wc_body d = body(dist[pass] + 1e-6f * (float)(n & 7), 0, 0, 0, -1, 0);
              sink += wc_eval_key(n & 1 ? 'I' : 'U', &me, &d, H, g, 2, C).margin;
          }
          printf("  %-26s %.3f us / eval\n", nm[pass], 1e6 * (double)(clock() - t0) / CLOCKS_PER_SEC / N);
      }
      (void)sink; }

    printf("\n%d passed, %d failed\n", passes, fails);
    return fails != 0;
}
