/* Regression tests for diagnostic predictor defects found during integration.
 * gcc -std=c99 -O2 -Wall -Wextra -pedantic will_connect_regression_test.c -lm
 * These test the geometric model, not official REK hit registration. */
#include <stdio.h>
#include <math.h>
#include "will_connect.h"

static int passed, failed;
#define CHECK(expr, label) do { if (expr) ++passed; else { ++failed; \
    printf("FAIL line %d: %s\n", __LINE__, label); } } while (0)

static wc_body body(float x, float y) {
    wc_body b = {{x,y,0}, {0,0,0}, {1,0,0}, 0};
    return b;
}
static int invalid(wc_result r) {
    return r.outcome == WC_INVALID && r.light == WC_OFF && r.t_contact < 0;
}
static int same_result(wc_result a, wc_result b) {
    return a.outcome == b.outcome && a.light == b.light && a.target == b.target &&
           fabsf(a.margin-b.margin) < 1e-5f && fabsf(a.t_contact-b.t_contact) < 1e-5f &&
           fabsf(a.rel_speed-b.rel_speed) < 1e-4f;
}
int main(void) {
    wc_body a, d;
    wc_result r, base;
    int hand, turning, v;
    const wc_v3 common[] = {{0,1,0}, {0.8f,-0.5f,0.25f}, {-0.4f,0.7f,-0.2f}};
    for (hand=0; hand<2; ++hand) for (turning=0; turning<2; ++turning) {
        const wc_attack *k = hand ? &WC_ATK_I : &WC_ATK_U;
        a=body(0,0); d=body(0.45f,0); a.yaw_rate=turning ? 0.8f : 0;
        base=wc_eval(k,&a,&d,&WC_HURTBOX_G1,0,0,&WC_CFG_DEFAULT);
        for (v=0; v<3; ++v) {
            a.vel=d.vel=common[v];
            r=wc_eval(k,&a,&d,&WC_HURTBOX_G1,0,0,&WC_CFG_DEFAULT);
            CHECK(same_result(base,r), "shared world velocity preserves the relative prediction");
        }
    }
    a=body(0,0); d=body(0.45f,0);
    d.pos.x=NAN;
    CHECK(invalid(wc_eval_key('U',&a,&d,&WC_HURTBOX_G1,0,0,&WC_CFG_DEFAULT)), "NaN position is unknown");
    d=body(0.45f,0); a.yaw_rate=NAN;
    CHECK(invalid(wc_eval_key('U',&a,&d,&WC_HURTBOX_G1,0,0,&WC_CFG_DEFAULT)), "NaN yaw is unknown");
    a.yaw_rate=INFINITY;
    CHECK(invalid(wc_eval_key('U',&a,&d,&WC_HURTBOX_G1,0,0,&WC_CFG_DEFAULT)), "infinite yaw is unknown");
    a.yaw_rate=1e30f;
    CHECK(invalid(wc_eval_key('U',&a,&d,&WC_HURTBOX_G1,0,0,&WC_CFG_DEFAULT)), "huge finite yaw is bounded before conversion");
    a=body(0,0); a.vel.z=INFINITY;
    CHECK(invalid(wc_eval_key('U',&a,&d,&WC_HURTBOX_G1,0,0,&WC_CFG_DEFAULT)), "infinite velocity is unknown");
    a=body(0,0);
    CHECK(invalid(wc_eval_key('U',&a,&d,&WC_HURTBOX_G1,0,1,&WC_CFG_DEFAULT)), "missing declared guard array is unknown");
    {
        wc_attack k=WC_ATK_U;
        wc_cfg cfg=WC_CFG_DEFAULT;
        wc_hurtbox hb=WC_HURTBOX_G1;
        wc_guard g={{0.3f,0.16f,0.3f},{0,0,0},NAN};
        CHECK(invalid(wc_eval(&k,&a,&d,&hb,&g,1,&cfg)), "NaN guard radius is unknown");
        k.reach=-0.4f;
        CHECK(invalid(wc_eval(&k,&a,&d,&hb,0,0,&cfg)), "negative reach is invalid");
        k=WC_ATK_U; k.startup=-0.1f;
        CHECK(invalid(wc_eval(&k,&a,&d,&hb,0,0,&cfg)), "negative timing is invalid");
        k=WC_ATK_U; k.min_speed=NAN;
        CHECK(invalid(wc_eval(&k,&a,&d,&hb,0,0,&cfg)), "NaN threshold is invalid");
        k=WC_ATK_U; hb.torso_lo=hb.torso_hi+0.1f;
        CHECK(invalid(wc_eval(&k,&a,&d,&hb,0,0,&cfg)), "reversed torso endpoints are invalid");
        hb=WC_HURTBOX_G1; cfg.amber_band=NAN;
        CHECK(invalid(wc_eval(&k,&a,&d,&hb,0,0,&cfg)), "NaN amber band is invalid");
        cfg=WC_CFG_DEFAULT; cfg.up=wc_v(0,0,0);
        CHECK(invalid(wc_eval(&k,&a,&d,&hb,0,0,&cfg)), "zero up vector is invalid");
    }
    {
        wc_attack k=WC_ATK_U;
        wc_cfg cfg=WC_CFG_DEFAULT;
        wc_hurtbox hb={0,0.10f,0,0,0.12f};
        wc__ctx c; wc__shape sh;
        float min_actual=1e9f; int i;
        k.startup=0; k.extend=0; k.hold=0.02f;
        k.sh_fwd=k.sh_right=k.sh_up=0; k.auto_aim=0; k.targets=WC_TGT_HEAD;
        a=body(0,0); a.yaw_rate=1;
        d=body(0.25999f*cosf(0.01f),0.25999f*sinf(0.01f));
        cfg.max_dt=0.02f;
        c.up=wc_v(0,0,1); c.fwd0=c.aim0=wc_v(1,0,0);
        c.t0=0; c.lh=0; c.k=&k; c.a=&a;
        sh.p0=d.pos; sh.v=wc_v(0,0,0); sh.a=sh.b=wc_v(0,0,0); sh.r=hb.head_r;
        for (i=0;i<=10000;i++) {
            float gap=wc__gap(&c,&sh,0.02f*(float)i/10000.0f);
            if (gap<min_actual) min_actual=gap;
        }
        r=wc_eval(&k,&a,&d,&hb,0,0,&cfg);
        CHECK(min_actual>0, "reference curved path does not overlap the target");
        CHECK(r.margin<0, "chord approximation crosses target in this constructed grazing case");
        CHECK(r.outcome==WC_MISS && r.t_contact<0 && r.target==0,
              "chord intersection cannot fabricate an actual curved contact");
        CHECK(r.light==WC_AMBER, "grazing approximation remains visibly uncertain");
    }
    {
        wc_cfg cfg=WC_CFG_DEFAULT; wc_attack k=WC_ATK_U;
        cfg.amber_band=1.0f; k.auto_aim=0; a=body(0,0); d=body(1.3f,0.16f);
        r=wc__eval_impl(&k,&a,&d,&WC_HURTBOX_G1,0,0,&cfg,1);
        base=wc__eval_impl(&k,&a,&d,&WC_HURTBOX_G1,0,0,&cfg,0);
        CHECK(same_result(base,r) && r.light==WC_AMBER,
              "broad phase respects an amber band larger than WC_FAR");
    }
    printf("%d passed, %d failed\n",passed,failed);
    return failed!=0;
}
