// Top-down arena view for the REK combat env. Shows both G1s, their facing,
// the live hit volume during a move's active frames, and the scoreboard the
// round is actually decided on.

#pragma once

#include "raylib.h"
#include "rek.h"

#define REK_SCREEN_W 1080
#define REK_SCREEN_H 720
#define REK_PIXELS_PER_M 150.0f

static const Color REK_RED = (Color){187, 0, 0, 255};
static const Color REK_CYAN = (Color){0, 187, 187, 255};
static const Color REK_WHITE = (Color){241, 241, 241, 255};
static const Color REK_GREY = (Color){96, 108, 108, 255};
static const Color REK_BACKGROUND = (Color){6, 24, 24, 255};

struct Client {
    float cx;
    float cy;
};

Client* make_client(Rek* env) {
    (void)env;
    InitWindow(REK_SCREEN_W, REK_SCREEN_H, "PufferLib REK");
    SetTargetFPS((int)REK_TICK_HZ);
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->cx = REK_SCREEN_W / 2.0f;
    client->cy = REK_SCREEN_H / 2.0f;
    return client;
}

void close_client(Client* client) {
    free(client);
    if (IsWindowReady()) CloseWindow();
}

static inline Vector2 rek_to_screen(const Client* c, float x, float z) {
    return (Vector2){c->cx + x * REK_PIXELS_PER_M, c->cy + z * REK_PIXELS_PER_M};
}

static void rek_draw_fighter(const Client* c, const Rek* env, int slot, Color col) {
    const Fighter* f = &env->fighters[slot];
    Vector2 p = rek_to_screen(c, f->x, f->z);
    float body_px = env->body_radius * REK_PIXELS_PER_M;

    // A downed robot is drawn flat and dim so the state reads at a glance.
    if (f->down_timer > 0) {
        DrawEllipse((int)p.x, (int)p.y, body_px * 1.4f, body_px * 0.5f, Fade(col, 0.45f));
        DrawText("DOWN", (int)p.x - 22, (int)p.y - 8, 16, REK_WHITE);
        return;
    }

    DrawCircleV(p, body_px, Fade(col, 0.85f));

    // Facing.
    Vector2 nose = rek_to_screen(c,
        f->x + cosf(f->yaw) * env->body_radius * 1.7f,
        f->z + sinf(f->yaw) * env->body_radius * 1.7f);
    DrawLineEx(p, nose, 3.0f, REK_WHITE);

    if (f->guard) {
        DrawCircleLines((int)p.x, (int)p.y, body_px + 6.0f, REK_WHITE);
    }

    // Live hit volume, drawn only while the move can actually connect.
    if (rek_committed(f)) {
        const MoveDef* m = &REK_MOVE_TABLE[f->move];
        bool live = f->frame >= m->startup && f->frame < m->startup + m->active;
        float reach = m->reach * env->dr_reach;
        Vector2 h = rek_to_screen(c, f->x + cosf(f->yaw) * reach, f->z + sinf(f->yaw) * reach);
        float hr = m->radius * REK_PIXELS_PER_M;
        if (live) {
            DrawCircleV(h, hr, Fade(REK_WHITE, 0.5f));
        } else {
            DrawCircleLines((int)h.x, (int)h.y, hr, Fade(REK_GREY, 0.8f));
        }
        DrawText(m->name, (int)p.x - 20, (int)p.y - (int)body_px - 20, 14, REK_WHITE);
    }

    // Balance meter above the head — this is what decides knockdowns.
    float bw = 46.0f;
    float by = p.y - body_px - 36.0f;
    DrawRectangle((int)(p.x - bw / 2), (int)by, (int)bw, 5, REK_GREY);
    DrawRectangle((int)(p.x - bw / 2), (int)by, (int)(bw * (f->balance > 1.0f ? 1.0f : f->balance)), 5,
        f->balance > 0.7f ? REK_RED : REK_WHITE);
}

void c_render(Rek* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* c = env->client;

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(REK_BACKGROUND);

    // Ring.
    DrawCircleLines((int)c->cx, (int)c->cy, env->arena_radius * REK_PIXELS_PER_M, REK_GREY);

    rek_draw_fighter(c, env, 0, REK_CYAN);
    rek_draw_fighter(c, env, 1, REK_RED);

    // Scoreboard. REK scores hits and docks a point per down, so show both.
    const Fighter* f0 = &env->fighters[0];
    const Fighter* f1 = &env->fighters[1];
    float secs_left = (float)(env->round_frames - env->tick) / REK_TICK_HZ;
    if (secs_left < 0.0f) secs_left = 0.0f;

    DrawText(TextFormat("%.1f", secs_left), REK_SCREEN_W / 2 - 30, 20, 34, REK_WHITE);
    DrawText(TextFormat("SLOT 0   score %d   hits %d   downs %d/%d",
        rek_score(f0), f0->hits, f0->downs, REK_DOWNS_TO_LOSE), 24, 20, 20, REK_CYAN);
    DrawText(TextFormat("SLOT 1   score %d   hits %d   downs %d/%d",
        rek_score(f1), f1->hits, f1->downs, REK_DOWNS_TO_LOSE),
        24, REK_SCREEN_H - 40, 20, REK_RED);

    EndDrawing();
}
