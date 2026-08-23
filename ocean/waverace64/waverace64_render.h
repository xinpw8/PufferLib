#pragma once

// State-based human evaluator for Wave Race 64. This file is included only
// when PUFFER_WAVERACE64_RENDER is enabled. Training never allocates Client or
// calls any Raylib function.

#include <raylib.h>

#define WR64_RENDER_SCALE 0.01f
#define WR64_WAKE_POINTS 64

struct Client {
    WR64RenderState state;
    WR64RenderState terminal_state;
    uint64_t state_hash;
    int has_state;
    int has_terminal;
    int last_tick;
    int wake_start;
    int wake_count;
    Vector3 wake[WR64_WAKE_POINTS];
};

static inline float wr64_render_clampf(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static inline Vector3 wr64_render_v3(float x, float y, float z) {
    Vector3 result = {x, y, z};
    return result;
}

static inline Vector3 wr64_render_add(Vector3 a, Vector3 b) {
    return wr64_render_v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline Vector3 wr64_render_sub(Vector3 a, Vector3 b) {
    return wr64_render_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline Vector3 wr64_render_mul(Vector3 value, float scale) {
    return wr64_render_v3(value.x * scale, value.y * scale, value.z * scale);
}

static inline float wr64_render_dot(Vector3 a, Vector3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline float wr64_render_length(Vector3 value) {
    return sqrtf(wr64_render_dot(value, value));
}

static inline Vector3 wr64_render_normalize(Vector3 value, Vector3 fallback) {
    float length = wr64_render_length(value);
    if (!isfinite(length) || length < 1e-5f) return fallback;
    return wr64_render_mul(value, 1.f / length);
}

static inline Vector3 wr64_render_world(float x, float y, float z) {
    return wr64_render_v3(
        x * WR64_RENDER_SCALE,
        y * WR64_RENDER_SCALE,
        z * WR64_RENDER_SCALE);
}

static inline Vector3 wr64_render_position(const WR64RenderState* state) {
    return wr64_render_world(
        state->position[0], state->position[1], state->position[2]);
}

static inline Vector3 wr64_render_forward(const WR64RenderState* state) {
    Vector3 forward = wr64_render_v3(
        state->basis[0], state->basis[1], state->basis[2]);
    Vector3 fallback = wr64_render_v3(
        state->heading[0], 0.f, state->heading[1]);
    fallback = wr64_render_normalize(fallback, wr64_render_v3(0.f, 0.f, 1.f));
    forward = wr64_render_normalize(forward, fallback);
    if (forward.x*state->heading[0] + forward.z*state->heading[1] < 0.f) {
        forward = wr64_render_mul(forward, -1.f);
    }
    return forward;
}

static inline Vector3 wr64_render_up(const WR64RenderState* state) {
    return wr64_render_normalize(
        wr64_render_v3(state->basis[3], state->basis[4], state->basis[5]),
        wr64_render_v3(0.f, 1.f, 0.f));
}

static inline Vector3 wr64_render_right(const WR64RenderState* state) {
    Vector3 right = wr64_render_normalize(
        wr64_render_v3(state->basis[6], state->basis[7], state->basis[8]),
        wr64_render_v3(state->heading[1], 0.f, -state->heading[0]));
    Vector3 expected = wr64_render_v3(
        state->heading[1], 0.f, -state->heading[0]);
    if (wr64_render_dot(right, expected) < 0.f) {
        right = wr64_render_mul(right, -1.f);
    }
    return right;
}

static inline Vector3 wr64_render_basis_point(Vector3 center,
        Vector3 right, Vector3 up, Vector3 forward,
        float x, float y, float z) {
    Vector3 result = center;
    result = wr64_render_add(result, wr64_render_mul(right, x));
    result = wr64_render_add(result, wr64_render_mul(up, y));
    result = wr64_render_add(result, wr64_render_mul(forward, z));
    return result;
}

static void wr64_render_triangle_double(
        Vector3 a, Vector3 b, Vector3 c, Color color) {
    DrawTriangle3D(a, b, c, color);
    DrawTriangle3D(c, b, a, color);
}

static void wr64_render_water(const WR64RenderState* state) {
    const int dim = WR64_RENDER_WATER_DIM;
    for (int row = 0; row < dim - 1; row++) {
        for (int col = 0; col < dim - 1; col++) {
            int i00 = row*dim + col;
            int i10 = row*dim + col + 1;
            int i01 = (row + 1)*dim + col;
            int i11 = (row + 1)*dim + col + 1;
            float x0 = state->water_origin_x + (float)col*state->water_spacing;
            float x1 = x0 + state->water_spacing;
            float z0 = state->water_origin_z + (float)row*state->water_spacing;
            float z1 = z0 + state->water_spacing;
            Vector3 p00 = wr64_render_world(x0, state->water[i00], z0);
            Vector3 p10 = wr64_render_world(x1, state->water[i10], z0);
            Vector3 p01 = wr64_render_world(x0, state->water[i01], z1);
            Vector3 p11 = wr64_render_world(x1, state->water[i11], z1);
            float mean = 0.25f * (state->water[i00] + state->water[i10]
                + state->water[i01] + state->water[i11]);
            int lift = (int)wr64_render_clampf(
                (mean - state->water_level) * 1.8f, -24.f, 30.f);
            Color a = {(unsigned char)(19 + lift/3),
                (unsigned char)(126 + lift),
                (unsigned char)(178 + lift), 255};
            Color b = {(unsigned char)(15 + lift/3),
                (unsigned char)(111 + lift),
                (unsigned char)(165 + lift), 255};
            if (((row + col) & 1) == 0) {
                wr64_render_triangle_double(p00, p01, p11, a);
                wr64_render_triangle_double(p00, p11, p10, b);
            } else {
                wr64_render_triangle_double(p00, p01, p10, b);
                wr64_render_triangle_double(p10, p01, p11, a);
            }
        }
    }

    Color grid = {132, 224, 235, 75};
    for (int row = 0; row < dim; row += 4) {
        for (int col = 0; col < dim - 1; col++) {
            int a = row*dim + col;
            int b = a + 1;
            float x0 = state->water_origin_x + (float)col*state->water_spacing;
            float x1 = x0 + state->water_spacing;
            float z = state->water_origin_z + (float)row*state->water_spacing;
            DrawLine3D(wr64_render_world(x0, state->water[a] + 2.f, z),
                wr64_render_world(x1, state->water[b] + 2.f, z), grid);
        }
    }
    for (int col = 0; col < dim; col += 4) {
        for (int row = 0; row < dim - 1; row++) {
            int a = row*dim + col;
            int b = (row + 1)*dim + col;
            float x = state->water_origin_x + (float)col*state->water_spacing;
            float z0 = state->water_origin_z + (float)row*state->water_spacing;
            float z1 = z0 + state->water_spacing;
            DrawLine3D(wr64_render_world(x, state->water[a] + 2.f, z0),
                wr64_render_world(x, state->water[b] + 2.f, z1), grid);
        }
    }
}

static Color wr64_render_buoy_color(int32_t type) {
    if (type == 0) return (Color){220, 49, 55, 255};
    if (type == 1) return (Color){249, 203, 47, 255};
    return (Color){78, 224, 226, 255};
}

static void wr64_render_route(const WR64RenderState* state) {
    for (int i = 0; i < state->node_count; i++) {
        const WR64RenderNode* node = &state->nodes[i];
        if (!node->valid || node->next < 0 || node->next >= state->node_count) {
            continue;
        }
        const WR64RenderNode* next = &state->nodes[node->next];
        if (!next->valid) continue;
        Vector3 a = wr64_render_world(node->live_x, node->live_y + 12.f,
            node->live_z);
        Vector3 b = wr64_render_world(next->live_x, next->live_y + 12.f,
            next->live_z);
        Color line = {132, 244, 235, 125};
        DrawLine3D(a, b, line);
    }

    for (int i = 0; i < state->node_count; i++) {
        const WR64RenderNode* node = &state->nodes[i];
        if (!node->valid) continue;
        if (node->type == 0 || node->type == 1) {
            Color color = wr64_render_buoy_color(node->type);
            Vector3 base = wr64_render_world(
                node->anchor_x, node->live_y - 20.f, node->anchor_z);
            Vector3 top = wr64_render_world(
                node->anchor_x, node->live_y + 145.f, node->anchor_z);
            DrawCylinderEx(base, top, 0.30f, 0.24f, 12, color);
            DrawSphere(top, 0.32f, RAYWHITE);
            DrawSphere(wr64_render_add(top, wr64_render_v3(0.f, 0.03f, 0.f)),
                0.23f, color);

            if (i == state->target_node || i == state->next_node) {
                Vector3 pass = wr64_render_world(
                    node->pass_x, node->live_y + 45.f, node->pass_z);
                DrawCylinderEx(top, pass, 0.045f, 0.045f, 8,
                    (Color){92, 255, 184, 220});
                DrawSphere(pass, i == state->target_node ? 0.34f : 0.22f,
                    (Color){92, 255, 184, 255});
            }
            if (i == state->target_node) {
                Vector3 beacon = wr64_render_add(top, wr64_render_v3(0.f, 4.f, 0.f));
                DrawCylinderEx(top, beacon, 0.055f, 0.015f, 8,
                    (Color){255, 255, 255, 185});
            }
        } else if (node->type == 3) {
            Vector3 lateral = wr64_render_normalize(
                wr64_render_v3(node->lateral_x, 0.f, node->lateral_z),
                wr64_render_v3(1.f, 0.f, 0.f));
            Vector3 center = wr64_render_world(
                node->live_x, node->live_y, node->live_z);
            Vector3 left = wr64_render_add(center, wr64_render_mul(lateral, -2.2f));
            Vector3 right = wr64_render_add(center, wr64_render_mul(lateral, 2.2f));
            Vector3 rise = wr64_render_v3(0.f, 2.2f, 0.f);
            DrawCylinderEx(left, wr64_render_add(left, rise), 0.12f, 0.12f,
                8, RAYWHITE);
            DrawCylinderEx(right, wr64_render_add(right, rise), 0.12f, 0.12f,
                8, RAYWHITE);
            DrawCylinderEx(wr64_render_add(left, rise),
                wr64_render_add(right, rise), 0.10f, 0.10f, 8,
                (Color){35, 42, 61, 255});
        }
    }
}

static void wr64_render_vehicle(const WR64RenderState* state) {
    Vector3 center = wr64_render_position(state);
    Vector3 right = wr64_render_right(state);
    Vector3 up = wr64_render_up(state);
    Vector3 forward = wr64_render_forward(state);
    center = wr64_render_add(center, wr64_render_mul(up, 0.18f));

    Vector3 nose = wr64_render_basis_point(
        center, right, up, forward, 0.f, 0.f, 1.45f);
    Vector3 rear_left = wr64_render_basis_point(
        center, right, up, forward, -0.58f, -0.05f, -1.05f);
    Vector3 rear_right = wr64_render_basis_point(
        center, right, up, forward, 0.58f, -0.05f, -1.05f);
    Vector3 keel = wr64_render_basis_point(
        center, right, up, forward, 0.f, -0.36f, -0.25f);
    Vector3 deck = wr64_render_basis_point(
        center, right, up, forward, 0.f, 0.26f, -0.20f);
    Color hull = {243, 66, 121, 255};
    Color hull_dark = {111, 25, 76, 255};
    wr64_render_triangle_double(nose, rear_left, keel, hull_dark);
    wr64_render_triangle_double(nose, keel, rear_right, hull_dark);
    wr64_render_triangle_double(nose, rear_right, deck, hull);
    wr64_render_triangle_double(nose, deck, rear_left, hull);
    wr64_render_triangle_double(rear_left, deck, rear_right,
        (Color){250, 226, 79, 255});

    Vector3 torso_base = wr64_render_basis_point(
        center, right, up, forward, 0.f, 0.38f, -0.25f);
    Vector3 shoulders = wr64_render_basis_point(
        center, right, up, forward, 0.f, 1.25f, -0.05f);
    DrawCylinderEx(torso_base, shoulders, 0.25f, 0.32f, 10,
        (Color){39, 194, 111, 255});
    Vector3 head = wr64_render_basis_point(
        center, right, up, forward, 0.f, 1.62f, 0.02f);
    DrawSphere(head, 0.28f, (Color){246, 192, 151, 255});
    Vector3 bar_left = wr64_render_basis_point(
        center, right, up, forward, -0.48f, 0.72f, 0.55f);
    Vector3 bar_right = wr64_render_basis_point(
        center, right, up, forward, 0.48f, 0.72f, 0.55f);
    DrawCylinderEx(bar_left, bar_right, 0.045f, 0.045f, 8,
        (Color){34, 36, 48, 255});
    DrawCylinderEx(shoulders, bar_left, 0.07f, 0.06f, 8,
        (Color){246, 192, 151, 255});
    DrawCylinderEx(shoulders, bar_right, 0.07f, 0.06f, 8,
        (Color){246, 192, 151, 255});
}

static void wr64_render_wake(const Client* client) {
    if (client->wake_count < 2) return;
    for (int i = 1; i < client->wake_count; i++) {
        int previous = (client->wake_start + i - 1) % WR64_WAKE_POINTS;
        int current = (client->wake_start + i) % WR64_WAKE_POINTS;
        float alpha = (float)i / (float)client->wake_count;
        Color color = {224, 251, 250, (unsigned char)(40.f + 190.f*alpha)};
        Vector3 a = wr64_render_add(client->wake[previous],
            wr64_render_v3(0.f, 0.06f, 0.f));
        Vector3 b = wr64_render_add(client->wake[current],
            wr64_render_v3(0.f, 0.06f, 0.f));
        DrawCylinderEx(a, b, 0.04f + 0.06f*alpha,
            0.05f + 0.08f*alpha, 6, color);
    }
}

static void wr64_render_minimap(const WR64RenderState* state,
        int x, int y, int width, int height) {
    DrawRectangle(x, y, width, height, (Color){8, 20, 35, 215});
    DrawRectangleLines(x, y, width, height, (Color){112, 226, 230, 180});
    DrawText("COURSE STATE", x + 10, y + 7, 14, (Color){170, 244, 241, 255});
    float min_x = INFINITY;
    float max_x = -INFINITY;
    float min_z = INFINITY;
    float max_z = -INFINITY;
    for (int i = 0; i < state->node_count; i++) {
        const WR64RenderNode* node = &state->nodes[i];
        if (!node->valid) continue;
        min_x = fminf(min_x, node->live_x);
        max_x = fmaxf(max_x, node->live_x);
        min_z = fminf(min_z, node->live_z);
        max_z = fmaxf(max_z, node->live_z);
    }
    if (!isfinite(min_x) || max_x - min_x < 1.f || max_z - min_z < 1.f) return;
    float pad = 12.f;
    float map_w = (float)width - 2.f*pad;
    float map_h = (float)height - 34.f - pad;
    float sx = map_w / (max_x - min_x);
    float sz = map_h / (max_z - min_z);
    float scale = fminf(sx, sz);
    float used_w = (max_x - min_x)*scale;
    float used_h = (max_z - min_z)*scale;
    float ox = (float)x + 0.5f*((float)width - used_w) - min_x*scale;
    float oy = (float)y + 26.f + 0.5f*(map_h - used_h) + max_z*scale;

    for (int i = 0; i < state->node_count; i++) {
        const WR64RenderNode* node = &state->nodes[i];
        if (!node->valid || node->next < 0 || node->next >= state->node_count) {
            continue;
        }
        const WR64RenderNode* next = &state->nodes[node->next];
        if (!next->valid) continue;
        DrawLine((int)(ox + node->live_x*scale),
            (int)(oy - node->live_z*scale),
            (int)(ox + next->live_x*scale),
            (int)(oy - next->live_z*scale),
            (Color){109, 214, 218, 180});
    }
    for (int i = 0; i < state->node_count; i++) {
        const WR64RenderNode* node = &state->nodes[i];
        if (!node->valid || (node->type != 0 && node->type != 1)) continue;
        int px = (int)(ox + node->live_x*scale);
        int py = (int)(oy - node->live_z*scale);
        DrawCircle(px, py, i == state->target_node ? 5.f : 3.f,
            wr64_render_buoy_color(node->type));
        if (i == state->target_node) DrawCircleLines(px, py, 8.f, RAYWHITE);
    }
    int px = (int)(ox + state->position[0]*scale);
    int py = (int)(oy - state->position[2]*scale);
    Vector2 nose = {(float)px + state->heading[0]*8.f,
        (float)py - state->heading[1]*8.f};
    Vector2 left = {(float)px - state->heading[0]*4.f - state->heading[1]*4.f,
        (float)py + state->heading[1]*4.f - state->heading[0]*4.f};
    Vector2 right = {(float)px - state->heading[0]*4.f + state->heading[1]*4.f,
        (float)py + state->heading[1]*4.f + state->heading[0]*4.f};
    DrawTriangle(nose, left, right, (Color){255, 255, 255, 255});
}

static const WR64RenderNode* wr64_render_target_node(
        const WR64RenderState* state) {
    if (state->target_node < 0 || state->target_node >= state->node_count) {
        return NULL;
    }
    return &state->nodes[state->target_node];
}

static void wr64_render_hud(const Client* client) {
    const WR64RenderState* state = &client->state;
    int width = GetScreenWidth();
    int height = GetScreenHeight();
    DrawRectangle(0, 0, width, 38, (Color){5, 15, 27, 235});
    DrawText("WAVE RACE 64", 14, 8, 22, RAYWHITE);
    const char* subtitle = width < 760
        ? "SUNNY BEACH | TIME TRIAL | STATE EVAL"
        : "SUNNY BEACH  |  TIME TRIAL  |  STATE EVALUATOR";
    DrawText(subtitle, width < 760 ? 190 : 194, 11,
        width < 760 ? 12 : 14, (Color){150, 235, 235, 255});

    DrawRectangle(12, 50, 236, 154, (Color){6, 20, 35, 220});
    DrawRectangleLines(12, 50, 236, 154, (Color){112, 226, 230, 180});
    int display_lap = state->lap < 1 ? 1 : state->lap;
    DrawText(TextFormat("LAP %d / %d", display_lap, state->target_laps),
        24, 62, 24, RAYWHITE);
    DrawText(TextFormat("GATE %d    CLEARED %d", state->target_node,
        state->checkpoints), 24, 93, 16, (Color){184, 237, 235, 255});
    DrawText(TextFormat("SPEED %7.1f units/s", state->speed_per_second),
        24, 116, 16, (Color){184, 237, 235, 255});
    DrawText(TextFormat("TIME  %6.2f s", (float)state->tick/(float)WR_GAME_UPDATE_HZ),
        24, 139, 16, (Color){184, 237, 235, 255});
    DrawText("MISSES", 24, 169, 15, (Color){184, 237, 235, 255});
    for (int i = 0; i < 5; i++) {
        int mx = 94 + 25*i;
        Color color = i < state->misses
            ? (Color){247, 78, 88, 255} : (Color){75, 111, 126, 255};
        DrawLine(mx - 6, 171, mx + 6, 183, color);
        DrawLine(mx + 6, 171, mx - 6, 183, color);
    }

    const WR64RenderNode* target = wr64_render_target_node(state);
    DrawRectangle(12, 214, 236, 58, (Color){6, 20, 35, 220});
    DrawRectangleLines(12, 214, 236, 58, (Color){112, 226, 230, 180});
    if (target && target->type == 0) {
        DrawCircle(30, 233, 9.f, wr64_render_buoy_color(0));
        DrawText("RED BUOY", 47, 222, 16, RAYWHITE);
        DrawText("PASS TO ITS RIGHT", 47, 244, 16,
            (Color){92, 255, 184, 255});
    } else if (target && target->type == 1) {
        DrawCircle(30, 233, 9.f, wr64_render_buoy_color(1));
        DrawText("YELLOW BUOY", 47, 222, 16, RAYWHITE);
        DrawText("PASS TO ITS LEFT", 47, 244, 16,
            (Color){92, 255, 184, 255});
    } else {
        DrawText("FOLLOW THE ROUTE MARKER", 24, 234, 15,
            (Color){92, 255, 184, 255});
    }

    int map_w = width < 800 ? 210 : 250;
    int map_h = height < 600 ? 168 : 205;
    wr64_render_minimap(state, width - map_w - 12, 50, map_w, map_h);

    int control_y = height - 54;
    DrawRectangle(0, control_y, width, 54, (Color){5, 15, 27, 235});
    DrawText("HOLD LEFT SHIFT FOR MANUAL CONTROL", 14, control_y + 7, 15,
        (Color){255, 226, 111, 255});
    DrawText("W throttle   A/D steer   arrows lean   S damp waves   SPACE slide",
        14, control_y + 29, 14, (Color){174, 231, 232, 255});
    DrawText(TextFormat("stick %+d,%+d  A%d B%d R%d",
        (int)state->pad_stick_x, (int)state->pad_stick_y,
        (state->pad_buttons & WR_BTN_A) != 0,
        (state->pad_buttons & WR_BTN_B) != 0,
        (state->pad_buttons & WR_BTN_R) != 0),
        width - 260, control_y + 7, 14, RAYWHITE);

    if (state->recovery) {
        const char* text = state->recovery == 2 ? "RECOVERY" : "UNSTABLE";
        int text_width = MeasureText(text, 34);
        DrawRectangle(width/2 - text_width/2 - 16, 54,
            text_width + 32, 48, (Color){209, 67, 70, 225});
        DrawText(text, width/2 - text_width/2, 61, 34, RAYWHITE);
    }
    if (client->has_terminal) {
        const WR64RenderState* terminal = &client->terminal_state;
        const char* label = terminal->success ? "LAST RACE: FINISH"
            : (terminal->disqualified ? "LAST RACE: DISQUALIFIED"
            : "LAST RACE: FAILED");
        Color color = terminal->success
            ? (Color){37, 185, 110, 235} : (Color){204, 62, 71, 235};
        int text_width = MeasureText(label, 17);
        DrawRectangle(width/2 - text_width/2 - 12, height - 86,
            text_width + 24, 26, color);
        DrawText(label, width/2 - text_width/2, height - 82, 17, RAYWHITE);
    }
}

static void wr64_render_human_controls(WaveRace64* env) {
    if (!env->client || !IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) return;
    float* action = env->agents[0].actions;
    int steer = 7;
    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) steer = 0;
    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) steer = 14;
    int lean = 4;
    if (IsKeyDown(KEY_DOWN)) lean = 0;
    if (IsKeyDown(KEY_UP)) lean = 8;
    action[0] = (float)steer;
    action[1] = (float)lean;
    action[2] = IsKeyDown(KEY_W) ? 1.f : 0.f;
    action[3] = IsKeyDown(KEY_S) ? 1.f : 0.f;
    action[4] = IsKeyDown(KEY_SPACE) ? 1.f : 0.f;
}

static void wr64_render_capture_terminal(WaveRace64* env) {
    if (!env->client) return;
    wr64_capture_render_state(env, &env->client->terminal_state);
    env->client->has_terminal = 1;
}

static void wr64_render_append_wake(Client* client,
        const WR64RenderState* state) {
    Vector3 position = wr64_render_position(state);
    if (client->wake_count > 0) {
        int last = (client->wake_start + client->wake_count - 1)
            % WR64_WAKE_POINTS;
        if (wr64_render_length(wr64_render_sub(position, client->wake[last]))
                < 0.08f) return;
    }
    if (client->wake_count < WR64_WAKE_POINTS) {
        int index = (client->wake_start + client->wake_count)
            % WR64_WAKE_POINTS;
        client->wake[index] = position;
        client->wake_count++;
    } else {
        client->wake[client->wake_start] = position;
        client->wake_start = (client->wake_start + 1) % WR64_WAKE_POINTS;
    }
}

static void wr64_render_close(WaveRace64* env) {
    if (!env->client) return;
    if (IsWindowReady()) CloseWindow();
    free(env->client);
    env->client = NULL;
}

static void wr64_render_draw(WaveRace64* env) {
    if (!env->client) {
        env->client = (Client*)calloc(1, sizeof(Client));
        if (!env->client) {
            fprintf(stderr, "[waverace64] renderer allocation failed\n");
            abort();
        }
    }
    if (!IsWindowReady()) {
        int width = 960;
        int height = 540;
        const char* width_env = getenv("WR64_RENDER_WIDTH");
        const char* height_env = getenv("WR64_RENDER_HEIGHT");
        if (width_env && atoi(width_env) >= 640) width = atoi(width_env);
        if (height_env && atoi(height_env) >= 480) height = atoi(height_env);
        InitWindow(width, height, "Wave Race 64 | PufferLib state evaluator");
        SetTargetFPS(60);
    }

    wr64_render_human_controls(env);
    WR64RenderState next;
    wr64_capture_render_state(env, &next);
    uint64_t hash = wr64_render_state_hash(&next);
    Client* client = env->client;
    if (!client->has_state || hash != client->state_hash) {
        if (client->has_state && next.tick <= client->last_tick) {
            client->wake_start = 0;
            client->wake_count = 0;
        }
        if (!client->has_state || next.tick != client->last_tick) {
            wr64_render_append_wake(client, &next);
        }
        client->state = next;
        client->state_hash = hash;
        client->last_tick = next.tick;
        client->has_state = 1;
    }

    const WR64RenderState* state = &client->state;
    Vector3 rider = wr64_render_position(state);
    Vector3 horizontal_forward = wr64_render_normalize(
        wr64_render_v3(state->heading[0], 0.f, state->heading[1]),
        wr64_render_v3(0.f, 0.f, 1.f));
    Camera3D camera = {0};
    camera.position = wr64_render_add(
        wr64_render_sub(rider, wr64_render_mul(horizontal_forward, 9.5f)),
        wr64_render_v3(0.f, 5.2f, 0.f));
    camera.target = wr64_render_add(
        wr64_render_add(rider, wr64_render_mul(horizontal_forward, 4.0f)),
        wr64_render_v3(0.f, 0.7f, 0.f));
    camera.up = wr64_render_v3(0.f, 1.f, 0.f);
    camera.fovy = 52.f;
    camera.projection = CAMERA_PERSPECTIVE;

    BeginDrawing();
    ClearBackground((Color){94, 196, 226, 255});
    DrawCircle(GetScreenWidth()/2 + 30, 76, 38.f,
        (Color){255, 235, 141, 255});
    BeginMode3D(camera);
    wr64_render_water(state);
    wr64_render_route(state);
    wr64_render_wake(client);
    wr64_render_vehicle(state);
    EndMode3D();
    wr64_render_hud(client);
    EndDrawing();
}
