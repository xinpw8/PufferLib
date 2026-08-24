#pragma once

// State-based human evaluator for Wave Race 64. This file is included only
// when PUFFER_WAVERACE64_RENDER is enabled. Training never allocates Client or
// calls any Raylib function.

#include <raylib.h>
#include <rlgl.h>

#define WR64_RENDER_SCALE 0.01f
#define WR64_WAKE_POINTS 64
#define WR64_CAMERA_TAU_SECONDS 0.60f
#define WR64_PUFFER_MODEL_PATH "resources/shared/puffer.glb"
#define WR64_PUFFER_SCALE 260.0f
#define WR64_BUOY_ARROW_BLINK_TICKS ((int)WR_GAME_UPDATE_HZ / 4)

struct Client {
    WR64RenderState state;
    WR64RenderState terminal_state;
    uint64_t state_hash;
    int has_state;
    int has_terminal;
    int human_control;
    int toggle_chord_down;
    int terminal_hold;
    int last_tick;
    int camera_ready;
    int camera_tick;
    float camera_y;
    int hud_lap;
    int final_lap_until_tick;
    int wake_start;
    int wake_count;
    Vector3 wake[WR64_WAKE_POINTS];
    Model puffer;
    int puffer_loaded;
};

static inline float wr64_render_clampf(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static void wr64_render_update_control_mode(Client* client, int chord_down) {
    chord_down = chord_down != 0;
    if (chord_down && !client->toggle_chord_down) {
        client->human_control = !client->human_control;
    }
    client->toggle_chord_down = chord_down;
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

static inline Vector3 wr64_render_cross(Vector3 a, Vector3 b) {
    return wr64_render_v3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x);
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

static void wr64_render_update_camera_anchor(Client* client,
        const WR64RenderState* state) {
    float rider_y = state->position[1] * WR64_RENDER_SCALE;
    int tick_delta = state->tick - client->camera_tick;
    int snap = !client->camera_ready || tick_delta <= 0
        || tick_delta > WR_GAME_UPDATE_HZ || state->recovery != 0;
    if (snap) {
        client->camera_y = rider_y;
        client->camera_ready = 1;
    } else {
        float seconds = (float)tick_delta / (float)WR_GAME_UPDATE_HZ;
        float alpha = 1.f - expf(-seconds / WR64_CAMERA_TAU_SECONDS);
        client->camera_y += alpha * (rider_y - client->camera_y);
    }
    client->camera_tick = state->tick;
}

static void wr64_render_update_hud_state(Client* client,
        const WR64RenderState* state) {
    if (client->hud_lap >= 0
            && client->hud_lap < state->target_laps
            && state->lap >= state->target_laps) {
        client->final_lap_until_tick = state->tick
            + 3 * WR_GAME_UPDATE_HZ;
    }
    client->hud_lap = state->lap;
}

static inline Vector3 wr64_render_position(const WR64RenderState* state) {
    return wr64_render_world(
        state->position[0], state->position[1], state->position[2]);
}

static Camera3D wr64_render_camera(const Client* client,
        const WR64RenderState* state) {
    Vector3 rider = wr64_render_position(state);
    Vector3 horizontal_forward = wr64_render_normalize(
        wr64_render_v3(state->heading[0], 0.f, state->heading[1]),
        wr64_render_v3(0.f, 0.f, 1.f));
    float camera_y = client->camera_ready ? client->camera_y : rider.y;
    Vector3 anchor = wr64_render_v3(rider.x, camera_y, rider.z);
    Camera3D camera = {0};
    camera.position = wr64_render_add(
        wr64_render_sub(anchor, wr64_render_mul(horizontal_forward, 9.5f)),
        wr64_render_v3(0.f, 5.2f, 0.f));
    camera.target = wr64_render_add(
        wr64_render_add(anchor, wr64_render_mul(horizontal_forward, 4.0f)),
        wr64_render_v3(0.f, 0.7f, 0.f));
    camera.up = wr64_render_v3(0.f, 1.f, 0.f);
    camera.fovy = 52.f;
    camera.projection = CAMERA_PERSPECTIVE;
    return camera;
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

static void wr64_render_triangle_double(
        Vector3 a, Vector3 b, Vector3 c, Color color) {
    DrawTriangle3D(a, b, c, color);
    DrawTriangle3D(c, b, a, color);
}

static void wr64_render_quad_double(
        Vector3 a, Vector3 b, Vector3 c, Vector3 d, Color color) {
    wr64_render_triangle_double(a, b, c, color);
    wr64_render_triangle_double(a, c, d, color);
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

static inline int wr64_render_buoy_arrow_visible(
        const WR64RenderState* state, int32_t node_index) {
    if (node_index != state->target_node) return 1;
    const int half_period = WR64_BUOY_ARROW_BLINK_TICKS > 0
        ? WR64_BUOY_ARROW_BLINK_TICKS : 1;
    const int32_t tick = state->tick > 0 ? state->tick : 0;
    return (tick / half_period) % 2 == 0;
}

static inline Vector3 wr64_render_buoy_arrow_direction(
        const WR64RenderNode* node) {
    const float side = node->type == 0 ? -1.f : 1.f;
    Vector3 fallback = wr64_render_normalize(
        wr64_render_v3(side * node->lateral_x, 0.f,
            side * node->lateral_z),
        wr64_render_v3(side, 0.f, 0.f));
    return wr64_render_normalize(
        wr64_render_v3(node->pass_x - node->live_x, 0.f,
            node->pass_z - node->live_z),
        fallback);
}

static inline Vector3 wr64_render_buoy_arrow_axis(
        const WR64RenderNode* node) {
    Vector3 pass = wr64_render_buoy_arrow_direction(node);
    return wr64_render_normalize(
        wr64_render_add(pass, wr64_render_v3(0.f, -1.f, 0.f)),
        wr64_render_v3(0.70710678f, -0.70710678f, 0.f));
}

static inline void wr64_render_buoy_arrow_geometry(
        const WR64RenderNode* node, Vector3 buoy_top,
        Vector3 front[7], Vector3 back[7]) {
    const Vector3 pass = wr64_render_buoy_arrow_direction(node);
    const Vector3 axis = wr64_render_buoy_arrow_axis(node);
    const Vector3 across = wr64_render_normalize(
        wr64_render_add(pass, wr64_render_v3(0.f, 1.f, 0.f)),
        wr64_render_v3(0.70710678f, 0.70710678f, 0.f));
    const Vector3 tangent = wr64_render_normalize(
        wr64_render_v3(node->tangent_x, 0.f, node->tangent_z),
        wr64_render_v3(pass.z, 0.f, -pass.x));
    const Vector3 center = wr64_render_add(
        wr64_render_add(buoy_top, wr64_render_v3(0.f, 0.72f, 0.f)),
        wr64_render_mul(pass, 0.10f));
    const float along[7] = {
        -0.675f, 0.125f, 0.125f, 0.675f, 0.125f, 0.125f, -0.675f};
    const float width[7] = {
        -0.170f, -0.170f, -0.350f, 0.f, 0.350f, 0.170f, 0.170f};
    for (int i = 0; i < 7; i++) {
        Vector3 point = wr64_render_add(center,
            wr64_render_add(wr64_render_mul(axis, along[i]),
                wr64_render_mul(across, width[i])));
        front[i] = wr64_render_add(point, wr64_render_mul(tangent, 0.08f));
        back[i] = wr64_render_add(point, wr64_render_mul(tangent, -0.08f));
    }
}

static void wr64_render_buoy_arrow(const WR64RenderNode* node,
        Vector3 buoy_top, Color color) {
    Vector3 front[7];
    Vector3 back[7];
    wr64_render_buoy_arrow_geometry(node, buoy_top, front, back);
    wr64_render_triangle_double(front[0], front[1], front[5], color);
    wr64_render_triangle_double(front[0], front[5], front[6], color);
    wr64_render_triangle_double(front[2], front[3], front[4], color);
    wr64_render_triangle_double(back[0], back[5], back[1], color);
    wr64_render_triangle_double(back[0], back[6], back[5], color);
    wr64_render_triangle_double(back[2], back[4], back[3], color);

    Color side = node->type == 0
        ? (Color){126, 25, 31, 255} : (Color){171, 117, 18, 255};
    for (int i = 0; i < 7; i++) {
        int next = (i + 1) % 7;
        wr64_render_quad_double(
            front[i], front[next], back[next], back[i], side);
        DrawCylinderEx(front[i], front[next],
            0.025f, 0.025f, 6, side);
    }
}

static void wr64_render_buoy_letter_stroke(
        Vector3 center, Vector3 right, Vector3 up,
        float x0, float y0, float x1, float y1) {
    Vector3 a = wr64_render_add(center,
        wr64_render_add(wr64_render_mul(right, x0),
            wr64_render_mul(up, y0)));
    Vector3 b = wr64_render_add(center,
        wr64_render_add(wr64_render_mul(right, x1),
            wr64_render_mul(up, y1)));
    DrawCylinderEx(a, b, 0.047f, 0.047f, 7, (Color){18, 19, 22, 255});
}

static void wr64_render_buoy_letter(
        Vector3 center, Vector3 right, int32_t type) {
    const Vector3 up = wr64_render_v3(0.f, 1.f, 0.f);
    if (type == 0) {
        wr64_render_buoy_letter_stroke(
            center, right, up, -0.16f, -0.25f, -0.16f, 0.25f);
        wr64_render_buoy_letter_stroke(
            center, right, up, -0.16f, 0.25f, 0.11f, 0.25f);
        wr64_render_buoy_letter_stroke(
            center, right, up, 0.11f, 0.25f, 0.13f, 0.02f);
        wr64_render_buoy_letter_stroke(
            center, right, up, 0.13f, 0.02f, -0.16f, 0.02f);
        wr64_render_buoy_letter_stroke(
            center, right, up, -0.02f, 0.02f, 0.18f, -0.25f);
    } else {
        wr64_render_buoy_letter_stroke(
            center, right, up, -0.14f, 0.25f, -0.14f, -0.25f);
        wr64_render_buoy_letter_stroke(
            center, right, up, -0.14f, -0.25f, 0.18f, -0.25f);
    }
}

static Vector3 wr64_render_buoy_body(
        const WR64RenderNode* node, Color color) {
    const Vector3 up = wr64_render_v3(0.f, 1.f, 0.f);
    const Vector3 pass = wr64_render_buoy_arrow_direction(node);
    const Vector3 tangent = wr64_render_normalize(
        wr64_render_v3(node->tangent_x, 0.f, node->tangent_z),
        wr64_render_v3(pass.z, 0.f, -pass.x));
    const Vector3 center = wr64_render_add(
        wr64_render_world(node->anchor_x, node->live_y, node->anchor_z),
        wr64_render_v3(0.f, 0.28f, 0.f));
    const Vector3 tail_root = wr64_render_add(
        wr64_render_sub(center, wr64_render_mul(pass, 0.28f)),
        wr64_render_v3(0.f, -0.10f, 0.f));
    const Vector3 tail_tip = wr64_render_add(
        wr64_render_sub(center, wr64_render_mul(pass, 1.15f)),
        wr64_render_v3(0.f, -0.10f, 0.f));
    const Color dark = {23, 24, 27, 255};
    for (int segment = 0; segment < 4; segment++) {
        float t0 = 0.25f * (float)segment;
        float t1 = 0.25f * (float)(segment + 1);
        Vector3 a = wr64_render_add(
            tail_root, wr64_render_mul(wr64_render_sub(tail_tip, tail_root), t0));
        Vector3 b = wr64_render_add(
            tail_root, wr64_render_mul(wr64_render_sub(tail_tip, tail_root), t1));
        float radius_a = 0.34f + (0.10f - 0.34f) * t0;
        float radius_b = 0.34f + (0.10f - 0.34f) * t1;
        DrawCylinderEx(a, b, radius_a, radius_b, 10,
            (segment & 1) ? color : dark);
    }

    DrawCylinderEx(
        wr64_render_sub(center, wr64_render_mul(tangent, 0.17f)),
        wr64_render_add(center, wr64_render_mul(tangent, 0.17f)),
        0.52f, 0.52f, 18, color);
    DrawSphere(center, 0.50f, color);

    Vector3 incoming = wr64_render_mul(tangent, -1.f);
    Vector3 face_right = wr64_render_normalize(
        wr64_render_cross(incoming, up), pass);
    wr64_render_buoy_letter(
        wr64_render_add(center, wr64_render_mul(incoming, 0.505f)),
        face_right, node->type);
    Vector3 outgoing = tangent;
    Vector3 back_right = wr64_render_normalize(
        wr64_render_cross(outgoing, up), wr64_render_mul(pass, -1.f));
    wr64_render_buoy_letter(
        wr64_render_add(center, wr64_render_mul(outgoing, 0.505f)),
        back_right, node->type);
    return wr64_render_add(center, wr64_render_mul(up, 0.52f));
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
            Vector3 top = wr64_render_buoy_body(node, color);

            if (wr64_render_buoy_arrow_visible(state, i)) {
                wr64_render_buoy_arrow(node, top, color);
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

static void wr64_render_puffer(const Client* client,
        const WR64RenderState* state) {
    if (!client->puffer_loaded) return;

    Vector3 center = wr64_render_position(state);
    Vector3 right = wr64_render_right(state);
    Vector3 up = wr64_render_up(state);
    Vector3 forward = wr64_render_forward(state);
    center = wr64_render_add(center, wr64_render_mul(forward, 0.24f));
    center = wr64_render_add(center, wr64_render_mul(up, -0.10f));

    // After the Tower Climb root correction below, the shared Puffer model's
    // local axes are +X right, +Y up, and +Z nose. Map those axes onto the
    // authoritative vehicle basis.
    const float basis[16] = {
        right.x, right.y, right.z, 0.f,
        up.x, up.y, up.z, 0.f,
        forward.x, forward.y, forward.z, 0.f,
        0.f, 0.f, 0.f, 1.f,
    };
    rlPushMatrix();
    rlTranslatef(center.x, center.y, center.z);
    rlMultMatrixf(basis);
    // Tower Climb applies the same correction to undo the GLB root rotation.
    rlRotatef(-90.f, 0.f, 0.f, 1.f);
    rlScalef(WR64_PUFFER_SCALE, WR64_PUFFER_SCALE, WR64_PUFFER_SCALE);
    DrawModel(client->puffer, wr64_render_v3(0.f, 0.f, 0.f), 1.f, WHITE);
    rlPopMatrix();
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
    DrawRectangle(x, y, width, height, (Color){8, 20, 35, 112});
    DrawRectangleLines(x, y, width, height, (Color){235, 252, 250, 145});
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
    float pad = 7.f;
    float map_w = (float)width - 2.f*pad;
    float map_h = (float)height - 2.f*pad;
    float sx = map_w / (max_x - min_x);
    float sz = map_h / (max_z - min_z);
    float scale = fminf(sx, sz);
    float used_w = (max_x - min_x)*scale;
    float used_h = (max_z - min_z)*scale;
    float ox = (float)x + 0.5f*((float)width - used_w) - min_x*scale;
    float oy = (float)y + pad + 0.5f*(map_h - used_h) + max_z*scale;

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
        DrawCircle(px, py, i == state->target_node ? 3.5f : 2.f,
            wr64_render_buoy_color(node->type));
        if (i == state->target_node) DrawCircleLines(px, py, 5.5f, RAYWHITE);
    }
    int px = (int)(ox + state->position[0]*scale);
    int py = (int)(oy - state->position[2]*scale);
    Vector2 nose = {(float)px + state->heading[0]*6.f,
        (float)py - state->heading[1]*6.f};
    Vector2 left = {(float)px - state->heading[0]*3.f - state->heading[1]*3.f,
        (float)py + state->heading[1]*3.f - state->heading[0]*3.f};
    Vector2 right = {(float)px - state->heading[0]*3.f + state->heading[1]*3.f,
        (float)py + state->heading[1]*3.f + state->heading[0]*3.f};
    DrawTriangle(nose, left, right, (Color){255, 255, 255, 255});
}

static const WR64RenderNode* wr64_render_target_node(
        const WR64RenderState* state) {
    if (state->target_node < 0 || state->target_node >= state->node_count) {
        return NULL;
    }
    return &state->nodes[state->target_node];
}

static void wr64_render_format_time(char* output, size_t size, int total_ms) {
    if (total_ms < 0) total_ms = 0;
    int minutes = total_ms / 60000;
    int seconds = (total_ms / 1000) % 60;
    int millis = total_ms % 1000;
    snprintf(output, size, "%d'%02d\"%03d", minutes, seconds, millis);
}

static void wr64_render_draw_outlined(const char* text,
        int x, int y, int font_size, Color color) {
    Color edge = {5, 12, 18, 235};
    DrawText(text, x - 1, y, font_size, edge);
    DrawText(text, x + 1, y, font_size, edge);
    DrawText(text, x, y - 1, font_size, edge);
    DrawText(text, x, y + 1, font_size, edge);
    DrawText(text, x, y, font_size, color);
}

static void wr64_render_draw_centered(const char* text,
        int center_x, int y, int font_size, Color color) {
    wr64_render_draw_outlined(text,
        center_x - MeasureText(text, font_size)/2, y, font_size, color);
}

static void wr64_render_hud(const Client* client) {
    const WR64RenderState* state = &client->state;
    int width = GetScreenWidth();
    int height = GetScreenHeight();
    int label_size = width < 600 ? 10 : 12;
    int value_size = width < 600 ? 17 : 20;
    Color yellow = {250, 232, 38, 255};
    Color orange = {255, 139, 24, 255};
    Color green = {78, 236, 94, 255};
    Color secondary = {213, 244, 242, 255};

    char time_text[32];
    wr64_render_format_time(time_text, sizeof(time_text), state->race_time_ms);
    wr64_render_draw_outlined("TIME", 10, 7, label_size, yellow);
    wr64_render_draw_outlined(time_text, 10, 20, value_size, orange);

    int display_lap = state->lap < 1 ? 1 : state->lap;
    char lap_text[24];
    snprintf(lap_text, sizeof(lap_text), "%d / %d",
        display_lap, state->target_laps);
    wr64_render_draw_centered("LAP", width/2, 7, label_size, yellow);
    wr64_render_draw_centered(lap_text, width/2, 20, value_size, green);

    char speed_text[24];
    snprintf(speed_text, sizeof(speed_text), "%d", state->speed_kmh);
    int speed_center = width - (width < 600 ? 48 : 56);
    wr64_render_draw_centered("SPEED", speed_center, 7, label_size, yellow);
    int speed_width = MeasureText(speed_text, value_size);
    int speed_x = speed_center - (speed_width + 28)/2;
    wr64_render_draw_outlined(speed_text, speed_x, 20, value_size, orange);
    wr64_render_draw_outlined("km/h", speed_x + speed_width + 3,
        27, 8, RAYWHITE);

    int split_y = 44;
    for (int lap = 0; lap < 3; lap++) {
        if (state->lap_splits_ms[lap] <= 0) continue;
        char split_time[32];
        char split_text[40];
        wr64_render_format_time(split_time, sizeof(split_time),
            state->lap_splits_ms[lap]);
        snprintf(split_text, sizeof(split_text), "%s  L%d",
            split_time, lap + 1);
        wr64_render_draw_outlined(split_text, 10, split_y, 9,
            lap == 1 ? (Color){255, 75, 62, 255} : secondary);
        split_y += 11;
    }
    const char* mode = client->human_control ? "HUMAN" : "POLICY";
    Color mode_color = client->human_control
        ? (Color){225, 146, 38, 235} : (Color){26, 155, 174, 235};

    const WR64RenderNode* target = wr64_render_target_node(state);
    char gate_text[64];
    if (target && target->type == 0) {
        snprintf(gate_text, sizeof(gate_text),
            "GATE %d  RED: PASS RIGHT", state->target_node);
    } else if (target && target->type == 1) {
        snprintf(gate_text, sizeof(gate_text),
            "GATE %d  YELLOW: PASS LEFT", state->target_node);
    } else {
        snprintf(gate_text, sizeof(gate_text),
            "GATE %d  FOLLOW ROUTE", state->target_node);
    }
    wr64_render_draw_centered(gate_text, width/2, 58, 9, green);

    int map_w = width < 760 ? 94 : 106;
    int map_h = width < 760 ? 68 : 78;
    wr64_render_minimap(state, width - map_w - 8, 62, map_w, map_h);

    int miss_y = height - 18;
    wr64_render_draw_outlined("MISS", 10, miss_y - 2, 10, RAYWHITE);
    int remaining = 5 - state->misses;
    if (remaining < 0) remaining = 0;
    if (remaining > 5) remaining = 5;
    for (int i = 0; i < 5; i++) {
        int x = 51 + 13*i;
        Color marker = i < remaining
            ? (Color){238, 54, 42, 255} : (Color){70, 73, 82, 220};
        DrawCircle(x, miss_y + 2, 4.5f, (Color){4, 9, 15, 235});
        DrawCircle(x, miss_y + 1, 3.5f, marker);
        if (i >= remaining) {
            DrawLine(x - 2, miss_y - 1, x + 2, miss_y + 3, RAYWHITE);
        }
    }

    int power = state->power;
    if (power < 0) power = 0;
    if (power > 5) power = 5;
    wr64_render_draw_centered(power == 5 ? "MAX POWER" : "POWER",
        width/2, height - 34, power == 5 ? 13 : 10,
        power == 5 ? yellow : RAYWHITE);
    static const Color power_colors[5] = {
        {90, 126, 255, 255}, {69, 220, 142, 255},
        {232, 230, 57, 255}, {255, 163, 42, 255},
        {244, 70, 42, 255},
    };
    int power_x = width/2 - 31;
    for (int i = 0; i < 5; i++) {
        Color color = i < power
            ? power_colors[i] : (Color){66, 74, 82, 210};
        DrawRectangle(power_x + 13*i, height - 17, 10, 6, color);
        DrawTriangle(
            (Vector2){(float)(power_x + 10 + 13*i), (float)(height - 17)},
            (Vector2){(float)(power_x + 13 + 13*i), (float)(height - 14)},
            (Vector2){(float)(power_x + 10 + 13*i), (float)(height - 11)},
            color);
    }

    int mode_width = MeasureText(mode, 9) + 14;
    int mode_x = width - mode_width - 8;
    DrawRectangle(mode_x, height - 24, mode_width, 16, mode_color);
    wr64_render_draw_centered(mode, mode_x + mode_width/2,
        height - 21, 9, RAYWHITE);

    if (client->human_control) {
        wr64_render_draw_centered(
            "SHIFT+UP policy | W throttle | A/D steer | UP/DOWN lean | DOWN+steer quick turn | S damp waves | SPACE slide",
            width/2, height - 49, width < 600 ? 7 : 8, secondary);
    }

    if (state->recovery) {
        const char* text = state->recovery == 2 ? "RECOVERY" : "UNSTABLE";
        wr64_render_draw_centered(text, width/2, 70, 15,
            (Color){255, 82, 75, 255});
    }
    if (!client->terminal_hold && client->final_lap_until_tick > 0
            && client->final_lap_until_tick >= state->tick) {
        wr64_render_draw_centered("FINAL LAP", width/2, 79,
            width < 600 ? 23 : 28, green);
    }
    if (client->terminal_hold && client->has_terminal) {
        const WR64RenderState* terminal = &client->terminal_state;
        const char* label = terminal->success ? "FINISH"
            : (terminal->disqualified ? "DISQUALIFIED" : "RACE FAILED");
        const char* detail = terminal->success
            ? "Official three-lap finish" : "Episode ended";
        Color color = terminal->success
            ? (Color){37, 185, 110, 235} : (Color){204, 62, 71, 235};
        int box_width = 210;
        int box_x = width/2 - box_width/2;
        int box_y = height/2 - 30;
        int text_width = MeasureText(label, 20);
        DrawRectangle(box_x, box_y, box_width, 60, color);
        DrawRectangleLines(box_x, box_y, box_width, 60, RAYWHITE);
        wr64_render_draw_outlined(label,
            width/2 - text_width/2, box_y + 5, 20, RAYWHITE);
        int detail_width = MeasureText(detail, 11);
        DrawText(detail, width/2 - detail_width/2, box_y + 29, 11, RAYWHITE);
        const char* restart = "ENTER: new race";
        int restart_width = MeasureText(restart, 10);
        DrawText(restart, width/2 - restart_width/2,
            box_y + 45, 10, RAYWHITE);
    }
}

static void wr64_render_poll_control_mode(WaveRace64* env) {
    if (!env->client || !IsWindowReady()) return;
    int shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    wr64_render_update_control_mode(
        env->client, shift && IsKeyDown(KEY_UP));
}

static void wr64_render_human_controls(WaveRace64* env) {
    if (!env->client || !IsWindowReady() || !env->client->human_control) return;
    float* action = env->agents[0].actions;
    int steer = 7;
    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) steer = 0;
    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) steer = 14;
    int lean = 4;
    if (IsKeyDown(KEY_DOWN)) lean = 0;
    if (IsKeyDown(KEY_UP) && !env->client->toggle_chord_down) lean = 8;
    action[0] = (float)steer;
    action[1] = (float)lean;
    action[2] = IsKeyDown(KEY_W) ? 1.f : 0.f;
    action[3] = IsKeyDown(KEY_S) ? 1.f : 0.f;
    action[4] = IsKeyDown(KEY_SPACE) ? 1.f : 0.f;
}

static void wr64_render_capture_terminal(WaveRace64* env) {
    if (!env->client) return;
    Client* client = env->client;
    wr64_capture_render_state(env, &client->terminal_state);
    wr64_render_update_camera_anchor(client, &client->terminal_state);
    wr64_render_update_hud_state(client, &client->terminal_state);
    client->state = client->terminal_state;
    client->state_hash = wr64_render_state_hash(&client->state);
    client->last_tick = client->state.tick;
    client->has_state = 1;
    client->has_terminal = 1;
    client->terminal_hold = 1;
}

static void wr64_render_reset_episode(WaveRace64* env) {
    if (!env->client) return;
    env->client->has_terminal = 0;
    env->client->terminal_hold = 0;
    env->client->camera_ready = 0;
    env->client->camera_tick = 0;
    env->client->hud_lap = -1;
    env->client->final_lap_until_tick = -1;
    env->client->wake_start = 0;
    env->client->wake_count = 0;
}

static int wr64_render_is_paused(const WaveRace64* env) {
    return env->client && env->client->terminal_hold;
}

static int wr64_render_terminal_ready(const WaveRace64* env) {
    return env->client && env->client->terminal_hold
        && env->client->has_terminal;
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
    if (env->client->puffer_loaded && IsWindowReady()) {
        UnloadModel(env->client->puffer);
        env->client->puffer_loaded = 0;
    }
    if (IsWindowReady()) CloseWindow();
    free(env->client);
    env->client = NULL;
}

static void wr64_render_load_puffer(Client* client) {
    if (!FileExists(WR64_PUFFER_MODEL_PATH)) {
        fprintf(stderr, "[waverace64] missing Puffer model: %s\n",
            WR64_PUFFER_MODEL_PATH);
        abort();
    }
    client->puffer = LoadModel(WR64_PUFFER_MODEL_PATH);
    if (!IsModelValid(client->puffer)) {
        fprintf(stderr, "[waverace64] invalid Puffer model: %s\n",
            WR64_PUFFER_MODEL_PATH);
        abort();
    }
    client->puffer_loaded = 1;
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
        int width = 800;
        int height = 450;
        const char* width_env = getenv("WR64_RENDER_WIDTH");
        const char* height_env = getenv("WR64_RENDER_HEIGHT");
        if (width_env && atoi(width_env) >= 480) width = atoi(width_env);
        if (height_env && atoi(height_env) >= 270) height = atoi(height_env);
        InitWindow(width, height, "Wave Race 64 | PufferLib state evaluator");
        SetTargetFPS(60);
    }

    Client* client = env->client;
    if (!client->puffer_loaded) wr64_render_load_puffer(client);
    wr64_render_poll_control_mode(env);
    if (client->terminal_hold && IsKeyPressed(KEY_ENTER)) {
        wr64_render_reset_episode(env);
    }
    wr64_render_human_controls(env);
    if (!client->terminal_hold) {
        WR64RenderState next;
        wr64_capture_render_state(env, &next);
        uint64_t hash = wr64_render_state_hash(&next);
        if (!client->has_state || hash != client->state_hash) {
            if (client->has_state && next.tick <= client->last_tick) {
                client->wake_start = 0;
                client->wake_count = 0;
            }
            if (!client->has_state || next.tick != client->last_tick) {
                wr64_render_append_wake(client, &next);
            }
            wr64_render_update_camera_anchor(client, &next);
            wr64_render_update_hud_state(client, &next);
            client->state = next;
            client->state_hash = hash;
            client->last_tick = next.tick;
            client->has_state = 1;
        }
    }

    const WR64RenderState* state = &client->state;
    Camera3D camera = wr64_render_camera(client, state);

    BeginDrawing();
    ClearBackground((Color){94, 196, 226, 255});
    DrawCircle(GetScreenWidth()/2 + 30, 76, 38.f,
        (Color){255, 235, 141, 255});
    BeginMode3D(camera);
    wr64_render_water(state);
    wr64_render_route(state);
    wr64_render_wake(client);
    wr64_render_puffer(client, state);
    EndMode3D();
    wr64_render_hud(client);
    EndDrawing();
}
