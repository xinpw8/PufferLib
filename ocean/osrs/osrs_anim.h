#ifndef OSRS_ANIM_H
#define OSRS_ANIM_H

#include "osrs_assets.h"
#include "osrs_binary_io.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANIM2_MAGIC 0x324D4E41
#define ANIM_FORMAT_VERSION_MIN 2
#define ANIM_FORMAT_VERSION_MAX 3
#define ANIM_HEADER_SIZE_V2 24
#define ANIM_MAX_SLOTS 256
#define ANIM_MAX_LABELS 256
#define ANIM_SINE_COUNT 2048
#define ANIM_MAX_BASES 65535
#define ANIM_MAX_SEQUENCES 65535

static int anim_sine[ANIM_SINE_COUNT];
static int anim_cosine[ANIM_SINE_COUNT];
static int anim_trig_initialized = 0;

static void anim_init_trig(void) {
    if (anim_trig_initialized) return;
    for (int i = 0; i < ANIM_SINE_COUNT; i++) {
        double angle = (double)i * (2.0 * 3.14159265358979323846 / ANIM_SINE_COUNT);
        anim_sine[i] = (int)(65536.0 * sin(angle));
        anim_cosine[i] = (int)(65536.0 * cos(angle));
    }
    anim_trig_initialized = 1;
}

typedef struct {
    uint16_t base_id;
    uint8_t  slot_count;
    uint8_t* types;
    uint8_t* map_lengths;
    uint8_t** frame_maps;
} AnimFrameBase;

typedef struct {
    uint8_t  slot_index;
    int16_t  dx, dy, dz;
} AnimTransform;

typedef enum {
    ANIM_FRAME_LEGACY = 0,
    ANIM_FRAME_MAYA_BAKED = 1,
} AnimFrameKind;

typedef struct {
    uint8_t        kind;
    uint16_t       framebase_id;
    uint8_t        transform_count;
    AnimTransform* transforms;
    uint16_t       maya_vertex_count;
    int16_t*       maya_vertices;
} AnimFrameData;

typedef struct {
    uint16_t delay;
    AnimFrameData frame;
} AnimSequenceFrame;

typedef struct {
    uint16_t           seq_id;
    uint16_t           frame_count;
    uint8_t            interleave_count;
    uint8_t*           interleave_order;
    int8_t             walk_flag;
    AnimSequenceFrame* frames;
} AnimSequence;

typedef enum {
    ANIM_PLAY_LOOP = 0,
    ANIM_PLAY_ONCE = 1,
} AnimPlaybackMode;

typedef struct {
    int           seq_id;
    AnimSequence* sequence;
    int           model_vert_count;
    int           frame_idx;
    int           ticks_in_frame;
    int           completed_loops;
    AnimPlaybackMode mode;
} AnimPlayback;

static inline void anim_playback_reset(AnimPlayback* pb) {
    pb->seq_id = -1;
    pb->sequence = NULL;
    pb->model_vert_count = -1;
    pb->frame_idx = 0;
    pb->ticks_in_frame = 0;
    pb->completed_loops = 0;
    pb->mode = ANIM_PLAY_LOOP;
}

static inline void anim_playback_restart(
    AnimPlayback* pb, int seq_id, AnimPlaybackMode mode
) {
    pb->seq_id = seq_id;
    pb->mode = mode;
    pb->sequence = NULL;
    pb->frame_idx = 0;
    pb->ticks_in_frame = 0;
    pb->completed_loops = 0;
}

static inline void anim_playback_set_seq(
    AnimPlayback* pb, int seq_id, AnimPlaybackMode mode
) {
    if (pb->seq_id == seq_id && pb->mode == mode) return;
    anim_playback_restart(pb, seq_id, mode);
}

static inline void anim_playback_advance(AnimPlayback* pb) {
    AnimSequence* seq = pb->sequence;
    if (!seq || seq->frame_count <= 0) return;
    int fidx = pb->frame_idx % seq->frame_count;
    int delay = seq->frames[fidx].delay > 0 ? seq->frames[fidx].delay : 1;
    pb->ticks_in_frame++;
    if (pb->ticks_in_frame >= delay) {
        pb->ticks_in_frame = 0;
        int next = (fidx + 1) % seq->frame_count;
        pb->frame_idx = next;
        if (next == 0) pb->completed_loops++;
    }
}

typedef struct {
    AnimFrameBase* bases;
    int            base_count;
    uint16_t*      base_ids;

    AnimSequence*  sequences;
    int            seq_count;
} AnimCache;

typedef struct {
    int16_t* verts;
    int      vert_count;

    int**    groups;
    int*     group_counts;

    uint8_t* base_face_alphas;
    uint8_t* face_alphas;
    int      face_count;
    int**    face_alpha_groups;
    int*     face_alpha_group_counts;
} AnimModelState;

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    const char* path;
} AnimReader;

static void anim_reader_need(AnimReader* r, size_t n) {
    if ((size_t)(r->end - r->p) < n) {
        fprintf(stderr, "anim_cache_load: truncated %s\n", r->path);
        abort();
    }
}

static uint8_t anim_read_u8(AnimReader* r) {
    anim_reader_need(r, 1);
    uint8_t v = r->p[0];
    r->p++;
    return v;
}

static int8_t anim_read_i8(AnimReader* r) {
    return (int8_t)anim_read_u8(r);
}

static uint16_t anim_read_u16(AnimReader* r) {
    anim_reader_need(r, 2);
    uint16_t v = (uint16_t)(r->p[0]) | ((uint16_t)(r->p[1]) << 8);
    r->p += 2;
    return v;
}

static int16_t anim_read_i16(AnimReader* r) {
    return (int16_t)anim_read_u16(r);
}

static uint32_t anim_read_u32(AnimReader* r) {
    anim_reader_need(r, 4);
    uint32_t v = (uint32_t)(r->p[0])
              | ((uint32_t)(r->p[1]) << 8)
              | ((uint32_t)(r->p[2]) << 16)
              | ((uint32_t)(r->p[3]) << 24);
    r->p += 4;
    return v;
}

static void anim_skip(AnimReader* r, size_t n) {
    anim_reader_need(r, n);
    r->p += n;
}

static AnimCache* anim_cache_load(const char* path) {
    FILE* f = osrs_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "anim_cache_load: cannot open %s\n", path);
        return NULL;
    }

    long size = osrs_file_size_or_abort(f, path);
    if (size < 8) {
        fprintf(stderr, "anim_cache_load: file too small: %s (%ld bytes)\n", path, size);
        abort();
    }

    uint8_t* buf = (uint8_t*)osrs_malloc_or_abort((size_t)size, "animation file");
    osrs_read_exact(f, buf, 1, (size_t)size, path, "animation file");
    fclose(f);

    AnimReader r = { buf, buf + size, path };
    uint32_t magic = anim_read_u32(&r);
    uint32_t sequence_frames_read = 0;
    if (magic != ANIM2_MAGIC) {
        fprintf(stderr, "anim_cache_load: bad magic 0x%08X in %s, expected ANM2\n",
            magic, path);
        abort();
    }

    uint16_t version = anim_read_u16(&r);
    uint16_t header_size = anim_read_u16(&r);
    if (version < ANIM_FORMAT_VERSION_MIN ||
            version > ANIM_FORMAT_VERSION_MAX ||
            header_size < ANIM_HEADER_SIZE_V2) {
        fprintf(stderr,
            "anim_cache_load: unsupported ANM2 header version=%u size=%u in %s\n",
            version, header_size, path);
        abort();
    }
    uint32_t base_count = anim_read_u32(&r);
    uint32_t seq_count = anim_read_u32(&r);
    uint32_t declared_sequence_frames = anim_read_u32(&r);
    uint32_t flags = anim_read_u32(&r);
    if (header_size > ANIM_HEADER_SIZE_V2) {
        anim_skip(&r, (size_t)(header_size - ANIM_HEADER_SIZE_V2));
    }
    if (base_count > ANIM_MAX_BASES || seq_count > ANIM_MAX_SEQUENCES) {
        fprintf(stderr,
            "anim_cache_load: invalid counts bases=%u sequences=%u in %s\n",
            base_count, seq_count, path);
        abort();
    }

    AnimCache* cache = (AnimCache*)osrs_calloc_or_abort(
        1, sizeof(AnimCache), "animation cache");
    cache->base_count = (int)base_count;
    cache->seq_count = (int)seq_count;

    cache->bases = (AnimFrameBase*)osrs_calloc_or_abort(
        cache->base_count, sizeof(AnimFrameBase), "animation framebases");
    cache->base_ids = (uint16_t*)osrs_malloc_or_abort(
        cache->base_count * sizeof(uint16_t), "animation framebase ids");

    for (int i = 0; i < cache->base_count; i++) {
        AnimFrameBase* fb = &cache->bases[i];
        fb->base_id = anim_read_u16(&r);
        cache->base_ids[i] = fb->base_id;
        fb->slot_count = anim_read_u8(&r);

        fb->types = (uint8_t*)osrs_malloc_or_abort(
            fb->slot_count, "animation framebase slot types");
        for (int s = 0; s < fb->slot_count; s++) {
            fb->types[s] = anim_read_u8(&r);
        }

        fb->map_lengths = (uint8_t*)osrs_malloc_or_abort(
            fb->slot_count, "animation framebase map lengths");
        fb->frame_maps = (uint8_t**)osrs_malloc_or_abort(
            fb->slot_count * sizeof(uint8_t*), "animation frame maps");
        for (int s = 0; s < fb->slot_count; s++) {
            uint8_t ml = anim_read_u8(&r);
            fb->map_lengths[s] = ml;
            fb->frame_maps[s] = (uint8_t*)osrs_malloc_or_abort(
                ml, "animation frame map labels");
            for (int j = 0; j < ml; j++) {
                fb->frame_maps[s][j] = anim_read_u8(&r);
            }
        }
    }

    cache->sequences = (AnimSequence*)osrs_calloc_or_abort(
        cache->seq_count, sizeof(AnimSequence), "animation sequences");
    for (int i = 0; i < cache->seq_count; i++) {
        AnimSequence* seq = &cache->sequences[i];
        seq->seq_id = anim_read_u16(&r);
        seq->frame_count = anim_read_u16(&r);

        seq->interleave_count = anim_read_u8(&r);
        if (seq->interleave_count > 0) {
            seq->interleave_order = (uint8_t*)osrs_malloc_or_abort(
                seq->interleave_count, "animation interleave order");
            for (int j = 0; j < seq->interleave_count; j++) {
                seq->interleave_order[j] = anim_read_u8(&r);
            }
        }

        seq->walk_flag = anim_read_i8(&r);

        seq->frames = (AnimSequenceFrame*)osrs_calloc_or_abort(
            seq->frame_count, sizeof(AnimSequenceFrame), "animation sequence frames");
        for (int fi = 0; fi < seq->frame_count; fi++) {
            AnimSequenceFrame* sf = &seq->frames[fi];
            sf->delay = anim_read_u16(&r);
            sf->frame.kind = version >= 3 ? anim_read_u8(&r) : ANIM_FRAME_LEGACY;
            sequence_frames_read++;

            if (sf->frame.kind == ANIM_FRAME_LEGACY) {
                sf->frame.framebase_id = anim_read_u16(&r);
                sf->frame.transform_count = anim_read_u8(&r);
                if (sf->frame.transform_count > 0) {
                    sf->frame.transforms = (AnimTransform*)osrs_malloc_or_abort(
                        sf->frame.transform_count * sizeof(AnimTransform),
                        "animation transforms");
                    for (int t = 0; t < sf->frame.transform_count; t++) {
                        sf->frame.transforms[t].slot_index = anim_read_u8(&r);
                        sf->frame.transforms[t].dx = anim_read_i16(&r);
                        sf->frame.transforms[t].dy = anim_read_i16(&r);
                        sf->frame.transforms[t].dz = anim_read_i16(&r);
                    }
                }
            } else if (sf->frame.kind == ANIM_FRAME_MAYA_BAKED) {
                sf->frame.framebase_id = 0xFFFF;
                sf->frame.transform_count = 0;
                sf->frame.maya_vertex_count = anim_read_u16(&r);
                if (sf->frame.maya_vertex_count == 0) {
                    fprintf(stderr,
                        "anim_cache_load: Maya baked frame has zero vertices in %s\n",
                        path);
                    abort();
                }
                sf->frame.maya_vertices = (int16_t*)osrs_malloc_or_abort(
                    (size_t)sf->frame.maya_vertex_count * 3 * sizeof(int16_t),
                    "Maya baked animation vertices");
                for (int v = 0; v < sf->frame.maya_vertex_count * 3; v++) {
                    sf->frame.maya_vertices[v] = anim_read_i16(&r);
                }
            } else {
                fprintf(stderr,
                    "anim_cache_load: unknown frame kind %u in sequence %u from %s\n",
                    sf->frame.kind, seq->seq_id, path);
                abort();
            }
        }
    }
    if (declared_sequence_frames != sequence_frames_read) {
        fprintf(stderr,
            "anim_cache_load: ANM2 frame count mismatch declared=%u read=%u in %s\n",
            declared_sequence_frames, sequence_frames_read, path);
        abort();
    }
    if (r.p != r.end) {
        fprintf(stderr, "anim_cache_load: ignored %ld trailing bytes in %s\n",
            (long)(r.end - r.p), path);
    }

    free(buf);
    anim_init_trig();

    fprintf(stderr,
        "anim_cache_load: loaded ANM%d flags=0x%08X %d framebases, "
        "%d sequences, %u sequence frames from %s\n",
        version, flags, cache->base_count, cache->seq_count,
        sequence_frames_read, path);
    return cache;
}

static AnimSequence* anim_get_sequence(AnimCache* cache, uint16_t seq_id) {
    if (!cache) return NULL;
    for (int i = 0; i < cache->seq_count; i++) {
        if (cache->sequences[i].seq_id == seq_id) {
            return &cache->sequences[i];
        }
    }
    return NULL;
}

static AnimFrameBase* anim_get_framebase(AnimCache* cache, uint16_t base_id) {
    if (!cache) return NULL;
    for (int i = 0; i < cache->base_count; i++) {
        if (cache->bases[i].base_id == base_id) {
            return &cache->bases[i];
        }
    }
    return NULL;
}

static AnimModelState* anim_model_state_create_with_face_alpha(
    const uint8_t* vertex_skins,
    int base_vert_count,
    const uint8_t* face_alpha_labels,
    const uint8_t* base_face_alphas,
    int face_count
) {
    AnimModelState* state = (AnimModelState*)osrs_calloc_or_abort(
        1, sizeof(AnimModelState), "animation model state");
    state->vert_count = base_vert_count;
    state->verts = (int16_t*)osrs_calloc_or_abort(
        base_vert_count * 3, sizeof(int16_t), "animation model vertices");

    state->groups = (int**)osrs_calloc_or_abort(
        ANIM_MAX_LABELS, sizeof(int*), "animation model groups");
    state->group_counts = (int*)osrs_calloc_or_abort(
        ANIM_MAX_LABELS, sizeof(int), "animation model group counts");

    int label_counts[ANIM_MAX_LABELS] = {0};
    for (int v = 0; v < base_vert_count; v++) {
        uint8_t label = vertex_skins[v];
        label_counts[label]++;
    }

    for (int l = 0; l < ANIM_MAX_LABELS; l++) {
        if (label_counts[l] > 0) {
            state->groups[l] = (int*)osrs_malloc_or_abort(
                label_counts[l] * sizeof(int), "animation model group vertices");
            state->group_counts[l] = 0;
        }
    }

    for (int v = 0; v < base_vert_count; v++) {
        uint8_t label = vertex_skins[v];
        state->groups[label][state->group_counts[label]++] = v;
    }

    if (face_count > 0 && face_alpha_labels && base_face_alphas) {
        state->face_count = face_count;
        state->base_face_alphas = (uint8_t*)osrs_malloc_or_abort(
            face_count, "animation model base face alphas");
        state->face_alphas = (uint8_t*)osrs_malloc_or_abort(
            face_count, "animation model face alphas");
        memcpy(state->base_face_alphas, base_face_alphas, face_count);
        memcpy(state->face_alphas, base_face_alphas, face_count);

        state->face_alpha_groups = (int**)osrs_calloc_or_abort(
            ANIM_MAX_LABELS, sizeof(int*), "animation face alpha groups");
        state->face_alpha_group_counts = (int*)osrs_calloc_or_abort(
            ANIM_MAX_LABELS, sizeof(int), "animation face alpha group counts");

        int face_label_counts[ANIM_MAX_LABELS] = {0};
        for (int face = 0; face < face_count; face++) {
            uint8_t label = face_alpha_labels[face];
            if (label == 255) continue;
            face_label_counts[label]++;
        }
        for (int label = 0; label < ANIM_MAX_LABELS; label++) {
            if (face_label_counts[label] > 0) {
                state->face_alpha_groups[label] = (int*)osrs_malloc_or_abort(
                    face_label_counts[label] * sizeof(int),
                    "animation face alpha group faces");
                state->face_alpha_group_counts[label] = 0;
            }
        }
        for (int face = 0; face < face_count; face++) {
            uint8_t label = face_alpha_labels[face];
            if (label == 255) continue;
            state->face_alpha_groups[label][
                state->face_alpha_group_counts[label]++] = face;
        }
    }

    return state;
}

static AnimModelState* anim_model_state_create(
    const uint8_t* vertex_skins,
    int base_vert_count
) {
    return anim_model_state_create_with_face_alpha(
        vertex_skins, base_vert_count, NULL, NULL, 0);
}

static void anim_model_state_free(AnimModelState* state) {
    if (!state) return;
    free(state->verts);
    for (int l = 0; l < ANIM_MAX_LABELS; l++) {
        free(state->groups[l]);
    }
    free(state->groups);
    free(state->group_counts);
    if (state->face_alpha_groups) {
        for (int l = 0; l < ANIM_MAX_LABELS; l++) {
            free(state->face_alpha_groups[l]);
        }
    }
    free(state->face_alpha_groups);
    free(state->face_alpha_group_counts);
    free(state->base_face_alphas);
    free(state->face_alphas);
    free(state);
}

static void anim_apply_rest_pose(
    AnimModelState* state,
    const int16_t* base_verts_src
) {
    memcpy(state->verts, base_verts_src, state->vert_count * 3 * sizeof(int16_t));
    if (state->face_alphas && state->base_face_alphas) {
        memcpy(state->face_alphas, state->base_face_alphas, state->face_count);
    }
}

static int anim_clamp_alpha(int alpha) {
    if (alpha < 0) return 0;
    if (alpha > 255) return 255;
    return alpha;
}

static void anim_apply_alpha_transform(
    AnimModelState* state,
    const uint8_t* labels,
    uint8_t map_len,
    int dx
) {
    if (!state->face_alphas || !state->face_alpha_groups ||
        !state->face_alpha_group_counts) {
        return;
    }

    int delta = dx * 8;
    for (int m = 0; m < map_len; m++) {
        uint8_t label = labels[m];
        for (int fi = 0; fi < state->face_alpha_group_counts[label]; fi++) {
            int face = state->face_alpha_groups[label][fi];
            state->face_alphas[face] = (uint8_t)anim_clamp_alpha(
                (int)state->face_alphas[face] + delta);
        }
    }
}

static void anim_apply_single_transform(
    AnimModelState* state,
    int type, const uint8_t* labels, uint8_t map_len,
    int dx, int dy, int dz,
    int* pivot_x, int* pivot_y, int* pivot_z);

static void anim_apply_frame(
    AnimModelState* state,
    const int16_t* base_verts_src,
    const AnimFrameData* frame,
    const AnimFrameBase* fb
) {
    if (frame->kind != ANIM_FRAME_LEGACY) {
        fprintf(stderr, "anim_apply_frame: non-legacy frame passed to legacy path\n");
        abort();
    }
    anim_apply_rest_pose(state, base_verts_src);

    int pivot_x = 0, pivot_y = 0, pivot_z = 0;

    for (int t = 0; t < frame->transform_count; t++) {
        uint8_t slot_idx = frame->transforms[t].slot_index;
        if (slot_idx >= fb->slot_count) continue;

        anim_apply_single_transform(
            state, fb->types[slot_idx],
            fb->frame_maps[slot_idx], fb->map_lengths[slot_idx],
            frame->transforms[t].dx,
            frame->transforms[t].dy,
            frame->transforms[t].dz,
            &pivot_x, &pivot_y, &pivot_z);
    }
}

static void anim_apply_maya_baked_frame(
    AnimModelState* state,
    const AnimFrameData* frame
) {
    if (frame->kind != ANIM_FRAME_MAYA_BAKED) {
        fprintf(stderr, "anim_apply_maya_baked_frame: non-Maya frame passed\n");
        abort();
    }
    if ((int)frame->maya_vertex_count != state->vert_count) {

        return;
    }
    memcpy(state->verts, frame->maya_vertices,
        (size_t)state->vert_count * 3 * sizeof(int16_t));
    if (state->face_alphas && state->base_face_alphas) {
        memcpy(state->face_alphas, state->base_face_alphas, state->face_count);
    }
}

static void anim_apply_single_transform(
    AnimModelState* state,
    int type, const uint8_t* labels, uint8_t map_len,
    int dx, int dy, int dz,
    int* pivot_x, int* pivot_y, int* pivot_z
) {
    if (type == 0) {
        int count = 0, sx = 0, sy = 0, sz = 0;
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                sx += state->verts[v * 3];
                sy += state->verts[v * 3 + 1];
                sz += state->verts[v * 3 + 2];
                count++;
            }
        }
        if (count > 0) {
            *pivot_x = sx / count + dx;
            *pivot_y = sy / count + dy;
            *pivot_z = sz / count + dz;
        } else {
            *pivot_x = dx;
            *pivot_y = dy;
            *pivot_z = dz;
        }
    } else if (type == 1) {
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                state->verts[v * 3]     += (int16_t)dx;
                state->verts[v * 3 + 1] += (int16_t)dy;
                state->verts[v * 3 + 2] += (int16_t)dz;
            }
        }
    } else if (type == 2) {
        int ax = (dx & 0xFF) * 8, ay = (dy & 0xFF) * 8, az = (dz & 0xFF) * 8;
        int sin_x = anim_sine[ax & 2047], cos_x = anim_cosine[ax & 2047];
        int sin_y = anim_sine[ay & 2047], cos_y = anim_cosine[ay & 2047];
        int sin_z = anim_sine[az & 2047], cos_z = anim_cosine[az & 2047];
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                int vx = state->verts[v * 3]     - *pivot_x;
                int vy = state->verts[v * 3 + 1] - *pivot_y;
                int vz = state->verts[v * 3 + 2] - *pivot_z;
                int rx = (vx * cos_z + vy * sin_z) >> 16;
                int ry = (vy * cos_z - vx * sin_z) >> 16;
                vx = rx; vy = ry;
                ry = (vy * cos_x - vz * sin_x) >> 16;
                int rz = (vy * sin_x + vz * cos_x) >> 16;
                vy = ry; vz = rz;
                rx = (vx * cos_y + vz * sin_y) >> 16;
                rz = (vz * cos_y - vx * sin_y) >> 16;
                state->verts[v * 3]     = (int16_t)(rx + *pivot_x);
                state->verts[v * 3 + 1] = (int16_t)(vy + *pivot_y);
                state->verts[v * 3 + 2] = (int16_t)(rz + *pivot_z);
            }
        }
    } else if (type == 3) {
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                int vx = state->verts[v * 3]     - *pivot_x;
                int vy = state->verts[v * 3 + 1] - *pivot_y;
                int vz = state->verts[v * 3 + 2] - *pivot_z;
                state->verts[v * 3]     = (int16_t)((vx * dx) / 128 + *pivot_x);
                state->verts[v * 3 + 1] = (int16_t)((vy * dy) / 128 + *pivot_y);
                state->verts[v * 3 + 2] = (int16_t)((vz * dz) / 128 + *pivot_z);
            }
        }
    } else if (type == 5) {
        anim_apply_alpha_transform(state, labels, map_len, dx);
    }
}

static void anim_apply_frame_interleaved(
    AnimModelState* state,
    const int16_t* base_verts_src,
    const AnimFrameData* secondary_frame, const AnimFrameBase* secondary_fb,
    const AnimFrameData* primary_frame, const AnimFrameBase* primary_fb,
    const uint8_t* interleave_order, int interleave_count
) {
    if (secondary_frame->kind != ANIM_FRAME_LEGACY ||
            primary_frame->kind != ANIM_FRAME_LEGACY) {
        fprintf(stderr, "anim_apply_frame_interleaved: Maya frames cannot be interleaved\n");
        abort();
    }
    anim_apply_rest_pose(state, base_verts_src);

    uint8_t secondary_slot[256];
    memset(secondary_slot, 0, sizeof(secondary_slot));
    for (int i = 0; i < interleave_count; i++) {
        secondary_slot[interleave_order[i]] = 1;
    }

    int pivot_x = 0, pivot_y = 0, pivot_z = 0;
    for (int t = 0; t < primary_frame->transform_count; t++) {
        uint8_t slot_idx = primary_frame->transforms[t].slot_index;
        if (slot_idx >= primary_fb->slot_count) continue;

        int type = primary_fb->types[slot_idx];
        int in_interleave = secondary_slot[slot_idx];

        if (!in_interleave || type == 0) {
            anim_apply_single_transform(
                state, type,
                primary_fb->frame_maps[slot_idx],
                primary_fb->map_lengths[slot_idx],
                primary_frame->transforms[t].dx,
                primary_frame->transforms[t].dy,
                primary_frame->transforms[t].dz,
                &pivot_x, &pivot_y, &pivot_z);
        }
    }

    pivot_x = 0; pivot_y = 0; pivot_z = 0;
    for (int t = 0; t < secondary_frame->transform_count; t++) {
        uint8_t slot_idx = secondary_frame->transforms[t].slot_index;
        if (slot_idx >= secondary_fb->slot_count) continue;

        int type = secondary_fb->types[slot_idx];
        int in_interleave = secondary_slot[slot_idx];

        if (in_interleave || type == 0) {
            anim_apply_single_transform(
                state, type,
                secondary_fb->frame_maps[slot_idx],
                secondary_fb->map_lengths[slot_idx],
                secondary_frame->transforms[t].dx,
                secondary_frame->transforms[t].dy,
                secondary_frame->transforms[t].dz,
                &pivot_x, &pivot_y, &pivot_z);
        }
    }
}

static void anim_update_mesh(
    float* mesh_vertices,
    const AnimModelState* state,
    const uint16_t* face_indices,
    int face_count
) {
    for (int fi = 0; fi < face_count; fi++) {
        int a = face_indices[fi * 3];
        int b = face_indices[fi * 3 + 1];
        int c = face_indices[fi * 3 + 2];

        int vi = fi * 9;
        mesh_vertices[vi]     = (float)state->verts[a * 3];
        mesh_vertices[vi + 1] = (float)(-state->verts[a * 3 + 1]);
        mesh_vertices[vi + 2] = (float)state->verts[a * 3 + 2];

        mesh_vertices[vi + 3] = (float)state->verts[b * 3];
        mesh_vertices[vi + 4] = (float)(-state->verts[b * 3 + 1]);
        mesh_vertices[vi + 5] = (float)state->verts[b * 3 + 2];

        mesh_vertices[vi + 6] = (float)state->verts[c * 3];
        mesh_vertices[vi + 7] = (float)(-state->verts[c * 3 + 1]);
        mesh_vertices[vi + 8] = (float)state->verts[c * 3 + 2];
    }
}

static void anim_update_mesh_alpha(
    unsigned char* mesh_colors,
    const AnimModelState* state,
    int face_count
) {
    if (!mesh_colors || !state || !state->face_alphas) return;
    int count = face_count < state->face_count ? face_count : state->face_count;
    for (int fi = 0; fi < count; fi++) {
        unsigned char alpha = (unsigned char)(255 - state->face_alphas[fi]);
        for (int corner = 0; corner < 3; corner++) {
            mesh_colors[(fi * 3 + corner) * 4 + 3] = alpha;
        }
    }
}

static void anim_cache_free(AnimCache* cache) {
    if (!cache) return;

    for (int i = 0; i < cache->base_count; i++) {
        AnimFrameBase* fb = &cache->bases[i];
        free(fb->types);
        free(fb->map_lengths);
        for (int s = 0; s < fb->slot_count; s++) {
            free(fb->frame_maps[s]);
        }
        free(fb->frame_maps);
    }
    free(cache->bases);
    free(cache->base_ids);

    for (int i = 0; i < cache->seq_count; i++) {
        AnimSequence* seq = &cache->sequences[i];
        free(seq->interleave_order);
        for (int fi = 0; fi < seq->frame_count; fi++) {
            free(seq->frames[fi].frame.transforms);
            free(seq->frames[fi].frame.maya_vertices);
        }
        free(seq->frames);
    }
    free(cache->sequences);
    free(cache);
}

#endif
