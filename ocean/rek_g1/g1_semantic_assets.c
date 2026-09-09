#include "g1_semantic_assets.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

typedef struct RekG1PinnedFile {
    const char* name;
    size_t bytes;
    const char* sha256;
} RekG1PinnedFile;

typedef struct RekG1PinnedClip {
    int32_t npz_path_id;
    size_t frames;
    RekG1PinnedFile dof;
    RekG1PinnedFile root_wxyz;
} RekG1PinnedClip;

static const RekG1PinnedFile MANIFEST_FILE = {
    "semantic_duel_assets_manifest.json",
    14951u,
    REK_G1_SEMANTIC_ASSET_MANIFEST_SHA256,
};

static const RekG1PinnedFile MODEL_FILE = {
    "model.two_fighter_arena.xml",
    131586u,
    "d7320a568a40c6aa4cce2bbefbcf3430d626db3b6c5f57ec8ac6a49dc12bc4f2",
};

static const RekG1PinnedFile IDLE_ROOT_POSITION_FILE = {
    "motion_377_root_position.f32le",
    468u,
    "77da7f0aca6ca61aceb086ca8df81b608741fe8842e2a7add22feab2afc096a0",
};

static const RekG1PinnedFile IDLE_ROOT_XYZW_FILE = {
    "motion_377_root_rotation_xyzw.f32le",
    624u,
    "ff8eaafd7e227017535c14968c0040cdd98ce48422f8a5efd286b145e1b461c6",
};

static const RekG1PinnedClip PINNED_CLIPS[REK_G1_SEMANTIC_UNIQUE_CLIP_COUNT] = {
    {
        370, 36u,
        {"motion_370_dof_position.f32le", 4176u,
         "9f183da41eac22ca3b723173c69a27ec078769064285df08fb4333c3db050f03"},
        {"motion_370_root_rotation_wxyz.f32le", 576u,
         "23678c4b27998ab443781e028d14c2a6eb669abdc81d973797783b176a7242cf"},
    },
    {
        371, 146u,
        {"motion_371_dof_position.f32le", 16936u,
         "516343ec455cd8516606ea5e9d5ba93bda35e82d900ff113b4da102f114f367d"},
        {"motion_371_root_rotation_wxyz.f32le", 2336u,
         "79fc5df843fe4eed36323b19a3fe214d9b0229f1e323a017abf15835559aca77"},
    },
    {
        372, 159u,
        {"motion_372_dof_position.f32le", 18444u,
         "423fac2cce7c4eb4542c1f5963fb40532444b51e945341c53793896b10e2be8c"},
        {"motion_372_root_rotation_wxyz.f32le", 2544u,
         "6f430735d1ccff64148d3856e725f1def853b5f920882efc31e8163dead5c508"},
    },
    {
        375, 47u,
        {"motion_375_dof_position.f32le", 5452u,
         "cffc77fc223bfdf8a58f8231c80e4ca910c4ff63b1a9cea6f86432ff01113315"},
        {"motion_375_root_rotation_wxyz.f32le", 752u,
         "4d7ce0d78b52dc1787cc15dfa14607d55f0ecf1a0627d661c4df793c51a8184a"},
    },
    {
        377, 39u,
        {"motion_377_dof_position.f32le", 4524u,
         "e103e7c047741bd8ce718baed8f7b8a1f6a1e1d891f19f3766f8b8b8180d19e3"},
        {"motion_377_root_rotation_wxyz.f32le", 624u,
         "bf4bc5a1fbcf24a84174b8bd7c52c9c553bca01032514eae7ec3d47bd2485e60"},
    },
    {
        380, 140u,
        {"motion_380_dof_position.f32le", 16240u,
         "c25665584ed58e79d28b62903e1a387e488064b909901e7745ec7f9f06749e90"},
        {"motion_380_root_rotation_wxyz.f32le", 2240u,
         "8753568140a5e913aa16d1d0b6af52b00b3bb1fe2886e3ce79f93b2484201f2a"},
    },
    {
        388, 35u,
        {"motion_388_dof_position.f32le", 4060u,
         "95f60a264f8e5f6b106bad1236a2fc171a35492f825220a2ec8fca0cd7ba8d57"},
        {"motion_388_root_rotation_wxyz.f32le", 560u,
         "97a3adfb93ea341792dc457af0ee331fbc5dced29989a6b676562f955d5f4abb"},
    },
    {
        392, 158u,
        {"motion_392_dof_position.f32le", 18328u,
         "801ee8ee90b64c831d25ed3f6be9e9e46371a12ed00db45b83e197d99ed252bb"},
        {"motion_392_root_rotation_wxyz.f32le", 2528u,
         "718576ba8ff2fe5f2b3c6f7f3d55de815d69ab8261bfdf55e5aa7b9dd5d65982"},
    },
};

static void set_error(
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (error == NULL || error_capacity == 0) return;
    if (operation == NULL) operation = "load G1 semantic assets";
    if (detail == NULL) detail = "unknown failure";
    (void)snprintf(error, error_capacity, "%s: %s", operation, detail);
    error[error_capacity - 1] = '\0';
}

static int join_path(
        char output[REK_G1_SEMANTIC_ASSET_PATH_CAPACITY],
        const char* root,
        const char* name) {
    if (output == NULL || root == NULL || name == NULL || root[0] == '\0'
            || name[0] == '\0' || strchr(name, '/') != NULL
            || strchr(name, '\\') != NULL || strstr(name, "..") != NULL) {
        return 0;
    }
    size_t root_length = strlen(root);
    const int separator = root_length > 0 && root[root_length - 1] == '/'
        ? 0 : 1;
    const int written = snprintf(
        output,
        REK_G1_SEMANTIC_ASSET_PATH_CAPACITY,
        separator ? "%s/%s" : "%s%s",
        root,
        name);
    return written > 0
        && written < REK_G1_SEMANTIC_ASSET_PATH_CAPACITY;
}

static int hex_digest(const unsigned char digest[32], char output[65]) {
    static const char HEX[] = "0123456789abcdef";
    if (digest == NULL || output == NULL) return 0;
    for (size_t index = 0; index < 32u; index++) {
        output[index * 2u] = HEX[digest[index] >> 4u];
        output[index * 2u + 1u] = HEX[digest[index] & 0x0fu];
    }
    output[64] = '\0';
    return 1;
}

static RekG1SemanticAssetsStatus read_pinned_file(
        const char* root,
        const RekG1PinnedFile* spec,
        void* destination,
        char* resolved_path,
        char* error,
        size_t error_capacity) {
    char path[REK_G1_SEMANTIC_ASSET_PATH_CAPACITY];
    struct stat file_stat;
    unsigned char buffer[65536];
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_length = 0;
    EVP_MD_CTX* digest_context = NULL;
    size_t offset = 0;
    int fd = -1;
    RekG1SemanticAssetsStatus status = REK_G1_SEMANTIC_ASSETS_FILE_INVALID;

    if (root == NULL || spec == NULL || spec->name == NULL
            || spec->sha256 == NULL || spec->bytes == 0u) {
        set_error(error, error_capacity, "read pinned file", "invalid contract");
        return REK_G1_SEMANTIC_ASSETS_NULL_ARGUMENT;
    }
    if (!join_path(path, root, spec->name)) {
        set_error(error, error_capacity, "read pinned file", "path is invalid");
        return REK_G1_SEMANTIC_ASSETS_PATH_INVALID;
    }
    fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode)
            || file_stat.st_size < 0
            || (uintmax_t)file_stat.st_size != (uintmax_t)spec->bytes) {
        set_error(error, error_capacity, "read pinned file", spec->name);
        if (fd >= 0) close(fd);
        return REK_G1_SEMANTIC_ASSETS_FILE_INVALID;
    }
    digest_context = EVP_MD_CTX_new();
    if (digest_context == NULL
            || EVP_DigestInit_ex(digest_context, EVP_sha256(), NULL) != 1) {
        set_error(error, error_capacity, "hash pinned file", "EVP init failed");
        status = REK_G1_SEMANTIC_ASSETS_ALLOCATION_FAILED;
        goto done;
    }
    while (offset < spec->bytes) {
        const size_t remaining = spec->bytes - offset;
        const size_t requested = remaining < sizeof(buffer)
            ? remaining : sizeof(buffer);
        ssize_t count = read(fd, buffer, requested);
        if (count <= 0) {
            set_error(error, error_capacity, "read pinned file", spec->name);
            goto done;
        }
        if (EVP_DigestUpdate(digest_context, buffer, (size_t)count) != 1) {
            set_error(error, error_capacity, "hash pinned file", spec->name);
            goto done;
        }
        if (destination != NULL) {
            memcpy((unsigned char*)destination + offset, buffer, (size_t)count);
        }
        offset += (size_t)count;
    }
    if (EVP_DigestFinal_ex(
            digest_context, digest, &digest_length) != 1
            || digest_length != 32u) {
        set_error(error, error_capacity, "hash pinned file", spec->name);
        goto done;
    }
    char actual[65];
    if (!hex_digest(digest, actual) || strcmp(actual, spec->sha256) != 0) {
        set_error(error, error_capacity, "hash pinned file", spec->name);
        status = REK_G1_SEMANTIC_ASSETS_HASH_MISMATCH;
        goto done;
    }
    if (resolved_path != NULL) {
        memcpy(resolved_path, path, strlen(path) + 1u);
    }
    status = REK_G1_SEMANTIC_ASSETS_OK;

done:
    EVP_MD_CTX_free(digest_context);
    if (fd >= 0 && close(fd) != 0
            && status == REK_G1_SEMANTIC_ASSETS_OK) {
        set_error(error, error_capacity, "close pinned file", spec->name);
        status = REK_G1_SEMANTIC_ASSETS_FILE_INVALID;
    }
    return status;
}

static int finite_values(const float* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

static int unit_wxyz(const float* values, size_t frames) {
    if (values == NULL) return 0;
    for (size_t frame = 0; frame < frames; frame++) {
        double norm_squared = 0.0;
        for (size_t axis = 0; axis < 4u; axis++) {
            const double value = values[frame * 4u + axis];
            norm_squared += value * value;
        }
        if (!isfinite(norm_squared)
                || fabs(sqrt(norm_squared) - 1.0) > 1e-4) {
            return 0;
        }
    }
    return 1;
}

/*
 * REK's SonicMotionComposer.NormalizeClipHeading mutates each decoded NPZ
 * clip before it can become an active layer.  It removes the yaw of frame 0
 * by left-multiplying every WXYZ root quaternion by that yaw's inverse.
 * Keep the binary artifacts raw and hash-verifiable, then reproduce that
 * current-build load-time transform in memory.
 */
static int normalize_clip_heading_wxyz(float* values, size_t frames) {
    if (values == NULL || frames == 0u) return 0;
    const float w = values[0];
    const float x = values[1];
    const float y = values[2];
    const float z = values[3];
    const float heading = atan2f(
        2.0f * (x * y + z * w),
        1.0f - 2.0f * (y * y + z * z));
    if (!isfinite(heading)) return 0;
    if (fabsf(heading) < 1.0e-6f) return 1;

    const float half_inverse_heading = -0.5f * heading;
    const float cosine = cosf(half_inverse_heading);
    const float sine = sinf(half_inverse_heading);
    if (!isfinite(cosine) || !isfinite(sine)) return 0;
    for (size_t frame = 0; frame < frames; frame++) {
        float* quaternion = values + frame * 4u;
        const float old_w = quaternion[0];
        const float old_x = quaternion[1];
        const float old_y = quaternion[2];
        const float old_z = quaternion[3];
        quaternion[0] = old_w * cosine - old_z * sine;
        quaternion[1] = old_x * cosine - old_y * sine;
        quaternion[2] = old_x * sine + old_y * cosine;
        quaternion[3] = old_z * cosine + old_w * sine;
    }
    return finite_values(values, frames * 4u) && unit_wxyz(values, frames);
}

static RekG1SemanticClipStorage* clip_by_path_id(
        RekG1SemanticAssets* assets,
        int32_t npz_path_id) {
    if (assets == NULL) return NULL;
    for (size_t index = 0; index < REK_G1_SEMANTIC_UNIQUE_CLIP_COUNT; index++) {
        if (assets->clips[index].npz_path_id == npz_path_id) {
            return &assets->clips[index];
        }
    }
    return NULL;
}

const char* rek_g1_semantic_assets_status_string(
        RekG1SemanticAssetsStatus status) {
    switch (status) {
        case REK_G1_SEMANTIC_ASSETS_OK: return "ok";
        case REK_G1_SEMANTIC_ASSETS_NULL_ARGUMENT: return "null argument";
        case REK_G1_SEMANTIC_ASSETS_PATH_INVALID: return "invalid asset path";
        case REK_G1_SEMANTIC_ASSETS_FILE_INVALID: return "invalid asset file";
        case REK_G1_SEMANTIC_ASSETS_HASH_MISMATCH: return "asset hash mismatch";
        case REK_G1_SEMANTIC_ASSETS_ALLOCATION_FAILED: return "allocation failed";
        case REK_G1_SEMANTIC_ASSETS_CONTENT_INVALID: return "invalid binary32 content";
        case REK_G1_SEMANTIC_ASSETS_ROUTE_MISMATCH: return "route table mismatch";
        case REK_G1_SEMANTIC_ASSETS_UNSUPPORTED_ENDIAN: return "unsupported byte order";
        default: return "unknown asset status";
    }
}

RekG1SemanticAssetsStatus rek_g1_semantic_assets_load(
        RekG1SemanticAssets* assets,
        const char* root,
        const uint32_t configured_compositor_kick_duration_ticks[
            REK_G1_REQUIRED_KICK_COUNT],
        char* error,
        size_t error_capacity) {
    RekG1SemanticAssetsStatus status;
    if (assets == NULL || root == NULL
            || configured_compositor_kick_duration_ticks == NULL) {
        set_error(error, error_capacity, "load G1 semantic assets", "null argument");
        return REK_G1_SEMANTIC_ASSETS_NULL_ARGUMENT;
    }
    memset(assets, 0, sizeof(*assets));
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    set_error(error, error_capacity, "load G1 semantic assets", "little endian required");
    return REK_G1_SEMANTIC_ASSETS_UNSUPPORTED_ENDIAN;
#endif
    const size_t root_length = strlen(root);
    if (root_length == 0u
            || root_length >= REK_G1_SEMANTIC_ASSET_PATH_CAPACITY) {
        set_error(error, error_capacity, "load G1 semantic assets", "root path is invalid");
        return REK_G1_SEMANTIC_ASSETS_PATH_INVALID;
    }
    for (size_t index = 0; index < REK_G1_REQUIRED_KICK_COUNT; index++) {
        if (configured_compositor_kick_duration_ticks[index] == 0u) {
            set_error(error, error_capacity, "load G1 semantic assets", "kick duration missing");
            return REK_G1_SEMANTIC_ASSETS_CONTENT_INVALID;
        }
    }
    memcpy(assets->root, root, root_length + 1u);
    status = read_pinned_file(
        root, &MANIFEST_FILE, NULL, NULL, error, error_capacity);
    if (status != REK_G1_SEMANTIC_ASSETS_OK) goto fail;
    status = read_pinned_file(
        root, &MODEL_FILE, NULL, assets->model_path, error, error_capacity);
    if (status != REK_G1_SEMANTIC_ASSETS_OK) goto fail;

    for (size_t index = 0; index < REK_G1_SEMANTIC_UNIQUE_CLIP_COUNT; index++) {
        const RekG1PinnedClip* pinned = &PINNED_CLIPS[index];
        RekG1SemanticClipStorage* storage = &assets->clips[index];
        storage->npz_path_id = pinned->npz_path_id;
        storage->frame_count = pinned->frames;
        storage->dof_position_mujoco = malloc(pinned->dof.bytes);
        storage->root_quaternion_wxyz = malloc(pinned->root_wxyz.bytes);
        if (storage->dof_position_mujoco == NULL
                || storage->root_quaternion_wxyz == NULL) {
            set_error(error, error_capacity, "load G1 semantic assets", "clip allocation failed");
            status = REK_G1_SEMANTIC_ASSETS_ALLOCATION_FAILED;
            goto fail;
        }
        status = read_pinned_file(
            root,
            &pinned->dof,
            storage->dof_position_mujoco,
            NULL,
            error,
            error_capacity);
        if (status != REK_G1_SEMANTIC_ASSETS_OK) goto fail;
        status = read_pinned_file(
            root,
            &pinned->root_wxyz,
            storage->root_quaternion_wxyz,
            NULL,
            error,
            error_capacity);
        if (status != REK_G1_SEMANTIC_ASSETS_OK) goto fail;
        const size_t dof_count = pinned->frames * GEAR_SONIC_ACTION_DIM;
        if (pinned->dof.bytes != dof_count * sizeof(float)
                || pinned->root_wxyz.bytes != pinned->frames * 4u * sizeof(float)
                || !finite_values(storage->dof_position_mujoco, dof_count)
                || !finite_values(
                    storage->root_quaternion_wxyz, pinned->frames * 4u)
                || !unit_wxyz(storage->root_quaternion_wxyz, pinned->frames)
                || !normalize_clip_heading_wxyz(
                    storage->root_quaternion_wxyz, pinned->frames)) {
            set_error(error, error_capacity, "load G1 semantic assets", "clip content invalid");
            status = REK_G1_SEMANTIC_ASSETS_CONTENT_INVALID;
            goto fail;
        }
    }

    assets->idle_root_position_m = malloc(IDLE_ROOT_POSITION_FILE.bytes);
    assets->idle_root_rotation_xyzw = malloc(IDLE_ROOT_XYZW_FILE.bytes);
    if (assets->idle_root_position_m == NULL
            || assets->idle_root_rotation_xyzw == NULL) {
        set_error(error, error_capacity, "load G1 semantic assets", "idle allocation failed");
        status = REK_G1_SEMANTIC_ASSETS_ALLOCATION_FAILED;
        goto fail;
    }
    status = read_pinned_file(
        root,
        &IDLE_ROOT_POSITION_FILE,
        assets->idle_root_position_m,
        NULL,
        error,
        error_capacity);
    if (status != REK_G1_SEMANTIC_ASSETS_OK) goto fail;
    status = read_pinned_file(
        root,
        &IDLE_ROOT_XYZW_FILE,
        assets->idle_root_rotation_xyzw,
        NULL,
        error,
        error_capacity);
    if (status != REK_G1_SEMANTIC_ASSETS_OK) goto fail;
    if (!finite_values(assets->idle_root_position_m, 39u * 3u)
            || !finite_values(assets->idle_root_rotation_xyzw, 39u * 4u)) {
        set_error(error, error_capacity, "load G1 semantic assets", "idle content invalid");
        status = REK_G1_SEMANTIC_ASSETS_CONTENT_INVALID;
        goto fail;
    }

    const RekG1NativeMotionRouteTable* route_table =
        rek_g1_native_static_motion_routes();
    if (!rek_g1_native_validate_static_motion_routes(route_table)) {
        set_error(error, error_capacity, "load G1 semantic assets", "route table invalid");
        status = REK_G1_SEMANTIC_ASSETS_ROUTE_MISMATCH;
        goto fail;
    }
    for (size_t index = 0; index < route_table->count; index++) {
        const RekG1NativeMotionRoute* route = &route_table->routes[index];
        RekG1SemanticClipStorage* storage = clip_by_path_id(
            assets, route->npz_path_id);
        if (storage == NULL || storage->frame_count != route->asset_frames
                || (size_t)route->id >= REK_G1_STATIC_ROUTE_COUNT) {
            set_error(error, error_capacity, "load G1 semantic assets", "route clip mismatch");
            status = REK_G1_SEMANTIC_ASSETS_ROUTE_MISMATCH;
            goto fail;
        }
        uint32_t duration = 0u;
        if (route->kind == REK_G1_NATIVE_ROUTE_KICK) {
            if (route->runtime_move_index < 6u || route->runtime_move_index > 9u) {
                set_error(error, error_capacity, "load G1 semantic assets", "kick index mismatch");
                status = REK_G1_SEMANTIC_ASSETS_ROUTE_MISMATCH;
                goto fail;
            }
            duration = configured_compositor_kick_duration_ticks[
                (size_t)route->runtime_move_index - 6u];
        }
        assets->route_assets[(size_t)route->id] =
            (RekG1SemanticDuelRouteAsset){
                .route_id = route->id,
                .clip = {
                    .dof_position_mujoco = storage->dof_position_mujoco,
                    .root_quaternion_wxyz = storage->root_quaternion_wxyz,
                    .dof_position_count =
                        storage->frame_count * GEAR_SONIC_ACTION_DIM,
                    .root_quaternion_count = storage->frame_count * 4u,
                    .frame_count = storage->frame_count,
                    .fps = route->asset_fps,
                },
                .configured_compositor_duration_ticks = duration,
            };
    }

    RekG1SemanticClipStorage* idle = clip_by_path_id(assets, 377);
    if (idle == NULL || idle->frame_count != 39u) {
        set_error(error, error_capacity, "load G1 semantic assets", "idle clip missing");
        status = REK_G1_SEMANTIC_ASSETS_ROUTE_MISMATCH;
        goto fail;
    }
    for (size_t frame = 0; frame < idle->frame_count; frame++) {
        const float* wxyz = idle->root_quaternion_wxyz + frame * 4u;
        float* xyzw = assets->idle_root_rotation_xyzw + frame * 4u;
        xyzw[0] = wxyz[1];
        xyzw[1] = wxyz[2];
        xyzw[2] = wxyz[3];
        xyzw[3] = wxyz[0];
    }
    assets->fixed_idle = (GearSonicNativeMotion){
        .dof_position_mujoco = idle->dof_position_mujoco,
        .root_position_m = assets->idle_root_position_m,
        .root_rotation_xyzw = assets->idle_root_rotation_xyzw,
        .frames = idle->frame_count,
        .loop = 1,
    };
    assets->loaded = 1u;
    return REK_G1_SEMANTIC_ASSETS_OK;

fail:
    rek_g1_semantic_assets_close(assets);
    return status;
}

void rek_g1_semantic_assets_close(RekG1SemanticAssets* assets) {
    if (assets == NULL) return;
    for (size_t index = 0; index < REK_G1_SEMANTIC_UNIQUE_CLIP_COUNT; index++) {
        free(assets->clips[index].dof_position_mujoco);
        free(assets->clips[index].root_quaternion_wxyz);
    }
    free(assets->idle_root_position_m);
    free(assets->idle_root_rotation_xyzw);
    memset(assets, 0, sizeof(*assets));
}
