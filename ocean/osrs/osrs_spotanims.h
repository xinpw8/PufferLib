#ifndef OSRS_SPOTANIMS_H
#define OSRS_SPOTANIMS_H

#include "osrs_assets.h"
#include "osrs_binary_io.h"
#include <stdint.h>
#include <limits.h>

#define OSRS_SPOTANIM_MAGIC 0x544F5053u
#define OSRS_SPOTANIM_VERSION 1u
#define OSRS_SPOTANIM_MODEL_BASE 0xA2000000u
#define OSRS_SPOTANIM_RECOLOR_MODEL_BASE 0x000D0000u

typedef struct {
    uint32_t id;
    int32_t model_id;
    int32_t animation_id;
    uint32_t resize_xy;
    uint32_t resize_z;
    uint32_t rotation;
    int32_t brightness;
    int32_t shadow;
} OsrsSpotAnimDef;

typedef struct {
    OsrsSpotAnimDef* defs;
    int count;
    int loaded;
} OsrsSpotAnimSet;

static void osrs_spotanims_free(OsrsSpotAnimSet* set);

static OsrsSpotAnimSet* osrs_spotanims_load(const char* path) {
    FILE* f = osrs_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "osrs_spotanims_load: cannot open %s\n", path);
        return NULL;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    osrs_read_exact(f, &magic, sizeof(magic), 1, path, "spotanim magic");
    osrs_read_exact(f, &version, sizeof(version), 1, path, "spotanim version");
    osrs_read_exact(f, &count, sizeof(count), 1, path, "spotanim count");
    if (magic != OSRS_SPOTANIM_MAGIC || version != OSRS_SPOTANIM_VERSION) {
        fprintf(stderr, "%s: bad spotanim header magic=0x%08X version=%u\n",
            path, magic, version);
        abort();
    }
    if (count > (uint32_t)INT_MAX) {
        fprintf(stderr, "%s: spotanim count too large: %u\n", path, count);
        abort();
    }

    OsrsSpotAnimSet* set = (OsrsSpotAnimSet*)osrs_calloc_or_abort(
        1, sizeof(*set), "spotanims");
    set->defs = (OsrsSpotAnimDef*)osrs_calloc_or_abort(
        count, sizeof(*set->defs), "spotanim rows");
    set->count = (int)count;
    for (uint32_t i = 0; i < count; i++) {
        osrs_read_exact(f, &set->defs[i], sizeof(set->defs[i]), 1,
            path, "spotanim row");
    }
    fclose(f);
    set->loaded = 1;
    fprintf(stderr, "osrs_spotanims_load: loaded %d from %s\n",
        set->count, path);
    return set;
}

static const OsrsSpotAnimDef* osrs_spotanim_find(
    const OsrsSpotAnimSet* set,
    int id
) {
    if (!set || !set->loaded || id < 0) return NULL;
    for (int i = 0; i < set->count; i++) {
        if ((int)set->defs[i].id == id) return &set->defs[i];
    }
    return NULL;
}

static void osrs_spotanims_free(OsrsSpotAnimSet* set) {
    if (!set) return;
    free(set->defs);
    free(set);
}

#endif
