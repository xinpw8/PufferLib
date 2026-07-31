#include <stdio.h>
#include <string.h>

#include "ocean/osrs/osrs_assets.h"
#include "ocean/osrs/encounters/encounter_colosseum.h"

typedef struct {
    int checked;
    int failures;
} SpriteAssetCheck;

static int path_has_png_signature(const char* path) {
    unsigned char signature[8];
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(signature, 1, sizeof(signature), f);
    fclose(f);
    static const unsigned char png_signature[8] = {
        0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'
    };
    return n == sizeof(signature) &&
        memcmp(signature, png_signature, sizeof(signature)) == 0;
}

static void check_colosseum_consumable_sprite(int raw_osrs_id, void* ctx) {
    SpriteAssetCheck* check = (SpriteAssetCheck*)ctx;
    char logical_path[64];
    snprintf(logical_path, sizeof(logical_path), "sprites/items/%d.png", raw_osrs_id);
    const char* path = osrs_asset_path(logical_path);
    check->checked++;
    if (path_has_png_signature(path)) return;

    check->failures++;
    fprintf(stderr, "missing or invalid sprite for colosseum raw id %d: %s\n",
        raw_osrs_id,
        path);
}

int main(void) {
    SpriteAssetCheck check = {0};
    col_for_each_reachable_consumable_dose_raw_osrs_id(
        check_colosseum_consumable_sprite,
        &check);
    if (check.checked == 0) {
        fprintf(stderr, "no colosseum consumable dose sprites checked\n");
        return 1;
    }
    if (check.failures != 0) {
        fprintf(stderr, "test_colosseum_consumable_sprite_assets: %d failure(s)\n",
            check.failures);
        return 1;
    }
    printf("test_colosseum_consumable_sprite_assets: OK (%d sprites)\n",
        check.checked);
    return 0;
}
