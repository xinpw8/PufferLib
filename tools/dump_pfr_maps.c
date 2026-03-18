/* dump_pfr_maps.c — Dump all pfr_native map metadata as JSON.
 * Compile: clang -O2 -o dump_pfr_maps tools/dump_pfr_maps.c \
 *   ../pokefirered-native/build/pfr_native/pfr_native_renamed.o \
 *   ../pokefirered-native/build/pfr_native/pfr_native_data.o \
 *   -I../pokefirered-native/src -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include "pfr_native_data.h"

int main(void) {
    size_t n = gPfrNativeMapCount;

    printf("{\n  \"map_count\": %zu,\n  \"maps\": [\n", n);
    for (size_t i = 0; i < n; i++) {
        const PfrNativeMap *m = &gPfrNativeMaps[i];
        printf("    {\n");
        printf("      \"id\": %u,\n", (unsigned)m->map_id);
        printf("      \"name\": \"%s\",\n", m->name ? m->name : "");
        printf("      \"id_symbol\": \"%s\",\n", m->id_symbol ? m->id_symbol : "");
        printf("      \"group\": %u,\n", (unsigned)m->map_group);
        printf("      \"num\": %u,\n", (unsigned)m->map_num);
        printf("      \"width\": %u,\n", (unsigned)m->width);
        printf("      \"height\": %u,\n", (unsigned)m->height);

        /* Warps */
        printf("      \"warps\": [");
        for (size_t w = 0; w < m->warp_count; w++) {
            const PfrNativeWarp *wp = &m->warps[w];
            if (w > 0) printf(",");
            printf("\n        {\"x\": %d, \"y\": %d, \"dest_map\": %u, \"dest_warp_id\": %u, \"supported\": %u}",
                (int)wp->x, (int)wp->y, (unsigned)wp->dest_map, (unsigned)wp->dest_warp_id, (unsigned)wp->supported);
        }
        if (m->warp_count > 0) printf("\n      ");
        printf("],\n");

        /* Connections */
        printf("      \"connections\": [");
        for (size_t c = 0; c < m->connection_count; c++) {
            const PfrNativeConnection *cn = &m->connections[c];
            if (c > 0) printf(",");
            printf("\n        {\"direction\": %u, \"offset\": %d, \"dest_map\": %u}",
                (unsigned)cn->direction, (int)cn->offset, (unsigned)cn->dest_map);
        }
        if (m->connection_count > 0) printf("\n      ");
        printf("]\n");

        printf("    }%s\n", (i + 1 < n) ? "," : "");
    }
    printf("  ]\n}\n");
    return 0;
}
