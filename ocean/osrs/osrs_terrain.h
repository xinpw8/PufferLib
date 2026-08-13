#ifndef OSRS_TERRAIN_H
#define OSRS_TERRAIN_H

#include "raylib.h"
#include "osrs_assets.h"
#include "osrs_binary_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TERR_MAGIC 0x54455252

typedef struct {
    Model model;
    int vertex_count;
    int region_count;
    int min_world_x;
    int min_world_y;
    int loaded;
    float* heightmap;
    int hm_min_x;
    int hm_min_y;
    int hm_width;
    int hm_height;
} TerrainMesh;

static TerrainMesh* terrain_load(const char* path) {
    FILE* f = osrs_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "terrain_load: could not open %s\n", path);
        return NULL;
    }

    uint32_t magic, vert_count, region_count;
    int32_t min_wx, min_wy;
    osrs_read_exact(f, &magic, 4, 1, path, "magic");
    if (magic != TERR_MAGIC) {
        fprintf(stderr, "terrain_load: bad magic %08x\n", magic);
        abort();
    }
    osrs_read_exact(f, &vert_count, 4, 1, path, "vertex count");
    osrs_read_exact(f, &region_count, 4, 1, path, "region count");
    osrs_read_exact(f, &min_wx, 4, 1, path, "min world x");
    osrs_read_exact(f, &min_wy, 4, 1, path, "min world y");

    fprintf(stderr, "terrain_load: %u verts, %u regions, origin (%d, %d)\n",
            vert_count, region_count, min_wx, min_wy);

    float* raw_verts = (float*)osrs_malloc_or_abort(
        vert_count * 3 * sizeof(float), "terrain vertices");
    osrs_read_exact(f, raw_verts, sizeof(float), vert_count * 3, path, "vertices");

    unsigned char* raw_colors = (unsigned char*)osrs_malloc_or_abort(
        vert_count * 4, "terrain colors");
    osrs_read_exact(f, raw_colors, 1, vert_count * 4, path, "colors");

    Mesh mesh = { 0 };
    mesh.vertexCount = (int)vert_count;
    mesh.triangleCount = (int)(vert_count / 3);
    mesh.vertices = raw_verts;
    mesh.colors = raw_colors;

    mesh.normals = (float*)osrs_calloc_or_abort(
        vert_count * 3, sizeof(float), "terrain normals");
    for (int i = 0; i < mesh.triangleCount; i++) {
        int base = i * 9;
        float ax = raw_verts[base + 0], ay = raw_verts[base + 1], az = raw_verts[base + 2];
        float bx = raw_verts[base + 3], by = raw_verts[base + 4], bz = raw_verts[base + 5];
        float cx = raw_verts[base + 6], cy = raw_verts[base + 7], cz = raw_verts[base + 8];

        float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 0.0001f) { nx /= len; ny /= len; nz /= len; }

        for (int v = 0; v < 3; v++) {
            mesh.normals[i * 9 + v * 3 + 0] = nx;
            mesh.normals[i * 9 + v * 3 + 1] = ny;
            mesh.normals[i * 9 + v * 3 + 2] = nz;
        }
    }

    UploadMesh(&mesh, false);

    TerrainMesh* tm = (TerrainMesh*)osrs_calloc_or_abort(
        1, sizeof(TerrainMesh), "terrain mesh");
    tm->model = LoadModelFromMesh(mesh);
    tm->vertex_count = (int)vert_count;
    tm->region_count = (int)region_count;
    tm->min_world_x = min_wx;
    tm->min_world_y = min_wy;
    tm->loaded = 1;

    int32_t hm_min_x, hm_min_y;
    uint32_t hm_w, hm_h;
    size_t has_heightmap = fread(&hm_min_x, 4, 1, f);
    if (has_heightmap == 1) {
        osrs_read_exact(f, &hm_min_y, 4, 1, path, "heightmap min y");
        osrs_read_exact(f, &hm_w, 4, 1, path, "heightmap width");
        osrs_read_exact(f, &hm_h, 4, 1, path, "heightmap height");
        if (hm_w == 0 || hm_h == 0 || hm_w > 4096 || hm_h > 4096) {
            fprintf(stderr, "terrain_load: invalid heightmap dimensions %ux%u\n",
                hm_w, hm_h);
            abort();
        }
        tm->hm_min_x = hm_min_x;
        tm->hm_min_y = hm_min_y;
        tm->hm_width = (int)hm_w;
        tm->hm_height = (int)hm_h;
        tm->heightmap = (float*)osrs_malloc_or_abort(
            hm_w * hm_h * sizeof(float), "terrain heightmap");
        osrs_read_exact(f, tm->heightmap, sizeof(float),
            hm_w * hm_h, path, "heightmap values");
        fprintf(stderr, "terrain heightmap: %dx%d, origin (%d, %d)\n",
                tm->hm_width, tm->hm_height, tm->hm_min_x, tm->hm_min_y);
    } else if (!feof(f)) {
        fprintf(stderr, "terrain_load: failed probing heightmap header\n");
        abort();
    }

    fclose(f);
    return tm;
}

static void terrain_offset(TerrainMesh* tm, int wx, int wy) {
    if (!tm || !tm->loaded) return;
    float dx = (float)wx;
    float dz = (float)wy;
    float* verts = tm->model.meshes[0].vertices;
    for (int i = 0; i < tm->vertex_count; i++) {
        verts[i * 3 + 0] -= dx;
        verts[i * 3 + 2] += dz;
    }
    UpdateMeshBuffer(tm->model.meshes[0], 0, verts,
                     tm->vertex_count * 3 * sizeof(float), 0);
    tm->min_world_x -= wx;
    tm->min_world_y -= wy;
    if (tm->heightmap) {
        tm->hm_min_x -= wx;
        tm->hm_min_y -= wy;
    }
    fprintf(stderr, "terrain_offset: shifted by (%d, %d), new origin (%d, %d)\n",
            wx, wy, tm->min_world_x, tm->min_world_y);
}

static float terrain_height_at(TerrainMesh* tm, int world_x, int world_y) {
    if (!tm || !tm->heightmap) return -2.0f;
    int lx = world_x - tm->hm_min_x;
    int ly = world_y - tm->hm_min_y;
    if (lx < 0 || lx >= tm->hm_width || ly < 0 || ly >= tm->hm_height)
        return -2.0f;
    return tm->heightmap[lx + ly * tm->hm_width];
}

static float terrain_height_avg(TerrainMesh* tm, int world_x, int world_y) {
    float h00 = terrain_height_at(tm, world_x, world_y);
    float h10 = terrain_height_at(tm, world_x + 1, world_y);
    float h01 = terrain_height_at(tm, world_x, world_y + 1);
    float h11 = terrain_height_at(tm, world_x + 1, world_y + 1);
    return (h00 + h10 + h01 + h11) * 0.25f;
}

static void terrain_free(TerrainMesh* tm) {
    if (!tm) return;
    if (tm->loaded) {
        UnloadModel(tm->model);
    }
    free(tm->heightmap);
    free(tm);
}

#endif
