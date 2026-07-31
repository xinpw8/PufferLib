#ifndef OSRS_ASSET_RAYLIB_H
#define OSRS_ASSET_RAYLIB_H

#include "osrs_assets.h"

#if __has_include("raylib.h")
#include "raylib.h"
#elif __has_include("raylib-5.5_macos/include/raylib.h")
#include "raylib-5.5_macos/include/raylib.h"
#else
#error "raylib.h not found"
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline const char* osrs_asset_ext(const char* path, const char* fallback) {
    const char* dot = path ? strrchr(path, '.') : NULL;
    return dot && dot[0] ? dot : fallback;
}

static inline Image osrs_asset_load_image(const char* path) {
    Image empty = {0};
    OsrsAssetBytes bytes = osrs_asset_read_all(path);
    if (!bytes.data || bytes.size == 0) {
        osrs_asset_bytes_free(&bytes);
        return empty;
    }
    if (bytes.size > (size_t)INT_MAX) {
        fprintf(stderr, "image asset too large: %s (%zu bytes)\n", path, bytes.size);
        abort();
    }
    Image image = LoadImageFromMemory(
        osrs_asset_ext(path, ".png"), bytes.data, (int)bytes.size);
    osrs_asset_bytes_free(&bytes);
    return image;
}

static inline Texture2D osrs_asset_load_texture(const char* path) {
    Texture2D empty = {0};
    Image image = osrs_asset_load_image(path);
    if (!image.data) return empty;
    Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);
    return texture;
}

static inline Font osrs_asset_load_font(const char* path, int font_size) {
    Font empty = {0};
    OsrsAssetBytes bytes = osrs_asset_read_all(path);
    if (!bytes.data || bytes.size == 0) {
        osrs_asset_bytes_free(&bytes);
        return empty;
    }
    if (bytes.size > (size_t)INT_MAX) {
        fprintf(stderr, "font asset too large: %s (%zu bytes)\n", path, bytes.size);
        abort();
    }
    Font font = LoadFontFromMemory(
        osrs_asset_ext(path, ".ttf"), bytes.data, (int)bytes.size, font_size, NULL, 95);
    osrs_asset_bytes_free(&bytes);
    return font.texture.id != 0 ? font : empty;
}

#endif
