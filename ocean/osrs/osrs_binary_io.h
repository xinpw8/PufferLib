#ifndef OSRS_BINARY_IO_H
#define OSRS_BINARY_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline void* osrs_malloc_or_abort(size_t size, const char* label) {
    if (size == 0) return NULL;
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "%s: malloc failed for %zu bytes\n", label, size);
        abort();
    }
    return ptr;
}

static inline void* osrs_calloc_or_abort(size_t count, size_t size, const char* label) {
    if (count == 0 || size == 0) return NULL;
    if (count > SIZE_MAX / size) {
        fprintf(stderr, "%s: allocation overflow for %zu * %zu bytes\n",
            label, count, size);
        abort();
    }
    void* ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "%s: calloc failed for %zu * %zu bytes\n",
            label, count, size);
        abort();
    }
    return ptr;
}

static inline void osrs_read_exact(
    FILE* f, void* dst, size_t size, size_t count, const char* path, const char* field
) {
    if (count == 0) return;
    size_t got = fread(dst, size, count, f);
    if (got != count) {
        fprintf(stderr, "%s: short read for %s (%zu/%zu)\n",
            path, field, got, count);
        abort();
    }
}

static inline void osrs_seek_or_abort(FILE* f, long offset, const char* path) {
    if (fseek(f, offset, SEEK_SET) != 0) {
        fprintf(stderr, "%s: fseek failed at offset %ld\n", path, offset);
        abort();
    }
}

static inline long osrs_file_size_or_abort(FILE* f, const char* path) {
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: fseek end failed\n", path);
        abort();
    }
    long size = ftell(f);
    if (size < 0) {
        fprintf(stderr, "%s: ftell failed\n", path);
        abort();
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: fseek start failed\n", path);
        abort();
    }
    return size;
}

#endif
