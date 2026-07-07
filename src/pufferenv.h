#pragma once

#include "ini.h"
#ifndef Env
#error "Env must be defined before including pufferenv.h"
#endif

typedef struct Agent {
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;

void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

typedef uint16_t bf16;

static inline bf16 f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(bf16 b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline void store_f32x8_as_bf16(bf16* dst, __m256 v) {
    __m256i vi = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
    __m128i lo = _mm256_castsi256_si128(vi);
    __m128i hi = _mm256_extracti128_si256(vi, 1);
    _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(lo, hi));
}
