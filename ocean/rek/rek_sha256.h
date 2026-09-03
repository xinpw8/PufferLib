// Minimal streaming SHA-256 used to bind REK_MJCF_PATH to measured artifacts.

#pragma once

#include <stdint.h>
#include <stdio.h>

typedef struct RekSha256 {
    uint32_t state[8];
    uint64_t bit_length;
    unsigned char block[64];
    size_t block_length;
} RekSha256;

static uint32_t rek_sha256_rotr(uint32_t value, unsigned int amount) {
    return (value >> amount) | (value << (32U - amount));
}

static void rek_sha256_transform(RekSha256* context) {
    static const uint32_t constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    uint32_t words[64];
    for (int i = 0; i < 16; i++) {
        size_t offset = (size_t)i * 4U;
        words[i] = ((uint32_t)context->block[offset] << 24U)
            | ((uint32_t)context->block[offset + 1U] << 16U)
            | ((uint32_t)context->block[offset + 2U] << 8U)
            | (uint32_t)context->block[offset + 3U];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t prior15 = words[i - 15];
        uint32_t prior2 = words[i - 2];
        uint32_t sigma0 = rek_sha256_rotr(prior15, 7U)
            ^ rek_sha256_rotr(prior15, 18U) ^ (prior15 >> 3U);
        uint32_t sigma1 = rek_sha256_rotr(prior2, 17U)
            ^ rek_sha256_rotr(prior2, 19U) ^ (prior2 >> 10U);
        words[i] = words[i - 16] + sigma0 + words[i - 7] + sigma1;
    }

    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t sum1 = rek_sha256_rotr(e, 6U) ^ rek_sha256_rotr(e, 11U)
            ^ rek_sha256_rotr(e, 25U);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 = h + sum1 + choice + constants[i] + words[i];
        uint32_t sum0 = rek_sha256_rotr(a, 2U) ^ rek_sha256_rotr(a, 13U)
            ^ rek_sha256_rotr(a, 22U);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void rek_sha256_init(RekSha256* context) {
    context->state[0] = 0x6a09e667U;
    context->state[1] = 0xbb67ae85U;
    context->state[2] = 0x3c6ef372U;
    context->state[3] = 0xa54ff53aU;
    context->state[4] = 0x510e527fU;
    context->state[5] = 0x9b05688cU;
    context->state[6] = 0x1f83d9abU;
    context->state[7] = 0x5be0cd19U;
    context->bit_length = 0;
    context->block_length = 0;
}

static void rek_sha256_update(RekSha256* context, const unsigned char* data, size_t length) {
    context->bit_length += (uint64_t)length * 8U;
    for (size_t i = 0; i < length; i++) {
        context->block[context->block_length++] = data[i];
        if (context->block_length == sizeof(context->block)) {
            rek_sha256_transform(context);
            context->block_length = 0;
        }
    }
}

static void rek_sha256_final(RekSha256* context, unsigned char digest[32]) {
    context->block[context->block_length++] = 0x80U;
    if (context->block_length > 56U) {
        while (context->block_length < 64U) context->block[context->block_length++] = 0;
        rek_sha256_transform(context);
        context->block_length = 0;
    }
    while (context->block_length < 56U) context->block[context->block_length++] = 0;
    for (int i = 0; i < 8; i++) {
        context->block[63 - i] = (unsigned char)(context->bit_length >> (8U * i));
    }
    rek_sha256_transform(context);
    for (int i = 0; i < 8; i++) {
        digest[4 * i] = (unsigned char)(context->state[i] >> 24U);
        digest[4 * i + 1] = (unsigned char)(context->state[i] >> 16U);
        digest[4 * i + 2] = (unsigned char)(context->state[i] >> 8U);
        digest[4 * i + 3] = (unsigned char)context->state[i];
    }
}

static int rek_sha256_file(const char* path, char hex_digest[65]) {
    FILE* source = fopen(path, "rb");
    if (source == NULL) return 0;
    RekSha256 context;
    rek_sha256_init(&context);
    unsigned char buffer[65536];
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), source)) > 0) {
        rek_sha256_update(&context, buffer, count);
    }
    int read_succeeded = !ferror(source);
    int close_succeeded = fclose(source) == 0;
    if (!read_succeeded || !close_succeeded) return 0;

    unsigned char digest[32];
    static const char hex[] = "0123456789abcdef";
    rek_sha256_final(&context, digest);
    for (int i = 0; i < 32; i++) {
        hex_digest[2 * i] = hex[digest[i] >> 4U];
        hex_digest[2 * i + 1] = hex[digest[i] & 0x0fU];
    }
    hex_digest[64] = '\0';
    return 1;
}
