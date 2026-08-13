// Custom ocean env encoders. Included by algo.cu.
// Per-env nets live under ocean/<env>/<env>.cu and are pulled in below.

// Normal(0, std). Used by custom ocean encoders for embeddings.
void puf_normal_init(Prec* dst, float std, ulong seed, cudaStream_t stream) {
    long n = numel(dst->shape);
    assert(n > 0);
    long rand_count = (n % 2 == 0) ? n : n + 1;
    float* buf;
    cudaMalloc(&buf, rand_count * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, buf, rand_count, 0.0f, std);
    curandDestroyGenerator(gen);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

#include "../ocean/nmmo3/nmmo3.cu"
#include "../ocean/minimal/minimal.cu"

#include "../ocean/osrs/osrs_item_obs_generated.h"
__device__ static const float OSRS_ITEM_OBS_TABLE_DEV
    [OSRS_ITEM_OBS_TABLE_ROWS][OSRS_ITEM_OBS_TABLE_COLS] = {
#include "../ocean/osrs/osrs_item_obs_table.inc"
};
#include "../ocean/osrs/osrs_entity_encoder.cu"
#include "../ocean/osrs_colosseum/osrs_colosseum.cu"
#include "../ocean/osrs_inferno/osrs_inferno.cu"
#ifdef PUFFER_NETHACK
#include "../ocean/nethack/nethack.cu"
#endif

// Override encoder vtable for known ocean environments. No-op for unknown envs.
static void create_custom_encoder(const char* env_name, Encoder* enc) {
#ifdef PUFFER_NETHACK
    if (strcmp(env_name, "nethack") == 0) {
        create_nethack_encoder(enc);
        return;
    }
#endif
    if (strcmp(env_name, "nmmo3") == 0) {
        create_nmmo3_encoder(enc);
        return;
    }
    if (strcmp(env_name, "minimal") == 0) {
        create_minimal_encoder(enc);
        return;
    }
    if (strcmp(env_name, "osrs_colosseum") == 0) {
        create_osrs_colosseum_encoder(enc);
        return;
    }
    if (strcmp(env_name, "osrs_inferno") == 0) {
        create_osrs_inferno_encoder(enc);
        return;
    }
    if (strcmp(env_name, "osrs_zulrah") == 0) {
        create_osrs_zulrah_encoder(enc);
        return;
    }
    if (strcmp(env_name, "osrs_pvp") == 0) {
        create_osrs_pvp_encoder(enc);
        return;
    }
}

static void create_custom_decoder(const char* env_name, Decoder* dec) {
#ifdef PUFFER_NETHACK
    if (strcmp(env_name, "nethack") == 0) {
        create_nethack_decoder(dec);
        return;
    }
#endif
}
