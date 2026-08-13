#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ocean/osrs/encounters/encounter_inferno.h"

#define EXACT_MAGIC "INFEXACTv1"
#define EXACT_VERSION 1u
#define EXACT_CHUNK_BYTES 65536
#define EXACT_ROLLOUT_STEPS 64
#define EXACT_ENV_SEED 0x01FEC0DEu
#define EXACT_ACTION_SEED 0xD1B54A32D192ED03ULL

typedef struct {
    char magic[16];
    uint32_t version;
    uint32_t state_size;
    uint32_t forecast_size;
    uint32_t forecast_obs_size;
    uint32_t obs_size;
    uint32_t action_features;
    uint32_t record_count;
} InfExactFileHeader;

typedef struct {
    uint32_t scenario_id;
    uint32_t step_index;
    uint32_t tick;
    uint32_t public_start_wave;
    uint32_t wave;
    uint32_t terminal;
    uint32_t winner;
    uint32_t state_size;
    uint32_t obs_size;
    uint32_t forecast_size;
    uint64_t state_hash;
    uint64_t forecast_hash;
    uint64_t forecast_obs_hash;
    uint64_t obs_hash;
    float reward;
} InfExactRecordHeader;

typedef struct {
    FILE* file;
    uint32_t record_count;
} InfExactWriter;

static uint64_t exact_fnv_bytes(uint64_t h, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t exact_hash_bytes(const void* data, size_t size) {
    return exact_fnv_bytes(1469598103934665603ULL, data, size);
}

static void exact_write_all(FILE* file, const void* data, size_t size) {
    if (fwrite(data, 1, size, file) != size) {
        perror("write inferno exact fixture");
        abort();
    }
}

static void exact_readable_path(
    char* out,
    size_t out_size,
    const char* dir,
    const char* file_name
) {
    int n = snprintf(out, out_size, "%s/%s", dir, file_name);
    if (n < 0 || (size_t)n >= out_size) {
        fprintf(stderr, "fixture path too long: %s/%s\n", dir, file_name);
        abort();
    }
}

static void exact_mkdir_if_needed(const char* dir) {
    if (mkdir(dir, 0777) == 0) return;
    if (errno == EEXIST) return;
    perror("mkdir inferno exact fixture dir");
    abort();
}

static void exact_writer_open(InfExactWriter* writer, const char* path) {
    memset(writer, 0, sizeof(*writer));
    writer->file = fopen(path, "wb");
    if (!writer->file) {
        perror("open inferno exact fixture");
        abort();
    }

    InfExactFileHeader header = {0};
    memcpy(header.magic, EXACT_MAGIC, sizeof(EXACT_MAGIC));
    header.version = EXACT_VERSION;
    header.state_size = (uint32_t)sizeof(InfernoState);
    header.forecast_size = (uint32_t)sizeof(InfStepOutForecast);
    header.forecast_obs_size = INF_STEP_OUT_FORECAST_OBS_SIZE;
    header.obs_size = INF_NUM_OBS;
    header.action_features = INF_STEP_OUT_FORECAST_ACTION_FEATURES;
    exact_write_all(writer->file, &header, sizeof(header));
}

static void exact_writer_close(InfExactWriter* writer) {
    InfExactFileHeader header = {0};
    memcpy(header.magic, EXACT_MAGIC, sizeof(EXACT_MAGIC));
    header.version = EXACT_VERSION;
    header.state_size = (uint32_t)sizeof(InfernoState);
    header.forecast_size = (uint32_t)sizeof(InfStepOutForecast);
    header.forecast_obs_size = INF_STEP_OUT_FORECAST_OBS_SIZE;
    header.obs_size = INF_NUM_OBS;
    header.action_features = INF_STEP_OUT_FORECAST_ACTION_FEATURES;
    header.record_count = writer->record_count;
    if (fseek(writer->file, 0, SEEK_SET) != 0) {
        perror("seek inferno exact fixture");
        abort();
    }
    exact_write_all(writer->file, &header, sizeof(header));
    if (fclose(writer->file) != 0) {
        perror("close inferno exact fixture");
        abort();
    }
    writer->file = NULL;
}

static void exact_capture(
    InfExactWriter* writer,
    uint32_t scenario_id,
    uint32_t step_index,
    uint32_t public_start_wave,
    InfernoState* s,
    InfernoContext* ctx
) {
    InfStepOutForecast forecast;
    float forecast_obs[INF_STEP_OUT_FORECAST_OBS_SIZE];
    float obs[INF_NUM_OBS];

    inf_build_step_out_forecast_ctx(s, ctx, &forecast);
    inf_write_obs_ctx((EncounterState*)s, (EncounterContext*)ctx, obs);
    int forecast_obs_offset =
        INF_PLAYER_OBS_SIZE + INF_PILLAR_OBS_SIZE + INF_TOTAL_NPC_OBS_SIZE;
    memcpy(forecast_obs, &obs[forecast_obs_offset], sizeof(forecast_obs));

    InfExactRecordHeader record = {0};
    record.scenario_id = scenario_id;
    record.step_index = step_index;
    record.tick = (uint32_t)s->tick;
    record.public_start_wave = public_start_wave;
    record.wave = (uint32_t)s->wave;
    record.terminal = (uint32_t)inf_is_terminal_ctx(
        (EncounterState*)s, (EncounterContext*)ctx);
    record.winner = (uint32_t)s->winner;
    record.state_size = (uint32_t)sizeof(*s);
    record.obs_size = INF_NUM_OBS;
    record.forecast_size = (uint32_t)sizeof(forecast);
    record.state_hash = exact_hash_bytes(s, sizeof(*s));
    record.forecast_hash = exact_hash_bytes(&forecast, sizeof(forecast));
    record.forecast_obs_hash = exact_hash_bytes(forecast_obs, sizeof(forecast_obs));
    record.obs_hash = exact_hash_bytes(obs, sizeof(obs));
    record.reward = inf_get_reward_ctx((EncounterState*)s, (EncounterContext*)ctx);

    exact_write_all(writer->file, &record, sizeof(record));
    exact_write_all(writer->file, &forecast, sizeof(forecast));
    exact_write_all(writer->file, forecast_obs, sizeof(forecast_obs));
    exact_write_all(writer->file, obs, sizeof(obs));
    exact_write_all(writer->file, s, sizeof(*s));
    writer->record_count++;
}

static uint64_t exact_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void exact_trace_actions(
    uint64_t* rng,
    int actions[INF_NUM_ACTION_HEADS]
) {
    for (int head = 0; head < INF_NUM_ACTION_HEADS; head++) {
        actions[head] = (int)(exact_splitmix64(rng) %
            (uint64_t)INF_ACTION_DIMS[head]);
    }
}

static void exact_init_state(
    InfernoState* s,
    InfernoContext* ctx,
    int public_start_wave,
    uint32_t seed
) {
    inf_init_context_typed(ctx);
    memset(s, 0, sizeof(*s));
    inf_put_int_ctx(
        (EncounterState*)s,
        (EncounterContext*)ctx,
        "start_wave",
        public_start_wave);
    inf_put_int_ctx(
        (EncounterState*)s,
        (EncounterContext*)ctx,
        "step_out_forecast_obs_mode",
        INF_STEP_OUT_FORECAST_MODE_EXACT_ROLLOUT);
    inf_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
}

static void exact_run_wave_rollout(
    InfExactWriter* writer,
    uint32_t scenario_id,
    int public_start_wave
) {
    InfernoState s;
    InfernoContext ctx;
    exact_init_state(&s, &ctx, public_start_wave, EXACT_ENV_SEED);

    uint32_t capture_idx = 0;
    uint64_t action_rng = EXACT_ACTION_SEED;
    int actions[INF_NUM_ACTION_HEADS];
    exact_capture(writer, scenario_id, capture_idx++, public_start_wave, &s, &ctx);
    for (int step = 0; step < EXACT_ROLLOUT_STEPS; step++) {
        exact_trace_actions(&action_rng, actions);
        inf_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
        exact_capture(writer, scenario_id, capture_idx++, public_start_wave, &s, &ctx);
        if (s.episode_over) break;
    }
}

static void exact_generate_fixture(const char* path) {
    InfExactWriter writer;
    exact_writer_open(&writer, path);

    exact_run_wave_rollout(&writer, 1u, 1);
    exact_run_wave_rollout(&writer, 62u, 62);
    exact_run_wave_rollout(&writer, 67u, 67);
    exact_run_wave_rollout(&writer, 69u, 69);

    exact_writer_close(&writer);
}

static int exact_compare_files(const char* expected_path, const char* actual_path) {
    FILE* expected = fopen(expected_path, "rb");
    if (!expected) {
        perror("open expected inferno exact fixture");
        abort();
    }
    FILE* actual = fopen(actual_path, "rb");
    if (!actual) {
        perror("open actual inferno exact fixture");
        abort();
    }

    uint8_t expected_buf[EXACT_CHUNK_BYTES];
    uint8_t actual_buf[EXACT_CHUNK_BYTES];
    uint64_t offset = 0;
    for (;;) {
        size_t ne = fread(expected_buf, 1, sizeof(expected_buf), expected);
        size_t na = fread(actual_buf, 1, sizeof(actual_buf), actual);
        if (ne != na) {
            printf("inferno exact mismatch: size differs at byte %llu\n",
                (unsigned long long)offset);
            fclose(expected);
            fclose(actual);
            return 1;
        }
        if (ne == 0) break;
        if (memcmp(expected_buf, actual_buf, ne) != 0) {
            for (size_t i = 0; i < ne; i++) {
                if (expected_buf[i] == actual_buf[i]) continue;
                printf("inferno exact mismatch at byte %llu: expected %u got %u\n",
                    (unsigned long long)(offset + i),
                    (unsigned)expected_buf[i],
                    (unsigned)actual_buf[i]);
                fclose(expected);
                fclose(actual);
                return 1;
            }
        }
        offset += (uint64_t)ne;
    }

    fclose(expected);
    fclose(actual);
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3 ||
            (strcmp(argv[1], "--write-golden") != 0 &&
             strcmp(argv[1], "--compare") != 0)) {
        fprintf(stderr,
            "usage: %s --write-golden DIR | --compare DIR\n", argv[0]);
        return 2;
    }

    inf_build_npc_stats();

    char fixture_path[1024];
    char current_path[1024];
    exact_mkdir_if_needed(argv[2]);
    exact_readable_path(
        fixture_path, sizeof(fixture_path), argv[2],
        "inferno_forecast_exact.bin");

    if (strcmp(argv[1], "--write-golden") == 0) {
        exact_generate_fixture(fixture_path);
        printf("inferno exact golden wrote %s\n", fixture_path);
        return 0;
    }

    exact_readable_path(
        current_path, sizeof(current_path), argv[2],
        "inferno_forecast_exact.current.bin");
    exact_generate_fixture(current_path);
    int failed = exact_compare_files(fixture_path, current_path);
    if (failed) return 1;
    printf("inferno exact golden compare PASS: %s\n", fixture_path);
    return 0;
}
