#define _POSIX_C_SOURCE 200809L

#include "g1_model_identity.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static unsigned int assertions = 0u;

#define CHECK(expression) do { \
    assertions += 1u; \
    if (!(expression)) { \
        (void)fprintf(stderr, "assertion failed at %s:%d: %s\n", \
            __FILE__, __LINE__, #expression); \
        exit(1); \
    } \
} while (0)

static const unsigned char ENCODER_PAYLOAD[] = "encoder-fixture-v1\n";
static const unsigned char DECODER_PAYLOAD[] = "decoder-fixture-v1\n";

static RekG1ModelIdentityContract valid_contract(void) {
    const RekG1ModelIdentityContract contract = {
        .schema = REK_G1_EXPLICIT_BATCH_SCHEMA,
        .classification = REK_G1_PUBLIC_FAMILY_CLASSIFICATION,
        .manifest_sha256 =
            "14b876e6b05140f4037c9e985766fdb0b82c470f032599b33c497d1a5fbd745f",
        .robot_batch = 8u,
        .rek_parity_claim = 0u,
        .current_steam_authority = 0u,
        .training_enabled = 0u,
        .encoder = {
            .source_sha256 = REK_G1_PINNED_ENCODER_SOURCE_SHA256,
            .output_sha256 =
                "b6d853a41f688e053d77db3c5a5f84539eb630f354787d090a9b4a1cd144cc97",
            .output_bytes = sizeof(ENCODER_PAYLOAD) - 1u,
        },
        .decoder = {
            .source_sha256 = REK_G1_PINNED_DECODER_SOURCE_SHA256,
            .output_sha256 =
                "841d03770417aba27ffecae85371b86dc694ebcc6048e4252dd0223477c275c0",
            .output_bytes = sizeof(DECODER_PAYLOAD) - 1u,
        },
    };
    return contract;
}

static void write_exact(
        const char* path,
        const unsigned char* payload,
        size_t payload_bytes) {
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(descriptor >= 0);
    size_t offset = 0u;
    while (offset < payload_bytes) {
        const ssize_t count = write(
            descriptor,
            payload + offset,
            payload_bytes - offset);
        if (count < 0 && errno == EINTR) continue;
        CHECK(count > 0);
        offset += (size_t)count;
    }
    CHECK(close(descriptor) == 0);
}

static RekG1ModelIdentityStatus verify(
        const RekG1ModelIdentityContract* contract,
        const char* encoder,
        const char* decoder,
        size_t batch,
        char error[256]) {
    memset(error, 0xa5, 256u);
    return rek_g1_model_identity_verify(
        contract,
        encoder,
        decoder,
        batch,
        error,
        256u);
}

static RekG1ModelIdentityStatus load_verified(
        const RekG1ModelIdentityContract* contract,
        const char* encoder,
        const char* decoder,
        size_t batch,
        RekG1VerifiedModelPair* verified_models,
        char error[256]) {
    memset(error, 0xa5, 256u);
    return rek_g1_model_identity_load_verified(
        contract,
        encoder,
        decoder,
        batch,
        verified_models,
        error,
        256u);
}

int main(void) {
    char directory_template[] = "/tmp/rek-g1-model-identity-XXXXXX";
    char* directory = mkdtemp(directory_template);
    CHECK(directory != NULL);
    char encoder_path[512];
    char decoder_path[512];
    char symlink_path[512];
    CHECK(snprintf(
        encoder_path,
        sizeof(encoder_path),
        "%s/encoder.onnx",
        directory) > 0);
    CHECK(snprintf(
        decoder_path,
        sizeof(decoder_path),
        "%s/decoder.onnx",
        directory) > 0);
    CHECK(snprintf(
        symlink_path,
        sizeof(symlink_path),
        "%s/encoder-link.onnx",
        directory) > 0);
    write_exact(
        encoder_path,
        ENCODER_PAYLOAD,
        sizeof(ENCODER_PAYLOAD) - 1u);
    write_exact(
        decoder_path,
        DECODER_PAYLOAD,
        sizeof(DECODER_PAYLOAD) - 1u);

    char error[256];
    RekG1ModelIdentityContract contract = valid_contract();
    CHECK(verify(&contract, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_OK);
    CHECK(error[0] == '\0');
    CHECK(strcmp(
        rek_g1_model_identity_status_string(REK_G1_MODEL_IDENTITY_OK),
        "ok") == 0);

    RekG1VerifiedModelPair verified_models = {0};
    CHECK(load_verified(
        &contract,
        encoder_path,
        decoder_path,
        8u,
        &verified_models,
        error) == REK_G1_MODEL_IDENTITY_OK);
    CHECK(error[0] == '\0');
    CHECK(verified_models.encoder.byte_count == sizeof(ENCODER_PAYLOAD) - 1u);
    CHECK(verified_models.decoder.byte_count == sizeof(DECODER_PAYLOAD) - 1u);
    CHECK(memcmp(
        verified_models.encoder.data,
        ENCODER_PAYLOAD,
        sizeof(ENCODER_PAYLOAD) - 1u) == 0);
    CHECK(memcmp(
        verified_models.decoder.data,
        DECODER_PAYLOAD,
        sizeof(DECODER_PAYLOAD) - 1u) == 0);

    unsigned char replaced_encoder[sizeof(ENCODER_PAYLOAD) - 1u];
    memcpy(replaced_encoder, ENCODER_PAYLOAD, sizeof(replaced_encoder));
    replaced_encoder[0] ^= 1u;
    write_exact(encoder_path, replaced_encoder, sizeof(replaced_encoder));
    CHECK(memcmp(
        verified_models.encoder.data,
        ENCODER_PAYLOAD,
        sizeof(ENCODER_PAYLOAD) - 1u) == 0);
    CHECK(load_verified(
        &contract,
        encoder_path,
        decoder_path,
        8u,
        &verified_models,
        error) == REK_G1_MODEL_IDENTITY_INVALID_ARGUMENT);
    CHECK(strstr(error, "already initialized") != NULL);
    CHECK(memcmp(
        verified_models.encoder.data,
        ENCODER_PAYLOAD,
        sizeof(ENCODER_PAYLOAD) - 1u) == 0);
    rek_g1_model_identity_release_verified(&verified_models);
    CHECK(verified_models.encoder.data == NULL);
    CHECK(verified_models.encoder.byte_count == 0u);
    CHECK(verified_models.decoder.data == NULL);
    CHECK(verified_models.decoder.byte_count == 0u);
    rek_g1_model_identity_release_verified(&verified_models);
    rek_g1_model_identity_release_verified(NULL);
    write_exact(
        encoder_path,
        ENCODER_PAYLOAD,
        sizeof(ENCODER_PAYLOAD) - 1u);

    CHECK(verify(&contract, encoder_path, decoder_path, 6u, error)
        == REK_G1_MODEL_IDENTITY_BATCH_MISMATCH);
    CHECK(strstr(error, "robot batch") != NULL);

    unsigned char tampered_encoder[sizeof(ENCODER_PAYLOAD) - 1u];
    memcpy(
        tampered_encoder,
        ENCODER_PAYLOAD,
        sizeof(tampered_encoder));
    tampered_encoder[0] ^= 1u;
    write_exact(encoder_path, tampered_encoder, sizeof(tampered_encoder));
    CHECK(verify(&contract, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_FILE_HASH_MISMATCH);
    CHECK(strstr(error, "SHA-256") != NULL);
    CHECK(load_verified(
        &contract,
        encoder_path,
        decoder_path,
        8u,
        &verified_models,
        error) == REK_G1_MODEL_IDENTITY_FILE_HASH_MISMATCH);
    CHECK(verified_models.encoder.data == NULL);
    CHECK(verified_models.encoder.byte_count == 0u);
    CHECK(verified_models.decoder.data == NULL);
    CHECK(verified_models.decoder.byte_count == 0u);
    write_exact(
        encoder_path,
        ENCODER_PAYLOAD,
        sizeof(ENCODER_PAYLOAD) - 1u);

    write_exact(
        decoder_path,
        DECODER_PAYLOAD,
        sizeof(DECODER_PAYLOAD) - 2u);
    CHECK(verify(&contract, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_FILE_SIZE_MISMATCH);
    CHECK(strstr(error, "byte count") != NULL);
    write_exact(
        decoder_path,
        DECODER_PAYLOAD,
        sizeof(DECODER_PAYLOAD) - 1u);

    CHECK(symlink(encoder_path, symlink_path) == 0);
    CHECK(verify(&contract, symlink_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_FILE_OPEN_FAILED);
    CHECK(strstr(error, "non-symlink") != NULL);

    RekG1ModelIdentityContract invalid = contract;
    invalid.encoder.source_sha256 =
        "113ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3";
    CHECK(verify(&invalid, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_CONTRACT);
    invalid = contract;
    invalid.current_steam_authority = 1u;
    CHECK(verify(&invalid, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_CONTRACT);
    invalid = contract;
    invalid.training_enabled = 1u;
    CHECK(verify(&invalid, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_CONTRACT);
    invalid = contract;
    invalid.rek_parity_claim = 1u;
    CHECK(verify(&invalid, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_CONTRACT);
    invalid = contract;
    invalid.schema = "rek.g1_gear_sonic_explicit_batch.v2";
    CHECK(verify(&invalid, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_CONTRACT);
    invalid = contract;
    invalid.manifest_sha256 =
        "14B876E6B05140F4037C9E985766FDB0B82C470F032599B33C497D1A5FBD745F";
    CHECK(verify(&invalid, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_CONTRACT);
    CHECK(verify(&contract, NULL, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_ARGUMENT);
    CHECK(verify(NULL, encoder_path, decoder_path, 8u, error)
        == REK_G1_MODEL_IDENTITY_INVALID_ARGUMENT);

    CHECK(unlink(symlink_path) == 0);
    CHECK(unlink(encoder_path) == 0);
    CHECK(unlink(decoder_path) == 0);
    CHECK(rmdir(directory) == 0);
    (void)printf("g1_model_identity: %u assertions passed\n", assertions);
    return 0;
}
