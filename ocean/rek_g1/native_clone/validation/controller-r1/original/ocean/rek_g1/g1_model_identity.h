#ifndef REK_G1_MODEL_IDENTITY_H
#define REK_G1_MODEL_IDENTITY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define REK_G1_EXPLICIT_BATCH_SCHEMA "rek.g1_gear_sonic_explicit_batch.v1"
#define REK_G1_PUBLIC_FAMILY_CLASSIFICATION "public_family_candidate"
#define REK_G1_PINNED_ENCODER_SOURCE_SHA256 \
    "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
#define REK_G1_PINNED_DECODER_SOURCE_SHA256 \
    "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"

typedef enum RekG1ModelIdentityStatus {
    REK_G1_MODEL_IDENTITY_OK = 0,
    REK_G1_MODEL_IDENTITY_INVALID_ARGUMENT,
    REK_G1_MODEL_IDENTITY_INVALID_CONTRACT,
    REK_G1_MODEL_IDENTITY_BATCH_MISMATCH,
    REK_G1_MODEL_IDENTITY_FILE_OPEN_FAILED,
    REK_G1_MODEL_IDENTITY_FILE_NOT_REGULAR,
    REK_G1_MODEL_IDENTITY_FILE_SIZE_MISMATCH,
    REK_G1_MODEL_IDENTITY_ALLOCATION_FAILED,
    REK_G1_MODEL_IDENTITY_FILE_READ_FAILED,
    REK_G1_MODEL_IDENTITY_DIGEST_FAILED,
    REK_G1_MODEL_IDENTITY_FILE_HASH_MISMATCH,
} RekG1ModelIdentityStatus;

typedef struct RekG1ModelFileIdentity {
    const char* source_sha256;
    const char* output_sha256;
    uint64_t output_bytes;
} RekG1ModelFileIdentity;

typedef struct RekG1ModelIdentityContract {
    const char* schema;
    const char* classification;
    const char* manifest_sha256;
    size_t robot_batch;
    uint8_t rek_parity_claim;
    uint8_t current_steam_authority;
    uint8_t training_enabled;
    RekG1ModelFileIdentity encoder;
    RekG1ModelFileIdentity decoder;
} RekG1ModelIdentityContract;

typedef struct RekG1VerifiedModelBytes {
    unsigned char* data;
    size_t byte_count;
} RekG1VerifiedModelBytes;

typedef struct RekG1VerifiedModelPair {
    RekG1VerifiedModelBytes encoder;
    RekG1VerifiedModelBytes decoder;
} RekG1VerifiedModelPair;

/*
 * Open each path once with O_NOFOLLOW, read the regular file through that
 * descriptor, and retain exactly the bytes whose size and SHA-256 match the
 * compiled contract. The output must be zero-initialized and is left empty on
 * failure. Release a successful output with
 * rek_g1_model_identity_release_verified.
 */
RekG1ModelIdentityStatus rek_g1_model_identity_load_verified(
    const RekG1ModelIdentityContract* contract,
    const char* encoder_path,
    const char* decoder_path,
    size_t actual_robot_batch,
    RekG1VerifiedModelPair* verified_models,
    char* error,
    size_t error_capacity);

void rek_g1_model_identity_release_verified(
    RekG1VerifiedModelPair* verified_models);

/*
 * Verify the exact regular files by loading and hashing descriptor-backed bytes
 * and then releasing them. Session construction should instead call
 * rek_g1_model_identity_load_verified and pass those returned bytes directly
 * to the inference runtime.
 */
RekG1ModelIdentityStatus rek_g1_model_identity_verify(
    const RekG1ModelIdentityContract* contract,
    const char* encoder_path,
    const char* decoder_path,
    size_t actual_robot_batch,
    char* error,
    size_t error_capacity);

const char* rek_g1_model_identity_status_string(
    RekG1ModelIdentityStatus status);

#ifdef __cplusplus
}
#endif

#endif
