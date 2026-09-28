#define _POSIX_C_SOURCE 200809L

#include "g1_model_identity.h"

#include <errno.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#error "the REK G1 model-identity gate requires O_CLOEXEC"
#endif

#ifndef O_NOFOLLOW
#error "the REK G1 model-identity gate requires O_NOFOLLOW"
#endif

enum {
    REK_G1_SHA256_BYTES = 32,
    REK_G1_SHA256_HEX_CHARS = 64,
    REK_G1_READ_CHUNK_BYTES = 64 * 1024,
};

static void set_error(char* error, size_t capacity, const char* format, ...) {
    if (error == NULL || capacity == 0u) return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
    error[capacity - 1u] = '\0';
}

static int is_lowercase_sha256(const char* text) {
    if (text == NULL || strlen(text) != REK_G1_SHA256_HEX_CHARS) return 0;
    unsigned int nonzero = 0u;
    for (size_t index = 0u; index < REK_G1_SHA256_HEX_CHARS; index++) {
        const unsigned char value = (unsigned char)text[index];
        const int valid = (value >= (unsigned char)'0' && value <= (unsigned char)'9')
            || (value >= (unsigned char)'a' && value <= (unsigned char)'f');
        if (!valid) return 0;
        nonzero |= (unsigned int)(value != (unsigned char)'0');
    }
    return nonzero != 0u;
}

static int valid_file_contract(
        const RekG1ModelFileIdentity* file,
        const char* pinned_source_sha256) {
    return file != NULL
        && file->output_bytes > 0u
        && is_lowercase_sha256(file->source_sha256)
        && is_lowercase_sha256(file->output_sha256)
        && strcmp(file->source_sha256, pinned_source_sha256) == 0;
}

static int valid_contract(const RekG1ModelIdentityContract* contract) {
    return contract != NULL
        && contract->schema != NULL
        && strcmp(contract->schema, REK_G1_EXPLICIT_BATCH_SCHEMA) == 0
        && contract->classification != NULL
        && strcmp(
            contract->classification,
            REK_G1_PUBLIC_FAMILY_CLASSIFICATION) == 0
        && is_lowercase_sha256(contract->manifest_sha256)
        && contract->robot_batch > 0u
        && contract->robot_batch % 2u == 0u
        && contract->rek_parity_claim == 0u
        && contract->current_steam_authority == 0u
        && contract->training_enabled == 0u
        && valid_file_contract(
            &contract->encoder,
            REK_G1_PINNED_ENCODER_SOURCE_SHA256)
        && valid_file_contract(
            &contract->decoder,
            REK_G1_PINNED_DECODER_SOURCE_SHA256);
}

static void digest_to_hex(
        const unsigned char digest[REK_G1_SHA256_BYTES],
        char output[REK_G1_SHA256_HEX_CHARS + 1]) {
    static const char HEX[] = "0123456789abcdef";
    for (size_t index = 0u; index < REK_G1_SHA256_BYTES; index++) {
        output[2u * index] = HEX[digest[index] >> 4u];
        output[2u * index + 1u] = HEX[digest[index] & 0x0fu];
    }
    output[REK_G1_SHA256_HEX_CHARS] = '\0';
}

static RekG1ModelIdentityStatus load_verified_file(
        const char* role,
        const char* path,
        const RekG1ModelFileIdentity* expected,
        RekG1VerifiedModelBytes* output,
        char* error,
        size_t error_capacity) {
    const int descriptor = open(
        path,
        O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) {
        set_error(
            error,
            error_capacity,
            "%s model is not an openable non-symlink file: %s",
            role,
            strerror(errno));
        return REK_G1_MODEL_IDENTITY_FILE_OPEN_FAILED;
    }

    struct stat metadata;
    if (fstat(descriptor, &metadata) != 0 || !S_ISREG(metadata.st_mode)) {
        set_error(error, error_capacity, "%s model is not a regular file", role);
        (void)close(descriptor);
        return REK_G1_MODEL_IDENTITY_FILE_NOT_REGULAR;
    }
    if (metadata.st_size < 0
            || (uint64_t)metadata.st_size != expected->output_bytes) {
        set_error(
            error,
            error_capacity,
            "%s model byte count differs from the compiled manifest",
            role);
        (void)close(descriptor);
        return REK_G1_MODEL_IDENTITY_FILE_SIZE_MISMATCH;
    }
    if (expected->output_bytes > (uint64_t)SIZE_MAX) {
        set_error(
            error,
            error_capacity,
            "%s model is too large for this process",
            role);
        (void)close(descriptor);
        return REK_G1_MODEL_IDENTITY_FILE_SIZE_MISMATCH;
    }

    const size_t byte_count = (size_t)expected->output_bytes;
    unsigned char* bytes = (unsigned char*)malloc(byte_count);
    if (bytes == NULL) {
        set_error(error, error_capacity, "%s model allocation failed", role);
        (void)close(descriptor);
        return REK_G1_MODEL_IDENTITY_ALLOCATION_FAILED;
    }

    EVP_MD_CTX* digest_context = EVP_MD_CTX_new();
    if (digest_context == NULL
            || EVP_DigestInit_ex(digest_context, EVP_sha256(), NULL) != 1) {
        set_error(error, error_capacity, "%s SHA-256 setup failed", role);
        EVP_MD_CTX_free(digest_context);
        free(bytes);
        (void)close(descriptor);
        return REK_G1_MODEL_IDENTITY_DIGEST_FAILED;
    }

    RekG1ModelIdentityStatus status = REK_G1_MODEL_IDENTITY_OK;
    size_t offset = 0u;
    while (offset < byte_count) {
        size_t remaining = byte_count - offset;
        if (remaining > REK_G1_READ_CHUNK_BYTES) {
            remaining = REK_G1_READ_CHUNK_BYTES;
        }
        const ssize_t count = read(descriptor, bytes + offset, remaining);
        if (count > 0) {
            if (EVP_DigestUpdate(
                    digest_context, bytes + offset, (size_t)count) != 1) {
                set_error(error, error_capacity, "%s SHA-256 update failed", role);
                status = REK_G1_MODEL_IDENTITY_DIGEST_FAILED;
                break;
            }
            offset += (size_t)count;
            continue;
        }
        if (count == 0) {
            set_error(
                error,
                error_capacity,
                "%s model ended before its verified byte count",
                role);
            status = REK_G1_MODEL_IDENTITY_FILE_READ_FAILED;
            break;
        }
        if (errno == EINTR) continue;
        set_error(
            error,
            error_capacity,
            "%s model read failed: %s",
            role,
            strerror(errno));
        status = REK_G1_MODEL_IDENTITY_FILE_READ_FAILED;
        break;
    }

    unsigned char digest[REK_G1_SHA256_BYTES];
    unsigned int digest_bytes = 0u;
    if (status == REK_G1_MODEL_IDENTITY_OK
            && (EVP_DigestFinal_ex(
                    digest_context,
                    digest,
                    &digest_bytes) != 1
                || digest_bytes != REK_G1_SHA256_BYTES)) {
        set_error(error, error_capacity, "%s SHA-256 finalization failed", role);
        status = REK_G1_MODEL_IDENTITY_DIGEST_FAILED;
    }
    EVP_MD_CTX_free(digest_context);
    (void)close(descriptor);

    if (status == REK_G1_MODEL_IDENTITY_OK) {
        char actual_sha256[REK_G1_SHA256_HEX_CHARS + 1];
        digest_to_hex(digest, actual_sha256);
        if (strcmp(actual_sha256, expected->output_sha256) != 0) {
            set_error(
                error,
                error_capacity,
                "%s model SHA-256 differs from the compiled manifest",
                role);
            status = REK_G1_MODEL_IDENTITY_FILE_HASH_MISMATCH;
        }
    }
    if (status != REK_G1_MODEL_IDENTITY_OK) {
        free(bytes);
        return status;
    }
    output->data = bytes;
    output->byte_count = byte_count;
    return status;
}

void rek_g1_model_identity_release_verified(
        RekG1VerifiedModelPair* verified_models) {
    if (verified_models == NULL) return;
    free(verified_models->decoder.data);
    free(verified_models->encoder.data);
    memset(verified_models, 0, sizeof(*verified_models));
}

RekG1ModelIdentityStatus rek_g1_model_identity_load_verified(
        const RekG1ModelIdentityContract* contract,
        const char* encoder_path,
        const char* decoder_path,
        size_t actual_robot_batch,
        RekG1VerifiedModelPair* verified_models,
        char* error,
        size_t error_capacity) {
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    if (contract == NULL || encoder_path == NULL || encoder_path[0] == '\0'
            || decoder_path == NULL || decoder_path[0] == '\0'
            || verified_models == NULL) {
        set_error(error, error_capacity, "model identity verification has a null argument");
        return REK_G1_MODEL_IDENTITY_INVALID_ARGUMENT;
    }
    if (verified_models->encoder.data != NULL
            || verified_models->encoder.byte_count != 0u
            || verified_models->decoder.data != NULL
            || verified_models->decoder.byte_count != 0u) {
        set_error(
            error,
            error_capacity,
            "verified model output is already initialized");
        return REK_G1_MODEL_IDENTITY_INVALID_ARGUMENT;
    }
    if (!valid_contract(contract)) {
        set_error(error, error_capacity, "compiled model identity contract is invalid");
        return REK_G1_MODEL_IDENTITY_INVALID_CONTRACT;
    }
    if (actual_robot_batch != contract->robot_batch) {
        set_error(
            error,
            error_capacity,
            "runtime robot batch differs from the compiled explicit-batch manifest");
        return REK_G1_MODEL_IDENTITY_BATCH_MISMATCH;
    }

    RekG1ModelIdentityStatus status = load_verified_file(
        "encoder",
        encoder_path,
        &contract->encoder,
        &verified_models->encoder,
        error,
        error_capacity);
    if (status != REK_G1_MODEL_IDENTITY_OK) return status;
    status = load_verified_file(
        "decoder",
        decoder_path,
        &contract->decoder,
        &verified_models->decoder,
        error,
        error_capacity);
    if (status != REK_G1_MODEL_IDENTITY_OK) {
        rek_g1_model_identity_release_verified(verified_models);
    }
    return status;
}

RekG1ModelIdentityStatus rek_g1_model_identity_verify(
        const RekG1ModelIdentityContract* contract,
        const char* encoder_path,
        const char* decoder_path,
        size_t actual_robot_batch,
        char* error,
        size_t error_capacity) {
    RekG1VerifiedModelPair verified_models = {0};
    const RekG1ModelIdentityStatus status =
        rek_g1_model_identity_load_verified(
            contract,
            encoder_path,
            decoder_path,
            actual_robot_batch,
            &verified_models,
            error,
            error_capacity);
    rek_g1_model_identity_release_verified(&verified_models);
    return status;
}

const char* rek_g1_model_identity_status_string(
        RekG1ModelIdentityStatus status) {
    switch (status) {
        case REK_G1_MODEL_IDENTITY_OK: return "ok";
        case REK_G1_MODEL_IDENTITY_INVALID_ARGUMENT: return "invalid_argument";
        case REK_G1_MODEL_IDENTITY_INVALID_CONTRACT: return "invalid_contract";
        case REK_G1_MODEL_IDENTITY_BATCH_MISMATCH: return "batch_mismatch";
        case REK_G1_MODEL_IDENTITY_FILE_OPEN_FAILED: return "file_open_failed";
        case REK_G1_MODEL_IDENTITY_FILE_NOT_REGULAR: return "file_not_regular";
        case REK_G1_MODEL_IDENTITY_FILE_SIZE_MISMATCH: return "file_size_mismatch";
        case REK_G1_MODEL_IDENTITY_ALLOCATION_FAILED: return "allocation_failed";
        case REK_G1_MODEL_IDENTITY_FILE_READ_FAILED: return "file_read_failed";
        case REK_G1_MODEL_IDENTITY_DIGEST_FAILED: return "digest_failed";
        case REK_G1_MODEL_IDENTITY_FILE_HASH_MISMATCH: return "file_hash_mismatch";
        default: return "unknown";
    }
}
