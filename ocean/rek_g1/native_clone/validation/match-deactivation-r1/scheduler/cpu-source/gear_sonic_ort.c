#include "gear_sonic_ort.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void gear_sonic_set_error(
    char* error,
    size_t capacity,
    const char* operation,
    const char* detail
) {
    if (error == NULL || capacity == 0) return;
    if (detail == NULL) detail = "unknown ONNX Runtime error";
    snprintf(error, capacity, "%s: %s", operation, detail);
    error[capacity - 1] = '\0';
}

static int gear_sonic_status_ok(
    GearSonicOrtBatch* batch,
    OrtStatus* status,
    const char* operation,
    char* error,
    size_t error_capacity
) {
    if (status == NULL) return 1;
    const char* detail = batch->api->GetErrorMessage(status);
    gear_sonic_set_error(error, error_capacity, operation, detail);
    batch->api->ReleaseStatus(status);
    return 0;
}

static int gear_sonic_size_product(
    size_t left,
    size_t right,
    size_t* result
) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int gear_sonic_validate_tensor(
    GearSonicOrtBatch* batch,
    OrtSession* session,
    int is_input,
    const char* expected_name,
    int64_t expected_width,
    char* error,
    size_t error_capacity
) {
    size_t count = 0;
    OrtStatus* status = is_input
        ? batch->api->SessionGetInputCount(session, &count)
        : batch->api->SessionGetOutputCount(session, &count);
    if (!gear_sonic_status_ok(
            batch, status, "read graph tensor count", error, error_capacity)) {
        return 0;
    }
    if (count != 1) {
        gear_sonic_set_error(
            error, error_capacity, "validate graph", "expected exactly one tensor"
        );
        return 0;
    }

    OrtAllocator* allocator = NULL;
    if (!gear_sonic_status_ok(
            batch,
            batch->api->GetAllocatorWithDefaultOptions(&allocator),
            "get default allocator",
            error,
            error_capacity)) {
        return 0;
    }
    char* name = NULL;
    status = is_input
        ? batch->api->SessionGetInputName(session, 0, allocator, &name)
        : batch->api->SessionGetOutputName(session, 0, allocator, &name);
    if (!gear_sonic_status_ok(
            batch, status, "read graph tensor name", error, error_capacity)) {
        return 0;
    }
    int name_matches = name != NULL && strcmp(name, expected_name) == 0;
    allocator->Free(allocator, name);
    if (!name_matches) {
        gear_sonic_set_error(
            error, error_capacity, "validate graph", "tensor name mismatch"
        );
        return 0;
    }

    OrtTypeInfo* type_info = NULL;
    status = is_input
        ? batch->api->SessionGetInputTypeInfo(session, 0, &type_info)
        : batch->api->SessionGetOutputTypeInfo(session, 0, &type_info);
    if (!gear_sonic_status_ok(
            batch, status, "read graph tensor type", error, error_capacity)) {
        return 0;
    }
    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
    int ok = gear_sonic_status_ok(
        batch,
        batch->api->CastTypeInfoToTensorInfo(type_info, &tensor_info),
        "read graph tensor shape",
        error,
        error_capacity
    );
    if (!ok || tensor_info == NULL) {
        batch->api->ReleaseTypeInfo(type_info);
        if (ok) {
            gear_sonic_set_error(
                error, error_capacity, "validate graph", "value is not a tensor"
            );
        }
        return 0;
    }

    ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    int64_t dimensions[2] = {0, 0};
    ok = gear_sonic_status_ok(
        batch,
        batch->api->GetTensorElementType(tensor_info, &element_type),
        "read graph element type",
        error,
        error_capacity
    );
    if (ok) {
        ok = gear_sonic_status_ok(
            batch,
            batch->api->GetDimensionsCount(tensor_info, &rank),
            "read graph rank",
            error,
            error_capacity
        );
    }
    if (ok && rank == 2) {
        ok = gear_sonic_status_ok(
            batch,
            batch->api->GetDimensions(tensor_info, dimensions, 2),
            "read graph dimensions",
            error,
            error_capacity
        );
    }
    batch->api->ReleaseTypeInfo(type_info);
    if (!ok) return 0;
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || rank != 2
            || dimensions[0] != (int64_t)batch->batch_size
            || dimensions[1] != expected_width) {
        gear_sonic_set_error(
            error,
            error_capacity,
            "validate graph",
            "tensor type or exact batch shape mismatch"
        );
        return 0;
    }
    return 1;
}

static int gear_sonic_validate_session(
    GearSonicOrtBatch* batch,
    OrtSession* session,
    const char* input_name,
    int64_t input_width,
    const char* output_name,
    int64_t output_width,
    char* error,
    size_t error_capacity
) {
    return gear_sonic_validate_tensor(
            batch,
            session,
            1,
            input_name,
            input_width,
            error,
            error_capacity)
        && gear_sonic_validate_tensor(
            batch,
            session,
            0,
            output_name,
            output_width,
            error,
            error_capacity);
}

void gear_sonic_ort_close(GearSonicOrtBatch* batch) {
    if (batch == NULL) return;
    if (batch->api != NULL) {
        if (batch->cpu_memory != NULL) {
            batch->api->ReleaseMemoryInfo(batch->cpu_memory);
        }
        if (batch->decoder != NULL) batch->api->ReleaseSession(batch->decoder);
        if (batch->encoder != NULL) batch->api->ReleaseSession(batch->encoder);
        if (batch->session_options != NULL) {
            batch->api->ReleaseSessionOptions(batch->session_options);
        }
        if (batch->environment != NULL) batch->api->ReleaseEnv(batch->environment);
    }
    memset(batch, 0, sizeof(*batch));
}

static int gear_sonic_ort_open_internal(
    GearSonicOrtBatch* batch,
    const char* encoder_path,
    const char* decoder_path,
    const void* encoder_data,
    size_t encoder_byte_count,
    const void* decoder_data,
    size_t decoder_byte_count,
    int from_memory,
    size_t batch_size,
    char* error,
    size_t error_capacity
) {
    if (batch == NULL
            || (from_memory
                ? encoder_data == NULL || encoder_byte_count == 0u
                    || decoder_data == NULL || decoder_byte_count == 0u
                : encoder_path == NULL || decoder_path == NULL)) {
        gear_sonic_set_error(error, error_capacity, "open", "null argument");
        return 0;
    }
    if (batch->api != NULL
            || batch->environment != NULL
            || batch->session_options != NULL
            || batch->encoder != NULL
            || batch->decoder != NULL
            || batch->cpu_memory != NULL) {
        gear_sonic_set_error(
            error, error_capacity, "open", "batch is already initialized"
        );
        return 0;
    }
    memset(batch, 0, sizeof(*batch));
    if (batch_size == 0 || batch_size > INT64_MAX) {
        gear_sonic_set_error(error, error_capacity, "open", "invalid batch size");
        return 0;
    }
    batch->batch_size = batch_size;

    const OrtApiBase* api_base = OrtGetApiBase();
    if (api_base == NULL || api_base->GetApi == NULL) {
        gear_sonic_set_error(error, error_capacity, "open", "OrtGetApiBase failed");
        return 0;
    }
    const char* version = api_base->GetVersionString();
    if (version == NULL || strcmp(version, GEAR_SONIC_ORT_VERSION) != 0) {
        gear_sonic_set_error(
            error, error_capacity, "open", "ONNX Runtime version mismatch"
        );
        return 0;
    }
    batch->api = api_base->GetApi(ORT_API_VERSION);
    if (batch->api == NULL) {
        gear_sonic_set_error(error, error_capacity, "open", "C API version unavailable");
        return 0;
    }

    if (!gear_sonic_status_ok(
            batch,
            batch->api->CreateEnv(
                ORT_LOGGING_LEVEL_WARNING,
                "rek-g1-gear-sonic",
                &batch->environment),
            "create environment",
            error,
            error_capacity)
        || !gear_sonic_status_ok(
            batch,
            batch->api->CreateSessionOptions(&batch->session_options),
            "create session options",
            error,
            error_capacity)
        || !gear_sonic_status_ok(
            batch,
            batch->api->SetIntraOpNumThreads(batch->session_options, 1),
            "set intra-op threads",
            error,
            error_capacity)
        || !gear_sonic_status_ok(
            batch,
            batch->api->SetInterOpNumThreads(batch->session_options, 1),
            "set inter-op threads",
            error,
            error_capacity)
        || !gear_sonic_status_ok(
            batch,
            batch->api->SetSessionExecutionMode(
                batch->session_options, ORT_SEQUENTIAL),
            "set sequential execution",
            error,
            error_capacity)) {
        gear_sonic_ort_close(batch);
        return 0;
    }

    OrtStatus* encoder_session_status = from_memory
        ? batch->api->CreateSessionFromArray(
            batch->environment,
            encoder_data,
            encoder_byte_count,
            batch->session_options,
            &batch->encoder)
        : batch->api->CreateSession(
            batch->environment,
            encoder_path,
            batch->session_options,
            &batch->encoder);
    int opened_encoder = gear_sonic_status_ok(
        batch,
        encoder_session_status,
        "create encoder session",
        error,
        error_capacity);
    OrtStatus* decoder_session_status = NULL;
    if (opened_encoder) {
        decoder_session_status = from_memory
            ? batch->api->CreateSessionFromArray(
                batch->environment,
                decoder_data,
                decoder_byte_count,
                batch->session_options,
                &batch->decoder)
            : batch->api->CreateSession(
                batch->environment,
                decoder_path,
                batch->session_options,
                &batch->decoder);
    }

    if (!opened_encoder
        || !gear_sonic_status_ok(
            batch,
            decoder_session_status,
            "create decoder session",
            error,
            error_capacity)
        || !gear_sonic_status_ok(
            batch,
            batch->api->CreateCpuMemoryInfo(
                OrtArenaAllocator, OrtMemTypeDefault, &batch->cpu_memory),
            "create CPU memory info",
            error,
            error_capacity)
        || !gear_sonic_validate_session(
            batch,
            batch->encoder,
            "obs_dict",
            GEAR_SONIC_ENCODER_INPUT_WIDTH,
            "encoded_tokens",
            GEAR_SONIC_ENCODER_OUTPUT_WIDTH,
            error,
            error_capacity)
        || !gear_sonic_validate_session(
            batch,
            batch->decoder,
            "obs_dict",
            GEAR_SONIC_DECODER_INPUT_WIDTH,
            "action",
            GEAR_SONIC_DECODER_OUTPUT_WIDTH,
            error,
            error_capacity)) {
        gear_sonic_ort_close(batch);
        return 0;
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_ort_open(
        GearSonicOrtBatch* batch,
        const char* encoder_path,
        const char* decoder_path,
        size_t batch_size,
        char* error,
        size_t error_capacity) {
    return gear_sonic_ort_open_internal(
        batch,
        encoder_path,
        decoder_path,
        NULL,
        0u,
        NULL,
        0u,
        0,
        batch_size,
        error,
        error_capacity);
}

int gear_sonic_ort_open_from_memory(
        GearSonicOrtBatch* batch,
        const void* encoder_data,
        size_t encoder_byte_count,
        const void* decoder_data,
        size_t decoder_byte_count,
        size_t batch_size,
        char* error,
        size_t error_capacity) {
    return gear_sonic_ort_open_internal(
        batch,
        NULL,
        NULL,
        encoder_data,
        encoder_byte_count,
        decoder_data,
        decoder_byte_count,
        1,
        batch_size,
        error,
        error_capacity);
}

static int gear_sonic_run(
    GearSonicOrtBatch* batch,
    OrtSession* session,
    const char* input_name,
    size_t input_width,
    float* input,
    const char* output_name,
    size_t output_width,
    float* output,
    char* error,
    size_t error_capacity
) {
    if (batch == NULL || batch->api == NULL || session == NULL
            || input == NULL || output == NULL) {
        gear_sonic_set_error(error, error_capacity, "run", "uninitialized argument");
        return 0;
    }
    size_t input_count = 0;
    size_t output_count = 0;
    if (!gear_sonic_size_product(batch->batch_size, input_width, &input_count)
            || !gear_sonic_size_product(batch->batch_size, output_width, &output_count)
            || input_count > SIZE_MAX / sizeof(float)
            || output_count > SIZE_MAX / sizeof(float)) {
        gear_sonic_set_error(error, error_capacity, "run", "tensor size overflow");
        return 0;
    }
    for (size_t index = 0; index < input_count; index++) {
        if (!isfinite(input[index])) {
            gear_sonic_set_error(error, error_capacity, "run", "non-finite input");
            return 0;
        }
    }

    int64_t input_shape[2] = {(int64_t)batch->batch_size, (int64_t)input_width};
    int64_t output_shape[2] = {(int64_t)batch->batch_size, (int64_t)output_width};
    OrtValue* input_value = NULL;
    OrtValue* output_value = NULL;
    int ok = gear_sonic_status_ok(
        batch,
        batch->api->CreateTensorWithDataAsOrtValue(
            batch->cpu_memory,
            input,
            input_count * sizeof(float),
            input_shape,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_value),
        "create input tensor",
        error,
        error_capacity
    );
    if (ok) {
        ok = gear_sonic_status_ok(
            batch,
            batch->api->CreateTensorWithDataAsOrtValue(
                batch->cpu_memory,
                output,
                output_count * sizeof(float),
                output_shape,
                2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &output_value),
            "create output tensor",
            error,
            error_capacity
        );
    }
    if (ok) {
        const char* input_names[1] = {input_name};
        const char* output_names[1] = {output_name};
        const OrtValue* input_values[1] = {input_value};
        OrtValue* output_values[1] = {output_value};
        ok = gear_sonic_status_ok(
            batch,
            batch->api->Run(
                session,
                NULL,
                input_names,
                input_values,
                1,
                output_names,
                1,
                output_values),
            "run graph",
            error,
            error_capacity
        );
    }
    if (output_value != NULL) batch->api->ReleaseValue(output_value);
    if (input_value != NULL) batch->api->ReleaseValue(input_value);
    if (!ok) return 0;

    for (size_t index = 0; index < output_count; index++) {
        if (!isfinite(output[index])) {
            gear_sonic_set_error(error, error_capacity, "run", "non-finite output");
            return 0;
        }
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_ort_encode(
    GearSonicOrtBatch* batch,
    float* observations,
    float* tokens,
    char* error,
    size_t error_capacity
) {
    return gear_sonic_run(
        batch,
        batch == NULL ? NULL : batch->encoder,
        "obs_dict",
        GEAR_SONIC_ENCODER_INPUT_WIDTH,
        observations,
        "encoded_tokens",
        GEAR_SONIC_ENCODER_OUTPUT_WIDTH,
        tokens,
        error,
        error_capacity
    );
}

int gear_sonic_ort_decode(
    GearSonicOrtBatch* batch,
    float* observations,
    float* actions,
    char* error,
    size_t error_capacity
) {
    return gear_sonic_run(
        batch,
        batch == NULL ? NULL : batch->decoder,
        "obs_dict",
        GEAR_SONIC_DECODER_INPUT_WIDTH,
        observations,
        "action",
        GEAR_SONIC_DECODER_OUTPUT_WIDTH,
        actions,
        error,
        error_capacity
    );
}
