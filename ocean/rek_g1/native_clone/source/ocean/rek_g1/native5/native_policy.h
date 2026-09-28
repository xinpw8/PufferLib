#ifndef REK_NATIVE5_NATIVE_POLICY_H
#define REK_NATIVE5_NATIVE_POLICY_H
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#define REK_NATIVE_POLICY_ABI 2
enum RekNativePolicyPrecision { REK_NATIVE_POLICY_BF16 = 0, REK_NATIVE_POLICY_FP32 = 1 };
typedef struct RekNativePolicyConfig {
    uint32_t abi_version;
    const char* checkpoint_path;
    const char* expected_sha256; /* optional, actual digest is always exposed */
    int hidden_size;
    int num_layers;
    int batch;
    int precision;
    uint64_t seed;
    int legacy_fast_hidden; /* old src/models.cu polynomial negative-hidden branch */
} RekNativePolicyConfig;
typedef struct RekNativePolicy RekNativePolicy;
#ifdef __cplusplus
extern "C" {
#endif
/* Loads a flat FP32 checkpoint in pinned native5 encoder, decoder, MinGRU
 * order. Native5 uses BF16 inference; older FP32 runs must select it explicitly.
 * No checkpoint-layout or observation-encoding inference is performed. */
RekNativePolicy* rek_native_policy_create(const RekNativePolicyConfig*, cudaStream_t);
/* All arrays are device-resident. For one fighter in paired arrays use
 * offset=0 or 1, stride=2; for contiguous arrays use offset=0,stride=1.
 * Exactly config.batch rows are read/written. observations are already encoded
 * as the checkpoint requires. terminals reset recurrent state BEFORE inference.
 * deterministic!=0 is masked argmax; zero uses seeded Philox categorical draws.
 * Calls are allocation-free and capture-safe after successful creation. */
int rek_native_policy_step_rows(RekNativePolicy*, const float* observations,
    const uint8_t* masks, const float* terminals, float* actions,
    int row_offset, int row_stride, int deterministic, cudaStream_t);
int rek_native_policy_reset(RekNativePolicy*, cudaStream_t);
/* Legacy horizon resets clear only recurrent state, preserving sampler RNG. */
int rek_native_policy_reset_recurrent(RekNativePolicy*, cudaStream_t);
/* Reporting only, synchronizes. Invalid logits, masks or input remain failures. */
int rek_native_policy_check_status(RekNativePolicy*, cudaStream_t);
void rek_native_policy_destroy(RekNativePolicy*);
const char* rek_native_policy_sha256(const RekNativePolicy*);
const char* rek_native_policy_error(void);
/* Diagnostic device FP32 logits [batch][34], including the final value head.
 * Values are copied from the actual precision used in inference. */
const float* rek_native_policy_logits(const RekNativePolicy*);
#ifdef __cplusplus
}
#endif
#endif
