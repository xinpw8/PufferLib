#define PRECISION_FLOAT
#define ENV_HEADER "../ocean/minimal/minimal.h"
#define PUFFER_ENV_NAME "minimal"
#define NUM_GEAR_SLOTS 11
#include "../src/pufferl.cu"

extern "C" {

static Encoder test_encoder;
static OsrsEntityEncoderWeights* test_weights;
static OsrsEntityEncoderActivations* test_activations;
static Allocator test_params;
static Allocator test_acts;
static Allocator test_grads;
static bool test_cublas_initialized;

static void test_free_allocator(Allocator* allocator) {
    if (allocator->mem) cudaFree(allocator->mem);
    free(allocator->regs);
    *allocator = {};
}

static void test_reset() {
    test_free_allocator(&test_params);
    test_free_allocator(&test_acts);
    test_free_allocator(&test_grads);
    free(test_weights);
    free(test_activations);
    test_weights = nullptr;
    test_activations = nullptr;
}

void osrs_entity_test_init(int kind, int batch, int obs_size, int hidden) {
    test_reset();
    if (!test_cublas_initialized) {
        cublas_init_handle();
        test_cublas_initialized = true;
    }
    test_encoder = {};
    test_encoder.in_dim = obs_size;
    test_encoder.out_dim = hidden;
    const char* env_name = kind == 0 ? "osrs_colosseum" : "osrs_inferno";
    create_custom_encoder(env_name, &test_encoder);
    test_weights = (OsrsEntityEncoderWeights*)test_encoder.create_weights(&test_encoder);
    test_encoder.reg_params(test_weights, &test_params);
    alloc_create(&test_params);
    test_activations = (OsrsEntityEncoderActivations*)calloc(
        1, test_encoder.activation_size);
    test_encoder.reg_train(
        test_weights, test_activations, &test_acts, &test_grads, batch);
    alloc_create(&test_acts);
    alloc_create(&test_grads);
}

void osrs_entity_test_set_weights(
        void* global_w, void* inventory_l1_w, void* inventory_l2_w,
        void* equipment_l1_w, void* equipment_l2_w,
        void* npc_l1_w, void* npc_l2_w) {
    OsrsEntityBranchWeights* branches = osrs_entity_branch_weights(test_weights);
    Prec* dst[] = {
        &test_weights->global_w,
        &test_weights->inv_l1_w,
        &test_weights->inv_l2_w,
        &branches[0].l1_w,
        &branches[0].l2_w,
        &branches[1].l1_w,
        &branches[1].l2_w,
    };
    void* src[] = {
        global_w,
        inventory_l1_w,
        inventory_l2_w,
        equipment_l1_w,
        equipment_l2_w,
        npc_l1_w,
        npc_l2_w,
    };
    for (int i = 0; i < 7; i++) {
        cudaMemcpy(dst[i]->data, src[i], numel(dst[i]->shape) * sizeof(float),
            cudaMemcpyDeviceToDevice);
    }
}

void osrs_entity_test_forward(void* output, void* observations, int batch, int obs_size) {
    Prec input = {.data = (precision_t*)observations, .shape = {batch, obs_size}};
    Prec result = test_encoder.forward(test_weights, test_activations, input, 0);
    cudaMemcpy(output, result.data, (int64_t)batch * test_encoder.out_dim * sizeof(float),
        cudaMemcpyDeviceToDevice);
}

void osrs_entity_test_backward(void* grad, int batch, int hidden) {
    Prec output_grad = {.data = (precision_t*)grad, .shape = {batch, hidden}};
    test_encoder.backward(test_weights, test_activations, output_grad, 0);
}

void osrs_entity_test_get_grad(int index, void* output) {
    OsrsEntityBranchActivations* branches =
        osrs_entity_branch_activations(test_activations);
    Prec* grads[] = {
        &test_activations->global_wgrad,
        &test_activations->inv_l1_wgrad,
        &test_activations->inv_l2_wgrad,
        &branches[0].l1_wgrad,
        &branches[0].l2_wgrad,
        &branches[1].l1_wgrad,
        &branches[1].l2_wgrad,
    };
    Prec* grad = grads[index];
    cudaMemcpy(output, grad->data, numel(grad->shape) * sizeof(float),
        cudaMemcpyDeviceToDevice);
}

}
