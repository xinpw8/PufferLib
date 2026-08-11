#define PRECISION_FLOAT
#define ENV_HEADER "../ocean/minimal/minimal.h"
#define PUFFER_ENV_NAME "minimal"
#include "../src/pufferl.cu"

extern "C" {

static Encoder test_encoder;
static ColosseumEntityEncoderWeights* test_weights;
static ColosseumEntityEncoderActivations* test_activations;
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

void osrs_entity_test_contract(int kind, int* values) {
    if (kind == 0) {
        values[0] = COLO_ENT_OBS_SIZE;
        values[1] = COLO_ENT_NPC_START;
        values[2] = COLO_ENT_NUM_NPCS;
        values[3] = COLO_ENT_FEATS;
        values[4] = COLO_ENT_TYPE_ONEHOT;
        values[5] = COLO_ENT_INV_START;
        values[6] = COLO_ENT_INV_NUM_CELLS;
        values[7] = COLO_ENT_INV_FEATS;
        return;
    }
    values[0] = INF_ENT_OBS_SIZE;
    values[1] = INF_ENT_NPC_START;
    values[2] = INF_ENT_NUM_NPCS;
    values[3] = INF_ENT_FEATS;
    values[4] = INF_ENT_TYPE_ONEHOT;
    values[5] = INF_ENT_INV_START;
    values[6] = INF_ENT_INV_NUM_CELLS;
    values[7] = INF_ENT_INV_FEATS;
}

int osrs_entity_test_init(int kind, int batch, int obs_size, int hidden) {
    if (kind == 0 && obs_size != COLO_ENT_OBS_SIZE) return 1;
    if (kind == 1 && obs_size != INF_ENT_OBS_SIZE) return 1;
    test_reset();
    if (!test_cublas_initialized) {
        cublas_init_handle();
        test_cublas_initialized = true;
    }
    test_encoder = {};
    test_encoder.in_dim = obs_size;
    test_encoder.out_dim = hidden;
    create_custom_encoder(kind == 0 ? "osrs_colosseum" : "osrs_inferno", &test_encoder);
    test_weights = (ColosseumEntityEncoderWeights*)test_encoder.create_weights(&test_encoder);
    test_encoder.reg_params(test_weights, &test_params);
    alloc_create(&test_params);
    test_activations = (ColosseumEntityEncoderActivations*)calloc(1, test_encoder.activation_size);
    test_encoder.reg_train(test_weights, test_activations, &test_acts, &test_grads, batch);
    alloc_create(&test_acts);
    alloc_create(&test_grads);
    return 0;
}

void osrs_entity_test_set_weights(
        void* global_w, void* entity_l1_w, void* entity_l2_w,
        void* inv_l1_w, void* inv_l2_w) {
    Prec* dst[] = {
        &test_weights->global_w,
        &test_weights->entity_l1_w,
        &test_weights->entity_l2_w,
        &test_weights->inv_l1_w,
        &test_weights->inv_l2_w,
    };
    void* src[] = {global_w, entity_l1_w, entity_l2_w, inv_l1_w, inv_l2_w};
    for (int i = 0; i < 5; i++) {
        cudaMemcpy(dst[i]->data, src[i], numel(dst[i]->shape) * sizeof(float),
            cudaMemcpyDeviceToDevice);
    }
}

void osrs_entity_test_forward(void* output, void* observations, int batch, int obs_size) {
    Prec input = {.data = (precision_t*)observations, .shape = {batch, obs_size}};
    Prec result = test_encoder.forward(test_weights, test_activations, input, 0);
    cudaMemcpy(output, result.data, (int64_t)batch * test_encoder.out_dim * sizeof(float),
        cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void osrs_entity_test_backward(void* grad, int batch, int hidden) {
    Prec output_grad = {.data = (precision_t*)grad, .shape = {batch, hidden}};
    test_encoder.backward(test_weights, test_activations, output_grad, 0);
    cudaDeviceSynchronize();
}

void osrs_entity_test_get_grad(int index, void* output) {
    Prec* grads[] = {
        &test_activations->global_wgrad,
        &test_activations->entity_l1_wgrad,
        &test_activations->entity_l2_wgrad,
        &test_activations->inv_l1_wgrad,
        &test_activations->inv_l2_wgrad,
    };
    Prec* grad = grads[index];
    cudaMemcpy(output, grad->data, numel(grad->shape) * sizeof(float),
        cudaMemcpyDeviceToDevice);
}

}
