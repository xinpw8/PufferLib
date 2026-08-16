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

static const OsrsEntityEncoderDescriptor* test_descriptor(int kind) {
    return kind == 0
        ? &OSRS_COLOSSEUM_ENTITY_DESCRIPTOR
        : &OSRS_INFERNO_ENTITY_DESCRIPTOR;
}

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
    const OsrsEntityEncoderDescriptor* descriptor = test_descriptor(kind);
    const OsrsEntityBranchDescriptor* equipment = &descriptor->branches[0];
    const OsrsEntityBranchDescriptor* npc = &descriptor->branches[1];
    values[0] = descriptor->obs_size;
    values[1] = npc->obs_start;
    values[2] = npc->num_records;
    values[3] = npc->obs_features;
    values[4] = osrs_entity_branch_features(npc);
    values[5] = npc->type_onehot;
    values[6] = npc->type_code_scale;
    values[7] = equipment->obs_start;
    values[8] = equipment->num_records;
    values[9] = equipment->obs_features;
    values[10] = osrs_entity_branch_features(equipment);
    values[11] = OSRS_ENTITY_INV_START;
    values[12] = OSRS_ENTITY_INV_NUM_RECORDS;
    values[13] = OSRS_ENTITY_INV_FEATURES;
}

int osrs_entity_test_init(int kind, int batch, int obs_size, int hidden) {
    const OsrsEntityEncoderDescriptor* descriptor = test_descriptor(kind);
    if (obs_size != descriptor->obs_size) return 1;
    test_reset();
    if (!test_cublas_initialized) {
        cublas_init_handle();
        test_cublas_initialized = true;
    }
    test_encoder = {};
    test_encoder.in_dim = obs_size;
    test_encoder.out_dim = hidden;
    create_custom_encoder(descriptor->env_name, &test_encoder);
    test_weights = (OsrsEntityEncoderWeights*)test_encoder.create_weights(&test_encoder);
    test_encoder.reg_params(test_weights, &test_params);
    alloc_create(&test_params);
    test_activations = (OsrsEntityEncoderActivations*)calloc(1, test_encoder.activation_size);
    test_encoder.reg_train(test_weights, test_activations, &test_acts, &test_grads, batch);
    alloc_create(&test_acts);
    alloc_create(&test_grads);
    return 0;
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
    cudaDeviceSynchronize();
}

void osrs_entity_test_backward(void* grad, int batch, int hidden) {
    Prec output_grad = {.data = (precision_t*)grad, .shape = {batch, hidden}};
    test_encoder.backward(test_weights, test_activations, output_grad, 0);
    cudaDeviceSynchronize();
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
