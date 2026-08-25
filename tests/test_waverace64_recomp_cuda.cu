#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/waverace64/waverace64_cuda_runtime.cuh"
#include "waverace64_recomp_device.inc"

extern "C" {
void* wr64_cuda_oracle_create(const char* rom_path);
int wr64_cuda_oracle_copy_rdram(void* oracle, uint8_t* destination);
uint64_t wr64_cuda_oracle_ticks(void* oracle);
int wr64_cuda_oracle_overlay(void* oracle);
int wr64_cuda_oracle_step(void* oracle, int8_t stick_x, int8_t stick_y,
    uint8_t a, uint8_t b, uint8_t z, uint8_t r);
void wr64_cuda_oracle_destroy(void* oracle);
}

#define WR64_RDRAM_SIZE (8u * 1024u * 1024u)
#define WR64_PHYSICS_OFFSET (UINT32_C(0x80192690) - UINT32_C(0x80000000))
#define WR64_PHYSICS_BYTES (UINT32_C(0x1718) * 4u)
#define WR64_RIDER_OFFSET (UINT32_C(0x801C2938) - UINT32_C(0x80000000))
#define WR64_RIDER_BYTES (UINT32_C(0x0378) * 4u)
#define WR64_HELPER_OFFSET (UINT32_C(0x801C3C60) - UINT32_C(0x80000000))
#define WR64_HELPER_BYTES UINT32_C(0xE8)
#define WR64_WATER_OFFSET (UINT32_C(0x80162420) - UINT32_C(0x80000000))
#define WR64_WATER_BYTES (384u * 128u * 4u)
#define WR64_GAME_STATE_OFFSET (UINT32_C(0x800DAB24) - UINT32_C(0x80000000))

__global__ void wr64_recomp_one_frame(
        uint8_t* rdram, WR64DeviceMachine* machine,
        int8_t stick_x, int8_t stick_y, uint16_t buttons) {
    if (blockIdx.x || threadIdx.x) return;
    machine->pad_buttons = buttons;
    machine->pad_stick_x = stick_x;
    machine->pad_stick_y = stick_y;
    recomp_context ctx;
    wr64_device_context_init(&ctx, machine);
    ctx.r29 = (gpr)(int64_t)(int32_t)UINT32_C(0x80153140);
    func_800922E4(rdram, &ctx);
    func_i1_802C5DF4(rdram, &ctx);
    MEM_W(0, UINT32_C(0x80151960)) += 1;
}

static size_t first_difference(
        const uint8_t* expected, const uint8_t* actual,
        size_t offset, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (expected[offset + i] != actual[offset + i]) return offset + i;
    }
    return SIZE_MAX;
}

static int compare_region(const char* name,
        const uint8_t* expected, const uint8_t* actual,
        size_t offset, size_t size) {
    size_t difference = first_difference(expected, actual, offset, size);
    if (difference == SIZE_MAX) {
        printf("PASS %s %zu bytes\n", name, size);
        return 0;
    }
    printf("FAIL %s offset=0x%zx expected=%02x actual=%02x\n",
        name, difference, expected[difference], actual[difference]);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s ROM\n", argv[0]);
        return 2;
    }
    void* oracle = wr64_cuda_oracle_create(argv[1]);
    if (!oracle) {
        fprintf(stderr, "failed to boot oracle\n");
        return 3;
    }
    uint8_t* initial = (uint8_t*)malloc(WR64_RDRAM_SIZE);
    uint8_t* expected = (uint8_t*)malloc(WR64_RDRAM_SIZE);
    uint8_t* actual = (uint8_t*)malloc(WR64_RDRAM_SIZE);
    if (!initial || !expected || !actual) return 4;
    if (wr64_cuda_oracle_copy_rdram(oracle, initial) != 0) return 5;
    if (wr64_cuda_oracle_step(oracle, 20, 56, 1, 0, 0, 0) < 0) return 6;
    if (wr64_cuda_oracle_copy_rdram(oracle, expected) != 0) return 7;

    uint8_t* device_rdram = NULL;
    WR64DeviceMachine* device_machine = NULL;
    cudaMalloc((void**)&device_rdram, WR64_RDRAM_SIZE);
    cudaMalloc((void**)&device_machine, sizeof(WR64DeviceMachine));
    cudaError_t stack_status = cudaDeviceSetLimit(cudaLimitStackSize, 256u * 1024u);
    if (stack_status != cudaSuccess) {
        fprintf(stderr, "cannot set CUDA stack: %s\n", cudaGetErrorString(stack_status));
        return 8;
    }
    cudaMemcpy(device_rdram, initial, WR64_RDRAM_SIZE, cudaMemcpyHostToDevice);
    WR64DeviceMachine machine = {};
    machine.ticks = wr64_cuda_oracle_ticks(oracle);
    machine.pad_buttons = UINT16_C(0x8000);
    machine.pad_stick_x = 20;
    machine.pad_stick_y = 56;
    machine.resident_overlay = wr64_cuda_oracle_overlay(oracle);
    cudaMemcpy(device_machine, &machine, sizeof(machine), cudaMemcpyHostToDevice);
    wr64_recomp_one_frame<<<1, 1>>>(
        device_rdram, device_machine, 20, 56, UINT16_C(0x8000));
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(status));
        return 8;
    }
    cudaMemcpy(actual, device_rdram, WR64_RDRAM_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(&machine, device_machine, sizeof(machine), cudaMemcpyDeviceToHost);
    printf("device_error=%d indirect_target=0x%08x\n",
        machine.error, machine.indirect_target);

    int failures = 0;
    failures += compare_region("physics", expected, actual,
        WR64_PHYSICS_OFFSET, WR64_PHYSICS_BYTES);
    failures += compare_region("rider", expected, actual,
        WR64_RIDER_OFFSET, WR64_RIDER_BYTES);
    failures += compare_region("helper", expected, actual,
        WR64_HELPER_OFFSET, WR64_HELPER_BYTES);
    failures += compare_region("water", expected, actual,
        WR64_WATER_OFFSET, WR64_WATER_BYTES);
    failures += compare_region("game_state", expected, actual,
        WR64_GAME_STATE_OFFSET, sizeof(uint32_t));

    uint32_t lcg = UINT32_C(0x12345678);
    int exact_frames = 1;
    int terminal_seen = 0;
    for (int frame = 1; failures == 0 && frame < 800; frame++) {
        lcg = lcg * UINT32_C(1664525) + UINT32_C(1013904223);
        int8_t stick_x = (int8_t)((int32_t)((lcg >> 24) & 0x7f) - 63);
        int8_t stick_y = (int8_t)(48 + (int32_t)((lcg >> 16) & 0x1f));
        uint8_t b = (uint8_t)((lcg & UINT32_C(0x1ff)) == 0);
        uint8_t r = (uint8_t)((lcg & UINT32_C(0x3ff)) == 1);
        uint16_t buttons = UINT16_C(0x8000)
            | (b ? UINT16_C(0x4000) : 0)
            | (r ? UINT16_C(0x0010) : 0);
        wr64_cuda_oracle_step(oracle, stick_x, stick_y, 1, b, 0, r);
        wr64_cuda_oracle_copy_rdram(oracle, expected);
        wr64_recomp_one_frame<<<1, 1>>>(
            device_rdram, device_machine, stick_x, stick_y, buttons);
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            fprintf(stderr, "frame %d kernel failed: %s\n",
                frame, cudaGetErrorString(status));
            failures++;
            break;
        }
        cudaMemcpy(actual + WR64_PHYSICS_OFFSET,
            device_rdram + WR64_PHYSICS_OFFSET, WR64_PHYSICS_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(actual + WR64_RIDER_OFFSET,
            device_rdram + WR64_RIDER_OFFSET, WR64_RIDER_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(actual + WR64_HELPER_OFFSET,
            device_rdram + WR64_HELPER_OFFSET, WR64_HELPER_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(actual + WR64_WATER_OFFSET,
            device_rdram + WR64_WATER_OFFSET, WR64_WATER_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(actual + WR64_GAME_STATE_OFFSET,
            device_rdram + WR64_GAME_STATE_OFFSET, sizeof(uint32_t),
            cudaMemcpyDeviceToHost);
        if (first_difference(expected, actual,
                WR64_PHYSICS_OFFSET, WR64_PHYSICS_BYTES) != SIZE_MAX
                || first_difference(expected, actual,
                    WR64_RIDER_OFFSET, WR64_RIDER_BYTES) != SIZE_MAX
                || first_difference(expected, actual,
                    WR64_HELPER_OFFSET, WR64_HELPER_BYTES) != SIZE_MAX
                || first_difference(expected, actual,
                    WR64_WATER_OFFSET, WR64_WATER_BYTES) != SIZE_MAX
                || first_difference(expected, actual,
                    WR64_GAME_STATE_OFFSET, sizeof(uint32_t)) != SIZE_MAX) {
            printf("FAIL multi-frame parity at frame %d\n", frame);
            failures++;
            break;
        }
        exact_frames++;
        uint32_t game_state;
        memcpy(&game_state, expected + WR64_GAME_STATE_OFFSET,
            sizeof(game_state));
        if (game_state == UINT32_C(0x29)
                || game_state == UINT32_C(0x2A)) {
            terminal_seen = 1;
            printf("terminal game_state=0x%02x frame=%d\n",
                game_state, exact_frames);
            break;
        }
    }
    if (!terminal_seen) {
        printf("FAIL terminal was not reached\n");
        failures++;
    }
    printf("%s terminal-boundary parity frames=%d\n",
        failures ? "FAIL" : "PASS", exact_frames);

    cudaFree(device_machine);
    cudaFree(device_rdram);
    free(actual);
    free(expected);
    free(initial);
    wr64_cuda_oracle_destroy(oracle);
    return failures || machine.error ? 1 : 0;
}
