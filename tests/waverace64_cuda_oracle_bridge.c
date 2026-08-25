#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "wr_env.h"

typedef struct WR64CudaOracle {
    WRMachine machine;
} WR64CudaOracle;

void wr_init_overlay_table(void);
void wr_dma_copy(uint8_t*, uint32_t, const uint8_t*, uint32_t);

void* wr64_cuda_oracle_create(const char* rom_path) {
    WR64CudaOracle* oracle = (WR64CudaOracle*)calloc(1, sizeof(*oracle));
    if (oracle == NULL) return NULL;
    if (wr_machine_init(&oracle->machine, rom_path) != 0) {
        free(oracle);
        return NULL;
    }
    wr_current = &oracle->machine;
    wr_init_overlay_table();
    wr_dma_copy(oracle->machine.rdram, UINT32_C(0x80046800),
        oracle->machine.rom + 0x1000, 0xA95D0 - 0x1000);
    if (wr_boot_to_race(&oracle->machine, 8000) < 0) {
        wr_machine_free(&oracle->machine);
        free(oracle);
        return NULL;
    }
    return oracle;
}

int wr64_cuda_oracle_copy_rdram(void* opaque, uint8_t* destination) {
    WR64CudaOracle* oracle = (WR64CudaOracle*)opaque;
    if (oracle == NULL || destination == NULL) return -1;
    memcpy(destination, oracle->machine.rdram, WR_RDRAM_SIZE);
    return 0;
}

uint64_t wr64_cuda_oracle_ticks(void* opaque) {
    WR64CudaOracle* oracle = (WR64CudaOracle*)opaque;
    return oracle ? oracle->machine.ticks : 0;
}

int wr64_cuda_oracle_overlay(void* opaque) {
    WR64CudaOracle* oracle = (WR64CudaOracle*)opaque;
    return oracle ? oracle->machine.resident_overlay : -1;
}

int wr64_cuda_oracle_step(void* opaque, int8_t stick_x, int8_t stick_y,
        uint8_t a, uint8_t b, uint8_t z, uint8_t r) {
    WR64CudaOracle* oracle = (WR64CudaOracle*)opaque;
    if (oracle == NULL) return -1;
    WRPad pad = {
        .stick_x = stick_x,
        .stick_y = stick_y,
        .a = a,
        .b = b,
        .z = z,
        .r = r,
    };
    return (int)wr_env_step(&oracle->machine, &pad, 1);
}

void wr64_cuda_oracle_destroy(void* opaque) {
    WR64CudaOracle* oracle = (WR64CudaOracle*)opaque;
    if (oracle == NULL) return;
    wr_machine_free(&oracle->machine);
    free(oracle);
}
