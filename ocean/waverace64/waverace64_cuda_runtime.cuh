#pragma once

#include "waverace64_recomp_device.cuh"

// Host runtime services reached by the retained race-frame recompilation
// closure. They are device implementations, so the training hot path never
// leaves the Puffer CUDA stream.
__device__ void cop0_status_write(recomp_context* ctx, gpr value) {
    ctx->status_reg = (uint32_t)value;
}

__device__ gpr cop0_status_read(recomp_context* ctx) {
    return ctx->status_reg;
}

__device__ void switch_error(const char*, uint32_t vram, uint32_t table) {
    (void)vram;
    (void)table;
    asm("trap;");
}

__device__ void do_break(uint32_t vram) {
    (void)vram;
    asm("trap;");
}

__device__ void recomp_syscall_handler(
        uint8_t*, recomp_context*, int32_t) {
}

__device__ void pause_self(uint8_t*) {
}

__device__ void osContGetReadData_recomp(
        uint8_t* rdram, recomp_context* ctx) {
    uint32_t pad = (uint32_t)ctx->r4;
    WR64DeviceMachine* machine = ctx->machine;
    MEM_H(0, pad) = (int16_t)machine->pad_buttons;
    MEM_B(2, pad) = machine->pad_stick_x;
    MEM_B(3, pad) = machine->pad_stick_y;
    MEM_B(4, pad) = 0;
    ctx->r2 = 0;
}

__device__ void osGetTime_recomp(uint8_t*, recomp_context* ctx) {
    ctx->r2 = (int32_t)(ctx->machine->ticks >> 32);
    ctx->r3 = (int32_t)(uint32_t)ctx->machine->ticks;
}

__device__ void osInvalDCache_recomp(uint8_t*, recomp_context* ctx) {
    ctx->r2 = 0;
}

__device__ void osPiStartDma_recomp(uint8_t*, recomp_context* ctx) {
    // Cartridge DMA is a boot/menu service. Reaching it from the supported
    // resident-overlay race closure is an environment fault.
    ctx->machine->error = 2;
    ctx->r2 = -1;
}

__device__ void osRecvMesg_recomp(uint8_t* rdram, recomp_context* ctx) {
    uint32_t queue = (uint32_t)ctx->r4;
    uint32_t output = (uint32_t)ctx->r5;
    uint32_t valid = queue ? (uint32_t)MEM_W(8, queue) : 0;
    if (queue && valid) {
        uint32_t first = (uint32_t)MEM_W(12, queue);
        uint32_t count = (uint32_t)MEM_W(16, queue);
        uint32_t base = (uint32_t)MEM_W(20, queue);
        uint32_t divisor = count ? count : 1;
        uint32_t message = (uint32_t)MEM_W((first % divisor) * 4, base);
        if (output) MEM_W(0, output) = (int32_t)message;
        MEM_W(12, queue) = (int32_t)(count ? (first + 1) % count : 0);
        MEM_W(8, queue) = (int32_t)(valid - 1);
    } else if (output) {
        MEM_W(0, output) = 0;
    }
    ctx->r2 = 0;
}

__device__ void osSendMesg_recomp(uint8_t* rdram, recomp_context* ctx) {
    uint32_t queue = (uint32_t)ctx->r4;
    uint32_t message = (uint32_t)ctx->r5;
    if (!queue) {
        ctx->r2 = -1;
        return;
    }
    uint32_t valid = (uint32_t)MEM_W(8, queue);
    uint32_t first = (uint32_t)MEM_W(12, queue);
    uint32_t count = (uint32_t)MEM_W(16, queue);
    uint32_t base = (uint32_t)MEM_W(20, queue);
    if (!count || valid >= count) {
        ctx->r2 = -1;
        return;
    }
    MEM_W(((first + valid) % count) * 4, base) = (int32_t)message;
    MEM_W(8, queue) = (int32_t)(valid + 1);
    ctx->r2 = 0;
}

__device__ void proutSprintf_recomp(uint8_t* rdram, recomp_context* ctx) {
    uint32_t destination = (uint32_t)ctx->r4;
    uint32_t source = (uint32_t)ctx->r5;
    uint32_t size = (uint32_t)ctx->r6;
    for (uint32_t i = 0; i < size; i++) {
        MEM_B(i, destination) = MEM_B(i, source);
    }
    ctx->r2 = (int32_t)(destination + size);
}
