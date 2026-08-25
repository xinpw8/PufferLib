#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

typedef uint64_t gpr;

typedef union {
    double d;
    struct { float fl, fh; };
    struct { uint32_t u32l, u32h; };
    uint64_t u64;
} fpr;

typedef struct WR64DeviceMachine {
    uint64_t ticks;
    uint16_t pad_buttons;
    int8_t pad_stick_x;
    int8_t pad_stick_y;
    int32_t resident_overlay;
    int32_t rounding_mode;
    int32_t error;
    uint32_t indirect_target;
} WR64DeviceMachine;

typedef struct recomp_context {
    gpr r0,  r1,  r2,  r3,  r4,  r5,  r6,  r7;
    gpr r8,  r9,  r10, r11, r12, r13, r14, r15;
    gpr r16, r17, r18, r19, r20, r21, r22, r23;
    gpr r24, r25, r26, r27, r28, r29, r30, r31;
    fpr f0,  f1,  f2,  f3,  f4,  f5,  f6,  f7;
    fpr f8,  f9,  f10, f11, f12, f13, f14, f15;
    fpr f16, f17, f18, f19, f20, f21, f22, f23;
    fpr f24, f25, f26, f27, f28, f29, f30, f31;
    uint64_t hi, lo;
    uint32_t* f_odd;
    uint32_t status_reg;
    uint8_t mips3_float_mode;
    WR64DeviceMachine* machine;
} recomp_context;

#define SIGNED(v) ((int64_t)(v))
#define S32(v) ((int32_t)(v))
#define U32(v) ((uint32_t)(v))
#define S64(v) ((int64_t)(v))
#define U64(v) ((uint64_t)(v))
#define ADD32(a, b) ((gpr)(int32_t)((a) + (b)))
#define SUB32(a, b) ((gpr)(int32_t)((a) - (b)))

#define WR64_DEVICE_OFFSET(address) ((uint32_t)(address) - UINT32_C(0x80000000))
#define MEM_W(offset, reg) \
    (*(int32_t*)(rdram + WR64_DEVICE_OFFSET((reg) + (offset))))
#define MEM_H(offset, reg) \
    (*(int16_t*)(rdram + WR64_DEVICE_OFFSET(((reg) + (offset)) ^ 2u)))
#define MEM_B(offset, reg) \
    (*(int8_t*)(rdram + WR64_DEVICE_OFFSET(((reg) + (offset)) ^ 3u)))
#define MEM_HU(offset, reg) \
    (*(uint16_t*)(rdram + WR64_DEVICE_OFFSET(((reg) + (offset)) ^ 2u)))
#define MEM_BU(offset, reg) \
    (*(uint8_t*)(rdram + WR64_DEVICE_OFFSET(((reg) + (offset)) ^ 3u)))

#define SD(val, offset, reg) do { \
    MEM_W((offset) + 4, (reg)) = (uint32_t)((gpr)(val)); \
    MEM_W((offset), (reg)) = (uint32_t)((gpr)(val) >> 32); \
} while (0)

__device__ static inline uint64_t load_doubleword(
        uint8_t* rdram, gpr reg, gpr offset) {
    uint64_t lo = (uint64_t)(uint32_t)MEM_W(reg, offset + 4);
    uint64_t hi = (uint64_t)(uint32_t)MEM_W(reg, offset);
    return lo | (hi << 32);
}

#define LD(offset, reg) load_doubleword(rdram, (offset), (reg))

__device__ static inline gpr do_lwl(
        uint8_t* rdram, gpr initial, gpr offset, gpr reg) {
    gpr address = offset + reg;
    uint32_t loaded = MEM_W(0, address & ~UINT64_C(3));
    gpr shift = address & 3u;
    gpr masked = initial & (gpr)(uint32_t)~(UINT32_C(0xFFFFFFFF) << (shift * 8));
    return (gpr)(int32_t)(masked | ((gpr)loaded << (shift * 8)));
}

__device__ static inline gpr do_lwr(
        uint8_t* rdram, gpr initial, gpr offset, gpr reg) {
    gpr address = offset + reg;
    uint32_t loaded = MEM_W(0, address & ~UINT64_C(3));
    gpr shift = address & 3u;
    gpr masked = initial & (gpr)(uint32_t)~(UINT32_C(0xFFFFFFFF) >> (24 - shift * 8));
    return (gpr)(int32_t)(masked | ((gpr)loaded >> (24 - shift * 8)));
}

__device__ static inline void do_swl(
        uint8_t* rdram, gpr offset, gpr reg, gpr value) {
    gpr address = offset + reg;
    gpr aligned = address & ~UINT64_C(3);
    gpr shift = address & 3u;
    uint32_t initial = MEM_W(0, aligned);
    MEM_W(0, aligned) = (initial & ~(UINT32_C(0xFFFFFFFF) >> (shift * 8)))
        | ((uint32_t)value >> (shift * 8));
}

__device__ static inline void do_swr(
        uint8_t* rdram, gpr offset, gpr reg, gpr value) {
    gpr address = offset + reg;
    gpr aligned = address & ~UINT64_C(3);
    gpr shift = address & 3u;
    uint32_t initial = MEM_W(0, aligned);
    MEM_W(0, aligned) = (initial & ~(UINT32_C(0xFFFFFFFF) << (24 - shift * 8)))
        | ((uint32_t)value << (24 - shift * 8));
}

__device__ static inline gpr do_ldl(
        uint8_t* rdram, gpr initial, gpr offset, gpr reg) {
    gpr address = offset + reg;
    gpr shift = address & 7u;
    uint64_t loaded = load_doubleword(rdram, 0, address & ~UINT64_C(7));
    return (initial & ~(UINT64_MAX << (shift * 8))) | (loaded << (shift * 8));
}

__device__ static inline gpr do_ldr(
        uint8_t* rdram, gpr initial, gpr offset, gpr reg) {
    gpr address = offset + reg;
    gpr shift = address & 7u;
    uint64_t loaded = load_doubleword(rdram, 0, address & ~UINT64_C(7));
    return (initial & ~(UINT64_MAX >> (56 - shift * 8)))
        | (loaded >> (56 - shift * 8));
}

__device__ static inline void do_sdl(
        uint8_t* rdram, gpr offset, gpr reg, gpr value) {
    gpr address = offset + reg;
    gpr aligned = address & ~UINT64_C(7);
    gpr shift = address & 7u;
    uint64_t initial = load_doubleword(rdram, 0, aligned);
    uint64_t result = (initial & ~(UINT64_MAX >> (shift * 8)))
        | ((uint64_t)value >> (shift * 8));
    MEM_W(4, aligned) = (uint32_t)result;
    MEM_W(0, aligned) = (uint32_t)(result >> 32);
}

__device__ static inline void do_sdr(
        uint8_t* rdram, gpr offset, gpr reg, gpr value) {
    gpr address = offset + reg;
    gpr aligned = address & ~UINT64_C(7);
    gpr shift = address & 7u;
    uint64_t initial = load_doubleword(rdram, 0, aligned);
    uint64_t result = (initial & ~(UINT64_MAX << (56 - shift * 8)))
        | ((uint64_t)value << (56 - shift * 8));
    MEM_W(4, aligned) = (uint32_t)result;
    MEM_W(0, aligned) = (uint32_t)(result >> 32);
}

#define MUL_S(a, b) ((a) * (b))
#define MUL_D(a, b) ((a) * (b))
#define DIV_S(a, b) ((a) / (b))
#define DIV_D(a, b) ((a) / (b))
#define CVT_S_W(v) ((float)(int32_t)(v))
#define CVT_D_W(v) ((double)(int32_t)(v))
#define CVT_D_L(v) ((double)(int64_t)(v))
#define CVT_S_L(v) ((float)(int64_t)(v))
#define CVT_D_S(v) ((double)(v))
#define CVT_S_D(v) ((float)(v))
#define TRUNC_W_S(v) ((int32_t)(v))
#define TRUNC_W_D(v) ((int32_t)(v))
#define TRUNC_L_S(v) ((int64_t)(v))
#define TRUNC_L_D(v) ((int64_t)(v))
#define DEFAULT_ROUNDING_MODE 0

__device__ static inline int32_t wr64_device_cvt_w_s(
        recomp_context* ctx, float value) {
    switch (ctx->machine->rounding_mode & 3) {
        case 1: return __float2int_rz(value);
        case 2: return __float2int_ru(value);
        case 3: return __float2int_rd(value);
        default: return __float2int_rn(value);
    }
}

__device__ static inline int32_t wr64_device_cvt_w_d(
        recomp_context* ctx, double value) {
    switch (ctx->machine->rounding_mode & 3) {
        case 1: return __double2int_rz(value);
        case 2: return __double2int_ru(value);
        case 3: return __double2int_rd(value);
        default: return __double2int_rn(value);
    }
}

#define CVT_W_S(v) wr64_device_cvt_w_s(ctx, (v))
#define CVT_W_D(v) wr64_device_cvt_w_d(ctx, (v))
#define CVT_L_S(v) ((int64_t)wr64_device_cvt_w_s(ctx, (v)))
#define CVT_L_D(v) ((int64_t)wr64_device_cvt_w_d(ctx, (v)))
#define CHECK_FR(ctx_value, index) ((void)0)
#define NAN_CHECK(value) ((void)0)

#define LO16(x) ((x) & 0xFFFF)
#define HI16(x) (((x) >> 16) + (((x) >> 15) & 1))
#define RELOC_HI16(section_index, offset) \
    HI16(wr64_device_section_addresses[(section_index)] + (offset))
#define RELOC_LO16(section_index, offset) \
    LO16(wr64_device_section_addresses[(section_index)] + (offset))

__device__ static inline uint32_t wr64_device_get_cop1_cs(
        recomp_context* ctx) {
    return (uint32_t)ctx->machine->rounding_mode;
}

__device__ static inline void wr64_device_set_cop1_cs(
        recomp_context* ctx, uint32_t value) {
    ctx->machine->rounding_mode = (int32_t)(value & 3u);
}

__device__ static inline void wr64_device_context_init(
        recomp_context* ctx, WR64DeviceMachine* machine) {
    uint8_t* bytes = (uint8_t*)ctx;
    for (size_t i = 0; i < sizeof(*ctx); i++) bytes[i] = 0;
    ctx->f_odd = &ctx->f0.u32h;
    ctx->machine = machine;
}

__device__ void wr64_device_lookup(
    int32_t vram, uint8_t* rdram, recomp_context* ctx);

__device__ void cop0_status_write(recomp_context* ctx, gpr value);
__device__ gpr cop0_status_read(recomp_context* ctx);
__device__ void switch_error(const char* function, uint32_t vram, uint32_t table);
__device__ void do_break(uint32_t vram);
__device__ void recomp_syscall_handler(
    uint8_t* rdram, recomp_context* ctx, int32_t instruction_vram);
__device__ void pause_self(uint8_t* rdram);
