#ifndef PUFFERLIB_PRECISION_H
#define PUFFERLIB_PRECISION_H

#include <stdint.h>

#define PUF_MAX_DIMS 8

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} FloatTensor;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} ByteTensor;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} LongTensor;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} IntTensor;

#ifdef __CUDACC__
#include <cuda_bf16.h>

#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

#endif

#ifdef __CUDACC__
typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;
#else
#ifdef PRECISION_FLOAT
typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;
#else
typedef struct {
    uint16_t* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;
#endif
#endif

#endif
