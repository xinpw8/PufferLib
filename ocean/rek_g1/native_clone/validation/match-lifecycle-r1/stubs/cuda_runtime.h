#pragma once
#include "cuda_runtime_api.h"
#define __global__
#define __device__
#define __constant__
struct Dim {unsigned x=1;}; static Dim blockIdx{0},blockDim{1},threadIdx{0},gridDim{1};
