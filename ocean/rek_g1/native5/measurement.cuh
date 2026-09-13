#pragma once
#include "physics.cuh"
#include "device_storage.cuh"

namespace rek5 {
struct CombatMeasurement {
    DeviceStorage storage;
    void* descriptor = nullptr;
    Physics* physics = nullptr;
    int roots[2]{}, left[2]{}, right[2]{};
    float floor_half_height = 0;
    float *upright = nullptr, *standing = nullptr;
    uint8_t *calibrated = nullptr;
    float *fall_floats = nullptr;
    int64_t *fall_integers = nullptr;
    uint8_t *fall_valid = nullptr;
    int64_t *hit_integers = nullptr, *order = nullptr, *offsets = nullptr, *counts = nullptr;
    float *hit_floats = nullptr;
    uint8_t *candidate_valid = nullptr, *scan_valid = nullptr;
    int64_t *indices = nullptr, *sorted_keys = nullptr;
    void* sort_temp = nullptr;
    size_t sort_bytes = 0;
    void* scan_temp = nullptr;
    size_t scan_bytes = 0;
    ~CombatMeasurement();
};
CombatMeasurement* measurement_create(Physics* physics);
void measurement_reset(CombatMeasurement*, cudaStream_t);
void measurement_clear_contacts(CombatMeasurement*, const uint8_t* arena_mask, cudaStream_t);
void measurement_sample(CombatMeasurement*, int substep, cudaStream_t);
// Replaces fall facts only for reset rows, preserving the last substep for others.
void measurement_sample_reset_fall(CombatMeasurement*, const uint8_t* rows, cudaStream_t);
}
