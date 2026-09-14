#pragma once
#include "warp_abi.h"
#include <cuda.h>
#include <mujoco/mujoco.h>
#include <map>
#include <string>
#include <vector>

namespace rek_mjgpu {
enum class Element { F32, I32, U8 };
struct ArrayStorage {
    WarpArray view;
    Element element = Element::F32;
    int components = 1;
    std::size_t bytes = 0;
    bool owns_memory = true;
};
struct ModelDataConfig {
    int nworld = 1;
    int nconmax = 128;
    int njmax = 512;
    // Explicit dense upper bound for sparse storage; no state is truncated.
    int njmax_nnz = 0;
};
// Caller owns the current CUDA context and the source mjModel. Construction
// packs model constants/topology and allocates GPU arrays; it does not invoke
// Python, mj_step, mj_forward, or any CPU physics state evolution.
class ModelData {
public:
    ModelData(const mjModel* model, const ModelDataConfig& config);
    ~ModelData();
    ModelData(const ModelData&) = delete;
    ModelData& operator=(const ModelData&) = delete;
    WarpArray array(const std::string& name) const;
    const WarpArray& array_ref(const std::string& name) const;
    const int& integer_ref(const std::string& name) const;
    const float& scalar_ref(const std::string& name) const;
    int integer(const std::string& name) const;
    float scalar(const std::string& name) const;
    std::size_t byte_size(const std::string& name) const;
    bool contains(const std::string& name) const;
    const std::map<std::string,ArrayStorage>& arrays() const { return arrays_; }
    const std::map<std::string,int>& integers() const { return integers_; }
    const std::vector<int>& geometry_pair_type_counts() const { return pair_type_counts_; }
    // Schedule-owned persistent scratch, allocated before graph capture.
    WarpArray allocate(const std::string& name, const std::vector<int>& shape,
                       Element element=Element::F32, int components=1, bool broadcast=false);
    // Same storage, different contiguous dimensions; registry owns it once.
    WarpArray reshape_alias(const std::string& name, const std::string& source,
                            const std::vector<int>& shape);
    void upload(const std::string& name, const void* data, std::size_t bytes);
    void zero(const std::string& name, CUstream stream=nullptr);
    void set_initial_state(const float* qpos, const float* qvel=nullptr);
private:
    std::map<std::string,ArrayStorage> arrays_;
    std::map<std::string,int> integers_;
    std::map<std::string,float> scalars_;
    std::vector<int> pair_type_counts_;
    void build(const mjModel*,const ModelDataConfig&);
};
}
