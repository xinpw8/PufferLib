#pragma once
#include "native_module.h"
#include <functional>
#include <string>

namespace rek_mjgpu {
// Inserts a GPU-controlled while node during graph construction. No CPU reads
// the condition, and no Python/host callback participates in loop execution.
// Body builder appends a single ordered chain and returns its last node.
class DeviceWhile {
public:
    DeviceWhile(const std::string& ptx,const std::string& sha256);
    ~DeviceWhile();
    DeviceWhile(const DeviceWhile&)=delete;
    DeviceWhile& operator=(const DeviceWhile&)=delete;
    void capture(CUstream stream,CUdeviceptr condition,
                 const std::function<CUgraphNode(CUgraph)>& build_body)const;
private:
    RekMjGpuModule* module_=nullptr;
    CUfunction setter_=nullptr;
};
}
