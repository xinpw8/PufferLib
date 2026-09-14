#pragma once
#include <cuda.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rek_mjgpu {
// Executes body only when a device mask contains at least one nonzero byte.
// No mask or condition is read by the host. Body is a graph-construction
// callback and must submit all its operations on the supplied stream.
class DeviceIf {
public:
    DeviceIf(const std::string& ptx,const std::string& sha256);
    ~DeviceIf();
    DeviceIf(const DeviceIf&)=delete;
    DeviceIf& operator=(const DeviceIf&)=delete;
    // Active capture inserts IF directly into the caller's graph. Otherwise
    // a graph is cached by mask pointer/count and launched on the caller's
    // stream. For a given key the callback's operations and addresses must
    // stay fixed. Calls on one instance must be serialized with its world.
    void execute(CUstream,const std::uint8_t* mask,int count,
                 const std::function<void(CUstream)>& body);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
