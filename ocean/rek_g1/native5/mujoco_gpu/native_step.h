#pragma once
#include "model_data.h"
#include <memory>
#include <string>

namespace rek_mjgpu {
// Native launch schedule for the pinned rigid REK model. Construction performs
// GPU graph setup; model/state arrays remain owned by the caller. The runtime
// contains no Python, CPU dynamics, or CPU convergence loop.
class NativeStep {
public:
    NativeStep(ModelData&,const std::string& catalog,const std::string& conditional_ptx,
               const std::string& conditional_sha256);
    ~NativeStep();
    NativeStep(const NativeStep&)=delete;
    NativeStep& operator=(const NativeStep&)=delete;
    void step(CUstream) const;
    // Recompute current transforms and spatial velocity for state export after
    // integration/reset. Does not advance physics or generate new contacts.
    void refresh(CUstream) const;
    // Reset boundary: discard old contacts and regenerate from current poses.
    void refresh_contacts(CUstream) const;
    std::size_t phase_nodes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
