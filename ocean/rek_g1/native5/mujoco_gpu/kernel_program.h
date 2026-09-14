#pragma once
#include "schedule.h"
#include <memory>
#include <string>

namespace rek_mjgpu {

// Startup binds all metadata, arguments and persistent GPU addresses. launch()
// submits native Driver operations only and is CUDA-graph capture compatible.
// ModelData and the loading CUDA context must outlive this program.
class KernelProgram {
public:
    KernelProgram(ModelData&, const std::string& catalog_path, const ScheduleSpec&);
    ~KernelProgram();
    KernelProgram(const KernelProgram&) = delete;
    KernelProgram& operator=(const KernelProgram&) = delete;
    void launch(CUstream stream) const;
    // Diagnostics can synchronize between individual nodes to locate a fault.
    void launch_one(std::size_t index, CUstream stream) const;
    // Graph construction only. All body nodes are native GPU operations.
    CUgraphNode append_to_graph(CUgraph graph, CUgraphNode predecessor=nullptr) const;
    std::size_t nodes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
