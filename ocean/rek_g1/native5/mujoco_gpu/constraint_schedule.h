#pragma once
#include "schedule.h"

namespace rek_mjgpu {
// Installed MuJoCo-Warp constraint kernels. The supported model deliberately
// has no equality, tendon, flex, or ball-limit constraints.
void append_constraints(ScheduleSpec&, ModelData&);
struct ConstraintCapacityReport {
    int collisions=0,contacts=0,max_rows=0,max_nonzeros=0;
    int first_invalid_world=-1;
    bool overflow=false;
};
// Explicit diagnostic boundary only. Synchronizes and reads integer counters;
// never call this from a captured training hot loop or treat it as physics.
ConstraintCapacityReport inspect_constraint_capacity(const ModelData&, CUstream=nullptr);
}
