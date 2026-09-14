#pragma once
#include "schedule.h"

namespace rek_mjgpu {
// Actual installed NXN broadphase, convex GJK/EPA, and primitive narrowphase.
// Allocations occur only while constructing the schedule, before CUDA capture.
// After execution, d.ncollision, d.nacon, and collision.nccd must be checked
// against their capacities by the caller; overflowing trials are invalid.
void append_collision(ScheduleSpec&, ModelData&);
}
