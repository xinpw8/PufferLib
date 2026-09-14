#pragma once
#include "schedule.h"

namespace rek_mjgpu {
// Exact sparse Newton/elliptic solver. Initialization resets the persistent
// context and nsolving through device operations on every physics step.
// The iteration schedule is the body of a GPU conditional WHILE graph node.
struct SolverSpec {
    ScheduleSpec initialize;
    ScheduleSpec iteration;
    std::string condition_field = "solver.nsolving";
    int iteration_limit = 0;
};
// Allocates persistent GPU context at construction, verifies the available
// cached specializations, and describes the original convergence-controlled
// solve. This does not execute or use a host-side iteration condition.
SolverSpec build_newton_solver(ModelData&);
}
