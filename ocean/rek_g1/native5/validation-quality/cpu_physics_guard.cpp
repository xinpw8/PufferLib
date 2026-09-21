#include <mujoco/mujoco.h>
#include <cstdlib>
extern "C" void __wrap_mj_step(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step1(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step2(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_kinematics(const mjModel*,mjData*){std::abort();}
