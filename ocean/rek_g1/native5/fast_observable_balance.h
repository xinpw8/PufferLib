#ifndef REK_NATIVE5_FAST_OBSERVABLE_BALANCE_H
#define REK_NATIVE5_FAST_OBSERVABLE_BALANCE_H

#include "observable_balance.h"

#ifdef __CUDACC__
#define REK_FAST_OBSERVABLE_FN __host__ __device__ inline
#else
#define REK_FAST_OBSERVABLE_FN inline
#endif

namespace rek_fast_observable {
// Candidate root origins and composed clip quaternions, not integrated bodies.
// No compact joint correspondence or referee/count producer is established.
struct Input {
    float root[2][7]; // Common XYZ followed by WXYZ, absolute fighter slots.
    uint64_t round_key;
    double sample_seconds;
    float round_duration_seconds,round_remaining_seconds;
    int points[2],round_active,terminal;
};
struct History {
    rek_observable_balance::Snapshot previous;
    bool available;
};

// Invoke once per exported observation. Repeated policy-buffer reads must not
// advance history. Explicit reset clears History; round identity masks history
// at round boundaries. Same-round body/route changes do not clear this state.
REK_FAST_OBSERVABLE_FN rek_observable_balance::Status project(
        History& history,const Input& input,float* output) {
    namespace ob=rek_observable_balance;
    ob::Snapshot now{};
    for(int side=0;side<2;side++){
        for(int k=0;k<3;k++)now.fighter[side].root_xyz[k]=input.root[side][k];
        for(int k=0;k<4;k++)now.fighter[side].root_wxyz[k]=input.root[side][3+k];
        now.fighter[side].joint_pose_available=0;
        now.points[side]=input.points[side];
    }
    now.round_key=input.round_key;now.sample_seconds=input.sample_seconds;
    now.round_duration_seconds=input.round_duration_seconds;
    now.round_remaining_seconds=input.round_remaining_seconds;
    now.round_active=input.round_active;now.terminal=input.terminal;
    now.referee_available=0;now.count_mask=0;
    auto previous=history.previous;
    for(int side=0;side<2;side++){
        now.actor_slot=side;previous.actor_slot=side;
        const auto status=ob::project(now,history.available?&previous:nullptr,
            output?output+side*ob::kFeatures:nullptr);
        if(status!=ob::kOk){
            history.available=false;
            if(output)for(int k=0;k<2*ob::kFeatures;k++)output[k]=0;
            return status;
        }
    }
    now.actor_slot=0;history.previous=now;history.available=true;
    return ob::kOk;
}
}
#undef REK_FAST_OBSERVABLE_FN
#endif
