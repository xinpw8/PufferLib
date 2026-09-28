#ifndef REK_NATIVE5_OBSERVABLE_PREV_ACTION_H
#define REK_NATIVE5_OBSERVABLE_PREV_ACTION_H

#include "observable_balance.h"

#ifdef __CUDACC__
#define REK_PREV_ACTION_FN __host__ __device__ inline
#else
#define REK_PREV_ACTION_FN inline
#endif

// Policy-owned sample history layered over observable_balance.v1. A sample is
// known even when later delivery, rejection or execution is unknown. This
// contract does not expose controller state or infer any held command.
namespace rek_observable_prev_action {
constexpr const char* kSchema="rek.native5.observable_balance_prev_action.v1";
constexpr int kActionCount=33;
constexpr int kAvailableColumn=221;
constexpr int kAddedFeatures=34;

// Only base-v1 metadata padding is reused. Both 86-column pose blocks and all
// existing base-v1 features retain their meanings and numerical values.
REK_PREV_ACTION_FN int column(int action) {
    if(action<0||action>=kActionCount)return -1;
    if(action<8)return 176+action;
    if(action<10)return 186+(action-8);
    if(action<20)return 192+(action-10);
    if(action<31)return 206+(action-20);
    return 219+(action-31);
}
REK_PREV_ACTION_FN bool history_column(int index) {
    return (index>=176&&index<=183)||(index>=186&&index<=187)||
        (index>=192&&index<=201)||(index>=206&&index<=216)||
        (index>=219&&index<=221);
}
REK_PREV_ACTION_FN bool structurally_available(int index) {
    return rek_observable_balance::structurally_available(index)||history_column(index);
}
REK_PREV_ACTION_FN void feature_mask(unsigned char* out) {
    for(int i=0;i<rek_observable_balance::kFeatures;i++)out[i]=structurally_available(i)?1:0;
}
struct History { int action=0; int available=0; };
REK_PREV_ACTION_FN void clear(History& history) {history.action=0;history.available=0;}
REK_PREV_ACTION_FN bool valid(const History& history) {
    return (history.available==0||history.available==1)&&
        (!history.available||(history.action>=0&&history.action<kActionCount));
}
// Call only after a successful policy sample. Do not call for bootstrap value
// evaluation, observation copies, protocol rejection or a failed inference.
// Downstream request rejection does not erase a sample that actually occurred.
REK_PREV_ACTION_FN bool record(History& history,float sampled_action) {
    if(!isfinite(sampled_action)||sampled_action<0||sampled_action>=kActionCount||
            sampled_action!=float(int(sampled_action)))return false;
    history.action=int(sampled_action);history.available=1;return true;
}
// Call before the next policy forward. Invalid input leaves the observation
// untouched. Unknown is all zero; sampled action0 has its own one-hot plus flag.
REK_PREV_ACTION_FN bool write(float* observation,const History& history) {
    if(!observation||!valid(history))return false;
    for(int a=0;a<kActionCount;a++)observation[column(a)]=0.f;
    observation[kAvailableColumn]=float(history.available);
    if(history.available)observation[column(history.action)]=1.f;
    return true;
}
REK_PREV_ACTION_FN bool valid_features(const float* observation) {
    if(!observation)return false;
    const float available=observation[kAvailableColumn];
    if(available!=0.f&&available!=1.f)return false;
    int count=0;
    for(int a=0;a<kActionCount;a++){
        const float value=observation[column(a)];
        if(value!=0.f&&value!=1.f)return false;
        count+=value==1.f;
    }
    return count==(available==1.f?1:0);
}
// Clear alongside the policy's genuine episode/reset boundary, never just a
// rollout horizon, minibatch boundary, counted-fall pose reset or elapsed gap.
// Training writes exact sampled env.actions into the NEXT observation after
// stepping, with terminal outputs cleared. Live stores the last successful
// sample locally and clears it with the worker's recurrent reset.
}
#undef REK_PREV_ACTION_FN
#endif
