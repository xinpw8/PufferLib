#ifndef REK_NATIVE5_ACTION_CADENCE_H
#define REK_NATIVE5_ACTION_CADENCE_H
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Mask-only experiment. Physics, recurrent inference, reward and discount clocks
// remain 50 Hz. Category 0 retains owned input; it does not repeat an attack.
namespace rek_action_cadence {
constexpr const char* kContract="rek.policy_action_cadence.ready_ordinal.v1";
inline int parse(const char* value){
    if(!value||!std::strcmp(value,"1"))return 1;
    if(!std::strcmp(value,"5"))return 5;
    throw std::invalid_argument("action_stride_must_be_1_or_5");
}
#ifdef __CUDACC__
#define REK_CADENCE_HD __host__ __device__
#else
#define REK_CADENCE_HD
#endif
REK_CADENCE_HD inline bool decision(int stride,std::uint64_t ordinal){
    return stride==1||ordinal%unsigned(stride)==0;
}
REK_CADENCE_HD inline bool permit(int stride,std::uint64_t ordinal,int action,
                                  bool learner=true,bool terminal=false){
    return !learner||terminal||action==0||decision(stride,ordinal);
}
#undef REK_CADENCE_HD
}
#endif
