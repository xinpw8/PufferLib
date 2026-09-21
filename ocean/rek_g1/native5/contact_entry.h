#pragma once
#include "primitive_motion.cuh"
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Native ContactTrackingManager keys min(geomA,geomB)<<32|max(geomA,geomB),
// not body pairs. The verified compact subset has 12 distinct striker geoms
// and 9 distinct target geoms. Same-body colliders retain separate histories.
namespace rek_contact_entry {
constexpr int Strikers=12,Targets=9,Pairs=Strikers*Targets;
enum class Mode { LegacyLimbUnion=0, GeomPair=1 };
inline Mode parse(const char* value){
    if(!value||!std::strcmp(value,"legacy_limb_union_v1"))return Mode::LegacyLimbUnion;
    if(!std::strcmp(value,"geom_pair_v1"))return Mode::GeomPair;
    throw std::invalid_argument("REK_FAST_CONTACT_ENTRY_must_be_legacy_limb_union_v1_or_geom_pair_v1");
}
inline const char* name(Mode mode){return mode==Mode::GeomPair?"geom_pair_v1":"legacy_limb_union_v1";}
struct State {std::uint64_t words[2];};
#ifdef __CUDACC__
#define REK_CONTACT_HD __host__ __device__ inline
#else
#define REK_CONTACT_HD inline
#endif
REK_CONTACT_HD bool update(State& state,int pair,bool touching){
    const std::uint64_t bit=std::uint64_t(1)<<(pair%64);
    auto& word=state.words[pair/64];const bool entered=touching&&!(word&bit);
    if(touching)word|=bit;else word&=~bit;
    return entered;
}
REK_CONTACT_HD void unscored_endpoint(State& state,int pair,
        const rek5_primitive::Shape& a,const rek5_primitive::Shape& b){
    // When scoring is impossible, sampled entry flags are unobserved. The full
    // sampler retains only t=1, so this preserves the next tick's history.
    update(state,pair,rek5_primitive::overlap(a,b));
}
struct Sampled {bool entered,overlap_after_start;};
REK_CONTACT_HD Sampled sample(State& state,int pair,
        const rek5_primitive::Shape& old_a,const rek5_primitive::Shape& a,
        const rek5_primitive::Shape& old_b,const rek5_primitive::Shape& b,int samples){
    Sampled out{};
    // t=0 also reconciles the existing instantaneous canned-route switch.
    // No cross-clip interpolation is introduced. Only t=1 remains latched;
    // a sampled enter followed by exit must not persist into the next tick.
    for(int i=0;i<=samples;i++){
        const float t=float(i)/samples;
        const bool touch=rek5_primitive::overlap(rek5_primitive::interpolate_shape(old_a,a,t),
                                                rek5_primitive::interpolate_shape(old_b,b,t));
        out.entered=update(state,pair,touch)||out.entered;
        out.overlap_after_start=out.overlap_after_start||(i>0&&touch);
    }
    return out;
}
#undef REK_CONTACT_HD
}
