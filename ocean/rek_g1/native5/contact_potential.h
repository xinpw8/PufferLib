#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define REK_CONTACT_FN __host__ __device__ inline
#else
#define REK_CONTACT_FN inline
#endif

// A positive-only positioning prior. This is neither a hit classifier nor a
// collision test. No action identity or negative outcome is inferred here.
namespace rek5_contact_potential {
constexpr int max_samples=64;
struct Model {
    int count=0;
    float scale[3]={};
    float samples[max_samples][3]={};
};
REK_CONTACT_FN float potential(const Model& model,float distance,float bearing){
    const float features[3]={distance,cosf(bearing),sinf(bearing)};
    float nearest=INFINITY;
    for(int i=0;i<model.count;i++){
        float d2=0;
        for(int j=0;j<3;j++){
            const float d=(features[j]-model.samples[i][j])/model.scale[j];
            d2+=d*d;
        }
        nearest=fminf(nearest,d2);
    }
    const float distance_in_feature_space=sqrtf(nearest);
    return -distance_in_feature_space/(1.f+distance_in_feature_space);
}
REK_CONTACT_FN float shaping_delta(float previous,float next,bool terminal,float gamma){
    return gamma*(terminal?0.f:next)-previous;
}
}
#undef REK_CONTACT_FN
