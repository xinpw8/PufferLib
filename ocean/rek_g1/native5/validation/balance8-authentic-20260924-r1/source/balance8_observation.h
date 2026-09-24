#pragma once
#include "observable_balance.h"
#include <cmath>
#include <initializer_list>
#include <cstring>

namespace rek_balance8 {
constexpr const char* kSchema="rek.native5.scaled_polar_xy.balance8_v1";
constexpr const char* kBaseSchema="rek.native5.scaled_polar_xy.v1";
constexpr int kColumns[8]={9,95,72,158,202,203,204,205};
constexpr int kAddedFeatures=8;
inline bool column(int value){for(int c:kColumns)if(c==value)return true;return false;}
inline bool overlay(float* base,const float* measured){
    if(!base||!measured)return false;
    for(int c:kColumns)if(!std::isfinite(measured[c]))return false;
    for(int c:{72,158})if(measured[c]<0||measured[c]>1)return false;
    for(int c:{202,203,204,205})if(measured[c]!=0&&measured[c]!=1)return false;
    if((measured[202]==0&&(measured[204]!=0||measured[205]!=0))||
       (measured[203]==0&&(measured[9]!=0||measured[95]!=0)))return false;
    for(int c:kColumns)base[c]=measured[c];
    return true;
}
inline rek_observable_balance::Status project(const rek_observable_balance::Snapshot& now,
        const rek_observable_balance::Snapshot* previous,float* base){
    float measured[223];auto status=rek_observable_balance::project(now,previous,measured);
    if(status!=rek_observable_balance::kOk)return status;
    return overlay(base,measured)?status:rek_observable_balance::kInvalidCurrent;
}
}
