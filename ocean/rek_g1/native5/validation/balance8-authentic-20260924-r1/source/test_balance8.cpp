#include <cmath>
#include <initializer_list>
#include "balance8_observation.h"
#include <array>
#include <cstdio>
#include <stdexcept>

void check(bool value){if(!value)throw std::runtime_error("balance8 CPU check failed");}
int main(){
    namespace b=rek_observable_balance;std::array<float,223> base;
    for(int i=0;i<223;i++)base[i]=float(i)-111.5f;base[0]=-0.f;
    auto original=base;b::Snapshot now{};
    now.round_key=1;now.sample_seconds=1;now.round_duration_seconds=120;now.round_remaining_seconds=100;now.round_active=1;
    for(int s=0;s<2;s++){now.fighter[s].root_wxyz[0]=1;now.fighter[s].root_xyz[2]=1.f;}
    check(rek_balance8::project(now,nullptr,base.data())==b::kOk);
    int retained=0,selected=0;for(int i=0;i<223;i++){
        if(rek_balance8::column(i)){selected++;check(base[i]==0);}
        else{retained++;check(std::memcmp(&base[i],&original[i],4)==0);}
    }check(retained==215&&selected==8);
    auto previous=now;now.sample_seconds=1.02;now.fighter[0].root_xyz[2]=1.1f;
    now.fighter[1].root_wxyz[0]=0;now.fighter[1].root_wxyz[1]=1;
    now.referee_available=1;now.count_mask=1;
    check(rek_balance8::project(now,&previous,base.data())==b::kOk);
    check(std::abs(base[9]-5)<1e-4&&base[95]==0&&base[72]==0&&base[158]==1&&base[202]==1&&base[203]==1&&base[204]==1&&base[205]==0);
    now.actor_slot=1;previous.actor_slot=1;check(rek_balance8::project(now,&previous,base.data())==b::kOk);
    check(base[72]==1&&base[158]==0&&base[204]==0&&base[205]==1);
    now.sample_seconds=1.251;check(rek_balance8::project(now,&previous,base.data())==b::kOk);check(base[203]==0&&base[9]==0&&base[95]==0);
    now.sample_seconds=1;check(rek_balance8::project(now,&previous,base.data())==b::kOk);check(base[203]==0);
    now.sample_seconds=1.02;now.round_key++;check(rek_balance8::project(now,&previous,base.data())==b::kOk);check(base[203]==0);
    now.referee_available=0;check(rek_balance8::project(now,&previous,base.data())==b::kOk);check(base[202]==0&&base[204]==0&&base[205]==0);
    auto before=base;now.fighter[0].root_xyz[2]=NAN;check(rek_balance8::project(now,&previous,base.data())==b::kInvalidCurrent);check(std::memcmp(base.data(),before.data(),sizeof(base))==0);
    float bad[223]={};bad[72]=1.01f;check(!rek_balance8::overlay(base.data(),bad));bad[72]=0;bad[204]=1;check(!rek_balance8::overlay(base.data(),bad));
    std::puts("{\"balance8_cpu_tests\":\"passed\",\"columns\":8,\"protected_columns_bitwise\":215,\"tests\":12}");
}
