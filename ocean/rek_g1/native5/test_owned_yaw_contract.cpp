// CPU contract tests. No GPU, authentic process, or policy execution.
#include "owned_yaw_observation.h"
#include <cstdio>
int main(){try{
    unsigned checks=0;auto check=[&](bool ok){checks++;if(!ok)throw std::runtime_error("owned_yaw_contract_failed");};
    check(!rek_owned_yaw::enabled(nullptr));check(!rek_owned_yaw::enabled(rek_owned_yaw::kLegacySchema));
    check(rek_owned_yaw::enabled(rek_owned_yaw::kSchema));
    for(const char* invalid:{"","unknown","rek.native5.scaled_polar_xy.v2"}){
        bool rejected=false;try{rek_owned_yaw::enabled(invalid);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
    }
    const float expected[]={0,0,0,0,0,0,1,-1,1,-1,1,-1,1,-1,1,-1};
    for(int i=0;i<16;i++){
        check(rek_owned_yaw::valid_desired(i)==(i>=1));check(rek_owned_yaw::desired_yaw(i)==expected[i]);
        check(rek_owned_yaw::pending_value(false,true,i)==0);check(rek_owned_yaw::pending_value(true,false,i)==0);
        check(rek_owned_yaw::pending_value(true,true,i)==expected[i]);
    }
    for(const char* backend:{static_cast<const char*>(nullptr),"semantic_cuda","puffysics_cuda","mujoco_cpu_eval"})
        for(const char* frozen:{static_cast<const char*>(nullptr),"","None","legacy.bin","migrated.bin"}){
            check(rek_owned_yaw::training_compatible(false,backend,frozen));
            const bool expected_ok=backend&&!std::strcmp(backend,"semantic_cuda")&&(!frozen||!frozen[0]||!std::strcmp(frozen,"None"));
            check(rek_owned_yaw::training_compatible(true,backend,frozen)==expected_ok);
        }
    printf("{\"test\":\"owned_yaw_v2_cpu_contract\",\"checks\":%u,\"passed\":true}\n",checks);return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 2;}}
