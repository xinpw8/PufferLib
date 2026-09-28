#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
struct RuntimePhaseProfile {
    bool enabled=false,pending=false;
    cudaEvent_t events[38]{};
    double host[38]{};
    std::map<std::string,std::vector<double>> values;
    static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
    RuntimePhaseProfile(){const char* p=std::getenv("REK_RUNTIME_PHASE_PROFILE");enabled=p&&std::string(p)=="1";
        if(enabled)for(auto& event:events)check(cudaEventCreate(&event));}
    void mark(int index,cudaStream_t stream){if(!enabled)return;
        if(index==0){if(pending)throw std::runtime_error("Runtime phase profile requires reporting after every step");
            cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(stream,&status));
            if(status!=cudaStreamCaptureStatusNone)throw std::runtime_error("Runtime phase profiling requires no outer graph capture");}
        check(cudaEventRecord(events[index],stream));host[index]=now();if(index==37)pending=true;
    }
    double interval(const char* name,int begin,int end){float ms=0;check(cudaEventElapsedTime(&ms,events[begin],events[end]));
        values[std::string("cuda_")+name].push_back(ms);values[std::string("host_")+name].push_back(host[end]-host[begin]);return ms;}
    void collect(){if(!enabled||!pending)return;
        // Called only after the original check_status stream synchronization.
        interval("preparation_reset_composer",0,1);interval("motor_observation_pack",1,2);
        interval("motor_encoder",2,3);interval("motor_decoder_input",3,4);interval("motor_decoder",4,5);interval("motor_apply",5,6);
        double drive=0,physics=0,tail=0;
        for(int i=0;i<10;i++){int at=6+3*i;drive+=interval("substep_drive",at,at+1);physics+=interval("physics_step_refresh",at+1,at+2);tail+=interval("measurement_referee_reset",at+2,at+3);}
        values["cuda_all_ten_drive"].push_back(drive);values["cuda_all_ten_physics_step_refresh"].push_back(physics);values["cuda_all_ten_measurement_referee_reset"].push_back(tail);
        interval("final_export",36,37);interval("whole_runtime",0,37);pending=false;
    }
    ~RuntimePhaseProfile(){if(!enabled)return;
        std::fprintf(stderr,"rek_runtime_phase_profile {\"schema\":\"rek.runtime_phase_profile.v1\",\"units\":\"ms\",\"intervals_include_host_enqueue_gaps\":true,\"phases\":{");bool comma=false;
        for(auto& item:values){auto v=item.second;if(v.empty())continue;std::sort(v.begin(),v.end());double sum=0;for(double x:v)sum+=x;
            std::fprintf(stderr,"%s\"%s\":{\"count\":%zu,\"sum\":%.9g,\"mean\":%.9g,\"min\":%.9g,\"median\":%.9g,\"max\":%.9g}",comma?",":"",item.first.c_str(),v.size(),sum,sum/v.size(),v.front(),v[v.size()/2],v.back());comma=true;}
        std::fprintf(stderr,"},\"uncollected_final_step\":%s}\n",pending?"true":"false");
        for(auto event:events)if(event)cudaEventDestroy(event);
    }
};
