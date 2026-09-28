#include "eval_step_graph.h"
#include <cstdio>
#include <cstdlib>
static int checks=0;
static void require(bool value,const char* why){++checks;if(!value){fprintf(stderr,"FAILED: %s\n",why);exit(1);}}
template<class F>void throws(F fn,const char* why){bool caught=false;try{fn();}catch(...){caught=true;}require(caught,why);}
int main(){
    int device_command=0,state=0,enqueues=0;
    auto enqueue=[&]{++enqueues;require(capturing,"only enqueue during capture");recorded=[&]{state+=device_command;};};
    {
        EvalStepGraph g;
        throws([&]{g.launch(nullptr);},"unavailable graph cannot silently eager fallback");
        g.capture(nullptr,enqueue);
        require(g.ready()&&enqueues==1&&state==0,"capture does not advance simulation");
        device_command=3;g.launch(nullptr);require(state==3,"first launch reads current persistent input");
        device_command=-2;g.launch(nullptr);require(state==1,"subsequent launch reads changed input");
        state=0;g.capture(nullptr,enqueue);
        require(state==0&&graphs==1&&executions==1,"recapture after reset replaces graph without executing");
        device_command=5;g.launch(nullptr);require(state==5,"new capture uses same reset state and pointer");
        fail_launch=4;throws([&]{g.launch(nullptr);},"launch failure reported");fail_launch=0;
        g.clear();g.clear();require(!g.ready()&&graphs==0&&executions==0,"idempotent explicit clear");
        const int ends=end_calls;
        throws([&]{g.capture(nullptr,[&]{throw std::runtime_error("runtime enqueue failure");});},"runtime capture exception reported");
        require(!capturing&&end_calls==ends+1&&graphs==0&&executions==0&&!g.ready(),"throw always ends origin capture and destroys temporary graph");
        fail_end=5;throws([&]{g.capture(nullptr,enqueue);},"invalidated capture reported");fail_end=0;
        require(!capturing&&graphs==0&&executions==0&&!g.ready(),"invalid capture no execution or resource leak");
        fail_instantiate=6;throws([&]{g.capture(nullptr,enqueue);},"instantiate failure reported");fail_instantiate=0;
        require(graphs==0&&executions==0&&!g.ready(),"instantiate failure destroys candidate graph");
        fail_begin=7;throws([&]{g.capture(nullptr,enqueue);},"begin failure reported");fail_begin=0;
        require(!capturing&&graphs==0&&executions==0,"begin failure no resources");
        g.capture(nullptr,enqueue);require(g.ready(),"capture can be explicitly retried after an error");
    }
    require(graphs==0&&executions==0,"destructor releases both CUDA objects");
    printf("{\"checks\":%d,\"failures\":0,\"gpu_calls\":0,\"scope\":\"mock CUDA lifecycle, no trajectory or throughput claim\"}\n",checks);
}
