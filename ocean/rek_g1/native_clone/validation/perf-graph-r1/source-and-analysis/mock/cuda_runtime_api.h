#pragma once
#include <functional>
#include <vector>
struct FakeGraph{std::function<void()> run;};
struct FakeExec{std::function<void()> run;};
using cudaGraph_t=FakeGraph*;using cudaGraphExec_t=FakeExec*;using cudaStream_t=void*;
using cudaError_t=int;constexpr int cudaSuccess=0,cudaStreamCaptureModeThreadLocal=1;
inline bool capturing=false;
inline int fail_begin=0,fail_end=0,fail_instantiate=0,fail_launch=0;
inline int graphs=0,executions=0,end_calls=0,launch_calls=0,synchronizations=0;
inline std::function<void()> recorded;
inline const char* cudaGetErrorString(cudaError_t){return "mock CUDA failure";}
inline int cudaStreamSynchronize(cudaStream_t){++synchronizations;return 0;}
inline int cudaStreamBeginCapture(cudaStream_t,int){if(fail_begin)return fail_begin;capturing=true;recorded={};return 0;}
inline int cudaStreamEndCapture(cudaStream_t,cudaGraph_t* out){++end_calls;capturing=false;if(fail_end){*out=nullptr;return fail_end;}*out=new FakeGraph{recorded};++graphs;return 0;}
inline int cudaGraphInstantiate(cudaGraphExec_t* out,cudaGraph_t graph,unsigned long long){if(fail_instantiate)return fail_instantiate;*out=new FakeExec{graph->run};++executions;return 0;}
inline int cudaGraphLaunch(cudaGraphExec_t exec,cudaStream_t){++launch_calls;if(fail_launch)return fail_launch;exec->run();return 0;}
inline int cudaGraphDestroy(cudaGraph_t graph){delete graph;--graphs;return 0;}
inline int cudaGraphExecDestroy(cudaGraphExec_t exec){delete exec;--executions;return 0;}
