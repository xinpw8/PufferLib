#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>

/* Captures one runtime control step only. Every command and policy input must
 * already have a persistent device address; its contents may change between
 * launches on this stream. Host inspection is never part of this graph. */
class EvalStepGraph {
    cudaGraph_t graph_=nullptr;
    cudaGraphExec_t executable_=nullptr;
    static void check(cudaError_t status) {
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    }
public:
    EvalStepGraph()=default;
    EvalStepGraph(const EvalStepGraph&)=delete;
    EvalStepGraph& operator=(const EvalStepGraph&)=delete;
    ~EvalStepGraph(){clear();}
    void clear() noexcept {
        if(executable_)cudaGraphExecDestroy(executable_);
        if(graph_)cudaGraphDestroy(graph_);
        executable_=nullptr;graph_=nullptr;
    }
    template<class Enqueue> void capture(cudaStream_t stream,Enqueue enqueue) {
        clear();
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        cudaGraph_t candidate=nullptr;
        try { enqueue(); }
        catch(...) {
            // Even an invalidated capture must be ended on its origin stream.
            cudaStreamEndCapture(stream,&candidate);
            if(candidate)cudaGraphDestroy(candidate);
            throw;
        }
        const auto ended=cudaStreamEndCapture(stream,&candidate);
        if(ended!=cudaSuccess){if(candidate)cudaGraphDestroy(candidate);check(ended);}
        cudaGraphExec_t execution=nullptr;
        const auto instantiated=cudaGraphInstantiate(&execution,candidate,0);
        if(instantiated!=cudaSuccess){
            if(execution)cudaGraphExecDestroy(execution);
            cudaGraphDestroy(candidate);check(instantiated);
        }
        graph_=candidate;executable_=execution;
    }
    void launch(cudaStream_t stream) {
        if(!executable_)throw std::runtime_error("Runtime step graph is unavailable; reset/recreate the worker");
        check(cudaGraphLaunch(executable_,stream));
    }
    bool ready()const{return executable_!=nullptr;}
};
