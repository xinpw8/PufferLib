#define REK_OBSERVABLE_ENCODER_NO_MAIN
#include "encode_observable_balance.cpp"

extern "C" void* balance8_create(const char* model){return new ObservableEncoder(file_sha(model),true,1,false);}
extern "C" void balance8_destroy(void* state){delete static_cast<ObservableEncoder*>(state);}
extern "C" void balance8_reset(void* state){auto* e=static_cast<ObservableEncoder*>(state);e->reset();e->reset_cadence();}
extern "C" cJSON* balance8_process(void* state,const cJSON* source){
    auto* e=static_cast<ObservableEncoder*>(state);
    try{return e->process(source).release();}
    catch(const std::exception& error){e->reset();return e->unavailable(error.what()).release();}
}
