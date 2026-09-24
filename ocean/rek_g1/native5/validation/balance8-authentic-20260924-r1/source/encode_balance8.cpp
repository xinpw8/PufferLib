#define REK_ENCODER_NO_MAIN
#include "encode_live.cpp"
#include "../balance8_observation.h"

extern "C" void* balance8_create(const char*);
extern "C" void balance8_destroy(void*);
extern "C" void balance8_reset(void*);
extern "C" cJSON* balance8_process(void*,const cJSON*);

int main(int argc,char** argv){try{
    std::string model,schema,projection,busy;
    for(int i=1;i<argc;i++){
        require(i+1<argc,"missing_cli_value");std::string option=argv[i],value=argv[++i];
        if(option=="--model")model=value;
        else if(option=="--observation-schema")schema=value;
        else if(option=="--projection")projection=value;
        else if(option=="--busy-projection")busy=value;
        else throw std::runtime_error("unknown_balance8_cli_option");
    }
    require(!model.empty()&&schema==rek_balance8::kSchema&&projection==PROJECTION&&
        busy=="dispatched_request_v4_duration","explicit_balance8_schema_model_projection_and_busy_contract_required");
    Calibration calibration(model);Encoder legacy(calibration,true,false,1);
    std::unique_ptr<void,decltype(&balance8_destroy)> measured(balance8_create(model.c_str()),balance8_destroy);
    auto manifest=legacy.manifest();cJSON_ReplaceItemInObjectCaseSensitive(manifest.get(),"observation_schema",cJSON_CreateString(rek_balance8::kSchema));
    text(manifest.get(),"base_observation_schema",rek_balance8::kBaseSchema);
    text(manifest.get(),"balance8_contract","215 legacy cells and action mask retained; measured vertical root rates and normalized root tilt; validated fresh received referee availability and count bits; no inferred falls");
    cJSON_ReplaceItemInObjectCaseSensitive(manifest.get(),"referee_semantics",cJSON_CreateString("fresh hash-bound lifecycle-bound bridge receipt; available at202, count bits204/205; history203; no server-current-state claim"));
    auto* fields=cJSON_GetObjectItemCaseSensitive(manifest.get(),"fields");
    const char* descriptions[]={"actor vertical root velocity m/s; zero padding without history203","opponent vertical root velocity m/s; zero padding without history203","actor normalized root tilt/pi, no fall threshold","opponent normalized root tilt/pi, no fall threshold","validated fresh received referee available","preceding same-round same-slot valid source interval in(0,250ms]","actor received count-active bit, unknown padding if202=0","opponent received count-active bit, unknown padding if202=0"};
    for(int k=0;k<8;k++){auto* f=cJSON_GetArrayItem(fields,rek_balance8::kColumns[k]);cJSON_ReplaceItemInObjectCaseSensitive(f,"kind",cJSON_CreateString(k<4?"measured_derived":"availability_or_received"));cJSON_ReplaceItemInObjectCaseSensitive(f,"source",cJSON_CreateString(descriptions[k]));}
    emit(manifest.get());std::string line;
    while(std::getline(std::cin,line)){
        try{
            require(line.size()<=1048576,"source_line_too_large");Json source(cJSON_ParseWithLengthOpts(line.c_str(),line.size()+1,nullptr,1),cJSON_Delete);
            require(source&&cJSON_IsObject(source.get()),"invalid_source_JSON");
            if(auto* type=optional(source.get(),"type")){
                if(str(type)=="close")break;
                if(str(type)=="reset"){legacy.reset();legacy.reset_cadence();balance8_reset(measured.get());auto out=object();text(out.get(),"event","projection_reset");emit(out.get());continue;}
            }
            Json extra(balance8_process(measured.get(),source.get()),cJSON_Delete);
            auto out=legacy.process(source.get());
            if(cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(out.get(),"ready"))){
                require(cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(extra.get(),"ready")),"balance8_measured_projection_unavailable");
                auto* request=cJSON_GetObjectItemCaseSensitive(out.get(),"worker_request");
                auto* obs=cJSON_GetObjectItemCaseSensitive(request,"observation");
                auto* projected=get(get(extra.get(),"worker_request"),"observation");
                float check[223]={},values[223]={};
                for(int column:rek_balance8::kColumns)values[column]=float(num(cJSON_GetArrayItem(projected,column)));
                require(rek_balance8::overlay(check,values),"invalid_balance8_measurement");
                for(int column:rek_balance8::kColumns)cJSON_ReplaceItemInArray(obs,column,cJSON_CreateNumber(values[column]));
                cJSON_ReplaceItemInObjectCaseSensitive(request,"observation_schema",cJSON_CreateString(rek_balance8::kSchema));
                cJSON_AddItemToObject(cJSON_GetObjectItemCaseSensitive(out.get(),"provenance"),"balance8",cJSON_Duplicate(get(extra.get(),"provenance"),1));
            }
            emit(out.get());
        }catch(const std::exception& e){legacy.reset();balance8_reset(measured.get());emit(legacy.unavailable(e.what()).get());}
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"encode-balance8: "<<e.what()<<'\n';return 2;}}
