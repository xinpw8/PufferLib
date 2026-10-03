#pragma once
// Host loader for rek.lite_falls.v1 models written by fit_lite_falls.py.
#include "lite_falls.h"
#include "../../../vendor/cJSON.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <openssl/sha.h>

namespace rek_lite_falls {
constexpr const char* kSchema="rek.lite_falls.v1";

struct Loaded {
    Model model{};
    std::string model_id,file_sha256,provenance;
};
inline void require(bool ok,const std::string& message){
    if(!ok)throw std::runtime_error("Lite falls model: "+message);
}
inline const cJSON* field(const cJSON* object,const char* name){
    const auto* value=cJSON_GetObjectItemCaseSensitive(object,name);
    require(value!=nullptr,std::string("missing ")+name);
    return value;
}
inline float number(const cJSON* value,const char* name){
    require(cJSON_IsNumber(value)&&std::isfinite(value->valuedouble),std::string("nonfinite ")+name);
    const float out=float(value->valuedouble);
    require(std::isfinite(out),std::string("float32 overflow ")+name);
    return out;
}
inline void numbers(const cJSON* object,const char* name,float* out,int count){
    const auto* array=field(object,name);
    require(cJSON_IsArray(array)&&cJSON_GetArraySize(array)==count,
        std::string(name)+" must have "+std::to_string(count)+" values");
    for(int i=0;i<count;i++)out[i]=number(cJSON_GetArrayItem(array,i),name);
}
inline void rows(const cJSON* object,const char* name,float* out,int row_count,int width){
    const auto* array=field(object,name);
    require(cJSON_IsArray(array)&&cJSON_GetArraySize(array)==row_count,
        std::string(name)+" must have "+std::to_string(row_count)+" rows");
    for(int r=0;r<row_count;r++){
        const auto* row=cJSON_GetArrayItem(array,r);
        require(cJSON_IsArray(row)&&cJSON_GetArraySize(row)==width,std::string(name)+" row width");
        for(int c=0;c<width;c++)out[r*width+c]=number(cJSON_GetArrayItem(row,c),name);
    }
}
inline void dimension(const cJSON* json,const char* name,int expected){
    require(number(field(json,name),name)==float(expected),std::string(name)+" mismatch");
}
inline std::string sha256_hex(const std::string& bytes){
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size(),digest);
    char text[2*SHA256_DIGEST_LENGTH+1];
    for(int i=0;i<SHA256_DIGEST_LENGTH;i++)std::snprintf(text+2*i,3,"%02x",digest[i]);
    return text;
}

inline Loaded parse(const std::string& text){
    std::unique_ptr<cJSON,decltype(&cJSON_Delete)> root(cJSON_Parse(text.c_str()),cJSON_Delete);
    require(root&&cJSON_IsObject(root.get()),"invalid JSON");
    const cJSON* json=root.get();Loaded out;Model& m=out.model;
    const auto* schema=field(json,"schema");
    require(cJSON_IsString(schema)&&std::string(schema->valuestring)==kSchema,"schema");
    const auto* id=field(json,"model_id");
    require(cJSON_IsString(id)&&id->valuestring&&*id->valuestring,"model_id");
    out.model_id=id->valuestring;
    dimension(json,"phase_bins",kPhaseBins);dimension(json,"move_bins",kMoveBins);
    dimension(json,"distance_bins",kDistanceBins);dimension(json,"closing_bins",kClosingBins);
    dimension(json,"quantiles",kQuantiles);
    numbers(json,"distance_edges_m",m.distance_edges,kDistanceBins-1);
    numbers(json,"closing_edges_m_s",m.closing_edges,kClosingBins-1);
    for(int i=1;i<kDistanceBins-1;i++)require(m.distance_edges[i]>m.distance_edges[i-1],"distance edges must increase");
    for(int i=1;i<kClosingBins-1;i++)require(m.closing_edges[i]>m.closing_edges[i-1],"closing edges must increase");
    m.bias=number(field(json,"bias"),"bias");
    numbers(json,"own_move",m.own_move,kMoveBins);
    numbers(json,"opponent_move",m.opponent_move,kMoveBins);
    numbers(json,"distance",m.distance,kDistanceBins);
    numbers(json,"closing",m.closing,kClosingBins);
    rows(json,"own_move_distance",m.own_move_distance,kMoveBins,kDistanceBins);
    rows(json,"opponent_move_distance",m.opponent_move_distance,kMoveBins,kDistanceBins);
    m.struck=number(field(json,"struck"),"struck");
    rows(json,"outcome_probability",m.outcome_probability,kOutcomeClasses,kOutcomes);
    for(int c=0;c<kOutcomeClasses;c++){
        float sum=0;
        for(int o=0;o<kOutcomes;o++){
            const float p=m.outcome_probability[c*kOutcomes+o];
            require(p>=0&&p<=1,"outcome probability range");sum+=p;
        }
        require(std::fabs(sum-1.f)<=1e-4f,"outcome probabilities must sum to 1");
    }
    rows(json,"delay_quantiles_ticks",m.delay_quantiles,kOutcomes,kQuantiles);
    for(int o=0;o<kOutcomes;o++)for(int q=0;q<kQuantiles;q++){
        const float v=m.delay_quantiles[o*kQuantiles+q];
        require(v>=0&&v<=kTicksPerSecond*600,"delay quantile range");
        if(q)require(v>=m.delay_quantiles[o*kQuantiles+q-1],"delay quantiles must not decrease");
    }
    const auto* observation=field(json,"observation");
    require(cJSON_IsObject(observation),"observation");
    m.falling_tilt_degrees=number(field(observation,"falling_tilt_degrees"),"falling_tilt_degrees");
    m.fallen_tilt_degrees=number(field(observation,"fallen_tilt_degrees"),"fallen_tilt_degrees");
    m.falling_height_ratio=number(field(observation,"falling_height_ratio"),"falling_height_ratio");
    m.fallen_height_ratio=number(field(observation,"fallen_height_ratio"),"fallen_height_ratio");
    const auto* provenance=cJSON_GetObjectItemCaseSensitive(json,"provenance");
    if(provenance){
        std::unique_ptr<char,decltype(&cJSON_free)> printed(cJSON_PrintUnformatted(provenance),cJSON_free);
        require(printed!=nullptr,"provenance");out.provenance=printed.get();
    }else out.provenance="null";
    m.enabled=1;
    return out;
}
inline Loaded load(const char* path){
    require(path&&*path,"empty path");
    std::ifstream file(path,std::ios::binary);require(bool(file),std::string("cannot read ")+path);
    std::string text((std::istreambuf_iterator<char>(file)),{});
    Loaded out=parse(text);out.file_sha256=sha256_hex(text);
    return out;
}
}  // namespace rek_lite_falls
