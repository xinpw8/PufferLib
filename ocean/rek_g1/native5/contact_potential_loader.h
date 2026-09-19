#pragma once
#include "contact_potential.h"
#include "../../../vendor/cJSON.h"
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <cstring>
#include <openssl/sha.h>

namespace rek5_contact_potential {
struct Loaded {
    Model model;
    std::string model_id,source_sha256,file_sha256;
};
inline void require(bool ok,const char* message){
    if(!ok)throw std::runtime_error(std::string("Contact potential: ")+message);
}
inline const cJSON* field(const cJSON* object,const char* name){
    return cJSON_GetObjectItemCaseSensitive(object,name);
}
inline std::string string_field(const cJSON* object,const char* name){
    const auto* value=field(object,name);
    require(cJSON_IsString(value)&&value->valuestring,name);
    return value->valuestring;
}
inline bool hash_string(const std::string& text){
    if(text.size()!=64)return false;
    for(char c:text)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    return true;
}
inline float number(const cJSON* value){
    require(cJSON_IsNumber(value)&&std::isfinite(value->valuedouble),"nonfinite/missing number");
    const float result=float(value->valuedouble);
    require(std::isfinite(result),"float32 overflow");return result;
}
inline Loaded parse(const char* text){
    std::unique_ptr<cJSON,decltype(&cJSON_Delete)> root(cJSON_Parse(text),cJSON_Delete);
    require(root&&cJSON_IsObject(root.get()),"invalid JSON");
    const auto* json=root.get();Loaded out;
    require(string_field(json,"schema")=="rek.contact_potential.v1","schema");
    const char* feature_names[]={"distance_unity_numeric","cos_bearing_positive_x","sin_bearing_positive_x"};
    const auto* order=field(json,"feature_order");
    require(cJSON_IsArray(order)&&cJSON_GetArraySize(order)==3,"feature order");
    for(int i=0;i<3;i++){
        const auto* value=cJSON_GetArrayItem(order,i);
        require(cJSON_IsString(value)&&value->valuestring&&std::string(value->valuestring)==feature_names[i],"feature convention");
    }
    const auto* units=field(json,"unit_mapping");
    require(cJSON_IsObject(units)&&number(field(units,"unity_units_per_candidate_unit"))==1,"unit convention");
    require(cJSON_IsFalse(field(json,"action_execution_inferred"))&&number(field(json,"miss_labels"))==0,"positive-only contract");
    out.model_id=string_field(json,"model_id");out.source_sha256=string_field(json,"source_sha256");
    require(hash_string(out.model_id)&&hash_string(out.source_sha256),"hash identity");
    require(number(field(json,"fit_round"))==1,"fit_round must be 1");
    const float count=number(field(json,"sample_count"));
    require(count>=2&&count<=max_samples&&count==floorf(count),"sample_count");
    out.model.count=int(count);
    const auto* scales=field(json,"feature_scale");const auto* samples=field(json,"samples");
    require(cJSON_IsArray(scales)&&cJSON_GetArraySize(scales)==3,"feature_scale");
    require(cJSON_IsArray(samples)&&cJSON_GetArraySize(samples)==out.model.count,"samples");
    for(int j=0;j<3;j++){
        out.model.scale[j]=number(cJSON_GetArrayItem(scales,j));
        require(out.model.scale[j]>0,"positive feature scale required");
    }
    const auto* training_ids=field(json,"training_event_ids");
    const auto* heldout_ids=field(json,"holdout_event_ids");
    require(cJSON_IsArray(training_ids)&&cJSON_GetArraySize(training_ids)==out.model.count,"training IDs");
    require(cJSON_IsArray(heldout_ids)&&cJSON_GetArraySize(heldout_ids)>0,"holdout IDs");
    std::set<std::string> ids;
    for(int split=0;split<2;split++){
        const auto* list=split?heldout_ids:training_ids;
        for(int i=0;i<cJSON_GetArraySize(list);i++){
            const auto* item=cJSON_GetArrayItem(list,i);
            require(cJSON_IsString(item)&&item->valuestring,"event ID string");
            std::string id=item->valuestring;
            require(id.rfind(split?"R2-E":"R1-E",0)==0,"round leakage");
            require(ids.insert(id).second,"duplicate event ID");
        }
    }
    for(int i=0;i<out.model.count;i++){
        const auto* row=cJSON_GetArrayItem(samples,i);
        require(cJSON_IsArray(row)&&cJSON_GetArraySize(row)==3,"sample row");
        for(int j=0;j<3;j++)out.model.samples[i][j]=number(cJSON_GetArrayItem(row,j));
        const auto* s=out.model.samples[i];
        require(s[0]>0&&fabsf(s[1]*s[1]+s[2]*s[2]-1.f)<1e-4f,"distance/angular sample");
    }
    return out;
}
inline Loaded load(const char* path){
    std::ifstream stream(path,std::ios::binary);require(bool(stream),"cannot open model");
    std::string bytes((std::istreambuf_iterator<char>(stream)),std::istreambuf_iterator<char>());
    require(!stream.bad()&&!bytes.empty()&&bytes.size()<=1024*1024,"invalid model length/read");
    auto out=parse(bytes.c_str());unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size(),digest);
    constexpr char hex[]="0123456789abcdef";
    for(unsigned char b:digest){out.file_sha256+=hex[b>>4];out.file_sha256+=hex[b&15];}
    return out;
}
}
