// Offline schema migration. Main never initializes CUDA or executes a kernel.
// The pinned trainer is included only to derive its real parameter registration.
#include "puffer5_bc_core.cuh"
#include "observable_prev_action.h"
#include "cJSON.h"
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace pa=rek_observable_prev_action;
constexpr int O=223,A=33,H=256,L=2;
constexpr const char* kAlgoSha="8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92";
using Bytes=std::vector<unsigned char>;
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
bool sha_valid(const std::string& text){return text.size()==64&&std::all_of(text.begin(),text.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});}
Bytes read(const std::filesystem::path& path){
    std::ifstream file(path,std::ios::binary|std::ios::ate);require(bool(file),"input open failed");
    auto size=file.tellg();require(size>=0&&size<=64*1024*1024,"input size invalid");
    Bytes bytes(size);file.seekg(0);file.read(reinterpret_cast<char*>(bytes.data()),size);
    require(bool(file),"input read failed");return bytes;
}
std::string sha(const void* data,size_t size){
    unsigned char digest[32];unsigned length=0;
    require(EVP_Digest(data,size,digest,&length,EVP_sha256(),nullptr)==1&&length==32,"SHA256 failed");
    std::string result;constexpr char hex[]="0123456789abcdef";
    for(auto byte:digest){result+=hex[byte>>4];result+=hex[byte&15];}return result;
}
void publish(const std::filesystem::path& path,const void* data,size_t size){
    FILE* file=fopen(path.c_str(),"wbx");require(file,"exclusive output creation failed");
    bool written=fwrite(data,1,size,file)==size;int closed=fclose(file);require(written&&closed==0,"output write failed");
    auto saved=read(path);require(saved.size()==size&&std::memcmp(saved.data(),data,size)==0,"output readback failed");
}
struct Layout{size_t count=0,encoder_begin=0;};
Layout layout(){
    Arch arch=build_arch("rek_native5",O,H,L,A,false,128);Allocator parameters{};
    Weights weights=weights_create(&arch,&parameters);
    auto* encoder=static_cast<EncoderWeights*>(weights.encoder);
    auto* network=static_cast<MinGRUWeights*>(weights.network);
    require(parameters.num_regs==4,"unexpected parameter registration count");
    const long expected[4][2]={{H,O},{A+1,H},{3*H,H},{3*H,H}};
    Layout out;int encoder_found=0;
    for(int i=0;i<parameters.num_regs;i++){
        auto& entry=parameters.regs[i];
        require(ndim(entry.shape)==2&&entry.shape[0]==expected[i][0]&&entry.shape[1]==expected[i][1]&&
            entry.elem_size==sizeof(precision_t)&&(out.count*sizeof(precision_t))%16==0,"pinned architecture layout changed");
        if(entry.data_ptr==reinterpret_cast<void**>(&encoder->weight.data)){
            require(i==0,"encoder registration order changed");out.encoder_begin=out.count;encoder_found++;
        }
        out.count+=numel(entry.shape);
    }
    require(encoder_found==1&&out.count==459008&&out.count==size_t(parameters.total_elems)&&
        parameters.total_bytes==long(out.count*sizeof(precision_t)),"flat checkpoint layout changed");
    free(network->weights);free(weights.network);free(weights.decoder);free(weights.encoder);free(parameters.regs);
    return out;
}
bool changed_index(size_t index,const Layout& l){
    return index>=l.encoder_begin&&index<l.encoder_begin+size_t(H)*O&&pa::history_column(int((index-l.encoder_begin)%O));
}
std::vector<float> transform(const std::vector<float>& input,const Layout& l){
    require(input.size()==l.count,"checkpoint shape mismatch");
    for(float value:input)require(std::isfinite(value),"nonfinite checkpoint weight");
    auto output=input;
    for(int h=0;h<H;h++)for(int column=0;column<O;column++)if(pa::history_column(column))
        output[l.encoder_begin+size_t(h)*O+column]=0.f;
    return output;
}
void verify(const std::vector<float>& input,const std::vector<float>& output,const Layout& l){
    require(input.size()==l.count&&output.size()==l.count,"checkpoint shape mismatch");
    const float zero=0;
    for(size_t i=0;i<l.count;i++){
        require(std::isfinite(input[i])&&std::isfinite(output[i]),"nonfinite checkpoint weight");
        require(std::memcmp(&output[i],changed_index(i,l)?&zero:&input[i],sizeof(float))==0,
            "migration changed a protected byte or failed to zero a new input weight");
    }
}
Json parse_sidecar(const Bytes& bytes,const std::string& digest){
    require(std::find(bytes.begin(),bytes.end(),0)==bytes.end(),"embedded NUL in sidecar");
    std::string text(bytes.begin(),bytes.end());const char* end=nullptr;
    Json json(cJSON_ParseWithOpts(text.c_str(),&end,1),cJSON_Delete);require(json&&cJSON_IsObject(json.get()),"invalid source sidecar");
    for(const cJSON* a=json->child;a;a=a->next)for(const cJSON* b=a->next;b;b=b->next)
        require(a->string&&b->string&&std::strcmp(a->string,b->string),"duplicate source sidecar key");
    const auto* schema=cJSON_GetObjectItemCaseSensitive(json.get(),"observation_schema");
    const auto* hash=cJSON_GetObjectItemCaseSensitive(json.get(),"checkpoint_sha256");
    require(cJSON_IsString(schema)&&schema->valuestring&&std::string(schema->valuestring)==rek_observable_balance::kSchema,
        "source checkpoint is not bound to observable_balance.v1");
    require(cJSON_IsString(hash)&&hash->valuestring&&std::string(hash->valuestring)==digest,"source sidecar SHA mismatch");
    for(auto shape:std::array<std::pair<const char*,int>,4>{{{"observations",O},{"actions",A},{"hidden_size",H},{"num_layers",L}}}){
        auto* item=cJSON_GetObjectItemCaseSensitive(json.get(),shape.first);
        require(cJSON_IsNumber(item)&&item->valuedouble==shape.second,"source sidecar architecture mismatch");
    }
    return json;
}
void replace_string(cJSON* json,const char* name,const std::string& value){
    cJSON_DeleteItemFromObjectCaseSensitive(json,name);require(cJSON_AddStringToObject(json,name,value.c_str()),"JSON allocation failed");
}
void migrate(const char* path,const char* expected,const char* output_path){
    require(sha_valid(expected),"expected SHA must be 64 lowercase hexadecimal characters");
    const std::filesystem::path input=std::filesystem::canonical(path);
    const std::filesystem::path output=std::filesystem::absolute(output_path);
    const std::string old_sidecar=input.string()+".policy-schema.json",new_sidecar=output.string()+".policy-schema.json";
    require(!std::filesystem::exists(output)&&!std::filesystem::exists(new_sidecar),"fresh checkpoint and sidecar paths required");
    const auto l=layout();const auto bytes=read(input);require(bytes.size()==l.count*sizeof(float),"checkpoint byte count mismatch");
    const auto original_sha=sha(bytes.data(),bytes.size());require(original_sha==expected,"input checkpoint SHA mismatch");
    const auto receipt_bytes=read(old_sidecar);auto source_receipt=parse_sidecar(receipt_bytes,original_sha);
    std::vector<float> original(l.count);std::memcpy(original.data(),bytes.data(),bytes.size());
    const auto migrated=transform(original,l);verify(original,migrated,l);
    const auto migrated_sha=sha(migrated.data(),bytes.size());
    size_t actually_changed=0;for(size_t i=0;i<l.count;i++)actually_changed+=std::memcmp(&original[i],&migrated[i],sizeof(float))!=0;
    auto tool_bytes=read(std::filesystem::canonical("/proc/self/exe"));
    Json receipt(cJSON_CreateObject(),cJSON_Delete);require(bool(receipt),"JSON allocation failed");
    replace_string(receipt.get(),"observation_schema",pa::kSchema);replace_string(receipt.get(),"checkpoint_sha256",migrated_sha);
    replace_string(receipt.get(),"checkpoint",output.string());replace_string(receipt.get(),"migration","zero_previous_sample_encoder_columns");
    replace_string(receipt.get(),"source_observation_schema",rek_observable_balance::kSchema);
    replace_string(receipt.get(),"source_checkpoint",input.string());replace_string(receipt.get(),"source_checkpoint_sha256",original_sha);
    replace_string(receipt.get(),"source_schema_receipt",old_sidecar);
    replace_string(receipt.get(),"source_schema_receipt_sha256",sha(receipt_bytes.data(),receipt_bytes.size()));
    replace_string(receipt.get(),"migration_tool_sha256",sha(tool_bytes.data(),tool_bytes.size()));
    replace_string(receipt.get(),"registered_trainer_algo_sha256",kAlgoSha);
    replace_string(receipt.get(),"action_history_semantics","previous successful own policy sample; delivery acceptance execution unknown");
    cJSON_AddNumberToObject(receipt.get(),"observations",O);cJSON_AddNumberToObject(receipt.get(),"actions",A);
    cJSON_AddNumberToObject(receipt.get(),"hidden_size",H);cJSON_AddNumberToObject(receipt.get(),"num_layers",L);
    cJSON_AddNumberToObject(receipt.get(),"parameters",l.count);cJSON_AddNumberToObject(receipt.get(),"encoder_begin",l.encoder_begin);
    cJSON_AddNumberToObject(receipt.get(),"encoder_shape_rows",H);cJSON_AddNumberToObject(receipt.get(),"encoder_shape_columns",O);
    auto* columns=cJSON_AddArrayToObject(receipt.get(),"zeroed_encoder_columns");
    for(int i=0;i<O;i++)if(pa::history_column(i))cJSON_AddItemToArray(columns,cJSON_CreateNumber(i));
    cJSON_AddNumberToObject(receipt.get(),"zeroed_weight_entries",H*pa::kAddedFeatures);
    cJSON_AddNumberToObject(receipt.get(),"actually_changed_weight_entries",actually_changed);
    cJSON_AddNumberToObject(receipt.get(),"other_parameter_bytes_preserved",(l.count-H*pa::kAddedFeatures)*sizeof(float));
    cJSON_AddBoolToObject(receipt.get(),"other_parameters_bitwise_equal",true);
    cJSON_AddBoolToObject(receipt.get(),"gpu_equivalence_verified",false);
    cJSON_AddBoolToObject(receipt.get(),"training_performed",false);cJSON_AddNumberToObject(receipt.get(),"cuda_calls",0);
    cJSON_AddItemToObject(receipt.get(),"source_binding",source_receipt.release());
    std::unique_ptr<char,decltype(&cJSON_free)> text(cJSON_Print(receipt.get()),cJSON_free);require(bool(text),"JSON serialization failed");
    publish(output,migrated.data(),bytes.size());publish(new_sidecar,text.get(),std::strlen(text.get()));
    std::cout<<text.get()<<'\n';
}
void cpu_test(){
    const auto l=layout();require(l.count==459008&&l.encoder_begin==0,"registered layout test failed");
    std::vector<float> original(l.count);for(size_t i=0;i<l.count;i++)original[i]=float(int(i%257)-128)/128.f;
    original[0]=-0.f;auto fresh=transform(original,l);verify(original,fresh,l);
    size_t changed=0;for(size_t i=0;i<l.count;i++)changed+=changed_index(i,l);require(changed==8704,"wrong column count");
    for(size_t i:std::array<size_t,6>{0,175,222,size_t(O*H-1),size_t(O*H),l.count-1}){
        auto broken=fresh;broken[i]+=1;bool rejected=false;
        try{verify(original,broken,l);}catch(const std::exception&){rejected=true;}require(rejected,"protected mutation accepted");
    }
    auto broken=fresh;broken[pa::column(0)]=-0.f;bool rejected=false;
    try{verify(original,broken,l);}catch(const std::exception&){rejected=true;}require(rejected,"noncanonical zero accepted");
    auto invalid=original;invalid.back()=NAN;rejected=false;
    try{transform(invalid,l);}catch(const std::exception&){rejected=true;}require(rejected,"nonfinite checkpoint accepted");
    require(sha_valid(std::string(64,'a'))&&!sha_valid(std::string(63,'a'))&&!sha_valid(std::string(64,'G')),"SHA syntax guard failed");
    std::array<float,O> base{},augmented{};
    for(int i=0;i<O;i++)base[i]=rek_observable_balance::structurally_available(i)?float((i%17)-8)/8.f:0.f;
    unsigned dot_products=0;
    for(int action=-1;action<A;action++){
        augmented=base;pa::History history;if(action>=0)require(pa::record(history,float(action)),"sample fixture failed");
        require(pa::write(augmented.data(),history),"history fixture failed");
        for(int h=0;h<H;h++){
            double old_sum=0,new_sum=0;
            for(int i=0;i<O;i++){old_sum+=double(base[i])*original[h*O+i];new_sum+=double(augmented[i])*fresh[h*O+i];}
            require(old_sum==new_sum,"CPU encoder equivalence failed");dot_products++;
        }
    }
    std::cout<<"{\"cpu_migration_tests\":\"passed\",\"registered_parameters\":"<<l.count
        <<",\"encoder_begin\":0,\"zeroed_entries\":"<<changed<<",\"protected_floats\":450304,\"encoder_dot_products\":"
        <<dot_products<<",\"gpu_equivalence_verified\":false,\"cuda_calls\":0}\n";
}
}
int main(int argc,char** argv){try{
    if(argc==2&&std::string(argv[1])=="--cpu-self-test"){cpu_test();return 0;}
    require(argc==5&&std::string(argv[1])=="--migrate","usage: observable-prev-action-migration --cpu-self-test | --migrate INPUT EXPECTED_SHA NEW_CHECKPOINT");
    migrate(argv[2],argv[3],argv[4]);return 0;
}catch(const std::exception& e){std::cerr<<"previous-action migration: "<<e.what()<<'\n';return 1;}}
