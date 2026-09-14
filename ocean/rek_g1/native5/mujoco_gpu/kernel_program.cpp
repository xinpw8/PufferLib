#include "kernel_program.h"
#include "native_module.h"
#include "cJSON.h"
#include <algorithm>
#include <climits>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <stdexcept>

namespace rek_mjgpu {
namespace {
using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
std::string string_field(const cJSON* object, const char* name) {
    const auto* field=cJSON_GetObjectItemCaseSensitive(object,name);
    if(!cJSON_IsString(field)||!field->valuestring)throw std::runtime_error(std::string("Missing catalog string: ")+name);
    return field->valuestring;
}
void checked(CUresult result, const char* name) {
    if(result==CUDA_SUCCESS)return;
    const char* message=nullptr;cuGetErrorString(result,&message);
    throw std::runtime_error(std::string(name)+": "+(message?message:"CUDA error"));
}
struct ModuleDeleter {void operator()(RekMjGpuModule* p)const{rek_mjgpu_module_unload(p);}};
using Module = std::unique_ptr<RekMjGpuModule,ModuleDeleter>;

// Array ABI has a uniform layout, but the element width is compiled into the
// generated kernel. Validate both scalar type and vector/matrix width at bind.
void validate_array(const std::string& declared,const ArrayStorage& storage,const std::string& name) {
    std::string type=std::regex_replace(declared,std::regex("\\s+"),"");
    if(type.rfind("wp::array_t<",0)!=0||type.back()!='>')throw std::runtime_error("Expected array argument for "+name);
    type=type.substr(12,type.size()-13);
    int components=1;
    std::smatch match;
    if(std::regex_match(type,match,std::regex("wp::vec_t<([0-9]+),(.+)>"))){
        components=std::stoi(match[1]);type=match[2];
    }else if(std::regex_match(type,match,std::regex("wp::mat_t<([0-9]+),([0-9]+),(.+)>"))){
        components=std::stoi(match[1])*std::stoi(match[2]);type=match[3];
    }else if(std::regex_match(type,match,std::regex("wp::quat_t<(.+)>"))){
        components=4;type=match[1];
    }
    const bool same=(type=="wp::float32"&&storage.element==Element::F32)
        ||(type=="wp::int32"&&storage.element==Element::I32)
        ||(type=="bool"&&storage.element==Element::U8);
    if(!same||storage.components!=components)
        throw std::runtime_error("Array type mismatch for "+name+": kernel expects "+declared);
}
struct Value {
    WarpArray array;
    int32_t integer=0;
    float scalar=0;
    bool boolean=false;
};
struct Operation {
    ScheduleNode::Kind kind;
    CUfunction function=nullptr;
    LaunchBounds bounds;
    unsigned grid[3]={1,1,1},block[3]={1,1,1},shared_bytes=0;
    std::vector<Value> values;
    std::vector<void*> parameters;
    CUdeviceptr dst=0,src=0;
    std::size_t bytes=0;
};
}

struct KernelProgram::Impl {
    std::map<std::string,Module> modules;
    std::vector<std::unique_ptr<Operation>> operations;
};

KernelProgram::KernelProgram(ModelData& data,const std::string& catalog_path,const ScheduleSpec& spec)
    :impl_(std::make_unique<Impl>()) {
    std::ifstream input(catalog_path);
    if(!input)throw std::runtime_error("Cannot open native kernel catalog");
    const std::string bytes((std::istreambuf_iterator<char>(input)),{});
    Json document(cJSON_Parse(bytes.c_str()),cJSON_Delete);
    if(!document||string_field(document.get(),"schema")!="rek-native-mujoco-kernel-catalog-v1")
        throw std::runtime_error("Invalid native kernel catalog");
    const auto* modules=cJSON_GetObjectItemCaseSensitive(document.get(),"modules");
    if(!cJSON_IsArray(modules))throw std::runtime_error("Missing native module inventory");
    for(const auto& node:spec){
        auto operation=std::make_unique<Operation>();operation->kind=node.kind;
        if(node.kind!=ScheduleNode::Kind::Kernel){
            const auto dst=data.array(node.destination);
            operation->dst=reinterpret_cast<CUdeviceptr>(dst.data);
            operation->bytes=data.byte_size(node.destination);
            if(node.kind==ScheduleNode::Kind::Copy){
                const auto src=data.array(node.source);
                if(data.byte_size(node.source)!=operation->bytes)throw std::runtime_error("Native copy sizes differ");
                operation->src=reinterpret_cast<CUdeviceptr>(src.data);
            }
            if(operation->bytes)impl_->operations.push_back(std::move(operation));
            continue;
        }
        if(!node.bounds.size)continue;
        const cJSON* module=nullptr;
        for(const auto* candidate=modules->child;candidate;candidate=candidate->next){
            if(string_field(candidate,"name")==node.module){
                if(module)throw std::runtime_error("Duplicate native module name");
                module=candidate;
            }
        }
        if(!module)throw std::runtime_error("Missing cached native module: "+node.module);
        const auto* block=cJSON_GetObjectItemCaseSensitive(module,"blockDim");
        if(!cJSON_IsNumber(block)||block->valuedouble!=block->valueint||block->valueint<1||block->valueint>1024)
            throw std::runtime_error("Missing valid compiled CUDA block dimension");
        if(node.block_dim&&node.block_dim!=block->valueint)
            throw std::runtime_error("Launch block dimension differs from compiled WP_TILE_BLOCK_DIM: "+node.module);
        operation->block[0]=block->valueint;
        if(node.shared_bytes<0)throw std::runtime_error("Negative dynamic shared memory");
        operation->shared_bytes=node.shared_bytes;
        operation->bounds=node.bounds;
        const std::size_t grid=(node.bounds.size-1)/operation->block[0]+1;
        if(grid>INT_MAX)throw std::runtime_error("Native CUDA grid overflow");
        operation->grid[0]=static_cast<unsigned>(grid);
        const auto* kernels=cJSON_GetObjectItemCaseSensitive(module,"kernels");
        if(!cJSON_IsArray(kernels))throw std::runtime_error("Missing kernel inventory");
        const cJSON* kernel=nullptr;
        for(const auto* candidate=kernels->child;candidate;candidate=candidate->next){
            const auto symbol=string_field(candidate,"symbol");
            if(symbol.rfind(node.entry_prefix,0)==0){
                if(kernel)throw std::runtime_error("Ambiguous native entry prefix: "+node.entry_prefix);
                kernel=candidate;
            }
        }
        if(!kernel)throw std::runtime_error("Missing native kernel: "+node.entry_prefix);
        const auto* shared=cJSON_GetObjectItemCaseSensitive(kernel,"sharedBytes");
        if(!cJSON_IsNumber(shared)||shared->valuedouble!=shared->valueint||shared->valueint<0)
            throw std::runtime_error("Missing kernel dynamic shared-memory metadata");
        if(node.shared_bytes&&node.shared_bytes!=shared->valueint)
            throw std::runtime_error("Launch dynamic shared memory differs from generated metadata: "+node.entry_prefix);
        operation->shared_bytes=shared->valueint;
        const auto* parameters=cJSON_GetObjectItemCaseSensitive(kernel,"parameters");
        if(!cJSON_IsArray(parameters)||cJSON_GetArraySize(parameters)!=static_cast<int>(node.parameters.size()+1)
            ||string_field(cJSON_GetArrayItem(parameters,0),"type")!="wp::launch_bounds_t")
            throw std::runtime_error("Native kernel argument count or launch-bounds ABI mismatch: "+node.entry_prefix);
        auto& loaded=impl_->modules[node.module];
        if(!loaded){
            loaded.reset(rek_mjgpu_module_load(string_field(module,"ptx").c_str(),string_field(module,"ptxSha256").c_str()));
            if(!loaded)throw std::runtime_error(rek_mjgpu_error());
        }
        if(rek_mjgpu_module_function(loaded.get(),string_field(kernel,"symbol").c_str(),&operation->function))
            throw std::runtime_error(rek_mjgpu_error());
        operation->values.resize(node.parameters.size());
        operation->parameters.resize(node.parameters.size()+1);
        operation->parameters[0]=&operation->bounds;
        for(std::size_t i=0;i<node.parameters.size();i++){
            const auto& parameter=node.parameters[i];auto& value=operation->values[i];
            const auto declared=string_field(cJSON_GetArrayItem(parameters,static_cast<int>(i+1)),"type");
            void* address=nullptr;
            switch(parameter.kind){
            case ScheduleParameter::Kind::Array:{
                const auto found=data.arrays().find(parameter.field);
                if(found==data.arrays().end())throw std::runtime_error("Missing native array: "+parameter.field);
                validate_array(declared,found->second,parameter.field);
                value.array=found->second.view;address=&value.array;break;
            }
            case ScheduleParameter::Kind::I32:
                if(declared!="wp::int32")throw std::runtime_error("Expected int32 kernel argument");
                value.integer=parameter.integer;address=&value.integer;break;
            case ScheduleParameter::Kind::F32:
                if(declared!="wp::float32")throw std::runtime_error("Expected float32 kernel argument");
                value.scalar=parameter.scalar;address=&value.scalar;break;
            case ScheduleParameter::Kind::Boolean:
                if(declared!="bool")throw std::runtime_error("Expected bool kernel argument");
                value.boolean=parameter.boolean;address=&value.boolean;break;
            }
            operation->parameters[i+1]=address;
        }
        impl_->operations.push_back(std::move(operation));
    }
}
KernelProgram::~KernelProgram()=default;
std::size_t KernelProgram::nodes()const{return impl_->operations.size();}
CUgraphNode KernelProgram::append_to_graph(CUgraph graph,CUgraphNode predecessor)const{
    CUcontext context=nullptr;checked(cuCtxGetCurrent(&context),"get graph context");
    for(const auto& operation:impl_->operations){
        CUgraphNode next=nullptr;
        const auto* dependencies=predecessor?&predecessor:nullptr;
        const std::size_t count=predecessor?1:0;
        switch(operation->kind){
        case ScheduleNode::Kind::Kernel:{
            CUDA_KERNEL_NODE_PARAMS p={};p.func=operation->function;
            p.gridDimX=operation->grid[0];p.gridDimY=operation->grid[1];p.gridDimZ=operation->grid[2];
            p.blockDimX=operation->block[0];p.blockDimY=operation->block[1];p.blockDimZ=operation->block[2];
            p.sharedMemBytes=operation->shared_bytes;p.kernelParams=operation->parameters.data();
            checked(cuGraphAddKernelNode(&next,graph,dependencies,count,&p),"add bound GPU kernel node");break;
        }
        case ScheduleNode::Kind::Copy:{
            CUDA_MEMCPY3D p={};p.srcMemoryType=CU_MEMORYTYPE_DEVICE;p.srcDevice=operation->src;
            p.dstMemoryType=CU_MEMORYTYPE_DEVICE;p.dstDevice=operation->dst;
            p.srcPitch=p.dstPitch=operation->bytes;p.srcHeight=p.dstHeight=1;
            p.WidthInBytes=operation->bytes;p.Height=1;p.Depth=1;
            checked(cuGraphAddMemcpyNode(&next,graph,dependencies,count,&p,context),"add GPU copy node");break;
        }
        case ScheduleNode::Kind::Zero:{
            CUDA_MEMSET_NODE_PARAMS p={};p.dst=operation->dst;p.elementSize=1;p.width=operation->bytes;p.height=1;
            checked(cuGraphAddMemsetNode(&next,graph,dependencies,count,&p,context),"add GPU zero node");break;
        }
        }
        predecessor=next;
    }
    return predecessor;
}
void KernelProgram::launch(CUstream stream)const{
    for(std::size_t i=0;i<impl_->operations.size();i++)launch_one(i,stream);
}
void KernelProgram::launch_one(std::size_t index,CUstream stream)const{
        const auto& operation=impl_->operations.at(index);
        switch(operation->kind){
        case ScheduleNode::Kind::Kernel:
            if(rek_mjgpu_launch(operation->function,operation->grid,operation->block,
                    operation->shared_bytes,operation->parameters.data(),stream))
                throw std::runtime_error(rek_mjgpu_error());
            break;
        case ScheduleNode::Kind::Copy:
            checked(cuMemcpyDtoDAsync(operation->dst,operation->src,operation->bytes,stream),"native device copy");break;
        case ScheduleNode::Kind::Zero:
            checked(cuMemsetD8Async(operation->dst,0,operation->bytes,stream),"native device zero");break;
        }
}
}
