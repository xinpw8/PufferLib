#include "sonic_onnx_reader.h"
#include <iostream>
#include <stdexcept>
#include <vector>

using sonic_onnx::Tensor;
std::vector<unsigned char> bytes(const Tensor& t) {
    if(t.external)throw std::runtime_error("External tensor storage");
    if(t.raw.data)return {t.raw.data,t.raw.data+t.raw.size};
    if(t.type==1&&!t.float_data.empty()) {
        const auto* first=reinterpret_cast<const unsigned char*>(t.float_data.data());
        return {first,first+t.float_data.size()*sizeof(float)};
    }
    throw std::runtime_error("Unrepresented tensor data");
}
void compare(const std::map<std::string,Tensor>& a,const std::map<std::string,Tensor>& b,bool constants) {
    if(a.size()!=b.size())throw std::runtime_error("Tensor count mismatch");
    size_t identical=0,batch_shapes=0,coefficients=0;
    for(const auto& item:a) {
        auto found=b.find(item.first);
        if(found==b.end())throw std::runtime_error("Tensor name mismatch: "+item.first);
        const auto& x=item.second;const auto& y=found->second;
        if(x.type!=y.type||x.dimensions!=y.dimensions)throw std::runtime_error("Tensor shape/type mismatch: "+item.first);
        auto av=bytes(x),bv=bytes(y);
        if(x.type==1)coefficients+=av.size()/4;
        if(av==bv){identical++;continue;}
        bool allowed=constants&&x.type==7&&av.size()==bv.size()&&av.size()>=8;
        if(allowed){
            std::int64_t first_a=0,first_b=0;
            std::memcpy(&first_a,av.data(),8);std::memcpy(&first_b,bv.data(),8);
            allowed=first_a==2&&first_b==8&&std::equal(av.begin()+8,av.end(),bv.begin()+8);
        }
        if(!allowed)throw std::runtime_error("Non-batch coefficient change: "+item.first);
        batch_shapes++;
    }
    std::cout<<"{\"group\":\""<<(constants?"constants":"initializers")<<"\",\"identical\":"<<identical
        <<",\"leading_int64_2_vs_8\":"<<batch_shapes<<",\"float32_coefficients\":"<<coefficients<<"}\n";
}
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::runtime_error("Expected batch2 encoder/decoder then batch8 encoder/decoder");
        for(int i=0;i<2;i++) {
            auto a=sonic_onnx::load(argv[1+i]),b=sonic_onnx::load(argv[3+i]);
            sonic_onnx::validate_io(a,2,i==0);sonic_onnx::validate_io(b,8,i==0);
            std::cout<<"{\"model\":\""<<(i==0?"encoder":"decoder")<<"\"}\n";
            compare(a.initializers,b.initializers,false);compare(a.constants,b.constants,true);
        }
        std::cout<<"{\"passed\":true,\"gpu_used\":false,\"inference_tested\":false}\n";
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
