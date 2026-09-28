#include "sonic_controller.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void cuda_ok(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
uint32_t bits(float value){uint32_t word;std::memcpy(&word,&value,sizeof(word));return word;}
float probe(size_t row,size_t column,size_t salt){return (float((row*131+column*17+salt*29)%2003)-1001.f)/317.f;}
struct Difference {size_t values=0,unequal_bits=0,unequal_values=0,first=SIZE_MAX;double max_abs=0;};
Difference difference(const std::vector<float>& a,const std::vector<float>& b){
    require(a.size()==b.size(),"comparison length mismatch");Difference d;d.values=a.size();
    for(size_t i=0;i<a.size();i++){require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite result");
        if(bits(a[i])!=bits(b[i])){d.unequal_bits++;if(d.first==SIZE_MAX)d.first=i;}
        d.unequal_values+=a[i]!=b[i];d.max_abs=std::max(d.max_abs,std::abs(double(a[i])-double(b[i])));}
    return d;
}
Difference report(size_t fixture,const char* label,const std::vector<float>& a,const std::vector<float>& b){
    auto d=difference(a,b);std::printf("{\"kind\":\"comparison\",\"fixture\":%zu,\"label\":\"%s\",\"values\":%zu,\"unequal_bits\":%zu,\"unequal_values\":%zu,\"max_abs\":%.12g,\"first\":",fixture,label,d.values,d.unequal_bits,d.unequal_values,d.max_abs);
    if(d.first==SIZE_MAX)std::printf("null");else std::printf("{\"index\":%zu,\"batch2\":%.9g,\"batch8\":%.9g,\"bits2\":%u,\"bits8\":%u}",d.first,a[d.first],b[d.first],bits(a[d.first]),bits(b[d.first]));
    std::printf("}\n");return d;
}
std::vector<float> first_two(const std::vector<float>& x,size_t width){require(x.size()==8*width,"expected eight rows");return {x.begin(),x.begin()+2*width};}
std::vector<float> connect(std::vector<float> history,const std::vector<float>& tokens,size_t batch){require(history.size()==batch*994&&tokens.size()==batch*64,"connected shape mismatch");for(size_t row=0;row<batch;row++)std::copy_n(tokens.data()+row*64,64,history.data()+row*994);return history;}
std::vector<float> read(const char* path,size_t count){std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"input file unavailable");require(f.tellg()==std::streamoff(count*4),"input file size mismatch");f.seekg(0);std::vector<float> data(count);f.read(reinterpret_cast<char*>(data.data()),count*4);require(bool(f),"input file read failed");for(float x:data)require(std::isfinite(x),"nonfinite input");return data;}
struct Controller {
    cudaStream_t stream{};SonicController* native=nullptr;size_t batch;
    float* input=nullptr;float* output=nullptr;char error[1024]{};
    Controller(size_t b,const char* encoder,const char* decoder):batch(b){cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));native=sonic_controller_create(encoder,decoder,batch,stream,error,sizeof(error));if(!native)throw std::runtime_error(error);cuda_ok(cudaMalloc(reinterpret_cast<void**>(&input),batch*1762*4));cuda_ok(cudaMalloc(reinterpret_cast<void**>(&output),batch*64*4));}
    ~Controller(){if(stream)cudaStreamSynchronize(stream);cudaFree(input);cudaFree(output);sonic_controller_destroy(native);if(stream)cudaStreamDestroy(stream);}
    std::vector<float> run(bool encode,const std::vector<float>& host){require(host.size()==batch*(encode?1762:994),"native input shape mismatch");cuda_ok(cudaMemcpyAsync(input,host.data(),host.size()*4,cudaMemcpyHostToDevice,stream));int ok=encode?sonic_controller_encode(native,input,output,error,sizeof(error)):sonic_controller_decode(native,input,output,error,sizeof(error));if(!ok)throw std::runtime_error(error);std::vector<float> result(batch*(encode?64:29));cuda_ok(cudaMemcpyAsync(result.data(),output,result.size()*4,cudaMemcpyDeviceToHost,stream));cuda_ok(cudaStreamSynchronize(stream));return result;}
};
void self_test(){std::vector<float> a(8*994),t(8*64);for(size_t i=0;i<a.size();i++)a[i]=float(i);for(size_t i=0;i<t.size();i++)t[i]=-float(i+1);auto c=connect(a,t,8);require(first_two(c,994)==connect(first_two(a,994),first_two(t,64),2),"first-two row/token copy");require(std::equal(c.begin()+64,c.begin()+994,a.begin()+64),"history changed");auto b=first_two(t,64);auto d=b;d[65]+=1;require(difference(b,d).unequal_bits==1&&difference(b,d).first==65,"mismatch comparison");std::vector<float> zeros{0.f},negative{-0.f};require(difference(zeros,negative).unequal_bits==1&&difference(zeros,negative).unequal_values==0,"signed zero accounting");std::puts("{\"kind\":\"self_test\",\"passed\":true,\"gpu_used\":false}");}
}
int main(int argc,char** argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
    require(argc==5||(argc==8&&std::string(argv[5])=="--inputs"),"usage: crossbatch encoder2 decoder2 encoder8 decoder8 [--inputs encoder8.f32 history8.f32]");
    Controller two(2,argv[1],argv[2]),eight(8,argv[3],argv[4]);
    std::printf("{\"kind\":\"header\",\"schema\":\"rek.native_crossbatch.v1\",\"comparison\":\"same first two rows, identical 930 history values\",\"encoder2\":\"%s\",\"decoder2\":\"%s\",\"encoder8\":\"%s\",\"decoder8\":\"%s\",\"physics_claim\":false}\n",sonic_controller_encoder_sha256(two.native),sonic_controller_decoder_sha256(two.native),sonic_controller_encoder_sha256(eight.native),sonic_controller_decoder_sha256(eight.native));
    const size_t starts[]={0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,128,254,384,510,640,766,768,1022};
    bool exact=true;size_t fixtures=argc==8?1:32;
    for(size_t fixture=0;fixture<fixtures;fixture++){
        std::vector<float> enc(8*1762,0),history(8*994);
        if(argc==8){enc=read(argv[6],enc.size());history=read(argv[7],history.size());}
        else for(size_t row=0;row<8;row++){size_t r=starts[fixture]+row;for(size_t c=0;c<580;c++)enc[row*1762+4+c]=probe(r,c,1);for(size_t c=0;c<60;c++)enc[row*1762+601+c]=probe(r,c,2);for(size_t c=0;c<994;c++)history[row*994+c]=probe(r,c,3);}
        auto enc2=first_two(enc,1762),h2=first_two(history,994);
        require(std::memcmp(enc2.data(),enc.data(),enc2.size()*4)==0&&std::memcmp(h2.data(),history.data(),h2.size()*4)==0,"crossbatch inputs differ");
        auto t2=two.run(true,enc2),t8=eight.run(true,enc);
        exact&=report(fixture,"encoder_tokens",t2,first_two(t8,64)).unequal_bits==0;
        auto direct2=two.run(false,h2),direct8=eight.run(false,history);
        exact&=report(fixture,"decoder_identical_raw_input",direct2,first_two(direct8,29)).unequal_bits==0;
        auto conn8=connect(history,t8,8),own2=connect(h2,t2,2),same2=connect(h2,first_two(t8,64),2);
        auto a8=eight.run(false,conn8),a2=two.run(false,own2),same_actions2=two.run(false,same2);
        exact&=report(fixture,"connected_own_tokens",a2,first_two(a8,29)).unequal_bits==0;
        exact&=report(fixture,"decoder_identical_batch8_tokens",same_actions2,first_two(a8,29)).unequal_bits==0;
    }
    std::printf("{\"kind\":\"summary\",\"fixtures\":%zu,\"tested_common_rows\":%zu,\"bit_exact_all\":%s,\"tolerance_relaxed\":false,\"unity_or_physics_parity_claimed\":false}\n",fixtures,fixtures*2,exact?"true":"false");return exact?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"crossbatch error: %s\n",e.what());return 2;}}
