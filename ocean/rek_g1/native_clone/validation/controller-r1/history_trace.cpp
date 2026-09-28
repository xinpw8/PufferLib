#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "native5/robot_history.h"
#include "g1_heading_native.h"

static int atan_backend(void*,float y,float x,float* out) {*out=atan2f(y,x);return 1;}
static int sincos_backend(void*,float a,float* s,float* c) {*s=sinf(a);*c=cosf(a);return 1;}
static void read(void* p,size_t n,FILE* f) {if(fread(p,1,n,f)!=n)std::exit(2);}
int main(int argc,char** argv) {
    if(argc!=4)return 2;
    FILE* in=fopen(argv[1],"rb");FILE* out=fopen(argv[2],"wb");FILE* heading=fopen(argv[3],"wb");
    if(!in||!out||!heading)return 2;
    float history[930]{},tokens[64];int operations;
    read(tokens,sizeof(tokens),in);read(&operations,sizeof(operations),in);
    for(int op=0;op<operations;++op) {
        int kind;read(&kind,sizeof(kind),in);
        if(kind==0)std::memset(history,0,sizeof(history));
        else if(kind==1) {
            float channels[93];read(channels,sizeof(channels),in);
            const int widths[5]={3,29,29,29,3},groups[5]={0,30,320,610,900};int index=0;
            for(int g=0;g<5;++g)for(int c=0;c<widths[g];++c)
                rek_native_history_push_channel(history,groups[g],widths[g],c,channels[index++]);
        }else if(kind==2){
            float decoder[994];for(int i=0;i<994;++i)decoder[i]=rek_native_decoder_value(tokens,history,i);
            if(fwrite(decoder,sizeof(decoder),1,out)!=1)return 2;
        }else return 2;
    }
    int count;read(&count,sizeof(count),in);
    const SonicMotionComposerNativeBackends backends{nullptr,atan_backend,sincos_backend,nullptr,nullptr};
    for(int i=0;i<count;++i){
        float q[4],angle,yaw[4],identity[4]={1,0,0,0};read(q,sizeof(q),in);
        if(!rek_g1_heading_angle(&backends,q,&angle)||!rek_g1_initial_heading(&backends,q,identity,yaw))return 2;
        fwrite(&angle,sizeof(angle),1,heading);fwrite(yaw,sizeof(yaw),1,heading);
    }
    if(fgetc(in)!=EOF)return 2;
    fclose(in);fclose(out);fclose(heading);return 0;
}
