/* CPU-only capture/substitution boundary. No approximations or missing-key fallback. */
#include <stdint.h>

static FILE* boundary_log;
static uint64_t boundary_calls;
static int boundary_trial=-1,boundary_tick=-1;
static const char* boundary_phase="setup";
typedef struct { uint32_t input[9],output[4]; } BoundaryRecord;
static BoundaryRecord* boundary_table;
static uint32_t boundary_count;
static int boundary_replay;
static int boundary_compare(const void* lhs,const void* rhs){
    const uint32_t* a=lhs;const uint32_t* b=rhs;
    for(int i=0;i<9;i++){if(a[i]<b[i])return -1;if(a[i]>b[i])return 1;}return 0;
}
static void boundary_words(const uint32_t* p,int n){
    fprintf(boundary_log,"[");for(int i=0;i<n;i++)fprintf(boundary_log,"%s\"0x%08x\"",i?",":"",p[i]);fprintf(boundary_log,"]");
}
static void boundary_open(const char* log,const char* table){
    const uint32_t endian=1;if(*(const uint8_t*)&endian!=1){fprintf(stderr,"little endian required\n");exit(2);}
    boundary_log=fopen(log,"wx");if(!boundary_log){perror(log);exit(2);}
    boundary_replay=table!=NULL;
    if(table){
        FILE* f=fopen(table,"rb");char magic[8];
        if(!f||fread(magic,1,8,f)!=8||memcmp(magic,"RSLPTB1\0",8)
                ||fread(&boundary_count,4,1,f)!=1||!boundary_count||boundary_count>100000)exit(2);
        boundary_table=malloc((size_t)boundary_count*sizeof(*boundary_table));if(!boundary_table)exit(2);
        if(fread(boundary_table,sizeof(*boundary_table),boundary_count,f)!=boundary_count||fgetc(f)!=EOF)exit(2);
        fclose(f);
        for(uint32_t i=0;i<boundary_count;i++){
            if(i&&boundary_compare(boundary_table+i-1,boundary_table+i)>=0)exit(2);
            for(int j=0;j<4;j++){float x;memcpy(&x,boundary_table[i].output+j,4);if(!isfinite(x))exit(2);}
        }
    }
    fprintf(boundary_log,"{\"event\":\"header\",\"schema\":\"rek.slerp_calls.v1\",\"fixture_sha256\":\"%s\",\"mode\":\"%s\",\"input_order\":\"a_wxyz,b_wxyz,t\",\"bits_authoritative\":true}\n",FIXTURE_SHA256,table?"oracle_substitution":"libm_capture");
}
static int boundary_slerp(void* context,const float* a,const float* b,float t,float* result){
    uint32_t input[9],output[4];memcpy(input,a,16);memcpy(input+4,b,16);memcpy(input+8,&t,4);
    if(boundary_replay){
        const BoundaryRecord* found=bsearch(input,boundary_table,boundary_count,sizeof(*boundary_table),boundary_compare);
        if(!found){
            fprintf(boundary_log,"{\"event\":\"missing_tuple\",\"call\":%llu,\"trial\":%d,\"tick\":%d,\"phase\":\"%s\",\"input_bits\":",(unsigned long long)boundary_calls,boundary_trial,boundary_tick,boundary_phase);
            boundary_words(input,9);fprintf(boundary_log,"}\n");fflush(boundary_log);return 0;
        }
        memcpy(result,found->output,16);
    }else if(!sonic_motion_composer_libm_candidate_quaternion_slerp(context,a,b,t,result))return 0;
    memcpy(output,result,16);
    fprintf(boundary_log,"{\"event\":\"call\",\"call\":%llu,\"trial\":%d,\"tick\":%d,\"phase\":\"%s\",\"input_bits\":",(unsigned long long)boundary_calls,boundary_trial,boundary_tick,boundary_phase);
    boundary_words(input,9);fprintf(boundary_log,",\"output_bits\":");boundary_words(output,4);fprintf(boundary_log,"}\n");boundary_calls++;return 1;
}
static void boundary_close(void){
    fprintf(boundary_log,"{\"event\":\"end\",\"complete\":true,\"calls\":%llu}\n",(unsigned long long)boundary_calls);
    if(ferror(boundary_log)||fclose(boundary_log))exit(2);
    free(boundary_table);
}
