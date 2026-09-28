/* Synthetic callback lookup test only; never initializes a composer. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#define FIXTURE_SHA256 "synthetic-boundary-test"
static int fallback_calls;
static int sonic_motion_composer_libm_candidate_quaternion_slerp(void* c,const float*a,const float*b,float t,float*out){
    (void)c;(void)a;(void)b;(void)t;(void)out;fallback_calls++;return 0;
}
#include "slerp_boundary.h"
int main(int argc,char**argv){
    (void)&boundary_close;
    if(argc!=2)return 2;
    char table[4096],log[4096];snprintf(table,sizeof(table),"%s/table.bin",argv[1]);snprintf(log,sizeof(log),"%s/calls.jsonl",argv[1]);
    float a[4]={1,0,0,0},b[4]={0,0,0,1},t=.25f;BoundaryRecord r;memcpy(r.input,a,16);memcpy(r.input+4,b,16);memcpy(r.input+8,&t,4);
    const float expected[4]={.9f,.1f,.2f,.3f};memcpy(r.output,expected,16);uint32_t n=1;
    FILE* f=fopen(table,"wx");if(!f)return 2;fwrite("RSLPTB1\0",1,8,f);fwrite(&n,4,1,f);fwrite(&r,sizeof(r),1,f);fclose(f);
    boundary_open(log,table);float out[4]={0};
    if(!boundary_slerp(NULL,a,b,t,out)||memcmp(out,expected,16)||fallback_calls)return 1;
    memcpy(out,a,16);
    if(boundary_slerp(NULL,a,b,.5f,out)||memcmp(out,a,16)||fallback_calls||boundary_calls!=1)return 1;
    /* A miss returns false without output mutation or libm fallback. The real
       fixture's ok() exits2 immediately, so no complete trace is emitted. */
    if(fclose(boundary_log))return 2;
    free(boundary_table);
    puts("{\"synthetic_lookup_checks\":5,\"failures\":0,\"composer_or_unity_executed\":false}");return 0;
}
