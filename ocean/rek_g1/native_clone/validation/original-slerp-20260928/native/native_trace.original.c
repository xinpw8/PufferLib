/* CPU/file-only instrumentation. Composer, entry matcher and math backend are
 * immutable snapshots; this file supplies explicit calls and JSONL observations. */
#include "sonic_motion_composer_native.h"
#include "sonic_motion_composer_libm_candidate.h"
#include "sonic_motion_entry_matcher_native.h"
#include "fixture_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static FILE* out;
static SonicMotionComposerNativeClip clips[2];
static void ok(int status){if(status){fprintf(stderr,"native status %d\n",status);exit(2);}}
static void scalar(float x){
    uint32_t b;memcpy(&b,&x,4);if(!isfinite(x)){fprintf(stderr,"nonfinite trace\n");exit(2);}
    fprintf(out,"{\"value\":%.9g,\"bits\":\"0x%08x\"}",x,b);
}
static void array(const float* x,int n){
    fprintf(out,"{\"values\":[");for(int i=0;i<n;i++){if(!isfinite(x[i]))exit(2);fprintf(out,"%s%.9g",i?",":"",x[i]);}
    fprintf(out,"],\"bits\":[");for(int i=0;i<n;i++){uint32_t b;memcpy(&b,x+i,4);fprintf(out,"%s\"0x%08x\"",i?",":"",b);}fprintf(out,"]}");
}
static int clip_id(const SonicMotionComposerNativeLayer* l){
    if(!l->has_clip)return 0;
    for(int i=0;i<2;i++)if(l->clip.dof_position_mujoco==clips[i].dof_position_mujoco)return i?370:377;
    fprintf(stderr,"unrecognized clip pointer\n");exit(2);
}
static void layer(const SonicMotionComposerNativeLayer* l){
    fprintf(out,"{\"clip_id\":%d,\"has_clip\":%d,\"has_config\":%d,\"active\":%d,\"heading_valid\":%d,\"heading_resync\":%d,\"start_frame\":%d,\"end_frame\":%d",
        clip_id(l),l->has_clip,l->has_config,l->active,l->heading_valid,l->heading_resync,l->start_frame,l->end_frame);
#define LF(name) fprintf(out,",\"" #name "\":");scalar(l->name)
    LF(cursor);LF(speed);LF(per_tick);LF(prev_heading);LF(last_heading_delta);
#undef LF
    fprintf(out,"}");
}
static void state(const SonicMotionComposerNative* c){
    float own;ok(sonic_motion_composer_native_heading_clip_ownership(c,&own));
    fprintf(out,"{\"current\":");layer(&c->current_layer);fprintf(out,",\"from\":");layer(&c->from_layer);
    fprintf(out,",\"xt\":%d,\"w_in\":%d,\"w_out\":%d,\"w_total\":%d,\"action_playing\":%d,\"action_move_id\":%d,\"pending_heading_delta\":",
        c->xt,c->w_in,c->w_out,c->w_total,c->action_playing,c->action_move_id);
    scalar(c->pending_heading_delta);fprintf(out,",\"heading_ownership\":");scalar(own);fprintf(out,"}");
}
static float* load(const char* dir,int id,const char* kind,size_t n){
    char p[4096];snprintf(p,sizeof(p),"%s/motion_%d_%s.f32le",dir,id,kind);
    FILE*f=fopen(p,"rb");if(!f){perror(p);exit(2);}float* a=malloc(n*4);if(!a)exit(2);
    if(fread(a,4,n,f)!=n||fgetc(f)!=EOF)exit(2);
    fclose(f);
    for(size_t i=0;i<n;i++)if(!isfinite(a[i]))exit(2);
    return a;
}
/* Unchanged ordered native5 clip-loader preprocessing, not a Unity oracle. */
static void normalize_heading(float* q,size_t n){
    float w=q[0],x=q[1],y=q[2],z=q[3];
    volatile float xy=x*y,zw=z*w,yy=y*y,zz=z*z,cross=xy+zw,sq=yy+zz;
    volatile float num=2.f*cross,twice=2.f*sq,den=1.f-twice;
    float angle=atan2f(num,den);if(fabsf(angle)<1e-6f)return;
    volatile float half=-.5f*angle;float c=cosf(half),s=sinf(half);
    for(size_t i=0;i<n;i+=4){float w0=q[i],x0=q[i+1],y0=q[i+2],z0=q[i+3];
        volatile float wc=w0*c,zs=z0*s,xc=x0*c,ys=y0*s,xs=x0*s,yc=y0*c,zc=z0*c,ws=w0*s;
        q[i]=wc-zs;q[i+1]=xc-ys;q[i+2]=xs+yc;q[i+3]=zc+ws;}
}
int main(int argc,char**argv){
    if(argc!=3){fprintf(stderr,"usage: native-trace ASSETS NEW_JSONL\n");return 2;}
    out=fopen(argv[2],"wx");if(!out){perror(argv[2]);return 2;}
    const int ids[2]={377,370},frames[2]={39,36},offsets[5]={0,1,5,10,15};
    float* features[2];
    for(int i=0;i<2;i++){
        float* p=load(argv[1],ids[i],"dof_position",frames[i]*29);
        float* q=load(argv[1],ids[i],"root_rotation_wxyz",frames[i]*4);normalize_heading(q,frames[i]*4);
        clips[i]=(SonicMotionComposerNativeClip){p,q,frames[i]*29,frames[i]*4,frames[i],50};
        features[i]=load(argv[1],ids[i],"foot_features",frames[i]*6);
    }
    fprintf(out,"{\"event\":\"header\",\"schema\":\"rek.native_composer_trace.v1\",\"producer\":\"native_cpu_port\",\"fixture_sha256\":\"%s\",\"fp32_bits_authoritative\":true,\"root_quaternion_order\":\"wxyz\",\"gpu\":false,\"physics\":false}\n",FIXTURE_SHA256);
    for(int order=0;order<2;order++)for(int rep=0;rep<2;rep++){
        SonicMotionEntryMatcherNative m;SonicMotionEntryMatcherNativeFeatureSlot slots[2];
        ok(sonic_motion_entry_matcher_native_init(&m,50,slots,2));
        for(int i=0;i<2;i++)ok(sonic_motion_entry_matcher_native_register(&m,&clips[i],features[i],frames[i]*6));
        SonicMotionComposerNativeBackends b={sonic_motion_composer_libm_candidate_quaternion_slerp,
            sonic_motion_composer_libm_candidate_atan2_f,sonic_motion_composer_libm_candidate_sin_cos_f,
            sonic_motion_entry_matcher_native_callback,&m};
        SonicMotionComposerNative c;SonicMotionComposerNativeAdvanceResult adv;float delta;
        ok(sonic_motion_composer_native_init(&c,50,&b));
        ok(sonic_motion_composer_native_play_action_immediate(&c,&clips[0],fixture_configs));
        for(int t=0;t<17;t++){ok(sonic_motion_composer_native_advance(&c,&adv));ok(sonic_motion_composer_native_consume_heading_delta(&c,&delta));}
        fprintf(out,"{\"event\":\"warmup_end\",\"trial\":%d,\"state\":",order*2+rep);state(&c);fprintf(out,"}\n");
        if(order==0)ok(sonic_motion_composer_native_reset(&c));
        ok(sonic_motion_composer_native_play_action(&c,&clips[0],fixture_configs));
        if(order==1)ok(sonic_motion_composer_native_reset(&c));
        for(int t=0;t<160;t++){
            if(t==50)ok(sonic_motion_composer_native_play_action(&c,&clips[1],fixture_configs+1));
            if(t==110)ok(sonic_motion_composer_native_play_action(&c,&clips[0],fixture_configs));
            ok(sonic_motion_composer_native_set_locomotion_speed(&c,1.f));
            SonicMotionComposerNativeReferenceTiming timing;
            for(int i=0;i<10;i++){timing.current_offsets[i]=offsets[i%5];timing.next_offsets[i]=offsets[i%5]+1;}
            float p[290],next[290],q[40];
            SonicMotionComposerNativeReferenceOutput ref={p,next,q,290,290,40};
            ok(sonic_motion_composer_native_build_reference_rows(&c,&timing,NULL,&ref));
            fprintf(out,"{\"event\":\"row\",\"trial\":%d,\"order\":\"%s\",\"repeat\":%d,\"tick\":%d,\"pre\":",
                order*2+rep,order?"idle_then_reset":"reset_then_idle",rep,t);state(&c);
            fprintf(out,",\"references\":[");
            for(int i=0;i<5;i++){
                float wxyz[4]={q[4*i+3],q[4*i],q[4*i+1],q[4*i+2]},vel[29];
                for(int j=0;j<29;j++){volatile float d=next[i*29+j]-p[i*29+j];volatile float v=d*50.f;vel[j]=v;}
                fprintf(out,"%s{\"frames_ahead\":%d,\"dof_position_mujoco\":",i?",":"",offsets[i]);array(p+i*29,29);
                fprintf(out,",\"root_quaternion_wxyz\":");array(wxyz,4);
                fprintf(out,",\"native_reference_velocity_estimate\":");array(vel,29);
                fprintf(out,",\"reference_root_position\":null}");
            }
            ok(sonic_motion_composer_native_advance(&c,&adv));fprintf(out,"],\"post_advance\":");state(&c);
            fprintf(out,",\"wrapped\":{\"current\":%d,\"from\":%d}",adv.current.wrapped,adv.outgoing.wrapped);
            ok(sonic_motion_composer_native_consume_heading_delta(&c,&delta));fprintf(out,",\"consumed_heading_delta\":");scalar(delta);
            fprintf(out,",\"post_consume\":");state(&c);fprintf(out,"}\n");
        }
        fprintf(out,"{\"event\":\"trial_end\",\"trial\":%d,\"rows\":160}\n",order*2+rep);
    }
    fprintf(out,"{\"event\":\"native_end\",\"complete\":true,\"rows\":640,\"trials\":4}\n");
    if(ferror(out)||fclose(out))return 2;
    for(int i=0;i<2;i++){free((void*)clips[i].dof_position_mujoco);free((void*)clips[i].root_quaternion_wxyz);free(features[i]);}
    return 0;
}
