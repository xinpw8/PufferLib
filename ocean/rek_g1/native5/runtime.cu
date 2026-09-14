#include "runtime_api.h"
#include "physics.cuh"
#include "motion_assets.cuh"
#include "measurement.cuh"
#include "robot_state.cuh"
#include "sonic_controller.cuh"
#include "device_storage.cuh"
#include "../g1_native_combat_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

using rek5::cuda_check;
namespace {
thread_local std::string runtime_error;
struct Combat {
    RekG1CudaNativeCombatState* states;
    RekG1HitContact* contacts;
    RekG1ImpactEvent* impacts;
    int32_t *route_offsets,*route_counts,*routes,*statuses,*score_delta;
    uint32_t *fall_events,*signals,*referee,*attributed,*scored;
    uint8_t *terminal,*episode_reset,*input_reset,*dampened,*begin,*complete,*clear;
    float *fall,*fight,*rewards,*terminals;
};
struct RuntimeView {
    rek5::PhysicsDescriptor p;
    RekNative5Buffers out;
    Combat c;
    RekG1CudaSemanticRow* scheduler;
    int *qindices,*vindices,*aindices,*roots,*com_roots;
    float *initial_qpos,*spawn_roots,*initial_heading;
    float *joints,*velocities,*omega,*base,*local_velocity,*entities,*observations;
    float *actions,*returns,*lengths,*hit_totals,*invalid_totals;
    int *dummy_offset,*failures,*phase;
    int* physics_stats;
    uint8_t *suspended,*enabled,*row_mask,*union_mask,*completed,*terminal_rows;
    uint8_t *robot_dampened,*robot_resetting,*masks;
    uint8_t* learner_masks;
    const float* external_actions;
    const uint8_t* external_override;
    RekNative5RoundResult* round_results;
    float round_seconds;
};
__global__ void apply_round_duration(RuntimeView v) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=v.p.arenas)return;
    auto& fight=v.c.states[a].combat.fight;
    if(fight.round_result==REK_G1_ROUND_IN_PROGRESS&&fight.time_remaining_seconds==fight.round_duration_seconds){
        fight.round_duration_seconds=v.round_seconds;fight.time_remaining_seconds=v.round_seconds;
    }
}
__global__ void reset_pose(RuntimeView v,const uint8_t* mask,bool reset_clock) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=v.p.arenas||!mask[a])return;
    for(int k=0;k<72;k++)v.p.qpos[a*72+k]=v.initial_qpos[k];
    for(int k=0;k<70;k++)v.p.qvel[a*70+k]=0;
    if(reset_clock)v.p.time[a]=0;
}
__global__ void counted_reset(RuntimeView v) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=v.p.arenas)return;
    bool begin=v.c.begin[a],complete=v.c.complete[a];v.union_mask[a]=begin||complete;
    for(int s=0;s<2;s++) {
        int row=2*a+s;v.row_mask[row]=complete;v.completed[row]|=begin||complete;
        if(begin||complete)for(int k=0;k<7;k++)v.p.qpos[a*72+s*36+k]=v.spawn_roots[s*7+k];
        if(complete)for(int j=0;j<29;j++) {
            v.p.qpos[a*72+v.qindices[s*29+j]]=0;v.p.qvel[a*70+v.vindices[s*29+j]]=0;
        }
    }
}
__global__ void clear_ctrl(float* ctrl,const uint8_t* mask,int arenas) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<arenas*58&&mask[i/58])ctrl[i]=0;
}
__global__ void arena_to_rows(const uint8_t* arena,uint8_t* rows,int count) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<count)rows[r]=arena[r/2];
}
__global__ void retain_dampening(uint8_t* desired,const uint8_t* previous,int rows) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows)desired[r]|=previous[r];
}
__global__ void gather(RuntimeView v,bool full) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=v.p.arenas*2)return;
    int a=row/2,s=row%2;
    float* entity=v.entities+row*86;
    for(int j=0;j<29;j++){
        float q=v.p.qpos[a*72+v.qindices[s*29+j]],dq=v.p.qvel[a*70+v.vindices[s*29+j]];
        v.joints[row*29+j]=q;v.velocities[row*29+j]=dq;
        if(full){entity[13+j]=q;entity[42+j]=dq;}
    }
    if(!full)return;
    for(int k=0;k<7;k++)entity[k]=v.p.qpos[a*72+s*36+k];
    for(int k=0;k<4;k++)v.base[row*4+k]=entity[3+k];
    int b=a*v.p.bodies+v.roots[s],root=a*v.p.bodies+v.com_roots[s];
    double angular[3],linear[3],lever[3];
    for(int k=0;k<3;k++){angular[k]=v.p.cvel[b*6+k];lever[k]=double(v.p.xipos[b*3+k])-v.p.com[root*3+k];}
    for(int k=0;k<3;k++)linear[k]=double(v.p.cvel[b*6+3+k])
        +(angular[(k+1)%3]*lever[(k+2)%3]-angular[(k+2)%3]*lever[(k+1)%3]);
    for(int k=0;k<3;k++) {
        double av=0,lv=0;
        for(int j=0;j<3;j++){double m=v.p.ximat[b*9+j*3+k];av+=m*angular[j];lv+=m*linear[j];}
        v.local_velocity[row*6+k]=v.omega[row*3+k]=entity[10+k]=float(av);
        v.local_velocity[row*6+3+k]=entity[7+k]=float(lv);
    }
}
__global__ void observation_pack(RuntimeView v,const float* semantic) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=v.p.arenas*2)return;
    float* o=v.observations+row*223;
    for(int side=0;side<2;side++){
        int source=side?(row^1):row;
        for(int k=0;k<71;k++)o[side*86+k]=v.entities[source*86+k];
        for(int k=0;k<15;k++)o[side*86+71+k]=v.c.fall[source*15+k];
    }
    for(int k=0;k<12;k++)o[172+k]=semantic[row*12+k];
    for(int k=0;k<39;k++)o[184+k]=v.c.fight[row*39+k];
}
__global__ void export_masks(RuntimeView v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<v.p.arenas*33&&v.learner_masks)v.learner_masks[i]=v.masks[(i/33)*66+i%33];
}
__device__ void direction(const float* o,double& fx,double& fy,double& norm,double& distance,double& bearing) {
    double w=o[3],x=o[4],y=o[5],z=o[6];norm=sqrt(w*w+x*x+y*y+z*z);
    w/=norm;x/=norm;y/=norm;z/=norm;
    fx=1-2*(y*y+z*z);fy=2*(w*z+x*y);
    double dx=double(o[86])-o[0],dy=double(o[87])-o[1];distance=sqrt(dx*dx+dy*dy);
    bearing=distance==0?0:atan2(-fy*dx+fx*dy,fx*dx+fy*dy);
    if(fx*fx+fy*fy==0&&distance!=0)bearing=NAN;
}
__device__ bool encode_observation(const float* o,float* dst) {
    double fx,fy,norm,distance,bearing;direction(o,fx,fy,norm,distance,bearing);
    for(int k=0;k<223;k++)dst[k]=o[k];
    dst[86]=float(distance);dst[87]=float(double(float(bearing))/3.14159265358979323846);
    dst[72]=float(double(dst[72])/180);dst[158]=float(double(dst[158])/180);
    dst[188]=float(double(dst[188])/120);dst[189]=float(double(dst[189])/120);
    bool finite=isfinite(norm)&&norm>0&&isfinite(bearing);
    for(int k=0;k<223;k++)finite=finite&&isfinite(o[k])&&isfinite(dst[k]);
    return finite;
}
__global__ void encode_fighters(RuntimeView v,float* dst) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=v.p.arenas*2)return;
    if(!encode_observation(v.observations+row*223,dst+row*223))atomicOr(v.failures+row/2,1);
}
__device__ void validate_external_action(RuntimeView v,int row,float action) {
    int category=isfinite(action)?int(action):-1;
    if(category<0||category>=33||float(category)!=action)atomicOr(v.failures+row/2,16);
    else if(!v.masks[row*33+category])v.invalid_totals[row/2]+=1;
}
__global__ void choose_actions(RuntimeView v) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=v.p.arenas)return;
    float learner=v.external_override&&v.external_override[a*2]
        ?v.external_actions[a*2]:v.out.actions[a];int category=isfinite(learner)?int(learner):-1;
    if(category<0||category>=33||float(category)!=learner){atomicOr(v.failures+a,16);category=0;}
    v.actions[a*2]=learner;
    if(category>=0&&category<33&&!v.masks[a*66+category])v.invalid_totals[a]+=1;
    if(v.external_override&&v.external_override[a*2+1]){
        float action=v.external_actions[a*2+1];validate_external_action(v,a*2+1,action);
        v.actions[a*2+1]=action;return;
    }
    if(v.c.terminals[a*2]!=0)v.dummy_offset[a]=0;
    const float* o=v.observations+(a*2+1)*223;
    const uint8_t* masks=v.masks+(a*2+1)*33;
    double fx,fy,norm,distance,bearing;direction(o,fx,fy,norm,distance,bearing);
    // CandidateApproachDummy evaluates yaw then sin/cos before its bearing.
    double yaw=atan2(fy,fx),dx=double(o[86])-o[0],dy=double(o[87])-o[1];
    double fwd=cos(yaw)*dx+sin(yaw)*dy,lat=-sin(yaw)*dx+cos(yaw)*dy;
    bearing=atan2(lat,fwd);
    int preferred=v.dummy_offset[a]+16;
    if(distance<.72)preferred=3;if(distance>1.25)preferred=2;
    if(fabs(bearing)>.16)preferred=bearing>0?6:7;
    if(o[79]!=0)preferred=1;
    int selected=masks[preferred]?preferred:(masks[1]?1:0);
    if(!(masks[preferred]||masks[1]||masks[0]))atomicOr(v.failures+a,32);
    for(int k=0;k<33;k++)if(masks[k]>1)atomicOr(v.failures+a,32);
    for(int k=180;k<184;k++)if(o[k]!=0&&o[k]!=1)atomicOr(v.failures+a,32);
    if(o[79]==0&&(!isfinite(norm)||norm<=1e-9))atomicOr(v.failures+a,32);
    v.actions[a*2+1]=float(selected);
    if(selected>=16&&selected<32)v.dummy_offset[a]=(selected-15)%16;
}
__global__ void flags(RuntimeView v,bool post) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=v.p.arenas*2)return;
    v.suspended[r]=v.robot_dampened[r]||v.robot_resetting[r];
    v.enabled[r]=!v.suspended[r]&&v.scheduler[r].status==0;
    v.c.routes[r]=v.scheduler[r].active_route_id;
    if(post){v.phase[r]=int(v.c.fall[r*15+8]);v.terminal_rows[r]=v.c.terminals[r]!=0;}
}
__global__ void scatter_controls(RuntimeView v,const float* controls,float* out) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=v.p.arenas*2)return;
    for(int j=0;j<29;j++)out[(r/2)*58+v.aindices[(r%2)*29+j]]=controls[r*29+j];
}
__global__ void export_observation(RuntimeView v,bool transition) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=v.p.arenas)return;
    const float* o=v.observations+a*446;float* dst=v.out.observations+a*223;
    if(v.learner_masks)for(int k=0;k<33;k++)v.learner_masks[a*33+k]=v.masks[a*66+k];
    bool finite=encode_observation(o,dst);
    for(int k=0;k<446;k++)finite=finite&&isfinite(o[k]);
    for(int k=0;k<223;k++)finite=finite&&isfinite(dst[k]);
    if(!finite)atomicOr(v.failures+a,1);
    if(v.c.statuses[a])atomicOr(v.failures+a,2);
    if(v.scheduler[a*2].status||v.scheduler[a*2+1].status)atomicOr(v.failures+a,4);
    if(v.physics_stats[a*4]>=128||v.physics_stats[a*4+1]||v.physics_stats[a*4+2])atomicOr(v.failures+a,8);
    v.out.rewards[a]=transition?v.c.rewards[a*2]:0;
    v.out.terminals[a]=transition?v.c.terminals[a*2]:0;
    const auto& fight=v.c.states[a].combat.fight;
    auto& result=v.round_results[a];bool terminal=transition&&v.c.terminals[a*2]!=0;
    if(terminal&&!result.terminal&&v.failures[a]==0){
        result.completed_rounds++;
        if(fight.round_result==REK_G1_ROUND_TIE)result.ties++;
        else if(fight.round_result==REK_G1_ROUND_REDO)result.redos++;
        else if((fight.round_result==REK_G1_ROUND_WON_BY_POINTS||fight.round_result==REK_G1_ROUND_WON_BY_KO)
                &&fight.round_winner_index>=0&&fight.round_winner_index<2)result.wins[fight.round_winner_index]++;
        else result.unclassified++;
        for(int s=0;s<2;s++)result.completed_points[s]+=fight.clean_hits[s];
    }
    result.phase=fight.phase;result.round_number=fight.current_round_number;
    result.round_result=fight.round_result;result.round_winner=fight.round_winner_index;
    result.fight_result=fight.fight_result;result.fight_winner=fight.fight_winner_index;
    for(int s=0;s<2;s++){result.points[s]=fight.clean_hits[s];result.falls[s]=fight.falls[s];}
    result.time_remaining_seconds=fight.time_remaining_seconds;result.failure_bits=v.failures[a];result.terminal=terminal;
    if(!transition)return;
    v.returns[a]+=v.c.rewards[a*2];v.lengths[a]+=1;v.hit_totals[a]+=v.c.scored[a];
    if(v.c.terminals[a*2]!=0&&v.failures[a]==0) {
        auto* log=reinterpret_cast<RekNative5Log*>(reinterpret_cast<char*>(v.out.logs)+a*v.out.log_stride_bytes);
        const auto& fight=v.c.states[a].combat.fight;
        log->score+=fight.clean_hits[0];log->episode_return+=v.returns[a];log->episode_length+=v.lengths[a];
        log->hits+=v.hit_totals[a];log->falls+=fight.falls[0];
        log->wins+=fight.round_winner_index==0;log->losses+=fight.round_winner_index==1;
        log->draws+=fight.round_winner_index<0;log->actions_invalid+=v.invalid_totals[a];log->n+=1;
        v.returns[a]=v.lengths[a]=v.hit_totals[a]=v.invalid_totals[a]=0;
    }
}
}

struct RekNative5Runtime {
    rek5::DeviceStorage storage;
    rek5::Physics* physics=nullptr;
    RekNative5Motion* motion=nullptr;
    rek5::CombatMeasurement* measurement=nullptr;
    RobotState* robot=nullptr;
    SonicController* controller=nullptr;
    RuntimeView view{};
    float *tokens=nullptr,*raw_actions=nullptr;
    uint8_t* all_arenas=nullptr;
    cudaStream_t stream=nullptr;
    char controller_error[1024]{};
    ~RekNative5Runtime(){
        sonic_controller_destroy(controller);robot_state_destroy(robot);delete measurement;delete motion;rek5::physics_close(physics);
    }
    unsigned rows_grid() const{return (view.p.arenas*2+127)/128;}
    unsigned arena_grid() const{return (view.p.arenas+127)/128;}
    void bind(cudaStream_t s){
        stream=s;physics->stream=s;robot->stream=s;
        if(!sonic_controller_set_stream(controller,s,controller_error,sizeof(controller_error)))throw std::runtime_error(controller_error);
    }
    void observe(){auto& c=view.c;auto* m=measurement;
        cuda_check(rek_g1_cuda_native_combat_observe(c.states,m->fall_floats,m->fall_integers,m->fall_valid,
            c.fall_events,c.signals,c.referee,c.score_delta,c.attributed,c.scored,c.terminal,
            c.fall,c.fight,c.rewards,c.terminals,c.statuses,view.p.arenas,stream));
    }
    void physical_full(const uint8_t* arenas,bool clock){
        reset_pose<<<arena_grid(),128,0,stream>>>(view,arenas,clock);
        clear_ctrl<<<(view.p.arenas*58+127)/128,128,0,stream>>>(physics->ctrl,arenas,view.p.arenas);
        rek5::physics_forward_selected(physics,arenas);
        arena_to_rows<<<rows_grid(),128,0,stream>>>(arenas,view.row_mask,view.p.arenas*2);
        cuda_check(robot_state_complete_reset(robot,view.row_mask));motion->reset(view.row_mask,view.initial_heading,stream);
    }
    void reset(){
        physical_full(all_arenas,true);rek5::measurement_reset(measurement,stream);
        auto& c=view.c;size_t a=view.p.arenas,rows=a*2;
        cuda_check(rek_g1_cuda_native_combat_init(c.states,c.statuses,a,stream));
        if(view.round_seconds>0)apply_round_duration<<<arena_grid(),128,0,stream>>>(view);
        for(auto p:{c.fall_events})cuda_check(cudaMemsetAsync(p,0,rows*sizeof(uint32_t),stream));
        for(auto p:{c.signals,c.referee,c.attributed,c.scored})cuda_check(cudaMemsetAsync(p,0,a*sizeof(uint32_t),stream));
        cuda_check(cudaMemsetAsync(c.score_delta,0,rows*sizeof(int32_t),stream));
        for(auto p:{c.terminal,c.episode_reset,c.begin,c.complete,c.clear})cuda_check(cudaMemsetAsync(p,0,a,stream));
        for(auto p:{c.input_reset,c.dampened,view.completed})cuda_check(cudaMemsetAsync(p,0,rows,stream));
        for(auto p:{view.returns,view.lengths,view.hit_totals,view.invalid_totals})cuda_check(cudaMemsetAsync(p,0,a*sizeof(float),stream));
        cuda_check(cudaMemsetAsync(view.dummy_offset,0,a*sizeof(int),stream));
        cuda_check(cudaMemsetAsync(view.round_results,0,a*sizeof(RekNative5RoundResult),stream));
        rek5::measurement_sample_reset_fall(measurement,motion->all_flags,stream);observe();
        gather<<<rows_grid(),128,0,stream>>>(view,true);observation_pack<<<rows_grid(),128,0,stream>>>(view,motion->observation12);
        export_observation<<<arena_grid(),128,0,stream>>>(view,false);cuda_check(cudaGetLastError());
    }
    void step(){
        auto& c=view.c;auto* m=measurement;int a=view.p.arenas,rows=a*2;
        choose_actions<<<arena_grid(),128,0,stream>>>(view);
        cuda_check(rek_g1_cuda_native_combat_begin_tick(c.states,c.fall_events,c.signals,c.referee,c.score_delta,
            c.attributed,c.scored,c.terminal,c.episode_reset,c.input_reset,c.statuses,a,stream));
        if(view.round_seconds>0)apply_round_duration<<<arena_grid(),128,0,stream>>>(view);
        rek5::measurement_clear_contacts(m,c.episode_reset,stream);physical_full(c.episode_reset,false);
        cuda_check(cudaMemsetAsync(view.completed,0,rows,stream));
        gather<<<rows_grid(),128,0,stream>>>(view,true);flags<<<rows_grid(),128,0,stream>>>(view,false);
        motion->pre(view.actions,view.local_velocity,view.suspended,stream);
        flags<<<rows_grid(),128,0,stream>>>(view,false);cuda_check(robot_state_set_active(robot,view.enabled));
        cuda_check(robot_state_prepare(robot,view.base,view.omega,view.joints,view.velocities,motion->heading,
            motion->positions,motion->next_positions,motion->rotations));
        if(!sonic_controller_encode(controller,robot->encoder_observations,tokens,controller_error,sizeof(controller_error)))throw std::runtime_error(controller_error);
        cuda_check(robot_state_decoder_input(robot,tokens));
        if(!sonic_controller_decode(controller,robot->decoder_observations,raw_actions,controller_error,sizeof(controller_error)))throw std::runtime_error(controller_error);
        cuda_check(robot_state_apply_actions(robot,raw_actions));
        for(int substep=0;substep<10;substep++){
            if(substep)gather<<<rows_grid(),128,0,stream>>>(view,false);
            cuda_check(robot_state_drive_prepare(robot,view.joints,view.velocities,substep));
            scatter_controls<<<rows_grid(),128,0,stream>>>(view,robot->controls,physics->ctrl);
            rek5::physics_step(physics,physics->ctrl);rek5::measurement_sample(m,substep,stream);
            flags<<<rows_grid(),128,0,stream>>>(view,false);
            cuda_check(rek_g1_cuda_native_combat_post_step_deferred(c.states,m->fall_floats,m->fall_integers,m->fall_valid,
                m->hit_integers,m->hit_floats,m->candidate_valid,m->order,m->offsets,m->counts,m->scan_valid,view.p.time,
                motion->composers,c.routes,c.impacts,c.route_offsets,c.route_counts,c.contacts,c.fall_events,c.signals,c.referee,
                c.score_delta,c.attributed,c.scored,c.terminal,c.input_reset,c.dampened,c.begin,c.complete,c.clear,c.statuses,
                a,size_t(view.p.capacity)*2,stream));
            rek5::measurement_clear_contacts(m,c.clear,stream);
            retain_dampening<<<rows_grid(),128,0,stream>>>(c.dampened,robot->dampened,rows);
            cuda_check(robot_state_set_dampened(robot,c.dampened,robot->controls));
            arena_to_rows<<<rows_grid(),128,0,stream>>>(c.begin,view.row_mask,rows);
            cuda_check(robot_state_begin_reset(robot,view.row_mask,robot->controls));
            counted_reset<<<arena_grid(),128,0,stream>>>(view);
            clear_ctrl<<<(a*58+127)/128,128,0,stream>>>(physics->ctrl,c.complete,a);
            rek5::physics_forward_selected(physics,view.union_mask);
            cuda_check(robot_state_complete_reset(robot,view.row_mask));motion->reset(view.row_mask,view.initial_heading,stream);
        }
        gather<<<rows_grid(),128,0,stream>>>(view,true);
        rek5::measurement_sample_reset_fall(m,view.completed,stream);observe();
        flags<<<rows_grid(),128,0,stream>>>(view,true);
        motion->post(view.local_velocity,view.phase,view.suspended,c.input_reset,view.completed,view.terminal_rows,stream);
        observation_pack<<<rows_grid(),128,0,stream>>>(view,motion->observation12);
        export_observation<<<arena_grid(),128,0,stream>>>(view,true);cuda_check(cudaGetLastError());
    }
};

extern "C" RekNative5Runtime* rek_native5_create(const RekNative5Config* cfg,const RekNative5Buffers* out,cudaStream_t s) {
    try{
        runtime_error.clear();
        if(!cfg||!out||cfg->abi_version!=REK_NATIVE5_RUNTIME_ABI||cfg->arenas<=0||!out->observations||!out->actions
           ||!out->rewards||!out->terminals||!out->logs||out->log_stride_bytes<sizeof(RekNative5Log))
            throw std::runtime_error("Invalid native5 configuration or device buffers");
        auto r=std::make_unique<RekNative5Runtime>();auto& v=r->view;auto& d=r->storage;int a=cfg->arenas,rows=a*2;
        if(!std::isfinite(cfg->round_seconds)||cfg->round_seconds<0||cfg->round_seconds>3600)
            throw std::runtime_error("round_seconds must be zero or finite in (0,3600]");
        v.round_seconds=cfg->round_seconds;
        r->physics=rek5::physics_create(cfg->model_path,cfg->physics_export_path,a,s);v.p=r->physics->data;v.out=*out;
        r->motion=new RekNative5Motion(*cfg,s);r->measurement=rek5::measurement_create(r->physics);
        auto* p=r->physics;auto* model=p->model;
        std::array<uint8_t,58> limited{};std::array<float,116> ranges{};
        for(int j=0;j<58;j++){
            int joint=model->actuator_trnid[p->actuator_ids[j]*2];limited[j]=model->jnt_limited[joint];
            ranges[j*2]=float(model->jnt_range[joint*2]);ranges[j*2+1]=float(model->jnt_range[joint*2+1]);
        }
        r->robot=robot_state_create(rows,s,limited.data(),ranges.data(),r->controller_error,sizeof(r->controller_error));
        if(!r->robot)throw std::runtime_error(r->controller_error);
        r->controller=sonic_controller_create(cfg->controller_encoder_path,cfg->controller_decoder_path,rows,s,r->controller_error,sizeof(r->controller_error));
        if(!r->controller)throw std::runtime_error(r->controller_error);
        v.scheduler=r->motion->rows;v.masks=r->motion->masks;v.robot_dampened=r->robot->dampened;v.robot_resetting=r->robot->resetting;
        v.qindices=d.upload(p->joint_qpos.data(),58);v.vindices=d.upload(p->joint_qvel.data(),58);v.aindices=d.upload(p->actuator_ids.data(),58);
        v.roots=d.upload(p->root_bodies.data(),2);int com_roots[2]={model->body_rootid[p->root_bodies[0]],model->body_rootid[p->root_bodies[1]]};v.com_roots=d.upload(com_roots,2);
        std::vector<float> pose=p->model_qpos0,headings(rows*4),spawn(14);
        for(int side=0;side<2;side++){
            for(int k=0;k<7;k++)spawn[side*7+k]=pose[side*36+k];
            for(int j=0;j<29;j++){
                int k=side*29+j;float q=r->motion->idle_positions[j];
                if(limited[k])q=std::min(ranges[k*2+1],std::max(ranges[k*2],q));pose[p->joint_qpos[k]]=q;
            }
            auto yaw=[](double w,double x,double y,double z){return atan2(2*(w*z+x*y),1-2*(y*y+z*z));};
            const float* base=pose.data()+side*36+3;auto ref=r->motion->idle_root_xyzw;
            double by=yaw(base[0],base[1],base[2],base[3]),ry=yaw(ref[3],ref[0],ref[1],ref[2]);
            double bw=cos(.5*by),bz=sin(.5*by),rw=cos(.5*ry),rz=-sin(.5*ry);
            for(int arena=0;arena<a;arena++){int row=arena*2+side;headings[row*4]=float(bw*rw-bz*rz);headings[row*4+3]=float(bw*rz+bz*rw);}
        }
        v.initial_qpos=d.upload(pose);v.initial_heading=d.upload(headings);v.spawn_roots=d.upload(spawn);
        v.joints=d.alloc<float>(rows*29);v.velocities=d.alloc<float>(rows*29);v.omega=d.alloc<float>(rows*3);v.base=d.alloc<float>(rows*4);
        v.local_velocity=d.alloc<float>(rows*6);v.entities=d.alloc<float>(rows*86);v.observations=d.alloc<float>(rows*223);v.actions=d.alloc<float>(rows);
        v.returns=d.alloc<float>(a);v.lengths=d.alloc<float>(a);v.hit_totals=d.alloc<float>(a);v.invalid_totals=d.alloc<float>(a);
        v.dummy_offset=d.alloc<int>(a);v.failures=d.alloc<int>(a);v.phase=d.alloc<int>(rows);v.physics_stats=p->stats;
        v.round_results=d.alloc<RekNative5RoundResult>(a);
        v.suspended=d.alloc<uint8_t>(rows);v.enabled=d.alloc<uint8_t>(rows);v.row_mask=d.alloc<uint8_t>(rows);v.union_mask=d.alloc<uint8_t>(a);
        v.completed=d.alloc<uint8_t>(rows);v.terminal_rows=d.alloc<uint8_t>(rows);r->all_arenas=d.alloc<uint8_t>(a);cuda_check(cudaMemset(r->all_arenas,1,a));
        r->tokens=d.alloc<float>(rows*64);r->raw_actions=d.alloc<float>(rows*29);auto& c=v.c;
        c.states=d.alloc<RekG1CudaNativeCombatState>(a);c.contacts=d.alloc<RekG1HitContact>(size_t(v.p.capacity)*2);
        c.impacts=d.alloc<RekG1ImpactEvent>(rek_g1_cuda_native_combat_impact_event_count());c.route_offsets=d.alloc<int32_t>(24);c.route_counts=d.alloc<int32_t>(24);
        c.routes=d.alloc<int32_t>(rows);c.statuses=d.alloc<int32_t>(a);c.score_delta=d.alloc<int32_t>(rows);c.fall_events=d.alloc<uint32_t>(rows);
        c.signals=d.alloc<uint32_t>(a);c.referee=d.alloc<uint32_t>(a);c.attributed=d.alloc<uint32_t>(a);c.scored=d.alloc<uint32_t>(a);
        c.terminal=d.alloc<uint8_t>(a);c.episode_reset=d.alloc<uint8_t>(a);c.input_reset=d.alloc<uint8_t>(rows);c.dampened=d.alloc<uint8_t>(rows);
        c.begin=d.alloc<uint8_t>(a);c.complete=d.alloc<uint8_t>(a);c.clear=d.alloc<uint8_t>(a);
        c.fall=d.alloc<float>(rows*15);c.fight=d.alloc<float>(rows*39);c.rewards=d.alloc<float>(rows);c.terminals=d.alloc<float>(rows);
        cuda_check(rek_g1_cuda_native_combat_upload_catalog(c.impacts,c.route_offsets,c.route_counts));r->bind(s);r->reset();
        cuda_check(cudaStreamSynchronize(s));
        std::fprintf(stderr,"native5 runtime: physics_backend=%s arenas=%d fighters=%d controller_bytes=%zu\n",rek5::physics_backend_name(p),a,rows,sonic_controller_resident_bytes(r->controller));
        std::fprintf(stderr,"model_sha256=%s export_sha256=%s motion_manifest_sha256=%s\n",p->model_sha256.c_str(),p->export_sha256.c_str(),r->motion->manifest_sha256.c_str());
        std::fprintf(stderr,"encoder_sha256=%s decoder_sha256=%s\n",sonic_controller_encoder_sha256(r->controller),sonic_controller_decoder_sha256(r->controller));
        return r.release();
    }catch(const std::exception& e){runtime_error=e.what();return nullptr;}
}
extern "C" int rek_native5_reset(RekNative5Runtime* r,cudaStream_t s){
    try{if(!r)throw std::runtime_error("Null native5 runtime");r->bind(s);r->reset();return 0;}
    catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_step(RekNative5Runtime* r,cudaStream_t s){
    try{if(!r)throw std::runtime_error("Null native5 runtime");r->bind(s);r->step();return 0;}
    catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_bind_action_mask(RekNative5Runtime* r,uint8_t* mask,cudaStream_t s){
    try{
        if(!r||!mask)throw std::runtime_error("Null native5 mask binding");r->view.learner_masks=mask;
        export_masks<<<(r->view.p.arenas*33+127)/128,128,0,s>>>(r->view);cuda_check(cudaGetLastError());return 0;
    }catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_bind_external_actions(RekNative5Runtime* r,const float* actions,const uint8_t* overrides,cudaStream_t s){
    try{
        if(!r||bool(actions)!=bool(overrides))throw std::runtime_error("External actions and overrides must be supplied together");
        cudaStreamCaptureStatus capture;cuda_check(cudaStreamIsCapturing(s,&capture));
        if(capture!=cudaStreamCaptureStatusNone)throw std::runtime_error("Cannot change external action binding during graph capture");
        r->view.external_actions=actions;r->view.external_override=overrides;return 0;
    }catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_get_device_view(RekNative5Runtime* r,RekNative5DeviceView* out){
    try{
        if(!r||!out)throw std::runtime_error("Null native5 device view argument");auto& v=r->view;
        *out={v.p.arenas,72,70,v.observations,v.masks,v.p.qpos,v.p.qvel,v.actions,v.c.rewards,v.c.terminals,v.round_results};return 0;
    }catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_encode_fighter_observations(RekNative5Runtime* r,float* out,cudaStream_t s){
    try{
        if(!r||!out)throw std::runtime_error("Null native5 encoded observation argument");
        encode_fighters<<<r->rows_grid(),128,0,s>>>(r->view,out);cuda_check(cudaGetLastError());return 0;
    }catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_read_snapshot(RekNative5Runtime* r,int arena,RekNative5Snapshot* out,cudaStream_t s){
    try{
        if(!r||!out||arena<0||arena>=r->view.p.arenas)throw std::runtime_error("Invalid native5 snapshot argument");
        cudaStreamCaptureStatus capture;cuda_check(cudaStreamIsCapturing(s,&capture));
        if(capture!=cudaStreamCaptureStatusNone)throw std::runtime_error("Snapshot is not capture-safe");
        auto& v=r->view;out->arena=arena;
        auto copy=[&](void* dst,const void* src,size_t n){cuda_check(cudaMemcpyAsync(dst,src,n,cudaMemcpyDeviceToHost,s));};
        copy(out->raw_observations,v.observations+arena*446,sizeof(out->raw_observations));
        copy(out->action_masks,v.masks+arena*66,sizeof(out->action_masks));
        copy(out->qpos,v.p.qpos+arena*72,sizeof(out->qpos));copy(out->qvel,v.p.qvel+arena*70,sizeof(out->qvel));
        copy(out->actions,v.actions+arena*2,sizeof(out->actions));copy(out->rewards,v.c.rewards+arena*2,sizeof(out->rewards));
        copy(out->terminals,v.c.terminals+arena*2,sizeof(out->terminals));copy(&out->round,v.round_results+arena,sizeof(out->round));
        cuda_check(cudaStreamSynchronize(s));return 0;
    }catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_check_status(RekNative5Runtime* r,cudaStream_t s){
    try{
        if(!r)throw std::runtime_error("Null native5 runtime");cuda_check(cudaStreamSynchronize(s));
        std::vector<int> status(r->view.p.arenas);cuda_check(cudaMemcpy(status.data(),r->view.failures,status.size()*sizeof(int),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<status.size();i++)if(status[i])throw std::runtime_error("Native5 arena "+std::to_string(i)+" sticky failure bits="+std::to_string(status[i])+" (1=finite/encoding,2=combat,4=scheduler,8=physics,16=action,32=dummy)");
        rek5::physics_check_status(r->physics);r->motion->check_status(s);return 0;
    }catch(const std::exception& e){runtime_error=e.what();return 1;}
}
extern "C" int rek_native5_close(RekNative5Runtime* r){
    int result=r?rek_native5_check_status(r,r->stream):0;delete r;return result;
}
extern "C" const char* rek_native5_error(void){return runtime_error.c_str();}
