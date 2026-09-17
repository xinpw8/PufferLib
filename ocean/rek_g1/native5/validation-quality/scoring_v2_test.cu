// Synthetic contact-enter fixtures call the actual production adapter, with
// real pinned catalog impact times. They do not measure physical contacts.
#include "../fast_runtime.cu"
extern "C" {
#include "../../g1_strike_catalog.h"
}

namespace {
struct Case { int route,limb,legacy_points,recovered_points; };
const Case host_cases[]={{23,2,0,1},{10,1,0,2},{10,5,2,2},{7,0,2,2},{7,4,0,2},{23,3,0,0}};
__global__ void fixture_contacts(const Parameters* parameters,const float* phases,const Case* cases,int* points){
    int i=threadIdx.x;if(i>=12)return;const int ci=i/2;const Case test=cases[ci];Parameters p=*parameters;p.recovered_scoring=i%2?2:1;
    p.routes[0].offset=2;p.routes[0].count=2;
    p.routes[test.route].offset=0;p.routes[test.route].count=2;p.routes[test.route].start_frame=0;p.routes[test.route].end_frame=10000;p.routes[test.route].playback_speed=1;
    FastFrame frames[4]{};
    for(int f=0;f<4;f++)for(int limb=0;limb<6;limb++){frames[f].strike_xyz[limb][0]=limb==test.limb?0.f:100.f;frames[f].strike_radius[limb]=.01f;}
    for(int f=0;f<4;f++)for(int zone=0;zone<3;zone++)frames[f].target_radius[zone]=.01f;
    Arena arena{};auto& fighter=arena.fighter[0];fighter.route=fighter.old_route=test.route;fighter.phase=fighter.old_phase=phases[ci];fighter.old_x=-.1f;fighter.strike_active=1;fighter.move_instance=1;arena.elapsed=10;
    View v{};v.p=&p;v.frames=frames;int hits[2]={},scored[2]={};strike_contacts(v,arena,0,hits,scored);points[i]=scored[0];
}
__global__ void fixture_native_targets(const Parameters* parameters,const FastFrame* frames,float phase,int* points){
    const int i=threadIdx.x;if(i>=2*rek5_native_contact::TargetCount)return;
    const int target=i/2;Parameters p=*parameters;p.recovered_scoring=2;p.primitive_contacts=i%2;p.contact_substeps=4;
    p.routes[0].offset=2;p.routes[0].count=2;
    p.routes[23].offset=0;p.routes[23].count=2;p.routes[23].start_frame=0;p.routes[23].end_frame=10000;p.routes[23].playback_speed=1;
    Arena arena{};auto& fighter=arena.fighter[0];fighter.route=fighter.old_route=23;fighter.phase=fighter.old_phase=phase;
    fighter.old_x=-.1f;fighter.y=fighter.old_y=4.f*target;fighter.strike_active=1;fighter.move_instance=1;arena.elapsed=10;
    View v{};v.p=&p;v.frames=frames;int hits[2]={},scored[2]={};
    strike_contacts(v,arena,0,hits,scored);points[2*i]=scored[0];
    hits[0]=hits[1]=scored[0]=scored[1]=0;
    strike_contacts(v,arena,0,hits,scored);points[2*i+1]=scored[0];
}
}
int main(int argc,char** argv){try{
    const bool gpu=argc==2&&!strcmp(argv[1],"--gpu");if(argc>2||(argc==2&&!gpu))throw std::runtime_error("Use --gpu for one bounded device fixture; no argument runs CPU only");
    const auto* catalog=rek_g1_current_build_strike_catalog();if(!rek_g1_validate_strike_catalog(catalog))throw std::runtime_error("catalog invalid");
    Parameters p{};std::copy(catalog->impact_events,catalog->impact_events+catalog->impact_event_count,p.impact_events);
    for(size_t i=0;i<catalog->count;i++){const auto& e=catalog->entries[i];p.impact_offsets[e.route_id]=e.impact_event_offset;p.impact_counts[e.route_id]=e.impact_event_count;}
    p.recovered_hit_config=rek5_recovered::rek_g1_current_build_hit_detector_config();float phases[6]{};int count=0;
    const int strike_limb[12]={0,0,0,0,1,1,1,1,2,3,4,5};std::copy(strike_limb,strike_limb+12,p.strike_limb);
    for(const auto& test:host_cases){
        const auto* events=p.impact_events+p.impact_offsets[test.route];phases[count]=events[0].impact_time_seconds*50;
        RekG1StrikeIntent intent{};intent.impact_events=events;intent.impact_event_count=p.impact_counts[test.route];intent.clip_cursor_frames=phases[count];intent.clip_fps=50;intent.action_playing=intent.layer_active=1;intent.move_id=1;
        RekG1HitDetectorState state{};auto result=rek5_recovered::score(state,p.recovered_hit_config,intent,0,test.limb,5,10);
        if(result.points!=test.recovered_points)throw std::runtime_error("CPU real-catalog acceptance mismatch");count++;
    }
    printf("{\"event\":\"scoring_v2_catalog_fixture\",\"cpu_cases\":6,\"failures\":0,\"contact_geometry\":\"synthetic\",\"impact_catalog\":\"pinned_native\"}\n");
    if(gpu){
        Parameters* dp=nullptr;float* df=nullptr;Case* dc=nullptr;int* results=nullptr;
        rek5::cuda_check(cudaMalloc(&dp,sizeof(p)));rek5::cuda_check(cudaMalloc(&df,sizeof(phases)));rek5::cuda_check(cudaMalloc(&dc,sizeof(host_cases)));rek5::cuda_check(cudaMalloc(&results,12*sizeof(int)));
        rek5::cuda_check(cudaMemcpy(dp,&p,sizeof(p),cudaMemcpyHostToDevice));rek5::cuda_check(cudaMemcpy(df,phases,sizeof(phases),cudaMemcpyHostToDevice));rek5::cuda_check(cudaMemcpy(dc,host_cases,sizeof(host_cases),cudaMemcpyHostToDevice));
        fixture_contacts<<<1,32>>>(dp,df,dc,results);int actual[12];rek5::cuda_check(cudaMemcpy(actual,results,sizeof(actual),cudaMemcpyDeviceToHost));
        for(int i=0;i<6;i++){const auto& test=host_cases[i];if(actual[i*2]!=test.legacy_points||actual[i*2+1]!=test.recovered_points)throw std::runtime_error("production adapter acceptance mismatch case "+std::to_string(i));}
        printf("{\"event\":\"scoring_v2_production_adapter_fixture\",\"gpu_cases\":12,\"failures\":0,\"routes\":[23,10,10,7,7,23],\"limbs\":[2,1,5,0,4,3],\"v1_points\":[0,0,2,2,0,0],\"v2_points\":[1,2,2,2,2,0],\"authentic_parity\":false}\n");
        cudaFree(dp);cudaFree(df);cudaFree(dc);cudaFree(results);
        FastFrame host_frames[4]{};
        for(auto& frame:host_frames){
            for(int limb=0;limb<6;limb++){frame.strike_xyz[limb][0]=limb==2?0.f:100.f;frame.strike_radius[limb]=.02f;}
            for(int striker=0;striker<12;striker++){
                auto& shape=frame.strike_shapes[striker];shape.kind=rek5_primitive::Sphere;
                shape.center[0]=strike_limb[striker]==2?0.f:100.f;shape.axes[0]=shape.axes[4]=shape.axes[8]=1;shape.size[0]=.01f;
            }
            for(int target=0;target<rek5_native_contact::TargetCount;target++){
                auto& shape=frame.target_shapes[target];shape.kind=rek5_native_contact::TargetKinds[target];
                shape.center[1]=4.f*target;shape.axes[0]=shape.axes[4]=shape.axes[8]=1;
                shape.size[0]=shape.size[1]=.01f;shape.size[2]=shape.kind==rek5_primitive::Box?.01f:0.f;
                frame.target_shape_radius[target]=.02f;
                if(target<3){frame.target_xyz[target][1]=shape.center[1];frame.target_radius[target]=.02f;}
            }
        }
        FastFrame* target_frames=nullptr;int* target_results=nullptr;
        constexpr int result_count=4*rek5_native_contact::TargetCount;
        rek5::cuda_check(cudaMalloc(&dp,sizeof(p)));rek5::cuda_check(cudaMalloc(&target_frames,sizeof(host_frames)));
        rek5::cuda_check(cudaMalloc(&target_results,result_count*sizeof(int)));
        rek5::cuda_check(cudaMemcpy(dp,&p,sizeof(p),cudaMemcpyHostToDevice));
        rek5::cuda_check(cudaMemcpy(target_frames,host_frames,sizeof(host_frames),cudaMemcpyHostToDevice));
        fixture_native_targets<<<1,32>>>(dp,target_frames,phases[0],target_results);rek5::cuda_check(cudaGetLastError());
        int target_actual[result_count];rek5::cuda_check(cudaMemcpy(target_actual,target_results,sizeof(target_actual),cudaMemcpyDeviceToHost));
        for(int target=0;target<rek5_native_contact::TargetCount;target++)for(int primitive=0;primitive<2;primitive++){
            const int index=2*(2*target+primitive),expected=primitive||target<3?1:0;
            if(target_actual[index]!=expected||target_actual[index+1]!=0)throw std::runtime_error("native target isolation/latch mismatch target "+std::to_string(target));
        }
        std::printf("{\"event\":\"native_scoring_targets_production_adapter_fixture\",\"passed\":true,\"gpu_cases\":%d,\"target_contract\":\"%s\",\"primitive_target_count\":%d,\"legacy_target_count\":3,\"isolated_hip_cases\":6,\"contact_geometry\":\"synthetic\",\"persistent_contact_rescores\":0,\"authentic_parity\":false}\n",result_count,rek5_native_contact::TargetContract,rek5_native_contact::TargetCount);
        cudaFree(dp);cudaFree(target_frames);cudaFree(target_results);
    }
    return 0;
}catch(const std::exception& e){fprintf(stderr,"Scoring v2 test failed: %s\n",e.what());return 2;}}
