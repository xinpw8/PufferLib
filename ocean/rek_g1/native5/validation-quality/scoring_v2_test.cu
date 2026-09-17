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
}
int main(int argc,char** argv){try{
    const bool gpu=argc==2&&!strcmp(argv[1],"--gpu");if(argc>2||(argc==2&&!gpu))throw std::runtime_error("Use --gpu for one bounded device fixture; no argument runs CPU only");
    const auto* catalog=rek_g1_current_build_strike_catalog();if(!rek_g1_validate_strike_catalog(catalog))throw std::runtime_error("catalog invalid");
    Parameters p{};std::copy(catalog->impact_events,catalog->impact_events+catalog->impact_event_count,p.impact_events);
    for(size_t i=0;i<catalog->count;i++){const auto& e=catalog->entries[i];p.impact_offsets[e.route_id]=e.impact_event_offset;p.impact_counts[e.route_id]=e.impact_event_count;}
    p.recovered_hit_config=rek5_recovered::rek_g1_current_build_hit_detector_config();float phases[6]{};int count=0;
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
    }
    return 0;
}catch(const std::exception& e){fprintf(stderr,"Scoring v2 test failed: %s\n",e.what());return 2;}}
