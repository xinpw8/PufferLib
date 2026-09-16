// Diagnostic only. The included production runtime is unchanged. GPU observers
// read its states and never alter simulation, actions, observations, or rewards.
#include "../fast_runtime.cu"
#include "../native_policy.h"
#include "../../../../vendor/cJSON.h"
extern "C" {
#include "../../g1_strike_catalog.h"
}
#include <chrono>
#include <fstream>
#include <tuple>

namespace audit {
constexpr int N=64,TICKS=1000,CAPACITY=100000;
struct Event {
    int arena,side,tick,route,limb,invocation;
    float phase,old_phase,absolute_speed,relative_min,relative_max,gap,bearing;
};
struct Counters { unsigned count,errors; unsigned long long points[2]; };
struct Before { Arena arena; int terminal; };
__global__ void capture(View v,Before* prior){
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a<v.arenas)prior[a]={v.state[a],v.rounds[a].terminal};
}
__global__ void observe(View v,const Before* prior,int* invocations,Event* events,Counters* counters){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=v.arenas)return;
    const Arena& a=v.state[index];const Parameters& p=*v.p;
    // This bounded run is exactly one round. A reset would invalidate history.
    if(prior[index].terminal){atomicOr(&counters->errors,1u);return;}
    for(int side=0;side<2;side++){
        const Fighter& f=a.fighter[side];const Fighter& enemy=a.fighter[side^1];
        const Fighter& old=prior[index].arena.fighter[side];
        bool started=!attacking(old)&&f.strike_active;
        if(started)invocations[index*2+side]++;
        int reconstructed=0;
        if(f.strike_active&&f.route!=23){
            const FastFrame& now=v.frames[frame_index(p,f,false)];
            const FastFrame& before=v.frames[frame_index(p,f,true)];
            const FastFrame& target=v.frames[frame_index(p,enemy,false)];
            const FastFrame& old_target=v.frames[frame_index(p,enemy,true)];
            for(int limb=0;limb<6;limb++){
                if(!limb_enabled(f.route,limb))continue;
                float tip[3],old_tip[3];point(f,now.strike_xyz[limb],false,tip);point(f,before.strike_xyz[limb],true,old_tip);
                float dx=tip[0]-old_tip[0],dy=tip[1]-old_tip[1],dz=tip[2]-old_tip[2];
                float speed=sqrtf(dx*dx+dy*dy+dz*dz)/DT;
                bool touch=false;float relative_min=INFINITY,relative_max=0;
                for(int zone=0;zone<3;zone++){
                    float dst[3],old_dst[3],from[3],to[3];
                    point(enemy,target.target_xyz[zone],false,dst);point(enemy,old_target.target_xyz[zone],true,old_dst);
                    for(int k=0;k<3;k++){from[k]=old_tip[k]-old_dst[k];to[k]=tip[k]-dst[k];}
                    float radius=now.strike_radius[limb]+target.target_radius[zone];
                    bool contact=sweep_distance2(from,to)<=radius*radius;touch=touch||contact;
                    if(contact){float norm=0;for(int k=0;k<3;k++){float d=to[k]-from[k];norm+=d*d;}float rel=sqrtf(norm)/DT;relative_min=fminf(relative_min,rel);relative_max=fmaxf(relative_max,rel);}
                }
                unsigned latched=started?0:old.contact_latched;
                if(touch&&!(latched&(1u<<limb))&&max(0,old.cooldown[limb]-1)==0&&speed>=p.hit_speed){
                    reconstructed++;unsigned slot=atomicAdd(&counters->count,1u);
                    if(slot>=CAPACITY){atomicOr(&counters->errors,2u);continue;}
                    float dx=enemy.x-f.x,dy=enemy.y-f.y;
                    events[slot]={index,side,a.tick,f.route,limb,invocations[index*2+side],f.phase,f.old_phase,speed,relative_min,relative_max,hypotf(dx,dy),angle(atan2f(dy,dx)-f.yaw)};
                }
            }
        }
        if(reconstructed!=a.delta[side])atomicOr(&counters->errors,4u);
        atomicAdd(&counters->points[side],static_cast<unsigned long long>(a.delta[side]));
    }
}
void check(bool good,const std::string& message){if(!good)throw std::runtime_error(message);}
void ck(cudaError_t r){check(r==cudaSuccess,cudaGetErrorString(r));}
void rt(int r){check(r==0,rek_native5_error());}
void po(int r){check(r==0,rek_native_policy_error());}
std::string read(const std::string& path){std::ifstream f(path);check(bool(f),"missing input "+path);return std::string((std::istreambuf_iterator<char>(f)),{});}
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
Json parse(const std::string& text){Json p(cJSON_Parse(text.c_str()),cJSON_Delete);check(bool(p),"JSON parse failed");return p;}
const cJSON* field(const cJSON* p,const char* key){auto* v=cJSON_GetObjectItemCaseSensitive(p,key);check(v,std::string("missing ")+key);return v;}
std::string str(const cJSON* p){check(cJSON_IsString(p)&&p->valuestring,"expected string");return p->valuestring;}
template<class T>T* alloc(std::vector<void*>& owned,size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));owned.push_back(p);return p;}

void compatibility(const std::string& asset_path,const FastAssets& assets){
    auto manifest=parse(read(asset_path+"/semantic_duel_assets_manifest.json"));
    auto* catalog=rek_g1_current_build_strike_catalog();check(rek_g1_validate_strike_catalog(catalog),"invalid recovered catalog");
    check(str(field(manifest.get(),"build_fingerprint"))==catalog->build_fingerprint,"build fingerprint mismatch");
    check(str(field(field(manifest.get(),"source_contracts"),"route_contract_file_sha256"))==catalog->source_sha256,"strike source contract mismatch");
    auto* table=rek_g1_native_static_motion_routes();auto* routes=field(manifest.get(),"routes");
    check(cJSON_GetArraySize(routes)==24,"route count mismatch");
    const char* float_keys[]={"playback_speed","blend_in_seconds","blend_out_seconds","yaw_blend"};
    for(int i=0;i<24;i++){
        const auto& expected=table->routes[i];const auto& baked=assets.routes[i];auto* r=cJSON_GetArrayItem(routes,i);auto* config=field(r,"config");
        check(int(field(r,"route_id")->valuedouble)==i&&int(field(r,"npz_path_id")->valuedouble)==expected.npz_path_id,"route identity mismatch");
        const float floats[]={expected.playback_speed,expected.blend_in_seconds,expected.blend_out_seconds,expected.yaw_blend};
        for(int k=0;k<4;k++)check(float(field(config,float_keys[k])->valuedouble)==floats[k],"route float mismatch");
        check(int(field(config,"start_frame")->valuedouble)==expected.start_frame&&int(field(config,"end_frame")->valuedouble)==expected.end_frame&&int(field(config,"mirror")->valuedouble)==expected.mirror&&int(field(config,"loop")->valuedouble)==expected.loop,"route flags mismatch");
        if(i>=7)check(int(field(r,"runtime_move_index")->valuedouble)==expected.runtime_move_index&&baked.move==expected.runtime_move_index,"move mapping mismatch");
        bool found=false;auto* clips=field(manifest.get(),"clips");for(auto* c=clips->child;c;c=c->next)if(int(field(c,"npz_path_id")->valuedouble)==expected.npz_path_id){found=true;check(float(field(c,"fps")->valuedouble)==expected.asset_fps&&int(field(c,"frames")->valuedouble)==int(expected.asset_frames),"clip timing mismatch");}
        check(found,"missing clip");
    }
    printf("{\"event\":\"compatibility\",\"asset_build_fingerprint\":\"%s\",\"strike_source_sha256\":\"%s\",\"routes_verified\":24,\"asset_hash_checks\":true,\"authentic_server_build_verified\":false}\n",catalog->build_fingerprint,catalog->source_sha256);
}

// Partial acceptance audit only. No unmeasured rigid-body contact or upright
// fields are synthesized. Relative speed is explicitly a sphere-center proxy.
struct Summary {long long total=0,absolute_slow=0,proxy_all_slow=0,proxy_some_slow=0,apex_missing=0,apex_missing_both=0,cooldown_only=0,apex_dedup=0,sequential_speed=0,sequential_apex=0,sequential_cooldown=0,sequential_dedup=0,sequential_kept=0;};
struct GateState {float cooldown[6],combined[6];int invocation=-1;unsigned apex_mask=0,combined_mask=0;GateState(){std::fill(cooldown,cooldown+6,-100.f);std::fill(combined,combined+6,-100.f);}};
void summarize(const char* opponent,std::vector<Event> events,const FastAssets& assets,const Counters& counters){
    std::sort(events.begin(),events.end(),[](const Event& a,const Event& b){return std::tie(a.arena,a.side,a.tick,a.limb)<std::tie(b.arena,b.side,b.tick,b.limb);});
    auto* catalog=rek_g1_current_build_strike_catalog();auto config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    Summary totals[2],moves[2][17];GateState state[N*2];
    for(const Event& e:events){
        const auto& route=assets.routes[e.route];auto* entry=rek_g1_strike_catalog_entry_by_route(catalog,RekG1NativeRouteId(e.route));check(entry,"missing event route");
        RekG1StrikeIntent intent{};intent.impact_events=catalog->impact_events+entry->impact_event_offset;intent.impact_event_count=entry->impact_event_count;
        // Bake source cursor formula, after validated source FPS/config match.
        intent.clip_cursor_frames=std::clamp(route.start_frame+e.phase*route.playback_speed,float(route.start_frame),float(route.end_frame));intent.clip_fps=50;intent.move_id=e.invocation;intent.action_playing=intent.layer_active=1;
        auto part=e.limb<2?REK_G1_BODY_PART_FOOT:e.limb<4?REK_G1_BODY_PART_HAND:REK_G1_BODY_PART_SHIN;
        auto side=(e.limb&1)?REK_G1_HAND_RIGHT:REK_G1_HAND_LEFT;int apex=-1,old_apex=-1;float ramp=0,old_ramp=0;
        bool active=rek5_recovered::embedded_strike_intent_apex(&intent,part,side,config.apex_min_ramp,&apex,&ramp);
        intent.clip_cursor_frames=std::clamp(route.start_frame+e.old_phase*route.playback_speed,float(route.start_frame),float(route.end_frame));
        bool old_active=rek5_recovered::embedded_strike_intent_apex(&intent,part,side,config.apex_min_ramp,&old_apex,&old_ramp);
        GateState& g=state[e.arena*2+e.side];if(g.invocation!=e.invocation){g.invocation=e.invocation;g.apex_mask=g.combined_mask=0;}
        unsigned bit=active?1u<<std::min(apex,30):0;float seconds=float(e.tick)*DT;
        bool cooldown=seconds-g.cooldown[e.limb]<config.per_body_cooldown_seconds;if(!cooldown)g.cooldown[e.limb]=seconds;
        bool duplicate=active&&(g.apex_mask&bit);if(active&&!duplicate)g.apex_mask|=bit;
        int sequential;
        if(e.relative_max<config.speed_threshold_mps)sequential=0;
        else if(!active)sequential=1;
        else if(seconds-g.combined[e.limb]<config.per_body_cooldown_seconds)sequential=2;
        else if(g.combined_mask&bit)sequential=3;
        else{sequential=4;g.combined[e.limb]=seconds;g.combined_mask|=bit;}
        for(Summary* s:{&totals[e.side],&moves[e.side][route.move]}){
            s->total++;s->absolute_slow+=e.absolute_speed<config.speed_threshold_mps;s->proxy_all_slow+=e.relative_max<config.speed_threshold_mps;s->proxy_some_slow+=e.relative_min<config.speed_threshold_mps;
            s->apex_missing+=!active;s->apex_missing_both+=!active&&!old_active;s->cooldown_only+=cooldown;s->apex_dedup+=duplicate;
            s->sequential_speed+=sequential==0;s->sequential_apex+=sequential==1;s->sequential_cooldown+=sequential==2;s->sequential_dedup+=sequential==3;s->sequential_kept+=sequential==4;
        }
    }
    for(int side=0;side<2;side++)for(int move=-1;move<17;move++){
        const Summary& s=move<0?totals[side]:moves[side][move];if(!s.total)continue;
        printf("{\"event\":\"contact_gate_audit\",\"opponent\":\"%s\",\"fighter\":%d,\"move\":%d,\"compact_points\":%lld,\"absolute_striker_speed_below_1_75\":%lld,\"sphere_relative_all_touching_targets_below_1_75\":%lld,\"sphere_relative_any_touching_target_below_1_75\":%lld,\"apex_reject_post_cursor\":%lld,\"apex_reject_both_endpoint_cursors\":%lld,\"cooldown_0_3_only_reject\":%lld,\"apex_then_dedup_reject\":%lld,\"sequential_proxy_speed_reject\":%lld,\"sequential_apex_reject\":%lld,\"sequential_cooldown_reject\":%lld,\"sequential_dedup_reject\":%lld,\"sequential_kept\":%lld}\n",opponent,side,move,s.total,s.absolute_slow,s.proxy_all_slow,s.proxy_some_slow,s.apex_missing,s.apex_missing_both,s.cooldown_only,s.apex_dedup,s.sequential_speed,s.sequential_apex,s.sequential_cooldown,s.sequential_dedup,s.sequential_kept);
    }
    check(totals[0].total==static_cast<long long>(counters.points[0])&&totals[1].total==static_cast<long long>(counters.points[1]),"event totals mismatch");
}
}

int main(int argc,char** argv){try{
    using namespace audit;check(argc==4,"Usage: contact-audit RUNTIME_JSON CHECKPOINT SHA256");
    auto json=parse(read(argv[1]));check(str(field(json.get(),"backend"))=="semantic_cuda","backend mismatch");
    std::string model=str(field(json.get(),"model_path")),assets_path=str(field(json.get(),"assets_path")),features=str(field(json.get(),"motion_features_path"));
    RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=model.c_str();cfg.assets_path=assets_path.c_str();cfg.motion_features_path=features.c_str();cfg.arenas=N;cfg.seed=10001;cfg.round_seconds=20;cfg.locomotion_segment_ticks=1;
    const auto* durations=field(json.get(),"move_duration_ticks");check(cJSON_GetArraySize(durations)==17,"duration count");for(int i=0;i<17;i++)cfg.move_duration_ticks[i]=uint32_t(cJSON_GetArrayItem(durations,i)->valuedouble);
    FastAssets assets=load_fast_assets(cfg);compatibility(assets_path,assets);
    setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);setenv("REK_FAST_SCORING","v4_spheres",1);setenv("REK_FAST_RANDOM_RESETS","1",1);setenv("REK_FAST_SHAPING_WEIGHT","0",1);
    for(const char* key:{"REK_FAST_MOVE_SPEED","REK_FAST_YAW_SPEED","REK_FAST_BODY_RADIUS","REK_FAST_HIT_SPEED","REK_FAST_RESET_GAP_MIN","REK_FAST_RESET_GAP_MAX","REK_FAST_RESET_HEADING_SPREAD_RAD"})unsetenv(key);
    check(!cJSON_GetObjectItemCaseSensitive(json.get(),"fast"),"diagnostic requires verified default compact parameters");
    for(const char* mode:{"neutral","scripted"}){
        setenv("REK_FAST_OPPONENT_MODE",mode,1);std::vector<void*> owned;cudaStream_t stream;ck(cudaStreamCreate(&stream));
        RekNative5Buffers b{};b.observations=alloc<float>(owned,N*223);b.actions=alloc<float>(owned,N);b.rewards=alloc<float>(owned,N);b.terminals=alloc<float>(owned,N);b.logs=alloc<RekNative5Log>(owned,N);b.log_stride_bytes=sizeof(RekNative5Log);
        auto* runtime=rek_native5_create(&cfg,&b,stream);check(runtime,rek_native5_error());
        auto* prior=alloc<Before>(owned,N);auto* invocations=alloc<int>(owned,N*2);auto* events=alloc<Event>(owned,CAPACITY);auto* counters=alloc<Counters>(owned,1);
        RekNative5DeviceView v{};rt(rek_native5_get_device_view(runtime,&v));auto* masks=alloc<uint8_t>(owned,N*33);rt(rek_native5_bind_action_mask(runtime,masks,stream));
        RekNativePolicyConfig pc{};pc.abi_version=REK_NATIVE_POLICY_ABI;pc.checkpoint_path=argv[2];pc.expected_sha256=argv[3];pc.hidden_size=256;pc.num_layers=2;pc.batch=N;pc.precision=REK_NATIVE_POLICY_BF16;pc.seed=10001;
        auto* policy=rek_native_policy_create(&pc,stream);check(policy,rek_native_policy_error());
        constexpr int CHUNK=20;auto start=std::chrono::steady_clock::now();ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int t=0;t<CHUNK;t++){
            po(rek_native_policy_step_rows(policy,b.observations,masks,b.terminals,b.actions,0,1,0,stream));
            capture<<<1,128,0,stream>>>(runtime->view,prior);rt(rek_native5_step(runtime,stream));observe<<<1,128,0,stream>>>(runtime->view,prior,invocations,events,counters);
        }
        cudaGraph_t graph;ck(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t executable;ck(cudaGraphInstantiate(&executable,graph,0));
        for(int t=0;t<TICKS;t+=CHUNK)ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));
        rt(rek_native5_check_status(runtime,stream));po(rek_native_policy_check_status(policy,stream));Counters count{};ck(cudaMemcpy(&count,counters,sizeof(count),cudaMemcpyDeviceToHost));check(!count.errors&&count.count<CAPACITY,"contact reconstruction failure bits="+std::to_string(count.errors));
        std::vector<Event> host(count.count);ck(cudaMemcpy(host.data(),events,host.size()*sizeof(Event),cudaMemcpyDeviceToHost));
        std::vector<RekNative5RoundResult> rounds(N);ck(cudaMemcpy(rounds.data(),v.rounds,sizeof(rounds[0])*N,cudaMemcpyDeviceToHost));unsigned long long sums[2]={};int wins=0,losses=0,draws=0;
        for(const auto& r:rounds){check(r.terminal&&r.completed_rounds==1,"expected exactly one complete round");for(int s=0;s<2;s++)sums[s]+=r.points[s];wins+=r.round_winner==0;losses+=r.round_winner==1;draws+=r.round_winner<0;}
        check(sums[0]==count.points[0]&&sums[1]==count.points[1],"terminal reward accounting mismatch");
        printf("{\"event\":\"run\",\"opponent\":\"%s\",\"checkpoint_sha256\":\"%s\",\"arenas\":%d,\"ticks\":%d,\"policy_side\":0,\"seed\":10001,\"sampled\":true,\"shaping_weight\":0,\"contact_reconstruction_mismatches\":0,\"terminal_point_accounting_mismatches\":0,\"wins\":%d,\"losses\":%d,\"draws\":%d,\"execution_wall_seconds\":%.6f}\n",mode,rek_native_policy_sha256(policy),N,TICKS,wins,losses,draws,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        summarize(mode,std::move(host),assets,count);rek_native_policy_destroy(policy);rt(rek_native5_close(runtime));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));for(void* p:owned)ck(cudaFree(p));ck(cudaStreamDestroy(stream));
    }
    return 0;
}catch(const std::exception& e){fprintf(stderr,"contact audit failed: %s\n",e.what());return 2;}}
