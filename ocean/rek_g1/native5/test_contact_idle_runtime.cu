// Compare the optimized production function with the full-sampling function
// copied verbatim from commit 2981e7c8, except its name. No learner or game.
#include "fast_runtime.cu"

__device__ void reference_geom_pair_contacts(const View& v,Arena& a,int side,int* hits,int* points){
    using namespace rek5_primitive;
    const Parameters& p=*v.p;auto& f=a.fighter[side];const auto& enemy=a.fighter[side^1];
    const auto& now=v.frames[frame_index(p,f,false)];const auto& before=v.frames[frame_index(p,f,true)];
    const auto& target=v.frames[frame_index(p,enemy,false)];const auto& old_target=v.frames[frame_index(p,enemy,true)];
    for(int limb=0;limb<6;limb++){
        float tip[3],old_tip[3];point(f,now.strike_xyz[limb],false,tip);point(f,before.strike_xyz[limb],true,old_tip);
        bool entered=false;float max_relative_speed=0;
        for(int zone=0;zone<rek_contact_entry::Targets;zone++){
            float dst[3],old_dst[3],from[3],to[3];
            point(enemy,target.target_shapes[zone].center,false,dst);point(enemy,old_target.target_shapes[zone].center,true,old_dst);
            for(int k=0;k<3;k++){from[k]=old_tip[k]-old_dst[k];to[k]=tip[k]-dst[k];}
            const float radius=fmaxf(before.strike_radius[limb],now.strike_radius[limb])+
                fmaxf(old_target.target_shape_radius[zone],target.target_shape_radius[zone]);
            const bool broad=sweep_distance2(from,to)<=radius*radius;
            bool intersects=false;
            for(int i=0;i<rek_contact_entry::Strikers;i++)if(p.strike_limb[i]==limb){
                const int pair=i*rek_contact_entry::Targets+zone;
                if(!broad){rek_contact_entry::update(f.contact_pairs,pair,false);continue;}
                const auto result=rek_contact_entry::sample(f.contact_pairs,pair,
                    world_shape(before.strike_shapes[i],f.old_x,f.old_y,f.old_yaw),
                    world_shape(now.strike_shapes[i],f.x,f.y,f.yaw),
                    world_shape(old_target.target_shapes[zone],enemy.old_x,enemy.old_y,enemy.old_yaw),
                    world_shape(target.target_shapes[zone],enemy.x,enemy.y,enemy.yaw),p.contact_substeps);
                entered=entered||result.entered;intersects=intersects||result.overlap_after_start;
            }
            // Preserve the existing per-limb maximum sphere-center velocity
            // proxy, including its aggregation across intersecting targets.
            if(intersects){float d2=0;for(int k=0;k<3;k++){float d=to[k]-from[k];d2+=d*d;}max_relative_speed=fmaxf(max_relative_speed,sqrtf(d2)/DT);}
        }
        // History above advances even without intent and never resets at move
        // start. Native apex, body cooldown and invocation dedup stay unchanged.
        if(!f.strike_active||!entered)continue;
        const auto& route=p.routes[f.route];RekG1StrikeIntent intent{};
        intent.impact_events=p.impact_events+p.impact_offsets[f.route];intent.impact_event_count=p.impact_counts[f.route];
        intent.clip_cursor_frames=clampf(route.start_frame+f.phase*route.playback_speed,float(route.start_frame),float(route.end_frame));
        intent.clip_fps=50;intent.move_id=f.move_instance;intent.action_playing=intent.layer_active=1;
        const auto result=rek5_recovered::score(a.recovered_hits,p.recovered_hit_config,intent,side,limb,max_relative_speed,a.elapsed);
        if(result.points){hits[side]++;points[side]+=result.points;auto& other=a.fighter[side^1];other.last_hit_valid=1;other.last_hit_age=0;other.last_hit_speed=max_relative_speed;}
    }
}

struct IdleResults {unsigned checks,failures,first_failure,steps,idle_calls,active_calls,points,resets,invalid_apex_calls;};
__device__ void verify(IdleResults* out,bool value,unsigned label){
    out->checks++;if(!value){if(!out->failures)out->first_failure=label;out->failures++;}
}
__device__ void compare(IdleResults* out,const Arena& reference,const Arena& optimized,unsigned label){
    for(int side=0;side<2;side++){
        const auto& r=reference.fighter[side];const auto& o=optimized.fighter[side];
        for(int word=0;word<2;word++)verify(out,r.contact_pairs.words[word]==o.contact_pairs.words[word],label+word);
        verify(out,r.points==o.points&&r.last_hit_valid==o.last_hit_valid,label+2);
        verify(out,__float_as_uint(r.last_hit_age)==__float_as_uint(o.last_hit_age)&&
            __float_as_uint(r.last_hit_speed)==__float_as_uint(o.last_hit_speed),label+3);
        for(int limb=0;limb<6;limb++){
            verify(out,reference.recovered_hits.cooldown_seen[side][limb]==optimized.recovered_hits.cooldown_seen[side][limb],label+4);
            verify(out,__float_as_uint(reference.recovered_hits.last_score_time_seconds[side][limb])==
                __float_as_uint(optimized.recovered_hits.last_score_time_seconds[side][limb]),label+5);
        }
        verify(out,reference.recovered_hits.scored_move_id[side]==optimized.recovered_hits.scored_move_id[side]&&
            reference.recovered_hits.scored_apex_mask[side]==optimized.recovered_hits.scored_apex_mask[side]&&
            reference.recovered_hits.scored_move_seen[side]==optimized.recovered_hits.scored_move_seen[side],label+6);
    }
    verify(out,reference.contact_pairs_initialized==optimized.contact_pairs_initialized,label+7);
}
__device__ void make_shape(rek5_primitive::Shape& s,int kind,float x,float y,float turn){
    s={};s.kind=kind;s.center[0]=x;s.center[1]=y;s.size[0]=.2f;s.size[1]=.17f;s.size[2]=.12f;
    float q[4]={cosf(turn*.5f),0,0,sinf(turn*.5f)};rek5_primitive::quaternion_matrix(q,s.axes);
}
__global__ void exercise_idle(Parameters* params,FastFrame* frames,Arena* states,IdleResults* out){
    if(threadIdx.x||blockIdx.x)return;
    auto& p=*params;
    p.contact_entry=rek_contact_entry::Mode::GeomPair;p.primitive_contacts=1;p.recovered_scoring=2;p.contact_substeps=8;
    p.recovered_hit_config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    const RekG1AimLimb aims[4]={REK_G1_AIM_LIMB_LEFT_LOWER_BODY,REK_G1_AIM_LIMB_RIGHT_LOWER_BODY,REK_G1_AIM_LIMB_LEFT_UPPER_BODY,REK_G1_AIM_LIMB_RIGHT_UPPER_BODY};
    for(int i=0;i<4;i++)p.impact_events[i]={.02f,.1f,.1f,1.f,aims[i]};
    for(int route_index=0;route_index<2;route_index++){
        const int route=route_index?23:0;
        p.impact_counts[route]=4;p.routes[route].count=2;p.routes[route].offset=route?2:0;
        p.routes[route].end_frame=100;p.routes[route].playback_speed=1;
    }
    const int limbs[12]={0,0,0,0,1,1,1,1,2,3,4,5};
    for(int i=0;i<12;i++)p.strike_limb[i]=limbs[i];
    for(int frame=0;frame<4;frame++){
        for(int limb=0;limb<6;limb++){
            frames[frame].strike_xyz[limb][0]=.025f*limb;
            frames[frame].strike_radius[limb]=.5f;
        }
        for(int i=0;i<12;i++)make_shape(frames[frame].strike_shapes[i],i%3,.015f*i,.01f*frame,.07f*i*frame);
        for(int zone=0;zone<9;zone++){
            make_shape(frames[frame].target_shapes[zone],zone%3,.015f*zone,.01f*frame,.05f*zone*frame);
            frames[frame].target_shape_radius[zone]=.5f;
        }
    }
    View v{};v.p=params;v.frames=frames;
    Arena& reference=states[0];Arena& optimized=states[1];
    const float positions[16]={2,2,-1,1,0,0,2,0,0,2,-1,1,0,0,2,0};
    const bool active[16]={false,false,false,false,true,false,false,true,true,false,false,true,false,true,false,true};
    for(unsigned tick=0;tick<256;tick++){
        if(tick%32==0){
            RekNative5RoundResult rr{},ro{};
            round_reset(p,reference,rr,true,0);round_reset(p,optimized,ro,true,0);
            seed_contact_pairs(v,reference);seed_contact_pairs(v,optimized);out->resets++;
        }else if(tick%19==0){
            pose_reset(p,reference);pose_reset(p,optimized);
            seed_contact_pairs(v,reference);seed_contact_pairs(v,optimized);out->resets++;
        }
        for(int state=0;state<2;state++){
            auto& a=states[state];a.elapsed=float(tick)*.31f;
            for(int side=0;side<2;side++){
                auto& f=a.fighter[side];f.old_x=f.x;f.old_y=f.y;f.old_yaw=f.yaw;f.old_phase=f.phase;f.old_route=f.route;
                f.x=side?0:positions[tick%16];f.y=0;f.yaw=(int(tick%5)-2)*.1f*(side?-1:1);
                f.route=(tick/7)%2?23:0;
                f.phase=(tick%16==4||tick%16==7||tick%16==13)?50.f:1.f;
                f.move_instance=int(tick/2)+1;
                f.strike_active=active[tick%16]&&(side==0||tick%3==0);
            }
        }
        int rh[2]={},rp[2]={},oh[2]={},op[2]={};
        for(int side=0;side<2;side++){
            if(reference.fighter[side].strike_active)out->active_calls++;else out->idle_calls++;
            if(reference.fighter[side].strike_active&&reference.fighter[side].phase==50.f)out->invalid_apex_calls++;
            reference_geom_pair_contacts(v,reference,side,rh,rp);
            geom_pair_contacts(v,optimized,side,oh,op);
            verify(out,rh[side]==oh[side]&&rp[side]==op[side],100+tick);
            reference.fighter[side].points+=rp[side];optimized.fighter[side].points+=op[side];out->points+=rp[side];
            compare(out,reference,optimized,1000+tick*16);
        }
        out->steps++;
    }
    // Production attack acceptance must retain the histories established idle.
    p.routes[7].count=2;p.routes[7].move=0;p.action_to_route[16]=7;p.durations[0]=2;p.settle_speed=.01f;
    for(int i=0;i<2;i++){
        auto& a=states[i];a.fighter[0].attack_duration=0;a.fighter[0].held=1;a.fighter[0].vx=a.fighter[0].vy=0;
        a.fighter[0].contact_pairs.words[0]=0x5a5a5a5a5a5a5a5aull;a.fighter[0].contact_pairs.words[1]=0xa55a500055555555ull;
        advance_fighter(p,a.fighter[0],16,a);
        verify(out,a.fighter[0].contact_pairs.words[0]==0x5a5a5a5a5a5a5a5aull&&
            a.fighter[0].contact_pairs.words[1]==0xa55a500055555555ull,9000+i);
    }
    compare(out,reference,optimized,9100);
    verify(out,out->idle_calls>0&&out->active_calls>0&&out->points>0&&out->resets>0&&out->invalid_apex_calls>0,9999);
}
int main(){try{
    Parameters* params=nullptr;FastFrame* frames=nullptr;Arena* states=nullptr;IdleResults* out=nullptr;
    rek5::cuda_check(cudaMallocManaged(&params,sizeof(Parameters)));rek5::cuda_check(cudaMemset(params,0,sizeof(Parameters)));
    rek5::cuda_check(cudaMallocManaged(&frames,4*sizeof(FastFrame)));rek5::cuda_check(cudaMemset(frames,0,4*sizeof(FastFrame)));
    rek5::cuda_check(cudaMallocManaged(&states,2*sizeof(Arena)));rek5::cuda_check(cudaMemset(states,0,2*sizeof(Arena)));
    rek5::cuda_check(cudaMallocManaged(&out,sizeof(IdleResults)));rek5::cuda_check(cudaMemset(out,0,sizeof(IdleResults)));
    exercise_idle<<<1,1>>>(params,frames,states,out);rek5::cuda_check(cudaDeviceSynchronize());
    std::printf("{\"test\":\"contact_unscored_production_equivalence\",\"checks\":%u,\"failures\":%u,\"first_failure\":%u,\"steps\":%u,\"idle_calls\":%u,\"active_calls\":%u,\"points\":%u,\"resets\":%u,\"invalid_apex_calls\":%u,\"passed\":%s}\n",
        out->checks,out->failures,out->first_failure,out->steps,out->idle_calls,out->active_calls,out->points,out->resets,out->invalid_apex_calls,out->failures?"false":"true");
    const bool failed=out->failures;
    rek5::cuda_check(cudaFree(params));rek5::cuda_check(cudaFree(frames));rek5::cuda_check(cudaFree(states));rek5::cuda_check(cudaFree(out));return failed?2:0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
