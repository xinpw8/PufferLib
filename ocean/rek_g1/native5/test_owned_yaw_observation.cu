// Synthetic software-contract test of the exact compact runtime functions.
// No authentic game, training, or new motion rule is exercised.
#include "fast_runtime.cu"

struct YawResult {float legacy[223],v2[223];int equal_state;};
__global__ void exercise_owned_yaw(YawResult* out){
    if(threadIdx.x||blockIdx.x)return;
    Parameters legacy{},v2{};legacy.settle_speed=.01f;legacy.round_seconds=120;
    for(int i=0;i<24;i++){legacy.routes[i].count=1;legacy.routes[i].loop=1;}
    legacy.action_to_route[23]=14;legacy.routes[14].move=3;legacy.durations[3]=45;
    v2=legacy;v2.owned_yaw_observation=1;
    FastFrame frame{};frame.root_z=1;frame.root_wxyz[0]=1;
    Arena a{},b{};a.fighter[0].held=b.fighter[0].held=1;
    RekNative5RoundResult round{};round.phase=2;
    View av{},bv{};av.arenas=bv.arenas=1;av.p=&legacy;bv.p=&v2;
    av.state=&a;bv.state=&b;av.frames=bv.frames=&frame;av.rounds=bv.rounds=&round;
    const int actions[]={23,6,0,7,0,1,0};
    for(int t=0;t<7;t++){
        advance_fighter(legacy,a.fighter[0],actions[t],a);
        advance_fighter(v2,b.fighter[0],actions[t],b);
        bool equal=true;const auto* x=reinterpret_cast<const unsigned char*>(&a);
        const auto* y=reinterpret_cast<const unsigned char*>(&b);
        for(unsigned i=0;i<sizeof(Arena);i++)equal=equal&&x[i]==y[i];
        out[t].equal_state=equal;
        for(int i=0;i<223;i++){out[t].legacy[i]=raw_value(av,0,0,i);out[t].v2[i]=raw_value(bv,0,0,i);}
    }
    // Terminal rows and out-of-scope native Bot controller rows carry no owned
    // policy intent. They are never silently presented as owned pending yaw.
    b.fighter[0].held=6;round.terminal=1;out[7].v2[187]=raw_value(bv,0,0,187);
    round.terminal=0;b.fighter[0].bot_controlled=1;out[7].v2[188]=raw_value(bv,0,0,187);
    b.fighter[0].bot_controlled=0;b.fighter[0].attack_duration=0;out[7].v2[189]=raw_value(bv,0,0,187);
}
int main(){try{
    unsigned checks=0;auto check=[&](bool ok,const char* message){checks++;if(!ok)throw std::runtime_error(message);};
    check(!rek_owned_yaw::enabled(nullptr),"legacy default");check(rek_owned_yaw::enabled(rek_owned_yaw::kSchema),"explicit v2");
    check(!rek_owned_yaw::valid_desired(0)&&!rek_owned_yaw::valid_desired(-1)&&!rek_owned_yaw::valid_desired(16),"unknown source category");
    for(int i=1;i<=15;i++)check(rek_owned_yaw::valid_desired(i),"valid retained category");
    YawResult* result=nullptr;rek5::cuda_check(cudaMallocManaged(&result,8*sizeof(YawResult)));
    rek5::cuda_check(cudaMemset(result,0,8*sizeof(YawResult)));
    exercise_owned_yaw<<<1,1>>>(result);rek5::cuda_check(cudaDeviceSynchronize());
    const float expected[]={0,1,1,-1,-1,0,0};
    for(int t=0;t<7;t++){
        check(result[t].equal_state,"observation opt-in changed dynamics");
        check(result[t].legacy[187]==0,"legacy column changed");
        check(result[t].v2[187]==expected[t],"hold/release pending yaw mismatch");
        check(result[t].v2[178]==0&&result[t].v2[182]==1,"legacy busy fields changed");
        for(int i=0;i<223;i++)if(i!=187)check(result[t].legacy[i]==result[t].v2[i],"another feature changed");
    }
    for(int i=187;i<=189;i++)check(result[7].v2[i]==0,"terminal/nonowned/nonbusy intent");
    rek5::cuda_check(cudaFree(result));
    printf("{\"test\":\"owned_yaw_v2_runtime\",\"checks\":%u,\"passed\":true,\"synthetic_fixture\":true,\"physics_changed\":false}\n",checks);
    return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 2;}}
