// Exact compact advance_fighter integration, using a small synthetic fixture.
// The existing ABI fixture separately compares old/new default real-asset bytes.
#include "fast_runtime.cu"

constexpr int kTicks=120;
struct Result {Fighter legacy,current;int bot_equal;};
__global__ void exercise(Result* result,int* reset_ok){
    if(threadIdx.x||blockIdx.x)return;
    Parameters p{};p.settle_speed=.01f;p.move_speed=1;p.yaw_speed=1.8f;
    p.brake_rate=5;p.yaw_ramp=.5f;p.round_seconds=120;
    for(int i=0;i<24;i++){p.routes[i].count=1;p.routes[i].loop=1;}
    p.action_to_route[23]=14;p.routes[14].move=3;p.durations[3]=45;
    Parameters current=p;current.yaw_command=rek_keyboard_yaw::Mode::KeyboardReset;
    Arena legacy{},keyboard{},bot_old{},bot_new{};
    legacy.fighter[0].held=keyboard.fighter[0].held=1;
    bot_old.fighter[0].held=bot_new.fighter[0].held=1;
    for(int t=0;t<kTicks;t++){
        int action=0;
        if(t==0||t==31||t==34||t==55||t==91)action=6;
        if(t==30||t==50||t==90)action=1;
        if(t==32||t==40||t==85)action=7;
        if(t==35)action=23;
        advance_fighter(p,legacy.fighter[0],action,legacy);
        advance_fighter(current,keyboard.fighter[0],action,keyboard);
        rek5_bot1::Command command{};command.yaw=t<50?.7f:t<90?-.8f:0.f;
        advance_fighter<true>(p,bot_old.fighter[0],action,bot_old,&command);
        advance_fighter<true>(current,bot_new.fighter[0],action,bot_new,&command);
        result[t].legacy=legacy.fighter[0];result[t].current=keyboard.fighter[0];
        result[t].bot_equal=1;
        const auto* a=reinterpret_cast<const unsigned char*>(&bot_old);
        const auto* b=reinterpret_cast<const unsigned char*>(&bot_new);
        for(unsigned i=0;i<sizeof(Arena);i++)if(a[i]!=b[i])result[t].bot_equal=0;
    }
    pose_reset(current,keyboard);
    *reset_ok=keyboard.fighter[0].keyboard_yaw.ramp==0&&keyboard.fighter[0].keyboard_yaw.sign==0;
    keyboard.fighter[0].keyboard_yaw={1,1};RekNative5RoundResult round{};
    round_reset(current,keyboard,round,true,0);
    *reset_ok=*reset_ok&&keyboard.fighter[0].keyboard_yaw.ramp==0&&keyboard.fighter[0].keyboard_yaw.sign==0;
}
int main(){try{
    unsigned checks=0;auto check=[&](bool ok,const char* why){checks++;if(!ok)throw std::runtime_error(why);};
    Result* result=nullptr;int* reset_ok=nullptr;
    rek5::cuda_check(cudaMallocManaged(&result,kTicks*sizeof(Result)));
    rek5::cuda_check(cudaMallocManaged(&reset_ok,sizeof(int)));
    rek5::cuda_check(cudaMemset(result,0,kTicks*sizeof(Result)));
    exercise<<<1,1>>>(result,reset_ok);rek5::cuda_check(cudaDeviceSynchronize());
    rek_keyboard_yaw::State expected{};
    for(int t=0;t<kTicks;t++){
        const auto& r=result[t];check(r.bot_equal,"opt-in changed Bot1 state");
        check(r.legacy.keyboard_yaw.ramp==0&&r.legacy.keyboard_yaw.sign==0,"legacy advanced new command state");
        const bool busy=t>=35&&t<80;
        check(bool(r.current.strike_active)==busy,"fixture busy duration changed");
        const float raw=busy?0.f:r.current.held==6?1.f:r.current.held==7?-1.f:0.f;
        const float command=rek_keyboard_yaw::advance(expected,raw,.02f,.5f,1.f);
        check(r.current.keyboard_yaw.ramp==expected.ramp&&r.current.keyboard_yaw.sign==expected.sign,"command state mismatch");
        check(r.current.omega==command*1.8f,"second lag or incorrect reset response");
        if(busy)check(r.current.yaw==result[34].current.yaw,"busy root rotation changed");
    }
    check(result[30].current.omega==0&&result[30].legacy.omega>1.7f,"release counterexample missing");
    check(result[32].current.omega<0&&result[32].legacy.omega>0,"reversal counterexample missing");
    check(result[55].current.held==6&&result[79].current.held==6,"busy hold0 lost desired yaw");
    check(result[80].current.omega==.04f*1.8f,"postbusy retained yaw did not restart");
    check(result[1].current.keyboard_yaw.ramp==.08f,"hold0 did not continue ramp");
    check(*reset_ok,"pose/round reset did not clear command state");
    rek5::cuda_check(cudaFree(result));rek5::cuda_check(cudaFree(reset_ok));
    printf("{\"test\":\"keyboard_yaw_runtime\",\"ticks\":%d,\"checks\":%u,\"passed\":true,\"bot_state_bytes_equal\":true,\"physical_parity\":false}\n",kTicks,checks);
    return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 2;}}
