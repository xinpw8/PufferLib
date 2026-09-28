#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "kernels_cpu.inc"

static int checks=0;
static void check(bool ok,const char* what){++checks;if(!ok){fprintf(stderr,"FAIL %s\n",what);exit(1);}}
struct Tick {
    uint32_t falls[2]{},signals[1]{},calls[1]{},attributed[1]{},scored[1]{};
    int32_t score[2]{},status[1]{};
    uint8_t terminal[1]{},reset[1]{},input[2]{},dampened[2]{},begin[1]{},complete[1]{},clear[1]{};
};
static RekG1CudaNativeCombatState initial(){RekG1CudaNativeCombatState s{};int32_t st=-1;init_kernel(&s,&st,1);check(st==0,"actual init kernel");return s;}
template<bool MATCH> static void begin(RekG1CudaNativeCombatState& s,Tick& t,uint8_t ready=1){
    begin_tick_kernel<MATCH>(&s,t.falls,t.signals,t.calls,t.score,t.attributed,t.scored,t.terminal,t.reset,t.input,t.status,1,ready);
    check(t.status[0]==0,"begin status");
}
template<bool MATCH> static void post(RekG1CudaNativeCombatState& s,Tick& t,bool fallen=false){
    float floats[10]={fallen?120.f:0.f,fallen?0.1f:1.f,0.002f,0.f,0.f,0.f,1.f,0.002f,0.f,0.f};
    int64_t ints[14]={1,0,1,0,0,0,0,1,0,1,0,0,0,0};
    uint8_t valid[2]={1,1};int64_t offset[1]={0},count[1]={0};float now[1]={1.f};RekG1HitContact contact{};
    post_step_kernel<true,MATCH>(&s,floats,ints,valid,offset,count,now,&contact,t.falls,t.signals,t.calls,t.score,t.attributed,t.scored,t.terminal,t.input,t.dampened,t.begin,t.complete,t.clear,t.status,1);
    check(t.status[0]==0,"post status");
}
static void end_round(RekG1CudaNativeCombatState& s,int winner){
    check(s.combat.fight.phase==REK_G1_FIGHT_ROUND_ACTIVE,"finish active only");
    s.combat.fight.clean_hits[0]=winner==0?2:0;s.combat.fight.clean_hits[1]=winner==1?2:0;
    RekG1FightAdvanceInput input{};input.delta_seconds=0.002f;input.time_remaining_seconds=0.f;
    RekG1FightStepResult result{};
    check(rek_g1_fight_advance_active(&s.combat.fight,&input,&result)==REK_G1_FIGHT_OK,"actual fight resolves round");
    check((result.signals&REK_G1_FIGHT_SIGNAL_ROUND_ENDED)!=0,"actual round ended signal");
    s.combat.fight=result.next_state;s.pending_episode_reset=1;
}
static int prepare_next(RekG1CudaNativeCombatState& s,Tick& t){
    auto wins0=s.combat.fight.rounds_won[0],wins1=s.combat.fight.rounds_won[1];
    int ticks=0;
    while(s.combat.fight.phase==REK_G1_FIGHT_BETWEEN_ROUNDS){
        check(++ticks<=251,"bounded five second transition");begin<true>(s,t);
        check(s.combat.fight.rounds_won[0]==wins0&&s.combat.fight.rounds_won[1]==wins1,"round wins survive between ticks");
        if(s.combat.fight.phase==REK_G1_FIGHT_BETWEEN_ROUNDS)check(!t.reset[0]&&!t.signals[0],"no premature physical reset");
    }
    check(ticks>=250,"five second delay retained within one20ms tick");
    check(s.combat.fight.phase==REK_G1_FIGHT_ROUND_COUNTDOWN,"prepared remains inactive");
    check(t.reset[0]==1,"one physical reset at preparation");
    check(t.signals[0]==(REK_G1_FIGHT_SIGNAL_ROUND_PREPARED|REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN),"original preparation signals");
    auto snapshot=s;
    for(int i=0;i<10;i++)post<true>(s,t,true);
    check(memcmp(&s,&snapshot,sizeof(s))==0,"inactive substeps freeze falls and fight");
    return ticks;
}
int main(){
    check(sizeof(RekG1CudaNativeCombatState)==252,"state ABI size252");check(offsetof(RekG1CudaNativeCombatState,reset_pending)==248,"reset offset248");
    Tick t{};auto s=initial();end_round(s,0);auto legacy=s;
    begin<false>(legacy,t);check(legacy.combat.fight.rounds_won[0]==0&&legacy.combat.fight.current_round_number==1&&t.reset[0]==1,"legacy training episode behavior retained");
    prepare_next(s,t);check(s.combat.fight.rounds_won[0]==1&&s.combat.fight.current_round_number==2,"match round2 retains win");
    begin<true>(s,t,0);check(s.combat.fight.phase==REK_G1_FIGHT_ROUND_COUNTDOWN&&!t.reset[0],"caller can hold countdown");
    begin<true>(s,t);check(s.combat.fight.phase==REK_G1_FIGHT_ROUND_ACTIVE&&t.signals[0]==REK_G1_FIGHT_SIGNAL_ROUND_STARTED&&!t.reset[0],"activation after reset only");
    end_round(s,0);check(s.combat.fight.fight_result==REK_G1_FIGHT_WON_BY_ROUNDS&&s.combat.fight.fight_winner_index==0&&s.combat.fight.rounds_won[0]==2,"two wins end actual fight");
    int exits=0;
    for(int i=0;i<300;i++){begin<true>(s,t);check(!t.reset[0],"completed fight never respawns");exits+=(t.signals[0]&REK_G1_FIGHT_SIGNAL_FIGHT_EXITED)!=0;}
    check(exits==1&&s.combat.fight.phase==REK_G1_FIGHT_IDLE&&s.combat.fight.fight_winner_index==0&&s.combat.fight.rounds_won[0]==2,"fight exit retains outcome");
    auto finished=s;post<true>(s,t,true);check(memcmp(&s,&finished,sizeof(s))==0,"idle has no new fall or clock events");
    s=initial();end_round(s,-1);prepare_next(s,t);check(s.combat.fight.current_round_is_redo&&s.combat.fight.round_duration_seconds==30.f,"tie prepares30s redo");
    begin<true>(s,t);end_round(s,1);prepare_next(s,t);check(!s.combat.fight.current_round_is_redo&&s.combat.fight.round_duration_seconds==120.f,"redo win returns normal120s");
    begin<true>(s,t);end_round(s,-1);check(s.combat.fight.fight_result==REK_G1_FIGHT_WON_BY_ROUNDS&&s.combat.fight.rounds_won[0]==0&&s.combat.fight.rounds_won[1]==1,"native three-round tie loss tie cap0to1");
    // Exercise the same active substep and counted-reset kernel branches in both modes.
    auto a=initial(),b=a;Tick ta{},tb{};begin<false>(a,ta);begin<true>(b,tb);post<false>(a,ta);post<true>(b,tb);
    check(memcmp(&a,&b,sizeof(a))==0&&memcmp(&ta,&tb,sizeof(ta))==0,"active mode parity exact");
    a.reset_pending=1;a.reset_complete_not_before_seconds=1.f;b=a;ta={};tb={};begin<false>(a,ta);begin<true>(b,tb);post<false>(a,ta);post<true>(b,tb);
    check(memcmp(&a,&b,sizeof(a))==0&&memcmp(&ta,&tb,sizeof(ta))==0&&tb.complete[0]==1&&tb.input[0]==1,"counted reset completion unchanged");
    auto invalid=initial(),before=invalid;uint32_t sig=123;uint8_t reset=123;
    check(rek_g1_native_match_begin(&invalid,2,&sig,&reset)==REK_G1_CUDA_NATIVE_COMBAT_INPUT_INVALID,"invalid countdown rejected");
    check(memcmp(&invalid,&before,sizeof(before))==0&&sig==123&&reset==123,"failed request atomic");
    invalid.combat.fight.phase=static_cast<RekG1FightPhase>(99);before=invalid;
    check(rek_g1_native_match_begin(&invalid,1,&sig,&reset)!=0&&memcmp(&invalid,&before,sizeof(before))==0,"invalid phase rejected atomically");
    puts("{\"success\":true,\"scope\":\"actual CUDA kernel bodies on CPU, no GPU\"}");printf("checks=%d\n",checks);return 0;
}
