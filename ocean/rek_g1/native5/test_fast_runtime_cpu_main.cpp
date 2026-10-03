// CPU emulation tests for lite falls and move_start_v1 in fast_runtime.cu.
// Build and run through test_fast_runtime_cpu.sh.
#include "test_fast_runtime_cpu.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace rek_fast_cpu;
static int failures=0,checks=0;
static void check(bool ok,const std::string& what){checks++;if(!ok){failures++;std::printf("FAIL %s\n",what.c_str());}}
static void report(const char* name,const Totals& t){
    std::printf("{\"case\":\"%s\",\"ticks\":%llu,\"rounds\":%llu,\"learner_hh_starts\":%llu,\"learner_reward\":%.6f,"
        "\"onsets\":[%llu,%llu],\"recoveries\":[%llu,%llu],\"falls\":[%llu,%llu],\"knockouts\":%llu,\"double_knockouts\":%llu,"
        "\"knockout_points\":[%llu,%llu],\"spawn_resets\":%llu,\"stuck_cleared_by_reset\":%llu,\"deferred_round_end_ticks\":%llu}\n",
        name,(unsigned long long)t.ticks,(unsigned long long)t.rounds,(unsigned long long)t.learner_starts[7],t.learner_reward,
        (unsigned long long)t.onsets[0],(unsigned long long)t.onsets[1],(unsigned long long)t.recoveries[0],(unsigned long long)t.recoveries[1],
        (unsigned long long)t.falls[0],(unsigned long long)t.falls[1],(unsigned long long)t.knockouts,(unsigned long long)t.double_knockouts,
        (unsigned long long)t.knockout_points[0],(unsigned long long)t.knockout_points[1],(unsigned long long)t.spawn_resets,
        (unsigned long long)t.stuck_cleared_by_reset,(unsigned long long)t.deferred_round_end_ticks);
}
static void common_invariants(const char* name,const Totals& t){
    const std::string n=name;
    check(t.invariant_failures==0,n+": no failure bits or reward/state invariant breaks");
    check(t.hits_while_not_upright==0,n+": hits score only while both fighters are upright");
    check(t.nonzero_mask_while_down==0,n+": a falling or downed fighter may only continue");
    check(t.terminal_during_count==0,n+": the round never ends during a referee count");
}

int main(int argc,char** argv){
    if(argc!=2&&argc!=3){std::fprintf(stderr,"usage: test_fast_runtime_cpu SMOKE_MODEL.json [NEW_DATASET_DIRECTORY]\n");return 2;}
    const std::string smoke=argv[1];
#if HAVE_ORIGINAL
    // Disabled features are bit-identical to the original runtime.
    for(LearnerPolicy policy:{RepeatHH,ApproachAndHH,AlternateKicks,Idle}){
        Config c;c.learner=policy;c.ticks=3000;
        const Totals a=run_original(c),b=run_modified(c);
        check(a.trajectory_hash==b.trajectory_hash&&a.rounds==b.rounds,
            "disabled lite falls and move reward reproduce the original runtime, policy "+std::to_string(int(policy)));
    }
#else
    std::printf("original runtime not supplied; bit-identity comparison skipped\n");
#endif
    {   // Diagnostic reward: 0.01 on each accepted HH start, nothing else.
        Config c;c.learner=RepeatHH;c.move_reward=true;c.reward_move=7;c.ticks=6000;
        const Totals t=run_modified(c);report("move_reward_repeat_hh",t);common_invariants("move_reward_repeat_hh",t);
        check(t.reward_without_target_start==0&&t.target_start_without_reward==0,"reward exactly on accepted HH starts");
        check(t.nonzero_reward_ticks==t.learner_starts[7],"one rewarded tick per HH start");
        // 145-tick kick plus one decision tick before the next accepted start.
        const double max_starts=double(c.arenas)*c.ticks/146.0;
        check(t.learner_starts[7]>=0.95*max_starts&&t.learner_starts[7]<=max_starts+c.arenas,"repeat-HH reaches the kick rate limit");
        for(int m=0;m<17;m++)if(m!=7)check(t.learner_starts[m]==0,"repeat-HH starts no other move");
    }
    {   // Other moves never pay, including the neighbouring kick category 16.
        Config c;c.learner=AlternateKicks;c.move_reward=true;c.reward_move=7;c.ticks=3000;
        const Totals t=run_modified(c);common_invariants("move_reward_alternate",t);
        check(t.learner_starts[6]>0&&t.learner_starts[7]>0,"both kick categories start");
        check(t.reward_without_target_start==0&&t.target_start_without_reward==0,"only native move 7 is rewarded");
        check(t.nonzero_reward_ticks==t.learner_starts[7],"rewarded ticks equal HH starts only");
    }
    {   // Lite falls with the synthetic smoke model in close combat.
        Config c;c.learner=ApproachAndHH;c.lite_model=smoke;c.ticks=40000;c.round_seconds=60;c.arenas=16;
        const Totals t=run_modified(c);report("lite_smoke_close_combat",t);common_invariants("lite_smoke_close_combat",t);
        check(t.onsets[0]>0&&t.onsets[1]>0,"both fighters start falls");
        check(t.falls[0]>0&&t.falls[1]>0,"both fighters become fallen");
        check(t.recoveries[0]+t.recoveries[1]>0,"some falls recover before being counted");
        check(t.knockouts+t.double_knockouts>0,"counts expire into knockouts");
        check(t.counts_started>=t.knockouts+t.double_knockouts,"every knockout follows a count");
        check(t.spawn_resets>0,"knockouts reset both fighters to spawn");
        check(t.knockout_points[0]+t.knockout_points[1]==5*(t.knockouts+2*t.double_knockouts),"5 points per counted fighter");
        check(t.falls[0]+t.falls[1]>=t.knockouts+2*t.double_knockouts,"no knockout without a fall");
    }
    {   // The probe configuration: lite falls plus the HH-only reward.
        Config c;c.learner=RepeatHH;c.lite_model=smoke;c.move_reward=true;c.reward_move=7;c.ticks=6000;
        const Totals t=run_modified(c);report("lite_smoke_move_reward",t);common_invariants("lite_smoke_move_reward",t);
        check(t.reward_without_target_start==0&&t.target_start_without_reward==0,"falls and knockouts add no reward");
        check(t.nonzero_reward_ticks==t.learner_starts[7]&&t.learner_starts[7]>0,"HH starts still rewarded");
    }
    if(argc==3){
        // Recovery data: the lite runtime under the known smoke model, aggregated
        // by the dataset logger's code. Disjoint seeds for training and holdout.
        const std::string directory=argv[2];
        for(int split=0;split<2;split++){
            Config c;c.learner=ApproachAndHH;c.lite_model=smoke;c.round_seconds=60;
            c.arenas=split?32:64;c.ticks=60000;c.seed=split?1073:73;
            c.dataset_out=directory+(split?"/holdout.json":"/train.json");
            const Totals t=run_modified(c);common_invariants(split?"recovery_holdout":"recovery_train",t);
            report(split?"recovery_holdout":"recovery_train",t);
        }
    }
    std::printf("%s: %d checks, %d failures\n",failures?"FAIL":"PASS",checks,failures);
    return failures?1:0;
}
