#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define __global__
struct Dim{int x;};Dim blockIdx{0},blockDim{1},threadIdx{0};
enum {REK_G1_FIGHT_ROUND_ACTIVE=2};
struct State{struct{struct{int phase;}fight;}combat;};
struct RuntimeView{struct{int arenas;}p;struct{State*states;}c;uint8_t*row_mask;uint8_t*match_input_active;};
static int checks=0;
static void check(bool x,const char*n){++checks;if(!x){fprintf(stderr,"FAIL %s\n",n);exit(1);}}
__global__ void match_input_phase_edges(RuntimeView v) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=v.p.arenas*2)return;
    const uint8_t active=v.c.states[row/2].combat.fight.phase==REK_G1_FIGHT_ROUND_ACTIVE;
    // Original DeactivateControls runs once at round end. Keep ownership and
    // command storage intact; this mask only drives the input component edge.
    v.row_mask[row]=v.match_input_active[row]&&!active;
    v.match_input_active[row]=active;
}

int main(){
State states[2]{};states[0].combat.fight.phase=2;states[1].combat.fight.phase=2;
uint8_t mask[4]={9,9,9,9},active[4]={1,1,1,1};RuntimeView v{{2},{states},mask,active};
auto run=[&](){for(int i=0;i<4;i++){threadIdx.x=i;match_input_phase_edges(v);}};
run();for(int i=0;i<4;i++)check(active[i]==1&&mask[i]==0,"cold active no deactivate");
states[0].combat.fight.phase=3;run();for(int i=0;i<4;i++)check(active[i]==(i>=2)&&mask[i]==(i<2),"only ended arena emits edge");
run();for(int i=0;i<4;i++)check(mask[i]==0,"not repeated while inactive");
states[0].combat.fight.phase=1;run();for(int i=0;i<2;i++)check(active[i]==0&&mask[i]==0,"countdown not active");
states[0].combat.fight.phase=2;run();for(int i=0;i<4;i++)check(active[i]==1&&mask[i]==0,"reactivation preserves input ownership");
states[1].combat.fight.phase=4;run();for(int i=0;i<4;i++)check(active[i]==(i<2)&&mask[i]==(i>=2),"other arena fight over");
states[1].combat.fight.phase=0;run();for(int i=0;i<4;i++)check(mask[i]==0,"idle never repeats deactivation");
printf("{\"success\":true,\"checks\":%d}\n",checks);}
