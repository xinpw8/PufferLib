from pathlib import Path
import hashlib,subprocess,json
root=Path(__file__).resolve().parent
runtime=root.parent/'source/ocean/rek_g1/native5/runtime.cu'
code=runtime.read_text()
start=code.index('__global__ void match_input_phase_edges(')
end=code.index('__global__ void finish_bot(',start)
kernel=code[start:end]
test='''#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define __global__
struct Dim{int x;};Dim blockIdx{0},blockDim{1},threadIdx{0};
enum {REK_G1_FIGHT_ROUND_ACTIVE=2};
struct State{struct{struct{int phase;}fight;}combat;};
struct RuntimeView{struct{int arenas;}p;struct{State*states;}c;uint8_t*row_mask;uint8_t*match_input_active;};
static int checks=0;
static void check(bool x,const char*n){++checks;if(!x){fprintf(stderr,"FAIL %s\\n",n);exit(1);}}
'''+kernel+'''
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
printf("{\\"success\\":true,\\"checks\\":%d}\\n",checks);}
'''
(root/'test_runtime_edges.cpp').write_text(test,newline='\n')
(root/'RUNTIME-EDGE-SOURCE.json').write_text(json.dumps({'runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),'exact_kernel_body_sha256':hashlib.sha256(kernel.encode()).hexdigest(),'test_method':'exact kernel body, minimal CPU view fields and launch indices; no GPU'},indent=2)+'\n')
print('Prepared actual runtime edge kernel CPU fixture')
