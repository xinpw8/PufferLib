from pathlib import Path
import hashlib, json, shutil, difflib

ROOT=Path(__file__).resolve().parent
BASE=Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1')
SRC=ROOT/'source'
ORIG=ROOT/'original'
SRC.mkdir(exist_ok=True); ORIG.mkdir(exist_ok=True)
for name in ['g1_native_combat_cuda.cu','g1_native_combat_cuda.h']:
    original=(BASE/name).read_bytes()
    for target in [ORIG/name,SRC/name]:
        if target.exists(): raise RuntimeError(f'preserve existing {target}')
        target.write_bytes(original)
cu=(SRC/'g1_native_combat_cuda.cu').read_text()
cu=cu.replace('#include "g1_native_combat_cuda.h"','#include "g1_native_combat_cuda.h"\n#include "g1_match_lifecycle.h"',1)
cu=cu.replace('__global__ void begin_tick_kernel(', 'template <bool MATCH_MODE = false>\n__global__ void begin_tick_kernel(',1)
cu=cu.replace('int32_t* statuses,\n        size_t count) {\n    for', 'int32_t* statuses,\n        size_t count) {\n    for',1)
start=cu.index('__global__ void begin_tick_kernel(')
end=cu.index('__device__ bool assemble_intent',start)
section=cu[start:end]
section=section.replace('size_t count) {','size_t count,\n        uint8_t countdown_complete = 0) {',1)
needle='        if (!states[arena].pending_episode_reset) continue;'
addition='''        if constexpr (MATCH_MODE) {
            statuses[arena] = rek_g1_native_match_begin(
                &states[arena], countdown_complete,
                &tick_signals[arena], &episode_reset[arena]);
            continue;
        }
'''
assert section.count(needle)==1
section=section.replace(needle,addition+needle,1)
cu=cu[:start]+section+cu[end:]
cu=cu.replace('__global__ void pack_contacts_kernel(', 'template <bool MATCH_MODE = false>\n__global__ void pack_contacts_kernel(',1)
needle='        if (states[arena].reset_pending || terminals[arena]) continue;'
assert cu.count(needle)==1
cu=cu.replace(needle,needle+'''
        if constexpr (MATCH_MODE) {
            if (states[arena].combat.fight.phase != REK_G1_FIGHT_ROUND_ACTIVE) continue;
        }''',1)
cu=cu.replace('template <bool DEFER_OBSERVATION>','template <bool DEFER_OBSERVATION, bool MATCH_MODE = false>')
needle='        RekG1FallStepResult fall_result[2] = {};'
assert cu.count(needle)==1
cu=cu.replace(needle,'''        if constexpr (MATCH_MODE) {
            // Inactive phases are advanced only by begin_tick_match. Their
            // stale fall samples must not produce referee events or age the
            // active round clock. No physical/history reset is requested here.
            if (value.combat.fight.phase != REK_G1_FIGHT_ROUND_ACTIVE) {
                if constexpr (DEFER_OBSERVATION) {
                    bool valid = true;
                    for (size_t fighter = 0; fighter < 2; fighter++) {
                        const size_t fighter_row = row + fighter;
                        const int64_t* integers = fall_integers + fighter_row * FALL_INTEGER_FIELDS;
                        valid = valid && fall_valid[fighter_row]
                            && (integers[0] == 0 || integers[0] == 1)
                            && (integers[1] == 0 || integers[1] == 1)
                            && (integers[2] == 0 || integers[2] == 1)
                            && integers[3] >= 0 && uint64_t(integers[3]) <= UINT32_MAX
                            && (integers[4] == 0 || integers[4] == 1)
                            && (integers[5] == 0 || integers[5] == 1)
                            && (integers[6] == 0 || integers[6] == 1);
                    }
                    if (!valid) statuses[arena] = REK_G1_CUDA_NATIVE_COMBAT_INPUT_INVALID;
                }
                continue;
            }
        }

'''+needle,1)
cu=cu.replace('begin_tick_kernel<<<','begin_tick_kernel<false><<<')
cu=cu.replace('pack_contacts_kernel<<<','pack_contacts_kernel<MATCH_MODE><<<')
cu=cu.replace('post_step_kernel<DEFER_OBSERVATION><<<','post_step_kernel<DEFER_OBSERVATION, MATCH_MODE><<<')
start=cu.index('extern "C" cudaError_t rek_g1_cuda_native_combat_begin_tick(')
end=cu.index('template <bool DEFER_OBSERVATION',start)
new=cu[start:end].replace('rek_g1_cuda_native_combat_begin_tick(', 'rek_g1_cuda_native_combat_begin_tick_match(',1)
new=new.replace('        cudaStream_t stream) {','        cudaStream_t stream,\n        uint8_t countdown_complete) {',1)
new=new.replace('    if (arena_count == 0)', '    if (countdown_complete > 1) return cudaErrorInvalidValue;\n    if (arena_count == 0)',1)
new=new.replace('begin_tick_kernel<false>', 'begin_tick_kernel<true>',1)
new=new.replace('input_reset, statuses, arena_count);','input_reset, statuses, arena_count, countdown_complete);',1)
cu=cu[:end]+new+cu[end:]
start=cu.index('extern "C" cudaError_t rek_g1_cuda_native_combat_post_step_deferred(')
end=cu.index('extern "C" cudaError_t rek_g1_cuda_native_combat_observe(',start)
new=cu[start:end].replace('rek_g1_cuda_native_combat_post_step_deferred(', 'rek_g1_cuda_native_combat_post_step_deferred_match(',1)
new=new.replace('post_step_dispatch<true>', 'post_step_dispatch<true, true>',1)
cu=cu[:end]+new+cu[end:]
(SRC/'g1_native_combat_cuda.cu').write_text(cu,newline='\n')
h=(SRC/'g1_native_combat_cuda.h').read_text()
start=h.index('cudaError_t rek_g1_cuda_native_combat_begin_tick(')
end=h.index('\n);',start)+3
new=h[start:end].replace('rek_g1_cuda_native_combat_begin_tick(', 'rek_g1_cuda_native_combat_begin_tick_match(',1)
new=new.replace('    cudaStream_t stream\n','    cudaStream_t stream,\n    uint8_t countdown_complete\n',1)
h=h[:end]+'''\n\n/* Explicit match opt-in. Call together with post_step_deferred_match.
 * One 20 ms transition clock per call. episode_reset is emitted only for
 * ROUND_PREPARED; caller must physically reset before the next begin call.
 * That next call may activate COUNTDOWN when countdown_complete==1. The clone
 * uses 1 to omit the unknown presentation Timeline. Fight outcome remains
 * latched through OVER->IDLE until an explicit init. State layout unchanged. */
'''+new+h[end:]
start=h.index('cudaError_t rek_g1_cuda_native_combat_post_step_deferred(')
end=h.index('\n);',start)+3
new=h[start:end].replace('rek_g1_cuda_native_combat_post_step_deferred(', 'rek_g1_cuda_native_combat_post_step_deferred_match(',1)
h=h[:end]+'\n\n/* Match opt-in: inactive fight phases do not age falls, scores or round time. */\n'+new+h[end:]
(SRC/'g1_native_combat_cuda.h').write_text(h,newline='\n')
diff=''
for name in ['g1_native_combat_cuda.cu','g1_native_combat_cuda.h']:
    diff+=''.join(difflib.unified_diff((ORIG/name).read_text().splitlines(True),(SRC/name).read_text().splitlines(True),fromfile='original/'+name,tofile='source/'+name))
(ROOT/'match-lifecycle.diff').write_text(diff,newline='\n')
print('staged two additive APIs; original state layout unchanged')
