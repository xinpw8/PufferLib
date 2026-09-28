from pathlib import Path
root=Path(__file__).parent/'source-r1/ocean/rek_g1/native5'
(root/'robot_history.h').write_text('''#pragma once
#include "../g1_cuda_qualifiers.h"

/* One transformed channel at a time. The original configured decoder requests
 * ten step-1 snapshots, oldest first with zero padding after Clear. This dense
 * equivalent deliberately does not implement arbitrary ring queries. */
REK_G1_FN static inline void rek_native_history_push_channel(
        float history[930], int group, int width, int column, float entry) {
    float* channel=history+group+column;
    for (int slot=0;slot<9;++slot) channel[slot*width]=channel[(slot+1)*width];
    channel[9*width]=entry;
}
REK_G1_FN static inline float rek_native_decoder_value(
        const float tokens[64], const float history[930], int index) {
    return index<64 ? tokens[index] : history[index-64];
}
''',newline='\n')
p=root/'robot_state.cu';s=p.read_text()
s=s.replace('#include "robot_state.cuh"','#define REK_G1_CUDA_DEVICE 1\n#include "robot_state.cuh"\n#include "robot_history.h"',1)
old='''    float* history=s.history+row*930+group+column;
    for (int slot=0;slot<9;++slot) history[slot*width]=history[(slot+1)*width];
    history[9*width]=entry;'''
assert s.count(old)==1
s=s.replace(old,'''    rek_native_history_push_channel(s.history+row*930,group,width,column,entry);''')
old='s.decoder_observations[row*994+i]=i<64 ? tokens[row*64+i] : s.history[row*930+i-64];'
assert s.count(old)==1
s=s.replace(old,'s.decoder_observations[row*994+i]=rek_native_decoder_value(tokens+row*64,s.history+row*930,i);')
p.write_text(s,newline='\n')
print('exact existing history operations exposed to shared CPU/device contract')
