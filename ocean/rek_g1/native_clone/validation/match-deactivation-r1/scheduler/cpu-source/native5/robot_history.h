#pragma once
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
