#include "g1_fight_state.h"
#include "g1_hit_detector.h"

#include <stdio.h>

int main(void) {
    if (REK_G1_HAND_LEFT != 0 || REK_G1_HAND_RIGHT != 1) return 1;
    if (REK_G1_BODY_PART_HAND != 1 || REK_G1_BODY_PART_FOOT != 2
            || REK_G1_BODY_PART_SHIN != 8) {
        return 1;
    }
    if (REK_G1_BODY_ZONE_TORSO != 2
            || REK_G1_BODY_ZONE_PELVIS != 3
            || REK_G1_BODY_ZONE_RIGHT_ANKLE != 17) {
        return 1;
    }
    puts("g1 combat shared types: 9 assertions passed");
    return 0;
}
