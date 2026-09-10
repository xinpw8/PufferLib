#include "held_input.h"

/* Host-only entry point for checking the header's original input function. */
void rek_g1_test_host_held_input(RekG1HeldInputState* state,
        const RekG1InputFrame* frame, const RekG1InputTiming* timing,
        int settled, int busy, RekG1InputDecision* decision) {
    *decision = rek_g1_apply_input_frame(state, *frame, *timing, settled, busy);
}
