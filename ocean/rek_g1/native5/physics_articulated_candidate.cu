// Isolated opt-in experiment. Ordinary builds still compile physics.cu.
// Reuse the existing armature-aware ABA integration/contact implementation.
// Force laws, model assets, control clocks, scoring, and failure checks remain
// unchanged. This route is not a claim of MuJoCo or authentic REK parity.
#define REK_NATIVE5_EXPERIMENTAL_ARTICULATED 1
#define B3_ART_CONTACTS 1
#define RP_USE_ART_CACHE 1
#include "physics.cu"
