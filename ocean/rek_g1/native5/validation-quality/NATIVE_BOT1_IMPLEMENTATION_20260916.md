# Optional recovered Bot1 candidate and rendered observations

Implemented against repository baseline `bd9e2552`, with independent opt-in
modes. Existing deployed binaries and checkpoints were not overwritten.

| Setting | Default | Optional candidate |
| --- | --- | --- |
| `REK_FAST_OPPONENT` | `v4_scripted` | `recovered_bot1_v1` |
| `REK_FAST_OBSERVATION` | `v4_logical` | `rendered_pose_v1` |
| `REK_FAST_SCORING` | `v4_spheres` | Existing `recovered_hit_rules_v1`, new `recovered_hit_rules_v2` |

Runtime JSON uses `fast.opponent_controller`, `fast.observation_mode`, and
`fast.scoring_mode`. Omitted evaluator fields explicitly select legacy modes,
even if the shell contains an opt-in flag. Unknown values fail startup.
Training records these flags through its environment whitelist. Policy input
width 223 and discrete action width 33 are unchanged.

## Bot contract and approximation boundary

Source provenance, exact method coordinates, G1 applicability, and move pools
are in [the native evidence audit](../validation/quality-20260916/NATIVE_BOT1_GAP.md).
`native_bot1.cuh` implements the recovered high-level state transitions:
initial Settling, timed Engaging, Settling, Attacking, Recovering, randomized
Repositioning, and opponent-down GivingRoom with upright grace. Engagement
timeout restarts engagement. Settling expiry retains the native distance-OR-
facing attack gate. It does not add a hard range gate or a facing reward.

Attack category uses Bot1's 0.25 kick probability. Selection scans all 17
assigned moves, consumes full-category and preferred-side reservoir draws,
and prefers the requested side when eligible. Eligibility comes from the first
nonzero limb in the verified impact catalog. Index 16's left-hand impact makes
the emote eligible for the left-punch pool. It is not omitted by `%16`.

Additional native checks made during implementation:

- `RobotInputController.set_VelocityCommand`, RVA `0x2270C40`, copies the vector
  without clamp or sign reversal. Keyboard left strafe/yaw produce positive
  commands. G1 forward/strafe/yaw configuration scales are 1. Native signed
  opponent angle is the negative of compact positive-left bearing; its output
  yaw and strafe already use compact positive-left signs.
- The native locomotion jump table at RVA `0x2368558` sends states 1 through 6
  to `0x236829A`, `0x23683A1`, `0x23683CB`, `0x2368503`, `0x2368503`, and
  `0x2368422`. Attacking and Recovering command zero; Settling allows yaw.
- The reposition lower yaw literal at RVA `0x3E9C91C` is -0.3.
- The timeout invokes `RobotInputController.CancelPunch`, RVA `0x226C430`.
  This invokes `EngineAIPolicyRunner.CancelMove` and, when applicable,
  `SonicMotionComposer.CancelAction` followed by `PlayIdle`. This path has no
  RobotConfig move-interruption gate. It requests a changed commanded clip,
  not an instantaneous physical reset. The compact implementation ends its
  attack clip without modeling the native profile/layer blend.

Only scripted rows are replaced, including type-1/override-2 opponents and the
scripted portion of the existing mixed distribution. Frozen policy overrides,
neutral, retreat, and strafe modes remain independent. Bot decisions use both
fighters' pre-step roots and rendered headings, avoiding side-dependent
one-tick lookahead.

Commands remain continuous internally. Root response uses the same explicit
compact `move_speed`, `yaw_speed`, braking, and yaw-ramp calibration as policy
commands. For example, forward 0.8 requests 0.8 times configured compact
translation speed. Native AI bypasses the keyboard ramp, whereas this candidate
retains the shared compact actuator response. Continuous pose blending is not
available; the dominant translation chooses an existing canned pose route.

These differences are declared modeling choices: 50 Hz combined decision and
locomotion updates, round-local sinusoid time, private seeded xorshift32 RNG,
compact settling/move acceptance, shared slider actuator response, canned pose
selection, immediate clip cancellation, and unchanged sphere geometry. Native
server cadence/RNG and actual controller/physics response are unknown. Own
recovery is unsupported and fails closed if encountered; the compact model
does not synthesize balance dynamics or falls. No authentic Bot1 parity claim
is made.

## Observation and scoring changes

`rendered_pose_v1` snapshots the true previous frame/root independently of
collision sweep history. Cached rendered heading supplies local position-
finite-difference velocity, wrapped yaw rate, semantic heading, and opponent
bearing. Joint velocities use wrapped differences against the true previous
frame. Fields 221/222 become the sum of weighted point deltas, matching the
recovered CleanHits interpretation. Dynamics, internal command yaw, masks,
contact history, and rewards are unchanged by this mode.

`recovered_hit_rules_v2` removes two legacy eligibility restrictions that v1
intentionally retained: route 23 exclusion and `limb_enabled` route whitelists.
All six existing striker spheres are candidates; native catalog apex rules
decide limb eligibility. This permits emote left-hand attribution and both
foot/shin attribution for lower-body impact events. Geometry, relative sphere-
center speed proxy, contact-enter union latch, cooldown, and per-invocation
apex deduplication are unchanged. V4 and recovered v1 retain their prior
filters.

## Verification

CPU FSM fixture: 20,042 checks, zero failures. Deterministic gates, timeout and
execution-result transitions, command signs, reservoir fallback, repeatable
seeded draws, and pool restrictions are covered. There were 5,055 kicks in
20,000 draws. The independent rendered-pose helper has its own CPU fixtures.

The linked production runtime was exercised at 32 arenas, 1,200 ticks per
case, including a round reset, with captured CUDA graphs:

| Case | Observed result |
| --- | --- |
| Default V4 versus preserved old runtime object | First 1,000 ticks bitwise equal; small differences begin at randomized second-round reset, tick 1,001. |
| Recovered v1 versus preserved old runtime object | Same bounded numerical differences as V4. |
| Rendered versus logical observation on the new runtime | Zero bitwise differences in qpos, qvel, masks, actions, rewards, or terminals. 2,762,496 observation checks; maximum absolute error 0.0000123978 against exported-pose formulas. |
| Recovered Bot1, rendered observations, recovered v2 | No runtime failures. 135 accepted attacks as side 0 and 137 as side 1; all 17 assigned moves selected. Same rendered observation check count and error bound. |

The old-object comparison is **numerically preserved, not bitwise preserved**
across the second reset. Full-run mismatch counts were 4,800 qpos values, 305
qvel values, and 13,103 raw-observation values. Maximum absolute differences
were respectively `3.87430191e-7`, `5.01051545e-7`, and `1.66893005e-6`.
Masks, actions, rewards, and terminals were exactly equal. Large ULP counts
near zero are retained in the machine-readable output. The onset at reset is
consistent with changed float compilation; separate compiler-causality proof
was not performed. Existing legacy artifacts remain available.

The post-training v2 fixture completed with exit code 0: six real-catalog CPU
cases and twelve synthetic-contact GPU cases passed with zero failures. The
CUDA fixture calls the production `strike_contacts` adapter. Observed v1 points
were `[0,0,2,2,0,0]`, versus v2 `[1,2,2,2,2,0]`, for emote left hand, right-knee
right foot, right-knee right shin, left-kick left foot, left-kick left shin,
and emote right hand. These are acceptance fixtures, not measured native
contact trajectories. Recorded evidence:
[fixture output](../validation/quality-20260916/build-bot1/scoring-v2-gpu.stdout.txt),
[stderr](../validation/quality-20260916/build-bot1/scoring-v2-gpu.stderr.txt), and
[exit code](../validation/quality-20260916/build-bot1/scoring-v2-gpu.exit-code.txt).

Sanitized results: [native-bot1-runtime-20260916.jsonl](native-bot1-runtime-20260916.jsonl).
Private build/output directory:
`/home/spark-advantage/rek-training/policy-quality-20260916-r1/build-bot1`.
Private source snapshot is the sibling `bot1-source`. Build command is
`validation-quality/build_bot1_probe.sh SOURCE OLD_RECOVERED_BUILD BUILD`.
The v2 device fixture command is `scoring-v2-test --gpu`.

Verified production source SHA-256:
`a3b1b6c8ee4582948274ca7aa2e415479923eb87d20947c7bde3ede78b98b826`.
Verified `fast_runtime.o` SHA-256:
`8e2874d13bdd770af011f56ce6a39b4faa690a89659e0e082ec89d744147ea3d`.
No production source or object changes were made after the primary agent took the GPU
slot for policy evaluation/training.
