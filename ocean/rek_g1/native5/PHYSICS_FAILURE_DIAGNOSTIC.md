# Native Puffysics failure diagnostic

The native training failure can occur before any PPO update. A separate
executable using the real native SONIC controller, 512 arenas, and deterministic
currently valid semantic actions reproduced a physics failure twice at the
same boundary: physics call 417, zero-based semantic tick 41, substep 6, arena
12. Each trajectory completed in approximately 4.6 seconds. The executable
stopped at its first physics failure; its configured bound was 100 ticks.

The production engine and adapter were unchanged. The diagnostic executable
intercepts `physics_step` to save each preceding world on the GPU, executes the
existing kernel, and replays the first failed world's preceding state through
the same independent-solver stages. It performs no PPO updates. All controller
targets at the failure were finite, with maximum absolute target 3.08920002 rad.

## Observed failure

The native statistics were `[23, 1, 0, 0]`: maximum contact count 23, nonfinite
state present, no collision/solver status flag, and zero predictive-limit
impulses. The current manifold count changed from 17 to 16. Neither contact
capacity nor a cylinder GJK/EPA error caused this event.

| Replay boundary | Maximum linear velocity component, m/s | Maximum angular velocity component, rad/s | State |
| --- | ---: | ---: | --- |
| Before the step | 3.62785172 | 30.0326729 | Finite |
| After contact warm start | 10.9849787 | 1025.48425 | Finite |
| After joint warm start | 1.5733495e31 | 3.13031634e34 | Finite |
| After external-force velocity integration | 1.5733495e31 | 3.13031634e34 | Finite |
| After biased contact solve | 1.5733495e31 | 3.13031634e34 | Finite |
| After biased joint solve | 8.80550588e29 | 1.44044213e31 | First nonfinite linear velocity, native body 1 |

The reported maxima retain finite components when other components are NaN;
the separate validity scan detects those nonfinite components.

The preceding finite world already contained extreme cached joint impulses.
Joint 14 held an upper-limit impulse of `6.47818579e28`, perpendicular impulses
`[-1.53936916e29, 1.24942448e28]`, and a linear impulse whose largest component
was `-3.64560648e30`. Joint 13 held an upper-limit impulse of `6.33063734e28`.
These are the waist/torso portion of the player tree. Both the prior body state
and these caches remained representable as float32 before warm starting.

This establishes the immediate path to failure: very large cached joint
impulses amplify velocities during joint warm starting, followed by nonfinite
arithmetic in the biased joint sweep. The underlying reason those impulses
accumulated is unresolved. This result does not identify a validated production
solver fix.

The earlier independent-capsule, 512-arena, horizon-256 benchmark also failed:
84 arenas had nonfinite states, maximum contact count was 48, and collision,
solver-status, and contact-capacity failures were zero. That existing failure
predates the native5 port.

## Separate adapter audit

`physics_adapter_audit.cpp` performs model loading and arithmetic only, with
zero CPU physics steps and zero GPU initialization.

Eight exported hinge intervals cross the positive-pi branch boundary: joints
0, 6, 15, 22, 29, 35, 44, and 51. The PD/state adapter unwraps the hinge angle
relative to the interval midpoint; the independent solver's hinge-limit rows
use wrapped `b3_twist` directly. For joint 0, a legal relative angle
`3.45958567` rad in `[-1.63292122, 3.77757883]` becomes `-2.82359958` rad in the
solver and is falsely below the lower limit. This is a reproducible branch
disagreement. No such disagreement was present in the immediate pre-failure
state above. Its role earlier in that trajectory has not been established.

Twenty of 58 hinges exceed the isolated explicit passive-damping scalar
stability threshold `h*c*(axis dot (I_parent^-1 + I_child^-1) axis) > 2` at the
initial pose. The maximum is 15.6367903 at joint 43. This is a local
unconstrained diagnostic. Complete constrained-tree stability requires separate
analysis. This calculation does not prove the cause of the observed cache growth.

The unchanged independent mode uses 2 ms steps, one substep, two biased joint
sweeps, two relaxation joint sweeps, contact frequency 30 Hz, contact damping
10, joint-constraint frequency 90 Hz, and joint-constraint damping 2. It omits
the exported rotor armature from free integration. Its velocity caps occur
before the relaxation sweeps.

## Reproduction and evidence

- `physics_first_failure_probe.cu` contains the bounded native-controller
  trajectory and the isolated GPU stage replay. Compile it in place of
  `physics.cu`, retaining the existing runtime/controller/measurement/motion
  objects. Use the physics flags `-std=c++17 -O3 -arch=sm_121`, default FMA,
  the MuJoCo include path, and the runtime's existing link dependencies.
- Its six positional inputs are XML, compiled-model export, motion assets,
  motion features, controller encoder, and controller decoder. Use the pinned
  encoder and decoder with exactly 1024 robot rows. No Python is invoked.
- Run from a private evidence directory. On failure it writes
  `before.world.bin`, `after.world.bin`, and `before.controls.bin` there.
  These binary state records are private evidence and are excluded from git.
- `physics_adapter_audit.cpp` provides the separate host-only branch and
  damping calculation. It accepts XML and compiled-model export paths.

The next solver investigation should trace when the cached impulses begin
growing, starting with the saved preceding world and the player waist/torso
constraints. Clearing caches or clamping away invalid state would conceal this
failure and was not applied.

Sanitized stdout/stderr are in `validation/physics-first-failure/`. The private
binary records have also been preserved outside temporary storage at
`/home/spark-advantage/rek-training/native5-rek-20260913-v1/physics-first-failure/`.
