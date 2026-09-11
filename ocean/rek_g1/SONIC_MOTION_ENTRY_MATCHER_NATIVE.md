# Sonic motion loop-entry matcher

## Scope

`sonic_motion_entry_matcher_native.c` is a heap-free C implementation of the
pure part of the current REK build's active-source `MatchEntryCursor` path.  It
also exposes the clip-view, feature-bake boundary, registration, mirroring, and
sampling operations needed to supply that matcher.

This component does not claim motion or environment parity.  In particular,
it does not infer foot positions from joint angles.  The six root-local ankle
coordinates used by the game must be supplied or baked through the explicit
kinematics callback.

## Provenance

The implementation was recovered from these local, read-only artifacts:

| Artifact | SHA-256 | Relevant methods |
| --- | --- | --- |
| `controller-audit-isil/IsilDump/REKApp/REKApp/SonicMotionComposer.txt` | `b59ba1dfc9ce6072b61088b36ee5dd469e09899e82dfe5b7bb49cd27edefb954` | `RegisterFootFeatures` at line 6598, `GetClipIlDofPos` at 6835, `MakeFeat` at 7017, `SampleFeatureAt` at 7220, `SampleFeatureLerp` at 7560, `MatchEntryCursor` at 8397 |
| `controller-audit-isil/IsilDump/REKApp/REKApp/SonicPolicyRunner.txt` | `5c7668aa79591cd84dfd120856ecdf96554309c85a2d5a425e8f42636381ab58` | `PoseRobotKinematic` at line 17495, `BakeFootFeatures` at 17755 |
| `ocean/rek/evidence/g1_sonic_staging_policy_contract.v1.json` | `d6f3e9525b37023ade4c9c48edefcc00f1ce39933c5ef87bfc1454b764107c23` | Associates the dump inputs with the current Steam build |

The contract identifies the matching current Steam inputs as GameAssembly
SHA-256
`6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`
and global metadata SHA-256
`e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd`.

No game binary or decoded motion asset is copied into this component or its
tests.

## Recovered behavior

The runner bakes one six-float feature row for every decoded motion frame:

```text
[left ankle local x, left ankle local y, left ankle local z,
 right ankle local x, right ankle local y, right ankle local z]
```

For each frame, the runner applies the 29 decoded MuJoCo-order joint values
through its configured index and Unity conversion maps, synchronizes the
kinematic pose, obtains both `ankle_roll` body positions, and transforms those
positions into the root body's local frame.  The runner restores its saved
joint state after the bake.  The C bake function preserves the exact per-row
iteration boundary but requires a callback to perform those runtime operations.

The matcher performs the following binary32 sequence:

1. Convert `controller_rate_hz * blend_seconds` to an integer using ties-to-even
   rounding, then clamp each blend width to at least one tick.
2. Compute the transition center as
   `float(w_out) * float(w_in) / float(w_out + w_in)`.
3. Advance and wrap the outgoing cursor by
   `outgoing.per_tick * transition_center`.
4. Sample the outgoing six-coordinate feature with clamped adjacent frames and
   binary32 linear interpolation.
5. Compare it with each integer target frame from `start_frame` through
   `end_frame`, inclusive.  Mirroring exchanges feet and negates each exchanged
   x coordinate.
6. Accumulate squared distance in the native order:
   `left = ((left_y^2 + left_x^2) + left_z^2)`,
   `right = ((right_y^2 + right_x^2) + right_z^2)`, then
   `distance = right + left`.
7. Retain the first frame at the minimum distance and compute
   `entry = best_frame - transition_center * target.per_tick`.
8. Wrap a looping target or clamp a non-looping target.

The implementation uses explicit binary32 operation boundaries, rejects a
non-ties-to-even floating-point environment, and limits exact integer-to-float
frame conversion to `2^24` frames.

## Deliberate fail-closed boundary

The game falls back to the authored entry cursor when the source is inactive or
when either feature array is absent.  The native candidate instead returns
`SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURES_NOT_REGISTERED` when an active match
lacks either feature array.  Its callback returns zero, so
`sonic_motion_composer_native_play_action` reports backend failure and keeps the
previous layer and cursor unchanged.  This is intentional: an unmeasured foot
trajectory must not silently become training data.

`SonicMotionEntryMatcherNativeKinematicsSampler` is the remaining runtime
boundary.  It must return measured root-local ankle positions for the posed
29-DOF row.  A callback failure returns
`SONIC_MOTION_ENTRY_MATCHER_NATIVE_KINEMATICS_FAILURE`.  A non-finite output is
also rejected.  I don't know the current runtime feature rows because no
authoritative feature artifact has been supplied or produced in this component.

## Integration

Storage is caller-owned.  Allocate one matcher and feature registry per composer
when composers may execute concurrently.  Register every route clip before a
transition can use it.  If another native backend needs the composer's shared
`backends.context`, place the matcher inside that aggregate context and use a
small adapter callback that passes the embedded matcher to
`sonic_motion_entry_matcher_native_callback`.

## Deterministic validation

`test_sonic_motion_entry_matcher_native.c` contains no decoded game data.  It
checks mirror exchange, clamped integer and linear sampling, feature-bake
failure, registry replacement and capacity, ties-to-even transition arithmetic,
the native six-term summation order, first-minimum tie selection, loop and
non-loop cursor handling, maximum-float distance saturation, and transactional
composer failure when features are missing.

Evidence records are under `<evidence-run>/commands/`:

- `subagent-loop-entry-003-wsl-strict-tests`
- `subagent-loop-entry-004-wsl-sanitizers-clang`
- `subagent-loop-entry-005-spark-arm64-validation`
- `subagent-loop-entry-006-operation-order`
- `subagent-loop-entry-008-final-wsl-validation`
- `subagent-loop-entry-009-final-spark-validation`

The Spark record exercises strict GCC, GCC AddressSanitizer plus
UndefinedBehaviorSanitizer, strict Clang, and Clang static analysis on aarch64.
