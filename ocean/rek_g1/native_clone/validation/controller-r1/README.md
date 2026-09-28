# Staged native controller correction

The staged source preserves motion state during an in-match body reset. The original worktree is untouched. `SOURCE-MAP.json` lists exact original/current hashes and the four translation units to rebuild. The parent owns runtime integration, GPU compilation, and GPU execution.

Call `motion->reset_ordered(flags, actual_base_wxyz, REK_G1_RESET_RUNNER_THEN_INPUT, true, stream)` after physical reset/forward for a body or round reset. The last boolean preserves established direct/categorical dispatch. The alternative `REK_G1_RESET_INPUT_THEN_RUNNER` is also implemented. No reset order is chosen by the API. Source subscriptions do not establish global callback order; the clone's selected order remains an explicit candidate configuration.

The existing `motion->reset(...)` remains the cold reconstruction baseline, appropriate for an explicitly fresh episode. Root runtime also clears its independent dispatch mode to -1 on cold reset. A cold reset and the original in-match callbacks have different semantics.

The correction calls the previously validated original-semantics composer Reset, retains prior clip/config/speed, and performs ordinary idle playback in the explicit callback order. Heading initializes at the Runner callback point from the actual base quaternion and current reference. Locomotion route/last/brake command fields and Runner VelocityCommand remain; the four native reset flags clear. The external categorical adapter clears, and unselected fighters remain unchanged. No yaw sign or gain is changed.

The history change extracts the existing production step-1 append/packing operations into `native5/robot_history.h`. Their results match all 27,832 original compiled decoder values across 28 queries, covering empty/partial/full/wrapped history and Clear. Inputs are explicit pretransformed snapshots. This does not validate physical snapshot sampling, encoder transforms, model inference, or physics.

CPU tests execute actual scheduler kernel bodies: 216 ordered-reset checks and 21,839 existing scheduler checks, including 1,200 dispatch-equivalence ticks. Reproduce with `build_cpu.sh` under Linux/WSL. Original decoder comparison is `compare_history.py`; its strict assertion intentionally reports failure for the documented nonzero heading differences.

Heading follows recovered binary32 arithmetic and the existing declared libm callback backend. Four of 20 oracle values differ by one ULP: positive/negative yaw angle by 0.00000011920928955078125 rad and the corresponding cosine by 0.000000059604644775390625. Identity and the tilted fixture match. No tolerance was relaxed and no fitted correction was made. GPU backend outputs are not tested by these CPU results.

The original compiled oracle is pinned in `HISTORY-COMPARISON.json`. The prior 640-row composer fixture establishes the staged Reset component and motion references, while these tests establish its scheduler integration. Whole-runtime parity, event callback order in an actual spawned official robot, and training transfer remain separate unproven properties.
