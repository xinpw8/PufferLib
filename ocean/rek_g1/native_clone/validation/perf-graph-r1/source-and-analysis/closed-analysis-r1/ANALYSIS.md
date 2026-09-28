# CUDA graph comparison: disabled

The strict comparison failed in 356 of365 replies. Initial and cold-reset snapshots before the first step matched. The first difference was qpos[1] at tick1: 2.18278728e-11 in that component. Other first-tick components also differ; this was not an initial-state advancement.

Both workers exited0 and produced8 short-round terminals. Discrete terminal records differed in 0 replies; command-event records differed in 0 replies. Full per-field comparisons are in ANALYSIS.json.

Later trajectories differ substantially more than the first tick: maximum qpos component difference0.135804414749 at tick276/index21, and qvel component difference5.52377474308 at tick275/index12. Masks, selected actions, rewards and score arrays remained identical in this test. These are component differences; qpos/raw aggregate fields contain mixed quantities and should not be summarized as one distance.

The512-step whole-worker benchmark took 9.742656s eager and 8.747245s captured, 1.1138× speed. This includes upload, status, snapshot and protocol overhead, with no rendering.

The original summary's scope sentence incorrectly claimed exact agreement despite its correct failure flag and counts. The original trace/report are preserved. The future driver now reports pass/fail wording conditionally, covered by a regression test.

Keep graph mode disabled. This single comparison does not identify whether the differences arise from capture or existing parallel numerical nondeterminism. No extra GPU run or tolerance relaxation is required for the current release. Match-lifecycle work remains independent.
