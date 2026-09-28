# Direct-command API regression and source review

Source review found a real override defect: `choose_actions` still validated a row's categorical input before its enabled direct command replaced that value. A valid direct command could therefore acquire sticky failure bit 16 from an ignored NaN/out-of-range category. The owner fixed selection to bypass categorical reads and checks for each enabled direct row independently. This also prevents ignored masked placeholders from inflating invalid-action counts.

Read-only review of the revised runtime, worker and ordered-reset integration found no additional blocking issue. Command parsing validates before changing ownership mode. A batched request sends its move/cancel pulse once, while continuous velocity persists across ticks. Feedback is collected after every tick. Cold reset clears dispatch ownership; counted/round body resets preserve it and use the actual reset body quaternion. Current G1 has no get-up clips, so direct input recovery is false while motor suspension remains a separate gate. Callback ordering is an explicit hypothesis, with both alternatives available, rather than proven Unity subscription order.

`direct_api_test.cpp` links the real runtime and executes one control tick per process. Six scenarios test ignored NaN/99 categories under direct side 0 and side 1, then ensure invalid categories still produce failure bit 16 on the other categorical side. Positive cases require finite physical state, uncategorized action sentinel -1, and healthy direct-command feedback. Negative cases require the precise categorical failure bit and the expected sticky close status. All other arenas/rows use ordinary neutral categories. This bypasses JSON input validation intentionally to exercise the public runtime API contract directly.

The harness requires the root's completed build and the same full-physics eight-fighter motor configuration. It reuses every pinned build object except `eval_worker.o`, replacing the entry point only. `build.sh` does not execute CUDA. `run.py` is the explicit GPU entry point and runs six fresh processes, one tick each, with independent stdout/stderr/exit receipts and a 120-second process limit. It preserves the root's configured environment and never starts a UI or original REK process. Python syntax and shell syntax were checked; compilation and GPU execution are delegated to the root agent and must be reported from their actual receipts.

From `/home/spark-advantage/rek-training/rek-native-clone-20260927-r1`:

```sh
bash original-history/review-tests/build.sh
python3 original-history/review-tests/run.py --config run/worker.json --environment config/environment.json --output original-history/review-tests/run-r1
```

This supplemental folder was created after the immutable 44-file original-history publication. It does not rewrite that publication or its oracle report.
