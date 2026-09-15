# Episode-observation regression

The same CUDA behavioral probe was linked first against preserved version 1
objects, then version 2 objects. Both ran on `spark-4ae3`. Commands, host
identity, compiler output, runtime output, exit code, binary/object hashes and
process timing are retained in the two subdirectories. No model, motion or
checkpoint binaries are included.

Version 1 is the expected negative control. At automatic episode boundaries,
policy observation 186 rose to 2, 3 and 4. The GPU check failed in all three
paths: raw fighter observations, encoded fighter observations and learner
observations. Its executable exited 2.

Version 2 passed. The policy feature remains 1 in all three paths for both
fighters in all four arenas. The separate diagnostic round counter still
advances to 2, 3 and 4. Three neutral episodes each terminated after exactly
1,000 ticks, or 20 simulated seconds. Explicit reset cleared diagnostic
counters. The complete held-yaw, attack-duration, input-discard and contact
replay regressions also passed without changing their version 1 outcomes.

This regression establishes the observation fix and the preserved tested
behavior. It is not a policy-strength test, throughput benchmark or evidence
of authentic REK parity. The frozen-policy intervention that motivated this
fix is documented separately in the policy evaluation evidence.

`summary.json` is a compact extraction from the original outputs. The source
probe SHA-256 is
`19ad29f672d646e507c4059a8d8fbef2fe38259617275bbd4634aa8a56f8b330`.
