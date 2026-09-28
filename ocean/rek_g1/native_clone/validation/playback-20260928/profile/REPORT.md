# Native control-step timing

The isolated profile completed 256 control ticks and exited 0. The identity-bound live viewer stayed paused and healthy. No rendering, graph replay, deployment or live input occurred.

The disjoint sequential host scopes average 0.0185 ms for input/policy enqueue, 2.7574 ms for runtime enqueue, 19.4358 ms for the status call including outstanding runtime execution, and 0.3537 ms for snapshot plus feedback. These total 22.5654 ms before small event bookkeeping and response construction. The status number is primarily waiting for simulation work.

The overlapping CUDA stream timeline reports 20.8557 ms runtime and 1.1995 ms after-runtime validation. Runtime median is 19.7793 ms. One initial large event reaches 169.6682 ms; no sample was discarded. These event intervals include possible host enqueue gaps and cannot be added to host scopes.

State JSON construction averages 0.0843 ms, including 0.0233 ms for WillConnect. Serialization averages 0.2435 ms and stdout 0.0634 ms across 258 outputs, including startup and snapshot. These are not the dominant costs. The profiled RPC rate is 42.37 ticks/s; use the separate uninstrumented benchmark for throughput.

Status packet packing is not justified as the primary remedy. Even removing the entire measured post-runtime validation and readback interval would fall short of the approximately 5 ms app gap, and required failure checks cannot be removed. The next useful diagnostic is subdivision of the native runtime timeline into motor inference and physics substeps without changing arithmetic or checks.
