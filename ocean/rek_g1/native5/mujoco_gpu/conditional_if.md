# GPU reset-mask guard

`DeviceIf` skips a reset/refresh body when its GPU arena mask is entirely zero.
A single 256-thread block scans the mask and sets a CUDA conditional handle.
No mask or condition is copied to the host.

During parent graph capture, the helper inserts its conditional node directly
into that graph. The callback records its operations into the IF body using
`cuStreamBeginCaptureToGraph`. Outside capture, the helper records one graph
per mask pointer and count, then reuses that graph. The callback's addresses
and operations must stay fixed for each key. Calls on one instance must be
serialized together with the simulation state it guards.

The 2026-09-14 Spark probe passed 64 cases, covering masks of 1, 33, 512 and
1,025 bytes, zero masks, first/last active bytes, nonbinary nonzero bytes,
out-of-range bytes, direct parent capture, standalone graph reuse and launches
on two streams. Eight expected graph-body captures occurred; execution did
not recapture the body. Exact commands and results are in
`validation/conditional-if-20260914/`.

This validates the guard. The runtime's actual reset behavior and training
speed require separate integration measurements. The probe does not report
training SPS.
