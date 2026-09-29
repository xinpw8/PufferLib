# Passive viewer resources

`resource_watch.py` is a Linux-only observer. Bind it to the viewer's recorded Node PID and `/proc/PID/stat` start ticks, and supply the exact native executable path as returned by `/proc/CHILD/exe`. It discovers only direct children with that executable. It never reads process command lines or environments, sends input, signals viewer processes, or changes GPU settings.

Example invocation, substituting the new viewer's verified identity and paths:

```sh
python3 resource_watch.py --pid NODE_PID --start-ticks NODE_START_TICKS \
  --native-exe /absolute/path/to/rek-native-clone \
  --output /absolute/new/run/resource.jsonl \
  --stop-file /absolute/new/run/STOP_RESOURCE_WATCH
```

The output must be fresh. The observer exits when its own STOP marker appears or the parent disappears, becomes a zombie, or changes identity. It only closes its own sampler. A timed-out `nvidia-smi` subprocess is an owned diagnostic query; the observer never signals a game or viewer.

Process CPU, resident/virtual memory, thread count, last CPU, and I/O counters are sampled once per second. The first observation of each PID/start identity reports cumulative counters and null deltas. CPU percent uses one logical core as 100%, so multithreaded values may exceed 100%. Missing I/O remains null. UTC and monotonic time are both recorded.

One background job queries global GPU utilization, clocks, temperature, power and memory, plus numeric per-GPU-application `pmon` counters, at most every five seconds. Each of its two sequential queries has a two-second timeout. Only one job may be pending, and process sampling never waits for it. Cached GPU sample age and in-flight status are explicit. GPU PID rows are system-wide, so they can expose competing GPU activity; command-name columns are discarded. Unsupported or nonfinite measurements remain null, with errors reported. `pmon` SM utilization is sampled activity, not an exclusive share of GPU time.

JSONL records are line-flushed for incremental passive mirroring. Abrupt machine/process failure can leave an incomplete final line or no `stopped` record; neither is a successful shutdown receipt. There is no automatic application repair or restart.

`python3 -m unittest -v test_resource_watch.py` exercises fake `/proc`, fake query execution, and fake asynchronous jobs. It does not query a real GPU or control an application.
