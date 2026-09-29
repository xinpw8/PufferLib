# Follow-camera viewer deployment (app revision 10)

Serves the revision-10 app with the follow-camera native binary on loopback port 18775, next to the unchanged revision-9 viewer on 18774.

- `build-worker.sh`: revision-9 build recipe. It reuses the identical build-r6 objects and recompiles only `eval_worker.cpp`; `eval_renderer.h`, `eval_render_request.h` and `eval_worker.cpp` are the only source differences from revision 9. Output sha256 `dba48c455123a4fa7d31c34034102b3756dfd469c26963823eb8711c92b5661f`.
- `deployment_config.py`: `baseline/` holds copies of run-r8 `worker.json`, `server.json` and `identity.json` made on the host at deployment time (not committed). Pins app manifest `e8e52f78…`, the binary, the run-r8 baseline (graph on, one arena) and port 18775. The only worker change is the revision-10 presentation path.
- `prepare_viewer.py`, `start_viewer.py`: prepare a fresh run-r9 and start it paused through `tools/playback/launch_viewer.py`, with the same CPU affinity as 18774 and the four existing viewers as observation-only guards. `start_resource_watch.py`, `resource_watch.py` and `passive.py` are the unchanged files from `passive_support`, copied next to these.
- `disk_cap.py`: the app records every rendered frame with no cap (about 0.55 MB per frame, roughly 20 frames/s while playing). Every 30 s this sums the run directory and stops only its own viewer session above 60 GB or when the filesystem passes 78 %.
- `start_helpers.ps1`: Windows tunnel and NAS mirror for port 18775 and run-r9.
- `bench_render.py`: replays recorded frame requests through a render-only worker.

Measured on 1500 recorded run-r8 frame requests (ticks 0–3716): revision-9 overview mean 16.90 ms (p99 20.20), revision-10 overview mean 16.73 ms (p99 19.76) with PNGs byte-identical to revision 9 for all 1500 frames, revision-10 follow camera mean 11.12 ms (p99 14.42). CPU tests: app 68 run, 67 pass, 1 skipped; deployment 7/7. A live human real-time trial was not run for this revision.
