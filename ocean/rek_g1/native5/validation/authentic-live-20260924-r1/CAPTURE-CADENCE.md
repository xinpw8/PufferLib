# Capture and live cadence, 2026-09-24

Supporting diagnostic evidence for the live-candidate evaluation. This note adds no evaluation gate and establishes no causal attribution. No runtime, controller, bridge, or game state was changed for this inspection.

## Measured cadence

Rates use the first-to-last native QPC interval in completed `trial/encoder.stdin.jsonl` files. Unity rate uses the corresponding `clock.unity_frame` difference; source rate uses observation count minus one.

| Completed recordings | Source Hz | Unity-frame Hz | Median source interval, ms |
| --- | ---: | ---: | ---: |
| Original no-prior: s601, s602, s603-retry2, s604-retry4, s605-retry3 | 47.95–48.35 | 54.39–55.39 | 18.40–18.64 |
| authentic-s802-retry2 | 24.55 | 24.55 | 38.63 |
| balance8-s901-retry4 | 26.11 | 26.11 | 36.78 |

Every consecutive current observation advanced exactly one Unity frame. The older observations sometimes advanced two frames. The current reduction therefore includes Unity frame cadence; inference-worker throughput alone does not explain it. These are different policies/runs, not a controlled performance comparison.

## Capture versus rendering

- Old recorder: 1280×720 X11 input at 20 fps, no scaling.
- Current recorder: 1920×1080 X11 input at 20 fps, then `scale=1280:720`. Both encode with libx264 ultrafast, two encoder threads, and the same bitrate settings.
- This establishes 2.25× capture-readback pixels plus scaling work. It does **not** establish 2.25× rendering pixels. The old s601 video at 10 s visibly crops the scene: timer near x=960 and opponent HUD outside the 1280-pixel frame. This supports prior 1080p rendering, although the historical window dimensions were not directly measured.
- Current passive `xrandr --current` and `xwininfo -root -tree` confirmed a 1920×1080 framebuffer and REK window. Both launchers still request 1280×720. Stored preferences contain native 1920×1080 and windowed 1280×720 dimensions.
- Both use `STRONGMEM=2`; current launches additionally set `WEAKBARRIER=0`, while original launches leave it at the default. This is a concurrent confound.
- Saved r26/r31/r38/r42 logs identify the same NVIDIA GB10 OpenGL renderer, driver 580.95.05, and Unity D3D11 path. Runtime frame-cap/vsync settings were not measured. Stored native refresh 50 Hz does not establish a fixed 25 Hz cap.

## Sources

Spark stage prefix: `/home/spark-advantage/rek-training/`.

- Original observations and video: `f7-no-kick-prior-live-20260924-r1/noprior-{s601,s602,s603-retry2,s604-retry4,s605-retry3}/trial/encoder.stdin.jsonl`; s601 `media/authentic-rek-policy-fight.mp4`.
- Current observations: `authentic-ppo-live-20260924-r1/authentic-s802-retry2/trial/encoder.stdin.jsonl` and `balance8-live-20260924-r1/balance8-s901-retry4/trial/encoder.stdin.jsonl`.
- Each trial's `media/capture-command.json` records actual FFmpeg arguments. Recorder source lines 69–72 and launcher lines 58/64 expose the differences above.
- Renderer receipts: `/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-r{26,31,38,42}/glx-before.txt` and `unity.log`.

| Stage-relative source | SHA256 |
| --- | --- |
| `f7-no-kick-prior-live-20260924-r1/record_passive_defender.cjs` | `cd4e0b8e0897cdb6772bb313c69284e27d5d45a6c69eecbb2c92ed31736c53b5` |
| `balance8-live-20260924-r1/record_passive_defender.cjs` | `6fc354d170a24ca8e93d8a01c2ffb9520932ef6341904da6ea8e973a2712c928` |
| `f7-no-kick-prior-live-20260924-r1/root-campaign/relaunch.sh` | `f7f05e1670a3f829200a00b876f4d49d2d5d2e87376a2cf002c867bcfd13c544` |
| `balance8-live-20260924-r1/root-campaign/relaunch.sh` | `b46f9788e1f2092864f9dcd5bb8cc25c4f33dfc018b97a4fbbf317b01a5a6041` |

## Proposed controlled follow-up

Compare matched warm rounds with full-frame capture at 20 fps versus 5 fps, keeping S2/W0, checkpoint, policy seed, game resolution, codec, and driver settings fixed. Use the same prespecified active-round interval in both conditions to compare Unity/source Hz and QPC-gap percentiles; retain existing fairness and completion checks. This isolates capture-rate load without changing memory barriers or cropping evidence. It has not been run.
