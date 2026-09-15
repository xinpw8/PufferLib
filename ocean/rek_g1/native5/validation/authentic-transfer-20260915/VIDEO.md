# Fresh recorded authentic-client trial

The earlier `trial-r4` (15:32 clean hits) was not recorded. Its logs and still
images must not be presented as video evidence.

On 2026-09-15, `trial-video-r2` reran the exact V4 checkpoint with bridge 0.4.9
and the existing native encoder/CUDA worker. Recording began after the native
StartRound acknowledgment and before the policy stream started.

- Capture: actual X11 display `:98`, 1280 × 720, 30 fps, VP8/WebM.
- Duration: 134.592719 seconds, including countdown and post-round result.
- Actual result: policy slot 0 lost 6:16 clean hits to Sparring Bot 1.
- 3,631 policy decisions; 101 native move sends.
- Recorder exit 0; live driver exit 0; complete video decode exit 0.
- Browser playback verified at the middle and final result screen.
- No generated frames, added video overlay or Windows input injection.

SHA-256:

```
a0a4334f1537f60e2002e7bf0ae633f27d5b99f1de109fc51e7c99866cab50fc
```

Verified physical-server deliverable:

```
\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-15\video-proof-r1\authentic-rek-policy-fight.webm
```

The same directory contains `recording-manifest.json` and
`measured-report.json`. Complete raw capture/trial logs remain under
`/home/spark-advantage/codexrook-runtime/live-transfer-20260915/video-proof-r2`
and `trial-video-r2` on Spark. The video is intentionally not stored in Git.

`record_live_transfer.cjs` records a new trial alongside the existing driver.
`serve_fight_video.cjs` exposes only the fixed video/evidence files on loopback
with range requests for browser seeking. The local review page is
`http://127.0.0.1:18771/` while that helper is running.
