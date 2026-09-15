# V3 human evaluator deployment

The isolated V3 viewer was tested at `http://127.0.0.1:18770/` before replacing
the paused V2 viewer at `http://127.0.0.1:18769/`. Both execute on `spark-4ae3`.
The Windows browser reaches these loopback ports through WSL SSH forwarding.
No Windows application received synthetic input.

The isolated browser test loaded a 640-pixel-wide native frame, accepted the
on-screen I side-kick input, and advanced from tick 15 to tick 610. The timer
showed 4:48 of its 300-second round. Both falls counters remained zero, and
there were no input or browser console errors. This is a UI smoke test; the
paired-contact regression in `v3-paired-i.json` is the test of the reset fix.

The replacement live viewer was verified at tick 0, paused, with a 5:00 timer,
score 0:0, zero falls, and a decoded 640-pixel frame. The human controls orange;
the opponent is `compact-v3-33m-bf16-sampled`. The page shows the V3 points-only
limitation and the exact new training/frozen-evaluation report. Old league
rankings were not transferred. Services on 18766 and 18768 were preserved.
The temporary 18770 server and test tab were closed after verification.

## Identities

- Worker: `/home/spark-advantage/rek-training/semantic-fast-20260914-v1/eval-build-v3/rek-eval-worker`
- Worker SHA-256: `a750cb7fd153074fbbd58cfd7b501d2eec8349948b371c26da8903c51a58b4ec`
- Runtime object SHA-256: `88c97ba250b2b21aab56a0749bbc7e0fef74dd930929c60f8620ac5bf696470a`
- Environment configuration identity: `5b00c572f63a69dea8c2a1eef186a22891b4a9cb980872a092cb21ac64adbc07`
- Checkpoint SHA-256: `e209edd0d1301170c5253f622f5f11aabccb57daf7fd4117983942bccb989fc6`
- Live configuration: `/home/spark-advantage/rek-training/semantic-fast-20260914-v1/human-viewer-v3-20260915/live-run/server.json`
- New Node PID at deployment: `145654`; replaced verified V2 Node PID `141173`.

All 78 Node input, viewer, session, and league tests passed. Native GPU input
tests and the position regression passed separately. Physical knockdowns and
action-to-idle pose blending remain unmodeled; this deployment does not claim
to repair those separate limitations or establish authentic REK parity.
