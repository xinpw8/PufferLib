# F7 interrupt projection candidate

The copied native encoder adds default-off `--interrupt-on-opponent-score`, requiring the existing request-duration mode. A positive opponent `round.clean_hits` delta while a dispatched request is projected busy cancels that request's busy state. Cancellation persists for that request QPC; a later request can become busy normally. First samples, new rounds and explicit resets retain derivative warmup. Own-score changes do not cancel. Observation length stays 223. Source mask restrictions and held-translation restrictions remain intersected with the encoder mask.

This aligns the live projection with F7's native training assumption `REK_FAST_INTERRUPT_ON_HIT=1`. It does not establish true server interruption. The pinned encoder and repository remain unchanged.

Exact invocation on Spark:

```bash
/home/spark-advantage/rek-training/f7-live-interrupt-encoder-20260924-r1/build/encode-live \
  --model /home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact/model.two_fighter_arena.xml \
  --projection client_pose_projection_v1 \
  --busy-projection dispatched_request_v4_duration \
  --interrupt-on-opponent-score
```

Preserve checkpoint, worker, schema and other trial settings when isolating this change. Omitting the final flag preserves the pinned output behavior.

Verification:

- Native `build_encoder.sh` passed. The actual `Encoder::process` test suite passed 963 assertions, plus 145 hinge-projection checks. Tests cover first sample, positive/no/own/simultaneous score deltas, both player slots, continuing canceled requests, new requests, reset/new-round handling, unsent/expired requests, source-mask preservation, held translation and invalid mode selection.
- Full completed s402 trace replay with the option off produced output byte-identical to the pinned encoder. Both complete output SHA256 values were `2d827aadbca78afa18c7b52b0ee9693d12dbd40bb66e442324eea308658363fd`.
- Enabled replay changed 288 observation rows and masks. Changed observation columns were only 178, 179, 180, 182 and 183 in this trace. All observations retained 223 fields; no output mask enabled a source-disabled action. At sequence 10184 the original remained busy until sequence 10322, 2.866 s later; the candidate cleared projected busy at 10184.
- No bridge operation, input or GPU workload was run. Replay projects recorded poses and does not step a simulator.

Hashes:

| Artifact | SHA256 |
| --- | --- |
| Pinned source | `d17e93a99d254c295df1f113884f877d20d641d3a1d855b9b88c4047f630b755` |
| Candidate source | `1d99e5cc1d79e8a6ef85292236a5ae320b8b4b6adc2b93d0ba04a1a9d8ccec94` |
| Candidate executable | `fbed51f422f6d80820a05d06b59000defaaa8496aaa16b01884edba19ac42e39` |

The patch applies to the pinned source root at `ocean/rek_g1/native5/live_transfer/`. Build using its unchanged `build_encoder.sh NEW_BUILD_DIRECTORY`, then run `NEW_BUILD_DIRECTORY/encoder-test MODEL_XML`. Candidate source, copied dependencies, build log, executable and enabled replay are preserved under `/home/spark-advantage/rek-training/f7-live-interrupt-encoder-20260924-r1`.

Evidence supporting this bounded experiment:

The pinned encoder's lines 193-194 canceled duration only on a fallen flag or fall-counter rise. Its lines 216-220 and 236 derived busy features and mask restrictions from that duration. F7 training's `fast_runtime.cu` lines 707 and 819 clear attack duration on opponent awarded points; lines 896 and 947 then expose ready state and reopen allowed actions. In the saved s162 trace, five opponent score increases occurred during projected busy, retaining restrictions for 0.126 to 2.190 s. Completed current s401/s402 traces have 13 such observations; their actual encoder and worker masks stayed `[0,1,6,7]` until duration expiry, including 2.866 s after the s402 sequence 10184 score increment.

The existing snapshots cannot directly establish attack interruption: every one of the 11,543 s401/s402 source rows has `input.action_busy=null`, `input.punching=false`, `runner.current_move_index=null`, `runner.current_motion_name=null`, `runner.motion_frame_index=0`, and `runner.is_done=false`. `input.action_busy` would directly supply controller busy where available; a recognized current-move identity plus playback state could identify active motion. Neither is available on this visual-only client. Bone transforms measure visible motion but do not by themselves identify accepted attack/busy state. Request timestamps and returned-send flags measure dispatch, and the source mask explicitly does not establish server readiness.

Limits retained deliberately: the score delta spans two observations, so a simultaneous new request and score cannot be ordered within that interval. Existing resets, including a source gap over 250 ms, clear canceled-request state; after warmup a still-recent old request can project busy again. Score-counter increments do not expose authoritative limb attribution or interruption cause. This candidate tests temporal projection consistency only.
