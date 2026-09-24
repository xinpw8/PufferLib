# Attack readiness: measured held-translation restriction

Scope: all 787 worker decision rows from the earlier closed `human-attackbc-live-20260924-r1/humanbc-s1101`, checkpoint `5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4`, under the original 250 ms bridge. This is separate from the new timing cohort. No game connection, GPU job, policy change, or runtime edit was used for this audit.

## Measured mask accounting

Worker rows were joined to ready encoder outputs by sequence and to recorded source states by exact QPC. The following groups are disjoint and cover every row:

| State | Rows | Fraction | Final attack support |
| --- | ---: | ---: | --- |
| Held translation | 712 | 90.47% | None; already blocked by bridge |
| Estimated attack busy | 52 | 6.61% | None; encoder duration projection suppresses source-legal attacks |
| Pending attack transport | 6 | 0.76% | None; source pending_move=true |
| Attack ready | 17 | 2.16% | All 17 attack categories |

The encoder added zero translation exclusions beyond the bridge mask in these rows. All six pending-transport rows had every native settlement flag true and were immediately after an attack request. There was no residual nonbusy, released state blocked by an unsettled flag in this sample. The policy selected six attacks and selected release/Q/E only ten times while translation was held.

| Held category | Rows | Rows allowing attacks |
| --- | ---: | ---: |
| 1, release | 27 | 8 |
| 2 | 14 | 0 |
| 3 | 10 | 0 |
| 4 | 22 | 0 |
| 5 | 11 | 0 |
| 6, Q | 20 | 6 |
| 7, E | 28 | 3 |
| 8 | 19 | 0 |
| 9 | 35 | 0 |
| 10 | 81 | 0 |
| 11 | 155 | 0 |
| 12 | 8 | 0 |
| 13 | 227 | 0 |
| 14 | 14 | 0 |
| 15 | 116 | 0 |

These are retained held categories at the observation, not newly selected action counts. Category 0 preserves the prior held command.

## Training/live comparison

The suggested missing compact-training translation gate was not found. In the frozen F7 `fast_runtime.cu`, lines 121-125 classify held categories 2-5 and 8-15 as translating and require `!translating` for `settled()`. Line 947 requires settlement for attack legality. Attack dispatch at lines 298-313 tests settlement before any automatic neutralization. Therefore an attack cannot bypass held ASDW by auto-clearing it. Q/E remain compatible with attack readiness after translation settles.

This mask was wired into the compiled F7 trainer: disassembly of `build-calib2/pufferl.o` shows `create_pufferl` calling `rek_native5_bind_action_mask` at offset `0x13698`, beyond the standalone binder definition. The repository's `puffer_env.cu:298` and `pufferlib5_action_mask.patch` document that binding.

Live `G1PolicyStreamContract.cs:129` requires no held ASDW and zero translation command. `Plugin.G1PolicyStream.cs:282-292` then checks native transition settlement. The shipped `encode_live.cpp:244-248` preserves the source mask and additionally suppresses attacks while held translation or projected busy is present. Compact physical velocity settlement and live native transition settlement are different mechanisms; their timing equivalence is not established by this audit.

Movement occupancy, rather than an absent compact held-translation gate, dominates this sample's attack restriction. Adding a duplicate gate or automatic live release is unsupported. The human-BC update deliberately masks held/settled/busy inputs 176-183 and assigns movement/release rows zero supervised weight; it does not directly teach release-then-attack timing. A future learning change needs measured sequence data and the unchanged legal rules, not relabeling blocked requests as executed attacks.

## Source identities

Frozen F7 source: `C:\rekagent\work\attack-gate-handoff-20260923-r1\source\fast_runtime.cu`, SHA256 `ee6b31be666b65f5aa116954727381b2c57d78571321993a3c17d72ac72e15d2`.

Compiled F7 stage: `/home/spark-advantage/rek-training/attack-gate-sweep-20260921-r1/build-calib2`:

- `pufferl.o`: `360f50d65bfe1b2be5aabfddbbc597c3310ba7da52adb7d40e622d385eee6377`.
- `puffer-rek-native5`: `1dd9df4b5fc1fb0191f31a056f21eaf09b562cef50bfc3dff019efabb375834d`.

Shipped encoder source: `C:\rekagent\work\balance8-authentic-20260924-r1\source\encode_live.cpp`, SHA256 `54747e5c44995341503be21fa004eb5734c7ae552ef10294fd206575cb1baeee`.

Inspected bridge source under `ocean/rek/evidence/windows/RekUiBridgeAgent`:

- `G1PolicyStreamContract.cs`: `0335b0a99cb2a09afffa8c18918181859246b7a204c47bf0c9a89bc9db6eb5df`.
- `Plugin.G1PolicyStream.cs`: `bd602ea1d032bfe078deb1f5223d6551e12afbf0c113d2922102872418357adb5`.

Data files relative to the earlier attempt's `trial/`:

- `worker.stdin.jsonl`: `5403b918106442ed5c2addaabdd0e1d7bfa0f0b1a5fc61a8dcb4255cf88d0a00`.
- `encoder.stdout.jsonl`: `5cfd85d4025a39ff4350516a8c6d73adf65af8bb5f09cd4897f659d2e5fa1812`.
- `encoder.stdin.jsonl`: `e58cb777119473bb1ee08bc0c9b95728131d3357cfa01bd981420f4a5237f522`.
