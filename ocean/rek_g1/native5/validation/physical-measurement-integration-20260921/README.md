# Full physical measurement integration

The unchanged 777-tick physical schedule passed with both measurement corrections integrated on 2026-09-21. A second, identical updated-build run also passed. Both used native MuJoCo CUDA physics and SONIC, reported `cpu_physics_calls=0` and `ppo_updates=0`, and explicitly reported `authentic_parity=false`. No training or live client interaction occurred.

This integrates [raw linear cvel contact velocity](../physical-contact-cvel-20260921/README.md) and [lowest-foot standing-height calibration](../physical-standing-height-20260921/README.md). The latter changes the actual initial idle denominator from 0.792999975 m to 0.691509724 m on both fighters. Its existing floor-based numerator remains approximate. The separate measurement fixtures establish those quantities; this short integrated schedule does not exercise a scored contact or countout.

## Build and unchanged schedule

`runtime.cu` and `measurement.cu` were recompiled together against the corrected `measurement.cuh`. This is required because the appended foot geometry/radius metadata changes the measurement class size. Other pinned physical/controller/motion objects were reused, with every object hash retained. Five shared headers were byte-identical; `runtime_api.h` retained identical shared types and existing prototypes, with only comments and an unused compact autoreset declaration changed.

Archived and rebuilt `runtime.cu` are byte-identical, SHA256 `16e099cab4c0bdfa2c6259148099e1c85708668ccc94ef30a93aa69853b0cc84`. Runtime compiler flags match the preserved original `build_native.sh`: C++17, O2, sm_121, `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`, host PIC and `-ffp-contract=off`, native MuJoCo GPU enabled. Historical evidence consists of the saved script/invocation, build logs and object manifest; it is not an independent shell-expanded compiler trace.

The original probe source and runner were unchanged: four arenas, SONIC batch 8, reset-separated yaw conditions -1/0/+1, 259 semantic ticks per condition, 50 Hz and ten physical substeps per tick. Each condition has 50 idle ticks and 25 held-yaw lead ticks. Move 3/category 23 and move 10/category 26 each have an attack lane and a no-attack control. Their active windows remain 45 and 134 ticks. Masks are enforced without fallback. CPU `mj_step`, `mj_forward` and `mj_kinematics` wrappers abort if called by this executable.

The model, motion assets, controller graphs, kernel catalog and conditional PTX were unchanged. The model SHA256 remains `6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa`.

## Results and repeat variability

Each run produced 3,120 recorded arena snapshots and 12 attack/control summaries. The archived baseline and both updated runs have zero points, falls, terminal/reset samples and failure bits. All three completed without a confounding reset. Updated process wall times were **13.20 s** and **13.51 s**, including startup and logged snapshots. These are probe timings, not training throughput.

| Recorded-field comparison | Changed snapshots | Maximum root-coordinate difference | Maximum tilt difference |
|---|---:|---:|---:|
| Archived baseline versus updated run 1 | 3,108 / 3,120 | 0.393494 m | 13.856838 degrees |
| Updated run 1 versus identical-build run 2 | 3,108 / 3,120 | 0.354507 m | 13.539389 degrees |

Both comparisons first differ at tick 1; their 12 initial snapshots are identical. Maximum recorded root-qvel component differences are respectively 9.035361 and 10.793591. That six-component field mixes linear and angular units, so these maxima are not a single physical speed. Comparisons cover only recorded fields, without claiming equality or coverage of the complete unreported state.

Attack-window net XY displacement illustrates the variability:

| Yaw condition | Move | Archived baseline (m) | Updated run 1 (m) | Updated run 2 (m) |
|---|---:|---:|---:|---:|
| -1 | 3 | 0.032897 | 0.032325 | 0.030817 |
| 0 | 3 | 0.057256 | 0.055522 | 0.057120 |
| +1 | 3 | 0.083295 | 0.077663 | 0.085757 |
| -1 | 10 | 1.224072 | 1.210215 | 1.456038 |
| 0 | 10 | 0.483539 | 0.597788 | 0.659833 |
| +1 | 10 | 0.986965 | 1.233252 | 0.968050 |

The unchanged-build repeat establishes run-to-run variability, but its cause remains unknown. This comparison cannot attribute motion differences or improvement to the measurement corrections. Attack/control arenas also have measured pre-action state differences, retained in the comparison artifacts; they are not identical-state causal interventions. Two updated runs do not estimate a reliable variability distribution. No additional experiments were performed.

## Reproduction and preserved evidence

Private Spark stage: `/home/spark-advantage/rek-training/physical-measurement-integration-20260921-r1`. Windows mirror: `C:/rekagent/work/consistent-fighter-20260919-r1/physical-measurement-integration-r1`. Exact commands, compiler output, reused-object hashes, source copies and results are preserved. The first preparation stopped at an overly broad header byte-equality check before compilation; `source-r1`, the stopped script and its diagnostics remain intact. The reviewed final build uses fresh `source-r2` and `build-r2`.

```bash
stage=/home/spark-advantage/rek-training/physical-measurement-integration-20260921-r1
bash "$stage/build-r2.sh" 31ac31944b0abb608dfd9ed16168803b2eabdd33e3d24a04200ab042f3f76b38 bf736deae56f8a472587ab9a8007bc756c6782f854c35cbfc0aa0e7f79a6b02f
bash "$stage/run.sh" "$stage/build-r2" "$stage/run-r1"
bash "$stage/run.sh" "$stage/build-r2" "$stage/run-r2"
```

These scripts require fresh output paths. Each run is bounded to 120 s. Actual intervals were `01:51:02.662133206Z` to `01:51:15.869643533Z` and `01:56:34.625892442Z` to `01:56:48.142855911Z`, all on 2026-09-21. Both exited 0 and GPU ownership was released afterward.

| Artifact | SHA256 |
|---|---|
| Corrected `measurement.cu` | `31ac31944b0abb608dfd9ed16168803b2eabdd33e3d24a04200ab042f3f76b38` |
| Corrected `measurement.cuh` | `bf736deae56f8a472587ab9a8007bc756c6782f854c35cbfc0aa0e7f79a6b02f` |
| Rebuilt runtime object | `cfda968b970368412123bcfd77b87cf30b39ab35fab97ecc8a022c0168d4a9f4` |
| Rebuilt measurement object | `7b29e3a37f4ae1b519d5522a67bc161d820a771bcc70de80c617204ab68a4864` |
| Unchanged schedule source | `7cd67c58b9958488ecc5b6c7142f5194646d23bc16b5bb7e834894b38b1a5a9f` |
| Archived baseline executable | `adf666e5b68f837e984cca65ceecf4b396d99eb42f923bdfec1ac72b69ffedc8` |
| Updated executable, both runs | `dec9bb06dafd62c41831619b1ef39b96a53a45f26aa58fed366e956ecbb7ae0c` |
| Archived baseline stdout | `fba722e5c447cd2d1d4852bf9755edc046c7f1bd582c34f5cabcfdc5d5e8696d` |
| Updated run 1 stdout | `9d904b55485fcd2f21d549530bacce8723d8651bb9c031e7f3032540373edcaf` |
| Updated run 2 stdout | `feb36450fe92be94fd4cd371c3b549f265974812a17ddd1a91c23bce941b8b76` |
| Baseline comparison JSON | `52223eb59763e525b143f38a789c890ae061c30643d0c960e16ed5cbfe543ce6` |
| Identical-build repeat comparison JSON | `69ec0f23f198b10ef0fcbd29507926addfe90c7c874d45d7c6227c48f39ea699` |

Private `compare.cjs` performs only file-based comparisons; its baseline-versus-itself self-check reports zero differences. All 1,573 files in the finalized Spark inventory matched local mirror SHA256 readback. Its manifest SHA256 is `2fcf93564a51556f66718349e5f3d2494e626116707084bd44af12a22bea2d99`. Proprietary assets and source dumps are absent from the public report.

The complete private stage is archived at `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\native-physical-measurement-integration-r1`: 1,595 files, 75,090,008 bytes. Every source-before, NAS readback and source-after SHA256 matched. Both runs, the archived baseline, stopped preparation and source files were preserved. Archive manifest SHA256: `5a141ea7c375a96a161f2b6572b9991281655a1ad6885167a4f45d3718bbba1f`.
