# Physical contact scoring: raw cvel correction

Implemented and verified 2026-09-21. Contact candidate velocity in `measurement.cu` now copies the three raw linear `cvel` components, matching the recovered native `Robot.GetBodyLinearVelocity` getter. Previously it added `omega cross (xipos - subtree_com[root])`, which produces a different point velocity. The full physical backend already supplies dynamic cvel; this correction does not introduce the compact runtime's kinematic proxy.

The change is limited to contact candidate packing and its derived relative speed. Controller and observation point velocities in `runtime.cu` are untouched, as are standing-height calibration, contact history, scoring arithmetic, and physics/controller execution. `runtime.cu` retains SHA256 `16e099cab4c0bdfa2c6259148099e1c85708668ccc94ef30a93aa69853b0cc84`.

## Verification

`test_measurement_contact_velocity.cu` uses the actual two-fighter model with 12 combinations of root translation, rotation, tilt, and joint rates. CPU MuJoCo calls are restricted to reset and FK/comPos/comVel for the reference; no CPU simulation stepping or controller inference occurs. The production GPU copy is checked against the direct FP32-cast MuJoCo raw triplets for all 2,268 components. Point/COM pointers are null during this copy test, detecting any accidental dependency on the removed shift.

Results:

- All GPU components are bitwise equal to the direct reference.
- Maximum FP32 cast error relative to MuJoCo double values: 1.1894763308e-7 m/s.
- Maximum removed point-shift component in the rotating-body fixtures: 1.1833359435 m/s. This is a quantity difference, not a numerical tolerance.
- Five synthetic contact cases pass through the production contact packer, relative-speed calculation, and full recovered hit detector: angular-only/raw-zero rejects; raw-fast/point-cancelled scores; equal raw velocities reject; target raw motion scores; nonfinite raw velocity invalidates the candidate.

The existing `measurement_probe.cpp` CPU oracle was narrowly updated to raw cvel. Both the preserved original implementation/oracle and corrected implementation/oracle pass all 17 differential samples. These include actual initial calibration and native GPU physics contacts, directed order/speeds, geometry-pair history transactions, selective fall refresh, invalid calibration, and NaN timestep rejection. Both use `mujoco_cuda`, two arenas, and the existing native physics objects. Linked wrappers abort if that regression attempts CPU `mj_step`, `mj_forward`, or `mj_kinematics`.

GPU verification ran from `2026-09-21T01:32:56.395540233Z` to `2026-09-21T01:32:58.571059077Z`, exit 0. GPU ownership was then released. There was no training or live client interaction.

## Frozen sources and artifacts

| Artifact | SHA256 |
|---|---|
| Original `measurement.cu` | `1c50f74d8669a02f02c9caa40ea8450230931a579719435382d81fa99658e4e3` |
| Corrected `measurement.cu` | `d52a61565386140053c96356bb857fa512831fc87817bf8d5046a47c33eaba23` |
| Corrected `measurement_probe.cpp` | `ede7960a086f529d60778b4603ffba0006f3c2964164fe0405af149522e1f7c3` |
| New fixture source | `7768ba5418596d923bd7c332d52f0999d93d3091423aea0cff281820e43f0781` |
| Corrected measurement object | `76b5d62b664e01b6c7f9a9e021b86de673833e417ed2d356a2476778a6a7e784` |
| Direct-reference/scoring executable | `43ad3688c6c2390748661c6e61e775786d86f638e58a4aa54eac65a65b0afc00` |
| Original regression executable | `e8396888953e503fcd4986b51f9cbeca35c010b164e983038b211a5161024d8a` |
| Corrected regression executable | `27ca9741cf2eb23f14680650cdc1e9a287814b72b3d0b5f302713d516c4e7ae3` |
| Direct-reference/scoring result | `5dc4e0777bf3f29279abc9cb94162171478aace7282ef02004e01ea25d79ebfe` |

Private stage: `C:/rekagent/work/consistent-fighter-20260919-r1/physical-contact-cvel-r1`. Original source copies are in `baseline`. The complete Spark stage is mirrored in `spark-results`, including source, build outputs, original failed-link diagnostics, corrected link commands, stdout/stderr, and verification results. All 22 locally applicable artifact hashes in the build/verification manifests passed readback.

Spark stage: `/home/spark-advantage/rek-training/physical-contact-cvel-20260921-r1`. `build.sh` compiled the new measurement and reference fixture against pinned MuJoCo 3.7.0 and CUDA sm_121. Its first regression link omitted the abort-wrapper definitions and failed without running a probe. `finish-build.sh` added those existing-style definitions and completed both regression links; the failure remains preserved. `run.sh` contains the exact bounded GPU commands and environment. Each executable was limited to 60 s. `link-inputs.sha256` records every reused native object and the MuJoCo library; `build/hashes.txt` and `verification-r1/hashes.txt` record produced artifacts.

The model SHA256 remains `6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa`. The source/reference definitions are documented in the [contact velocity contract](../contact-velocity-contract-20260920/README.md). The [controller coefficient comparison](../controller-coefficient-comparison-20260921/README.md) establishes identity with the inspected old-copy payloads, not current service identity.

## Limits

This proves the contact measurement now uses the recovered velocity quantity and preserves the tested measurement transactions. Synthetic score cases do not establish actual contacts or fighting strength. The known pelvis-to-floor versus pelvis-to-lowest-foot standing calibration discrepancy remains unchanged. Current service configuration, dynamics parity, and authentic performance remain separate questions.

## Private NAS preservation

Both `controller-tensor-comparison-r1` and `physical-contact-cvel-r1`, including the frozen controller state-constant comparison, are archived at `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\native-physical-contact-cvel-r1`. The archive contains 830 files and 134,521,628 bytes. Every source-before, copied-file readback, and source-after SHA256 matched; source files and existing archives were preserved. `archive-manifest.json` SHA256: `c3c799922dab52acc3977baf948dcc86d5eae90e9fcec7bfc30f5c509e958c6a`.
