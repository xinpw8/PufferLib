# Physical standing-height calibration

The physical backend now calibrates pelvis height against the lowest foot sphere, matching the recovered native standing-height quantity for the pinned G1 model. At the actual runtime initial idle pose, the denominator is **0.691509724 m**, versus **0.792999975 m** under the previous pelvis-to-floor calculation. The difference is **0.101490251 m** on both fighters. This is a calibration correction, without a fighting-strength or full-body dynamics parity claim.

## Recovered contract and scope

Pinned `Robot.txt` SHA256 `4f61233092542b15773e49d8404790a8ed89352d3b656fa41b75bab9c8283ded` contains `CacheFootBodies` at line 52726, `TryGetBodyLowestPoint` at 16237, `MeasureStandingHeight` at 53385, and `get_PelvisHeightRatio` at 28781. Foot bodies come from `BodyPartTag.IsFoot` components, including inactive children, deduplicated by body. Only geometry directly attached to those bodies is scanned; collision masks do not filter that scan. Each lookup refreshes MuJoCo kinematics.

Native geometry support is evaluated in the downward direction, with double arithmetic and a float Unity point returned for each foot. Sphere support uses radius; boxes use oriented support vertices; capsules and cylinders share an axial-endpoint-plus-full-radius formula. Other geometry types use `geom_rbound` spherical fallback. These generic choices are documented, not newly implemented here.

The pinned model has exactly four spheres on each ankle-roll foot body, eight per fighter. Each radius is `0.004999999888241291 m`. The implementation asserts that body ownership/count and sphere/radius contract. Its calculation is:

```text
bottom[g] = float(double(device_geom_center_z[g]) - model_radius[g])
standing = float(pelvis_z - min(bottom[0..7]))
height_ratio = standing <= 0.0001f ? 1.0f : (pelvis_z - existing_floor_height) / standing
```

Casting finite support points before their minimum gives the same height as casting each foot's minimum. The native ratio threshold `0.0001f` and fallback `1.0f` were confirmed by static PE constant reads from installed `GameAssembly.dll`, SHA256 `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`. No installed DLL code ran.

Native missing pelvis/foot geometry leaves standing height zero and disables height detection. This patch retains the physical backend's existing strict missing-model, nonfinite, invalid-root and global-invalid-row handling. Finite zero, negative and small positive denominators use the native disabled-height branch. The denominator is floor-independent; the complete calibration function still validates its existing supported horizontal floor.

The numerator remains an approximation: native height ratio uses cached floor-contact height or a downward MuJoCo ray, while this backend still uses the pinned horizontal floor geometry. GPU centers are already FP32, whereas native geometry calculations start from double MuJoCo state. The float support/subtraction order is reproduced for the available device inputs, without claiming cross-backend bitwise FK parity.

## Measured verification

The fixture reconstructs the exact initial pose preparation in `runtime.cu`: float model-root state plus the first actual idle-clip joint frame, clipped to the float joint limits. All 72 values match the exported initial pose; 12 joints are clipped. It does not silently substitute `model_qpos0`.

| CPU FK case | Native foot denominator, fighter 0 / 1 | Previous floor denominator |
|---|---|---|
| Actual initial idle | 0.691509724 / 0.691509724 m | 0.792999975 m |
| Whole body translated upward 1 m | 0.691509724 / 0.691509724 m | 1.79299998 m |
| Actual idle with world roll 0.3 rad | 0.712336063 / 0.694361866 m | 0.792999975 m |

For these FK cases, maximum difference between the double-input native reference and FP32-input support calculation was zero. That finite case result does not guarantee zero error on arbitrary poses.

The production CUDA fixture passes **520 assertions**, including **12 GPU cases**: actual idle, translation, one lifted foot, paired geometry/radius reorder, zero/negative/small-positive disabled heights, nonfinite geometry, invalid quaternion, nonfinite pelvis, tilted FK, and exact/next-float threshold boundaries. Three constructor cases reject a non-sphere, nonfinite radius, and changed foot-body geometry membership. No synthetic case is presented as a physical trajectory.

The preserved raw-cvel baseline and corrected measurement regression each pass all **17 differential samples**, including actual native GPU physics contact processing, directed ordering/speeds, history transactions, selective fall refresh and invalid calibration. That regression aborts on CPU `mj_step`, `mj_forward` or `mj_kinematics`; the separate FK reference deliberately uses CPU kinematics and no simulation stepping or controller inference.

Final GPU verification ran `2026-09-21T01:47:32.004531785Z` through `01:47:34.164607142Z`, exit 0. GPU ownership was then released. A separate full-runtime integration rebuild is required because the appended measurement metadata changes the C++ class size; the measurement-only regression is not a full-runtime ABI claim.

## Preserved attempt and reproduction

The first fixture run stopped at the existing model-contract check before calibration kernels: it omitted the production host preparation that sets timestep to 0.002 s. The final fixture checks the exported runtime timestep and applies that same assignment. `verification-r1`, its source and binaries remain preserved. No production condition or numerical threshold was relaxed.

Private Spark stage: `/home/spark-advantage/rek-training/physical-standing-height-20260921-r1`. `build.sh` creates the original source/build and measurement regression; `finish-build-r2.sh` builds the corrected fixture in fresh `source-r2`/`build-r2`; `run-r2.sh` records the three bounded verification commands. Each GPU executable has a 60 s limit. Full compiler commands, reused object hashes, dependency/model/asset identities, stdout/stderr and times are retained.

```bash
bash /home/spark-advantage/rek-training/physical-standing-height-20260921-r1/build.sh
bash /home/spark-advantage/rek-training/physical-standing-height-20260921-r1/finish-build-r2.sh
bash /home/spark-advantage/rek-training/physical-standing-height-20260921-r1/run-r2.sh
```

These scripts require fresh output directories. The exact native CPU/GPU fixture interface is `test-measurement-standing-height MODEL_XML EXPORT_JSON ASSETS_DIRECTORY [--cpu-only]`. The unchanged two-fighter model hash is `6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa`.

| Artifact | SHA256 |
|---|---|
| `measurement.cu` | `31ac31944b0abb608dfd9ed16168803b2eabdd33e3d24a04200ab042f3f76b38` |
| `measurement.cuh` | `bf736deae56f8a472587ab9a8007bc756c6782f854c35cbfc0aa0e7f79a6b02f` |
| `measurement_probe.cpp` | `ac2739a5c25792f79768936e7922916569aed348677a554c9725c4b9ee0e341a` |
| `test_measurement_standing_height.cu` | `1577685102382e24b8fb61a06b9b1cab2c98ba0e5ee699c4539737e442725803` |
| Corrected measurement object | `b7534372a0d00bd99ba5e733893429d69e3cb45b9de72ce93b11a2380d036817` |
| Corrected measurement regression executable | `b98f314b2e9ebabb72e189c687621ac44e1e2fb2f1cb346a68683eed12da2be9` |
| Final fixture executable | `04b414eea7a6d0a45643923d46b7347dba99c28d667b5e62d8de5f756f044649` |
| Final fixture result | `f3f2f176a2dd4e25654c175e229ee96c3413f8a16be71b6041e7dfbca70d406a` |

The private Windows mirror is `C:/rekagent/work/consistent-fighter-20260919-r1/physical-standing-height-r1`, including byte-preserved pre-change raw-cvel sources. Proprietary model/motion payloads are read in place and are absent from the repository.

The complete private stage, including failed and successful attempts, is preserved at `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\native-standing-height-r1`. All 1,574 copied files, 66,604,804 bytes, matched source-before, destination-readback and source-after SHA256 checks. Archive manifest SHA256: `945b46e3eaf6e0be721f4cc5909f5d2b1e6a67980052a883906e80d682c4583f`. Existing source files and archives were preserved.
