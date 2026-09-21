# CPU contact-velocity composition probe

The proposed body-velocity composition agrees with direct pinned MuJoCo 3.7.0 calculations on the tested route states. The largest double-precision linear-component difference was **3.20e-14 m/s**. This establishes measured kinematic consistency for these inputs. It does not establish physical controller, contact-response, balance, recovery, or authentic fighting parity.

No runtime or asset source changed. The probe uses the unchanged `load_fast_assets` API, links no CUDA runtime, and performs no GPU work, optimizer update, game connection, or simulation step. The earlier [native contract report](../contact-velocity-contract-20260920/README.md) establishes the recovered `cvel` read; its version-unknown statement records the evidence available at that time.

Updated static evidence supplied during this probe identifies the installed `REK_Data/Plugins/x86_64/mujoco.dll` as 3.7.0: PE FileVersion/ProductVersion are 3.7.0 and the exported `mj_version` statically returns 3,007,000. That installed file is 3,556,352 bytes, SHA256 `a6f6b6fd6f0cc35923f57fdc43e18a5cfa43b451fc719bfa355c71c745b43eea`. No DLL code was executed for that identification, and active-process module loading was not demonstrated. This experiment executes the separate pinned Linux 3.7.0 library; matching release identities do not prove cross-platform bitwise equivalence.

## Method and coverage

The new `validation-quality/contact_velocity_probe.cpp` reconstructs all 72 generalized positions for both fighter trees from actual transformed `FastFrame` poses. Free-joint position and velocity addresses come from the model; hinge addresses come from `FastAssets`. Float root quaternions are normalized before manifold operations. Moving-edge generalized velocities use `mj_differentiatePos` with the existing 0.02 s interval.

Canonical and world references each call `mj_kinematics`, `mj_comPos`, and `mj_comVel`. For each of the 14 relevant bodies per fighter, the probe compares direct world `cvel[6*body+3..5]` against:

```text
L_world = Rz(yaw) * L_canonical + V_base
          + Omega_world cross (Rz(yaw) * C_root_subtree)
```

Direct world generalized velocity independently transforms root translation and adds `R(root_world)^T * Omega_world` to the free joint's local angular velocity. A second check takes a centered derivative of the composed pose path, using `mj_integratePos` only as a coordinate-manifold operation. Its interval is 2e-6 s. This operation has no forces, controller, contacts, or solver.

Coverage comprises all 24 routes and 1,768 baked frames on both model trees, with five paired external-motion scenarios: identity, two combined signed yaw/translation-rate cases, translation only, and yaw only. The actual catalog has three reverse-playback routes, seven loops, and **zero mirror-enabled routes**. A separate synthetic reflection sweep applies the loader's reflection map to those real frames; it is not presented as an actual mirrored route. Maximum tested root tilt was 0.880836 rad.

The two sweeps together produced:

- 17,920 paired pose/motion cases and 501,760 body-triplet comparisons.
- 3,870,720 directed geometry-pair relative-speed comparisons.
- 465,920 shared-body geometry checks, preserving distinct geometry identities.
- 1,270 independent centered-derivative checks of world generalized velocity.
- 48 reconciled route-start edges, 34 held non-loop endpoint edges, and 14 loop-wrap edges.

Both fighters' bodies are evaluated separately; inertial symmetry is not assumed. Same-body foot and torso colliders read the same body velocity while retaining their distinct geometry-pair identities. At reconciled starts and held endpoints, identical coordinate arrays explicitly produce zero clip velocity. Existing instantaneous route jumps are excluded from velocity: the discarded cross-route difference reached 41.7693 in a generalized-velocity component and is not a physical impulse estimate.

## Measured errors

The executable reports maxima before selecting a numerical tolerance. Exit 0 indicates finite results, required coverage, and structural checks; it is not a hidden numerical acceptance threshold.

| Comparison | Maximum absolute difference |
| --- | ---: |
| Double linear `cvel` component | 3.197442310920451e-14 m/s |
| Double relative-speed norm | 4.618527782440651e-14 m/s |
| Host-FP32 composed linear component versus double reference | 1.622815931412447e-6 m/s |
| Norm from FP32 composed triplets versus double reference | 1.748782727517551e-6 m/s |
| Transformed root-subtree COM component | 8.881784197001252e-16 m |
| Centered derivative versus analytic world `qvel` component | 3.030102835310800e-10 |

The last row combines translational and angular generalized components, whose units are m/s and rad/s respectively. FP32 composition is measured on the CPU with contraction disabled; the final norm in that diagnostic remains double precision. It does not measure CUDA arithmetic or a full FP32 contact kernel.

Two deliberately incorrect alternatives produce materially larger differences: adding a world angular vector directly to local free-joint angular velocity changes a linear `cvel` component by up to 0.094357 m/s; replacing the root-subtree COM reference with a striker geometry center changes the external-yaw linear term by up to 1.317219 m/s. These are sensitivity diagnostics, not measurements of authentic hit error.

## Preserved development failures

`probe-r1` failed its exact-zero check because MuJoCo quaternion self-differencing leaves roundoff for some identical normalized inputs. The final probe records that raw residual, maximum 2.278310523278522e-15, and assigns exact zero only after verifying the old/current coordinate arrays are identical. Moving-edge differencing is unchanged.

`probe-r2` failed a coverage assertion that incorrectly expected an actual mirror-enabled route. Source inspection confirmed all current mirror flags are zero. The final probe reports this inventory and labels its additional reflection sweep synthetic. Both failed executables, source archives, stderr, commands, timings, and exit codes remain preserved. `probe-r3` exited 0; no numerical threshold was relaxed.

## Reproduction and artifacts

Successful execution: 2026-09-21 00:28:47.803372124 to 00:28:48.045127314 UTC. `/usr/bin/time` records 0.23 s wall time, 0.23 s CPU user time, and 18,596 KiB maximum resident memory. This is a small CPU reference test, not a training-throughput benchmark.

Private Spark stage: `/home/spark-advantage/rek-training/contact-velocity-probe-20260920-r1`. Successful source and output directories are `source-r3` and `probe-r3`. The executed build/run command was:

```bash
bash /home/spark-advantage/rek-training/contact-velocity-probe-20260920-r1/source-r3/ocean/rek_g1/native5/validation-quality/run_contact_velocity_probe.sh \
  /home/spark-advantage/rek-training/contact-velocity-probe-20260920-r1/probe-r3 \
  /home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact/model.two_fighter_arena.xml \
  /home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact \
  /home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features
```

The script requires a new output directory. `commands.txt` records each C/C++ compiler invocation and the executable command. `provenance.txt` binds the included sources, dependency, model, and manifests. The compiler was GCC 13.3.0. Linker wrappers reject calls to `mj_step`, `mj_forward`, `mj_step1`, and `mj_step2`; the recorded forbidden-call count is zero. `dependencies.txt` contains no CUDA, Python, or Torch library dependency.

| Artifact | SHA256 |
| --- | --- |
| Probe source | `7d3f88c73eda784ec53781ef269411aa49b46b6a79ac6fd76fd9317c51884daa` |
| Build/run script | `681452bb5d13a239c5e2ed89d42cd70d27539d88fbac835009152fc1c3fd8392` |
| `source-r3.tar` | `77311ce9f655f43438600226b699aaa4a740bc27eb48af7001b1d2eb1bdd4f20` |
| Successful executable | `df2db91cfe665b3117f3cf8005db49fa2f790f2bbcd94e695c3cba355f9b4d05` |
| `probe-r3/result.jsonl` | `5c084f0a7c7d25f8f904b79962762eba6ce010571d1eec1bed255784832b4e63` |
| Pinned `libmujoco.so.3.7.0` | `ef77a7d7d1e5a83197674170b296d0ea072376db89edbf161cc0637508d7642f` |
| Model XML | `6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa` |
| Asset manifest | `7d4719a3ca1e9e5a8faf571bc3c5c70e2b4c9e841be34303fa0a3f47d3692a28` |
| Feature manifest | `3b69127f5004574057a6b99a976a31c0e3ed3489c22c193b94a0c4e5b4aa59bc` |

The complete stage is privately mirrored at `C:\rekagent\work\consistent-fighter-20260919-r1\contact-velocity-probe-r1\spark-results`. All 69 unique stage-local files named by the three run provenance/result manifests were rehashed locally and matched. The source archives also remain beside that mirror. Proprietary model and motion assets were read in place on Spark and were not added to the repository.

The probe stage and installed-version investigation are also preserved on the
existing private evidence server under
`2026-09-19/consistent-fighter-r1/native-contact-velocity-probe-r1`.
All 111 copied files (2,586,984 bytes) matched source-before, destination and
source-after SHA256 checks. Archive manifest SHA256:
`2f3df263bf51e48259f6827a549e1538696abc465d86dab86f1c789675d823dc`.

No proxy integration or training is included. A later opt-in implementation must preserve the legacy asset/runtime path and separately validate device precision, route-edge selection, and scoring effects.
