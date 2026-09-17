# Preserved native physical-runtime quality probe

Executed 2026-09-17 UTC on Spark GB10. **Passed, exit 0.** This diagnostic
used the existing articulated MuJoCo CUDA runtime and native SONIC controller.
No physics implementation, policy, or training rule was changed. No PPO update,
Python execution, CPU physics call, or gameplay input occurred.

The four arenas each ran 1,000 decision ticks, with ten 0.002 s physical steps
per decision: 20 simulated seconds per arena. Diagnostic loop wall time was
15.4871 s; complete process wall time was 17.00 s. These times include deliberate
per-tick host action selection, readback and logging. They are not training SPS.

## Observed results

| Arena / diagnostic schedule | Locally supplied attack requests | Final points | Final falls | KO / bilateral reset events | Minimum root height m | Maximum tilt degrees |
| --- | --- | --- | --- | --- | --- | --- |
| 0: idle / idle | 0 / 0 | 0:0 | 0:0 | 0 / 0 | 0.6684 / 0.6684 | 8.47 / 8.47 |
| 1: stationary attack cycle / idle | 12 / 0 | 0:0 | 0:0 | 0 / 0 | 0.5954 / 0.6684 | 40.80 / 8.47 |
| 2: approach, repeated right hook move 3 / idle | 6 / 0 | 2:0 | 0:0 | 0 / 0 | 0.6154 / 0.6684 | 34.48 / 11.03 |
| 3: approach and attack cycles, both fighters | 2 / 2 | 0:5 | 1:0 | 1 / 1 | 0.0890 / 0.6408 | 106.84 / 44.87 |

All selected actions passed the runtime's existing legality mask. The scripted
diagnostic uses range 0.65 to 1.05 m and bearing tolerance 0.16 rad for approach;
these are test inputs, not recovered native Bot1 parameters or training rewards.
No learned policy was loaded. Arena 0 isolates standing behavior; arena 1
exercises canned targets without engagement. The action cycles include all 17
route choices but the bound ends before every route is exercised in every lane.

Every checked runtime failure flag was zero. Linker wrappers would abort on
`mj_step`, `mj_forward`, or `mj_kinematics`; observed call count was zero,
including startup. The startup log confirms `mujoco_cuda`, eight controller
rows, 57,813,876 resident controller bytes, 164 native physical phase nodes,
2,048 pooled contacts, and 512 constraint rows per arena.

Three lanes reached one terminal sample. Arena 3 had 0.0011783 s remaining at
the final sample after its counted reset. This is a 20 s simulated-duration
probe, not a claim that all four rounds completed.

## Actual fall, native KO award, and physical reset

Arena 3 supplies the missing outcome directly, without a synthetic damage rule:

| Decision tick | Measured event |
| --- | --- |
| 714 | Fighter 0 enters falling phase, tilt 45.848 degrees, root height 0.685719 m; both feet are off the floor. |
| 761 | Still falling: tilt 97.853 degrees, height 0.100922 m, height ratio 0.114655, two non-foot floor-contact bodies, accumulated fall hold 0.396000 s. |
| 762 | Fallen phase becomes 2; fall count becomes 1. Tilt 96.408 degrees, height 0.102105 m, height ratio 0.116147, three non-foot floor-contact bodies. Referee bits 1 classify a slip; native no-recovery count duration is 3 s. |
| 911 | Fallen at height 0.102956 m, tilt 92.890 degrees, four non-foot floor-contact bodies; referee count elapsed 2.988028 s. |
| 912 | Native referee bits 16 report KO, score delta is [0,5], and fight signals 33 report score change plus bilateral spawn reset. Both fall phases return to upright. |

At tick 912, the two physical roots move 1.335726 m and 0.644183 m relative to
tick 911. Their returned heights are 0.802735 m and 0.802802 m, with tilts below
0.08 degrees. This is a runtime KO/reset event with explicit referee bits,
unlike the inferential attribution required for the visual-only authentic log.
The fall was classified as a slip and no ordinary score preceded it. The probe
does not prove that an opponent strike caused the fall.

The approach/right-hook lane does move during its 0.9 s request windows. Five
complete windows have net XY displacements 0.556751, 0.505509, 0.589692,
0.579238, and 0.427514 m. However, the diagnostic sends 34 to 41 legal yaw-right
actions within those windows. These are combined hook/yaw trajectories, not
isolated-hook measurements and not a matched comparison with authentic trials.

## Preserved inputs and reproducibility

Source and harness:
[physical_quality_probe.cpp](../../validation-quality/physical_quality_probe.cpp),
[physical_quality_probe.sh](../../validation-quality/physical_quality_probe.sh).
The harness links the existing objects in
`/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2`,
excluding `pufferl.o`, and adds the CPU-call wrappers. It compiles against that
build's preserved `native-source` ABI, instead of rebuilding the environment.
Its process timeout is 120 s.

Private output directory:
`/home/spark-advantage/rek-training/policy-quality-20260916-r1/physical-quality-probe-r2`.
It contains both sources, full build/run commands, input/object hashes, ELF
dependencies, compiler output, per-tick physical measurements, stderr, process
timing, and exit code. Existing artifacts were preserved. The preceding `r1`
compiled successfully but stopped before execution because the initial harness
had a multiline command-recording error; that failed harness/output is retained.

```text
probe executable  b2d981919ab2da28506b7457c4adf39c1b3cc202b5769377e8de0698b3e47ae9
probe C++ source  8e44e1f30a25521c2aa5b5719d94e011da9a565753c3aaba96dd757e51253b06
probe harness     0563b8cc0c8b049734ea8eee41a105028c3c39aaf4e456b656d2e568d13a3718
stdout.jsonl      3c3d8154923c54d3ab428c17d7b1cbdebafa6af0d36d447176929c4fc1a9ba8b
stderr.txt        cb328adb3c8f5c71147f95c552c7b32b7466b3f32363ab44465d2c81b153cfc9
model XML         6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa
controller encoder 2befe0f7be8e72a33824f04fd39c191287e1e9aab05807012d6ceeb3f688ff40
controller decoder 7f4cb9a01bd21bbd0ef96e3c82d2ee1fc015e7a31dfb032d5e844719af3e9de2
```

## Conclusion

The existing native physical backend can execute stable standing, canned G1
tracking, physical falls, native 5-point awards and bilateral resets. These
capabilities were bypassed by the compact `fast_runtime`. This probe establishes
their present executability and one realized failure transition. It does not
establish authentic contact/balance parity, successful learned tactics, or a
trained fighter's win rate. The remaining model-quality question requires
matched, isolated physical-versus-authentic trajectory measurements before
claiming that returning to this backend alone fixes transfer.
