# Physical observable-balance Constellation import, 2026-09-21

Three completed physical training runs were imported into a separate dataset
for the existing native Puffer Constellation viewer. Original INIs, the compact
sweep datasets, the original importer and the native converter were preserved.
The import itself ran no viewer, GPU, policy inference or game job. After
verification, the existing owned viewer was stopped and the new dataset was
opened on Spark's isolated display `:98` at 16:58:32 UTC. The native window and
its three-point plot were observed through the browser/VNC connection. The
Windows input desktop was untouched.

## Dataset and meaning

The dataset contains only `mujoco_cuda` runs with
`rek.native5.observable_balance.v1` and
`REK_NATIVE5_REWARD=normalized_points_falls_v1`.
The fresh reference and matched-initial warm-start comparison are separate
groups. Compact simulator results are absent.

| Completed run | Initialization | Command and INI LR | Native recorded own points | Native recorded wins | Newly reported rounds |
| --- | --- | ---: | ---: | ---: | ---: |
| `train-balance-physical-r1` | Fresh, seed 419 | 0.0001 | 13.5 | 0.5 | 4 |
| `train-warm-physical-continue-r1` | Physical final `390007e2...` | 0.0001 | 15.222222 | 0.444444 | 9 |
| `train-lr-lr001-r1` | Same physical final `390007e2...` | 0.001 | 12 | 0.25 | 4 |

These are the original INI's changing-policy, in-training measurements against
`CandidateApproachDummy`. `env/score` means own awarded points in the reported
completed rounds. `env/wins` is an in-training fraction; neither metric is an
authentic REK result or a frozen-policy strength estimate. The small samples in
this table cannot rank the policies reliably.

All three preserved INIs contain one objective-bearing row, at epoch 13 and
3,407,872 steps. All three processes completed 4,194,304 steps and have successful
completion receipts and final checkpoints. The cached row's step/time values
do not describe the entire training run. Native round-summary and process-time
receipts remain the source for complete-run aggregates.

## Native import checks

The separate importer accepts an explicit physical root and selected run names.
It refuses incomplete selections before creating a dataset. Learning rate and
twelve other numeric settings are checked between command provenance and the
unchanged INI. Both warm-start checkpoint hashes match:
`390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310`.
That hash is also present in each run's original provenance file.

The existing C `cache_data --full` converter retained all three native rows.
Eighty-four native metric/config values were verified against its float32 and
`%.6g` serialization. The converter expresses step counts in millions.
No reward or win metric was renamed into `env/score`.

The unchanged native converter skips metric keys containing the substring
`loss`, which also excludes `env/losses`. This behavior is explicitly recorded
in the new provenance JSON; the copied original INIs retain that field. An
initial verification attempt detected the omission and is retained separately
at `datasets/import-physical-20260921T165655790Z-3741777`. The verified dataset
below was created fresh with the exclusion documented.

## Paths and reproduction

Constellation stage:
`/home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation`.

Verified dataset, relative to that stage:
`datasets/import-physical-20260921T165745450Z-3743402`.
It contains original INI copies, native
`resources/constellation/experiments.ini`, converter stdout/stderr,
`provenance.json` and `import-command.json`.

Groups:

- `rek5_physical_fresh_normalized_points_falls_v1`
- `rek5_physical_warm_390007e2_normalized_points_falls_v1`

The new importer is `import-physical-constellation-r2.cjs`; the existing
`import-constellation.cjs` is unchanged. Executed remote command:

```sh
node /home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation/import-physical-constellation-r2.cjs \
  /home/spark-advantage/rek-training/physical-observable-balance-20260921-r1 \
  train-balance-physical-r1 train-warm-physical-continue-r1 train-lr-lr001-r1
```

Remote execution and transfer used WSL SSH/scp with host `spark`.
The existing `launch-constellation.sh` opened this verified dataset with viewer
PID 3745017. Its command, process identity and display-window checks are recorded
inside the dataset. The preceding compact-sweep viewer was PID 3527723; its
executable and working directory were verified before stopping that owned
process. The compact dataset remains available unchanged.

| Artifact | SHA256 |
| --- | --- |
| New importer, revision 2 | `12c231ca66b5c05f59707c674abdedad867a29f7112d39d976e46c4af1ee9ba2` |
| Original importer, unchanged | `97eb892726636ac37b3829dd7d2f2cd5b979cac397eec40a37661724cde01603` |
| Existing native converter | `53d78b920f98f5e6a6fa009e48712b172378eb702882e9a6cd3bb19b2c0eba4d` |
| Verified physical cache | `e511c4ea005e1a0ef8410e51e1581f39d5f074fcaa8dfe203470dcefbe166598` |
| Physical import provenance | `aecb4859ca7b9243aeeae6a576ba815e82bc590f7650f47f5f129e529c468fa3` |

Original source/copy hashes and every selected command, configuration and
checkpoint identity are preserved in the private provenance sidecar. No
checkpoint weights or proprietary binaries are published in this report.

## Private archive

Verified fresh NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\constellation-r1`.

`physical-constellation-r1.tar.gz` is 14,193 bytes and preserves 18 selected
files plus an inventory: importer revision 2, the existing launcher, all three
unchanged INIs, the native cache, import command/provenance, converter outputs,
and viewer launch command, PID, process, start time, display checks and logs.
Viewer stdout/stderr are explicitly identified as point-in-time snapshots.
Shared resources and existing viewer/converter binaries remain covered by the
original private normalized-sweep archive; their reference hashes are retained.

Source-before/source-after/snapshot hashes, native tar comparison, and
Spark/local/NAS archive hashes passed. All NAS files were created without
overwrite. Archiving performed no viewer, GPU, input or game action.

- Archive SHA256: `8ddf1610ddcf3e66b08164ef98667d5be4d0634f95e1cea00f87f58bf4a8cc16`.
- Receipt: `nas-copy-receipt.json` in the NAS directory above; SHA256
  `6f979a40369e676d955717d50e153806cb3fd2c068b2d329656f386b84076789`.
- Local helper, exact execution and transfer receipts:
  `C:\rekagent\work\physical-constellation-archive-20260921-r1`.
