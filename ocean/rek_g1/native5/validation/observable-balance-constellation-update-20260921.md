# Five-run physical Constellation dataset, 2026-09-21

A fresh CPU-only native-cache import extends the three-run dataset documented
in `observable-balance-constellation-20260921.md` with the completed dummy
opponent LR 0.015 arm and completed recovered-Bot1 LR 0.015 arm. The prior
dataset, original INIs and importers were preserved. The CPU-only import did
not launch a viewer, GPU job, inference worker or game. After separate review
and authorization, the owned viewer was replaced on Spark display `:98`.

## Recorded rows and interpretation

All five runs completed 4,194,304 transitions. Each unchanged native INI
contains one objective-bearing row, at epoch 13 and 3,407,872 steps.

| Run | Initialization | Training opponent | LR | INI own points | INI win fraction | INI rounds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `train-balance-physical-r1` | Fresh | CandidateApproachDummy | 0.0001 | 13.5 | 0.5 | 4 |
| `train-warm-physical-continue-r1` | `390007e2...` | CandidateApproachDummy | 0.0001 | 15.222222 | 0.444444 | 9 |
| `train-lr-lr001-r1` | `390007e2...` | CandidateApproachDummy | 0.001 | 12 | 0.25 | 4 |
| `train-lr-lr015-r1` | `390007e2...` | CandidateApproachDummy | 0.015 | 13 | 0.25 | 4 |
| `train-bot1-lr015-r1` | `390007e2...` | `recovered_bot1_g1_v1` | 0.015 | 18.866667 | 0.333333 | 15 |

These are changing-policy, in-training measurements from the recorded INI
window. They are not whole-run aggregates, authentic REK outcomes, or
frozen-policy strength estimates. `env/score` remains own awarded points;
no reward or win metric was renamed. The higher recovered-Bot1 score is from
a different opponent and a different small round sample. It cannot establish
improved authentic strength. Cached timing and SPS fields likewise describe
the recorded row, not the complete training process.

All four warm starts have the exact initial SHA256
`390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310`.
The recovered-Bot1 process and receipt both exited 0 before import. Its final
checkpoint SHA256 is
`7c34eaa9f00c1ee97f8cbd8abf894d773d8a838547a10e496bcb5980de821c84`.

## Import and audit

Importer revision 3 adds an explicit second permitted physical root and
retains each stage's existing build-manifest files. Opponent identity is
recorded separately. The recovered-Bot1 group requires both the exact command
assignment and the actual runtime startup mode. The legacy stage retains its
pinned CandidateApproachDummy default with no opponent override.

The unchanged native `cache_data --full` retained all five rows. All 140
checked native metric/configuration values matched its float32 and `%.6g`
serialization. Original source and copied INI hashes matched before and after
the import. Learning rate and twelve other numeric settings matched the
unchanged INIs and command provenance. Local downloaded cache and provenance
hashes also matched Spark.

The native converter still drops metric keys containing `loss`. In these INIs
the excluded key is `env/losses`; it remains in the original copies and is
listed in `provenance.json`. Step counts remain expressed in millions in the
cache. No metric was fabricated or filled in from complete-run receipts.

Group names fit the unchanged native viewer's 64-byte table-name storage:

- `rek5_phys_fresh_dummy_norm_points_falls_v1`
- `rek5_phys_warm_390007e2_dummy_norm_points_falls_v1`
- `rek5_phys_warm_390007e2_recovered_bot1_norm_points_falls_v1`

## Paths and commands

Stage: `/home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation`.

Fresh dataset: `datasets/import-physical-20260921T175116011Z-3854001`.

The importer, original INI copies, cache, converter outputs, exact command,
source/checkpoint hashes and provenance are preserved privately. Local copy:
`C:\rekagent\work\physical-constellation-update-20260921-r1`.

Executed through WSL SSH host `spark`:

```sh
node /home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation/import-physical-constellation-r3.cjs \
  /home/spark-advantage/rek-training/physical-observable-balance-20260921-r1 \
  train-balance-physical-r1 train-warm-physical-continue-r1 train-lr-lr001-r1 train-lr-lr015-r1 \
  --root /home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1 \
  train-bot1-lr015-r1
```

The existing viewer PID 3745017 was verified still running from the earlier
three-run dataset after import. Following separate authorization, its exact
executable and working directory were checked again immediately before an
exact-PID TERM. Its exit was confirmed within the bounded 10 s check. No other
process was signaled. The existing launcher then executed:

```sh
bash /home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation/launch-constellation.sh \
  /home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation/datasets/import-physical-20260921T175116011Z-3854001
```

The new viewer is PID 3859169. Its executable, new dataset working directory
and `DISPLAY=:98` were verified. Window metadata records
`0x200007 "Puffer Constellation"`, 1920 by 1080 pixels, at 17:53:49 UTC.
The import and viewer replacement used no Windows or browser input. Subsequent
browser/VNC inspection confirmed the rendered native viewer. Its displayed
axes remained uptime and `env/perf`; attempted dropdown selection did not
produce a verified axis change. This view must not be read as an authentic
win-rate ranking. No Windows global input was emitted.

| Artifact | SHA256 |
| --- | --- |
| Importer revision 3 | `f1f78c7d1ab87460de04a80d7ca7538e1f14458400948ee293fe587f200291c5` |
| Revision 2, unchanged | `12c231ca66b5c05f59707c674abdedad867a29f7112d39d976e46c4af1ee9ba2` |
| Native converter, unchanged | `53d78b920f98f5e6a6fa009e48712b172378eb702882e9a6cd3bb19b2c0eba4d` |
| Five-row cache | `88bd117d338666a533bd1090a240016444a1e3b86ea8e0a9133780ed849294c3` |
| Import provenance | `0e2f98ab5b42e52aee38074a6f45623afc067423643c24e6e5abd89667209d46` |
| Existing viewer launcher | `4c7945e135dcda83e2179a43bf032684a98272dc87bf08d7cd7ede6146b3c8a7` |

## Private archive

Fresh NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\constellation-r2`.

`physical-constellation-r2.tar.gz` is 19,213 bytes. It preserves 25 selected
files plus the source inventory: importer revision 3, unchanged launcher,
owned-viewer replacement script/identity records, all five unchanged INIs,
native cache, import command/provenance, converter outputs and new viewer
PID/start/command/display/log evidence. Viewer logs are point-in-time snapshots.
Shared resources and existing native binaries retain reference hashes and
remain covered by the earlier normalized-sweep archive.

Source-before/source-after/snapshot hashes, native tar comparison and
Spark/local/NAS hashes passed. No existing archive was overwritten. Archiving
performed no viewer, GPU, input or game action.

- Archive SHA256: `9a18b6dcfb8a6cf81aa1d5182bbbe0f9d585f9de01be8e4e6c3813d8d5705991`.
- NAS receipt SHA256: `2bb0ee3c8905d4aa3a56e4dba2b29d4031b5bd1cc72aff38f6fe373b64e520f4`.
