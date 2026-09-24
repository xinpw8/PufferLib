# Closed native stage archive

Verified on 2026-09-24T11:39:33.507Z at:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\timing500-ppo-refresh-native-20260924-r1`

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| evidence.tar.gz | 23649408 | `0a3870f23aef5a4089533fb78442e60c037ffea28adf812f981af9445d88e1b8` |
| native-dependencies.tar.gz | 4319600 | `5afc4c9d60d77e093845e92c6d10355c7d60ac2f46889c894702a4b06c264bd1` |

The first archive contains exactly 104 regular source/metadata/data/replay/checkpoint files and 55 preserved symlinks. The second contains exactly the six pinned execution inputs: C2 behavior checkpoint, native worker, native object, all-ones mask, replayer and trainer. The final f147 checkpoint and both original-v3 and derived-v5 datasets are included in the first archive.

The inventory was frozen before the later private copies of `capture_resources.cjs` and `snapshot_controls.cjs`. Those two passive helpers and the changing live resource log are excluded from this closed-training archive; the archive has not been overwritten.

Source, symlink-target, native-capture and dependency hashes matched before and after transfer. Stream hashes matched NAS readback hashes. Archive member listings matched the explicit file allowlists. Symlink targets were preserved, not silently dereferenced. Neither original files nor the active live cohort were changed. No proprietary game binaries, credential stores or GPU work are included.

The existing closed parent archive remains a required dependency:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\timing500-ppo-live-20260924-r1-closed-20260924T110034Z`

- `evidence.tar.gz`: `2393472f31fd7b1060cf48674faa7da0699d6ea551bc9284adc072821fe68f30`.
- `native-captures.tar.gz`: `ac60c7db18bf634b4e1db59064578802035e452ea6e124dfd453c70719567c7d`.

Both parent NAS hashes were reverified. All 55 symlink targets match the parent's recorded member hashes and sizes. All five selected native captures match the archived native receipt. The parent includes the genuine raw worker/encoder/relay evidence and native packets; they are not copied to Git.

Restore the parent cohort at its original path or deliberately remap the symlinks. Restore native captures at the explicitly recorded native root, and fresh-stage evidence at its recorded stage path. The dependency tar uses paths relative to filesystem root: extract into a fresh recovery directory and intentionally map those paths rather than blindly overwriting live files. Rebuild source remains in the existing validation source bundles and the CUDA toolchain is external.

`receipts/closed-archive-receipt.json` records exact paths, sizes, hashes and restoration dependencies. NAS also retains the allowlisted archive/snapshot helpers and complete before/after inventories. No dataset, replay tensor, checkpoint, binary or raw capture is committed by this publication.
