# Manual attack mapping correction, app r8

App r7 interpreted saved policy categories as original native move indices by subtracting 16. Those orders differ, so ten of seventeen named attacks reached the wrong clip. App r8 changes only the conversion in `league/input.cjs`: categories 16 through 32 map to `[6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16]`.

HH now requests left front kick (native 7), UU right side kick (8), and L right jab (4). Previously those requests selected left jab (1), double uppercut (2), and right side kick (8). Keyboard gestures and buttons share this corrected boundary. The saved bindings, gesture parser, continuous axes, direct-command API, attack acceptance rules, fixed timestep and scheduler are unchanged.

All seventeen mappings and one-shot consumption pass an independent named-source check. The old source fails ten mappings. Tests derive expected identities from saved command names, original key getters, named motion clips, native routes and the pre-existing policy table. They do not derive expectations from the corrected lookup. The full CPU app suite reports 54 passed, one unchanged real-model preparation test skipped locally, and no failures. Three new named regression tests fail when run against the original r7 boundary, while its gesture-only test still passes.

The read-only Windows receipt verifies that all six specifically selected controls values match the September 25 export byte-for-byte. No preference was changed. The receipt contains only control value names, lengths and hashes; raw human traces and authentication values are excluded. The source-backed 300 ms double-tap constructor default remains unchanged, with the original live serialized seed value still unverified.

| Saved keys | Named move | Policy category | Native move index |
|---|---|---:|---:|
| I | left_hook | 20 | 0 |
| K | left_jab | 21 | 1 |
| Space + J | double_uppercut | 22 | 2 |
| O | right_hook | 23 | 3 |
| L | right_jab | 24 | 4 |
| Space + L | left_jab_right_uppercut | 25 | 5 |
| YY | left_side_kick | 16 | 6 |
| HH | left_front_kick | 17 | 7 |
| UU | right_side_kick | 18 | 8 |
| JJ | right_knee | 19 | 9 |
| Space + Y | 6_punch | 26 | 10 |
| Space + U | run_and_punch | 27 | 11 |
| ; | left_right_jab | 28 | 12 |
| ' | left_right_hook | 29 | 13 |
| Space + K | left_hook_right_jab | 30 | 14 |
| Space + H | double_hook | 31 | 15 |
| Space + I | butt_smack_emote | 32 | 16 |

`independent-review/MANIFEST.json` pins the independent review, named expected fixture, current Windows equality receipt and ten-failure regression. `semantic_duel_assets_manifest.json` is the exact original asset-route contract used by the tests. `app-r8.diff` preserves the complete parent-to-fix delta. `tests-r8.json` and `tests-r8.txt` preserve the closed CPU result. `controls-r7-negative.txt` preserves the failing old-boundary regression.

The app manifest is `456d96225d79a01821d5959742b9120bbcd14f8b58bf7094666abc2518e7fd10`; parent r7 manifest `6be9fa74673eca108da8c07cd414d27d55b6cc9f625db1ab70e2f5c3e6f02cca` remains in the app history. Repository copy verification covers all 176 payload files and the current manifest. This publication runs no native process and makes no new performance, trajectory, hit-outcome or physical-parity claim. Correctly routed requests can still be rejected by native action state. Deployment receipts are maintained separately by the task owner.

Regression check using an installed Node runtime and absolute paths:

```sh
node independent-review/verify_mapping.cjs /absolute/path/to/native_clone/app/league/input.cjs /absolute/path/to/controls-fix-20260929-r1/parent-r7/input.cjs
```
