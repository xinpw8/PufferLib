# G1 horizontal heading evidence

Read-only source/asset review, 2026-09-21. The installed G1 prefab and recovered
native getter support the shared projection's horizontal root-local +X heading.
They do not support adding a 90-degree or 180-degree heading correction. No
runtime, encoder, model, game process or input configuration was changed.

## Convention and identity

`observable_balance.h:77` projects local +X from the common-frame root
quaternion. `Plugin.G1PolicyStream.cs:448` obtains the live root from
`Robot.RootTransform`; line 461 publishes that root's rotation.

The private recovered `Robot.get_Forward` at RVA `0x23e0100` obtains the same
`RootTransform`, negates its right vector, and rotates the result by serialized
`forwardYawOffset` about Unity world up. Relevant evidence is
`C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp\Robot.txt:38200`
(root getter at line 38350, right vector at 38400, offset/rotation at
38240..38302). The rotation axis is also identified by
`C:\rekagent\work\controller-audit-isil\IsilDump\UnityEngine.CoreModule\UnityEngine\Vector3.txt:4827`.
Only derived semantics are recorded here.

The existing private asset probe is
`C:\rekagent\evidence\motion-parity-D21-20260903T071000Z\artifacts\subagent-g1-spawn-mujoco_asset_probe_v7.json`.
Its G1 `Robot` object is `sharedassets0.assets`, path ID 3430, owner
`g1_29dof_Prefab_SONIC`, imported robot `g1_29dof`, root body `pelvis`.
It records `forwardYawOffset = -180` at line 155132. The serialized object
SHA256 is `81d93f30e18968ebc76c63916f686103f923cfc705bac8d6f2674e3e314380b7`.
This is separate from the T800 object recorded in `controller_path.json`.

For Unity root-right vector `r = (rx, ry, rz)`, the recovered operation with
this offset produces `(rx, -ry, rz)`, apart from floating-point rotation
roundoff. Its horizontal XZ projection therefore equals root-right XZ.
The shared coordinate conversion maps Unity XYZ to common XZY, and quaternion
XYZW to WXYZ `(-w,x,z,y)`. Consequently the shared local +X horizontal vector
is exactly `(rx, rz)`. This conclusion also applies to tilted roots with a
nonzero horizontal projection; it does not equate the full 3D forward vectors.

## Provenance checked against installed files

Repository paths below are relative to `C:\Users\Daniel\codex-rek-puffysics-training-profile`.
The asset probe's inventory hash matches the repository inventory. Both installed
files were independently hashed during this review and match that inventory.
Build fingerprint: `f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659`.

| Source or artifact | SHA256 |
| --- | --- |
| `C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\GameAssembly.dll` | `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412` |
| `C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\REK_Data\sharedassets0.assets` | `37f7a476c56caae37f5a04d4fa1acf5954fdc2b90f20f521830369ecff05f355` |
| `ocean/rek/evidence/evidence_out/inventory.json` | `ea932824c7f1fa9781ab816716d4bfca9ec22b14e754466941c8c157910eff79` |
| Private asset probe named above | `2e94ebda205da7445c0767722250ddc1e1966a9a951a3a405e41822ac818ef36` |
| Private `Robot.txt` named above | `4f61233092542b15773e49d8404790a8ed89352d3b656fa41b75bab9c8283ded` |
| `ocean/rek_g1/native5/observable_balance.h` | `4b651a2f02b7a335cb781bb886b048d83025acc88dca4a522f62bc4cea7a0d3a` |
| `ocean/rek/evidence/windows/RekUiBridgeAgent/Plugin.G1PolicyStream.cs` | `bd602ea1d032bfe078deb1f5223d6551e12afb0c113d2922102872418357adb5` |
| `ocean/rek/evidence/windows/RekUiBridgeAgent/Plugin.cs` | `e421272a42dca22608528e06b3a6babf41567b11e83d7f98476be639953f87f7` |

## Limits and smallest runtime confirmation

The current bridge's continuous `forward_xyz` field is populated from
`Transform.forward` (`Plugin.cs:2501`, publication at line 3490). It is Unity
local +Z and is not an observed call to `Robot.Forward`. The G1 policy stream
records root rotation but does not record the native forward vector or offset.
The older `C:\rekagent\RekAgentBridge\Plugin.cs:333` does read `Robot.Forward`,
but its saved `C:\rekagent\agent_state.json` is dated 2026-07-19 and has
`inFight=false`; it cannot validate the current G1 runtime. That legacy plugin
also changes input configuration on load and was not invoked.

This establishes the serialized configuration/getter convention for the current
installed build. It does not measure a live instance's offset, establish server
execution or confirm full pose, motion, scale or policy parity. The heading
ambiguity in `reward-objective-20260919/action-interface-diagnosis.md` is narrowed
by this previously available G1 asset evidence.

The smallest direct runtime confirmation would passively record, in the same
sample and for a verified G1 instance, `Robot.Forward`, `RootTransform.rotation`,
the instance's `forwardYawOffset`, and its existing lifecycle/QPC identity.
Compare normalized horizontal native forward with the shared +X projection
across ordinary nondegenerate poses. This requires observation only; no movement
commands or input changes are needed.
