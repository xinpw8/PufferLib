# Native Sparring Bot1: recovered behavior and compact-candidate gap

Static evidence review, 2026-09-16. Repository baseline: `bd9e2552df214a461d1716817b8d1d48efb852c2`. No game interaction, GPU execution, new decompilation, or runtime changes were performed for this report. A future implementation should be labeled a **recovered-AI candidate** until its behavior is measured against the authoritative bot.

## Evidence and G1 applicability

The recovered native AI is `REKApp.AIOpponentController`, a randomized state machine with continuous locomotion commands. The extracted scene assets contain six brains, all with identical scalar tuning:

| Container | Fighter 1 path ID | Fighter 0 path ID |
| --- | ---: | ---: |
| level1 | 3262 | 3332 |
| level2 | 3266 | 3336 |
| level3 | 3263 | 3333 |

`FightCoordinator.PossessSlotIfChanged`, source line 4753, passes the slot's input controller to the brain, applies the difficulty, and activates it. `AIOpponentController.PossessSlot` binds that controller and the opposing fighter. This path has no T800/G1 type branch. The asset extraction contains no separate G1-prefab AI brain. These are scene-brain settings applicable to the occupied slot; the existing T800 contract's six-move subset and recovery assumptions must not be transplanted to G1.

G1-specific evidence is `sharedassets0.assets`, RobotConfig path ID `2722`, `RobotConfig_G1_UnitreeFighting`, `robotId=g1`. Its serialized SHA-256 is `3b0f5dfa78591b384ab022d85c4f3027182c8c4cf48f0c325a3b5170c5ebf1eb`. It assigns 17 moves, disables move interruption, E-stop, and special commands, and enables locomotion-transition settling. Forward/strafe/yaw configuration scales are 1; settle planar-speed/yaw-rate thresholds are approximately 0.03; braking rate is 2; keyboard yaw ramp is 0.5 s. These fields do not establish the authoritative server's actual physical response.

`FightCoordinator.get_SparringBotNumber`, line 66060, returns `clientAiDifficultyLevel + 1` on the remote-client path and `aiDifficultyLevel + 1` otherwise. Consequently, displayed **Bot1 corresponds to difficulty 0**. `ApplyDifficulty` at level 0 preserves the scene's base tuning. A new difficulty owner resets the stored level to zero. An arbitrary private AI encounter still requires observing its displayed difficulty before calling it Bot1.

## Distance, state, and control semantics

`DistanceToOpponent` measures **linear Euclidean root-to-root distance in Unity's horizontal XZ plane**, after zeroing Y. It does not return squared distance, collider separation, hand reach, or a policy-estimated gap. The square-root helper is the same one used by the recovered `UnityEngine.Vector3.Magnitude`. `AngleToOpponent` uses the projected robot forward direction and direction to the opposing root.

Rounded base tuning: stop distance 0.418 m, facing threshold 35 degrees, forward-command scale 0.8, yaw-command scale 1.5, initial delay 0.660 s, maximum engagement timer 2.150 s, minimum footwork 0.200 s, settle 0.300 s, recovery wait 0.258 s, maximum punching duration 3 s. Locomotion scales are command values; physical velocity equivalence remains unmeasured.

| Native condition | Recovered behavior |
| --- | --- |
| Start engaging, facing error <35 degrees and distance <= stop +0.15 m | Skip minimum-footwork delay; otherwise initialize it to 0.2 s. |
| Engaging, minimum-footwork delay finished, facing error <35 degrees, distance <= stop +0.30 m | Enter settling. This distance is approximately 0.718 m. |
| Engagement timer expires without settling | Call `StartEngaging` again, resetting `maxEngageTime`; expiry does not force an attack. |
| Settling, timer still positive | Re-engage only when distance > stop +0.50 m **and** facing error >=35 degrees. |
| Settling timer expires | Attempt attack if distance < stop +0.30 m **or** facing error <35 degrees; otherwise re-engage. This is not a hard maximum attack-distance gate. |
| Attack ends, or 3 s punching timeout | Clear punching; wait `recoveryTime`, then reposition with probability 0.05 or engage again. |
| Opponent down, except while already attacking | Give room; use 1.5 m spacing and 0.25 backward-command scale, followed by 1 s upright grace. |

The literals above were checked against the existing pinned native image: RVAs `0x3D57BBC = 0.15`, `0x3D57BC8 = 0.30`, `0x3D57BD8 = 0.50`, `0x3D57BF0 = 0.80`, and `0x3E9C560 = 45`. The `stop +0.1` attack threshold in our `ContinuousBotControllerContract.cs` is an audit-controller choice, not this native transition logic.

The native engagement command scales forward travel with `clamp((distance - stop)/0.8, 0, 1) * engageForwardSpeed`, with an additional close-facing footwork branch. That branch adds time-varying lateral motion and can override forward command to 0.24. Facing yaw has a 17.5-degree dead zone, then scales magnitude by `clamp(abs(angle)/45, 0, 1)` with the native sign convention. Repositioning uses randomized side, duration approximately 0.352 to 1.413 s, backward scale 0.15, strafe scale 0.434, and randomized yaw. `UpdateStateMachine` runs from `Update` using `Time.deltaTime`; locomotion runs from `FixedUpdate`. This evidence does not establish a fixed 50 Hz native decision cadence.

By comparison, `fast_runtime.cu:154` turns using discrete actions whenever bearing error exceeds 0.16 rad, approaches above 1.05 m, backs away below 0.55 m, waits for its own settled predicate, then cycles 16 attacks. It has no recovered native state timers, side-conditioned random attack pools, or native reposition/give-room behavior. Its default body radius is 0.22 m (`:531`); overlap correction enforces a 0.44 m root gap (`:336`). **That gap exceeds the native 0.418 m stop distance.** A port must preserve the actual distance definition, close-footwork branches, and `maxEngageTime` semantics. Changing stop distance is not a source-grounded fix. The native settle threshold near 0.718 m also means the 0.44/0.418 difference alone does not prove engagement deadlock.

## Assigned-move selection

`StartAttack` draws kick with probability 0.25 at Bot1, otherwise punch. Preferred side is right for a nonnegative signed opponent angle and left otherwise. It asks for a random assigned move in that category, favoring that side, and tries the other category only if no eligible move exists.

`RobotConfig.TryPickMove` scans assigned clips. A clip's category and side come from its **first non-null impact event with a nonzero limb**, through `MocapClipConfig.TryGetPrimaryLimb` and `AimLimbExtensions`. Reservoir sampling selects uniformly from the preferred-side pool when available, otherwise from the full requested-category pool. This creates the following static G1 pools, using zero-based indices in RobotConfig's assigned-move list:

| Pool | Indices | Serialized clip path IDs |
| --- | --- | --- |
| Left punch | 0, 1, 2, 5, 10, 12, 13, 14, 15, 16 | 2704, 2706, 2701, 2707, 2698, 2709, 2708, 2705, 2700, 2699 |
| Right punch | 3, 4, 11 | 2712, 2713, 2717 |
| Left kick | 6, 7 | 2710, 2703 |
| Right kick | 8, 9 | 2715, 2714 |

Index 16, `butt_smack_emote_processed`, has a left-hand first impact event and therefore enters the static left-punch pool. The compact loop's `%16` omits it. Clip names alone are insufficient for category selection: multi-strike clips follow their first qualifying impact event. Runtime move acceptance and execution can still reject a selected move.

## Source coordinates and provenance

Recovered text root: `C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp`. RVAs below are relative to the recorded GameAssembly image base, not live process addresses.

| Source and method | Source line | RVA |
| --- | ---: | --- |
| AIOpponentController.UpdateStateMachine | 192 | `0x2368570` |
| AIOpponentController.PossessSlot | 3497 | `0x23676B0` |
| AIOpponentController.ApplyDifficulty | 3873 | `0x2366AE0` |
| AIOpponentController.AngleToOpponent | 4476 | `0x2366600` |
| AIOpponentController.DistanceToOpponent | 5268 | `0x2367250` |
| AIOpponentController.StartEngaging | 5807 | `0x2367DE0` |
| AIOpponentController.ComputeFacingYaw | 6117 | `0x2366E20` |
| AIOpponentController.UpdateLocomotion | 6682 | `0x23681F0` |
| AIOpponentController.StartAttack | 7351 | `0x23679A0` |
| RobotInputController.TryGetRandomAssignedMove, preferred-side overload | 13515 | `0x226F900` |
| RobotConfig.TryPickMove | 2059 | `0x2267D80` |
| MocapClipConfig.TryGetPrimaryLimb | 1635 | `0x2266F10` |
| AimLimbExtensions.Category / Side | 78 / 42 | `0x2262590` / `0x22636C0` |

SHA-256 source identifiers:

| Source | SHA-256 |
| --- | --- |
| Recorded GameAssembly.dll | `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412` |
| Recorded global-metadata.dat | `e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd` |
| AIOpponentController.txt | `330956e4c257623fb78b0d619783c94b9a1f476bae4cf4e7c554491eb69c73e2` |
| FightCoordinator.txt | `9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5` |
| RobotInputController.txt | `f248df08449e3ff0706ce15ea07e4d58517f2fc9ed3f3143473fa48c4323bc21` |
| RobotConfig.txt | `9333f636923169b54c9ab927d3ba2168934cc420bf93bb8af6e7534da16a740b` |
| MocapClipConfig.txt | `6dfa8eaeef47220f3c0c13de09360fb24ecf43253a5e77cfff3b0d8219dd2db8` |
| AimLimbExtensions.txt | `ae5b56581e35263496f92a4a7650eea4189f08d29e05de244cabf7c7c9bb1ba1` |
| UnityEngine/Vector3.txt | `69b93812e19e7334ee5b5954a4479b85a2d124f702db95b0ca4591b103c3c6df` |
| mujoco_asset_probe_v8.json | `b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686` |
| Reviewed fast_runtime.cu | `64a58cd76a6a4c9035eda53a2201e8c51ebb386ebe5c0600ee6d7c3c49e6c4b1` |

The asset probe is at `C:\Users\Daniel\codex-wr64-puffer-scenario-7cc6ce1\ocean\rek\evidence\evidence_out\mujoco_asset_probe_v8.json`. Vector3 is under the sibling `IsilDump\UnityEngine.CoreModule\UnityEngine` directory. Repository provenance records are `ocean/rek/evidence/evidence_out/il2cpp_recovery.json` and `t800_strategy_contract_v2_20260903.json`; the latter's static-interface evidence does not establish simulator parity. Recorded build fingerprint: `f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659`.

## Remaining unknowns and validation boundary

I don't know whether the authoritative server uses exactly these client-image methods, asset values, random seed/state, or decision cadence. Those require version/configuration confirmation and synchronized observations of native state transitions, selected moves, accepted execution, and root motion in a permitted private encounter. Static source recovery alone cannot answer them.

The generic recovery code requests Dampen, Straighten, and orientation-selected get-up actions. G1's extracted configuration disables special commands and E-stop. Successful G1 recovery through that generic path remains unproven; T800 recovery behavior is not sufficient evidence. Controller/physics response, actual transition settling, move acceptance timing, falls, and contact scoring also need authoritative trajectory comparisons. The compact candidate explicitly does not integrate balance/fall dynamics. A high compact-opponent win rate consequently measures that candidate matchup and cannot establish success against authentic Bot1.
