# Existing private-bot robot selection

A bounded read-only inspection found no existing client-side setting or request argument that deterministically selects G1 as the private sparring opponent. No plugin, server, game, configuration or input was changed.

Recovered source root: `C:/rekagent/work/controller-audit-isil/IsilDump/REKApp/REKApp`.

- `FightCoordinator.txt:3779`, `PickAiRobotId`: reads the other human slot's robot identity, private `aiMirrorChance`/`aiRobotPool`, and Unity RNG. Random selection is visible at 4020, 4034 and 4051. The constructor at 75515 defaults mirror chance to 0.75; 75545/75561 establish the pool `g1,t800`. These are recovered defaults, not measured authoritative-server configuration or live probabilities.
- `FightCoordinator.txt:4163`, `MirrorAiRobotsToHumans`: selects for nonhuman slots not pinned by a scene. `FightCoordinator_NestedType__DelayedNetworkFightStart_d__72.txt:222` invokes it before ordinary `StartFight`. Thus selecting own G1 does not make opponent G1 deterministic.
- `CentralApiClient.txt:1001` exposes matchmaking with a flow argument and callbacks, without an opponent-model argument. `CentralApiClient_NestedType__FindMatchCoroutine_d__25.txt:481` constructs the flow-only request. `KothScreenController_NestedType__SoloFindLoop_d__93.txt:849` uses that ordinary path.
- `FightCoordinator.txt:49453` sends ordinary `REK_Ready` without a model request. Existing bridge `ocean/rek/evidence/windows/RekUiBridgeAgent/Plugin.cs:852` invokes `OnSoloClicked`; `Plugin.PrivateAiBootstrap.cs:11` sends native readiness only.
- `ArenaRobotOverride` contains private scene-prefab/robot fields, not an exposed client reservation setting. `SimClientDriver` has `-simRobots`, but `SimClientDriver_NestedType__SetupRobot_d__33.txt` uses the client's own garage/loadout, including `ActivateLoadout` at 1051. It does not select the sparring robot.

The matching inventory at `C:/rekagent/work/rek-current-interop-b20ca0d/inventory-dummy-rek.json` identifies the mirror/pool fields and selector as private. Searches of recovered settings, development settings, arena environment setup and simulation arguments found no supported opponent-selection control. Unknown server configuration and undiscovered official server capabilities remain outside this conclusion.

`FightCoordinator.txt` SHA256 was reverified as `9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5`. No proprietary recovered implementation is copied here. The practical conclusion is to retain the current measured mixed-pair exclusion; no launch flag, invented API field or client-state patch is justified by this inspection.
