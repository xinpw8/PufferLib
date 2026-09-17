using REKApp;
using RekEvidence;
using UnityEngine;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private bool _privateAiReadyRequested;

    private CommandResult ReadyPrivateAiSession()
    {
        var isolated = RequireBackgroundControl(out _) &&
            TryVerifyExplicitIsolatedSession(out var proof) &&
            proof == G1HeldInputScheduleContract.RequiredIsolationProof;
        var gameMenu = UnityEngine.Object.FindFirstObjectByType<GameMenuController>();
        var coordinator = gameMenu?.fightCoordinator ??
            UnityEngine.Object.FindFirstObjectByType<FightCoordinator>();
        var network = gameMenu?.networkSession ??
            UnityEngine.Object.FindFirstObjectByType<NetworkSession>();
        var context = GameContext.Instance ?? UnityEngine.Object.FindFirstObjectByType<GameContext>();
        if (coordinator is null || network is null || context is null || gameMenu is null)
            return CommandResult.Rejected("private_ai_bootstrap_controllers_not_found");

        var localSlot = coordinator.LocalFighterIndex;
        var opponentSlot = 1 - localSlot;
        var clients = coordinator.slotHasClient;
        var validSlots = localSlot is 0 or 1 && clients is not null && clients.Length > opponentSlot;
        var noHumanAi = validSlots && !clients![opponentSlot] &&
            (coordinator.clientHumanSlotMask & (1 << opponentSlot)) == 0 &&
            !coordinator.HumanInSlot(opponentSlot) && coordinator.OpponentIsAI &&
            coordinator.SlotIsAI(opponentSlot);
        var fighters = coordinator.Fighters;
        var visualPair = fighters is not null && fighters.Length >= 2 &&
            fighters[0] is not null && fighters[1] is not null &&
            fighters[0].IsVisualOnly && fighters[1].IsVisualOnly;
        var privateSolo = context.IsSolo && context.Mode == GameContext.SessionMode.Championship &&
            !context.IsKotH && !context.IsRanked && !context.AutoFindMatch &&
            !string.IsNullOrWhiteSpace(context.ArenaID) && !coordinator.IsRankedArena;
        var clientOnly = network.IsConnected && network.IsClient && !network.IsServer;
        var route = privateSolo && clientOnly && noHumanAi
            ? _soloRouteProofTracker.SnapshotForRuntimeSession(
                context.ArenaID, context.ArenaIP, context.ArenaPort,
                network.serverAddress, network.port, NativePointer(network).ToInt64(),
                bindRuntimeSession: false)
            : SoloRouteProofSnapshot.Unavailable("private_ai_bootstrap_route_not_proven");
        var facts = new PrivateAiBootstrapFacts(
            Isolated: isolated,
            ControlsIdle: !_g1PolicyRunning && !_scheduleRunning && !_singleMotionTrialRunning &&
                !_continuousControllerRunning && !_g1HeldScheduleRunning &&
                !_attackZoneTrialRunning && !_attackZoneRecoveryOnlyRunning && _freshRoundArm is null,
            ClientOnly: clientOnly, PrivateSolo: privateSolo,
            SlotsKnownNoHumanAi: noHumanAi, RemoteDriven: coordinator.IsRemoteDriven,
            IdleInactive: coordinator.CurrentPhase == FightPhase.Idle &&
                coordinator.CurrentRound is not { IsActive: true },
            NoVisualPair: !visualPair, NoSetup: coordinator.setupRoutine is null,
            MenuClosed: !gameMenu.IsMenuOpen,
            RouteProven: route.SoloRouteProven,
            RuntimeSessionConsistent: route.RuntimeSessionIdentityConsistent,
            AlreadyRequested: _privateAiReadyRequested);
        var reason = PrivateAiBootstrapContract.RejectReason(facts);
        if (reason is not null)
            return CommandResult.Rejected(reason);

        // OnFightButtonNetwork emits native REK_Ready. The
        // server schedules its ordinary StartFight. No local difficulty, pose,
        // controller command, score, or policy-eligibility field is written.
        _privateAiReadyRequested = true;
        coordinator.OnFightButtonNetwork();
        return CommandResult.RequestIssued("private_ai_ready_requested_opponent_unverified");
    }
}
