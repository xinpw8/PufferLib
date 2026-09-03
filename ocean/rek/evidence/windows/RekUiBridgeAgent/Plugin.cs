using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using BepInEx;
using BepInEx.Unity.IL2CPP;
using HarmonyLib;
using Il2CppInterop.Runtime;
using Il2CppInterop.Runtime.InteropTypes;
using REKApp;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.UIElements;
using Button = UnityEngine.UIElements.Button;

namespace RekUiBridgeAgent;

[BepInPlugin(PluginGuid, PluginName, PluginVersion)]
[BepInProcess("REK.exe")]
public sealed class Plugin : BasePlugin
{
    public const string PluginGuid = "rek.evidence.control.bridge";
    public const string PluginName = "REK Evidence Control Bridge";
    public const string PluginVersion = "0.2.6";

    private const string PipeName = "rek-ui-bridge-v1";
    private const int MaxPendingRequests = 32;
    private const int MaxRememberedRequests = 2048;
    private const string ExpectedGameAssemblySha256 =
        "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
    private const string ExpectedMetadataSha256 =
        "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
    private static readonly ScheduleStep[] MeasuredSchedule =
    {
        new(0, Vector3.zero, null, "neutral"),
        new(50, new Vector3(1f, 0f, 0f), null, "forward_1"),
        new(150, Vector3.zero, null, "neutral"),
        new(200, new Vector3(-1f, 0f, 0f), null, "backward_1"),
        new(300, Vector3.zero, null, "neutral"),
        new(350, new Vector3(0f, -1f, 0f), null, "strafe_left_1"),
        new(450, Vector3.zero, null, "neutral"),
        new(500, new Vector3(0f, 1f, 0f), null, "strafe_right_1"),
        new(600, Vector3.zero, null, "neutral"),
        new(650, new Vector3(0f, 0f, -1f), null, "yaw_left_1"),
        new(750, Vector3.zero, null, "neutral"),
        new(800, new Vector3(0f, 0f, 1f), null, "yaw_right_1"),
        new(900, Vector3.zero, 2, "move_2_punch_combo"),
        new(1100, Vector3.zero, 3, "move_3_right_kick"),
        new(1300, Vector3.zero, 4, "move_4_left_punch"),
        new(1500, Vector3.zero, 5, "move_5_right_punch"),
        new(1700, Vector3.zero, 9, "move_9_right_shoryuken_lm_dragon_punch"),
        new(1900, Vector3.zero, 10, "move_10_front_kick_L_left_kick"),
        new(2100, new Vector3(1f, 0f, 0f), 2, "forward_1_move_2"),
        new(2300, Vector3.zero, null, "neutral"),
        new(2400, new Vector3(-1f, 0f, 0f), 3, "backward_1_move_3"),
        new(2600, Vector3.zero, null, "neutral_complete"),
    };

    private readonly ConcurrentQueue<BridgeRequest> _pending = new();
    private readonly ConcurrentDictionary<string, byte> _remembered = new(StringComparer.Ordinal);
    private readonly ConcurrentQueue<string> _rememberedOrder = new();
    private LocalPipeServer? _pipe;
    private BridgeBehaviour? _behaviour;
    private Harmony? _harmony;
    private int _pendingCount;
    private long _stateSequence;
    private long _lastPublishedConnection;
    private string? _lastStateIdentity;
    private string _gameAssemblySha256 = string.Empty;
    private string _metadataSha256 = string.Empty;
    private string _pluginSha256 = string.Empty;
    private string _scheduleSha256 = string.Empty;
    private long _leaseConnectionId;
    private bool _scheduleRunning;
    private bool _scheduleAuthorizedWhileBackground;
    private int _scheduleTick;
    private int _scheduleFixedSubstep;
    private int _nextScheduleStep;
    private string? _scheduleRunId;
    private Vector3 _scheduleVelocity = Vector3.zero;
    private RobotInputController? _scheduleInput;
    private IntPtr _scheduleInputPointer;
    private int? _scheduleMoveAwaitingSend;
    private int? _scheduleMoveScheduleTick;
    private bool _scheduleMoveArmedObserved;
    private bool _scheduleMoveInvocationObserved;
    private int _scheduleMoveSendCompletedCount;
    private bool _scheduleFinalNeutralInvocationObserved;
    private bool _scheduleFinalNeutralSendObserved;
    private int _scheduleFightEpoch;
    private int _scheduleRoundNumber;
    private RuntimeIdentity? _scheduleIdentity;
    private bool _sendBoundaryPatchesVerified;

    internal static Plugin? Instance { get; private set; }

    public override void Load()
    {
        var gameAssemblyPath = Path.Combine(Paths.GameRootPath, "GameAssembly.dll");
        var metadataPath = Path.Combine(
            Paths.GameRootPath,
            "REK_Data",
            "il2cpp_data",
            "Metadata",
            "global-metadata.dat");
        _gameAssemblySha256 = HashFile(gameAssemblyPath);
        _metadataSha256 = HashFile(metadataPath);
        _pluginSha256 = HashFile(Assembly.GetExecutingAssembly().Location);
        _scheduleSha256 = HashText(BridgeScheduleContract.CanonicalJson);

        if (!string.Equals(
                _scheduleSha256,
                BridgeScheduleContract.ExpectedSha256,
                StringComparison.Ordinal) ||
            !ValidateEmbeddedSchedule())
        {
            Log.LogError($"Control bridge disabled: embedded schedule contract mismatch {_scheduleSha256}.");
            return;
        }

        if (!string.Equals(_gameAssemblySha256, ExpectedGameAssemblySha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(_metadataSha256, ExpectedMetadataSha256, StringComparison.OrdinalIgnoreCase))
        {
            Log.LogError(
                $"UI bridge disabled: build hash mismatch. " +
                $"GameAssembly={_gameAssemblySha256} metadata={_metadataSha256}");
            return;
        }

        Instance = this;
        _harmony = new Harmony(PluginGuid);
        _harmony.PatchAll(typeof(Plugin).Assembly);
        _sendBoundaryPatchesVerified =
            HasOwnedPatch(typeof(RobotInputController), "LateUpdate") &&
            HasOwnedPatch(typeof(RobotInputController), "SendVelocityCommand") &&
            HasOwnedPatch(typeof(RobotInputController), "SendMoveEvent");
        if (!_sendBoundaryPatchesVerified)
        {
            Log.LogError("Control bridge disabled: exact send-boundary Harmony ownership was not verified.");
            _harmony.UnpatchSelf();
            _harmony = null;
            Instance = null;
            return;
        }
        _behaviour = AddComponent<BridgeBehaviour>();
        _pipe = new LocalPipeServer(
            PipeName,
            TryAcceptRequest,
            message => Log.LogInfo(message),
            message => Log.LogWarning(message));
        _pipe.Start();
        Log.LogInfo(
            $"Build-pinned evidence control bridge listening on local current-user pipe {PipeName}. " +
            "Global input remains unavailable. Semantic commands require an exclusive connection lease " +
            "and private Bot 1 scope is rechecked on the Unity main thread.");
    }

    public override bool Unload()
    {
        _pipe?.Dispose();
        _pipe = null;
        StopSchedule("plugin_unload");
        _harmony?.UnpatchSelf();
        _harmony = null;
        _sendBoundaryPatchesVerified = false;
        _leaseConnectionId = 0;
        if (_behaviour is not null)
        {
            UnityEngine.Object.Destroy(_behaviour);
            _behaviour = null;
        }
        Instance = null;
        return true;
    }

    private bool TryAcceptRequest(BridgeRequest request)
    {
        var dedupeKey = $"{request.ConnectionId}:{request.RequestId}";
        if (!_remembered.TryAdd(dedupeKey, 0))
            return false;

        var count = Interlocked.Increment(ref _pendingCount);
        if (count > MaxPendingRequests)
        {
            Interlocked.Decrement(ref _pendingCount);
            _remembered.TryRemove(dedupeKey, out _);
            return false;
        }

        _rememberedOrder.Enqueue(dedupeKey);
        while (_remembered.Count > MaxRememberedRequests && _rememberedOrder.TryDequeue(out var expired))
            _remembered.TryRemove(expired, out _);
        _pending.Enqueue(request);
        return true;
    }

    internal void OnUnityUpdate()
    {
        var processed = 0;
        while (processed < 8 && _pending.TryDequeue(out var request))
        {
            processed++;
            Interlocked.Decrement(ref _pendingCount);
            if (request.Kind == RequestKind.GetState)
            {
                var state = CaptureState(request.RequestId);
                _pipe?.Send(request.ConnectionId, state.Payload);
                continue;
            }

            if (request.Kind == RequestKind.Command && request.Command is not null)
            {
                var result = ApplyCommand(request.ConnectionId, request.Command.Value);
                var measuredPairing = result.MeasuredPairing;
                if (request.Command.Value == BridgeCommand.StartMeasuredSchedule && measuredPairing is null)
                    measuredPairing = MeasuredPairingPayload(ReadMeasuredPairing(gameMenu: null));
                _pipe?.Send(request.ConnectionId, new
                {
                    @event = "ack",
                    protocol = "rek.ui_bridge.v1",
                    request_id = request.RequestId,
                    command = request.Command.Value.ToString(),
                    status = result.Status,
                    reason = result.Reason,
                    applied = result.Applied,
                    client_request_issued = result.ClientRequestIssued,
                    server_acceptance_observed = false,
                    lease_connection_id = _leaseConnectionId == 0 ? (long?)null : _leaseConnectionId,
                    schedule_run_id = _scheduleRunId,
                    schedule_id = BridgeScheduleContract.ScheduleId,
                    command_sequence_schema = BridgeScheduleContract.Schema,
                    command_sequence_sha256 = _scheduleSha256,
                    measured_pairing = measuredPairing,
                    unity_frame = Time.frameCount,
                    unity_time = Time.timeAsDouble,
                    unity_thread = "main",
                });
                continue;
            }

            _pipe?.Send(request.ConnectionId, new
            {
                @event = "ack",
                protocol = "rek.ui_bridge.v1",
                request_id = request.RequestId,
                command = request.Key?.ToString(),
                status = "rejected",
                reason = "verified_process_targeted_unity_input_delivery_not_implemented",
                applied = false,
                unity_frame = Time.frameCount,
                unity_time = Time.timeAsDouble,
                unity_thread = "main",
            });
        }
    }

    internal void OnUnityFixedUpdate()
    {
        try
        {
            var connectionId = _pipe?.CurrentConnectionId ?? 0;
            if (_leaseConnectionId != 0 && connectionId != _leaseConnectionId)
            {
                StopSchedule("lease_connection_lost");
                _leaseConnectionId = 0;
            }

            if (!_scheduleRunning)
                return;

            if (!RequireOwnedScheduleControl(out var foregroundReason))
            {
                StopSchedule(foregroundReason);
                return;
            }
            var atScheduleBoundary =
                _scheduleFixedSubstep % BridgeScheduleContract.FixedSubstepsPerScheduleTick == 0;
            var input = _scheduleInput;
            if (atScheduleBoundary)
            {
                if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var reason))
                {
                    StopSchedule($"scope_lost:{reason}");
                    return;
                }
                if (!ScopeMatchesSchedule(scope))
                {
                    StopSchedule("scheduled_round_identity_changed");
                    return;
                }
                var measuredPairing = ReadMeasuredPairing(
                    scope.Coordinator,
                    scope.LocalSlot,
                    scope.OpponentSlot);
                if (!measuredPairing.Validation.ExactT800VersusT800)
                {
                    StopSchedule($"scheduled_pairing_lost:{measuredPairing.Validation.Reason}");
                    return;
                }
                input = scope.Input;
                _scheduleTick = _scheduleFixedSubstep / BridgeScheduleContract.FixedSubstepsPerScheduleTick;
            }

            if (input is null || NativePointer(input) != _scheduleInputPointer)
            {
                StopSchedule("local_input_controller_changed");
                return;
            }

            while (atScheduleBoundary &&
                   _nextScheduleStep < MeasuredSchedule.Length &&
                   MeasuredSchedule[_nextScheduleStep].Tick == _scheduleTick)
            {
                var step = MeasuredSchedule[_nextScheduleStep++];
                _scheduleVelocity = step.Velocity;
                if (step.Tick == BridgeScheduleContract.FinalScheduleTick)
                    _scheduleFinalNeutralSendObserved = false;
                if (!SetVelocityExact(input, _scheduleVelocity))
                {
                    StopSchedule("velocity_readback_mismatch");
                    return;
                }

                if (step.MoveIndex is not null && _scheduleMoveAwaitingSend is not null)
                {
                    StopSchedule("previous_move_not_observed_at_late_update_boundary");
                    return;
                }
                if (step.MoveIndex is not null &&
                    (input.hasPendingMove || input.hasPendingSpecial || input.hasPendingEStop))
                {
                    StopSchedule("conflicting_pending_command_before_scheduled_move");
                    return;
                }
                var moveAcceptedLocally = step.MoveIndex is null ||
                                          input.ExecuteMoveByIndex(step.MoveIndex.Value);
                if (step.MoveIndex is not null && moveAcceptedLocally)
                {
                    _scheduleMoveAwaitingSend = step.MoveIndex;
                    _scheduleMoveScheduleTick = step.Tick;
                    _scheduleMoveArmedObserved = false;
                    _scheduleMoveInvocationObserved = false;
                }
                var stepEvent = new
                {
                    @event = "schedule_step",
                    protocol = "rek.ui_bridge.v1",
                    schedule_id = BridgeScheduleContract.ScheduleId,
                    command_sequence_schema = BridgeScheduleContract.Schema,
                    command_sequence_sha256 = _scheduleSha256,
                    schedule_run_id = _scheduleRunId,
                    schedule_tick = _scheduleTick,
                    client_fixed_substep = _scheduleFixedSubstep,
                    fixed_substeps_per_schedule_tick = BridgeScheduleContract.FixedSubstepsPerScheduleTick,
                    label = step.Label,
                    velocity_command_xyz = new[] { step.Velocity.x, step.Velocity.y, step.Velocity.z },
                    move_index = step.MoveIndex,
                    move_accepted_locally = moveAcceptedLocally,
                    server_acceptance_observed = false,
                    unity_frame = Time.frameCount,
                    unity_fixed_time = Time.fixedTimeAsDouble,
                };
                _pipe?.Send(_leaseConnectionId, stepEvent);
                Log.LogInfo(JsonSerializer.Serialize(stepEvent, BridgeJson.Options));

                if (!moveAcceptedLocally)
                {
                    StopSchedule($"move_{step.MoveIndex}_rejected_locally");
                    return;
                }
            }

            var completedFixedSubsteps = _scheduleFixedSubstep + 1;
            var requiredFixedSubsteps = BridgeScheduleContract.DurationScheduleTicks *
                                        BridgeScheduleContract.FixedSubstepsPerScheduleTick;
            if (completedFixedSubsteps >= requiredFixedSubsteps)
            {
                if (!_scheduleFinalNeutralSendObserved)
                    return;
                if (_scheduleMoveAwaitingSend is not null ||
                    _scheduleMoveSendCompletedCount != MeasuredSchedule.Count(step => step.MoveIndex is not null))
                {
                    StopSchedule("completion_move_send_count_mismatch");
                    return;
                }
                if (!TryGetPrivateAiContext(requireActiveRound: true, out var endScope, out var endReason) ||
                    !ScopeMatchesSchedule(endScope))
                {
                    StopSchedule($"completion_scope_lost:{endReason}");
                    return;
                }
                StopSchedule("complete");
                return;
            }
            _scheduleFixedSubstep++;
        }
        catch (Exception exception)
        {
            StopSchedule($"fixed_update_control_failed:{exception.GetType().Name}");
        }
    }

    private CommandResult ApplyCommand(long connectionId, BridgeCommand command)
    {
        try
        {
            if (connectionId <= 0 || connectionId != (_pipe?.CurrentConnectionId ?? 0))
                return CommandResult.Rejected("stale_or_disconnected_pipe_connection");

            if (command == BridgeCommand.AcquireExclusiveControl)
            {
                if (_leaseConnectionId != 0 && _leaseConnectionId != connectionId)
                    return CommandResult.Rejected("exclusive_control_lease_held_by_another_connection");
                if (!RequireBackgroundControl(out var foregroundReason))
                    return CommandResult.Rejected(foregroundReason);
                _leaseConnectionId = connectionId;
                return CommandResult.AppliedResult("exclusive_control_lease_acquired");
            }

            if (command == BridgeCommand.ReleaseExclusiveControl)
            {
                if (_leaseConnectionId != connectionId)
                    return CommandResult.Rejected("exclusive_control_lease_not_owned");
                StopSchedule("lease_released");
                _leaseConnectionId = 0;
                return CommandResult.AppliedResult("exclusive_control_lease_released");
            }

            if (_leaseConnectionId != connectionId)
                return CommandResult.Rejected("exclusive_control_lease_required");

            return command switch
            {
                BridgeCommand.ConfirmLoggedIn => ConfirmLoggedIn(),
                BridgeCommand.NavigateFreePlay => NavigateFreePlay(),
                BridgeCommand.EnterSolo => EnterSolo(),
                BridgeCommand.StartRound => StartRound(),
                BridgeCommand.ExitUnexpectedPrivateAiSession => ExitUnexpectedPrivateAiSession(),
                BridgeCommand.ExitLostPrivateSession => ExitLostPrivateSession(),
                BridgeCommand.StartMeasuredSchedule => StartMeasuredSchedule(),
                BridgeCommand.StopMeasuredSchedule => StopMeasuredSchedule(),
                _ => CommandResult.Rejected("unknown_semantic_command"),
            };
        }
        catch (Exception exception)
        {
            return CommandResult.Rejected($"command_failed:{exception.GetType().Name}");
        }
    }

    private static CommandResult ConfirmLoggedIn()
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        var lobby = UnityEngine.Object.FindFirstObjectByType<LobbyShellController>();
        if (lobby is null)
            return CommandResult.Rejected("lobby_shell_not_found");
        if (lobby.CurrentScreen != LobbyShellController.Screen.Login)
            return CommandResult.Rejected($"expected_login_screen:observed_{lobby.CurrentScreen}");

        var login = lobby.loginScreen;
        var button = login?.letsGoButton;
        if (login is null || button is null || !button.visible || !button.enabledInHierarchy)
            return CommandResult.Rejected("lets_go_control_not_visible_and_enabled");

        login.OnLetsGoClicked();
        return lobby.CurrentScreen == LobbyShellController.Screen.Home
            ? CommandResult.AppliedResult("home_screen_observed_after_lets_go")
            : CommandResult.Rejected("lets_go_postcondition_not_observed");
    }

    private static CommandResult NavigateFreePlay()
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        var lobby = UnityEngine.Object.FindFirstObjectByType<LobbyShellController>();
        if (lobby is null)
            return CommandResult.Rejected("lobby_shell_not_found");
        if (lobby.CurrentScreen != LobbyShellController.Screen.Home)
            return CommandResult.Rejected($"expected_home_screen:observed_{lobby.CurrentScreen}");
        if (AnyConnectedSession())
            return CommandResult.Rejected("network_session_already_connected");

        var home = lobby.homeScreen;
        var button = home?.freePlayTile;
        if (home is null || button is null || !button.visible || !button.enabledInHierarchy)
            return CommandResult.Rejected("free_play_control_not_visible_and_enabled");

        home.OnFreePlayClicked();
        return lobby.CurrentScreen == LobbyShellController.Screen.FreePlay
            ? CommandResult.AppliedResult("free_play_screen_observed")
            : CommandResult.Rejected("free_play_postcondition_not_observed");
    }

    private static CommandResult EnterSolo()
    {
        return CommandResult.Rejected(
            "public_solo_route_disabled_private_arena_entry_not_proven");
    }

    private static CommandResult StartRound()
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        if (!TryGetPrivateAiContext(requireActiveRound: false, out var scope, out var reason))
            return CommandResult.Rejected(reason);
        if (scope.RoundActive)
            return CommandResult.Rejected("round_already_active");

        var view = scope.GameMenu?.menuView;
        var postFightPrompt = scope.GameMenu is not null && scope.GameMenu.IsMenuOpen && view is not null &&
                              view.CurrentPane == GameMenuView.Pane.PostFight &&
                              view.postFightContinueButton is not null &&
                              view.postFightContinueButton.visible &&
                              view.postFightContinueButton.enabledInHierarchy;
        if (postFightPrompt)
        {
            var wasWinner = scope.GameMenu!.postFightIsWinner;
            if (!wasWinner)
                return CommandResult.Rejected("post_fight_loser_requires_explicit_exit");
            scope.GameMenu!.HandlePostFightContinue();
            return CommandResult.RequestIssued("post_fight_continue_request_issued");
        }

        if (scope.Coordinator.CurrentPhase == FightPhase.Idle)
        {
            if (scope.GameMenu is null || scope.GameMenu.IsMenuOpen)
                return CommandResult.Rejected("idle_fight_setup_requires_closed_game_menu");
            if (scope.Coordinator.IsRemoteDriven)
            {
                scope.Coordinator.OnFightButtonNetwork();
                return CommandResult.RequestIssued("remote_ready_request_issued");
            }
            if (scope.Coordinator.draining)
                return CommandResult.Rejected("arena_is_draining");
            if (scope.Coordinator.setupRoutine is not null)
                return CommandResult.Rejected("fight_setup_already_in_progress");
            scope.Coordinator.StartFight();
            return scope.Coordinator.setupRoutine is not null
                ? CommandResult.AppliedResult("native_start_fight_coroutine_observed")
                : CommandResult.Rejected("native_start_fight_postcondition_not_observed");
        }

        return CommandResult.Rejected(
            $"fight_phase_not_idle:observed_{scope.Coordinator.CurrentPhase}");
    }

    private static CommandResult ExitLostPrivateSession()
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        if (!TryGetPrivateAiContext(requireActiveRound: false, out var scope, out var reason))
            return CommandResult.Rejected(reason);
        if (scope.RoundActive)
            return CommandResult.Rejected("round_still_active");
        var view = scope.GameMenu?.menuView;
        var postFightPrompt = scope.GameMenu is not null && scope.GameMenu.IsMenuOpen && view is not null &&
                              view.CurrentPane == GameMenuView.Pane.PostFight &&
                              view.postFightContinueButton is not null &&
                              view.postFightContinueButton.visible &&
                              view.postFightContinueButton.enabledInHierarchy;
        if (!postFightPrompt)
            return CommandResult.Rejected("post_fight_prompt_not_observed");
        if (scope.GameMenu!.postFightIsWinner)
            return CommandResult.Rejected("winner_must_use_start_round");
        scope.GameMenu.HandlePostFightContinue();
        return CommandResult.RequestIssued("post_fight_loser_exit_request_issued");
    }

    private static CommandResult ExitUnexpectedPrivateAiSession()
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);

        var gameMenu = UnityEngine.Object.FindFirstObjectByType<GameMenuController>();
        var coordinator = gameMenu?.fightCoordinator ??
                          UnityEngine.Object.FindFirstObjectByType<FightCoordinator>();
        var network = gameMenu?.networkSession ??
                      UnityEngine.Object.FindFirstObjectByType<NetworkSession>();
        var context = GameContext.Instance ?? UnityEngine.Object.FindFirstObjectByType<GameContext>();
        if (gameMenu is null || coordinator is null)
            return CommandResult.Rejected("arena_session_controllers_not_found");
        if (network is null || !network.IsConnected || !network.IsClient || network.IsServer)
            return CommandResult.Rejected("client_only_connected_network_session_not_proven");
        if (context is null || !context.IsSolo || context.IsRanked || context.AutoFindMatch ||
            string.IsNullOrWhiteSpace(context.ArenaID) || coordinator.IsRankedArena)
            return CommandResult.Rejected("private_unranked_solo_arena_not_proven");

        var localSlot = coordinator.LocalFighterIndex;
        if (localSlot is < 0 or > 1)
            return CommandResult.Rejected("invalid_local_fighter_slot");
        var opponentSlot = 1 - localSlot;
        var slotHasClient = coordinator.slotHasClient;
        if (slotHasClient is null || slotHasClient.Length <= opponentSlot)
            return CommandResult.Rejected("opponent_client_occupancy_unknown");
        if (slotHasClient[opponentSlot] ||
            (coordinator.clientHumanSlotMask & (1 << opponentSlot)) != 0 ||
            coordinator.HumanInSlot(opponentSlot) ||
            !coordinator.OpponentIsAI ||
            !coordinator.SlotIsAI(opponentSlot))
            return CommandResult.Rejected("ai_only_opponent_not_proven");
        if (coordinator.clientAiDifficultyLevel == 0 && coordinator.SparringBotNumber == 1)
            return CommandResult.Rejected("exact_sparring_bot_1_session_must_not_be_discarded");
        if (coordinator.CurrentRound is { IsActive: true })
            return CommandResult.Rejected("round_still_active");
        if (coordinator.CurrentPhase != FightPhase.Idle)
            return CommandResult.Rejected($"fight_phase_not_idle:observed_{coordinator.CurrentPhase}");

        gameMenu.HandleExitHomeRequested();
        return CommandResult.RequestIssued("unexpected_private_ai_session_exit_home_requested");
    }

    private CommandResult StartMeasuredSchedule()
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        if (_scheduleRunning)
            return CommandResult.Rejected("schedule_already_running");
        if (!SameFloatBits(Time.fixedDeltaTime, BridgeScheduleContract.ExpectedFixedDeltaTime))
            return CommandResult.Rejected($"unexpected_fixed_delta_time:{Time.fixedDeltaTime:R}");
        if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var reason))
            return CommandResult.Rejected(reason);
        var measuredPairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        var measuredPairingPayload = MeasuredPairingPayload(measuredPairing);
        if (!measuredPairing.Validation.ExactT800VersusT800)
        {
            return CommandResult.Rejected(
                $"required_t800_vs_t800_pairing_not_proven:{measuredPairing.Validation.Reason}",
                measuredPairingPayload);
        }
        if (scope.Input is null || !scope.Input.IsActive || !scope.Input.networkInitialized)
            return CommandResult.Rejected("local_input_controller_not_active_and_network_initialized");
        if (scope.Input.hasPendingMove || scope.Input.hasPendingSpecial || scope.Input.hasPendingEStop)
            return CommandResult.Rejected("local_input_controller_has_pending_command");
        if (!_sendBoundaryPatchesVerified)
            return CommandResult.Rejected("send_boundary_patches_not_verified");

        _scheduleRunId = Guid.NewGuid().ToString("N");
        _scheduleTick = 0;
        _scheduleFixedSubstep = 0;
        _nextScheduleStep = 0;
        _scheduleVelocity = Vector3.zero;
        _scheduleInput = scope.Input;
        _scheduleInputPointer = NativePointer(scope.Input);
        _scheduleMoveAwaitingSend = null;
        _scheduleMoveScheduleTick = null;
        _scheduleMoveArmedObserved = false;
        _scheduleMoveInvocationObserved = false;
        _scheduleMoveSendCompletedCount = 0;
        _scheduleFinalNeutralInvocationObserved = false;
        _scheduleFinalNeutralSendObserved = false;
        _scheduleFightEpoch = scope.FightEpoch;
        _scheduleRoundNumber = scope.RoundNumber;
        _scheduleIdentity = RuntimeIdentity.From(scope);
        _scheduleAuthorizedWhileBackground = true;
        _scheduleRunning = true;
        Log.LogInfo(
            $"Measured schedule started run={_scheduleRunId} sha256={_scheduleSha256} " +
            $"fixed_delta_time={Time.fixedDeltaTime:R} " +
            $"fixed_substeps_per_schedule_tick={BridgeScheduleContract.FixedSubstepsPerScheduleTick}");
        return CommandResult.AppliedResult("measured_schedule_started", measuredPairingPayload);
    }

    private CommandResult StopMeasuredSchedule()
    {
        if (!_scheduleRunning)
            return CommandResult.Rejected("schedule_not_running");
        StopSchedule("requested");
        return CommandResult.AppliedResult("measured_schedule_stopped");
    }

    private void StopSchedule(string reason)
    {
        var wasRunning = _scheduleRunning;
        var runId = _scheduleRunId;
        var authorizedWhileBackground = _scheduleAuthorizedWhileBackground;
        if (!wasRunning)
            return;
        _scheduleRunning = false;
        _scheduleAuthorizedWhileBackground = false;
        TryCancelExactOwnedMove();
        TryNeutralOwnedController();
        _scheduleVelocity = Vector3.zero;
        _scheduleInput = null;
        _scheduleInputPointer = IntPtr.Zero;
        _scheduleMoveAwaitingSend = null;
        _scheduleMoveScheduleTick = null;
        _scheduleMoveArmedObserved = false;
        _scheduleMoveInvocationObserved = false;
        var moveSendCompletedCount = _scheduleMoveSendCompletedCount;
        var finalNeutralSendObserved = _scheduleFinalNeutralSendObserved;
        _scheduleMoveSendCompletedCount = 0;
        _scheduleFinalNeutralInvocationObserved = false;
        _scheduleFinalNeutralSendObserved = false;
        _scheduleFightEpoch = 0;
        _scheduleRoundNumber = 0;
        _scheduleIdentity = null;

        try
        {
            var payload = new
            {
                @event = "schedule_end",
                protocol = "rek.ui_bridge.v1",
                schedule_id = BridgeScheduleContract.ScheduleId,
                command_sequence_schema = BridgeScheduleContract.Schema,
                command_sequence_sha256 = _scheduleSha256,
                schedule_run_id = runId,
                schedule_tick = _scheduleTick,
                client_fixed_substep = _scheduleFixedSubstep,
                move_send_completed_count = moveSendCompletedCount,
                final_neutral_send_observed = finalNeutralSendObserved,
                reason,
                complete = string.Equals(reason, "complete", StringComparison.Ordinal),
                authorized_while_background = authorizedWhileBackground,
                server_acceptance_observed = false,
                unity_frame = Time.frameCount,
                unity_fixed_time = Time.fixedTimeAsDouble,
            };
            _pipe?.Send(_leaseConnectionId, payload);
            Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        }
        catch
        {
        }
    }

    private static bool SetVelocityExact(RobotInputController input, Vector3 velocity)
    {
        input.VelocityCommand = velocity;
        var observed = input.VelocityCommand;
        return SameFloatBits(observed.x, velocity.x) &&
               SameFloatBits(observed.y, velocity.y) &&
               SameFloatBits(observed.z, velocity.z);
    }

    internal void OnRobotInputLateUpdatePrefix(RobotInputController input)
    {
        if (!_scheduleRunning ||
            _leaseConnectionId == 0 ||
            _leaseConnectionId != (_pipe?.CurrentConnectionId ?? 0) ||
            _scheduleInputPointer == IntPtr.Zero ||
            NativePointer(input) != _scheduleInputPointer)
        {
            return;
        }

        try
        {
            if (!RequireOwnedScheduleControl(out var foregroundReason))
            {
                StopSchedule(foregroundReason);
                return;
            }
            if (_scheduleMoveAwaitingSend is null)
                return;
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var scopeReason) ||
                !ScopeMatchesSchedule(scope))
            {
                StopSchedule($"late_update_scope_lost:{scopeReason}");
                return;
            }

            var moveIndex = _scheduleMoveAwaitingSend.Value;
            if (input.hasPendingSpecial || input.hasPendingEStop)
            {
                StopSchedule("conflicting_pending_special_or_estop_at_late_update_boundary");
                return;
            }
            if (input.hasPendingMove && input.pendingMoveIndex != moveIndex)
            {
                StopSchedule("conflicting_pending_move_at_late_update_boundary");
                return;
            }
            var alreadyArmed = input.hasPendingMove && input.pendingMoveIndex == moveIndex;
            if (!alreadyArmed && !input.ExecuteMoveByIndex(moveIndex))
            {
                StopSchedule($"move_{moveIndex}_rejected_at_late_update_boundary");
                return;
            }
            if (!input.hasPendingMove || input.pendingMoveIndex != moveIndex)
            {
                StopSchedule($"move_{moveIndex}_not_armed_at_late_update_boundary");
                return;
            }

            if (!_scheduleMoveArmedObserved)
            {
                var payload = ScheduleMoveEvent("schedule_move_armed", input, moveIndex);
                _pipe?.Send(_leaseConnectionId, payload);
                Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
                _scheduleMoveArmedObserved = true;
            }
        }
        catch (Exception exception)
        {
            StopSchedule($"late_update_control_failed:{exception.GetType().Name}");
        }
    }

    internal void OnSendVelocityCommandPrefix(RobotInputController input)
    {
        if (!_scheduleRunning ||
            _leaseConnectionId == 0 ||
            _leaseConnectionId != (_pipe?.CurrentConnectionId ?? 0) ||
            _scheduleInputPointer == IntPtr.Zero ||
            NativePointer(input) != _scheduleInputPointer)
        {
            return;
        }

        try
        {
            if (!RequireOwnedScheduleControl(out var foregroundReason))
            {
                StopSchedule(foregroundReason);
                return;
            }
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var scopeReason) ||
                !ScopeMatchesSchedule(scope))
            {
                StopSchedule($"send_boundary_scope_lost:{scopeReason}");
                return;
            }
            if (!SetVelocityExact(input, _scheduleVelocity))
            {
                StopSchedule("velocity_readback_mismatch_at_send_boundary");
                return;
            }
            if (_scheduleTick == BridgeScheduleContract.FinalScheduleTick &&
                SameFloatBits(_scheduleVelocity.x, 0f) &&
                SameFloatBits(_scheduleVelocity.y, 0f) &&
                SameFloatBits(_scheduleVelocity.z, 0f))
            {
                _scheduleFinalNeutralInvocationObserved = true;
            }
        }
        catch (Exception exception)
        {
            StopSchedule($"send_boundary_control_failed:{exception.GetType().Name}");
        }
    }

    internal void OnSendVelocityCommandPostfix(RobotInputController input)
    {
        if (_scheduleRunning && _scheduleFinalNeutralInvocationObserved &&
            NativePointer(input) == _scheduleInputPointer)
        {
            _scheduleFinalNeutralInvocationObserved = false;
            _scheduleFinalNeutralSendObserved = true;
        }
    }

    internal void OnSendVelocityCommandFailure(RobotInputController input, Exception exception)
    {
        if (_scheduleRunning && _scheduleFinalNeutralInvocationObserved &&
            NativePointer(input) == _scheduleInputPointer)
        {
            _scheduleFinalNeutralInvocationObserved = false;
            StopSchedule($"velocity_send_failed:{exception.GetType().Name}");
        }
    }

    internal bool OnSendMoveEventPrefix(RobotInputController input)
    {
        if (!_scheduleRunning ||
            _scheduleMoveAwaitingSend is null ||
            _scheduleInputPointer == IntPtr.Zero ||
            NativePointer(input) != _scheduleInputPointer)
        {
            return true;
        }
        try
        {
            var moveIndex = _scheduleMoveAwaitingSend.Value;
            if (!RequireOwnedScheduleControl(out var foregroundReason))
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex;
                StopSchedule(foregroundReason);
                return allowUnownedSend;
            }
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var scopeReason))
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex;
                StopSchedule($"move_send_scope_lost:{scopeReason}");
                return allowUnownedSend;
            }
            if (!ScopeMatchesSchedule(scope) ||
                !input.hasPendingMove || input.pendingMoveIndex != moveIndex)
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex;
                StopSchedule("move_send_scope_or_pending_mismatch");
                return allowUnownedSend;
            }
            _scheduleMoveInvocationObserved = true;
            var payload = ScheduleMoveEvent("schedule_move_send_invoked", input, moveIndex);
            _pipe?.Send(_leaseConnectionId, payload);
            Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
            return true;
        }
        catch (Exception exception)
        {
            var moveIndex = _scheduleMoveAwaitingSend;
            var allowUnownedSend = moveIndex is null || input.pendingMoveIndex != moveIndex.Value;
            StopSchedule($"move_send_prefix_failed:{exception.GetType().Name}");
            return allowUnownedSend;
        }
    }

    internal void OnSendMoveEventPostfix(RobotInputController input)
    {
        if (!_scheduleRunning || !_scheduleMoveInvocationObserved ||
            _scheduleMoveAwaitingSend is null ||
            NativePointer(input) != _scheduleInputPointer)
        {
            return;
        }
        var moveIndex = _scheduleMoveAwaitingSend.Value;
        var payload = ScheduleMoveEvent("schedule_move_send_completed", input, moveIndex);
        _pipe?.Send(_leaseConnectionId, payload);
        Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        _scheduleMoveAwaitingSend = null;
        _scheduleMoveScheduleTick = null;
        _scheduleMoveArmedObserved = false;
        _scheduleMoveInvocationObserved = false;
        _scheduleMoveSendCompletedCount++;
    }

    internal void OnSendMoveEventFailure(RobotInputController input, Exception exception)
    {
        if (_scheduleRunning && _scheduleMoveInvocationObserved &&
            NativePointer(input) == _scheduleInputPointer)
        {
            StopSchedule($"move_send_failed:{exception.GetType().Name}");
        }
    }

    private object ScheduleMoveEvent(string eventName, RobotInputController input, int moveIndex) => new
    {
        @event = eventName,
        protocol = "rek.ui_bridge.v1",
        schedule_id = BridgeScheduleContract.ScheduleId,
        command_sequence_schema = BridgeScheduleContract.Schema,
        command_sequence_sha256 = _scheduleSha256,
        schedule_run_id = _scheduleRunId,
        schedule_tick = _scheduleMoveScheduleTick,
        move_index = moveIndex,
        pending_move_readback = input.hasPendingMove,
        pending_move_index_readback = input.pendingMoveIndex,
        server_acceptance_observed = false,
        unity_frame = Time.frameCount,
        unity_time = Time.timeAsDouble,
    };

    private static bool SameFloatBits(float left, float right) =>
        BitConverter.SingleToInt32Bits(left) == BitConverter.SingleToInt32Bits(right);

    private static bool ValidateEmbeddedSchedule()
    {
        try
        {
            using var document = JsonDocument.Parse(BridgeScheduleContract.CanonicalJson);
            var root = document.RootElement;
            if (root.GetProperty("schema").GetString() != BridgeScheduleContract.Schema ||
                root.GetProperty("schedule_id").GetString() != BridgeScheduleContract.ScheduleId ||
                root.GetProperty("duration_ticks").GetInt32() != BridgeScheduleContract.DurationScheduleTicks ||
                root.GetProperty("unity_fixed_rate_hz").GetInt32() != BridgeScheduleContract.UnityFixedRateHz ||
                root.GetProperty("schedule_rate_hz").GetInt32() != BridgeScheduleContract.ScheduleRateHz ||
                root.GetProperty("fixed_substeps_per_tick").GetInt32() !=
                    BridgeScheduleContract.FixedSubstepsPerScheduleTick)
            {
                return false;
            }

            var componentOrder = root.GetProperty("velocity_component_order");
            if (componentOrder.GetArrayLength() != 3 ||
                componentOrder[0].GetString() != "forward" ||
                componentOrder[1].GetString() != "strafe" ||
                componentOrder[2].GetString() != "yaw")
            {
                return false;
            }

            var previousTick = -1;
            foreach (var step in MeasuredSchedule)
            {
                if (step.Tick <= previousTick || step.Tick < 0 ||
                    step.Tick >= BridgeScheduleContract.DurationScheduleTicks)
                {
                    return false;
                }
                previousTick = step.Tick;
            }
            if (MeasuredSchedule[0].Tick != 0 ||
                MeasuredSchedule[^1].Tick != BridgeScheduleContract.FinalScheduleTick)
            {
                return false;
            }

            var manifestMoves = root.GetProperty("move_commands").EnumerateArray().ToArray();
            var embeddedMoves = MeasuredSchedule.Where(step => step.MoveIndex is not null).ToArray();
            if (manifestMoves.Length != embeddedMoves.Length)
                return false;
            for (var index = 0; index < manifestMoves.Length; index++)
            {
                if (manifestMoves[index].GetProperty("tick").GetInt32() != embeddedMoves[index].Tick ||
                    manifestMoves[index].GetProperty("move_index").GetInt32() != embeddedMoves[index].MoveIndex)
                {
                    return false;
                }
            }

            var segments = root.GetProperty("velocity_segments").EnumerateArray().ToArray();
            var cursor = 0;
            foreach (var segment in segments)
            {
                var start = segment.GetProperty("start").GetInt32();
                var stop = segment.GetProperty("stop").GetInt32();
                if (start != cursor || stop <= start || stop > BridgeScheduleContract.DurationScheduleTicks)
                    return false;
                cursor = stop;
            }
            if (cursor != BridgeScheduleContract.DurationScheduleTicks)
                return false;

            var embeddedVelocity = Vector3.zero;
            var stepIndex = 0;
            var segmentIndex = 0;
            for (var tick = 0; tick < BridgeScheduleContract.DurationScheduleTicks; tick++)
            {
                while (stepIndex < MeasuredSchedule.Length && MeasuredSchedule[stepIndex].Tick <= tick)
                    embeddedVelocity = MeasuredSchedule[stepIndex++].Velocity;
                while (segmentIndex + 1 < segments.Length &&
                       segments[segmentIndex].GetProperty("stop").GetInt32() <= tick)
                {
                    segmentIndex++;
                }
                var vector = segments[segmentIndex].GetProperty("velocity_command");
                if (vector.GetArrayLength() != 3 ||
                    !SameFloatBits(vector[0].GetSingle(), embeddedVelocity.x) ||
                    !SameFloatBits(vector[1].GetSingle(), embeddedVelocity.y) ||
                    !SameFloatBits(vector[2].GetSingle(), embeddedVelocity.z))
                {
                    return false;
                }
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool HasOwnedPatch(Type declaringType, string methodName)
    {
        var target = AccessTools.DeclaredMethod(declaringType, methodName);
        var patchInfo = target is null ? null : Harmony.GetPatchInfo(target);
        return patchInfo?.Owners?.Contains(PluginGuid, StringComparer.Ordinal) == true;
    }

    private bool ScopeMatchesSchedule(PrivateAiContext scope) =>
        _scheduleIdentity is not null && _scheduleIdentity.Equals(RuntimeIdentity.From(scope));

    private void TryNeutralOwnedController()
    {
        try
        {
            var input = _scheduleInput;
            if (input is not null && _scheduleInputPointer != IntPtr.Zero &&
                NativePointer(input) == _scheduleInputPointer)
            {
                input.VelocityCommand = Vector3.zero;
            }
        }
        catch
        {
        }
    }

    private void TryCancelExactOwnedMove()
    {
        try
        {
            var input = _scheduleInput;
            var moveIndex = _scheduleMoveAwaitingSend;
            if (input is not null && moveIndex is not null &&
                _scheduleInputPointer != IntPtr.Zero &&
                NativePointer(input) == _scheduleInputPointer &&
                input.hasPendingMove && input.pendingMoveIndex == moveIndex.Value)
            {
                input.hasPendingMove = false;
            }
        }
        catch
        {
        }
    }

    private static bool RequireBackgroundControl(out string reason)
    {
        if (!TryReadForeground(out var isRekForeground))
        {
            reason = "foreground_process_could_not_be_verified";
            return false;
        }
        if (isRekForeground)
        {
            reason = "rek_is_foreground_human_control_may_be_active";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    private bool RequireOwnedScheduleControl(out string reason)
    {
        var connectionId = _pipe?.CurrentConnectionId ?? 0;
        if (!_scheduleRunning || !_scheduleAuthorizedWhileBackground)
        {
            reason = "schedule_not_authorized_while_background";
            return false;
        }
        if (_leaseConnectionId == 0 || connectionId != _leaseConnectionId)
        {
            reason = "exclusive_schedule_lease_not_owned";
            return false;
        }
        if (!RequireBackgroundControl(out reason))
            return false;
        reason = string.Empty;
        return true;
    }

    private static bool TryReadForeground(out bool isRekForeground)
    {
        isRekForeground = false;
        var window = GetForegroundWindow();
        if (window == IntPtr.Zero)
            return false;
        _ = GetWindowThreadProcessId(window, out var processId);
        if (processId == 0)
            return false;
        isRekForeground = processId == (uint)Environment.ProcessId;
        return true;
    }

    private static bool AnyConnectedSession()
    {
        try
        {
            var network = UnityEngine.Object.FindFirstObjectByType<NetworkSession>();
            return network is not null && network.IsConnected;
        }
        catch
        {
            return true;
        }
    }

    private static bool TryGetPrivateAiContext(
        bool requireActiveRound,
        out PrivateAiContext scope,
        out string reason)
    {
        scope = null!;
        reason = "private_ai_scope_not_proven";
        try
        {
            var gameMenu = UnityEngine.Object.FindFirstObjectByType<GameMenuController>();
            var coordinator = gameMenu?.fightCoordinator ??
                              UnityEngine.Object.FindFirstObjectByType<FightCoordinator>();
            var network = gameMenu?.networkSession ??
                          UnityEngine.Object.FindFirstObjectByType<NetworkSession>();
            var context = GameContext.Instance ?? UnityEngine.Object.FindFirstObjectByType<GameContext>();
            if (coordinator is null)
            {
                reason = "fight_coordinator_not_found";
                return false;
            }
            if (network is null || !network.IsConnected || !network.IsClient || network.IsServer)
            {
                reason = "client_only_connected_network_session_not_proven";
                return false;
            }
            if (context is null || !context.IsSolo)
            {
                reason = "solo_context_not_proven";
                return false;
            }
            if (context.IsRanked || context.AutoFindMatch || string.IsNullOrWhiteSpace(context.ArenaID))
            {
                reason = "private_unranked_arena_identity_not_proven";
                return false;
            }
            if (!TryReadMultiplayerSessionPrivate(out var sessionIsPrivate, out var sessionPrivacyReason) ||
                !sessionIsPrivate)
            {
                reason = sessionPrivacyReason;
                return false;
            }
            if (coordinator.IsRankedArena)
            {
                reason = "ranked_coordinator_rejected";
                return false;
            }
            if (coordinator.clientAiDifficultyLevel != 0)
            {
                reason = "unexpected_sparring_bot_difficulty";
                return false;
            }
            if (string.IsNullOrWhiteSpace(network.serverAddress) || network.port <= 0)
            {
                reason = "network_endpoint_identity_not_proven";
                return false;
            }

            var localSlot = coordinator.LocalFighterIndex;
            if (localSlot is < 0 or > 1)
            {
                reason = "invalid_local_fighter_slot";
                return false;
            }
            var opponentSlot = 1 - localSlot;
            var slotHasClient = coordinator.slotHasClient;
            if (slotHasClient is null || slotHasClient.Length <= opponentSlot)
            {
                reason = "opponent_client_occupancy_unknown";
                return false;
            }
            if (slotHasClient[opponentSlot] ||
                (coordinator.clientHumanSlotMask & (1 << opponentSlot)) != 0 ||
                coordinator.HumanInSlot(opponentSlot) ||
                !coordinator.OpponentIsAI ||
                !coordinator.SlotIsAI(opponentSlot) ||
                coordinator.SparringBotNumber != 1)
            {
                reason = "exact_sparring_bot_1_scope_not_proven";
                return false;
            }

            var round = coordinator.CurrentRound;
            var roundActive = round is not null && round.IsActive;
            if (requireActiveRound &&
                (!roundActive || coordinator.CurrentPhase != FightPhase.RoundActive))
            {
                reason = "active_round_not_observed";
                return false;
            }

            RobotInputController? input = null;
            if (requireActiveRound)
            {
                var fighters = coordinator.Fighters;
                if (fighters is null || fighters.Length < 2 || fighters[0] is null || fighters[1] is null)
                {
                    reason = "two_fighters_not_observed";
                    return false;
                }
                if (!fighters[0].IsVisualOnly || !fighters[1].IsVisualOnly)
                {
                    reason = "client_visual_only_fighter_pair_not_proven";
                    return false;
                }

                input = coordinator.robotInput;
                if (input is null || input.Robot is null || input.networkIndex != localSlot ||
                    NativePointer(input.Robot) != NativePointer(fighters[localSlot]))
                {
                    reason = "local_input_controller_fighter_binding_not_proven";
                    return false;
                }
                if (!input.IsActive || !input.networkInitialized || input.hasPendingEStop)
                {
                    reason = "local_input_controller_not_ready_for_control";
                    return false;
                }
            }

            scope = new PrivateAiContext(
                gameMenu,
                coordinator,
                input,
                roundActive,
                localSlot,
                opponentSlot,
                coordinator.fightEpoch,
                round?.RoundNumber ?? -1,
                network,
                context,
                round);
            reason = "exact_sparring_bot_1_scope_proven";
            return true;
        }
        catch (Exception exception)
        {
            reason = $"private_ai_scope_probe_failed:{exception.GetType().Name}";
            return false;
        }
    }

    internal void OnUnityLateUpdate()
    {
        var connectionId = _pipe?.CurrentConnectionId ?? 0;
        if (connectionId <= 0)
            return;

        var state = CaptureState(requestId: null);
        if (connectionId == _lastPublishedConnection &&
            string.Equals(state.Identity, _lastStateIdentity, StringComparison.Ordinal))
        {
            return;
        }

        _lastPublishedConnection = connectionId;
        _lastStateIdentity = state.Identity;
        _pipe?.Send(connectionId, state.Payload);
    }

    private static bool TryReadMultiplayerSessionPrivate(
        out bool isPrivate,
        out string reason)
    {
        isPrivate = false;
        try
        {
            var manager = UnityEngine.Object.FindFirstObjectByType<XRMultiplayer.SessionManager>();
            if (manager is null)
            {
                reason = "multiplayer_session_manager_not_found";
                return false;
            }
            var session = manager.currentSession;
            if (session is null)
            {
                reason = "multiplayer_current_session_not_found";
                return false;
            }
            isPrivate = session.IsPrivate;
            reason = isPrivate
                ? "multiplayer_session_private"
                : "multiplayer_session_public_rejected";
            return true;
        }
        catch (Exception exception)
        {
            reason = $"multiplayer_session_privacy_probe_failed:{exception.GetType().Name}";
            return false;
        }
    }

    private CapturedState CaptureState(string? requestId)
    {
        LobbyShellController? lobby = null;
        GameMenuController? gameMenu = null;
        try
        {
            lobby = UnityEngine.Object.FindFirstObjectByType<LobbyShellController>();
        }
        catch
        {
        }
        try
        {
            gameMenu = UnityEngine.Object.FindFirstObjectByType<GameMenuController>();
        }
        catch
        {
        }

        var lobbyScreen = TryRead(() => lobby?.CurrentScreen.ToString());
        var menuOpen = TryReadNullable(() => gameMenu?.IsMenuOpen);
        var menuPane = TryRead(() => gameMenu?.menuView?.CurrentPane.ToString());
        var toolkitFocus = ReadToolkitFocus(lobby, gameMenu);
        var uguiFocus = ReadUguiFocus();
        var home = ReadHomeState(lobby, toolkitFocus.Pointer);
        var privateAi = ReadPrivateAiProof(gameMenu);
        var measuredPairing = MeasuredPairingPayload(ReadMeasuredPairing(gameMenu));
        var scene = TryRead(() => SceneManager.GetActiveScene().name);
        var foregroundKnown = TryReadForeground(out var rekForeground);

        var controlIdentity = new
        {
            lease_held = _leaseConnectionId != 0,
            schedule_running = _scheduleRunning,
            schedule_authorized_while_background = _scheduleAuthorizedWhileBackground,
            schedule_run_id = _scheduleRunId,
        };
        var identityObject = new
        {
            scene,
            lobby_screen = lobbyScreen,
            game_menu_open = menuOpen,
            game_menu_pane = menuPane,
            toolkit_focus = toolkitFocus.Payload,
            ugui_focus = uguiFocus,
            home,
            private_ai = privateAi,
            measured_pairing = measuredPairing,
            foreground_known = foregroundKnown,
            rek_is_foreground = foregroundKnown ? rekForeground : (bool?)null,
            control = controlIdentity,
        };
        var identity = JsonSerializer.Serialize(identityObject, BridgeJson.Options);
        var sequence = Interlocked.Increment(ref _stateSequence);
        var payload = new
        {
            @event = "state",
            protocol = "rek.ui_bridge.v1",
            request_id = requestId,
            state_sequence = sequence,
            observed_utc = DateTimeOffset.UtcNow,
            unity_frame = Time.frameCount,
            unity_time = Time.timeAsDouble,
            unity_unscaled_time = Time.unscaledTimeAsDouble,
            unity_thread = "main",
            scene,
            application_version = Application.version,
            unity_version = Application.unityVersion,
            build = new
            {
                game_assembly_sha256 = _gameAssemblySha256,
                global_metadata_sha256 = _metadataSha256,
                plugin_sha256 = _pluginSha256,
                plugin_version = PluginVersion,
            },
            lobby_screen = lobbyScreen,
            game_menu_open = menuOpen,
            game_menu_pane = menuPane,
            focus = new
            {
                effective_source = toolkitFocus.Payload is not null ? "ui_toolkit" : uguiFocus is not null ? "ugui" : null,
                ui_toolkit = toolkitFocus.Payload,
                ugui = uguiFocus,
            },
            home,
            private_ai = privateAi,
            measured_pairing = measuredPairing,
            foreground = new
            {
                known = foregroundKnown,
                rek_is_foreground = foregroundKnown ? rekForeground : (bool?)null,
                mutation_allowed = foregroundKnown && !rekForeground,
            },
            control = new
            {
                semantic_available = true,
                exclusive_lease_required = true,
                lease_held = _leaseConnectionId != 0,
                lease_connection_id = _leaseConnectionId == 0 ? (long?)null : _leaseConnectionId,
                schedule_id = BridgeScheduleContract.ScheduleId,
                command_sequence_schema = BridgeScheduleContract.Schema,
                command_sequence_sha256 = _scheduleSha256,
                schedule_running = _scheduleRunning,
                schedule_authorized_while_background = _scheduleAuthorizedWhileBackground,
                schedule_tick = _scheduleTick,
                client_fixed_substep = _scheduleFixedSubstep,
                fixed_substeps_per_schedule_tick = BridgeScheduleContract.FixedSubstepsPerScheduleTick,
                schedule_run_id = _scheduleRunId,
                send_boundary_patches_verified = _sendBoundaryPatchesVerified,
            },
            input = new
            {
                global_input_available = false,
                semantic_commands_available = true,
                autonomous = _leaseConnectionId != 0,
                global_input_unavailable_reason = "global_keyboard_mouse_and_gamepad_injection_deliberately_unavailable",
            },
        };
        return new CapturedState(identity, payload);
    }

    private static ToolkitFocus ReadToolkitFocus(
        LobbyShellController? lobby,
        GameMenuController? gameMenu)
    {
        try
        {
            var roots = new[]
            {
                lobby?.root,
                gameMenu?.menuView?.root,
            };
            foreach (var root in roots)
            {
                var focusable = root?.panel?.focusController?.focusedElement;
                if (focusable is null)
                    continue;
                var element = focusable.TryCast<VisualElement>();
                if (element is null)
                    continue;

                var pointer = NativePointer(element);
                var textElement = element.TryCast<TextElement>();
                var rawText = textElement?.text;
                var textRedacted = ShouldRedactText(element.name, textElement);
                var bounds = element.worldBound;
                return new ToolkitFocus(pointer, new
                {
                    name = SafeString(element.name, 128),
                    type = element.GetType().FullName,
                    path = ToolkitPath(element),
                    text = textRedacted ? null : SafeString(rawText, 256),
                    text_redacted = textRedacted,
                    focusable = element.focusable,
                    tab_index = element.tabIndex,
                    enabled = element.enabledInHierarchy,
                    visible = element.visible,
                    world_bound = new { x = bounds.x, y = bounds.y, width = bounds.width, height = bounds.height },
                });
            }
        }
        catch
        {
        }
        return new ToolkitFocus(IntPtr.Zero, null);
    }

    private static object? ReadUguiFocus()
    {
        try
        {
            var selected = EventSystem.current?.currentSelectedGameObject;
            if (selected is null)
                return null;
            var selectable = selected.GetComponent<Selectable>();
            return new
            {
                name = SafeString(selected.name, 128),
                path = GameObjectPath(selected),
                instance_id = selected.GetInstanceID(),
                active_self = selected.activeSelf,
                active_in_hierarchy = selected.activeInHierarchy,
                selectable = selectable is not null,
                interactable = selectable?.interactable,
            };
        }
        catch
        {
            return null;
        }
    }

    private static object? ReadHomeState(LobbyShellController? lobby, IntPtr focusedPointer)
    {
        try
        {
            var home = lobby?.homeScreen;
            if (home is null)
                return null;
            return new
            {
                user_display_text = SafeString(home.userNameLabel?.text, 128),
                cards = new object?[]
                {
                    Card("profile", home.profileButton, focusedPointer),
                    Card("settings", home.settingsButton, focusedPointer),
                    Card("workshop", home.workshopTile, focusedPointer),
                    Card("championship", home.championshipTile, focusedPointer),
                    Card("free_play", home.freePlayTile, focusedPointer),
                    Card("inbox", home.inboxButton, focusedPointer),
                    Card("inbox_close", home.inboxCloseButton, focusedPointer),
                    Card("logout", home.logoutButton, focusedPointer),
                    Card("exit", home.exitButton, focusedPointer),
                },
            };
        }
        catch
        {
            return null;
        }
    }

    private static object? Card(string role, Button? button, IntPtr focusedPointer)
    {
        if (button is null)
            return null;
        return new
        {
            role,
            name = SafeString(button.name, 128),
            text = ShouldRedactName(button.name) ? null : SafeString(button.text, 256),
            text_redacted = ShouldRedactName(button.name),
            enabled = button.enabledInHierarchy,
            visible = button.visible,
            focused = focusedPointer != IntPtr.Zero && NativePointer(button) == focusedPointer,
        };
    }

    private static MeasuredPairing ReadMeasuredPairing(GameMenuController? gameMenu)
    {
        try
        {
            var coordinator = gameMenu?.fightCoordinator ??
                              UnityEngine.Object.FindFirstObjectByType<FightCoordinator>();
            if (coordinator is null)
                return UnavailableMeasuredPairing("fight_coordinator_not_found");
            var localSlot = coordinator.LocalFighterIndex;
            if (localSlot is < 0 or > 1)
                return UnavailableMeasuredPairing("invalid_local_fighter_slot");
            return ReadMeasuredPairing(coordinator, localSlot, 1 - localSlot);
        }
        catch (Exception exception)
        {
            return UnavailableMeasuredPairing($"pairing_probe_failed:{exception.GetType().Name}");
        }
    }

    private static MeasuredPairing ReadMeasuredPairing(
        FightCoordinator coordinator,
        int localSlot,
        int opponentSlot)
    {
        try
        {
            if (localSlot is < 0 or > 1 || opponentSlot != 1 - localSlot)
                return UnavailableMeasuredPairing("invalid_fighter_slot_pair", localSlot, opponentSlot);

            var identities = coordinator.fighterIdentities;
            FighterIdentity? localIdentity = identities is not null && identities.Length > localSlot
                ? identities[localSlot]
                : null;
            FighterIdentity? opponentIdentity = identities is not null && identities.Length > opponentSlot
                ? identities[opponentSlot]
                : null;

            var fighters = coordinator.Fighters;
            Robot? localRobot = fighters is not null && fighters.Length > localSlot
                ? fighters[localSlot]
                : null;
            Robot? opponentRobot = fighters is not null && fighters.Length > opponentSlot
                ? fighters[opponentSlot]
                : null;

            var localFighter = new MeasuredFighter(
                localSlot,
                localIdentity?.RobotID,
                localRobot is null ? null : SafeString(localRobot.name, 128),
                ReadOrderedBoneNames(localRobot));
            var opponentFighter = new MeasuredFighter(
                opponentSlot,
                opponentIdentity?.RobotID,
                opponentRobot is null ? null : SafeString(opponentRobot.name, 128),
                ReadOrderedBoneNames(opponentRobot));
            var validation = BridgePairingContract.Validate(
                localFighter.SemanticRobotId,
                localFighter.BoneNames,
                opponentFighter.SemanticRobotId,
                opponentFighter.BoneNames);
            return new MeasuredPairing(localSlot, opponentSlot, localFighter, opponentFighter, validation);
        }
        catch (Exception exception)
        {
            return UnavailableMeasuredPairing(
                $"pairing_probe_failed:{exception.GetType().Name}",
                localSlot,
                opponentSlot);
        }
    }

    private static IReadOnlyList<string?>? ReadOrderedBoneNames(Robot? robot)
    {
        if (robot is null)
            return null;
        var bones = robot.boneTransforms;
        if (bones is null)
            return null;
        var names = new List<string?>();
        for (var index = 0; index < bones.Length; index++)
            names.Add(bones[index]?.name);
        return names;
    }

    private static MeasuredPairing UnavailableMeasuredPairing(
        string reason,
        int? localSlot = null,
        int? opponentSlot = null) =>
        new(
            localSlot,
            opponentSlot,
            null,
            null,
            new PairingValidation(
                reason,
                ExactT800VersusT800: false,
                LocalSemanticT800: false,
                OpponentSemanticT800: false,
                LocalExactT800BoneSignature: false,
                OpponentExactT800BoneSignature: false));

    private static object MeasuredPairingPayload(MeasuredPairing pairing) => new
    {
        required_pairing = BridgePairingContract.RequiredPairing,
        required_robot_id = BridgePairingContract.RequiredRobotId,
        required_t800_bone_count = BridgePairingContract.T800BoneNames.Length,
        required_t800_bone_signature_sha256 = BridgePairingContract.T800BoneSignatureSha256,
        semantic_identity_source = "FightCoordinator.fighterIdentities[slot].RobotID",
        bone_signature_source = "FightCoordinator.Fighters[slot].boneTransforms[index].name",
        exact_t800_vs_t800 = pairing.Validation.ExactT800VersusT800,
        reason = pairing.Validation.Reason,
        local_slot = pairing.LocalSlot,
        opponent_slot = pairing.OpponentSlot,
        local_fighter = MeasuredFighterPayload(
            pairing.LocalFighter,
            pairing.Validation.LocalSemanticT800,
            pairing.Validation.LocalExactT800BoneSignature),
        opponent_fighter = MeasuredFighterPayload(
            pairing.OpponentFighter,
            pairing.Validation.OpponentSemanticT800,
            pairing.Validation.OpponentExactT800BoneSignature),
    };

    private static object MeasuredFighterPayload(
        MeasuredFighter? fighter,
        bool semanticT800,
        bool exactT800BoneSignature) => new
    {
        slot = fighter?.Slot,
        semantic_robot_id = fighter?.SemanticRobotId,
        runtime_object_name = fighter?.RuntimeObjectName,
        bone_count = fighter?.BoneNames?.Count,
        bone_names = fighter?.BoneNames,
        semantic_t800 = semanticT800,
        exact_t800_bone_signature = exactT800BoneSignature,
    };

    private static object ReadPrivateAiProof(GameMenuController? gameMenu)
    {
        try
        {
            var coordinator = gameMenu?.fightCoordinator ??
                UnityEngine.Object.FindFirstObjectByType<FightCoordinator>();
            var network = gameMenu?.networkSession ??
                UnityEngine.Object.FindFirstObjectByType<NetworkSession>();
            var context = GameContext.Instance ?? UnityEngine.Object.FindFirstObjectByType<GameContext>();
            var view = gameMenu?.menuView;
            if (coordinator is null)
                return new { proven = false, reason = "no_fight_coordinator" };

            var localSlot = coordinator.LocalFighterIndex;
            if (localSlot is < 0 or > 1)
                return new { proven = false, reason = "invalid_local_slot" };
            var opponentSlot = 1 - localSlot;
            var slotHasClient = coordinator.slotHasClient;
            var opponentSlotClientKnown = slotHasClient is not null && slotHasClient.Length > opponentSlot;
            var opponentSlotHasClient = opponentSlotClientKnown && slotHasClient![opponentSlot];
            var opponentHumanBit = (coordinator.clientHumanSlotMask & (1 << opponentSlot)) != 0;
            var opponentIsAi = coordinator.OpponentIsAI;
            var opponentSlotIsAi = coordinator.SlotIsAI(opponentSlot);
            var humanInOpponentSlot = coordinator.HumanInSlot(opponentSlot);
            var networkClientOnly = network is not null && network.IsConnected && network.IsClient && !network.IsServer;
            var solo = context is not null && context.IsSolo;
            var sessionPrivacyKnown = TryReadMultiplayerSessionPrivate(
                out var sessionIsPrivate,
                out var sessionPrivacyReason);
            var exactBotOne = coordinator.clientAiDifficultyLevel == 0 &&
                              coordinator.SparringBotNumber == 1;
            var fighters = coordinator.Fighters;
            var clientVisualOnlyPair = fighters is not null && fighters.Length >= 2 &&
                                       fighters[0] is not null && fighters[1] is not null &&
                                       fighters[0].IsVisualOnly && fighters[1].IsVisualOnly;
            var sessionProven = TryGetPrivateAiContext(
                requireActiveRound: false,
                out _,
                out var sessionProofReason);
            var roundActive = coordinator.CurrentRound is not null && coordinator.CurrentRound.IsActive;
            var activeGameplayProven = TryGetPrivateAiContext(
                requireActiveRound: true,
                out _,
                out var activeGameplayProofReason);
            var roundInactive = coordinator.CurrentRound is null || !coordinator.CurrentRound.IsActive;
            var postFightPrompt = gameMenu is not null && gameMenu.IsMenuOpen && view is not null &&
                                  view.CurrentPane == GameMenuView.Pane.PostFight &&
                                  view.postFightContinueButton is not null &&
                                  view.postFightContinueButton.enabledInHierarchy &&
                                  view.postFightContinueButton.visible;
            return new
            {
                proven = sessionProven,
                active_gameplay_proven = activeGameplayProven,
                reason = sessionProofReason,
                active_gameplay_reason = activeGameplayProofReason,
                local_slot = localSlot,
                opponent_slot = opponentSlot,
                network_client_only = networkClientOnly,
                context_is_solo = solo,
                multiplayer_session_privacy_known = sessionPrivacyKnown,
                multiplayer_session_is_private = sessionPrivacyKnown ? sessionIsPrivate : null as bool?,
                multiplayer_session_privacy_reason = sessionPrivacyReason,
                opponent_is_ai = opponentIsAi,
                opponent_slot_is_ai = opponentSlotIsAi,
                human_in_opponent_slot = humanInOpponentSlot,
                opponent_slot_client_known = opponentSlotClientKnown,
                opponent_slot_has_client = opponentSlotHasClient,
                opponent_human_bit_set = opponentHumanBit,
                client_ai_difficulty = coordinator.clientAiDifficultyLevel,
                sparring_bot_number = coordinator.SparringBotNumber,
                exact_sparring_bot_1 = exactBotOne,
                client_visual_only_fighter_pair = clientVisualOnlyPair,
                fight_epoch = coordinator.fightEpoch,
                phase = coordinator.CurrentPhase.ToString(),
                phase_value = (int)coordinator.CurrentPhase,
                round_active = roundActive,
                round_number = coordinator.CurrentRound?.RoundNumber,
                round_inactive = roundInactive,
                post_fight_prompt = postFightPrompt,
                post_fight_is_winner = gameMenu?.postFightIsWinner,
                space_gate_would_allow = sessionProven && roundInactive && postFightPrompt &&
                                         gameMenu?.postFightIsWinner == true,
            };
        }
        catch (Exception exception)
        {
            return new
            {
                proven = false,
                reason = $"private_ai_probe_failed:{exception.GetType().Name}",
            };
        }
    }

    private static IntPtr NativePointer(Il2CppObjectBase value) =>
        IL2CPP.Il2CppObjectBaseToPtr(value);

    private static string? ToolkitPath(VisualElement element)
    {
        try
        {
            var parts = new List<string>();
            VisualElement? cursor = element;
            for (var depth = 0; cursor is not null && depth < 32; depth++)
            {
                var name = SafeString(cursor.name, 128);
                parts.Add(string.IsNullOrEmpty(name) ? cursor.GetType().Name : name);
                cursor = cursor.parent;
            }
            parts.Reverse();
            return string.Join("/", parts);
        }
        catch
        {
            return null;
        }
    }

    private static string? GameObjectPath(GameObject gameObject)
    {
        try
        {
            var parts = new List<string>();
            var cursor = gameObject.transform;
            for (var depth = 0; cursor is not null && depth < 32; depth++)
            {
                parts.Add(SafeString(cursor.name, 128) ?? "<unnamed>");
                cursor = cursor.parent;
            }
            parts.Reverse();
            return string.Join("/", parts);
        }
        catch
        {
            return null;
        }
    }

    private static bool ShouldRedactText(string? elementName, TextElement? textElement) =>
        textElement?.isInputField == true || ShouldRedactName(elementName);

    private static bool ShouldRedactName(string? elementName)
    {
        if (string.IsNullOrWhiteSpace(elementName))
            return false;
        var lowered = elementName.ToLowerInvariant();
        return lowered.Contains("password", StringComparison.Ordinal) ||
               lowered.Contains("secret", StringComparison.Ordinal) ||
               lowered.Contains("token", StringComparison.Ordinal) ||
               lowered.Contains("cookie", StringComparison.Ordinal) ||
               lowered.Contains("credential", StringComparison.Ordinal) ||
               lowered.Contains("recovery", StringComparison.Ordinal) ||
               lowered.Contains("mfa", StringComparison.Ordinal);
    }

    private static string? SafeString(string? value, int maxLength)
    {
        if (value is null)
            return null;
        var normalized = value.Replace('\r', ' ').Replace('\n', ' ');
        return normalized.Length <= maxLength ? normalized : normalized[..maxLength];
    }

    private static string? TryRead(Func<string?> read)
    {
        try
        {
            return read();
        }
        catch
        {
            return null;
        }
    }

    private static bool? TryReadNullable(Func<bool?> read)
    {
        try
        {
            return read();
        }
        catch
        {
            return null;
        }
    }

    private static string HashFile(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var sha256 = SHA256.Create();
        return Convert.ToHexString(sha256.ComputeHash(stream)).ToLowerInvariant();
    }

    private static string HashText(string value)
    {
        using var sha256 = SHA256.Create();
        return Convert.ToHexString(sha256.ComputeHash(Encoding.UTF8.GetBytes(value))).ToLowerInvariant();
    }

    private sealed record CapturedState(string Identity, object Payload);
    private sealed record ToolkitFocus(IntPtr Pointer, object? Payload);
    private sealed record ScheduleStep(int Tick, Vector3 Velocity, int? MoveIndex, string Label);
    private sealed record MeasuredFighter(
        int Slot,
        string? SemanticRobotId,
        string? RuntimeObjectName,
        IReadOnlyList<string?>? BoneNames);
    private sealed record MeasuredPairing(
        int? LocalSlot,
        int? OpponentSlot,
        MeasuredFighter? LocalFighter,
        MeasuredFighter? OpponentFighter,
        PairingValidation Validation);
    private sealed record PrivateAiContext(
        GameMenuController? GameMenu,
        FightCoordinator Coordinator,
        RobotInputController? Input,
        bool RoundActive,
        int LocalSlot,
        int OpponentSlot,
        int FightEpoch,
        int RoundNumber,
        NetworkSession Network,
        GameContext Context,
        RoundState? Round);

    private sealed record RuntimeIdentity(
        IntPtr CoordinatorPointer,
        IntPtr NetworkPointer,
        IntPtr RoundPointer,
        IntPtr ControllerPointer,
        int FightEpoch,
        int RoundNumber,
        int LocalSlot,
        int OpponentSlot,
        string ArenaId,
        string Endpoint)
    {
        internal static RuntimeIdentity From(PrivateAiContext scope) => new(
            NativePointer(scope.Coordinator),
            NativePointer(scope.Network),
            scope.Round is null ? IntPtr.Zero : NativePointer(scope.Round),
            scope.Input is null ? IntPtr.Zero : NativePointer(scope.Input),
            scope.FightEpoch,
            scope.RoundNumber,
            scope.LocalSlot,
            scope.OpponentSlot,
            scope.Context.ArenaID,
            $"{scope.Network.serverAddress}:{scope.Network.port}");
    }

    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);

    private sealed record CommandResult(
        string Status,
        string Reason,
        bool Applied,
        bool ClientRequestIssued,
        object? MeasuredPairing)
    {
        internal static CommandResult AppliedResult(string reason, object? measuredPairing = null) =>
            new(
                "accepted",
                reason,
                Applied: true,
                ClientRequestIssued: false,
                measuredPairing);

        internal static CommandResult RequestIssued(string reason) =>
            new(
                "accepted",
                reason,
                Applied: false,
                ClientRequestIssued: true,
                MeasuredPairing: null);

        internal static CommandResult Rejected(string reason, object? measuredPairing = null) =>
            new(
                "rejected",
                reason,
                Applied: false,
                ClientRequestIssued: false,
                measuredPairing);
    }
}

[HarmonyPatch(typeof(RobotInputController), "LateUpdate")]
internal static class RobotInputControllerLateUpdateControlPatch
{
    [HarmonyPrefix]
    [HarmonyPriority(Priority.First)]
    internal static void Prefix(RobotInputController __instance)
    {
        Plugin.Instance?.OnRobotInputLateUpdatePrefix(__instance);
    }
}

[HarmonyPatch(typeof(RobotInputController), "SendVelocityCommand")]
internal static class SendVelocityCommandControlPatch
{
    [HarmonyPrefix]
    [HarmonyPriority(Priority.First)]
    internal static void Prefix(RobotInputController __instance)
    {
        Plugin.Instance?.OnSendVelocityCommandPrefix(__instance);
    }

    [HarmonyPostfix]
    [HarmonyPriority(Priority.Last)]
    internal static void Postfix(RobotInputController __instance)
    {
        Plugin.Instance?.OnSendVelocityCommandPostfix(__instance);
    }

    [HarmonyFinalizer]
    internal static Exception? Finalizer(RobotInputController __instance, Exception? __exception)
    {
        if (__exception is not null)
            Plugin.Instance?.OnSendVelocityCommandFailure(__instance, __exception);
        return __exception;
    }
}

[HarmonyPatch(typeof(RobotInputController), "SendMoveEvent")]
internal static class SendMoveEventControlPatch
{
    [HarmonyPrefix]
    [HarmonyPriority(Priority.First)]
    internal static bool Prefix(RobotInputController __instance)
    {
        return Plugin.Instance?.OnSendMoveEventPrefix(__instance) ?? true;
    }

    [HarmonyPostfix]
    [HarmonyPriority(Priority.Last)]
    internal static void Postfix(RobotInputController __instance)
    {
        Plugin.Instance?.OnSendMoveEventPostfix(__instance);
    }

    [HarmonyFinalizer]
    internal static Exception? Finalizer(RobotInputController __instance, Exception? __exception)
    {
        if (__exception is not null)
            Plugin.Instance?.OnSendMoveEventFailure(__instance, __exception);
        return __exception;
    }
}

public sealed class BridgeBehaviour : MonoBehaviour
{
    public BridgeBehaviour(IntPtr pointer) : base(pointer)
    {
    }

    public void Update()
    {
        Plugin.Instance?.OnUnityUpdate();
    }

    public void FixedUpdate()
    {
        Plugin.Instance?.OnUnityFixedUpdate();
    }

    public void LateUpdate()
    {
        Plugin.Instance?.OnUnityLateUpdate();
    }
}
