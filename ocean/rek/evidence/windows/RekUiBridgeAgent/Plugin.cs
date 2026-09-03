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
public sealed partial class Plugin : BasePlugin
{
    public const string PluginGuid = "rek.evidence.control.bridge";
    public const string PluginName = "REK Evidence Control Bridge";
    public const string PluginVersion = "0.4.0";

    private const string PipeName = "rek-ui-bridge-v1";
    private const string IsolatedSessionMarker = "spark-x98";
    private static readonly IsolationProof ExplicitIsolatedSession = DetectExplicitIsolatedSession();
    private const int MaxPendingRequests = 32;
    private const int MaxRememberedRequests = 2048;
    private const int MaxConsumedTrialRounds = 128;
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
    private string _singleMotionTrialSha256 = string.Empty;
    private string _continuousControllerSha256 = string.Empty;
    private string _attackZoneContractSha256 = string.Empty;
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
    private bool _trialIsolationPatchesVerified;
    private readonly bool[] _renderedCommandMarkers =
        new bool[RenderedCommandMarkerContract.Specs.Length];
    private bool _renderedMarkerStripVisible;
    private Texture2D? _renderedMarkerPreTexture;
    private Texture2D? _renderedMarkerPostTexture;
    private FreshRoundArm? _freshRoundArm;
    private readonly HashSet<string> _consumedTrialRounds = new(StringComparer.Ordinal);
    private bool _singleMotionTrialRunning;
    private bool _singleMotionTrialAuthorizedWhileBackground;
    private SingleMotionSelector? _singleMotionTrialSelector;
    private string? _singleMotionTrialRunId;
    private string? _singleMotionTrialFreshRoundRequestId;
    private RuntimeIdentity? _singleMotionTrialIdentity;
    private string? _singleMotionTrialRoundIdentitySha256;
    private object? _singleMotionTrialInitialState;
    private string? _singleMotionTrialInitialStateSha256;
    private RobotInputController? _singleMotionTrialInput;
    private IntPtr _singleMotionTrialInputPointer;
    private Vector3 _singleMotionTrialVelocity = Vector3.zero;
    private int _singleMotionTrialTick;
    private int _singleMotionTrialFixedSubstep;
    private int _singleMotionTrialNonNeutralEdgeCount;
    private int _singleMotionTrialReleaseEdgeCount;
    private int _singleMotionTrialVelocityPressSendCompletedCount;
    private int _singleMotionTrialVelocityReleaseSendCompletedCount;
    private int _singleMotionTrialMoveSendCompletedCount;
    private bool _singleMotionTrialNeutralPreRollInvocationObserved;
    private bool _singleMotionTrialNeutralPreRollSendObserved;
    private string? _singleMotionTrialVelocityEdgeAwaitingSend;
    private bool _singleMotionTrialVelocityInvocationObserved;
    private int? _singleMotionTrialMoveAwaitingSend;
    private bool _singleMotionTrialMoveArmedObserved;
    private bool _singleMotionTrialMoveInvocationObserved;
    private bool _continuousControllerRunning;
    private bool _continuousControllerAuthorizedWhileBackground;
    private string? _continuousControllerRunId;
    private TrialSessionIdentity? _continuousControllerSessionIdentity;
    private RuntimeIdentity? _continuousControllerRoundIdentity;
    private string? _continuousControllerRoundIdentitySha256;
    private RobotInputController? _continuousControllerInput;
    private IntPtr _continuousControllerInputPointer;
    private Vector3 _continuousControllerVelocity = Vector3.zero;
    private int _continuousControllerFixedSubstep;
    private int _continuousControllerTick;
    private int _continuousControllerRoundTick;
    private int _continuousControllerRoundSequence;
    private int _continuousControllerTelemetrySequence;
    private string _continuousControllerPhase = "inactive";
    private string? _continuousControllerSuspendReason;
    private bool _continuousControllerRoundInactiveObserved;
    private int _continuousControllerRoundInactiveTick;
    private bool _continuousControllerRoundStartRequestIssued;
    private int _continuousControllerRoundStartRequestTick;
    private int _continuousControllerNextAttackIndex;
    private int _continuousControllerActionSequence;
    private ContinuousAttackProfile? _continuousControllerActiveAttack;
    private MocapClipConfig? _continuousControllerActiveClip;
    private IntPtr _continuousControllerActiveClipPointer;
    private int? _continuousControllerMoveAwaitingSend;
    private bool _continuousControllerMoveArmedObserved;
    private bool _continuousControllerMoveInvocationObserved;
    private bool _continuousControllerMoveRequestObserved;
    private bool _continuousControllerActionStartedObserved;
    private int _continuousControllerActionRequestTick;
    private int _continuousControllerActionStartTick;
    private int _continuousControllerSettleUntilTick;
    private string? _continuousControllerVelocityPurposeAwaitingSend;
    private bool _continuousControllerVelocityInvocationObserved;
    private SpecialCommand? _continuousControllerSpecialAwaitingSend;
    private bool _continuousControllerSpecialArmedObserved;
    private bool _continuousControllerSpecialInvocationObserved;
    private string? _continuousControllerSpecialPurpose;
    private int _continuousControllerLastRecoveryRequestTick = int.MinValue;
    private bool _continuousControllerStraightenIssued;
    private bool _continuousControllerEStopAwaitingSend;
    private bool _continuousControllerEStopInvocationObserved;
    private string _continuousControllerRecoveryStage = "inactive";
    private int _continuousControllerRecoveryStageTick;
    private int _continuousControllerRecoverySequence;
    private bool _continuousControllerRecoveryEpisodeActive;
    private ContinuousFrame? _continuousControllerLastFrame;
    private string? _continuousControllerLastRoundIdentitySha256;
    private ContinuousRoundMetrics? _continuousControllerLastRoundMetrics;

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
        _singleMotionTrialSha256 = HashText(SingleMotionTrialContract.CanonicalJson);
        _continuousControllerSha256 = HashText(ContinuousBotControllerContract.CanonicalJson);
        _attackZoneContractSha256 = HashText(AttackZoneTrialContract.CanonicalJson);

        if (!string.Equals(
                _scheduleSha256,
                BridgeScheduleContract.ExpectedSha256,
                StringComparison.Ordinal) ||
            !ValidateEmbeddedSchedule() ||
            !string.Equals(
                _singleMotionTrialSha256,
                SingleMotionTrialContract.ExpectedSha256,
                StringComparison.Ordinal) ||
            !ValidateSingleMotionTrialContract() ||
            !string.Equals(
                _continuousControllerSha256,
                ContinuousBotControllerContract.ExpectedSha256,
                StringComparison.Ordinal) ||
            !ValidateContinuousControllerContract() ||
            !string.Equals(
                _attackZoneContractSha256,
                AttackZoneTrialContract.ExpectedSha256,
                StringComparison.Ordinal) ||
            !AttackZoneTrialContract.ValidateEmbeddedContract(out _))
        {
            Log.LogError(
                $"Control bridge disabled: embedded control contract mismatch " +
                $"schedule={_scheduleSha256} single_motion_trial={_singleMotionTrialSha256} " +
                $"continuous_controller={_continuousControllerSha256} " +
                $"attack_zone={_attackZoneContractSha256}.");
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
        _trialIsolationPatchesVerified =
            HasOwnedPatch(typeof(RobotInputController), "SendSpecialEvent") &&
            HasOwnedPatch(typeof(RobotInputController), "SendEStopToggle") &&
            HasOwnedPatch(typeof(FightCoordinator), "OnHitReceived");
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
        StopSingleMotionTrial("plugin_unload");
        StopContinuousController("plugin_unload");
        _freshRoundArm = null;
        _harmony?.UnpatchSelf();
        _harmony = null;
        _sendBoundaryPatchesVerified = false;
        _trialIsolationPatchesVerified = false;
        _leaseConnectionId = 0;
        _renderedMarkerStripVisible = false;
        Array.Clear(_renderedCommandMarkers);
        if (_renderedMarkerPreTexture is not null)
        {
            UnityEngine.Object.Destroy(_renderedMarkerPreTexture);
            _renderedMarkerPreTexture = null;
        }
        if (_renderedMarkerPostTexture is not null)
        {
            UnityEngine.Object.Destroy(_renderedMarkerPostTexture);
            _renderedMarkerPostTexture = null;
        }
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
                var result = ApplyCommand(request);
                var measuredPairing = result.MeasuredPairing;
                if ((request.Command.Value is BridgeCommand.StartMeasuredSchedule or
                        BridgeCommand.StartSingleMotionTrial or
                        BridgeCommand.StartContinuousBotController or
                        BridgeCommand.StartAttackZoneTrial) &&
                    measuredPairing is null)
                    measuredPairing = MeasuredPairingPayload(ReadMeasuredPairing(gameMenu: null));
                _pipe?.Send(request.ConnectionId, new
                {
                    @event = "ack",
                    protocol = "rek.ui_bridge.v1",
                    request_id = request.RequestId,
                    command = request.Command.Value.ToString(),
                    selector = request.Selector,
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
                    single_motion_trial_schema = SingleMotionTrialContract.Schema,
                    single_motion_trial_sha256 = _singleMotionTrialSha256,
                    single_motion_trial_authority_scope = SingleMotionTrialContract.AuthorityScope,
                    single_motion_trial_authority_caveat = SingleMotionTrialContract.AuthorityCaveat,
                    authoritative_execution_observed = false,
                    single_motion_trial_run_id = _singleMotionTrialRunId,
                    single_motion_trial_selector = _singleMotionTrialSelector?.Selector,
                    single_motion_trial_round_identity_sha256 = _singleMotionTrialRoundIdentitySha256,
                    single_motion_trial_initial_state_sha256 = _singleMotionTrialInitialStateSha256,
                    single_motion_trial_initial_state = _singleMotionTrialInitialState,
                    continuous_controller_schema = ContinuousBotControllerContract.Schema,
                    continuous_controller_sha256 = _continuousControllerSha256,
                    continuous_controller_authority_scope =
                        ContinuousBotControllerContract.AuthorityScope,
                    continuous_controller_authority_caveat =
                        ContinuousBotControllerContract.AuthorityCaveat,
                    continuous_controller_range_angle_provenance =
                        ContinuousBotControllerContract.RangeAngleProvenance,
                    continuous_controller_facing_yaw_provenance =
                        ContinuousBotControllerContract.FacingYawProvenance,
                    continuous_controller_attack_selection_provenance =
                        ContinuousBotControllerContract.AttackSelectionProvenance,
                    continuous_controller_static_impact_timing_provenance =
                        ContinuousBotControllerContract.StaticImpactTimingProvenance,
                    continuous_controller_round_restart_limitation =
                        ContinuousBotControllerContract.RoundRestartLimitation,
                    continuous_controller_recovery_guard_provenance =
                        ContinuousBotControllerContract.RecoveryGuardProvenance,
                    continuous_controller_fault_estop_provenance =
                        ContinuousBotControllerContract.FaultEStopProvenance,
                    continuous_controller_dampen_guard =
                        ContinuousBotControllerContract.DampenGuard,
                    continuous_controller_straighten_guard =
                        ContinuousBotControllerContract.StraightenGuard,
                    continuous_controller_opponent_runtime_requirement =
                        ContinuousBotControllerContract.OpponentRuntimeRequirement,
                    continuous_controller_run_id = _continuousControllerRunId,
                    continuous_controller_running = _continuousControllerRunning,
                    continuous_controller_phase = _continuousControllerPhase,
                    continuous_controller_round_identity_sha256 =
                        _continuousControllerRoundIdentitySha256,
                    attack_zone_trial_schema = AttackZoneTrialContract.Schema,
                    attack_zone_trial_sha256 = _attackZoneContractSha256,
                    attack_zone_trial_running = _attackZoneTrialRunning,
                    attack_zone_recovery_only_running = _attackZoneRecoveryOnlyRunning,
                    attack_zone_recovery_ready_ticks = _attackZoneRecoveryReadyTicks,
                    attack_zone_trial_phase = _attackZonePhase,
                    attack_zone_trial_id = _attackZoneTarget?.Request.TrialId,
                    attack_zone_trial_schedule_sha256 =
                        _attackZoneTarget?.Request.ScheduleSha256,
                    attack_zone_trial_schedule_ordinal =
                        _attackZoneTarget?.Request.ScheduleOrdinal,
                    fresh_round_armed = _freshRoundArm is not null,
                    fresh_round_request_id = _freshRoundArm?.RequestId ??
                                             _singleMotionTrialFreshRoundRequestId,
                    fresh_round_invalid_reason = _freshRoundArm?.InvalidReason,
                    build = new
                    {
                        game_assembly_sha256 = _gameAssemblySha256,
                        global_metadata_sha256 = _metadataSha256,
                        plugin_sha256 = _pluginSha256,
                        plugin_version = PluginVersion,
                    },
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
                StopSingleMotionTrial("lease_connection_lost");
                StopContinuousController("lease_connection_lost");
                _freshRoundArm = null;
                _leaseConnectionId = 0;
            }

            if (_singleMotionTrialRunning)
                AdvanceSingleMotionTrial();

            if (_continuousControllerRunning)
                AdvanceContinuousController();

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
                ActivateRenderedCommandMarkers(step.Tick);
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
            StopContinuousController($"fixed_update_control_failed:{exception.GetType().Name}");
            StopSingleMotionTrial($"fixed_update_control_failed:{exception.GetType().Name}");
            StopSchedule($"fixed_update_control_failed:{exception.GetType().Name}");
        }
    }

    private CommandResult ApplyCommand(BridgeRequest request)
    {
        try
        {
            var connectionId = request.ConnectionId;
            var command = request.Command ?? throw new InvalidOperationException("command_request_missing_command");
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
                StopSingleMotionTrial("lease_released");
                StopContinuousController("lease_released");
                _freshRoundArm = null;
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
                BridgeCommand.StartRound => StartRound(connectionId, request.RequestId),
                BridgeCommand.ExitUnexpectedPrivateAiSession => ExitUnexpectedPrivateAiSession(),
                BridgeCommand.ExitLostPrivateSession => ExitLostPrivateSession(),
                BridgeCommand.StartMeasuredSchedule => StartMeasuredSchedule(),
                BridgeCommand.StopMeasuredSchedule => StopMeasuredSchedule(),
                BridgeCommand.StartSingleMotionTrial => StartSingleMotionTrial(request.Selector),
                BridgeCommand.StartContinuousBotController => StartContinuousBotController(),
                BridgeCommand.StopContinuousBotController => StopContinuousBotController(),
                BridgeCommand.StartAttackZoneTrial => StartAttackZoneTrial(request.AttackZoneTarget),
                BridgeCommand.StopAttackZoneTrial => StopAttackZoneTrial(),
                _ => CommandResult.Rejected("unknown_semantic_command"),
            };
        }
        catch (Exception exception)
        {
            if (request.Command == BridgeCommand.StartRound)
                _freshRoundArm = null;
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
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        var lobby = UnityEngine.Object.FindFirstObjectByType<LobbyShellController>();
        if (lobby is null || lobby.CurrentScreen != LobbyShellController.Screen.FreePlay)
            return CommandResult.Rejected("private_practice_requires_free_play_screen");
        if (AnyConnectedSession())
            return CommandResult.Rejected("network_session_already_connected");

        var koth = lobby.kothScreen;
        if (koth is null || koth.soloButton is null ||
            !koth.soloButton.visible || !koth.soloButton.enabledInHierarchy)
        {
            return CommandResult.Rejected("private_practice_control_not_visible_and_enabled");
        }
        if (koth.soloFinding)
            return CommandResult.Rejected("private_practice_reservation_already_in_progress");

        koth.OnSoloClicked();
        return koth.soloFinding
            ? CommandResult.RequestIssued("private_practice_reservation_requested")
            : CommandResult.Rejected("private_practice_reservation_postcondition_not_observed");
    }

    private CommandResult StartRound(long connectionId, string requestId)
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        if (_scheduleRunning || _singleMotionTrialRunning || _continuousControllerRunning)
            return CommandResult.Rejected("controlled_run_already_active");
        if (_freshRoundArm is not null)
            return CommandResult.Rejected("fresh_round_already_armed");
        if (!TryGetPrivateAiContext(requireActiveRound: false, out var scope, out var reason))
            return CommandResult.Rejected(reason);
        if (scope.RoundActive)
            return CommandResult.Rejected("round_already_active");
        if (!TryCreateFreshRoundArm(
                connectionId,
                requestId,
                scope,
                out var arm,
                out reason))
        {
            return CommandResult.Rejected(reason);
        }

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
            _freshRoundArm = arm;
            scope.GameMenu!.HandlePostFightContinue();
            return CommandResult.RequestIssued("post_fight_continue_request_issued");
        }

        if (scope.Coordinator.CurrentPhase == FightPhase.Idle)
        {
            if (scope.GameMenu is null || scope.GameMenu.IsMenuOpen)
                return CommandResult.Rejected("idle_fight_setup_requires_closed_game_menu");
            if (scope.Coordinator.IsRemoteDriven)
            {
                _freshRoundArm = arm;
                scope.Coordinator.OnFightButtonNetwork();
                return CommandResult.RequestIssued("remote_ready_request_issued");
            }
            if (scope.Coordinator.draining)
                return CommandResult.Rejected("arena_is_draining");
            if (scope.Coordinator.setupRoutine is not null)
                return CommandResult.Rejected("fight_setup_already_in_progress");
            _freshRoundArm = arm;
            scope.Coordinator.StartFight();
            if (scope.Coordinator.setupRoutine is null)
            {
                _freshRoundArm = null;
                return CommandResult.Rejected("native_start_fight_postcondition_not_observed");
            }
            return CommandResult.AppliedResult("native_start_fight_coroutine_observed");
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
        if (_singleMotionTrialRunning || _continuousControllerRunning)
            return CommandResult.Rejected("another_control_mode_already_running");
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
        _freshRoundArm = null;
        Array.Clear(_renderedCommandMarkers);
        _renderedMarkerStripVisible = true;
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
        if (!string.Equals(reason, "complete", StringComparison.Ordinal))
            _renderedMarkerStripVisible = false;
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

    private CommandResult StartSingleMotionTrial(string? selectorName)
    {
        if (!RequireBackgroundControl(out var foregroundReason))
            return CommandResult.Rejected(foregroundReason);
        if (_scheduleRunning)
            return CommandResult.Rejected("measured_schedule_already_running");
        if (_singleMotionTrialRunning)
            return CommandResult.Rejected("single_motion_trial_already_running");
        if (_continuousControllerRunning)
            return CommandResult.Rejected("continuous_bot_controller_already_running");
        if (!SingleMotionTrialContract.TryGet(selectorName, out var selector))
            return CommandResult.Rejected("invalid_or_disallowed_selector");
        if (!SameFloatBits(Time.fixedDeltaTime, BridgeScheduleContract.ExpectedFixedDeltaTime))
            return CommandResult.Rejected($"unexpected_fixed_delta_time:{Time.fixedDeltaTime:R}");
        if (!_sendBoundaryPatchesVerified || !_trialIsolationPatchesVerified)
            return CommandResult.Rejected("single_motion_trial_send_boundary_patches_not_verified");

        var arm = _freshRoundArm;
        if (arm is null)
            return CommandResult.Rejected("fresh_round_not_armed");
        _freshRoundArm = null;
        if (arm.ConnectionId != _leaseConnectionId ||
            arm.ConnectionId != (_pipe?.CurrentConnectionId ?? 0))
        {
            return CommandResult.Rejected("fresh_round_arm_connection_changed");
        }
        if (arm.InvalidReason is not null)
            return CommandResult.Rejected($"fresh_round_arm_invalid:{arm.InvalidReason}");
        if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var reason))
            return CommandResult.Rejected($"fresh_round_scope_not_proven:{reason}");

        var sessionIdentity = TrialSessionIdentity.From(scope);
        if (!sessionIdentity.IsComplete || !sessionIdentity.Equals(arm.SessionIdentity))
            return CommandResult.Rejected("fresh_round_session_scope_changed");
        if (!TryCreateTrialRoundIdentity(scope, out var roundIdentity, out reason))
            return CommandResult.Rejected(reason);
        if (roundIdentity.RoundPointer == arm.PriorRoundPointer ||
            (roundIdentity.FightEpoch == arm.PriorFightEpoch &&
             roundIdentity.RoundNumber == arm.PriorRoundNumber))
        {
            return CommandResult.Rejected("fresh_round_identity_not_unique");
        }

        var roundIdentitySha256 = HashTrialRoundIdentity(roundIdentity);
        if (_consumedTrialRounds.Contains(roundIdentitySha256))
            return CommandResult.Rejected("single_motion_trial_round_already_consumed");
        if (_consumedTrialRounds.Count >= MaxConsumedTrialRounds)
            return CommandResult.Rejected("single_motion_trial_round_history_capacity_reached");

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
        if (!TryCaptureSingleMotionInitialState(
                scope,
                roundIdentitySha256,
                out var initialState,
                out var initialStateSha256,
                out reason))
        {
            return CommandResult.Rejected($"single_motion_trial_initial_state_rejected:{reason}");
        }

        if (!_consumedTrialRounds.Add(roundIdentitySha256))
            return CommandResult.Rejected("single_motion_trial_round_already_consumed");

        _singleMotionTrialSelector = selector;
        _singleMotionTrialRunId = Guid.NewGuid().ToString("N");
        _singleMotionTrialFreshRoundRequestId = arm.RequestId;
        _singleMotionTrialIdentity = RuntimeIdentity.From(scope);
        _singleMotionTrialRoundIdentitySha256 = roundIdentitySha256;
        _singleMotionTrialInitialState = initialState;
        _singleMotionTrialInitialStateSha256 = initialStateSha256;
        _singleMotionTrialInput = scope.Input;
        _singleMotionTrialInputPointer = roundIdentity.ControllerPointer;
        _singleMotionTrialVelocity = Vector3.zero;
        _singleMotionTrialTick = 0;
        _singleMotionTrialFixedSubstep = 0;
        _singleMotionTrialNonNeutralEdgeCount = 0;
        _singleMotionTrialReleaseEdgeCount = 0;
        _singleMotionTrialVelocityPressSendCompletedCount = 0;
        _singleMotionTrialVelocityReleaseSendCompletedCount = 0;
        _singleMotionTrialMoveSendCompletedCount = 0;
        _singleMotionTrialNeutralPreRollInvocationObserved = false;
        _singleMotionTrialNeutralPreRollSendObserved = false;
        _singleMotionTrialVelocityEdgeAwaitingSend = "neutral_pre_roll";
        _singleMotionTrialVelocityInvocationObserved = false;
        _singleMotionTrialMoveAwaitingSend = null;
        _singleMotionTrialMoveArmedObserved = false;
        _singleMotionTrialMoveInvocationObserved = false;
        _singleMotionTrialAuthorizedWhileBackground = true;
        _singleMotionTrialRunning = true;

        Log.LogInfo(
            $"Single-motion trial started run={_singleMotionTrialRunId} selector={selector.Selector} " +
            $"contract_sha256={_singleMotionTrialSha256} round_sha256={roundIdentitySha256} " +
            $"initial_state_sha256={initialStateSha256}");
        return CommandResult.AppliedResult("single_motion_trial_started", measuredPairingPayload);
    }

    private void AdvanceSingleMotionTrial()
    {
        if (!_singleMotionTrialRunning)
            return;
        if (!RequireOwnedSingleMotionTrialControl(out var controlReason))
        {
            StopSingleMotionTrial(controlReason);
            return;
        }

        var atTrialBoundary =
            _singleMotionTrialFixedSubstep % SingleMotionTrialContract.FixedSubstepsPerTrialTick == 0;
        var input = _singleMotionTrialInput;
        PrivateAiContext? scope = null;
        if (atTrialBoundary)
        {
            if (!TryGetPrivateAiContext(requireActiveRound: true, out scope, out var scopeReason))
            {
                StopSingleMotionTrial($"scope_lost:{scopeReason}");
                return;
            }
            if (!ScopeMatchesSingleMotionTrial(scope))
            {
                StopSingleMotionTrial("single_motion_trial_scope_identity_changed");
                return;
            }
            var measuredPairing = ReadMeasuredPairing(
                scope.Coordinator,
                scope.LocalSlot,
                scope.OpponentSlot);
            if (!measuredPairing.Validation.ExactT800VersusT800)
            {
                StopSingleMotionTrial(
                    $"single_motion_trial_pairing_lost:{measuredPairing.Validation.Reason}");
                return;
            }
            input = scope.Input;
            _singleMotionTrialTick =
                _singleMotionTrialFixedSubstep / SingleMotionTrialContract.FixedSubstepsPerTrialTick;
        }

        if (input is null || _singleMotionTrialInputPointer == IntPtr.Zero ||
            NativePointer(input) != _singleMotionTrialInputPointer)
        {
            StopSingleMotionTrial("single_motion_trial_input_controller_changed");
            return;
        }
        if (input.hasPendingSpecial || input.hasPendingEStop ||
            (input.hasPendingMove &&
             (_singleMotionTrialMoveAwaitingSend is null ||
              input.pendingMoveIndex != _singleMotionTrialMoveAwaitingSend.Value)))
        {
            StopSingleMotionTrial("single_motion_trial_unexpected_pending_command");
            return;
        }
        if (!VelocityEquals(input.VelocityCommand, _singleMotionTrialVelocity))
        {
            StopSingleMotionTrial("single_motion_trial_velocity_changed_outside_owned_edge");
            return;
        }

        if (atTrialBoundary && _singleMotionTrialTick == SingleMotionTrialContract.ActionTick)
        {
            if (!_singleMotionTrialNeutralPreRollSendObserved ||
                _singleMotionTrialVelocityEdgeAwaitingSend is not null ||
                _singleMotionTrialNonNeutralEdgeCount != 0)
            {
                StopSingleMotionTrial("neutral_pre_roll_request_not_observed_before_action");
                return;
            }

            var selector = _singleMotionTrialSelector!;
            _singleMotionTrialNonNeutralEdgeCount++;
            if (selector.IsLocomotion)
            {
                _singleMotionTrialVelocity = new Vector3(
                    selector.Forward,
                    selector.Strafe,
                    selector.Yaw);
                if (!SetVelocityExact(input, _singleMotionTrialVelocity))
                {
                    StopSingleMotionTrial("single_motion_trial_action_velocity_readback_mismatch");
                    return;
                }
                _singleMotionTrialVelocityEdgeAwaitingSend = "action";
                _singleMotionTrialVelocityInvocationObserved = false;
            }
            else
            {
                if (selector.MoveIndex is null ||
                    input.hasPendingMove || input.hasPendingSpecial || input.hasPendingEStop)
                {
                    StopSingleMotionTrial("single_motion_trial_move_edge_not_clean");
                    return;
                }
                if (!input.ExecuteMoveByIndex(selector.MoveIndex.Value))
                {
                    StopSingleMotionTrial($"single_motion_trial_move_{selector.MoveIndex}_rejected_locally");
                    return;
                }
                _singleMotionTrialMoveAwaitingSend = selector.MoveIndex;
                _singleMotionTrialMoveArmedObserved = false;
                _singleMotionTrialMoveInvocationObserved = false;
            }
            EmitSingleMotionTrialCommandEdge("action");
        }

        if (atTrialBoundary &&
            _singleMotionTrialTick == SingleMotionTrialContract.LocomotionReleaseTick &&
            _singleMotionTrialSelector!.IsLocomotion)
        {
            if (_singleMotionTrialVelocityPressSendCompletedCount != 1 ||
                _singleMotionTrialVelocityEdgeAwaitingSend is not null)
            {
                StopSingleMotionTrial("single_motion_trial_action_request_not_completed_before_release");
                return;
            }
            _singleMotionTrialVelocity = Vector3.zero;
            if (!SetVelocityExact(input, _singleMotionTrialVelocity))
            {
                StopSingleMotionTrial("single_motion_trial_release_velocity_readback_mismatch");
                return;
            }
            _singleMotionTrialReleaseEdgeCount++;
            _singleMotionTrialVelocityEdgeAwaitingSend = "release";
            _singleMotionTrialVelocityInvocationObserved = false;
            EmitSingleMotionTrialCommandEdge("release");
        }

        var completedFixedSubsteps = _singleMotionTrialFixedSubstep + 1;
        var requiredFixedSubsteps = SingleMotionTrialContract.DurationTrialTicks *
                                    SingleMotionTrialContract.FixedSubstepsPerTrialTick;
        if (completedFixedSubsteps >= requiredFixedSubsteps)
        {
            if (!SingleMotionTrialCompletionCountsMatch())
            {
                StopSingleMotionTrial("single_motion_trial_completion_request_counts_mismatch");
                return;
            }
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var endScope, out var endReason) ||
                !ScopeMatchesSingleMotionTrial(endScope))
            {
                StopSingleMotionTrial($"single_motion_trial_completion_scope_lost:{endReason}");
                return;
            }
            var endPairing = ReadMeasuredPairing(
                endScope.Coordinator,
                endScope.LocalSlot,
                endScope.OpponentSlot);
            if (!endPairing.Validation.ExactT800VersusT800)
            {
                StopSingleMotionTrial(
                    $"single_motion_trial_completion_pairing_lost:{endPairing.Validation.Reason}");
                return;
            }
            StopSingleMotionTrial("complete");
            return;
        }

        _singleMotionTrialFixedSubstep++;
    }

    private bool SingleMotionTrialCompletionCountsMatch()
    {
        var selector = _singleMotionTrialSelector;
        if (selector is null || !_singleMotionTrialNeutralPreRollSendObserved ||
            _singleMotionTrialNeutralPreRollInvocationObserved ||
            _singleMotionTrialVelocityEdgeAwaitingSend is not null ||
            _singleMotionTrialMoveAwaitingSend is not null ||
            _singleMotionTrialNonNeutralEdgeCount != 1)
        {
            return false;
        }
        return selector.IsLocomotion
            ? _singleMotionTrialReleaseEdgeCount == 1 &&
              _singleMotionTrialVelocityPressSendCompletedCount == 1 &&
              _singleMotionTrialVelocityReleaseSendCompletedCount == 1 &&
              _singleMotionTrialMoveSendCompletedCount == 0
            : _singleMotionTrialReleaseEdgeCount == 0 &&
              _singleMotionTrialVelocityPressSendCompletedCount == 0 &&
              _singleMotionTrialVelocityReleaseSendCompletedCount == 0 &&
              _singleMotionTrialMoveSendCompletedCount == 1;
    }

    private void StopSingleMotionTrial(string reason)
    {
        if (!_singleMotionTrialRunning)
            return;

        var runId = _singleMotionTrialRunId;
        var selector = _singleMotionTrialSelector;
        var authorizedWhileBackground = _singleMotionTrialAuthorizedWhileBackground;
        var roundIdentitySha256 = _singleMotionTrialRoundIdentitySha256;
        var initialStateSha256 = _singleMotionTrialInitialStateSha256;
        var initialState = _singleMotionTrialInitialState;
        var neutralPreRollSendObserved = _singleMotionTrialNeutralPreRollSendObserved;
        var nonNeutralEdgeCount = _singleMotionTrialNonNeutralEdgeCount;
        var releaseEdgeCount = _singleMotionTrialReleaseEdgeCount;
        var velocityPressSendCompletedCount = _singleMotionTrialVelocityPressSendCompletedCount;
        var velocityReleaseSendCompletedCount = _singleMotionTrialVelocityReleaseSendCompletedCount;
        var moveSendCompletedCount = _singleMotionTrialMoveSendCompletedCount;
        var complete = string.Equals(reason, "complete", StringComparison.Ordinal);

        _singleMotionTrialRunning = false;
        _singleMotionTrialAuthorizedWhileBackground = false;
        TryCancelExactOwnedTrialMove();
        TryNeutralOwnedTrialController();
        _singleMotionTrialInput = null;
        _singleMotionTrialInputPointer = IntPtr.Zero;
        _singleMotionTrialIdentity = null;
        _singleMotionTrialVelocity = Vector3.zero;
        _singleMotionTrialVelocityEdgeAwaitingSend = null;
        _singleMotionTrialVelocityInvocationObserved = false;
        _singleMotionTrialMoveAwaitingSend = null;
        _singleMotionTrialMoveArmedObserved = false;
        _singleMotionTrialMoveInvocationObserved = false;

        try
        {
            var payload = new
            {
                @event = "single_motion_trial_end",
                protocol = "rek.ui_bridge.v1",
                single_motion_trial_schema = SingleMotionTrialContract.Schema,
                single_motion_trial_sha256 = _singleMotionTrialSha256,
                authority_scope = SingleMotionTrialContract.AuthorityScope,
                authority_caveat = SingleMotionTrialContract.AuthorityCaveat,
                single_motion_trial_run_id = runId,
                fresh_round_request_id = _singleMotionTrialFreshRoundRequestId,
                selector = selector?.Selector,
                selector_kind = selector?.Kind,
                command_identity = selector?.CommandIdentity,
                round_identity_sha256 = roundIdentitySha256,
                initial_state_sha256 = initialStateSha256,
                initial_state = initialState,
                trial_tick = _singleMotionTrialTick,
                client_fixed_substep = _singleMotionTrialFixedSubstep,
                fixed_substeps_per_trial_tick = SingleMotionTrialContract.FixedSubstepsPerTrialTick,
                neutral_pre_roll_send_observed = neutralPreRollSendObserved,
                non_neutral_edge_count = nonNeutralEdgeCount,
                release_edge_count = releaseEdgeCount,
                velocity_press_send_completed_count = velocityPressSendCompletedCount,
                velocity_release_send_completed_count = velocityReleaseSendCompletedCount,
                move_send_completed_count = moveSendCompletedCount,
                round_consumed = roundIdentitySha256 is not null &&
                                 _consumedTrialRounds.Contains(roundIdentitySha256),
                complete,
                reason,
                authorized_while_background = authorizedWhileBackground,
                client_request_edges_observed = complete,
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
                utc = DateTimeOffset.UtcNow,
                stopwatch_timestamp_ticks = System.Diagnostics.Stopwatch.GetTimestamp(),
                stopwatch_frequency_hz = System.Diagnostics.Stopwatch.Frequency,
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

    private void EmitSingleMotionTrialCommandEdge(string edge)
    {
        var selector = _singleMotionTrialSelector!;
        var payload = new
        {
            @event = "single_motion_trial_command_edge",
            protocol = "rek.ui_bridge.v1",
            single_motion_trial_schema = SingleMotionTrialContract.Schema,
            single_motion_trial_sha256 = _singleMotionTrialSha256,
            authority_scope = SingleMotionTrialContract.AuthorityScope,
            authority_caveat = SingleMotionTrialContract.AuthorityCaveat,
            single_motion_trial_run_id = _singleMotionTrialRunId,
            fresh_round_request_id = _singleMotionTrialFreshRoundRequestId,
            selector = selector.Selector,
            selector_kind = selector.Kind,
            command_identity = selector.CommandIdentity,
            round_identity_sha256 = _singleMotionTrialRoundIdentitySha256,
            initial_state_sha256 = _singleMotionTrialInitialStateSha256,
            edge,
            trial_tick = _singleMotionTrialTick,
            client_fixed_substep = _singleMotionTrialFixedSubstep,
            velocity_command_xyz = new[]
            {
                _singleMotionTrialVelocity.x,
                _singleMotionTrialVelocity.y,
                _singleMotionTrialVelocity.z,
            },
            move_index = selector.MoveIndex,
            local_command_value_set = true,
            client_request_edge_observed = false,
            server_acceptance_observed = false,
            authoritative_execution_observed = false,
            utc = DateTimeOffset.UtcNow,
            stopwatch_timestamp_ticks = System.Diagnostics.Stopwatch.GetTimestamp(),
            stopwatch_frequency_hz = System.Diagnostics.Stopwatch.Frequency,
            unity_frame = Time.frameCount,
            unity_fixed_time = Time.fixedTimeAsDouble,
        };
        _pipe?.Send(_leaseConnectionId, payload);
        Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
    }

    private void EmitSingleMotionTrialClientRequest(string requestKind, string phase, int? moveIndex)
    {
        var selector = _singleMotionTrialSelector!;
        var payload = new
        {
            @event = "single_motion_trial_client_request",
            protocol = "rek.ui_bridge.v1",
            single_motion_trial_schema = SingleMotionTrialContract.Schema,
            single_motion_trial_sha256 = _singleMotionTrialSha256,
            authority_scope = SingleMotionTrialContract.AuthorityScope,
            authority_caveat = SingleMotionTrialContract.AuthorityCaveat,
            single_motion_trial_run_id = _singleMotionTrialRunId,
            fresh_round_request_id = _singleMotionTrialFreshRoundRequestId,
            selector = selector.Selector,
            selector_kind = selector.Kind,
            command_identity = selector.CommandIdentity,
            round_identity_sha256 = _singleMotionTrialRoundIdentitySha256,
            initial_state_sha256 = _singleMotionTrialInitialStateSha256,
            request_kind = requestKind,
            request_phase = phase,
            command_edge_trial_tick = phase switch
            {
                "neutral_pre_roll" => 0,
                "action" => SingleMotionTrialContract.ActionTick,
                "release" => SingleMotionTrialContract.LocomotionReleaseTick,
                _ => -1,
            },
            observed_trial_tick = _singleMotionTrialTick,
            observed_client_fixed_substep = _singleMotionTrialFixedSubstep,
            velocity_command_xyz = new[]
            {
                _singleMotionTrialVelocity.x,
                _singleMotionTrialVelocity.y,
                _singleMotionTrialVelocity.z,
            },
            move_index = moveIndex,
            send_method = requestKind == "velocity"
                ? "RobotInputController.SendVelocityCommand"
                : "RobotInputController.SendMoveEvent",
            send_method_returned = true,
            client_request_edge_observed = true,
            server_acceptance_observed = false,
            authoritative_execution_observed = false,
            utc = DateTimeOffset.UtcNow,
            stopwatch_timestamp_ticks = System.Diagnostics.Stopwatch.GetTimestamp(),
            stopwatch_frequency_hz = System.Diagnostics.Stopwatch.Frequency,
            unity_frame = Time.frameCount,
            unity_time = Time.timeAsDouble,
        };
        _pipe?.Send(_leaseConnectionId, payload);
        Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
    }

    private CommandResult StartContinuousBotController()
    {
        if (!RequireBackgroundControl(out var controlReason))
            return CommandResult.Rejected(controlReason);
        if (_scheduleRunning || _singleMotionTrialRunning)
            return CommandResult.Rejected("another_control_mode_already_running");
        if (_continuousControllerRunning)
            return CommandResult.Rejected("continuous_bot_controller_already_running");
        if (!SameFloatBits(Time.fixedDeltaTime, BridgeScheduleContract.ExpectedFixedDeltaTime))
            return CommandResult.Rejected($"unexpected_fixed_delta_time:{Time.fixedDeltaTime:R}");
        if (!_sendBoundaryPatchesVerified || !_trialIsolationPatchesVerified)
            return CommandResult.Rejected("continuous_controller_send_boundary_patches_not_verified");
        if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var reason))
            return CommandResult.Rejected(reason);

        var measuredPairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        var measuredPairingPayload = MeasuredPairingPayload(measuredPairing);
        if (!TryValidateContinuousPairing(measuredPairing, out reason))
        {
            return CommandResult.Rejected(
                $"continuous_runtime_pairing_not_proven:{reason}",
                measuredPairingPayload);
        }
        if (scope.Input is null || !TryValidateContinuousMoveMap(scope.Input, out reason))
            return CommandResult.Rejected(reason, measuredPairingPayload);
        if (scope.Input.hasPendingMove || scope.Input.hasPendingSpecial || scope.Input.hasPendingEStop)
            return CommandResult.Rejected("continuous_controller_initial_pending_command", measuredPairingPayload);
        if (!Finite(scope.Input.VelocityCommand) ||
            !VelocityEquals(scope.Input.VelocityCommand, Vector3.zero))
        {
            return CommandResult.Rejected(
                "continuous_controller_initial_velocity_not_neutral",
                measuredPairingPayload);
        }
        if (!TryCreateTrialRoundIdentity(scope, out var roundIdentity, out reason))
            return CommandResult.Rejected(reason, measuredPairingPayload);
        if (!TryCaptureContinuousFrame(scope, measuredPairing, out var frame, out reason))
            return CommandResult.Rejected(reason, measuredPairingPayload);
        if (frame.LocalMotorShutdown)
        {
            return CommandResult.Rejected(
                "continuous_controller_initial_motor_shutdown_state_not_owned",
                measuredPairingPayload);
        }

        var sessionIdentity = TrialSessionIdentity.From(scope);
        if (!sessionIdentity.IsComplete)
            return CommandResult.Rejected("continuous_controller_session_identity_incomplete");

        _freshRoundArm = null;
        _continuousControllerRunId = Guid.NewGuid().ToString("N");
        _continuousControllerSessionIdentity = sessionIdentity;
        _continuousControllerRoundIdentity = RuntimeIdentity.From(scope);
        _continuousControllerRoundIdentitySha256 = HashTrialRoundIdentity(roundIdentity);
        _continuousControllerLastRoundIdentitySha256 = null;
        _continuousControllerInput = scope.Input;
        _continuousControllerInputPointer = NativePointer(scope.Input);
        _continuousControllerVelocity = Vector3.zero;
        _continuousControllerFixedSubstep = 0;
        _continuousControllerTick = 0;
        _continuousControllerRoundTick = 0;
        _continuousControllerRoundSequence = 1;
        _continuousControllerTelemetrySequence = 0;
        _continuousControllerPhase = "round_active_settling";
        _continuousControllerSuspendReason = null;
        _continuousControllerRoundInactiveObserved = false;
        _continuousControllerRoundInactiveTick = -1;
        _continuousControllerRoundStartRequestIssued = false;
        _continuousControllerRoundStartRequestTick = -1;
        _continuousControllerNextAttackIndex = 0;
        _continuousControllerActionSequence = 0;
        _continuousControllerRecoverySequence = 0;
        _continuousControllerRecoveryEpisodeActive = false;
        _continuousControllerSettleUntilTick = ContinuousBotControllerContract.SettleTicks;
        _continuousControllerLastRecoveryRequestTick = int.MinValue;
        _continuousControllerStraightenIssued = false;
        _continuousControllerRecoveryStage = "inactive";
        _continuousControllerRecoveryStageTick = 0;
        _continuousControllerLastFrame = frame;
        _continuousControllerLastRoundMetrics = null;
        ClearContinuousActionState();
        ClearContinuousPendingRequestState();
        _continuousControllerAuthorizedWhileBackground = true;
        _continuousControllerRunning = true;

        EmitContinuousEvent(
            "continuous_controller_start",
            "continuous_private_bot_controller_started",
            frame,
            new
            {
                initial_active_round_required = true,
                exact_private_sparring_bot_1_scope_proven = true,
                exact_local_t800_proven = true,
                exact_opponent_runtime_t800_proven = true,
                opponent_semantic_robot_id_used_for_acceptance = false,
                controller_has_finite_schedule = false,
                attack_selection_provenance =
                    ContinuousBotControllerContract.AttackSelectionProvenance,
                round_restart_limitation =
                    ContinuousBotControllerContract.RoundRestartLimitation,
            });
        EmitContinuousRoundObservation(frame, "round_bound");
        return CommandResult.AppliedResult(
            "continuous_private_bot_controller_started",
            measuredPairingPayload);
    }

    private CommandResult StopContinuousBotController()
    {
        if (!_continuousControllerRunning)
            return CommandResult.Rejected("continuous_bot_controller_not_running");
        StopContinuousController("requested");
        return CommandResult.AppliedResult("continuous_private_bot_controller_stopped");
    }

    private void AdvanceContinuousController()
    {
        if (!_continuousControllerRunning)
            return;
        if (_attackZoneRecoveryOnlyRunning)
        {
            AdvanceAttackZoneRecoveryOnly();
            return;
        }
        if (_attackZoneTrialRunning)
        {
            AdvanceAttackZoneTrial();
            return;
        }
        if (!RequireOwnedContinuousControl(out var controlReason))
        {
            StopContinuousController(controlReason);
            return;
        }

        var fixedSubstep = _continuousControllerFixedSubstep++;
        if (fixedSubstep % ContinuousBotControllerContract.FixedSubstepsPerControlTick != 0)
            return;
        _continuousControllerTick =
            fixedSubstep / ContinuousBotControllerContract.FixedSubstepsPerControlTick;

        if (!TryGetPrivateAiContext(requireActiveRound: false, out var sessionScope, out var sessionReason))
        {
            SuspendContinuousController($"private_bot1_scope_unproven:{sessionReason}", roundInactive: false);
            return;
        }
        var observedSession = TrialSessionIdentity.From(sessionScope);
        if (_continuousControllerSessionIdentity is null ||
            !observedSession.Equals(_continuousControllerSessionIdentity))
        {
            StopContinuousController("continuous_controller_session_identity_changed");
            return;
        }
        if (!sessionScope.RoundActive ||
            sessionScope.Coordinator.CurrentPhase != FightPhase.RoundActive)
        {
            SuspendContinuousController("round_inactive", roundInactive: true);
            TryIssueContinuousRoundStart(sessionScope);
            return;
        }
        if (!TryGetPrivateAiContext(
                requireActiveRound: true,
                out var scope,
                out var activeReason,
                allowOwnedPendingEStop: _continuousControllerEStopAwaitingSend))
        {
            SuspendContinuousController($"active_round_scope_unproven:{activeReason}", roundInactive: false);
            return;
        }
        if (!TryCreateTrialRoundIdentity(scope, out var trialRoundIdentity, out var identityReason))
        {
            SuspendContinuousController(identityReason, roundInactive: false);
            return;
        }

        var runtimeIdentity = RuntimeIdentity.From(scope);
        var roundIdentitySha256 = HashTrialRoundIdentity(trialRoundIdentity);
        if (_continuousControllerRoundIdentity is null)
        {
            var newRoundIdentityObserved = !string.Equals(
                roundIdentitySha256,
                _continuousControllerLastRoundIdentitySha256,
                StringComparison.Ordinal);
            if (!ContinuousBotControllerContract.CanBindRestartedRound(
                    _continuousControllerRoundInactiveObserved,
                    _continuousControllerRoundStartRequestIssued,
                    newRoundIdentityObserved))
            {
                var rejectionReason = !_continuousControllerRoundInactiveObserved
                    ? "new_round_observed_without_inactive_transition"
                    : !_continuousControllerRoundStartRequestIssued
                        ? "new_round_observed_without_owned_start_request"
                        : "continuous_controller_round_identity_reused";
                StopContinuousController(rejectionReason);
                return;
            }
            if (!TryBindContinuousRound(scope, runtimeIdentity, roundIdentitySha256, out var bindReason))
            {
                SuspendContinuousController(bindReason, roundInactive: false);
                return;
            }
        }
        else if (!_continuousControllerRoundIdentity.Equals(runtimeIdentity) ||
                 !string.Equals(
                     _continuousControllerRoundIdentitySha256,
                     roundIdentitySha256,
                     StringComparison.Ordinal))
        {
            StopContinuousController("continuous_controller_round_identity_changed_without_suspend");
            return;
        }

        var measuredPairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        if (!TryCaptureContinuousFrame(scope, measuredPairing, out var frame, out var frameReason))
        {
            SuspendContinuousController(frameReason, roundInactive: false);
            return;
        }
        if (!ContinuousRuntimeIdentityMatches(frame, _continuousControllerLastFrame))
        {
            StopContinuousController("continuous_runtime_fighter_identity_changed_within_round");
            return;
        }
        if (_continuousControllerInput is null ||
            _continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(_continuousControllerInput) != _continuousControllerInputPointer ||
            NativePointer(frame.Input) != _continuousControllerInputPointer)
        {
            StopContinuousController("continuous_controller_input_binding_changed");
            return;
        }
        if (!VelocityEquals(frame.Input.VelocityCommand, _continuousControllerVelocity))
        {
            StopContinuousController("continuous_controller_velocity_changed_outside_owned_edge");
            return;
        }
        if ((frame.Input.hasPendingEStop &&
             !_continuousControllerEStopAwaitingSend) ||
            (frame.Input.hasPendingMove &&
             (_continuousControllerMoveAwaitingSend is null ||
              frame.Input.pendingMoveIndex != _continuousControllerMoveAwaitingSend.Value)) ||
            (frame.Input.hasPendingSpecial &&
             (_continuousControllerSpecialAwaitingSend is null ||
              frame.Input.pendingSpecialCommand != (int)_continuousControllerSpecialAwaitingSend.Value)))
        {
            StopContinuousController("continuous_controller_unowned_pending_command_observed");
            return;
        }

        var previousPhase = _continuousControllerPhase;
        _continuousControllerPhase = "round_active";
        _continuousControllerSuspendReason = null;
        _continuousControllerRoundInactiveObserved = false;
        _continuousControllerRoundTick++;
        _continuousControllerLastFrame = frame;
        EmitContinuousRoundObservation(frame, "metrics_changed");

        if (!ObserveContinuousActionLifecycle(frame))
            return;

        if (frame.LocalResetting)
        {
            InterruptContinuousAction("local_resetting", frame);
            SetContinuousVelocity(frame.Input, Vector3.zero, "local_not_controllable");
            _continuousControllerPhase = "round_active_suspended_local_not_controllable";
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        if (frame.LocalMotorShutdown)
        {
            InterruptContinuousAction("motor_shutdown_fault_preempted_action", frame);
            SetContinuousVelocity(frame.Input, Vector3.zero, "motor_shutdown_fault_neutral");
            _continuousControllerPhase = "round_active_motor_shutdown_fault";
            DriveContinuousFaultEStopCycle(frame);
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        if (_continuousControllerRecoveryEpisodeActive &&
            IsContinuousFaultEStopStage(_continuousControllerRecoveryStage))
        {
            SetContinuousVelocity(frame.Input, Vector3.zero, "motor_shutdown_fault_neutral");
            _continuousControllerPhase = "round_active_motor_shutdown_fault";
            DriveContinuousFaultEStopCycle(frame);
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        if (frame.LocalFalling && !frame.LocalFallen)
        {
            InterruptContinuousAction("local_fall_preempted_action", frame);
            SetContinuousVelocity(frame.Input, Vector3.zero, "fall_observation_neutral");
            _continuousControllerPhase = "round_active_await_local_fallen_state";
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        if (frame.LocalFallen)
        {
            InterruptContinuousAction("local_fall_recovery_preempted_action", frame);
            SetContinuousVelocity(frame.Input, Vector3.zero, "fall_recovery_neutral");
            _continuousControllerPhase = "round_active_recovery";
            DriveContinuousRecovery(frame);
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        if (_continuousControllerRecoveryEpisodeActive)
        {
            _continuousControllerRecoveryEpisodeActive = false;
            _continuousControllerStraightenIssued = false;
            _continuousControllerSettleUntilTick = Math.Max(
                _continuousControllerSettleUntilTick,
                _continuousControllerTick + ContinuousBotControllerContract.SettleTicks);
            EmitContinuousEvent(
                "continuous_recovery_lifecycle",
                "local_upright_readiness_observed",
                frame,
                new
                {
                    recovery_sequence = _continuousControllerRecoverySequence,
                    lifecycle_stage = "local_upright_readiness_observed",
                });
        }

        if (_continuousControllerSpecialAwaitingSend is not null ||
            frame.Input.hasPendingSpecial || frame.LocalGetUpPending || frame.Input.IsRecovering)
        {
            SetContinuousVelocity(frame.Input, Vector3.zero, "recovery_request_pending");
            _continuousControllerPhase = "round_active_recovery_pending";
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        if (_continuousControllerActiveAttack is not null)
        {
            SetContinuousVelocity(frame.Input, Vector3.zero, "action_lifecycle_neutral");
            _continuousControllerPhase = "round_active_action_lifecycle";
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        if (frame.InputPunching || frame.ComposerActionPlaying || frame.ComposerBusy ||
            frame.Input.hasPendingMove)
        {
            SetContinuousVelocity(frame.Input, Vector3.zero, "await_local_motion_readiness");
            _continuousControllerPhase = "round_active_await_local_motion_readiness";
            EmitContinuousTelemetryIfDue(frame);
            return;
        }

        var attack = ContinuousBotControllerContract.Attacks[_continuousControllerNextAttackIndex];
        var locomotion = ContinuousBotControllerContract.DecideLocomotion(
            frame.Geometry,
            attack,
            frame.OpponentFalling || frame.OpponentFallen);
        SetContinuousVelocity(
            frame.Input,
            new Vector3(locomotion.Forward, locomotion.Strafe, locomotion.Yaw),
            locomotion.Reason);

        if (locomotion.AttackWindow &&
            _continuousControllerTick >= _continuousControllerSettleUntilTick &&
            ContinuousLocalActionReady(frame))
        {
            TryStartContinuousAttack(frame, attack);
        }
        else if (_continuousControllerTick < _continuousControllerSettleUntilTick)
        {
            _continuousControllerPhase = "round_active_settling";
        }
        else
        {
            _continuousControllerPhase = $"round_active_{locomotion.Reason}";
        }

        if (previousPhase.StartsWith("suspended", StringComparison.Ordinal) ||
            previousPhase.StartsWith("round_active_suspended", StringComparison.Ordinal))
        {
            EmitContinuousEvent(
                "continuous_controller_resume",
                "same_session_and_round_scope_revalidated",
                frame,
                null);
        }
        EmitContinuousTelemetryIfDue(frame);
    }

    private bool TryBindContinuousRound(
        PrivateAiContext scope,
        RuntimeIdentity runtimeIdentity,
        string roundIdentitySha256,
        out string reason)
    {
        reason = string.Empty;
        var measuredPairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        if (!TryCaptureContinuousFrame(scope, measuredPairing, out var frame, out reason))
            return false;
        var input = frame.Input;
        if (!TryValidateContinuousMoveMap(input, out reason))
            return false;
        if (input.hasPendingMove || input.hasPendingSpecial || input.hasPendingEStop)
        {
            reason = "continuous_controller_new_round_pending_command";
            return false;
        }
        if (!VelocityEquals(input.VelocityCommand, Vector3.zero))
        {
            reason = "continuous_controller_new_round_velocity_not_neutral";
            return false;
        }

        _continuousControllerRoundIdentity = runtimeIdentity;
        _continuousControllerRoundIdentitySha256 = roundIdentitySha256;
        _continuousControllerInput = input;
        _continuousControllerInputPointer = NativePointer(input);
        _continuousControllerVelocity = Vector3.zero;
        _continuousControllerRoundTick = 0;
        _continuousControllerRoundSequence++;
        _continuousControllerRoundInactiveObserved = false;
        _continuousControllerRoundInactiveTick = -1;
        _continuousControllerRoundStartRequestIssued = false;
        _continuousControllerRoundStartRequestTick = -1;
        _continuousControllerSettleUntilTick =
            _continuousControllerTick + ContinuousBotControllerContract.SettleTicks;
        _continuousControllerLastFrame = frame;
        _continuousControllerLastRoundMetrics = null;
        _continuousControllerStraightenIssued = false;
        _continuousControllerRecoveryEpisodeActive = false;
        ClearContinuousActionState();
        ClearContinuousPendingRequestState();
        EmitContinuousEvent(
            "continuous_controller_round_bound",
            "new_active_round_revalidated",
            frame,
            new { round_sequence = _continuousControllerRoundSequence });
        EmitContinuousRoundObservation(frame, "round_bound");
        return true;
    }

    private void TryIssueContinuousRoundStart(PrivateAiContext scope)
    {
        if (!_continuousControllerRunning || !_continuousControllerRoundInactiveObserved)
            return;
        if (_continuousControllerRoundStartRequestIssued)
        {
            if (_continuousControllerTick - _continuousControllerRoundStartRequestTick >
                ContinuousBotControllerContract.RoundStartObservationTimeoutTicks)
            {
                StopContinuousController("continuous_round_start_observation_timeout");
            }
            return;
        }

        var menu = scope.GameMenu;
        var view = menu?.menuView;
        var promptVisible = menu is not null && menu.IsMenuOpen && view is not null &&
                            view.CurrentPane == GameMenuView.Pane.PostFight &&
                            view.postFightContinueButton is not null &&
                            view.postFightContinueButton.visible;
        var promptEnabled = promptVisible &&
                            view!.postFightContinueButton!.enabledInHierarchy;
        var inactiveTicks = _continuousControllerRoundInactiveTick < 0
            ? 0
            : _continuousControllerTick - _continuousControllerRoundInactiveTick;
        if (inactiveTicks >
            ContinuousBotControllerContract.RoundStartObservationTimeoutTicks)
        {
            StopContinuousController(
                "continuous_round_start_prompt_observation_timeout");
            return;
        }
        var sessionMatches = _continuousControllerSessionIdentity is not null &&
                             TrialSessionIdentity.From(scope).Equals(
                                 _continuousControllerSessionIdentity);
        if (sessionMatches && promptEnabled && menu?.postFightIsWinner == false)
        {
            StopContinuousController(
                "continuous_round_restart_unavailable_after_loss_current_build_continue_exits_to_lobby");
            return;
        }
        if (!ContinuousBotControllerContract.ShouldIssueRoundStartRequest(
                sessionMatches,
                roundInactive: !scope.RoundActive,
                promptVisible,
                promptEnabled,
                postFightWinner: menu?.postFightIsWinner == true,
                requestAlreadyIssued: _continuousControllerRoundStartRequestIssued,
                inactiveTicks))
        {
            return;
        }

        _continuousControllerRoundStartRequestIssued = true;
        _continuousControllerRoundStartRequestTick = _continuousControllerTick;
        menu!.HandlePostFightContinue();
        EmitContinuousEvent(
            "continuous_round_start_lifecycle",
            "post_fight_continue_semantic_request_issued",
            frame: null,
            new
            {
                lifecycle_stage = "client_request_method_returned",
                space_gate_would_allow = true,
                semantic_method = "GameMenuController.HandlePostFightContinue",
                client_request_issued = true,
                global_space_input_emitted = false,
                request_is_one_shot_for_inactive_transition = true,
                inactive_ticks_before_request = inactiveTicks,
                observation_timeout_ticks =
                    ContinuousBotControllerContract.RoundStartObservationTimeoutTicks,
                two_minute_limit_ticks =
                    ContinuousBotControllerContract.TwoMinuteTicks,
            });
    }

    private void SuspendContinuousController(string reason, bool roundInactive)
    {
        if (_attackZoneTrialRunning)
        {
            StopAttackZoneTrialInternal(reason, "trial_interrupted");
            return;
        }
        if (!_continuousControllerRunning)
            return;
        var phase = roundInactive ? "suspended_round_inactive" : "suspended_scope_unproven";
        var changed = !string.Equals(_continuousControllerPhase, phase, StringComparison.Ordinal) ||
                      !string.Equals(_continuousControllerSuspendReason, reason, StringComparison.Ordinal);
        var lastFrame = _continuousControllerLastFrame;
        TryCancelExactOwnedContinuousPendingRequests();
        InterruptContinuousAction(reason, lastFrame);
        TryNeutralOwnedContinuousController();
        _continuousControllerVelocity = Vector3.zero;
        _continuousControllerPhase = phase;
        _continuousControllerSuspendReason = reason;
        if (roundInactive)
        {
            if (!_continuousControllerRoundInactiveObserved)
                _continuousControllerRoundInactiveTick = _continuousControllerTick;
            _continuousControllerRoundInactiveObserved = true;
            _continuousControllerLastRoundIdentitySha256 =
                _continuousControllerRoundIdentitySha256 ??
                _continuousControllerLastRoundIdentitySha256;
            _continuousControllerRoundIdentity = null;
            _continuousControllerRoundIdentitySha256 = null;
            _continuousControllerInput = null;
            _continuousControllerInputPointer = IntPtr.Zero;
            _continuousControllerLastFrame = null;
            _continuousControllerLastRoundMetrics = null;
            _continuousControllerStraightenIssued = false;
            _continuousControllerRecoveryEpisodeActive = false;
        }
        else
        {
            _continuousControllerStraightenIssued = false;
            _continuousControllerRecoveryEpisodeActive = false;
            _continuousControllerRecoveryStage = "inactive";
        }
        ClearContinuousPendingRequestState();
        if (changed)
        {
            EmitContinuousEvent(
                "continuous_controller_suspend",
                reason,
                lastFrame,
                new
                {
                    round_inactive = roundInactive,
                    requires_full_scope_revalidation_before_resume = true,
                });
        }
    }

    private void StopContinuousController(string reason)
    {
        if (!_continuousControllerRunning)
            return;
        if (_attackZoneTrialRunning)
        {
            StopAttackZoneTrialInternal(reason, "trial_interrupted");
            return;
        }
        var frame = _continuousControllerLastFrame;
        if (_attackZoneRecoveryOnlyRunning)
        {
            EmitAttackZoneEvent(
                "recovery_state_observed",
                "attack_zone_recovery_only_interrupted",
                frame,
                new
                {
                    interruption_reason = reason,
                    recovery_only = true,
                    attack_requested = false,
                    upright_readiness_consecutive_ticks = _attackZoneRecoveryReadyTicks,
                });
            _attackZoneRecoveryOnlyRunning = false;
            _attackZonePhase = "inactive";
            _attackZoneTarget = null;
            _attackZoneSettleTracker = null;
        }
        var runId = _continuousControllerRunId;
        var roundIdentity = _continuousControllerRoundIdentitySha256;
        var authorized = _continuousControllerAuthorizedWhileBackground;
        TryCancelExactOwnedContinuousPendingRequests();
        InterruptContinuousAction(reason, frame);
        _continuousControllerRunning = false;
        _continuousControllerAuthorizedWhileBackground = false;
        TryNeutralOwnedContinuousController();
        _continuousControllerPhase = "inactive";
        _continuousControllerSuspendReason = reason;
        ClearContinuousActionState();
        ClearContinuousPendingRequestState();

        try
        {
            var payload = new
            {
                @event = "continuous_controller_end",
                protocol = "rek.ui_bridge.v1",
                continuous_controller_schema = ContinuousBotControllerContract.Schema,
                continuous_controller_sha256 = _continuousControllerSha256,
                authority_scope = ContinuousBotControllerContract.AuthorityScope,
                authority_caveat = ContinuousBotControllerContract.AuthorityCaveat,
                continuous_controller_run_id = runId,
                round_identity_sha256 = roundIdentity,
                reason,
                authorized_while_background = authorized,
                client_request_observation_mode = true,
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
                utc = DateTimeOffset.UtcNow,
                stopwatch_timestamp_ticks = System.Diagnostics.Stopwatch.GetTimestamp(),
                stopwatch_frequency_hz = System.Diagnostics.Stopwatch.Frequency,
                unity_frame = Time.frameCount,
                unity_time = Time.timeAsDouble,
                unity_fixed_time = Time.fixedTimeAsDouble,
            };
            _pipe?.Send(_leaseConnectionId, payload);
            Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        }
        catch
        {
        }

        _continuousControllerSessionIdentity = null;
        _continuousControllerRoundIdentity = null;
        _continuousControllerRoundIdentitySha256 = null;
        _continuousControllerInput = null;
        _continuousControllerInputPointer = IntPtr.Zero;
        _continuousControllerVelocity = Vector3.zero;
        _continuousControllerLastFrame = null;
        _continuousControllerLastRoundMetrics = null;
        _continuousControllerRoundInactiveObserved = false;
        _continuousControllerRecoveryEpisodeActive = false;
        _attackZoneRecoveryReadyTicks = 0;
    }

    private static bool TryValidateContinuousPairing(
        MeasuredPairing pairing,
        out string reason)
    {
        if (pairing.LocalSlot is not (0 or 1) ||
            pairing.OpponentSlot != 1 - pairing.LocalSlot ||
            pairing.LocalFighter is null || pairing.OpponentFighter is null)
        {
            reason = "continuous_pairing_slots_or_fighters_missing";
            return false;
        }
        if (!ContinuousBotControllerContract.HasRequiredT800Pairing(
                pairing.Validation.LocalSemanticT800,
                pairing.Validation.LocalExactT800BoneSignature,
                pairing.Validation.OpponentExactT800BoneSignature))
        {
            reason = !pairing.Validation.LocalSemanticT800
                ? "continuous_local_semantic_robot_id_not_exact_t800"
                : !pairing.Validation.LocalExactT800BoneSignature
                    ? "continuous_local_runtime_t800_signature_not_exact"
                    : "continuous_opponent_runtime_t800_signature_not_exact";
            return false;
        }

        var opponent = pairing.OpponentFighter;
        if (string.IsNullOrWhiteSpace(opponent.RuntimeObjectName))
        {
            reason = "continuous_opponent_runtime_object_name_missing";
            return false;
        }
        if (opponent.BoneNames is null || opponent.BoneNames.Count == 0 ||
            opponent.BoneNames.Any(string.IsNullOrWhiteSpace))
        {
            reason = "continuous_opponent_runtime_bone_identity_missing_or_nonunique";
            return false;
        }
        if (opponent.BoneNames.Distinct(StringComparer.Ordinal).Count() !=
            opponent.BoneNames.Count)
        {
            reason = "continuous_opponent_runtime_bone_identity_nonunique";
            return false;
        }

        reason = "continuous_exact_runtime_t800_pairing_proven_with_opponent_semantic_id_non_authoritative";
        return true;
    }

    private static bool TryValidateContinuousMoveMap(
        RobotInputController input,
        out string reason)
    {
        reason = string.Empty;
        var config = input.Config;
        if (config is null)
        {
            reason = "continuous_local_robot_config_missing";
            return false;
        }
        foreach (var attack in ContinuousBotControllerContract.Attacks)
        {
            var clip = config.GetMove(attack.MoveIndex);
            if (clip is null ||
                !string.Equals(clip.name, attack.MoveName, StringComparison.Ordinal))
            {
                reason = $"continuous_move_{attack.MoveIndex}_runtime_identity_mismatch";
                return false;
            }
        }
        return true;
    }

    private static bool TryCaptureContinuousFrame(
        PrivateAiContext scope,
        MeasuredPairing pairing,
        out ContinuousFrame frame,
        out string reason)
    {
        frame = null!;
        if (!TryValidateContinuousPairing(pairing, out reason))
            return false;
        var fighters = scope.Coordinator.Fighters;
        var input = scope.Input;
        if (fighters is null || fighters.Length != 2 ||
            fighters[0] is null || fighters[1] is null || input is null)
        {
            reason = "continuous_runtime_fighter_or_input_missing";
            return false;
        }
        var localRobot = fighters[scope.LocalSlot];
        var opponentRobot = fighters[scope.OpponentSlot];
        if (NativePointer(localRobot) == IntPtr.Zero ||
            NativePointer(opponentRobot) == IntPtr.Zero ||
            NativePointer(localRobot) == NativePointer(opponentRobot) ||
            NativePointer(input.Robot) != NativePointer(localRobot))
        {
            reason = "continuous_runtime_fighter_pointer_identity_invalid";
            return false;
        }
        var localRoot = localRobot.RootTransform;
        var opponentRoot = opponentRobot.RootTransform;
        var composer = input.Composer;
        if (localRoot is null || opponentRoot is null || composer is null)
        {
            reason = "continuous_root_transform_or_composer_missing";
            return false;
        }

        var localPosition = localRoot.position;
        var opponentPosition = opponentRoot.position;
        var localRotation = localRoot.rotation;
        var opponentRotation = opponentRoot.rotation;
        var localForward = localRoot.forward;
        var opponentForward = opponentRoot.forward;
        var localLinearVelocity = localRobot.RootLinearVelocity;
        var localAngularVelocity = localRobot.RootAngularVelocity;
        var opponentLinearVelocity = opponentRobot.RootLinearVelocity;
        var opponentAngularVelocity = opponentRobot.RootAngularVelocity;
        if (!Finite(localPosition) || !Finite(opponentPosition) ||
            !Finite(localRotation) || !Finite(opponentRotation) ||
            !Finite(localForward) || !Finite(opponentForward) ||
            !Finite(localLinearVelocity) || !Finite(localAngularVelocity) ||
            !Finite(opponentLinearVelocity) || !Finite(opponentAngularVelocity) ||
            !ContinuousBotControllerContract.TryComputePlanarGeometry(
                localPosition.x,
                localPosition.z,
                localForward.x,
                localForward.z,
                opponentPosition.x,
                opponentPosition.z,
                opponentForward.x,
                opponentForward.z,
                out var geometry))
        {
            reason = "continuous_planar_fighter_geometry_not_finite_or_degenerate";
            return false;
        }

        var local = pairing.LocalFighter!;
        var opponent = pairing.OpponentFighter!;
        var localBoneSignature = HashText(string.Join("\n", local.BoneNames!));
        var opponentBoneSignature = HashText(string.Join("\n", opponent.BoneNames!));
        var opponentRuntimeIdentity = HashText(
            $"{opponent.RuntimeObjectName}\n{opponentBoneSignature}");
        var opponentRuntimeIsT800 = pairing.Validation.OpponentExactT800BoneSignature;
        var opponentSemanticDeclaresT800 = pairing.Validation.OpponentSemanticT800;
        var semanticRuntimeConsistency =
            ContinuousBotControllerContract.ClassifySemanticRuntimeConsistency(
                opponentSemanticDeclaresT800,
                opponentRuntimeIsT800);

        var round = scope.Round;
        var cleanHits = round?.CleanHits;
        var falls = round?.Falls;
        if (round is null || cleanHits is null || cleanHits.Length != 2 ||
            falls is null || falls.Length != 2)
        {
            reason = "continuous_round_metrics_missing";
            return false;
        }

        frame = new ContinuousFrame(
            input,
            localRobot,
            opponentRobot,
            local.SemanticRobotId,
            local.RuntimeObjectName!,
            local.BoneNames!.Count,
            localBoneSignature,
            opponent.SemanticRobotId,
            opponent.RuntimeObjectName!,
            opponent.BoneNames!.Count,
            opponentBoneSignature,
            opponentRuntimeIdentity,
            semanticRuntimeConsistency.Mismatch,
            semanticRuntimeConsistency.Classification,
            localPosition,
            localRotation,
            localForward,
            opponentPosition,
            opponentRotation,
            opponentForward,
            localLinearVelocity,
            localAngularVelocity,
            opponentLinearVelocity,
            opponentAngularVelocity,
            geometry,
            localRobot.IsFalling,
            localRobot.IsFallen,
            localRobot.IsDampened,
            localRobot.RecoveryArmed,
            localRobot.GetUpPending,
            localRobot.IsResetting,
            localRobot.IsMotorShutdown,
            localRobot.SuggestedGetUpOrientation,
            opponentRobot.IsFalling,
            opponentRobot.IsFallen,
            opponentRobot.IsDampened,
            opponentRobot.RecoveryArmed,
            opponentRobot.GetUpPending,
            opponentRobot.IsResetting,
            opponentRobot.IsMotorShutdown,
            input.IsPunching,
            input.IsRecovering,
            input.AllowMoveInterrupt,
            composer.IsActionPlaying,
            composer.IsBusy,
            composer.ActiveActionClip,
            composer.ActiveActionClip?.name,
            composer.ActiveActionClip is null
                ? IntPtr.Zero
                : NativePointer(composer.ActiveActionClip),
            composer.CurrentMoveId,
            composer.ActionClipFrame,
            composer.ActionClipFps,
            new ContinuousRoundMetrics(
                cleanHits[scope.LocalSlot],
                cleanHits[scope.OpponentSlot],
                falls[scope.LocalSlot],
                falls[scope.OpponentSlot]));
        reason = string.Empty;
        return true;
    }

    private static bool ContinuousRuntimeIdentityMatches(
        ContinuousFrame current,
        ContinuousFrame? previous) =>
        previous is null ||
        (string.Equals(
             current.LocalRuntimeObjectName,
             previous.LocalRuntimeObjectName,
             StringComparison.Ordinal) &&
         string.Equals(
             current.LocalBoneSignatureSha256,
             previous.LocalBoneSignatureSha256,
             StringComparison.Ordinal) &&
         string.Equals(
             current.OpponentRuntimeIdentitySha256,
             previous.OpponentRuntimeIdentitySha256,
             StringComparison.Ordinal));

    private bool ObserveContinuousActionLifecycle(ContinuousFrame frame)
    {
        var attack = _continuousControllerActiveAttack;
        if (attack is null)
            return true;

        if (!_continuousControllerMoveRequestObserved)
        {
            if (_continuousControllerTick - _continuousControllerActionRequestTick >
                ContinuousBotControllerContract.RequestStartTimeoutTicks)
            {
                StopContinuousController("continuous_move_client_request_edge_timeout");
                return false;
            }
            return true;
        }

        if (!_continuousControllerActionStartedObserved)
        {
            if (frame.ComposerActionPlaying)
            {
                if (frame.ActiveActionClipPointer == IntPtr.Zero ||
                    frame.ActiveActionClipPointer != _continuousControllerActiveClipPointer)
                {
                    StopContinuousController("unexpected_local_action_clip_started");
                    return false;
                }
                _continuousControllerActionStartedObserved = true;
                _continuousControllerActionStartTick = _continuousControllerTick;
                EmitContinuousEvent(
                    "continuous_action_lifecycle",
                    "local_motion_start_observed",
                    frame,
                    ContinuousActionDetail("local_motion_start_observed"));
                return true;
            }
            if (_continuousControllerTick - _continuousControllerActionRequestTick >
                ContinuousBotControllerContract.RequestStartTimeoutTicks)
            {
                StopContinuousController("continuous_local_motion_start_observation_timeout");
                return false;
            }
            return true;
        }

        if (frame.ComposerActionPlaying &&
            frame.ActiveActionClipPointer != _continuousControllerActiveClipPointer)
        {
            StopContinuousController("continuous_local_action_clip_changed_during_lifecycle");
            return false;
        }
        if (!frame.ComposerActionPlaying && !frame.ComposerBusy &&
            !frame.InputPunching && !frame.InputRecovering &&
            !frame.Input.hasPendingMove && !frame.Input.hasPendingSpecial &&
            !frame.Input.hasPendingEStop)
        {
            EmitContinuousEvent(
                "continuous_action_lifecycle",
                "local_motion_completion_and_readiness_observed",
                frame,
                ContinuousActionDetail("local_motion_completion_and_readiness_observed"));
            _continuousControllerNextAttackIndex =
                (_continuousControllerNextAttackIndex + 1) %
                ContinuousBotControllerContract.Attacks.Length;
            _continuousControllerSettleUntilTick =
                _continuousControllerTick + ContinuousBotControllerContract.SettleTicks;
            ClearContinuousActionState();
            return true;
        }
        if (_continuousControllerTick - _continuousControllerActionStartTick >
            ContinuousBotControllerContract.ActionCompletionTimeoutTicks)
        {
            StopContinuousController("continuous_local_motion_completion_timeout");
            return false;
        }
        return true;
    }

    private void TryStartContinuousAttack(
        ContinuousFrame frame,
        ContinuousAttackProfile attack)
    {
        if (_continuousControllerActiveAttack is not null ||
            frame.Input.hasPendingMove || frame.Input.hasPendingSpecial ||
            frame.Input.hasPendingEStop)
        {
            return;
        }
        var clip = frame.Input.Config?.GetMove(attack.MoveIndex);
        if (clip is null ||
            !string.Equals(clip.name, attack.MoveName, StringComparison.Ordinal))
        {
            StopContinuousController($"continuous_move_{attack.MoveIndex}_runtime_identity_lost");
            return;
        }

        _continuousControllerActionSequence++;
        _continuousControllerActiveAttack = attack;
        _continuousControllerActiveClip = clip;
        _continuousControllerActiveClipPointer = NativePointer(clip);
        _continuousControllerMoveAwaitingSend = attack.MoveIndex;
        _continuousControllerMoveArmedObserved = false;
        _continuousControllerMoveInvocationObserved = false;
        _continuousControllerMoveRequestObserved = false;
        _continuousControllerActionStartedObserved = false;
        _continuousControllerActionRequestTick = _continuousControllerTick;
        _continuousControllerActionStartTick = -1;

        if (!frame.Input.ExecuteMoveByIndex(attack.MoveIndex))
        {
            StopContinuousController($"continuous_move_{attack.MoveIndex}_rejected_locally");
            return;
        }
        EmitContinuousEvent(
            "continuous_action_lifecycle",
            "local_command_edge_set",
            frame,
            ContinuousActionDetail("local_command_edge_set"));
    }

    private static bool ContinuousLocalActionReady(ContinuousFrame frame) =>
        !frame.LocalFalling && !frame.LocalFallen && !frame.LocalResetting &&
        !frame.LocalMotorShutdown && !frame.LocalGetUpPending &&
        !frame.InputPunching && !frame.InputRecovering &&
        !frame.ComposerActionPlaying && !frame.ComposerBusy &&
        !frame.Input.hasPendingMove && !frame.Input.hasPendingSpecial &&
        !frame.Input.hasPendingEStop;

    private bool TryCaptureFreshContinuousMoveSendFrame(
        PrivateAiContext scope,
        RobotInputController input,
        out ContinuousFrame frame,
        out string reason)
    {
        var pairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        if (!TryCaptureContinuousFrame(scope, pairing, out frame, out reason))
            return false;
        if (_continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(input) != _continuousControllerInputPointer ||
            NativePointer(frame.Input) != _continuousControllerInputPointer ||
            !ContinuousRuntimeIdentityMatches(frame, _continuousControllerLastFrame))
        {
            reason = "continuous_move_send_runtime_identity_changed";
            return false;
        }
        if (ContinuousBotControllerContract.LocalBlocksFreshMoveSend(
                frame.LocalFalling,
                frame.LocalFallen,
                frame.LocalDampened,
                frame.LocalRecoveryArmed,
                frame.LocalGetUpPending,
                frame.LocalResetting,
                frame.LocalMotorShutdown,
                frame.InputRecovering))
        {
            reason = "continuous_move_send_local_down_or_recovering";
            return false;
        }
        if (ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(
                frame.OpponentFalling,
                frame.OpponentFallen,
                frame.OpponentDampened,
                frame.OpponentRecoveryArmed,
                frame.OpponentGetUpPending,
                frame.OpponentResetting,
                frame.OpponentMotorShutdown))
        {
            reason = "continuous_move_send_opponent_down_or_recovering";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    private void CancelContinuousMoveAtFreshSendGuard(
        RobotInputController input,
        ContinuousFrame? frame,
        string reason)
    {
        var moveIndex = _continuousControllerMoveAwaitingSend;
        if (moveIndex is null ||
            (input.hasPendingMove && input.pendingMoveIndex != moveIndex.Value))
        {
            StopContinuousController(
                $"{reason}:continuous_move_pending_state_not_owned_for_cancel");
            return;
        }
        if (input.hasPendingMove)
            input.hasPendingMove = false;
        _continuousControllerVelocity = Vector3.zero;
        if (!SetVelocityExact(input, Vector3.zero))
        {
            StopContinuousController($"{reason}:continuous_neutral_readback_mismatch");
            return;
        }
        if (frame is not null)
            _continuousControllerLastFrame = frame;
        if (_attackZoneTrialRunning)
        {
            var localRecovery = frame is not null &&
                AttackZoneTrialContract.ClassifyCensorDisposition(
                    frame.LocalFalling,
                    frame.LocalFallen,
                    frame.LocalDampened,
                    frame.LocalRecoveryArmed,
                    frame.LocalGetUpPending,
                    frame.LocalResetting,
                    frame.LocalMotorShutdown,
                    frame.InputRecovering,
                    AttackZoneOpponentUnhealthy(frame)) ==
                AttackZoneCensorDisposition.ContinueLocalRecovery;
            var downOrRecoveryGuard = reason is
                "continuous_move_send_local_down_or_recovering" or
                "continuous_move_send_opponent_down_or_recovering";
            StopAttackZoneTrialInternal(
                $"pre_start_move_cancelled_at_fresh_send_guard:{reason}",
                downOrRecoveryGuard ? "trial_censored" : "trial_interrupted",
                continueLocalRecovery: localRecovery);
            return;
        }
        InterruptContinuousAction(reason, frame);
        _continuousControllerPhase = "round_active_move_cancelled_at_fresh_send_guard";
    }

    private void DriveContinuousRecovery(ContinuousFrame frame)
    {
        if (!_continuousControllerRecoveryEpisodeActive)
        {
            _continuousControllerRecoveryEpisodeActive = true;
            _continuousControllerRecoverySequence++;
            _continuousControllerRecoveryStage = "normal_recovery_guard";
            _continuousControllerRecoveryStageTick = _continuousControllerTick;
            _continuousControllerStraightenIssued = false;
            EmitContinuousEvent(
                "continuous_recovery_lifecycle",
                "fall_recovery_episode_started",
                frame,
                ContinuousRecoveryDetail("fall_recovery_episode_started"));
        }

        if (_continuousControllerTick - _continuousControllerRecoveryStageTick >
                ContinuousBotControllerContract.RecoveryObservationTimeoutTicks)
        {
            StopContinuousController(
                $"continuous_recovery_observation_timeout:{_continuousControllerRecoveryStage}");
            return;
        }
        if (_continuousControllerEStopAwaitingSend ||
            _continuousControllerSpecialAwaitingSend is not null)
        {
            return;
        }
        switch (_continuousControllerRecoveryStage)
        {
            case "normal_recovery_guard":
            case "await_dampened_after_dampen":
            case "await_recovery_armed_after_straighten":
            {
                if (frame.LocalMotorShutdown)
                {
                    StopContinuousController("motor_shutdown_reappeared_before_get_up");
                    return;
                }
                var decision = ContinuousBotControllerContract.SelectRecoveryCommand(
                    fallen: frame.LocalFallen,
                    dampened: frame.LocalDampened,
                    recoveryArmed: frame.LocalRecoveryArmed,
                    motorShutdown: frame.LocalMotorShutdown,
                    straightenIssued: _continuousControllerStraightenIssued,
                    suggestedProne:
                        frame.SuggestedGetUpOrientation == GetUpOrientation.Prone);
                if (decision == ContinuousRecoveryCommand.Dampen)
                {
                    if (_continuousControllerRecoveryStage ==
                            "await_dampened_after_dampen" &&
                        (long)_continuousControllerTick -
                            _continuousControllerLastRecoveryRequestTick <
                            ContinuousBotControllerContract.RecoveryRetryTicks)
                    {
                        _continuousControllerPhase =
                            "round_active_recovery_await_dampened_after_dampen";
                        return;
                    }
                    RequestContinuousSpecial(frame, SpecialCommand.Dampen, "dampen");
                    return;
                }
                if (decision == ContinuousRecoveryCommand.Straighten)
                {
                    RequestContinuousSpecial(frame, SpecialCommand.Straighten, "straighten");
                    _continuousControllerStraightenIssued = true;
                    return;
                }
                if (decision ==
                    ContinuousRecoveryCommand.WaitForDampenedOrRecoveryArmed)
                {
                    _continuousControllerPhase =
                        "round_active_recovery_await_recovery_armed_after_straighten";
                    return;
                }
                if (decision == ContinuousRecoveryCommand.GetUpProne)
                {
                    RequestContinuousSpecial(frame, SpecialCommand.GetUpProne, "get_up_prone");
                    return;
                }
                if (decision == ContinuousRecoveryCommand.GetUpSupine)
                {
                    RequestContinuousSpecial(frame, SpecialCommand.GetUpSupine, "get_up_supine");
                    return;
                }
                return;
            }
            case "await_upright":
                return;
            default:
                StopContinuousController("continuous_recovery_stage_invalid");
                return;
        }
    }

    private static bool IsContinuousFaultEStopStage(string stage) =>
        stage is "fault_estop_delay" or "await_motor_shutdown" or
            "fault_estop_hold" or "need_estop_off" or "await_motor_running";

    private void DriveContinuousFaultEStopCycle(ContinuousFrame frame)
    {
        var straightenIssuedOnFaultEntry =
            ContinuousBotControllerContract.ResolveStraightenIssuedOnFaultEntry(
                _continuousControllerRecoveryEpisodeActive,
                _continuousControllerStraightenIssued);
        if (!_continuousControllerRecoveryEpisodeActive)
        {
            _continuousControllerRecoveryEpisodeActive = true;
            _continuousControllerRecoverySequence++;
            _continuousControllerStraightenIssued = straightenIssuedOnFaultEntry;
            _continuousControllerRecoveryStage = "fault_estop_delay";
            _continuousControllerRecoveryStageTick = _continuousControllerTick;
            EmitContinuousEvent(
                "continuous_recovery_lifecycle",
                "motor_shutdown_hold_fault_episode_started",
                frame,
                ContinuousRecoveryDetail("motor_shutdown_hold_fault_episode_started"));
        }
        else if (!IsContinuousFaultEStopStage(_continuousControllerRecoveryStage))
        {
            if (_continuousControllerSpecialAwaitingSend is not null ||
                frame.Input.hasPendingSpecial)
            {
                StopContinuousController(
                    "motor_shutdown_fault_observed_during_owned_special_request");
                return;
            }
            _continuousControllerStraightenIssued = straightenIssuedOnFaultEntry;
            _continuousControllerRecoveryStage = "fault_estop_delay";
            _continuousControllerRecoveryStageTick = _continuousControllerTick;
            EmitContinuousEvent(
                "continuous_recovery_lifecycle",
                "motor_shutdown_hold_fault_preempted_normal_recovery",
                frame,
                ContinuousRecoveryDetail(
                    "motor_shutdown_hold_fault_preempted_normal_recovery"));
        }

        if (_continuousControllerTick - _continuousControllerRecoveryStageTick >
                ContinuousBotControllerContract.RecoveryObservationTimeoutTicks)
        {
            StopContinuousController(
                $"continuous_fault_estop_observation_timeout:{_continuousControllerRecoveryStage}");
            return;
        }
        if (_continuousControllerEStopAwaitingSend)
            return;

        switch (_continuousControllerRecoveryStage)
        {
            case "fault_estop_delay":
                if (!frame.LocalMotorShutdown)
                {
                    StopContinuousController(
                        "motor_shutdown_hold_cleared_before_fault_estop_delay");
                    return;
                }
                if (_continuousControllerTick - _continuousControllerRecoveryStageTick <
                    ContinuousBotControllerContract.FaultEStopDelayTicks)
                {
                    return;
                }
                RequestContinuousEStopToggle(frame, "fault_estop_toggle_on");
                return;
            case "await_motor_shutdown":
                if (frame.LocalMotorShutdown)
                {
                    _continuousControllerRecoveryStage = "fault_estop_hold";
                    _continuousControllerRecoveryStageTick = _continuousControllerTick;
                    EmitContinuousEvent(
                        "continuous_recovery_lifecycle",
                        "motor_shutdown_observed_after_fault_estop_toggle_on",
                        frame,
                        ContinuousRecoveryDetail(
                            "motor_shutdown_observed_after_fault_estop_toggle_on"));
                }
                return;
            case "fault_estop_hold":
                if (!frame.LocalMotorShutdown)
                {
                    StopContinuousController(
                        "motor_running_observed_before_owned_fault_estop_toggle_off");
                    return;
                }
                if (_continuousControllerTick - _continuousControllerRecoveryStageTick <
                    ContinuousBotControllerContract.FaultEStopHoldTicks)
                {
                    return;
                }
                _continuousControllerRecoveryStage = "need_estop_off";
                _continuousControllerRecoveryStageTick = _continuousControllerTick;
                RequestContinuousEStopToggle(frame, "fault_estop_toggle_off");
                return;
            case "need_estop_off":
                if (!frame.LocalMotorShutdown)
                {
                    StopContinuousController(
                        "motor_shutdown_not_observed_at_fault_estop_toggle_off_edge");
                    return;
                }
                RequestContinuousEStopToggle(frame, "fault_estop_toggle_off");
                return;
            case "await_motor_running":
                if (!frame.LocalMotorShutdown)
                {
                    _continuousControllerRecoveryStage = frame.LocalFallen
                        ? "normal_recovery_guard"
                        : "await_upright";
                    _continuousControllerRecoveryStageTick = _continuousControllerTick;
                    EmitContinuousEvent(
                        "continuous_recovery_lifecycle",
                        "motor_running_observed_after_fault_estop_toggle_off",
                        frame,
                        ContinuousRecoveryDetail(
                            "motor_running_observed_after_fault_estop_toggle_off"));
                }
                return;
            default:
                StopContinuousController("continuous_fault_estop_stage_invalid");
                return;
        }
    }

    private void RequestContinuousEStopToggle(ContinuousFrame frame, string purpose)
    {
        if (frame.Input.hasPendingEStop || _continuousControllerEStopAwaitingSend)
        {
            StopContinuousController("continuous_recovery_estop_edge_not_clean");
            return;
        }
        _continuousControllerEStopAwaitingSend = true;
        _continuousControllerEStopInvocationObserved = false;
        _continuousControllerSpecialPurpose = purpose;
        _continuousControllerLastRecoveryRequestTick = _continuousControllerTick;
        frame.Input.ToggleEStop();
        if (!frame.Input.hasPendingEStop)
        {
            StopContinuousController("continuous_recovery_estop_not_armed_locally");
            return;
        }
        EmitContinuousEvent(
            "continuous_recovery_lifecycle",
            "local_estop_toggle_edge_set",
            frame,
            ContinuousRecoveryDetail("local_estop_toggle_edge_set"));
    }

    private void RequestContinuousSpecial(
        ContinuousFrame frame,
        SpecialCommand command,
        string purpose)
    {
        if (frame.Input.hasPendingSpecial ||
            _continuousControllerSpecialAwaitingSend is not null)
        {
            StopContinuousController("continuous_recovery_special_edge_not_clean");
            return;
        }
        _continuousControllerSpecialAwaitingSend = command;
        _continuousControllerSpecialArmedObserved = false;
        _continuousControllerSpecialInvocationObserved = false;
        _continuousControllerSpecialPurpose = purpose;
        _continuousControllerLastRecoveryRequestTick = _continuousControllerTick;
        if (!frame.Input.ExecuteSpecial(command) ||
            !frame.Input.hasPendingSpecial ||
            frame.Input.pendingSpecialCommand != (int)command)
        {
            StopContinuousController($"continuous_recovery_{purpose}_not_armed_locally");
            return;
        }
        _continuousControllerSpecialArmedObserved = true;
        EmitContinuousEvent(
            "continuous_recovery_lifecycle",
            "local_special_command_edge_set",
            frame,
            ContinuousRecoveryDetail("local_special_command_edge_set"));
    }

    private void SetContinuousVelocity(
        RobotInputController input,
        Vector3 velocity,
        string purpose)
    {
        if (!_continuousControllerRunning || VelocityEquals(_continuousControllerVelocity, velocity))
            return;
        if (_continuousControllerVelocityPurposeAwaitingSend is not null)
        {
            StopContinuousController("continuous_previous_velocity_request_not_observed");
            return;
        }
        _continuousControllerVelocity = velocity;
        _continuousControllerVelocityPurposeAwaitingSend = purpose;
        _continuousControllerVelocityInvocationObserved = false;
        if (!SetVelocityExact(input, velocity))
        {
            StopContinuousController("continuous_velocity_readback_mismatch");
            return;
        }
        var detail = new
        {
            lifecycle_stage = "local_velocity_command_edge_set",
            purpose,
            velocity_command_xyz = new[] { velocity.x, velocity.y, velocity.z },
        };
        if ((_attackZoneTrialRunning || _attackZoneRecoveryOnlyRunning) &&
            VelocityEquals(velocity, Vector3.zero))
        {
            EmitAttackZoneEvent(
                "neutral_command_edge_set",
                purpose,
                _continuousControllerLastFrame,
                detail);
        }
        else
        {
            EmitContinuousEvent(
                "continuous_velocity_lifecycle",
                "local_velocity_command_edge_set",
                _continuousControllerLastFrame,
                detail);
        }
    }

    private object ContinuousActionDetail(string stage) => new
    {
        action_sequence = _continuousControllerActionSequence,
        lifecycle_stage = stage,
        move_profile = _continuousControllerActiveAttack is null
            ? null
            : ContinuousAttackProfilePayload(_continuousControllerActiveAttack),
        local_command_tick = _continuousControllerActionRequestTick,
        local_motion_start_tick = _continuousControllerActionStartedObserved
            ? _continuousControllerActionStartTick
            : (int?)null,
        local_motion_readiness_observed =
            string.Equals(
                stage,
                "local_motion_completion_and_readiness_observed",
                StringComparison.Ordinal),
    };

    private object ContinuousRecoveryDetail(string stage) => new
    {
        recovery_sequence = _continuousControllerRecoverySequence,
        lifecycle_stage = stage,
        recovery_stage = _continuousControllerRecoveryStage,
        request_purpose = _continuousControllerSpecialPurpose,
        special_command = _continuousControllerSpecialAwaitingSend?.ToString(),
        estop_toggle_request = _continuousControllerEStopAwaitingSend,
        recovery_guard_provenance =
            ContinuousBotControllerContract.RecoveryGuardProvenance,
        fault_estop_provenance =
            ContinuousBotControllerContract.FaultEStopProvenance,
        dampen_guard = ContinuousBotControllerContract.DampenGuard,
        straighten_guard = ContinuousBotControllerContract.StraightenGuard,
        straighten_issued = _continuousControllerStraightenIssued,
    };

    private void InterruptContinuousAction(string reason, ContinuousFrame? frame)
    {
        if (_continuousControllerActiveAttack is null)
            return;
        EmitContinuousEvent(
            "continuous_action_lifecycle",
            "interrupted",
            frame,
            new
            {
                action_sequence = _continuousControllerActionSequence,
                lifecycle_stage = "interrupted",
                interruption_reason = reason,
                move_profile = ContinuousAttackProfilePayload(
                    _continuousControllerActiveAttack),
                client_request_edge_observed = _continuousControllerMoveRequestObserved,
                local_motion_start_observed = _continuousControllerActionStartedObserved,
            });
        ClearContinuousActionState();
    }

    private void ClearContinuousActionState()
    {
        _continuousControllerActiveAttack = null;
        _continuousControllerActiveClip = null;
        _continuousControllerActiveClipPointer = IntPtr.Zero;
        _continuousControllerMoveAwaitingSend = null;
        _continuousControllerMoveArmedObserved = false;
        _continuousControllerMoveInvocationObserved = false;
        _continuousControllerMoveRequestObserved = false;
        _continuousControllerActionStartedObserved = false;
        _continuousControllerActionRequestTick = -1;
        _continuousControllerActionStartTick = -1;
    }

    private void ClearContinuousPendingRequestState()
    {
        _continuousControllerVelocityPurposeAwaitingSend = null;
        _continuousControllerVelocityInvocationObserved = false;
        _continuousControllerSpecialAwaitingSend = null;
        _continuousControllerSpecialArmedObserved = false;
        _continuousControllerSpecialInvocationObserved = false;
        _continuousControllerSpecialPurpose = null;
        _continuousControllerEStopAwaitingSend = false;
        _continuousControllerEStopInvocationObserved = false;
    }

    private void EmitContinuousTelemetryIfDue(ContinuousFrame frame)
    {
        if (_continuousControllerTick % ContinuousBotControllerContract.TelemetryIntervalTicks != 0)
            return;
        _continuousControllerTelemetrySequence++;
        EmitContinuousEvent(
            "continuous_controller_telemetry",
            "periodic_measured_state",
            frame,
            new
            {
                telemetry_sequence = _continuousControllerTelemetrySequence,
                active_attack = _continuousControllerActiveAttack is null
                    ? null
                    : ContinuousAttackProfilePayload(_continuousControllerActiveAttack),
                next_attack = ContinuousAttackProfilePayload(
                    ContinuousBotControllerContract.Attacks[_continuousControllerNextAttackIndex]),
            });
    }

    private void EmitContinuousRoundObservation(ContinuousFrame frame, string reason)
    {
        if (_continuousControllerLastRoundMetrics is not null &&
            _continuousControllerLastRoundMetrics.Equals(frame.RoundMetrics))
        {
            return;
        }
        var previous = _continuousControllerLastRoundMetrics;
        _continuousControllerLastRoundMetrics = frame.RoundMetrics;
        EmitContinuousEvent(
            "continuous_round_observation",
            reason,
            frame,
            new
            {
                observation_is_not_causally_attributed_to_any_request = true,
                local_clean_hits = frame.RoundMetrics.LocalCleanHits,
                opponent_clean_hits = frame.RoundMetrics.OpponentCleanHits,
                local_falls = frame.RoundMetrics.LocalFalls,
                opponent_falls = frame.RoundMetrics.OpponentFalls,
                previous_local_clean_hits = previous?.LocalCleanHits,
                previous_opponent_clean_hits = previous?.OpponentCleanHits,
                previous_local_falls = previous?.LocalFalls,
                previous_opponent_falls = previous?.OpponentFalls,
            });
    }

    private void EmitContinuousEvent(
        string eventName,
        string reason,
        ContinuousFrame? frame,
        object? detail)
    {
        if (_attackZoneTrialRunning || _attackZoneRecoveryOnlyRunning)
        {
            EmitAttackZoneEvent(MapAttackZoneEventName(eventName, reason), reason, frame, detail);
            return;
        }
        try
        {
            var payload = new
            {
                @event = eventName,
                protocol = "rek.ui_bridge.v1",
                continuous_controller_schema = ContinuousBotControllerContract.Schema,
                continuous_controller_sha256 = _continuousControllerSha256,
                authority_scope = ContinuousBotControllerContract.AuthorityScope,
                authority_caveat = ContinuousBotControllerContract.AuthorityCaveat,
                range_angle_provenance = ContinuousBotControllerContract.RangeAngleProvenance,
                facing_yaw_provenance = ContinuousBotControllerContract.FacingYawProvenance,
                attack_selection_provenance =
                    ContinuousBotControllerContract.AttackSelectionProvenance,
                static_impact_timing_provenance =
                    ContinuousBotControllerContract.StaticImpactTimingProvenance,
                round_restart_limitation =
                    ContinuousBotControllerContract.RoundRestartLimitation,
                recovery_guard_provenance =
                    ContinuousBotControllerContract.RecoveryGuardProvenance,
                fault_estop_provenance =
                    ContinuousBotControllerContract.FaultEStopProvenance,
                dampen_guard = ContinuousBotControllerContract.DampenGuard,
                straighten_guard = ContinuousBotControllerContract.StraightenGuard,
                opponent_runtime_requirement =
                    ContinuousBotControllerContract.OpponentRuntimeRequirement,
                continuous_controller_run_id = _continuousControllerRunId,
                controller_phase = _continuousControllerPhase,
                controller_reason = reason,
                round_sequence = _continuousControllerRoundSequence,
                round_identity_sha256 = _continuousControllerRoundIdentitySha256,
                client_control_tick = _continuousControllerTick,
                client_fixed_substep = _continuousControllerFixedSubstep,
                fixed_substeps_per_control_tick =
                    ContinuousBotControllerContract.FixedSubstepsPerControlTick,
                measured_state = frame is null ? null : ContinuousFramePayload(frame),
                detail,
                client_request_observation_only = true,
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
                utc = DateTimeOffset.UtcNow,
                stopwatch_timestamp_ticks = System.Diagnostics.Stopwatch.GetTimestamp(),
                stopwatch_frequency_hz = System.Diagnostics.Stopwatch.Frequency,
                unity_frame = Time.frameCount,
                unity_time = Time.timeAsDouble,
                unity_fixed_time = Time.fixedTimeAsDouble,
            };
            _pipe?.Send(_leaseConnectionId, payload);
            Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        }
        catch (Exception exception)
        {
            Log.LogError(
                $"Continuous controller event emission failed: {exception.GetType().Name}");
            TryCancelExactOwnedContinuousPendingRequests();
            TryNeutralOwnedContinuousController();
            _continuousControllerRunning = false;
            _continuousControllerAuthorizedWhileBackground = false;
            _continuousControllerPhase = "inactive";
            _continuousControllerSuspendReason =
                $"continuous_event_emit_failed:{exception.GetType().Name}";
        }
    }

    private static object ContinuousFramePayload(ContinuousFrame frame) => new
    {
        local_identity = new
        {
            semantic_robot_id = frame.LocalSemanticRobotId,
            runtime_object_name = frame.LocalRuntimeObjectName,
            runtime_bone_count = frame.LocalBoneCount,
            runtime_bone_signature_sha256 = frame.LocalBoneSignatureSha256,
            exact_local_t800_proven = true,
        },
        opponent_identity = new
        {
            semantic_robot_id_untrusted_for_runtime_acceptance = frame.OpponentSemanticRobotId,
            runtime_object_name = frame.OpponentRuntimeObjectName,
            runtime_bone_count = frame.OpponentBoneCount,
            runtime_bone_signature_sha256 = frame.OpponentBoneSignatureSha256,
            runtime_identity_sha256 = frame.OpponentRuntimeIdentitySha256,
            semantic_runtime_mismatch = frame.OpponentSemanticRuntimeMismatch,
            semantic_runtime_consistency = frame.OpponentSemanticRuntimeConsistency,
            semantic_robot_id_used_for_acceptance = false,
        },
        geometry = new
        {
            planar_distance_m = frame.Geometry.DistanceMeters,
            local_bearing_to_opponent_deg =
                frame.Geometry.LocalBearingToOpponentDegrees,
            opponent_bearing_to_local_deg =
                frame.Geometry.OpponentBearingToLocalDegrees,
            local_heading_deg = frame.Geometry.LocalHeadingDegrees,
            opponent_heading_deg = frame.Geometry.OpponentHeadingDegrees,
        },
        local_root = new
        {
            position_xyz_m = new[]
            {
                frame.LocalPosition.x,
                frame.LocalPosition.y,
                frame.LocalPosition.z,
            },
            rotation_xyzw = new[]
            {
                frame.LocalRotation.x,
                frame.LocalRotation.y,
                frame.LocalRotation.z,
                frame.LocalRotation.w,
            },
            forward_xyz = new[]
            {
                frame.LocalForward.x,
                frame.LocalForward.y,
                frame.LocalForward.z,
            },
            linear_velocity_xyz_m_s = new[]
            {
                frame.LocalLinearVelocity.x,
                frame.LocalLinearVelocity.y,
                frame.LocalLinearVelocity.z,
            },
            angular_velocity_xyz_rad_s = new[]
            {
                frame.LocalAngularVelocity.x,
                frame.LocalAngularVelocity.y,
                frame.LocalAngularVelocity.z,
            },
        },
        opponent_root = new
        {
            position_xyz_m = new[]
            {
                frame.OpponentPosition.x,
                frame.OpponentPosition.y,
                frame.OpponentPosition.z,
            },
            rotation_xyzw = new[]
            {
                frame.OpponentRotation.x,
                frame.OpponentRotation.y,
                frame.OpponentRotation.z,
                frame.OpponentRotation.w,
            },
            forward_xyz = new[]
            {
                frame.OpponentForward.x,
                frame.OpponentForward.y,
                frame.OpponentForward.z,
            },
            linear_velocity_xyz_m_s = new[]
            {
                frame.OpponentLinearVelocity.x,
                frame.OpponentLinearVelocity.y,
                frame.OpponentLinearVelocity.z,
            },
            angular_velocity_xyz_rad_s = new[]
            {
                frame.OpponentAngularVelocity.x,
                frame.OpponentAngularVelocity.y,
                frame.OpponentAngularVelocity.z,
            },
        },
        local_state = new
        {
            falling = frame.LocalFalling,
            fallen = frame.LocalFallen,
            dampened = frame.LocalDampened,
            recovery_armed = frame.LocalRecoveryArmed,
            get_up_pending = frame.LocalGetUpPending,
            resetting = frame.LocalResetting,
            motor_shutdown = frame.LocalMotorShutdown,
            suggested_get_up_orientation = frame.SuggestedGetUpOrientation.ToString(),
            suggested_get_up_orientation_value = (int)frame.SuggestedGetUpOrientation,
        },
        opponent_state = new
        {
            falling = frame.OpponentFalling,
            fallen = frame.OpponentFallen,
            dampened = frame.OpponentDampened,
            recovery_armed = frame.OpponentRecoveryArmed,
            get_up_pending = frame.OpponentGetUpPending,
            resetting = frame.OpponentResetting,
            motor_shutdown = frame.OpponentMotorShutdown,
        },
        input_state = new
        {
            velocity_command_xyz = new[]
            {
                frame.Input.VelocityCommand.x,
                frame.Input.VelocityCommand.y,
                frame.Input.VelocityCommand.z,
            },
            punching = frame.InputPunching,
            recovering = frame.InputRecovering,
            allow_move_interrupt = frame.AllowMoveInterrupt,
            pending_move = frame.Input.hasPendingMove,
            pending_move_index = frame.Input.pendingMoveIndex,
            pending_special = frame.Input.hasPendingSpecial,
            pending_special_command = frame.Input.pendingSpecialCommand,
            pending_estop = frame.Input.hasPendingEStop,
        },
        local_motion = new
        {
            action_playing = frame.ComposerActionPlaying,
            busy = frame.ComposerBusy,
            active_action_clip = frame.ActiveActionClipName,
            current_move_id = frame.CurrentMoveId,
            action_clip_frame = frame.ActionClipFrame,
            action_clip_fps = frame.ActionClipFps,
        },
    };

    private static object ContinuousAttackProfilePayload(ContinuousAttackProfile attack) => new
    {
        move_index = attack.MoveIndex,
        move_name = attack.MoveName,
        display_name = attack.DisplayName,
        maximum_distance_m = attack.MaximumDistanceMeters,
        maximum_abs_bearing_degrees = attack.MaximumAbsBearingDegrees,
        range_angle_provenance = ContinuousBotControllerContract.RangeAngleProvenance,
        attack_selection_provenance =
            ContinuousBotControllerContract.AttackSelectionProvenance,
        serialized_asset_sha256 = attack.SerializedAssetSha256,
        static_impact_timing_provenance =
            ContinuousBotControllerContract.StaticImpactTimingProvenance,
        static_impact_events = attack.StaticImpactEvents.Select(value => new
        {
            impact_time_s = value.ImpactTimeSeconds,
            lead_time_s = value.LeadTimeSeconds,
            release_time_s = value.ReleaseTimeSeconds,
            limb = value.Limb,
            gain_boost = value.GainBoost,
        }).ToArray(),
    };

    private void ActivateRenderedCommandMarkers(int scheduleTick)
    {
        foreach (var marker in RenderedCommandMarkerContract.Specs)
        {
            if (marker.ScheduleTick != scheduleTick || _renderedCommandMarkers[marker.Index])
                continue;
            _renderedCommandMarkers[marker.Index] = true;
            var payload = new
            {
                @event = "rendered_command_marker_edge",
                protocol = "rek.ui_bridge.v1",
                marker_schema = RenderedCommandMarkerContract.Schema,
                render_binding = RenderedCommandMarkerContract.RenderBinding,
                transition = RenderedCommandMarkerContract.Transition,
                schedule_id = BridgeScheduleContract.ScheduleId,
                command_sequence_schema = BridgeScheduleContract.Schema,
                command_sequence_sha256 = _scheduleSha256,
                schedule_run_id = _scheduleRunId,
                schedule_tick = marker.ScheduleTick,
                client_fixed_substep = _scheduleFixedSubstep,
                selector = marker.Selector,
                command_identity = marker.CommandIdentity,
                region_px = new
                {
                    x = marker.X,
                    y = marker.Y,
                    width = marker.Width,
                    height = marker.Height,
                },
                pre_rgb = RenderedCommandMarkerContract.PreRgb,
                post_rgb = RenderedCommandMarkerContract.PostRgb,
                marker_state = "post",
                marker_persists_after_edge = true,
                server_acceptance_observed = false,
                unity_frame = Time.frameCount,
                unity_fixed_time = Time.fixedTimeAsDouble,
            };
            _pipe?.Send(_leaseConnectionId, payload);
            Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        }
    }

    internal void OnUnityGui()
    {
        if (!_renderedMarkerStripVisible || !ExplicitIsolatedSession.Verified)
            return;
        try
        {
            EnsureRenderedMarkerTextures();
            if (_renderedMarkerPreTexture is null || _renderedMarkerPostTexture is null)
                return;
            var previousDepth = GUI.depth;
            GUI.depth = -32000;
            try
            {
                foreach (var marker in RenderedCommandMarkerContract.Specs)
                {
                    var texture = _renderedCommandMarkers[marker.Index]
                        ? _renderedMarkerPostTexture
                        : _renderedMarkerPreTexture;
                    GUI.DrawTexture(
                        new Rect(marker.X, marker.Y, marker.Width, marker.Height),
                        texture);
                }
            }
            finally
            {
                GUI.depth = previousDepth;
            }
        }
        catch (Exception exception)
        {
            _renderedMarkerStripVisible = false;
            Log.LogError(
                $"Rendered command marker strip disabled: {exception.GetType().Name}");
        }
    }

    private void EnsureRenderedMarkerTextures()
    {
        _renderedMarkerPreTexture ??= CreateRenderedMarkerTexture(
            "rek-evidence-marker-pre", new Color(0f, 0f, 0f, 1f));
        _renderedMarkerPostTexture ??= CreateRenderedMarkerTexture(
            "rek-evidence-marker-post", new Color(1f, 0f, 1f, 1f));
    }

    private static Texture2D CreateRenderedMarkerTexture(string name, Color color)
    {
        var texture = new Texture2D(1, 1, TextureFormat.RGB24, false)
        {
            name = name,
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp,
            hideFlags = HideFlags.HideAndDontSave,
        };
        texture.SetPixel(0, 0, color);
        texture.Apply(updateMipmaps: false, makeNoLongerReadable: true);
        return texture;
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
        if (_continuousControllerRunning &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            OnContinuousControllerLateUpdatePrefix(input);
            return;
        }

        if (_singleMotionTrialRunning &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            OnSingleMotionTrialLateUpdatePrefix(input);
            return;
        }

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

    private void OnSingleMotionTrialLateUpdatePrefix(RobotInputController input)
    {
        try
        {
            if (!RequireOwnedSingleMotionTrialControl(out var controlReason))
            {
                StopSingleMotionTrial(controlReason);
                return;
            }
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var scopeReason) ||
                !ScopeMatchesSingleMotionTrial(scope))
            {
                StopSingleMotionTrial($"single_motion_trial_late_update_scope_lost:{scopeReason}");
                return;
            }
            var measuredPairing = ReadMeasuredPairing(
                scope.Coordinator,
                scope.LocalSlot,
                scope.OpponentSlot);
            if (!measuredPairing.Validation.ExactT800VersusT800)
            {
                StopSingleMotionTrial(
                    $"single_motion_trial_late_update_pairing_lost:{measuredPairing.Validation.Reason}");
                return;
            }
            if (input.hasPendingSpecial || input.hasPendingEStop)
            {
                StopSingleMotionTrial("single_motion_trial_conflicting_special_or_estop");
                return;
            }

            var moveIndex = _singleMotionTrialMoveAwaitingSend;
            if (moveIndex is null)
            {
                if (input.hasPendingMove)
                    StopSingleMotionTrial("single_motion_trial_unexpected_pending_move");
                return;
            }
            if (input.hasPendingMove && input.pendingMoveIndex != moveIndex.Value)
            {
                StopSingleMotionTrial("single_motion_trial_conflicting_pending_move");
                return;
            }
            if (!input.hasPendingMove && !input.ExecuteMoveByIndex(moveIndex.Value))
            {
                StopSingleMotionTrial($"single_motion_trial_move_{moveIndex}_rejected_at_late_update");
                return;
            }
            if (!input.hasPendingMove || input.pendingMoveIndex != moveIndex.Value)
            {
                StopSingleMotionTrial($"single_motion_trial_move_{moveIndex}_not_armed_at_late_update");
                return;
            }
            _singleMotionTrialMoveArmedObserved = true;
        }
        catch (Exception exception)
        {
            StopSingleMotionTrial($"single_motion_trial_late_update_failed:{exception.GetType().Name}");
        }
    }

    private void OnContinuousControllerLateUpdatePrefix(RobotInputController input)
    {
        try
        {
            if (!RequireOwnedContinuousControl(out var controlReason))
            {
                StopContinuousController(controlReason);
                return;
            }
            if (_continuousControllerRoundIdentity is null)
            {
                SuspendContinuousController(
                    "continuous_late_update_round_identity_unbound",
                    roundInactive: false);
                return;
            }
            if (!TryGetPrivateAiContext(
                    requireActiveRound: true,
                    out var scope,
                    out var scopeReason,
                    allowOwnedPendingEStop: _continuousControllerEStopAwaitingSend) ||
                !_continuousControllerRoundIdentity.Equals(RuntimeIdentity.From(scope)))
            {
                SuspendContinuousController(
                    $"continuous_late_update_scope_unproven:{scopeReason}",
                    roundInactive: false);
                return;
            }
            if (!VelocityEquals(input.VelocityCommand, _continuousControllerVelocity) ||
                !SetVelocityExact(input, _continuousControllerVelocity))
            {
                StopContinuousController("continuous_velocity_mismatch_at_late_update_boundary");
                return;
            }

            if (_continuousControllerMoveAwaitingSend is not null)
            {
                var moveIndex = _continuousControllerMoveAwaitingSend.Value;
                if (!TryCaptureFreshContinuousMoveSendFrame(
                        scope,
                        input,
                        out var sendFrame,
                        out var sendReason))
                {
                    CancelContinuousMoveAtFreshSendGuard(input, sendFrame, sendReason);
                    if (_continuousControllerRunning && sendReason is not (
                        "continuous_move_send_local_down_or_recovering" or
                        "continuous_move_send_opponent_down_or_recovering"))
                    {
                        StopContinuousController(sendReason);
                    }
                    return;
                }
                if (input.hasPendingSpecial || input.hasPendingEStop ||
                    (input.hasPendingMove && input.pendingMoveIndex != moveIndex))
                {
                    StopContinuousController("continuous_move_pending_conflict_at_late_update");
                    return;
                }
                if (!input.hasPendingMove && !input.ExecuteMoveByIndex(moveIndex))
                {
                    StopContinuousController($"continuous_move_{moveIndex}_rearm_rejected");
                    return;
                }
                if (!input.hasPendingMove || input.pendingMoveIndex != moveIndex)
                {
                    StopContinuousController($"continuous_move_{moveIndex}_not_armed_at_send_boundary");
                    return;
                }
                _continuousControllerMoveArmedObserved = true;
            }

            if (_continuousControllerSpecialAwaitingSend is not null)
            {
                var command = _continuousControllerSpecialAwaitingSend.Value;
                if (input.hasPendingMove || input.hasPendingEStop ||
                    (input.hasPendingSpecial && input.pendingSpecialCommand != (int)command))
                {
                    StopContinuousController("continuous_special_pending_conflict_at_late_update");
                    return;
                }
                if (!input.hasPendingSpecial && !input.ExecuteSpecial(command))
                {
                    StopContinuousController($"continuous_special_{command}_rearm_rejected");
                    return;
                }
                if (!input.hasPendingSpecial || input.pendingSpecialCommand != (int)command)
                {
                    StopContinuousController($"continuous_special_{command}_not_armed_at_send_boundary");
                    return;
                }
                _continuousControllerSpecialArmedObserved = true;
            }

            if (_continuousControllerEStopAwaitingSend)
            {
                if (input.hasPendingMove || input.hasPendingSpecial)
                {
                    StopContinuousController("continuous_estop_pending_conflict_at_late_update");
                    return;
                }
                if (!input.hasPendingEStop)
                    input.ToggleEStop();
                if (!input.hasPendingEStop)
                {
                    StopContinuousController("continuous_estop_not_armed_at_send_boundary");
                }
            }
        }
        catch (Exception exception)
        {
            StopContinuousController(
                $"continuous_late_update_failed:{exception.GetType().Name}");
        }
    }

    internal void OnSendVelocityCommandPrefix(RobotInputController input)
    {
        InvalidateFreshRoundArmFromVelocityRequest(input);
        if (_continuousControllerRunning &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            OnContinuousControllerVelocityPrefix(input);
            return;
        }
        if (_singleMotionTrialRunning &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            OnSingleMotionTrialVelocityPrefix(input);
            return;
        }

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

    private void OnSingleMotionTrialVelocityPrefix(RobotInputController input)
    {
        try
        {
            if (!RequireOwnedSingleMotionTrialControl(out var controlReason))
            {
                StopSingleMotionTrial(controlReason);
                return;
            }
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var scopeReason) ||
                !ScopeMatchesSingleMotionTrial(scope))
            {
                StopSingleMotionTrial($"single_motion_trial_velocity_scope_lost:{scopeReason}");
                return;
            }
            var measuredPairing = ReadMeasuredPairing(
                scope.Coordinator,
                scope.LocalSlot,
                scope.OpponentSlot);
            if (!measuredPairing.Validation.ExactT800VersusT800)
            {
                StopSingleMotionTrial(
                    $"single_motion_trial_velocity_pairing_lost:{measuredPairing.Validation.Reason}");
                return;
            }
            if (!VelocityEquals(input.VelocityCommand, _singleMotionTrialVelocity))
            {
                StopSingleMotionTrial("single_motion_trial_unowned_velocity_at_send_boundary");
                return;
            }
            if (!SetVelocityExact(input, _singleMotionTrialVelocity))
            {
                StopSingleMotionTrial("single_motion_trial_velocity_send_readback_mismatch");
                return;
            }

            var phase = _singleMotionTrialVelocityEdgeAwaitingSend;
            if (phase is null)
                return;
            if (phase == "neutral_pre_roll" &&
                _singleMotionTrialTick >= SingleMotionTrialContract.ActionTick)
            {
                StopSingleMotionTrial("neutral_pre_roll_velocity_request_observed_too_late");
                return;
            }
            _singleMotionTrialVelocityInvocationObserved = true;
            if (phase == "neutral_pre_roll")
                _singleMotionTrialNeutralPreRollInvocationObserved = true;
        }
        catch (Exception exception)
        {
            StopSingleMotionTrial($"single_motion_trial_velocity_prefix_failed:{exception.GetType().Name}");
        }
    }

    private void OnContinuousControllerVelocityPrefix(RobotInputController input)
    {
        try
        {
            if (!RequireOwnedContinuousControl(out var controlReason))
            {
                StopContinuousController(controlReason);
                return;
            }
            if (!VelocityEquals(input.VelocityCommand, _continuousControllerVelocity) ||
                !SetVelocityExact(input, _continuousControllerVelocity))
            {
                StopContinuousController("continuous_unowned_velocity_at_send_boundary");
                return;
            }
            if (_continuousControllerVelocityPurposeAwaitingSend is not null)
                _continuousControllerVelocityInvocationObserved = true;
        }
        catch (Exception exception)
        {
            StopContinuousController(
                $"continuous_velocity_prefix_failed:{exception.GetType().Name}");
        }
    }

    internal void OnSendVelocityCommandPostfix(RobotInputController input)
    {
        if (_continuousControllerRunning &&
            _continuousControllerVelocityInvocationObserved &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            var purpose = _continuousControllerVelocityPurposeAwaitingSend;
            if (purpose is null)
            {
                StopContinuousController("continuous_velocity_purpose_missing_at_postfix");
                return;
            }
            _continuousControllerVelocityInvocationObserved = false;
            _continuousControllerVelocityPurposeAwaitingSend = null;
            OnAttackZoneVelocityRequestReturned(purpose);
            var detail = new
            {
                lifecycle_stage = "client_request_method_returned",
                purpose,
                send_method = "RobotInputController.SendVelocityCommand",
                send_method_returned = true,
                velocity_command_xyz = new[]
                {
                    _continuousControllerVelocity.x,
                    _continuousControllerVelocity.y,
                    _continuousControllerVelocity.z,
                },
            };
            if (_attackZoneTrialRunning &&
                purpose.Contains("neutral", StringComparison.Ordinal) &&
                VelocityEquals(_continuousControllerVelocity, Vector3.zero))
            {
                EmitAttackZoneEvent(
                    "neutral_request_method_returned",
                    purpose,
                    _continuousControllerLastFrame,
                    detail);
            }
            else
            {
                EmitContinuousEvent(
                    "continuous_velocity_lifecycle",
                    "client_request_method_returned",
                    _continuousControllerLastFrame,
                    detail);
            }
            return;
        }

        if (_singleMotionTrialRunning &&
            _singleMotionTrialVelocityInvocationObserved &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            var phase = _singleMotionTrialVelocityEdgeAwaitingSend;
            if (phase is null)
            {
                StopSingleMotionTrial("single_motion_trial_velocity_phase_missing_at_postfix");
                return;
            }
            _singleMotionTrialVelocityInvocationObserved = false;
            _singleMotionTrialVelocityEdgeAwaitingSend = null;
            if (phase == "neutral_pre_roll")
            {
                _singleMotionTrialNeutralPreRollInvocationObserved = false;
                _singleMotionTrialNeutralPreRollSendObserved = true;
            }
            else if (phase == "action")
            {
                _singleMotionTrialVelocityPressSendCompletedCount++;
            }
            else if (phase == "release")
            {
                _singleMotionTrialVelocityReleaseSendCompletedCount++;
            }
            else
            {
                StopSingleMotionTrial("single_motion_trial_unknown_velocity_phase");
                return;
            }
            EmitSingleMotionTrialClientRequest("velocity", phase, moveIndex: null);
            return;
        }

        if (_scheduleRunning && _scheduleFinalNeutralInvocationObserved &&
            NativePointer(input) == _scheduleInputPointer)
        {
            _scheduleFinalNeutralInvocationObserved = false;
            _scheduleFinalNeutralSendObserved = true;
        }
    }

    internal void OnSendVelocityCommandFailure(RobotInputController input, Exception exception)
    {
        if (_continuousControllerRunning &&
            _continuousControllerVelocityInvocationObserved &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            _continuousControllerVelocityInvocationObserved = false;
            StopContinuousController(
                $"continuous_velocity_send_failed:{exception.GetType().Name}");
            return;
        }

        if (_singleMotionTrialRunning &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            _singleMotionTrialVelocityInvocationObserved = false;
            _singleMotionTrialNeutralPreRollInvocationObserved = false;
            StopSingleMotionTrial($"single_motion_trial_velocity_send_failed:{exception.GetType().Name}");
            return;
        }

        if (_scheduleRunning && _scheduleFinalNeutralInvocationObserved &&
            NativePointer(input) == _scheduleInputPointer)
        {
            _scheduleFinalNeutralInvocationObserved = false;
            StopSchedule($"velocity_send_failed:{exception.GetType().Name}");
        }
    }

    internal bool OnSendMoveEventPrefix(RobotInputController input)
    {
        InvalidateFreshRoundArmFromMoveRequest(input);
        if (_continuousControllerRunning &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            return OnContinuousControllerMovePrefix(input);
        }
        if (_singleMotionTrialRunning &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            return OnSingleMotionTrialMovePrefix(input);
        }

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

    private bool OnSingleMotionTrialMovePrefix(RobotInputController input)
    {
        var moveIndex = _singleMotionTrialMoveAwaitingSend;
        if (moveIndex is null)
        {
            StopSingleMotionTrial("single_motion_trial_unexpected_move_request");
            return true;
        }
        try
        {
            if (!RequireOwnedSingleMotionTrialControl(out var controlReason))
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                StopSingleMotionTrial(controlReason);
                return allowUnownedSend;
            }
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var scopeReason) ||
                !ScopeMatchesSingleMotionTrial(scope))
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                StopSingleMotionTrial($"single_motion_trial_move_scope_lost:{scopeReason}");
                return allowUnownedSend;
            }
            var measuredPairing = ReadMeasuredPairing(
                scope.Coordinator,
                scope.LocalSlot,
                scope.OpponentSlot);
            if (!measuredPairing.Validation.ExactT800VersusT800)
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                StopSingleMotionTrial(
                    $"single_motion_trial_move_pairing_lost:{measuredPairing.Validation.Reason}");
                return allowUnownedSend;
            }
            if (!_singleMotionTrialMoveArmedObserved ||
                !input.hasPendingMove || input.pendingMoveIndex != moveIndex.Value)
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                StopSingleMotionTrial("single_motion_trial_move_pending_state_mismatch");
                return allowUnownedSend;
            }
            _singleMotionTrialMoveInvocationObserved = true;
            return true;
        }
        catch (Exception exception)
        {
            var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
            StopSingleMotionTrial($"single_motion_trial_move_prefix_failed:{exception.GetType().Name}");
            return allowUnownedSend;
        }
    }

    private bool OnContinuousControllerMovePrefix(RobotInputController input)
    {
        var moveIndex = _continuousControllerMoveAwaitingSend;
        if (moveIndex is null)
        {
            StopContinuousController("continuous_unexpected_move_request");
            return true;
        }
        try
        {
            if (!RequireOwnedContinuousControl(out var controlReason))
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                StopContinuousController(controlReason);
                return allowUnownedSend;
            }
            if (_continuousControllerRoundIdentity is null ||
                !TryGetPrivateAiContext(
                    requireActiveRound: true,
                    out var scope,
                    out _) ||
                !_continuousControllerRoundIdentity.Equals(RuntimeIdentity.From(scope)))
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                SuspendContinuousController(
                    "continuous_move_send_scope_unproven",
                    roundInactive: false);
                return allowUnownedSend;
            }
            if (!TryCaptureFreshContinuousMoveSendFrame(
                    scope,
                    input,
                    out var sendFrame,
                    out var sendReason))
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                CancelContinuousMoveAtFreshSendGuard(input, sendFrame, sendReason);
                if (_continuousControllerRunning && sendReason is not (
                    "continuous_move_send_local_down_or_recovering" or
                    "continuous_move_send_opponent_down_or_recovering"))
                {
                    StopContinuousController(sendReason);
                }
                return allowUnownedSend;
            }
            if (!_continuousControllerMoveArmedObserved ||
                !input.hasPendingMove || input.pendingMoveIndex != moveIndex.Value)
            {
                var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
                StopContinuousController("continuous_move_pending_state_mismatch");
                return allowUnownedSend;
            }
            _continuousControllerMoveInvocationObserved = true;
            return true;
        }
        catch (Exception exception)
        {
            var allowUnownedSend = input.pendingMoveIndex != moveIndex.Value;
            StopContinuousController(
                $"continuous_move_prefix_failed:{exception.GetType().Name}");
            return allowUnownedSend;
        }
    }

    internal void OnSendMoveEventPostfix(RobotInputController input)
    {
        if (_continuousControllerRunning &&
            _continuousControllerMoveInvocationObserved &&
            _continuousControllerMoveAwaitingSend is not null &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            var moveIndex = _continuousControllerMoveAwaitingSend.Value;
            _continuousControllerMoveInvocationObserved = false;
            _continuousControllerMoveAwaitingSend = null;
            _continuousControllerMoveArmedObserved = false;
            _continuousControllerMoveRequestObserved = true;
            EmitContinuousEvent(
                "continuous_action_lifecycle",
                "client_request_method_returned",
                _continuousControllerLastFrame,
                new
                {
                    action_sequence = _continuousControllerActionSequence,
                    lifecycle_stage = "client_request_method_returned",
                    move_profile = _continuousControllerActiveAttack is null
                        ? null
                        : ContinuousAttackProfilePayload(_continuousControllerActiveAttack),
                    move_index = moveIndex,
                    send_method = "RobotInputController.SendMoveEvent",
                    send_method_returned = true,
                });
            return;
        }

        if (_singleMotionTrialRunning &&
            _singleMotionTrialMoveInvocationObserved &&
            _singleMotionTrialMoveAwaitingSend is not null &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            var moveIndex = _singleMotionTrialMoveAwaitingSend.Value;
            _singleMotionTrialMoveInvocationObserved = false;
            _singleMotionTrialMoveAwaitingSend = null;
            _singleMotionTrialMoveArmedObserved = false;
            _singleMotionTrialMoveSendCompletedCount++;
            EmitSingleMotionTrialClientRequest("move", "action", moveIndex);
            return;
        }

        if (!_scheduleRunning || !_scheduleMoveInvocationObserved ||
            _scheduleMoveAwaitingSend is null ||
            NativePointer(input) != _scheduleInputPointer)
        {
            return;
        }
        var scheduleMoveIndex = _scheduleMoveAwaitingSend.Value;
        var payload = ScheduleMoveEvent("schedule_move_send_completed", input, scheduleMoveIndex);
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
        if (_continuousControllerRunning &&
            _continuousControllerMoveInvocationObserved &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            _continuousControllerMoveInvocationObserved = false;
            StopContinuousController(
                $"continuous_move_send_failed:{exception.GetType().Name}");
            return;
        }

        if (_singleMotionTrialRunning &&
            _singleMotionTrialMoveInvocationObserved &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            StopSingleMotionTrial($"single_motion_trial_move_send_failed:{exception.GetType().Name}");
            return;
        }

        if (_scheduleRunning && _scheduleMoveInvocationObserved &&
            NativePointer(input) == _scheduleInputPointer)
        {
            StopSchedule($"move_send_failed:{exception.GetType().Name}");
        }
    }

    internal bool OnSendSpecialEventPrefix(RobotInputController input)
    {
        InvalidateFreshRoundArmFromUnexpectedRequest(input, "special");
        if (_singleMotionTrialRunning &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            StopSingleMotionTrial("single_motion_trial_unexpected_special_request");
            return true;
        }
        if (!_continuousControllerRunning ||
            _continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(input) != _continuousControllerInputPointer)
        {
            return true;
        }

        var command = _continuousControllerSpecialAwaitingSend;
        if (command is null)
        {
            StopContinuousController("continuous_unexpected_special_request");
            return true;
        }
        try
        {
            if (!RequireOwnedContinuousControl(out var controlReason))
            {
                var allowUnownedSend =
                    input.pendingSpecialCommand != (int)command.Value;
                StopContinuousController(controlReason);
                return allowUnownedSend;
            }
            if (_continuousControllerRoundIdentity is null ||
                !TryGetPrivateAiContext(
                    requireActiveRound: true,
                    out var scope,
                    out _) ||
                !_continuousControllerRoundIdentity.Equals(RuntimeIdentity.From(scope)))
            {
                var allowUnownedSend =
                    input.pendingSpecialCommand != (int)command.Value;
                SuspendContinuousController(
                    "continuous_special_send_scope_unproven",
                    roundInactive: false);
                return allowUnownedSend;
            }
            if (!_continuousControllerSpecialArmedObserved ||
                !input.hasPendingSpecial ||
                input.pendingSpecialCommand != (int)command.Value)
            {
                var allowUnownedSend =
                    input.pendingSpecialCommand != (int)command.Value;
                StopContinuousController("continuous_special_pending_state_mismatch");
                return allowUnownedSend;
            }
            _continuousControllerSpecialInvocationObserved = true;
            return true;
        }
        catch (Exception exception)
        {
            var allowUnownedSend =
                input.pendingSpecialCommand != (int)command.Value;
            StopContinuousController(
                $"continuous_special_prefix_failed:{exception.GetType().Name}");
            return allowUnownedSend;
        }
    }

    internal void OnSendSpecialEventPostfix(RobotInputController input)
    {
        if (!_continuousControllerRunning ||
            !_continuousControllerSpecialInvocationObserved ||
            _continuousControllerSpecialAwaitingSend is null ||
            _continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(input) != _continuousControllerInputPointer)
        {
            return;
        }
        var command = _continuousControllerSpecialAwaitingSend.Value;
        var purpose = _continuousControllerSpecialPurpose;
        _continuousControllerSpecialInvocationObserved = false;
        _continuousControllerSpecialAwaitingSend = null;
        _continuousControllerSpecialArmedObserved = false;
        if (command == SpecialCommand.Dampen)
        {
            _continuousControllerRecoveryStage = "await_dampened_after_dampen";
            _continuousControllerRecoveryStageTick = _continuousControllerTick;
        }
        else if (command == SpecialCommand.Straighten)
        {
            _continuousControllerRecoveryStage = "await_recovery_armed_after_straighten";
            _continuousControllerRecoveryStageTick = _continuousControllerTick;
        }
        else if (command is SpecialCommand.GetUpProne or SpecialCommand.GetUpSupine)
        {
            _continuousControllerRecoveryStage = "await_upright";
            _continuousControllerRecoveryStageTick = _continuousControllerTick;
        }
        EmitContinuousEvent(
            "continuous_recovery_lifecycle",
            "client_request_method_returned",
            _continuousControllerLastFrame,
            new
            {
                recovery_sequence = _continuousControllerRecoverySequence,
                lifecycle_stage = "client_request_method_returned",
                request_purpose = purpose,
                special_command = command.ToString(),
                special_command_value = (int)command,
                send_method = "RobotInputController.SendSpecialEvent",
                send_method_returned = true,
            });
        _continuousControllerSpecialPurpose = null;
    }

    internal void OnSendSpecialEventFailure(
        RobotInputController input,
        Exception exception)
    {
        if (_continuousControllerRunning &&
            _continuousControllerSpecialInvocationObserved &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            _continuousControllerSpecialInvocationObserved = false;
            StopContinuousController(
                $"continuous_special_send_failed:{exception.GetType().Name}");
        }
    }

    internal bool OnSendEStopTogglePrefix(RobotInputController input)
    {
        InvalidateFreshRoundArmFromUnexpectedRequest(input, "estop");
        if (_singleMotionTrialRunning &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            StopSingleMotionTrial("single_motion_trial_unexpected_estop_request");
            return true;
        }
        if (!_continuousControllerRunning ||
            _continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(input) != _continuousControllerInputPointer)
        {
            return true;
        }
        if (!_continuousControllerEStopAwaitingSend)
        {
            StopContinuousController("continuous_unexpected_estop_request");
            return true;
        }
        try
        {
            if (!RequireOwnedContinuousControl(out var controlReason))
            {
                StopContinuousController(controlReason);
                return false;
            }
            if (_continuousControllerRoundIdentity is null ||
                !TryGetPrivateAiContext(
                    requireActiveRound: true,
                    out var scope,
                    out _,
                    allowOwnedPendingEStop: true) ||
                !_continuousControllerRoundIdentity.Equals(RuntimeIdentity.From(scope)) ||
                !input.hasPendingEStop)
            {
                SuspendContinuousController(
                    "continuous_estop_send_scope_or_pending_unproven",
                    roundInactive: false);
                return false;
            }
            _continuousControllerEStopInvocationObserved = true;
            return true;
        }
        catch (Exception exception)
        {
            StopContinuousController(
                $"continuous_estop_prefix_failed:{exception.GetType().Name}");
            return false;
        }
    }

    internal void OnSendEStopTogglePostfix(RobotInputController input)
    {
        if (!_continuousControllerRunning ||
            !_continuousControllerEStopInvocationObserved ||
            !_continuousControllerEStopAwaitingSend ||
            _continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(input) != _continuousControllerInputPointer)
        {
            return;
        }
        var purpose = _continuousControllerSpecialPurpose;
        _continuousControllerEStopInvocationObserved = false;
        _continuousControllerEStopAwaitingSend = false;
        if (string.Equals(purpose, "fault_estop_toggle_on", StringComparison.Ordinal))
            _continuousControllerRecoveryStage = "await_motor_shutdown";
        else if (string.Equals(purpose, "fault_estop_toggle_off", StringComparison.Ordinal))
            _continuousControllerRecoveryStage = "await_motor_running";
        else
        {
            StopContinuousController("continuous_estop_purpose_invalid_at_postfix");
            return;
        }
        _continuousControllerRecoveryStageTick = _continuousControllerTick;
        EmitContinuousEvent(
            "continuous_recovery_lifecycle",
            "client_request_method_returned",
            _continuousControllerLastFrame,
            new
            {
                recovery_sequence = _continuousControllerRecoverySequence,
                lifecycle_stage = "client_request_method_returned",
                request_purpose = purpose,
                estop_toggle_request = true,
                send_method = "RobotInputController.SendEStopToggle",
                send_method_returned = true,
                semantic_estop_toggle_only = true,
                physical_key_mapping_asserted = false,
                move_request = false,
            });
        _continuousControllerSpecialPurpose = null;
    }

    internal void OnSendEStopToggleFailure(
        RobotInputController input,
        Exception exception)
    {
        if (_continuousControllerRunning &&
            _continuousControllerEStopInvocationObserved &&
            _continuousControllerInputPointer != IntPtr.Zero &&
            NativePointer(input) == _continuousControllerInputPointer)
        {
            _continuousControllerEStopInvocationObserved = false;
            StopContinuousController(
                $"continuous_estop_send_failed:{exception.GetType().Name}");
        }
    }

    internal void OnUnexpectedSingleMotionTrialRequest(
        RobotInputController input,
        string requestKind)
    {
        InvalidateFreshRoundArmFromUnexpectedRequest(input, requestKind);
        if (_singleMotionTrialRunning &&
            _singleMotionTrialInputPointer != IntPtr.Zero &&
            NativePointer(input) == _singleMotionTrialInputPointer)
        {
            StopSingleMotionTrial($"single_motion_trial_unexpected_{requestKind}_request");
        }
    }

    private void InvalidateFreshRoundArmFromVelocityRequest(RobotInputController input)
    {
        if (_freshRoundArm is null || VelocityEquals(input.VelocityCommand, Vector3.zero))
            return;
        InvalidateFreshRoundArmFromUnexpectedRequest(input, "non_neutral_velocity");
    }

    private void InvalidateFreshRoundArmFromMoveRequest(RobotInputController input) =>
        InvalidateFreshRoundArmFromUnexpectedRequest(input, "move");

    private void InvalidateFreshRoundArmFromUnexpectedRequest(
        RobotInputController input,
        string requestKind)
    {
        var arm = _freshRoundArm;
        if (arm is null || arm.InvalidReason is not null)
            return;
        try
        {
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out _) ||
                scope.Input is null || NativePointer(scope.Input) != NativePointer(input) ||
                !TrialSessionIdentity.From(scope).Equals(arm.SessionIdentity))
            {
                _freshRoundArm = arm with
                {
                    InvalidReason = $"unscoped_{requestKind}_request_before_trial",
                };
                return;
            }
            _freshRoundArm = arm with { InvalidReason = $"unexpected_{requestKind}_request_before_trial" };
        }
        catch
        {
            _freshRoundArm = arm with { InvalidReason = "request_scope_probe_failed_before_trial" };
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

    private static bool ValidateSingleMotionTrialContract()
    {
        try
        {
            using var document = JsonDocument.Parse(SingleMotionTrialContract.CanonicalJson);
            var root = document.RootElement;
            if (root.GetProperty("schema").GetString() != SingleMotionTrialContract.Schema ||
                root.GetProperty("authority_scope").GetString() !=
                SingleMotionTrialContract.AuthorityScope ||
                root.GetProperty("authority_caveat").GetString() !=
                SingleMotionTrialContract.AuthorityCaveat ||
                root.GetProperty("unity_fixed_rate_hz").GetInt32() !=
                SingleMotionTrialContract.UnityFixedRateHz ||
                root.GetProperty("trial_rate_hz").GetInt32() !=
                SingleMotionTrialContract.TrialRateHz ||
                root.GetProperty("fixed_substeps_per_tick").GetInt32() !=
                SingleMotionTrialContract.FixedSubstepsPerTrialTick ||
                root.GetProperty("neutral_pre_roll_ticks").GetInt32() !=
                SingleMotionTrialContract.NeutralPreRollTicks ||
                root.GetProperty("action_tick").GetInt32() !=
                SingleMotionTrialContract.ActionTick ||
                root.GetProperty("locomotion_release_tick").GetInt32() !=
                SingleMotionTrialContract.LocomotionReleaseTick ||
                root.GetProperty("duration_ticks").GetInt32() !=
                SingleMotionTrialContract.DurationTrialTicks)
            {
                return false;
            }

            var manifestSelectors = root.GetProperty("selectors").EnumerateArray().ToArray();
            if (manifestSelectors.Length != 12 ||
                manifestSelectors.Length != SingleMotionTrialContract.Selectors.Length ||
                SingleMotionTrialContract.Selectors.Select(value => value.Selector)
                    .Distinct(StringComparer.Ordinal).Count() != manifestSelectors.Length)
            {
                return false;
            }

            for (var index = 0; index < manifestSelectors.Length; index++)
            {
                var manifest = manifestSelectors[index];
                var embedded = SingleMotionTrialContract.Selectors[index];
                var velocity = manifest.GetProperty("velocity_command");
                if (manifest.GetProperty("selector").GetString() != embedded.Selector ||
                    manifest.GetProperty("kind").GetString() != embedded.Kind ||
                    manifest.GetProperty("command_identity").GetString() != embedded.CommandIdentity ||
                    velocity.GetArrayLength() != 3 ||
                    !SameFloatBits(velocity[0].GetSingle(), embedded.Forward) ||
                    !SameFloatBits(velocity[1].GetSingle(), embedded.Strafe) ||
                    !SameFloatBits(velocity[2].GetSingle(), embedded.Yaw))
                {
                    return false;
                }
                var manifestMoveIndex = manifest.GetProperty("move_index");
                if (embedded.MoveIndex is null)
                {
                    if (manifestMoveIndex.ValueKind != JsonValueKind.Null ||
                        !string.Equals(embedded.Kind, "locomotion", StringComparison.Ordinal))
                    {
                        return false;
                    }
                }
                else if (manifestMoveIndex.GetInt32() != embedded.MoveIndex.Value ||
                         !string.Equals(embedded.Kind, "move", StringComparison.Ordinal))
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

    private static bool ValidateContinuousControllerContract()
    {
        try
        {
            using var document = JsonDocument.Parse(
                ContinuousBotControllerContract.CanonicalJson);
            var root = document.RootElement;
            if (root.GetProperty("schema").GetString() !=
                    ContinuousBotControllerContract.Schema ||
                root.GetProperty("authority_scope").GetString() !=
                    ContinuousBotControllerContract.AuthorityScope ||
                root.GetProperty("authority_caveat").GetString() !=
                    ContinuousBotControllerContract.AuthorityCaveat ||
                root.GetProperty("range_angle_provenance").GetString() !=
                    ContinuousBotControllerContract.RangeAngleProvenance ||
                root.GetProperty("facing_yaw_provenance").GetString() !=
                    ContinuousBotControllerContract.FacingYawProvenance ||
                root.GetProperty("attack_selection_provenance").GetString() !=
                    ContinuousBotControllerContract.AttackSelectionProvenance ||
                root.GetProperty("static_impact_timing_provenance").GetString() !=
                    ContinuousBotControllerContract.StaticImpactTimingProvenance ||
                root.GetProperty("round_restart_limitation").GetString() !=
                    ContinuousBotControllerContract.RoundRestartLimitation ||
                root.GetProperty("round_restart_static_evidence").GetString() !=
                    ContinuousBotControllerContract.RoundRestartStaticEvidence ||
                root.GetProperty("recovery_guard_provenance").GetString() !=
                    ContinuousBotControllerContract.RecoveryGuardProvenance ||
                root.GetProperty("fault_estop_provenance").GetString() !=
                    ContinuousBotControllerContract.FaultEStopProvenance ||
                root.GetProperty("dampen_guard").GetString() !=
                    ContinuousBotControllerContract.DampenGuard ||
                root.GetProperty("straighten_guard").GetString() !=
                    ContinuousBotControllerContract.StraightenGuard ||
                root.GetProperty("opponent_runtime_requirement").GetString() !=
                    ContinuousBotControllerContract.OpponentRuntimeRequirement ||
                !SameFloatBits(
                    root.GetProperty("facing_deadband_factor").GetSingle(),
                    ContinuousBotControllerContract.FacingDeadbandFactor) ||
                !SameFloatBits(
                    root.GetProperty("facing_threshold_degrees").GetSingle(),
                    ContinuousBotControllerContract.FacingThresholdDegrees) ||
                !SameFloatBits(
                    root.GetProperty("facing_yaw_ramp_degrees").GetSingle(),
                    ContinuousBotControllerContract.FacingYawRampDegrees) ||
                !SameFloatBits(
                    root.GetProperty("engage_yaw_command").GetSingle(),
                    ContinuousBotControllerContract.EngageYawCommand) ||
                root.GetProperty("unity_fixed_rate_hz").GetInt32() !=
                    ContinuousBotControllerContract.UnityFixedRateHz ||
                root.GetProperty("control_rate_hz").GetInt32() !=
                    ContinuousBotControllerContract.ControlRateHz ||
                root.GetProperty("fixed_substeps_per_control_tick").GetInt32() !=
                    ContinuousBotControllerContract.FixedSubstepsPerControlTick ||
                root.GetProperty("recovery_observation_timeout_ticks").GetInt32() !=
                    ContinuousBotControllerContract.RecoveryObservationTimeoutTicks ||
                root.GetProperty("fault_estop_delay_ticks").GetInt32() !=
                    ContinuousBotControllerContract.FaultEStopDelayTicks ||
                root.GetProperty("fault_estop_hold_ticks").GetInt32() !=
                    ContinuousBotControllerContract.FaultEStopHoldTicks)
            {
                return false;
            }

            var manifestAttacks = root.GetProperty("attacks").EnumerateArray().ToArray();
            if (manifestAttacks.Length != ContinuousBotControllerContract.Attacks.Length ||
                ContinuousBotControllerContract.Attacks.Select(value => value.MoveIndex)
                    .Distinct().Count() != manifestAttacks.Length)
            {
                return false;
            }
            for (var attackIndex = 0; attackIndex < manifestAttacks.Length; attackIndex++)
            {
                var manifest = manifestAttacks[attackIndex];
                var attack = ContinuousBotControllerContract.Attacks[attackIndex];
                if (manifest.GetProperty("move_index").GetInt32() != attack.MoveIndex ||
                    manifest.GetProperty("move_name").GetString() != attack.MoveName ||
                    manifest.GetProperty("display_name").GetString() != attack.DisplayName ||
                    manifest.GetProperty("serialized_asset_sha256").GetString() !=
                        attack.SerializedAssetSha256 ||
                    !SameFloatBits(
                        manifest.GetProperty("maximum_distance_m").GetSingle(),
                        attack.MaximumDistanceMeters) ||
                    !SameFloatBits(
                        manifest.GetProperty("maximum_abs_bearing_degrees").GetSingle(),
                        attack.MaximumAbsBearingDegrees))
                {
                    return false;
                }

                var manifestImpacts = manifest.GetProperty("static_impact_events")
                    .EnumerateArray().ToArray();
                if (manifestImpacts.Length != attack.StaticImpactEvents.Count)
                    return false;
                for (var impactIndex = 0; impactIndex < manifestImpacts.Length; impactIndex++)
                {
                    var impactManifest = manifestImpacts[impactIndex];
                    var impact = attack.StaticImpactEvents[impactIndex];
                    if (!SameFloatBits(
                            impactManifest.GetProperty("impact_time_s").GetSingle(),
                            impact.ImpactTimeSeconds) ||
                        !SameFloatBits(
                            impactManifest.GetProperty("lead_time_s").GetSingle(),
                            impact.LeadTimeSeconds) ||
                        !SameFloatBits(
                            impactManifest.GetProperty("release_time_s").GetSingle(),
                            impact.ReleaseTimeSeconds) ||
                        !SameFloatBits(
                            impactManifest.GetProperty("gain_boost").GetSingle(),
                            impact.GainBoost) ||
                        impactManifest.GetProperty("limb").GetInt32() != impact.Limb)
                    {
                        return false;
                    }
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

    private bool ScopeMatchesSingleMotionTrial(PrivateAiContext scope) =>
        _singleMotionTrialIdentity is not null &&
        _singleMotionTrialIdentity.Equals(RuntimeIdentity.From(scope));

    private bool RequireOwnedSingleMotionTrialControl(out string reason)
    {
        var connectionId = _pipe?.CurrentConnectionId ?? 0;
        if (!_singleMotionTrialRunning || !_singleMotionTrialAuthorizedWhileBackground)
        {
            reason = "single_motion_trial_not_authorized_while_background";
            return false;
        }
        if (_leaseConnectionId == 0 || connectionId != _leaseConnectionId)
        {
            reason = "exclusive_single_motion_trial_lease_not_owned";
            return false;
        }
        if (!RequireBackgroundControl(out reason))
            return false;
        reason = string.Empty;
        return true;
    }

    private bool RequireOwnedContinuousControl(out string reason)
    {
        var connectionId = _pipe?.CurrentConnectionId ?? 0;
        if (!_continuousControllerRunning ||
            !_continuousControllerAuthorizedWhileBackground)
        {
            reason = "continuous_controller_not_authorized_while_background";
            return false;
        }
        if (_leaseConnectionId == 0 || connectionId != _leaseConnectionId)
        {
            reason = "exclusive_continuous_controller_lease_not_owned";
            return false;
        }
        if (!RequireBackgroundControl(out reason))
            return false;
        reason = string.Empty;
        return true;
    }

    private void TryNeutralOwnedContinuousController()
    {
        try
        {
            var input = _continuousControllerInput;
            if (input is not null && _continuousControllerInputPointer != IntPtr.Zero &&
                NativePointer(input) == _continuousControllerInputPointer)
            {
                input.VelocityCommand = Vector3.zero;
            }
        }
        catch
        {
        }
    }

    private void TryCancelExactOwnedContinuousPendingRequests()
    {
        try
        {
            var input = _continuousControllerInput;
            if (input is null || _continuousControllerInputPointer == IntPtr.Zero ||
                NativePointer(input) != _continuousControllerInputPointer)
            {
                return;
            }
            if (_continuousControllerMoveAwaitingSend is not null &&
                input.hasPendingMove &&
                input.pendingMoveIndex == _continuousControllerMoveAwaitingSend.Value)
            {
                input.hasPendingMove = false;
            }
            if (_continuousControllerSpecialAwaitingSend is not null &&
                input.hasPendingSpecial &&
                input.pendingSpecialCommand == (int)_continuousControllerSpecialAwaitingSend.Value)
            {
                input.hasPendingSpecial = false;
            }
            if (_continuousControllerEStopAwaitingSend && input.hasPendingEStop)
                input.hasPendingEStop = false;
        }
        catch
        {
        }
    }

    private void TryNeutralOwnedTrialController()
    {
        try
        {
            var input = _singleMotionTrialInput;
            if (input is not null && _singleMotionTrialInputPointer != IntPtr.Zero &&
                NativePointer(input) == _singleMotionTrialInputPointer)
            {
                input.VelocityCommand = Vector3.zero;
            }
        }
        catch
        {
        }
    }

    private void TryCancelExactOwnedTrialMove()
    {
        try
        {
            var input = _singleMotionTrialInput;
            var moveIndex = _singleMotionTrialMoveAwaitingSend;
            if (input is not null && moveIndex is not null &&
                _singleMotionTrialInputPointer != IntPtr.Zero &&
                NativePointer(input) == _singleMotionTrialInputPointer &&
                input.hasPendingMove && input.pendingMoveIndex == moveIndex.Value)
            {
                input.hasPendingMove = false;
            }
        }
        catch
        {
        }
    }

    private static bool VelocityEquals(Vector3 actual, Vector3 expected) =>
        SameFloatBits(actual.x, expected.x) &&
        SameFloatBits(actual.y, expected.y) &&
        SameFloatBits(actual.z, expected.z);

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

    private static bool TryCreateFreshRoundArm(
        long connectionId,
        string requestId,
        PrivateAiContext scope,
        out FreshRoundArm arm,
        out string reason)
    {
        arm = null!;
        if (connectionId <= 0 || string.IsNullOrEmpty(requestId))
        {
            reason = "fresh_round_request_identity_missing";
            return false;
        }
        var sessionIdentity = TrialSessionIdentity.From(scope);
        if (!sessionIdentity.IsComplete)
        {
            reason = "fresh_round_session_identity_incomplete";
            return false;
        }
        var priorRoundPointer = scope.Round is null ? IntPtr.Zero : NativePointer(scope.Round);
        if (scope.Round is not null && priorRoundPointer == IntPtr.Zero)
        {
            reason = "prior_round_identity_missing";
            return false;
        }
        arm = new FreshRoundArm(
            connectionId,
            requestId,
            sessionIdentity,
            priorRoundPointer,
            scope.FightEpoch,
            scope.RoundNumber,
            InvalidReason: null);
        reason = string.Empty;
        return true;
    }

    private static bool TryCreateTrialRoundIdentity(
        PrivateAiContext scope,
        out TrialRoundIdentity identity,
        out string reason)
    {
        identity = null!;
        if (scope.Round is null || scope.Input is null)
        {
            reason = "active_round_or_input_identity_missing";
            return false;
        }
        var sessionIdentity = TrialSessionIdentity.From(scope);
        var roundPointer = NativePointer(scope.Round);
        var controllerPointer = NativePointer(scope.Input);
        if (!sessionIdentity.IsComplete || roundPointer == IntPtr.Zero ||
            controllerPointer == IntPtr.Zero || scope.RoundNumber < 0 || scope.FightEpoch < 0)
        {
            reason = "active_round_identity_incomplete";
            return false;
        }
        identity = new TrialRoundIdentity(
            sessionIdentity,
            roundPointer,
            controllerPointer,
            scope.FightEpoch,
            scope.RoundNumber);
        reason = string.Empty;
        return true;
    }

    private static string HashTrialRoundIdentity(TrialRoundIdentity identity)
    {
        var canonical = JsonSerializer.Serialize(new
        {
            coordinator_pointer = identity.SessionIdentity.CoordinatorPointer.ToInt64().ToString("x16"),
            network_pointer = identity.SessionIdentity.NetworkPointer.ToInt64().ToString("x16"),
            context_pointer = identity.SessionIdentity.ContextPointer.ToInt64().ToString("x16"),
            round_pointer = identity.RoundPointer.ToInt64().ToString("x16"),
            controller_pointer = identity.ControllerPointer.ToInt64().ToString("x16"),
            identity.FightEpoch,
            identity.RoundNumber,
            identity.SessionIdentity.LocalSlot,
            identity.SessionIdentity.OpponentSlot,
            arena_id_sha256 = HashText(identity.SessionIdentity.ArenaId),
            endpoint_sha256 = HashText(identity.SessionIdentity.Endpoint),
        }, BridgeJson.Options);
        return HashText(canonical);
    }

    private static bool TryCaptureSingleMotionInitialState(
        PrivateAiContext scope,
        string roundIdentitySha256,
        out object state,
        out string stateSha256,
        out string reason)
    {
        state = null!;
        stateSha256 = string.Empty;
        reason = "initial_state_unknown";
        try
        {
            var input = scope.Input;
            var round = scope.Round;
            var fight = scope.Coordinator.Fight;
            var fighters = scope.Coordinator.Fighters;
            if (input is null || round is null || fight is null ||
                fighters is null || fighters.Length != 2 ||
                fighters[0] is null || fighters[1] is null)
            {
                reason = "required_initial_state_object_missing";
                return false;
            }
            if (!input.IsActive || !input.networkInitialized || input.networkIndex != scope.LocalSlot)
            {
                reason = "initial_input_controller_binding_unknown";
                return false;
            }
            if (input.hasPendingMove || input.hasPendingSpecial || input.hasPendingEStop)
            {
                reason = "initial_input_controller_has_pending_command";
                return false;
            }
            var velocity = input.VelocityCommand;
            if (!Finite(velocity) || !VelocityEquals(velocity, Vector3.zero))
            {
                reason = "initial_velocity_not_exact_neutral";
                return false;
            }
            var composer = input.Composer;
            if (composer is null)
            {
                reason = "initial_action_composer_missing";
                return false;
            }
            if (input.IsPunching || input.IsRecovering || composer.IsActionPlaying)
            {
                reason = "initial_action_state_not_neutral";
                return false;
            }

            var roundDuration = round.RoundDuration;
            var timeRemaining = round.TimeRemaining;
            var cleanHits = round.CleanHits;
            var falls = round.Falls;
            var roundsWon = fight.RoundsWon;
            if (!float.IsFinite(roundDuration) || !float.IsFinite(timeRemaining) ||
                roundDuration <= 0f || timeRemaining <= 0f || timeRemaining > roundDuration ||
                cleanHits is null || cleanHits.Length != 2 ||
                falls is null || falls.Length != 2 ||
                roundsWon is null || roundsWon.Length != 2)
            {
                reason = "initial_round_state_incomplete_or_invalid";
                return false;
            }
            if (!round.IsActive || round.IsRedo || round.KnockoutOccurred ||
                cleanHits[0] != 0 || cleanHits[1] != 0 || falls[0] != 0 || falls[1] != 0)
            {
                reason = "initial_round_state_not_fresh";
                return false;
            }

            if (!TryCaptureTrialFighterState(fighters[0], out var fighter0, out reason) ||
                !TryCaptureTrialFighterState(fighters[1], out var fighter1, out reason))
            {
                return false;
            }

            state = new
            {
                schema = "rek.single_motion_initial_state.v1",
                round_identity_sha256 = roundIdentitySha256,
                session_id_sha256 = HashText(scope.Context.ArenaID),
                endpoint_sha256 = HashText($"{scope.Network.serverAddress}:{scope.Network.port}"),
                fight_epoch = scope.FightEpoch,
                round = new
                {
                    number = round.RoundNumber,
                    duration = roundDuration,
                    time_remaining = timeRemaining,
                    active = round.IsActive,
                    redo = round.IsRedo,
                    clean_hits = new[] { cleanHits[0], cleanHits[1] },
                    falls = new[] { falls[0], falls[1] },
                    result = round.Result.ToString(),
                    result_value = (int)round.Result,
                    winner_index = round.WinnerIndex,
                    knockout = round.KnockoutOccurred,
                },
                fight = new
                {
                    format = fight.Format.ToString(),
                    format_value = (int)fight.Format,
                    current_round = fight.CurrentRoundNumber,
                    rounds_won = new[] { roundsWon[0], roundsWon[1] },
                    result = fight.Result.ToString(),
                    result_value = (int)fight.Result,
                    winner_index = fight.WinnerIndex,
                },
                input = new
                {
                    network_index = input.networkIndex,
                    network_initialized = input.networkInitialized,
                    active = input.IsActive,
                    punching = input.IsPunching,
                    recovering = input.IsRecovering,
                    velocity_command_xyz = new[] { velocity.x, velocity.y, velocity.z },
                    pending_move = input.hasPendingMove,
                    pending_special = input.hasPendingSpecial,
                    pending_estop = input.hasPendingEStop,
                    action_playing = composer.IsActionPlaying,
                    action_clip = composer.ActiveActionClip?.name,
                    action_clip_frame = composer.ActionClipFrame,
                    action_clip_fps = composer.ActionClipFps,
                },
                fighter_0 = fighter0,
                fighter_1 = fighter1,
                utc = DateTimeOffset.UtcNow,
                stopwatch_timestamp_ticks = System.Diagnostics.Stopwatch.GetTimestamp(),
                stopwatch_frequency_hz = System.Diagnostics.Stopwatch.Frequency,
                unity_frame = Time.frameCount,
                unity_time = Time.timeAsDouble,
                unity_fixed_time = Time.fixedTimeAsDouble,
            };
            stateSha256 = HashText(JsonSerializer.Serialize(state, BridgeJson.Options));
            reason = string.Empty;
            return true;
        }
        catch (Exception exception)
        {
            reason = $"initial_state_probe_failed:{exception.GetType().Name}";
            state = null!;
            stateSha256 = string.Empty;
            return false;
        }
    }

    private static bool TryCaptureTrialFighterState(
        Robot fighter,
        out object state,
        out string reason)
    {
        state = null!;
        var root = fighter.RootTransform;
        if (root is null)
        {
            reason = "initial_fighter_root_missing";
            return false;
        }
        var position = root.position;
        var rotation = root.rotation;
        var linearVelocity = fighter.RootLinearVelocity;
        var angularVelocity = fighter.RootAngularVelocity;
        if (!Finite(position) || !Finite(rotation) ||
            !Finite(linearVelocity) || !Finite(angularVelocity) ||
            !float.IsFinite(fighter.TiltAngle))
        {
            reason = "initial_fighter_state_nonfinite";
            return false;
        }
        state = new
        {
            visual_only = fighter.IsVisualOnly,
            player_controlled = fighter.IsPlayerControlled,
            falling = fighter.IsFalling,
            fallen = fighter.IsFallen,
            dampened = fighter.IsDampened,
            resetting = fighter.IsResetting,
            motor_shutdown = fighter.IsMotorShutdown,
            tilt_angle = fighter.TiltAngle,
            floor_contact_count = fighter.FloorContactCount,
            root_position_xyz = new[] { position.x, position.y, position.z },
            root_rotation_xyzw = new[] { rotation.x, rotation.y, rotation.z, rotation.w },
            root_linear_velocity_xyz = new[]
            {
                linearVelocity.x,
                linearVelocity.y,
                linearVelocity.z,
            },
            root_angular_velocity_xyz = new[]
            {
                angularVelocity.x,
                angularVelocity.y,
                angularVelocity.z,
            },
        };
        reason = string.Empty;
        return true;
    }

    private static bool Finite(Vector3 value) =>
        float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);

    private static bool Finite(Quaternion value) =>
        float.IsFinite(value.x) && float.IsFinite(value.y) &&
        float.IsFinite(value.z) && float.IsFinite(value.w);

    private static bool RequireBackgroundControl(out string reason)
    {
        if (TryVerifyExplicitIsolatedSession(out _))
        {
            reason = string.Empty;
            return true;
        }
        reason = "autonomous_mutation_requires_verified_isolated_spark_session";
        return false;
    }

    private static bool TryVerifyExplicitIsolatedSession(out string? proof)
    {
        proof = ExplicitIsolatedSession.Proof;
        return ExplicitIsolatedSession.Verified;
    }

    private static IsolationProof DetectExplicitIsolatedSession()
    {
        if (!string.Equals(
                Environment.GetEnvironmentVariable("REK_EVIDENCE_ISOLATED_SESSION"),
                IsolatedSessionMarker,
                StringComparison.Ordinal) ||
            !string.Equals(
                Environment.GetEnvironmentVariable("DISPLAY"),
                ":98",
                StringComparison.Ordinal) ||
            !string.Equals(
                Environment.GetEnvironmentVariable("WINEPREFIX"),
                "/opt/codexrook/wineprefix",
                StringComparison.Ordinal))
        {
            return new IsolationProof(false, null);
        }

        try
        {
            var versionPointer = WineGetVersion();
            var version = versionPointer == IntPtr.Zero
                ? null
                : Marshal.PtrToStringAnsi(versionPointer);
            if (!string.Equals(version, "11.13", StringComparison.Ordinal))
                return new IsolationProof(false, null);
            return new IsolationProof(
                true,
                "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98");
        }
        catch (EntryPointNotFoundException)
        {
            return new IsolationProof(false, null);
        }
        catch (DllNotFoundException)
        {
            return new IsolationProof(false, null);
        }
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
        out string reason,
        bool allowOwnedPendingEStop = false)
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
                if (!ContinuousBotControllerContract.IsInputReadyForControl(
                        input.IsActive,
                        input.networkInitialized,
                        input.hasPendingEStop,
                        allowOwnedPendingEStop))
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
        var isolatedSessionVerified = TryVerifyExplicitIsolatedSession(out var isolationProof);
        var attackZoneAvailability = CaptureAttackZoneAvailability();

        var controlIdentity = new
        {
            lease_held = _leaseConnectionId != 0,
            schedule_running = _scheduleRunning,
            schedule_authorized_while_background = _scheduleAuthorizedWhileBackground,
            schedule_run_id = _scheduleRunId,
            single_motion_trial_running = _singleMotionTrialRunning,
            single_motion_trial_authorized_while_background =
                _singleMotionTrialAuthorizedWhileBackground,
            single_motion_trial_run_id = _singleMotionTrialRunId,
            single_motion_trial_selector = _singleMotionTrialSelector?.Selector,
            continuous_controller_running = _continuousControllerRunning,
            continuous_controller_run_id = _continuousControllerRunId,
            continuous_controller_phase = _continuousControllerPhase,
            continuous_controller_suspend_reason = _continuousControllerSuspendReason,
            continuous_controller_round_identity_sha256 =
                _continuousControllerRoundIdentitySha256,
            attack_zone_trial_running = _attackZoneTrialRunning,
            attack_zone_recovery_only_running = _attackZoneRecoveryOnlyRunning,
            attack_zone_recovery_ready_ticks = _attackZoneRecoveryReadyTicks,
            attack_zone_trial_phase = _attackZonePhase,
            attack_zone_trial_id = _attackZoneTarget?.Request.TrialId,
            fresh_round_request_id = _freshRoundArm?.RequestId,
            fresh_round_invalid_reason = _freshRoundArm?.InvalidReason,
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
                isolated_session_verified = isolatedSessionVerified,
                isolated_session_proof = isolationProof,
                mutation_allowed = isolatedSessionVerified,
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
                rendered_command_marker_schema = RenderedCommandMarkerContract.Schema,
                rendered_command_marker_render_binding = RenderedCommandMarkerContract.RenderBinding,
                rendered_command_marker_count = RenderedCommandMarkerContract.Specs.Length,
                rendered_command_markers_visible = _renderedMarkerStripVisible,
                rendered_command_markers_post_count = _renderedCommandMarkers.Count(value => value),
                single_motion_trial_schema = SingleMotionTrialContract.Schema,
                single_motion_trial_sha256 = _singleMotionTrialSha256,
                single_motion_trial_authority_scope = SingleMotionTrialContract.AuthorityScope,
                single_motion_trial_authority_caveat = SingleMotionTrialContract.AuthorityCaveat,
                single_motion_trial_fixed_substeps_per_tick =
                    SingleMotionTrialContract.FixedSubstepsPerTrialTick,
                single_motion_trial_neutral_pre_roll_ticks =
                    SingleMotionTrialContract.NeutralPreRollTicks,
                single_motion_trial_action_tick = SingleMotionTrialContract.ActionTick,
                single_motion_trial_locomotion_release_tick =
                    SingleMotionTrialContract.LocomotionReleaseTick,
                single_motion_trial_duration_ticks = SingleMotionTrialContract.DurationTrialTicks,
                single_motion_trial_running = _singleMotionTrialRunning,
                single_motion_trial_authorized_while_background =
                    _singleMotionTrialAuthorizedWhileBackground,
                single_motion_trial_run_id = _singleMotionTrialRunId,
                single_motion_trial_selector = _singleMotionTrialSelector?.Selector,
                single_motion_trial_tick = _singleMotionTrialTick,
                single_motion_trial_client_fixed_substep = _singleMotionTrialFixedSubstep,
                single_motion_trial_round_identity_sha256 = _singleMotionTrialRoundIdentitySha256,
                single_motion_trial_initial_state_sha256 = _singleMotionTrialInitialStateSha256,
                single_motion_trial_rounds_consumed = _consumedTrialRounds.Count,
                single_motion_trial_history_capacity = MaxConsumedTrialRounds,
                fresh_round_armed = _freshRoundArm is not null,
                fresh_round_request_id = _freshRoundArm?.RequestId,
                fresh_round_invalid_reason = _freshRoundArm?.InvalidReason,
                trial_isolation_patches_verified = _trialIsolationPatchesVerified,
                continuous_controller_schema = ContinuousBotControllerContract.Schema,
                continuous_controller_sha256 = _continuousControllerSha256,
                continuous_controller_authority_scope =
                    ContinuousBotControllerContract.AuthorityScope,
                continuous_controller_authority_caveat =
                    ContinuousBotControllerContract.AuthorityCaveat,
                continuous_controller_range_angle_provenance =
                    ContinuousBotControllerContract.RangeAngleProvenance,
                continuous_controller_facing_yaw_provenance =
                    ContinuousBotControllerContract.FacingYawProvenance,
                continuous_controller_attack_selection_provenance =
                    ContinuousBotControllerContract.AttackSelectionProvenance,
                continuous_controller_static_impact_timing_provenance =
                    ContinuousBotControllerContract.StaticImpactTimingProvenance,
                continuous_controller_round_restart_limitation =
                    ContinuousBotControllerContract.RoundRestartLimitation,
                continuous_controller_running = _continuousControllerRunning,
                continuous_controller_authorized_while_background =
                    _continuousControllerAuthorizedWhileBackground,
                continuous_controller_run_id = _continuousControllerRunId,
                continuous_controller_phase = _continuousControllerPhase,
                continuous_controller_suspend_reason = _continuousControllerSuspendReason,
                continuous_controller_tick = _continuousControllerTick,
                continuous_controller_round_tick = _continuousControllerRoundTick,
                continuous_controller_round_sequence = _continuousControllerRoundSequence,
                continuous_controller_round_identity_sha256 =
                    _continuousControllerRoundIdentitySha256,
                continuous_controller_next_attack_index = _continuousControllerNextAttackIndex,
                continuous_controller_action_sequence = _continuousControllerActionSequence,
                continuous_controller_recovery_sequence = _continuousControllerRecoverySequence,
                continuous_controller_recovery_stage = _continuousControllerRecoveryStage,
                continuous_controller_recovery_guard_provenance =
                    ContinuousBotControllerContract.RecoveryGuardProvenance,
                continuous_controller_fault_estop_provenance =
                    ContinuousBotControllerContract.FaultEStopProvenance,
                continuous_controller_dampen_guard =
                    ContinuousBotControllerContract.DampenGuard,
                continuous_controller_straighten_guard =
                    ContinuousBotControllerContract.StraightenGuard,
                continuous_controller_opponent_runtime_requirement =
                    ContinuousBotControllerContract.OpponentRuntimeRequirement,
                continuous_controller_straighten_issued =
                    _continuousControllerStraightenIssued,
                continuous_controller_last_measured_state =
                    _continuousControllerLastFrame is null
                        ? null
                        : ContinuousFramePayload(_continuousControllerLastFrame),
                attack_zone_trial_schema = AttackZoneTrialContract.Schema,
                attack_zone_trial_sha256 = _attackZoneContractSha256,
                attack_zone_trial_authority_scope = AttackZoneTrialContract.AuthorityScope,
                attack_zone_trial_authority_caveat = AttackZoneTrialContract.AuthorityCaveat,
                attack_zone_trial_recorder_version =
                    AttackZoneTrialContract.ExpectedRecorderVersion,
                attack_zone_trial_recorder_plugin_sha256 =
                    AttackZoneTrialContract.ExpectedRecorderPluginSha256,
                attack_zone_trial_running = _attackZoneTrialRunning,
                attack_zone_recovery_only_running = _attackZoneRecoveryOnlyRunning,
                attack_zone_recovery_ready_ticks = _attackZoneRecoveryReadyTicks,
                attack_zone_trial_phase = _attackZonePhase,
                attack_zone_trial_last_outcome = _attackZoneLastOutcome,
                attack_zone_trial_id = _attackZoneTarget?.Request.TrialId,
                attack_zone_trial_target = _attackZoneTarget is null
                    ? null
                    : AttackZoneTargetPayload(_attackZoneTarget.Request),
                attack_zone_trial_availability = attackZoneAvailability,
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
        bool exactT800BoneSignature)
    {
        var runtimeBoneSignature = fighter?.BoneNames is { Count: > 0 } boneNames &&
                                   boneNames.All(value => !string.IsNullOrWhiteSpace(value))
            ? HashText(string.Join("\n", boneNames))
            : null;
        return new
        {
            slot = fighter?.Slot,
            semantic_robot_id = fighter?.SemanticRobotId,
            runtime_object_name = fighter?.RuntimeObjectName,
            bone_count = fighter?.BoneNames?.Count,
            bone_names = fighter?.BoneNames,
            runtime_bone_signature_sha256 = runtimeBoneSignature,
            semantic_t800 = semanticT800,
            exact_t800_bone_signature = exactT800BoneSignature,
            semantic_runtime_mismatch = semanticT800 != exactT800BoneSignature,
            semantic_robot_id_used_for_continuous_acceptance = false,
        };
    }

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
    private sealed record IsolationProof(bool Verified, string? Proof);
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
    private sealed record ContinuousRoundMetrics(
        int LocalCleanHits,
        int OpponentCleanHits,
        int LocalFalls,
        int OpponentFalls);
    private sealed record ContinuousFrame(
        RobotInputController Input,
        Robot LocalRobot,
        Robot OpponentRobot,
        string? LocalSemanticRobotId,
        string LocalRuntimeObjectName,
        int LocalBoneCount,
        string LocalBoneSignatureSha256,
        string? OpponentSemanticRobotId,
        string OpponentRuntimeObjectName,
        int OpponentBoneCount,
        string OpponentBoneSignatureSha256,
        string OpponentRuntimeIdentitySha256,
        bool OpponentSemanticRuntimeMismatch,
        string OpponentSemanticRuntimeConsistency,
        Vector3 LocalPosition,
        Quaternion LocalRotation,
        Vector3 LocalForward,
        Vector3 OpponentPosition,
        Quaternion OpponentRotation,
        Vector3 OpponentForward,
        Vector3 LocalLinearVelocity,
        Vector3 LocalAngularVelocity,
        Vector3 OpponentLinearVelocity,
        Vector3 OpponentAngularVelocity,
        PlanarCombatGeometry Geometry,
        bool LocalFalling,
        bool LocalFallen,
        bool LocalDampened,
        bool LocalRecoveryArmed,
        bool LocalGetUpPending,
        bool LocalResetting,
        bool LocalMotorShutdown,
        GetUpOrientation SuggestedGetUpOrientation,
        bool OpponentFalling,
        bool OpponentFallen,
        bool OpponentDampened,
        bool OpponentRecoveryArmed,
        bool OpponentGetUpPending,
        bool OpponentResetting,
        bool OpponentMotorShutdown,
        bool InputPunching,
        bool InputRecovering,
        bool AllowMoveInterrupt,
        bool ComposerActionPlaying,
        bool ComposerBusy,
        MocapClipConfig? ActiveActionClip,
        string? ActiveActionClipName,
        IntPtr ActiveActionClipPointer,
        int CurrentMoveId,
        int ActionClipFrame,
        float ActionClipFps,
        ContinuousRoundMetrics RoundMetrics);
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

    private sealed record TrialSessionIdentity(
        IntPtr CoordinatorPointer,
        IntPtr NetworkPointer,
        IntPtr ContextPointer,
        int LocalSlot,
        int OpponentSlot,
        string ArenaId,
        string Endpoint)
    {
        internal bool IsComplete =>
            CoordinatorPointer != IntPtr.Zero &&
            NetworkPointer != IntPtr.Zero &&
            ContextPointer != IntPtr.Zero &&
            LocalSlot is 0 or 1 &&
            OpponentSlot == 1 - LocalSlot &&
            !string.IsNullOrWhiteSpace(ArenaId) &&
            !string.IsNullOrWhiteSpace(Endpoint);

        internal static TrialSessionIdentity From(PrivateAiContext scope) => new(
            NativePointer(scope.Coordinator),
            NativePointer(scope.Network),
            NativePointer(scope.Context),
            scope.LocalSlot,
            scope.OpponentSlot,
            scope.Context.ArenaID ?? string.Empty,
            $"{scope.Network.serverAddress}:{scope.Network.port}");
    }

    private sealed record TrialRoundIdentity(
        TrialSessionIdentity SessionIdentity,
        IntPtr RoundPointer,
        IntPtr ControllerPointer,
        int FightEpoch,
        int RoundNumber);

    private sealed record FreshRoundArm(
        long ConnectionId,
        string RequestId,
        TrialSessionIdentity SessionIdentity,
        IntPtr PriorRoundPointer,
        int PriorFightEpoch,
        int PriorRoundNumber,
        string? InvalidReason);

    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);

    [DllImport("ntdll.dll", EntryPoint = "wine_get_version", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr WineGetVersion();

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

[HarmonyPatch(typeof(RobotInputController), "SendSpecialEvent")]
internal static class SendSpecialEventControlPatch
{
    [HarmonyPrefix]
    [HarmonyPriority(Priority.First)]
    internal static bool Prefix(RobotInputController __instance)
    {
        return Plugin.Instance?.OnSendSpecialEventPrefix(__instance) ?? true;
    }

    [HarmonyPostfix]
    [HarmonyPriority(Priority.Last)]
    internal static void Postfix(RobotInputController __instance)
    {
        Plugin.Instance?.OnSendSpecialEventPostfix(__instance);
    }

    [HarmonyFinalizer]
    internal static Exception? Finalizer(
        RobotInputController __instance,
        Exception? __exception)
    {
        if (__exception is not null)
            Plugin.Instance?.OnSendSpecialEventFailure(__instance, __exception);
        return __exception;
    }
}

[HarmonyPatch(typeof(RobotInputController), "SendEStopToggle")]
internal static class SendEStopToggleControlPatch
{
    [HarmonyPrefix]
    [HarmonyPriority(Priority.First)]
    internal static bool Prefix(RobotInputController __instance)
    {
        return Plugin.Instance?.OnSendEStopTogglePrefix(__instance) ?? true;
    }

    [HarmonyPostfix]
    [HarmonyPriority(Priority.Last)]
    internal static void Postfix(RobotInputController __instance)
    {
        Plugin.Instance?.OnSendEStopTogglePostfix(__instance);
    }

    [HarmonyFinalizer]
    internal static Exception? Finalizer(
        RobotInputController __instance,
        Exception? __exception)
    {
        if (__exception is not null)
            Plugin.Instance?.OnSendEStopToggleFailure(__instance, __exception);
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

    public void OnGUI()
    {
        Plugin.Instance?.OnUnityGui();
    }
}
