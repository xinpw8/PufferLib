using System.Buffers.Binary;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using BepInEx;
using BepInEx.Unity.IL2CPP;
using HarmonyLib;
using REKApp;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.Netcode;

namespace RekEvidenceRecorder;

[BepInPlugin(PluginGuid, PluginName, PluginVersion)]
[BepInProcess("REK.exe")]
public sealed class Plugin : BasePlugin
{
    public const string PluginGuid = "openai.rek.evidence.recorder";
    public const string PluginName = "REK Private AI Evidence Recorder";
    public const string PluginVersion = "0.5.1";

    private const string ExpectedGameAssemblySha256 =
        "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
    private const string ExpectedMetadataSha256 =
        "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
    private const string DefaultOutputRoot = @"C:\rekagent\evidence\runtime\rek-private-ai-protocol-v5";
    private static readonly string OutputRoot = ResolveOutputRoot();
    private const int ClientSampleStrideTicks = 10;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    internal static Plugin? Instance { get; private set; }

    private RecorderBehaviour? _behaviour;
    private Harmony? _harmony;
    private bool _harmonyArmed;
    private StreamWriter? _writer;
    private string? _partialPath;
    private string? _finalPath;
    private int _sampleCount;
    private long _clientFixedTick;
    private int _captureErrorCount;
    private ulong _transportInvocationSequence;
    private ulong _fightSnapshotSequence;
    private ulong _rawProtocolSequence;
    private ulong _rawFightStateSequence;
    private ulong _rawScoreSequence;
    private ulong _rawHitSequence;
    private ulong _rawBonePacketSequence;
    private ulong _boneSnapshotSequence;
    private int? _lastTransportInvocationFrame;
    private int? _lastFightSnapshotFrame;
    private double? _lastTransportInvocationTime;
    private double? _lastFightSnapshotTime;
    private readonly Dictionary<string, ulong> _transportInvocationCounts = new(StringComparer.Ordinal);
    private readonly Dictionary<int, BoneSnapshotCursor> _boneSnapshotCursors = new();
    private readonly Dictionary<string, bool?> _harmonyTargetStatus = new(StringComparer.Ordinal);
    private readonly HashSet<string> _loggedHookHits = new(StringComparer.Ordinal);
    private double _lastFlushTime;
    private string _gameAssemblySha256 = string.Empty;
    private string _metadataSha256 = string.Empty;
    private string _pluginSha256 = string.Empty;

    private static string ResolveOutputRoot()
    {
        var configured = Environment.GetEnvironmentVariable("REK_EVIDENCE_OUTPUT_ROOT");
        return string.IsNullOrWhiteSpace(configured)
            ? DefaultOutputRoot
            : Path.GetFullPath(configured);
    }

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

        if (!string.Equals(_gameAssemblySha256, ExpectedGameAssemblySha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(_metadataSha256, ExpectedMetadataSha256, StringComparison.OrdinalIgnoreCase))
        {
            Log.LogError(
                $"Recorder disabled: build hash mismatch. " +
                $"GameAssembly={_gameAssemblySha256} metadata={_metadataSha256}");
            return;
        }

        Instance = this;
        _behaviour = AddComponent<RecorderBehaviour>();
        Log.LogInfo(
            $"Recorder armed for private Sparring Bot 1 scope only. Output root: {OutputRoot}. " +
            "No input, network, authentication, registry, or game-state writes are implemented.");
    }

    public override bool Unload()
    {
        FinishCapture("plugin_unload");
        DisarmHarmony();
        if (_behaviour is not null)
        {
            UnityEngine.Object.Destroy(_behaviour);
            _behaviour = null;
        }
        Instance = null;
        return true;
    }

    internal void OnClientFixedUpdate()
    {
        ScopeSnapshot scope;
        try
        {
            scope = EvaluateScope();
        }
        catch (Exception exception)
        {
            FinishCapture($"scope_error:{exception.GetType().Name}");
            DisarmHarmony();
            return;
        }

        if (!scope.Allowed)
        {
            FinishCapture($"scope_exit:{scope.Reason}");
            DisarmHarmony();
            return;
        }

        try
        {
            if (_writer is null)
            {
                ArmHarmony();
                BeginCapture(scope);
            }

            if (_clientFixedTick % ClientSampleStrideTicks == 0)
            {
                WriteRecord(BuildSample(scope));
                _sampleCount++;
            }
            _clientFixedTick++;

            var now = Time.realtimeSinceStartupAsDouble;
            if (now - _lastFlushTime >= 1.0)
            {
                _writer?.Flush();
                _lastFlushTime = now;
            }
        }
        catch (Exception exception)
        {
            _captureErrorCount++;
            TryWriteError("OnClientFixedUpdate", exception);
            if (_captureErrorCount >= 3)
            {
                FinishCapture("capture_error_limit");
                DisarmHarmony();
            }
        }
    }

    private void ArmHarmony()
    {
        if (_harmonyArmed)
            return;
        _harmony = new Harmony(PluginGuid);
        try
        {
            _harmony.PatchAll(typeof(Plugin).Assembly);
            _harmonyArmed = true;
            AuditHarmonyTargets();
        }
        catch
        {
            _harmony.UnpatchSelf();
            _harmony = null;
            _harmonyArmed = false;
            throw;
        }
    }

    private void AuditHarmonyTargets()
    {
        _harmonyTargetStatus.Clear();
        AuditHarmonyTarget(typeof(RobotInputController), "SendVelocityCommand");
        AuditHarmonyTarget(typeof(RobotInputController), "SendMoveEvent");
        AuditHarmonyTarget(typeof(RobotInputController), "SendSpecialEvent");
        AuditHarmonyTarget(typeof(RobotInputController), "SendEStopToggle");
        AuditHarmonyTarget(typeof(FightCoordinator), "ApplyFightStateSnapshot");
        AuditHarmonyTarget(typeof(FightCoordinator), "OnScoreReceived");
        AuditHarmonyTarget(typeof(FightCoordinator), "OnHitReceived");
        AuditHarmonyTarget(typeof(Robot), "OnBoneMessageReceived");
    }

    private void AuditHarmonyTarget(Type declaringType, string methodName)
    {
        var target = AccessTools.DeclaredMethod(declaringType, methodName);
        bool? owned = null;
        if (target is not null)
        {
            var patchInfo = Harmony.GetPatchInfo(target);
            owned = patchInfo?.Owners.Contains(PluginGuid) ?? false;
        }
        var identity = $"{declaringType.FullName}.{methodName}";
        _harmonyTargetStatus[identity] = owned;
        Log.LogInfo(
            $"Harmony target {identity}: resolved={target is not null}; " +
            $"owned_patch={(owned.HasValue ? owned.Value.ToString() : "unknown")}");
    }

    private void DisarmHarmony()
    {
        if (!_harmonyArmed)
            return;
        _harmony?.UnpatchSelf();
        _harmony = null;
        _harmonyArmed = false;
    }

    private ScopeSnapshot EvaluateScope()
    {
        var coordinator = UnityEngine.Object.FindFirstObjectByType<FightCoordinator>();
        if (coordinator is null)
            return ScopeSnapshot.Denied("no_fight_coordinator");

        var network = UnityEngine.Object.FindFirstObjectByType<NetworkSession>();
        if (network is null)
            return ScopeSnapshot.Denied("no_network_session");
        if (!network.IsConnected)
            return ScopeSnapshot.Denied("network_not_connected");
        if (!network.IsClient || network.IsServer)
            return ScopeSnapshot.Denied("not_client_only");

        var localSlot = coordinator.LocalFighterIndex;
        if (localSlot is < 0 or > 1)
            return ScopeSnapshot.Denied("invalid_local_slot");
        var opponentSlot = 1 - localSlot;

        if (!coordinator.OpponentIsAI)
            return ScopeSnapshot.Denied("opponent_not_ai");
        if (coordinator.SparringBotNumber != 1)
            return ScopeSnapshot.Denied("opponent_not_sparring_bot_1");
        if (!coordinator.SlotIsAI(opponentSlot))
            return ScopeSnapshot.Denied("opponent_slot_not_ai");
        if (coordinator.HumanInSlot(opponentSlot))
            return ScopeSnapshot.Denied("human_in_opponent_slot");

        var slotHasClient = coordinator.slotHasClient;
        if (slotHasClient is null || slotHasClient.Length <= opponentSlot)
            return ScopeSnapshot.Denied("opponent_client_state_unknown");
        if (slotHasClient[opponentSlot])
            return ScopeSnapshot.Denied("opponent_slot_has_client");

        var opponentHumanBit = (coordinator.clientHumanSlotMask & (1 << opponentSlot)) != 0;
        if (opponentHumanBit)
            return ScopeSnapshot.Denied("opponent_human_bit_set");

        var fighters = coordinator.Fighters;
        if (fighters is null || fighters.Length < 2)
            return ScopeSnapshot.Denied("fighters_missing");
        var fighter0 = fighters[0];
        var fighter1 = fighters[1];
        if (fighter0 is null || fighter1 is null)
            return ScopeSnapshot.Denied("fighters_missing");
        if (!fighter0.IsVisualOnly || !fighter1.IsVisualOnly)
            return ScopeSnapshot.Denied("fighters_not_visual_only");

        var round = coordinator.CurrentRound;
        if (round is null || !round.IsActive)
            return ScopeSnapshot.Denied("round_not_active");

        return ScopeSnapshot.AllowedScope(
            coordinator,
            network,
            fighter0,
            fighter1,
            localSlot,
            opponentSlot,
            coordinator.SparringBotNumber,
            ReadServerIdentity(network));
    }

    private void BeginCapture(ScopeSnapshot scope)
    {
        Directory.CreateDirectory(OutputRoot);
        var stamp = DateTimeOffset.UtcNow.ToString("yyyyMMddTHHmmss.fffffffZ");
        var identity = Guid.NewGuid().ToString("N");
        var basename = $"rek-private-ai-raw-snapshot-{stamp}-pid{Environment.ProcessId}-{identity}.jsonl";
        _finalPath = Path.Combine(OutputRoot, basename);
        _partialPath = _finalPath + ".partial";
        _writer = new StreamWriter(
            new FileStream(_partialPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 1 << 20),
            new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            1 << 20)
        {
            AutoFlush = false,
        };
        _sampleCount = 0;
        _clientFixedTick = 0;
        _captureErrorCount = 0;
        _transportInvocationSequence = 0;
        _fightSnapshotSequence = 0;
        _rawProtocolSequence = 0;
        _rawFightStateSequence = 0;
        _rawScoreSequence = 0;
        _rawHitSequence = 0;
        _rawBonePacketSequence = 0;
        _boneSnapshotSequence = 0;
        _lastTransportInvocationFrame = null;
        _lastFightSnapshotFrame = null;
        _lastTransportInvocationTime = null;
        _lastFightSnapshotTime = null;
        _transportInvocationCounts.Clear();
        _boneSnapshotCursors.Clear();
        PrimeBoneSnapshotCursor(scope.Fighter0!);
        PrimeBoneSnapshotCursor(scope.Fighter1!);
        _lastFlushTime = Time.realtimeSinceStartupAsDouble;

        WriteRecord(new Dictionary<string, object?>
        {
            ["event"] = "capture_start",
            ["schema"] = "rek.private_ai.protocol.v5",
            ["utc"] = DateTimeOffset.UtcNow,
            ["pid"] = Environment.ProcessId,
            ["machine"] = Environment.MachineName,
            ["game_root"] = Paths.GameRootPath,
            ["scene"] = SceneManager.GetActiveScene().name,
            ["application_version"] = Application.version,
            ["unity_version"] = Application.unityVersion,
            ["plugin_version"] = PluginVersion,
            ["plugin_sha256"] = _pluginSha256,
            ["game_assembly_sha256"] = _gameAssemblySha256,
            ["global_metadata_sha256"] = _metadataSha256,
            ["sampling_semantics"] = "compact_state_every_10_client_Unity_FixedUpdate_calls_plus_exact_receive-boundary protocol packets and decoded bone snapshots",
            ["client_sample_stride_ticks"] = ClientSampleStrideTicks,
            ["tick_level_claim"] = false,
            ["tick_domain"] = "client_fixed_update",
            ["fixed_delta_time"] = Time.fixedDeltaTime,
            ["server_tick_available"] = false,
            ["server_tick_reason"] = "recovered FightStatePacket and BoneSnapshot layouts expose no server tick field",
            ["bone_wire_protocol"] = new Dictionary<string, object?>
            {
                ["message"] = "REK_Bones",
                ["body_layout"] = "uint8 networkIndex; uint8 boneCount; repeated float32 little-endian worldPosition.xyz and worldRotation.xyzw",
                ["body_size_formula_bytes"] = "2 + 28 * boneCount",
                ["t800_bone_count"] = 30,
                ["t800_body_bytes"] = 842,
                ["intended_send_interval_seconds"] = 0.02,
                ["intended_send_rate_hz"] = 50,
                ["delivery"] = "unreliable",
                ["native_method"] = "REKApp.Robot.ServerSendBones RVA 0x23D7D00",
                ["native_method_source_sha256"] = "4f61233092542b15773e49d8404790a8ed89352d3b656fa41b75bab9c8283ded",
            },
            ["fight_wire_protocol"] = new Dictionary<string, object?>
            {
                ["fight_state"] = "REK_FightState: packed 33-byte little-endian memcpy; reliable; nominal 0.1 s interval",
                ["score"] = "REK_Score: packed 7-byte little-endian memcpy; reliable; emitted per scoring event",
                ["hit"] = "REK_Hit: packed 29-byte little-endian memcpy; unreliable; effects telemetry without fighter identity",
                ["server_tick_available"] = false,
                ["native_source_sha256"] = "9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5",
            },
            ["outbound_request_protocol"] = new Dictionary<string, object?>
            {
                ["observation_boundary"] = "RobotInputController.Send* prefix calls reached from RobotInputController.LateUpdate",
                ["input"] = "REK_Input: uint8-truncated networkIndex plus velocity xyz as three float32 little-endian values; 13 bytes; unreliable",
                ["move"] = "REK_Move: uint8-truncated networkIndex plus uint8-truncated pendingMoveIndex; 2 bytes; reliable",
                ["special_and_estop"] = "invocation-only observations; wire layout and delivery deliberately unclaimed",
                ["server_tick_available"] = false,
                ["server_acceptance_available"] = false,
                ["acknowledgement_observed"] = false,
                ["native_source_sha256"] = "f248df08449e3ff0706ce15ea07e4d58517f2fc9ed3f3143473fa48c4323bc21",
            },
            ["instrumentation_hooks"] = new[]
            {
                "REKApp.RobotInputController.SendVelocityCommand:prefix_exact_REK_Input_request_projection",
                "REKApp.RobotInputController.SendMoveEvent:prefix_exact_REK_Move_request_projection",
                "REKApp.RobotInputController.SendSpecialEvent:prefix_observation",
                "REKApp.RobotInputController.SendEStopToggle:prefix_observation",
                "REKApp.FightCoordinator.ApplyFightStateSnapshot:prefix_raw_packet_copy_and_postfix_applied_state_correlation",
                "REKApp.FightCoordinator.OnScoreReceived:prefix_raw_packet_copy",
                "REKApp.FightCoordinator.OnHitReceived:prefix_raw_packet_copy",
                "REKApp.Robot.OnBoneMessageReceived:prefix_raw_packet_copy_and_postfix_decoded_snapshot_observation",
            },
            ["harmony_target_status"] = new Dictionary<string, bool?>(_harmonyTargetStatus),
            ["authority_semantics"] = "client_observation_of_remote_authoritative_private_AI_mode",
            ["server"] = scope.Server,
            ["scope"] = ScopeRecord(scope),
            ["fighter_0_bones"] = BoneNames(scope.Fighter0!),
            ["fighter_1_bones"] = BoneNames(scope.Fighter1!),
        });
        _writer.Flush();
        Log.LogInfo($"Private AI evidence capture started: {_partialPath}");
    }

    private Dictionary<string, object?> BuildSample(ScopeSnapshot scope)
    {
        var coordinator = scope.Coordinator!;
        var input = coordinator.robotInput;
        var round = coordinator.CurrentRound;
        var fight = coordinator.Fight;

        var record = new Dictionary<string, object?>
        {
            ["event"] = "sample",
            ["sample_index"] = _sampleCount,
            ["client_fixed_tick"] = _clientFixedTick,
            ["utc"] = DateTimeOffset.UtcNow,
            ["unity_frame"] = Time.frameCount,
            ["unity_time"] = Time.timeAsDouble,
            ["unity_fixed_time"] = Time.fixedTimeAsDouble,
            ["unity_unscaled_time"] = Time.unscaledTimeAsDouble,
            ["scene"] = SceneManager.GetActiveScene().name,
            ["fight_epoch"] = coordinator.fightEpoch,
            ["phase"] = coordinator.CurrentPhase.ToString(),
            ["phase_value"] = (int)coordinator.CurrentPhase,
            ["local_fighter_index"] = scope.LocalSlot,
            ["opponent_slot"] = scope.OpponentSlot,
            ["sparring_bot_number"] = scope.SparringBotNumber,
            ["client_ai_difficulty"] = coordinator.clientAiDifficultyLevel,
            ["transport_observation"] = new Dictionary<string, object?>
            {
                ["client_transport_invocation_sequence"] = _transportInvocationSequence,
                ["fight_state_snapshot_sequence"] = _fightSnapshotSequence,
                ["raw_protocol_sequence"] = _rawProtocolSequence,
                ["raw_fight_state_sequence"] = _rawFightStateSequence,
                ["raw_score_sequence"] = _rawScoreSequence,
                ["raw_hit_sequence"] = _rawHitSequence,
                ["last_client_transport_invocation_unity_frame"] = _lastTransportInvocationFrame,
                ["last_fight_snapshot_unity_frame"] = _lastFightSnapshotFrame,
                ["last_client_transport_invocation_realtime_since_startup"] = _lastTransportInvocationTime,
                ["last_fight_snapshot_unscaled_time"] = _lastFightSnapshotTime,
                ["server_tick"] = null,
            },
            ["input"] = InputRecord(input),
            ["round"] = RoundRecord(round),
            ["fight"] = FightRecord(fight),
            ["fighter_0"] = RobotRecord(scope.Fighter0!, includeBones: false),
            ["fighter_1"] = RobotRecord(scope.Fighter1!, includeBones: false),
        };
        return record;
    }

    internal void ObserveVelocityCommandRequest(RobotInputController input)
    {
        if (_writer is null)
            return;
        try
        {
            if (!EvaluateScope().Allowed)
                return;

            var velocity = input.VelocityCommand;
            var xBits = BitConverter.SingleToInt32Bits(velocity.x);
            var yBits = BitConverter.SingleToInt32Bits(velocity.y);
            var zBits = BitConverter.SingleToInt32Bits(velocity.z);
            var body = new byte[13];
            body[0x00] = unchecked((byte)input.networkIndex);
            BinaryPrimitives.WriteInt32LittleEndian(body.AsSpan(0x01, 4), xBits);
            BinaryPrimitives.WriteInt32LittleEndian(body.AsSpan(0x05, 4), yBits);
            BinaryPrimitives.WriteInt32LittleEndian(body.AsSpan(0x09, 4), zBits);

            _transportInvocationSequence++;
            _lastTransportInvocationFrame = Time.frameCount;
            _lastTransportInvocationTime = Time.realtimeSinceStartupAsDouble;
            _transportInvocationCounts.TryGetValue("SendVelocityCommand", out var methodCount);
            methodCount++;
            _transportInvocationCounts["SendVelocityCommand"] = methodCount;
            LogFirstHookHit("REKApp.RobotInputController.SendVelocityCommand");
            WriteRecord(new Dictionary<string, object?>
            {
                ["event"] = "outbound_request_projection",
                ["message"] = "REK_Input",
                ["request_sequence"] = _transportInvocationSequence,
                ["message_request_sequence"] = methodCount,
                ["client_fixed_tick_at_observation"] = _clientFixedTick,
                ["unity_frame"] = _lastTransportInvocationFrame,
                ["unity_realtime_since_startup"] = _lastTransportInvocationTime,
                ["wire_delivery"] = "unreliable",
                ["wire_body_bytes"] = body.Length,
                ["wire_body_sha256"] = HashBytes(body),
                ["wire_body_base64"] = Convert.ToBase64String(body),
                ["network_index_source_int32"] = input.networkIndex,
                ["network_index_wire_uint8"] = body[0x00],
                ["velocity_command_xyz"] = new[] { velocity.x, velocity.y, velocity.z },
                ["velocity_float32_bit_patterns"] = new[]
                {
                    $"0x{unchecked((uint)xBits):x8}",
                    $"0x{unchecked((uint)yBits):x8}",
                    $"0x{unchecked((uint)zBits):x8}",
                },
                ["server_tick"] = null,
                ["server_acceptance"] = null,
                ["ack_observed"] = false,
                ["request_only"] = true,
                ["native_method"] = "REKApp.RobotInputController.SendVelocityCommand RVA 0x226F110",
                ["provenance"] = "exact packed request projection from source fields at REKApp.RobotInputController.SendVelocityCommand prefix",
                ["semantic_limit"] = "prefix observation proves method invocation and exact projected body only; it does not prove send completion, delivery, server acceptance, execution, or policy state",
            });
        }
        catch (Exception exception)
        {
            _captureErrorCount++;
            TryWriteError("ObserveVelocityCommandRequest", exception);
        }
    }

    internal void ObserveMoveRequest(RobotInputController input)
    {
        if (_writer is null)
            return;
        try
        {
            if (!EvaluateScope().Allowed)
                return;

            var body = new byte[2];
            body[0x00] = unchecked((byte)input.networkIndex);
            body[0x01] = unchecked((byte)input.pendingMoveIndex);

            _transportInvocationSequence++;
            _lastTransportInvocationFrame = Time.frameCount;
            _lastTransportInvocationTime = Time.realtimeSinceStartupAsDouble;
            _transportInvocationCounts.TryGetValue("SendMoveEvent", out var methodCount);
            methodCount++;
            _transportInvocationCounts["SendMoveEvent"] = methodCount;
            LogFirstHookHit("REKApp.RobotInputController.SendMoveEvent");
            WriteRecord(new Dictionary<string, object?>
            {
                ["event"] = "outbound_request_projection",
                ["message"] = "REK_Move",
                ["request_sequence"] = _transportInvocationSequence,
                ["message_request_sequence"] = methodCount,
                ["client_fixed_tick_at_observation"] = _clientFixedTick,
                ["unity_frame"] = _lastTransportInvocationFrame,
                ["unity_realtime_since_startup"] = _lastTransportInvocationTime,
                ["wire_delivery"] = "reliable",
                ["wire_body_bytes"] = body.Length,
                ["wire_body_sha256"] = HashBytes(body),
                ["wire_body_base64"] = Convert.ToBase64String(body),
                ["network_index_source_int32"] = input.networkIndex,
                ["network_index_wire_uint8"] = body[0x00],
                ["move_index_source_int32"] = input.pendingMoveIndex,
                ["move_index_wire_uint8"] = body[0x01],
                ["server_tick"] = null,
                ["server_acceptance"] = null,
                ["ack_observed"] = false,
                ["request_only"] = true,
                ["native_method"] = "REKApp.RobotInputController.SendMoveEvent RVA 0x226ECB0",
                ["provenance"] = "exact packed request projection from source fields at REKApp.RobotInputController.SendMoveEvent prefix",
                ["semantic_limit"] = "prefix observation proves method invocation and exact projected body only; it does not prove send completion, delivery, server acceptance, move start, execution, or policy state",
            });
        }
        catch (Exception exception)
        {
            _captureErrorCount++;
            TryWriteError("ObserveMoveRequest", exception);
        }
    }

    internal void ObserveClientTransportInvocation(string methodName)
    {
        if (_writer is null)
            return;
        try
        {
            if (!EvaluateScope().Allowed)
                return;

            _transportInvocationSequence++;
            _lastTransportInvocationFrame = Time.frameCount;
            _lastTransportInvocationTime = Time.realtimeSinceStartupAsDouble;
            _transportInvocationCounts.TryGetValue(methodName, out var methodCount);
            methodCount++;
            _transportInvocationCounts[methodName] = methodCount;
            LogFirstHookHit($"REKApp.RobotInputController.{methodName}");
            WriteRecord(new Dictionary<string, object?>
            {
                ["event"] = "client_transport_method_invoked",
                ["request_sequence"] = _transportInvocationSequence,
                ["method_request_sequence"] = methodCount,
                ["client_fixed_tick_at_observation"] = _clientFixedTick,
                ["unity_frame"] = _lastTransportInvocationFrame,
                ["unity_realtime_since_startup"] = _lastTransportInvocationTime,
                ["method"] = methodName,
                ["message"] = methodName == "SendSpecialEvent" ? "REK_Special" : "REK_EStop",
                ["wire_body_bytes"] = null,
                ["wire_body_sha256"] = null,
                ["wire_body_base64"] = null,
                ["wire_delivery"] = null,
                ["server_tick"] = null,
                ["server_acceptance"] = null,
                ["ack_observed"] = false,
                ["request_only"] = true,
                ["provenance"] = $"REKApp.RobotInputController.{methodName} prefix invocation observation",
                ["semantic_limit"] = "wire layout and delivery are not claimed by this record; invocation does not prove send completion, delivery, acceptance, or execution",
            });
        }
        catch (Exception exception)
        {
            _captureErrorCount++;
            TryWriteError($"ObserveClientTransportInvocation:{methodName}", exception);
        }
    }

    private void LogFirstHookHit(string identity)
    {
        if (_loggedHookHits.Add(identity))
            Log.LogInfo($"Observed first runtime invocation of {identity}");
    }

    internal unsafe ulong ObserveRawFightState(FastBufferReader reader)
    {
        if (_writer is null)
            return 0;
        if (!EvaluateScope().Allowed)
            return 0;

        var body = CopyExactReaderBody(reader, 33, "REK_FightState");
        _rawProtocolSequence++;
        _rawFightStateSequence++;
        var protocolSequence = _rawProtocolSequence;
        LogFirstHookHit("REKApp.FightCoordinator.ApplyFightStateSnapshot:prefix");
        WriteRecord(new Dictionary<string, object?>
        {
            ["event"] = "raw_fight_state_packet",
            ["raw_protocol_sequence"] = protocolSequence,
            ["raw_fight_state_sequence"] = _rawFightStateSequence,
            ["client_fixed_tick_at_observation"] = _clientFixedTick,
            ["unity_frame"] = Time.frameCount,
            ["unity_time"] = Time.timeAsDouble,
            ["unity_unscaled_time"] = Time.unscaledTimeAsDouble,
            ["monotonic_receipt_time"] = Time.realtimeSinceStartupAsDouble,
            ["wire_body_bytes"] = body.Length,
            ["wire_body_sha256"] = HashBytes(body),
            ["wire_body_base64"] = Convert.ToBase64String(body),
            ["decoded"] = new Dictionary<string, object?>
            {
                ["phase"] = body[0x00],
                ["phase_name"] = FightPhaseName(body[0x00]),
                ["round_number"] = body[0x01],
                ["round_active"] = body[0x02],
                ["is_redo"] = body[0x03],
                ["time_remaining"] = ReadFloat32(body, 0x04),
                ["hits_0"] = BinaryPrimitives.ReadInt16LittleEndian(body.AsSpan(0x08, 2)),
                ["hits_1"] = BinaryPrimitives.ReadInt16LittleEndian(body.AsSpan(0x0A, 2)),
                ["knockout_occurred"] = body[0x0C],
                ["round_result"] = body[0x0D],
                ["round_result_name"] = RoundResultName(body[0x0D]),
                ["round_winner"] = unchecked((sbyte)body[0x0E]),
                ["rounds_won_0"] = body[0x0F],
                ["rounds_won_1"] = body[0x10],
                ["fight_result"] = body[0x11],
                ["fight_result_name"] = FightResultName(body[0x11]),
                ["fight_winner"] = unchecked((sbyte)body[0x12]),
                ["format"] = body[0x13],
                ["format_name"] = FightFormatName(body[0x13]),
                ["human_slot_mask"] = body[0x14],
                ["champion_slot"] = unchecked((sbyte)body[0x15]),
                ["fault_mask"] = body[0x16],
                ["fault_stress_0"] = body[0x17],
                ["fault_stress_1"] = body[0x18],
                ["referee_count_mask"] = body[0x19],
                ["referee_count_seconds"] = body[0x1A],
                ["referee_call_sequence"] = body[0x1B],
                ["referee_call_type"] = body[0x1C],
                ["referee_call_name"] = RefereeCallName(body[0x1C]),
                ["referee_call_faller"] = unchecked((sbyte)body[0x1D]),
                ["referee_call_points"] = body[0x1E],
                ["ai_level"] = body[0x1F],
                ["decided_winner_bits"] = body[0x20],
            },
            ["wire_delivery"] = "reliable",
            ["nominal_wire_interval_seconds"] = 0.1,
            ["native_sender"] = "REKApp.FightCoordinator.ServerSendFightState RVA 0x238BFA0",
            ["native_receiver"] = "REKApp.FightCoordinator.ApplyFightStateSnapshot RVA 0x2379E00",
            ["provenance"] = "read-only copy of FastBufferReader at REKApp.FightCoordinator.ApplyFightStateSnapshot prefix",
            ["semantic_limit"] = "packet has no server tick, monotonic server timestamp, command sequence, action acceptance, move identity, policy observation, policy output, or hidden state",
        });
        return protocolSequence;
    }

    internal unsafe void ObserveRawScore(FastBufferReader reader)
    {
        if (_writer is null || !EvaluateScope().Allowed)
            return;

        var body = CopyExactReaderBody(reader, 7, "REK_Score");
        _rawProtocolSequence++;
        _rawScoreSequence++;
        LogFirstHookHit("REKApp.FightCoordinator.OnScoreReceived");
        WriteRecord(new Dictionary<string, object?>
        {
            ["event"] = "raw_score_packet",
            ["raw_protocol_sequence"] = _rawProtocolSequence,
            ["raw_score_sequence"] = _rawScoreSequence,
            ["client_fixed_tick_at_observation"] = _clientFixedTick,
            ["unity_frame"] = Time.frameCount,
            ["unity_time"] = Time.timeAsDouble,
            ["unity_unscaled_time"] = Time.unscaledTimeAsDouble,
            ["monotonic_receipt_time"] = Time.realtimeSinceStartupAsDouble,
            ["wire_body_bytes"] = body.Length,
            ["wire_body_sha256"] = HashBytes(body),
            ["wire_body_base64"] = Convert.ToBase64String(body),
            ["decoded"] = new Dictionary<string, object?>
            {
                ["fighter_index"] = body[0x00],
                ["new_hit_count"] = BinaryPrimitives.ReadInt16LittleEndian(body.AsSpan(0x01, 2)),
                ["points_awarded"] = ReadFloat32(body, 0x03),
            },
            ["wire_delivery"] = "reliable",
            ["native_sender"] = "REKApp.FightCoordinator.OnPointScoredNetwork RVA 0x23867D0",
            ["native_receiver"] = "REKApp.FightCoordinator.OnScoreReceived RVA 0x2387010",
            ["provenance"] = "read-only copy of FastBufferReader at REKApp.FightCoordinator.OnScoreReceived prefix",
            ["semantic_limit"] = "packet has no server tick, timestamp, move identity, hit detail, or action acceptance",
        });
    }

    internal unsafe void ObserveRawHit(FastBufferReader reader)
    {
        if (_writer is null || !EvaluateScope().Allowed)
            return;

        var body = CopyExactReaderBody(reader, 29, "REK_Hit");
        _rawProtocolSequence++;
        _rawHitSequence++;
        LogFirstHookHit("REKApp.FightCoordinator.OnHitReceived");
        WriteRecord(new Dictionary<string, object?>
        {
            ["event"] = "raw_hit_packet",
            ["raw_protocol_sequence"] = _rawProtocolSequence,
            ["raw_hit_sequence"] = _rawHitSequence,
            ["client_fixed_tick_at_observation"] = _clientFixedTick,
            ["unity_frame"] = Time.frameCount,
            ["unity_time"] = Time.timeAsDouble,
            ["unity_unscaled_time"] = Time.unscaledTimeAsDouble,
            ["monotonic_receipt_time"] = Time.realtimeSinceStartupAsDouble,
            ["wire_body_bytes"] = body.Length,
            ["wire_body_sha256"] = HashBytes(body),
            ["wire_body_base64"] = Convert.ToBase64String(body),
            ["decoded"] = new Dictionary<string, object?>
            {
                ["position_xyz"] = new[]
                {
                    ReadFloat32(body, 0x00),
                    ReadFloat32(body, 0x04),
                    ReadFloat32(body, 0x08),
                },
                ["surface_normal_xyz"] = new[]
                {
                    ReadFloat32(body, 0x0C),
                    ReadFloat32(body, 0x10),
                    ReadFloat32(body, 0x14),
                },
                ["relative_speed"] = ReadFloat32(body, 0x18),
                ["is_kick"] = body[0x1C],
            },
            ["wire_delivery"] = "unreliable",
            ["native_sender"] = "REKApp.FightCoordinator.OnHitDetectedNetwork RVA 0x2385500",
            ["native_receiver"] = "REKApp.FightCoordinator.OnHitReceived RVA 0x2385810",
            ["provenance"] = "read-only copy of FastBufferReader at REKApp.FightCoordinator.OnHitReceived prefix",
            ["semantic_limit"] = "effects packet has no fighter identity, scorer, victim, move identity, body zone, impulse, timestamp, sequence, or action acceptance",
        });
    }

    internal void ObserveFightStateSnapshot(FightCoordinator coordinator, ulong rawProtocolSequence)
    {
        if (_writer is null)
            return;
        try
        {
            if (rawProtocolSequence == 0 && !EvaluateScope().Allowed)
                return;
            _fightSnapshotSequence++;
            _lastFightSnapshotFrame = Time.frameCount;
            _lastFightSnapshotTime = Time.unscaledTimeAsDouble;
            WriteRecord(new Dictionary<string, object?>
            {
                ["event"] = "fight_state_snapshot_applied",
                ["fight_state_snapshot_sequence"] = _fightSnapshotSequence,
                ["raw_protocol_sequence"] = rawProtocolSequence == 0 ? null : rawProtocolSequence,
                ["client_fixed_tick_at_observation"] = _clientFixedTick,
                ["unity_frame"] = _lastFightSnapshotFrame,
                ["unity_unscaled_time"] = _lastFightSnapshotTime,
                ["phase"] = coordinator.CurrentPhase.ToString(),
                ["phase_value"] = (int)coordinator.CurrentPhase,
                ["round"] = RoundRecord(coordinator.CurrentRound),
                ["fight"] = FightRecord(coordinator.Fight),
                ["provenance"] = "REKApp.FightCoordinator.ApplyFightStateSnapshot postfix",
            });
        }
        catch (Exception exception)
        {
            _captureErrorCount++;
            TryWriteError("ObserveFightStateSnapshot", exception);
        }
    }

    internal unsafe ulong ObserveRawBoneMessage(FastBufferReader reader)
    {
        if (_writer is null)
            return 0;

        var scope = EvaluateScope();
        if (!scope.Allowed)
            return 0;

        var remaining = reader.Length - reader.Position;
        if (remaining < 2 || remaining > 8192)
            throw new InvalidDataException($"REK_Bones body length {remaining} is outside the audited range.");

        var body = new byte[remaining];
        Marshal.Copy((IntPtr)reader.GetUnsafePtrAtCurrentPosition(), body, 0, body.Length);
        var networkIndex = body[0];
        var boneCount = body[1];
        var expectedLength = 2 + 28 * boneCount;
        if (body.Length != expectedLength)
            throw new InvalidDataException($"REK_Bones body length {body.Length} does not match decoded bone count {boneCount}.");

        Robot? robot = null;
        int? fighterSlot = null;
        if (scope.Fighter0!.networkIndex == networkIndex)
        {
            robot = scope.Fighter0;
            fighterSlot = 0;
        }
        else if (scope.Fighter1!.networkIndex == networkIndex)
        {
            robot = scope.Fighter1;
            fighterSlot = 1;
        }
        if (robot is null)
            throw new InvalidDataException($"REK_Bones network index {networkIndex} does not identify either scoped fighter.");

        var boneNames = BoneNames(robot);
        if (boneNames.Count != boneCount)
            throw new InvalidDataException($"REK_Bones bone count {boneCount} does not match scoped skeleton count {boneNames.Count}.");

        var worldPositions = new List<float>(boneCount * 3);
        var worldRotations = new List<float>(boneCount * 4);
        var offset = 2;
        for (var index = 0; index < boneCount; index++)
        {
            worldPositions.Add(ReadFloat32(body, ref offset));
            worldPositions.Add(ReadFloat32(body, ref offset));
            worldPositions.Add(ReadFloat32(body, ref offset));
            worldRotations.Add(ReadFloat32(body, ref offset));
            worldRotations.Add(ReadFloat32(body, ref offset));
            worldRotations.Add(ReadFloat32(body, ref offset));
            worldRotations.Add(ReadFloat32(body, ref offset));
        }
        if (offset != body.Length)
            throw new InvalidDataException("REK_Bones decoder did not consume the audited body length.");

        _rawBonePacketSequence++;
        WriteRecord(new Dictionary<string, object?>
        {
            ["event"] = "raw_bone_packet",
            ["raw_bone_packet_sequence"] = _rawBonePacketSequence,
            ["client_fixed_tick_at_observation"] = _clientFixedTick,
            ["unity_frame"] = Time.frameCount,
            ["unity_time"] = Time.timeAsDouble,
            ["unity_unscaled_time"] = Time.unscaledTimeAsDouble,
            ["monotonic_receipt_time"] = Time.realtimeSinceStartupAsDouble,
            ["fighter_slot"] = fighterSlot,
            ["network_index"] = networkIndex,
            ["bone_count"] = boneCount,
            ["wire_body_bytes"] = body.Length,
            ["wire_body_sha256"] = Convert.ToHexString(SHA256.HashData(body)).ToLowerInvariant(),
            ["wire_body_base64"] = Convert.ToBase64String(body),
            ["bone_names"] = boneNames,
            ["world_positions_xyz"] = worldPositions,
            ["world_rotations_xyzw"] = worldRotations,
            ["intended_wire_interval_seconds"] = 0.02,
            ["intended_wire_rate_hz"] = 50,
            ["wire_delivery"] = "unreliable",
            ["provenance"] = "read-only copy of FastBufferReader at REKApp.Robot.OnBoneMessageReceived prefix",
            ["semantic_limit"] = "timestamps are client receipt times; packet has no server tick, source timestamp, sequence, move identity, acceptance, acknowledgement, velocity, contact, or controller state",
        });
        return _rawBonePacketSequence;
    }

    internal void ObserveBoneMessageReceived(ulong rawBonePacketSequence)
    {
        if (_writer is null)
            return;
        try
        {
            var scope = EvaluateScope();
            if (!scope.Allowed)
                return;

            ObserveLatestBoneSnapshot(scope.Fighter0!, 0, rawBonePacketSequence);
            ObserveLatestBoneSnapshot(scope.Fighter1!, 1, rawBonePacketSequence);
        }
        catch (Exception exception)
        {
            _captureErrorCount++;
            TryWriteError("ObserveBoneMessageReceived", exception);
        }
    }

    private void PrimeBoneSnapshotCursor(Robot robot)
    {
        var ring = robot.snapshotRing;
        var head = robot.snapshotHead;
        var count = robot.snapshotCount;
        if (ring is null || ring.Length == 0 || count <= 0 || head < 0)
            return;

        var latestIndex = (head + ring.Length - 1) % ring.Length;
        var snapshot = ring[latestIndex];
        if (snapshot is null)
            return;

        _boneSnapshotCursors[robot.networkIndex] = new BoneSnapshotCursor(head, count, snapshot.receivedAt);
    }

    private void ObserveLatestBoneSnapshot(Robot robot, int fighterSlot, ulong rawBonePacketSequence)
    {
        var ring = robot.snapshotRing;
        var head = robot.snapshotHead;
        var count = robot.snapshotCount;
        if (ring is null || ring.Length == 0 || count <= 0 || head < 0)
            return;

        var latestIndex = (head + ring.Length - 1) % ring.Length;
        var snapshot = ring[latestIndex];
        if (snapshot is null)
            return;

        var networkIndex = robot.networkIndex;
        var cursor = new BoneSnapshotCursor(head, count, snapshot.receivedAt);
        if (_boneSnapshotCursors.TryGetValue(networkIndex, out var previous) && previous.Equals(cursor))
            return;
        _boneSnapshotCursors[networkIndex] = cursor;

        _boneSnapshotSequence++;
        LogFirstHookHit("REKApp.Robot.OnBoneMessageReceived");
        WriteRecord(new Dictionary<string, object?>
        {
            ["event"] = "decoded_bone_snapshot",
            ["bone_snapshot_sequence"] = _boneSnapshotSequence,
            ["raw_bone_packet_sequence"] = rawBonePacketSequence == 0 ? null : rawBonePacketSequence,
            ["client_fixed_tick_at_observation"] = _clientFixedTick,
            ["unity_frame"] = Time.frameCount,
            ["unity_time"] = Time.timeAsDouble,
            ["unity_unscaled_time"] = Time.unscaledTimeAsDouble,
            ["fighter_slot"] = fighterSlot,
            ["network_index"] = networkIndex,
            ["snapshot_ring_index"] = latestIndex,
            ["snapshot_ring_head_after_decode"] = head,
            ["snapshot_ring_count_after_decode"] = count,
            ["snapshot_received_at_client_time"] = snapshot.receivedAt,
            ["root_world_position"] = Vector(snapshot.rootWorldPos),
            ["root_world_rotation_xyzw"] = QuaternionRecord(snapshot.rootWorldRot),
            ["child_local_rotations_xyzw"] = QuaternionArray(snapshot.childLocalRot),
            ["bone_names"] = BoneNames(robot),
            ["provenance"] = "REKApp.Robot.OnBoneMessageReceived postfix; latest decoded Robot.BoneSnapshot ring element",
            ["semantic_limit"] = "receivedAt is client Time.time; the recovered packet contains no server tick, joint velocity, torque, controller observation, controller output, or profile identity",
        });
    }

    private static Dictionary<string, object?> ScopeRecord(ScopeSnapshot scope) => new()
    {
        ["allowed"] = true,
        ["network_connected"] = true,
        ["network_is_client"] = true,
        ["network_is_server"] = false,
        ["local_fighter_index"] = scope.LocalSlot,
        ["opponent_slot"] = scope.OpponentSlot,
        ["opponent_is_ai"] = true,
        ["opponent_slot_is_ai"] = true,
        ["human_in_opponent_slot"] = false,
        ["opponent_slot_has_client"] = false,
        ["opponent_human_bit_set"] = false,
        ["fighter_0_visual_only"] = true,
        ["fighter_1_visual_only"] = true,
        ["sparring_bot_number"] = scope.SparringBotNumber,
    };

    private static Dictionary<string, object?> ReadServerIdentity(NetworkSession network)
    {
        var context = GameContext.Instance ?? UnityEngine.Object.FindFirstObjectByType<GameContext>();
        var address = string.IsNullOrWhiteSpace(network.serverAddress) ? null : network.serverAddress;
        var endpoint = address is null ? null : $"{address}:{network.port}";
        var sessionIdentifier = string.IsNullOrWhiteSpace(context?.ArenaID) ? null : context.ArenaID;
        return new Dictionary<string, object?>
        {
            ["endpoint"] = endpoint,
            ["endpoint_provenance"] = "REKApp.NetworkSession.serverAddress+port",
            ["session_identifier_recorded"] = false,
            ["session_identifier_reason"] = "omitted as sensitive session data",
            ["session_id_sha256"] = sessionIdentifier is null ? null : HashText(sessionIdentifier),
            ["session_id_sha256_provenance"] = "SHA-256 of REKApp.GameContext.ArenaID; raw identifier is never written",
            ["protocol"] = "Unity.Netcode.Transports.UTP.UnityTransport",
            ["arena_region"] = string.IsNullOrWhiteSpace(context?.ArenaRegion) ? null : context.ArenaRegion,
            ["arena_scene"] = string.IsNullOrWhiteSpace(context?.ArenaScene) ? null : context.ArenaScene,
            ["server_reported_version"] = null,
        };
    }

    private static Dictionary<string, object?>? InputRecord(RobotInputController? input)
    {
        if (input is null)
            return null;
        var composer = input.Composer;
        var clip = composer?.ActiveActionClip;
        return new Dictionary<string, object?>
        {
            ["network_index"] = input.networkIndex,
            ["network_initialized"] = input.networkInitialized,
            ["active"] = input.IsActive,
            ["punching"] = input.IsPunching,
            ["recovering"] = input.IsRecovering,
            ["velocity_command"] = Vector(input.VelocityCommand),
            ["pending_move"] = input.hasPendingMove,
            ["pending_move_index"] = input.hasPendingMove ? input.pendingMoveIndex : null,
            ["pending_special"] = input.hasPendingSpecial,
            ["pending_special_command"] = input.hasPendingSpecial ? input.pendingSpecialCommand : null,
            ["action_playing"] = composer?.IsActionPlaying,
            ["action_clip"] = clip?.name,
            ["action_clip_frame"] = composer?.ActionClipFrame,
            ["action_clip_fps"] = composer?.ActionClipFps,
            ["action_state_is_server_acceptance_evidence"] = false,
            ["action_state_semantics"] = "local client composer observation only; no server acknowledgement or acceptance is exposed",
        };
    }

    private static Dictionary<string, object?>? RoundRecord(RoundState? round)
    {
        if (round is null)
            return null;
        return new Dictionary<string, object?>
        {
            ["number"] = round.RoundNumber,
            ["duration"] = round.RoundDuration,
            ["time_remaining"] = round.TimeRemaining,
            ["active"] = round.IsActive,
            ["redo"] = round.IsRedo,
            ["clean_hits"] = new int?[] { ReadInt(round.CleanHits, 0), ReadInt(round.CleanHits, 1) },
            ["falls"] = new int?[] { ReadInt(round.Falls, 0), ReadInt(round.Falls, 1) },
            ["result"] = round.Result.ToString(),
            ["result_value"] = (int)round.Result,
            ["winner_index"] = round.WinnerIndex,
            ["knockout"] = round.KnockoutOccurred,
        };
    }

    private static Dictionary<string, object?>? FightRecord(FightState? fight)
    {
        if (fight is null)
            return null;
        return new Dictionary<string, object?>
        {
            ["format"] = fight.Format.ToString(),
            ["format_value"] = (int)fight.Format,
            ["current_round"] = fight.CurrentRoundNumber,
            ["rounds_won"] = new int?[] { ReadInt(fight.RoundsWon, 0), ReadInt(fight.RoundsWon, 1) },
            ["result"] = fight.Result.ToString(),
            ["result_value"] = (int)fight.Result,
            ["winner_index"] = fight.WinnerIndex,
        };
    }

    private static Dictionary<string, object?> RobotRecord(Robot robot, bool includeBones = true)
    {
        var root = robot.RootTransform;
        return new Dictionary<string, object?>
        {
            ["visual_only"] = robot.IsVisualOnly,
            ["player_controlled"] = robot.IsPlayerControlled,
            ["falling"] = robot.IsFalling,
            ["fallen"] = robot.IsFallen,
            ["dampened"] = robot.IsDampened,
            ["resetting"] = robot.IsResetting,
            ["motor_shutdown"] = robot.IsMotorShutdown,
            ["tilt_angle"] = robot.TiltAngle,
            ["floor_contact_count"] = robot.FloorContactCount,
            ["root_position"] = root is null ? null : Vector(root.position),
            ["root_rotation"] = root is null ? null : QuaternionRecord(root.rotation),
            ["root_linear_velocity"] = Vector(robot.RootLinearVelocity),
            ["root_angular_velocity"] = Vector(robot.RootAngularVelocity),
            ["bones"] = includeBones ? BonePoseRecord(robot) : null,
        };
    }

    private static Dictionary<string, object?> BonePoseRecord(Robot robot)
    {
        var positions = new List<float?>();
        var rotations = new List<float?>();
        var localPositions = new List<float?>();
        var localRotations = new List<float?>();
        var bones = robot.boneTransforms;

        if (bones is null)
        {
            return new Dictionary<string, object?>
            {
                ["count"] = null,
                ["world_positions_xyz"] = positions,
                ["world_rotations_xyzw"] = rotations,
                ["local_positions_xyz"] = localPositions,
                ["local_rotations_xyzw"] = localRotations,
            };
        }

        for (var index = 0; index < bones.Length; index++)
        {
            var bone = bones[index];
            if (bone is null)
            {
                AppendMissing(positions, 3);
                AppendMissing(rotations, 4);
                AppendMissing(localPositions, 3);
                AppendMissing(localRotations, 4);
                continue;
            }

            Append(positions, bone.position);
            Append(rotations, bone.rotation);
            Append(localPositions, bone.localPosition);
            Append(localRotations, bone.localRotation);
        }

        return new Dictionary<string, object?>
        {
            ["count"] = bones.Length,
            ["world_positions_xyz"] = positions,
            ["world_rotations_xyzw"] = rotations,
            ["local_positions_xyz"] = localPositions,
            ["local_rotations_xyzw"] = localRotations,
        };
    }

    private static List<string?> BoneNames(Robot robot)
    {
        var names = new List<string?>();
        var bones = robot.boneTransforms;
        if (bones is null)
            return names;
        for (var index = 0; index < bones.Length; index++)
            names.Add(bones[index]?.name);
        return names;
    }

    private static int? ReadInt(Il2CppInterop.Runtime.InteropTypes.Arrays.Il2CppStructArray<int>? values, int index)
    {
        if (values is null || values.Length <= index)
            return null;
        return values[index];
    }

    private static float[] Vector(Vector3 value) => new[] { value.x, value.y, value.z };

    private static float[] QuaternionRecord(Quaternion value) => new[] { value.x, value.y, value.z, value.w };

    private static List<float?> QuaternionArray(
        Il2CppInterop.Runtime.InteropTypes.Arrays.Il2CppStructArray<Quaternion>? values)
    {
        var result = new List<float?>();
        if (values is null)
            return result;
        for (var index = 0; index < values.Length; index++)
            Append(result, values[index]);
        return result;
    }

    private static void Append(List<float?> destination, Vector3 value)
    {
        destination.Add(value.x);
        destination.Add(value.y);
        destination.Add(value.z);
    }

    private static void Append(List<float?> destination, Quaternion value)
    {
        destination.Add(value.x);
        destination.Add(value.y);
        destination.Add(value.z);
        destination.Add(value.w);
    }

    private static void AppendMissing(List<float?> destination, int count)
    {
        for (var index = 0; index < count; index++)
            destination.Add(null);
    }

    private void WriteRecord(Dictionary<string, object?> record)
    {
        if (_writer is null)
            throw new InvalidOperationException("Capture writer is not open.");
        _writer.WriteLine(JsonSerializer.Serialize(record, JsonOptions));
    }

    internal void ObserveHookError(string hook, Exception exception)
    {
        if (_writer is null)
            return;
        _captureErrorCount++;
        TryWriteError($"Harmony:{hook}", exception);
    }

    private void TryWriteError(string stage, Exception exception)
    {
        try
        {
            WriteRecord(new Dictionary<string, object?>
            {
                ["event"] = "capture_error",
                ["utc"] = DateTimeOffset.UtcNow,
                ["unity_frame"] = Time.frameCount,
                ["stage"] = stage,
                ["exception_type"] = exception.GetType().FullName,
                ["exception_message"] = exception.Message,
                ["exception_stack_trace"] = exception.StackTrace,
                ["error_count"] = _captureErrorCount,
            });
        }
        catch
        {
        }
    }

    private void FinishCapture(string reason)
    {
        if (_writer is null)
            return;

        try
        {
            WriteRecord(new Dictionary<string, object?>
            {
                ["event"] = "capture_end",
                ["utc"] = DateTimeOffset.UtcNow,
                ["reason"] = reason,
                ["client_fixed_tick_at_end"] = _clientFixedTick,
                ["unity_fixed_time_at_end"] = Time.fixedTimeAsDouble,
                ["sample_count"] = _sampleCount,
                ["capture_error_count"] = _captureErrorCount,
                ["client_transport_invocation_count"] = _transportInvocationSequence,
                ["client_transport_method_counts"] = new Dictionary<string, ulong>(_transportInvocationCounts),
                ["fight_state_snapshot_count"] = _fightSnapshotSequence,
                ["raw_protocol_packet_count"] = _rawProtocolSequence,
                ["raw_fight_state_packet_count"] = _rawFightStateSequence,
                ["raw_score_packet_count"] = _rawScoreSequence,
                ["raw_hit_packet_count"] = _rawHitSequence,
                ["raw_bone_packet_count"] = _rawBonePacketSequence,
                ["decoded_bone_snapshot_count"] = _boneSnapshotSequence,
            });
            _writer.Flush();
        }
        catch
        {
        }
        finally
        {
            _writer.Dispose();
            _writer = null;
        }

        try
        {
            if (_partialPath is not null && _finalPath is not null)
                File.Move(_partialPath, _finalPath);
            Log.LogInfo($"Private AI evidence capture finalized: {_finalPath}; reason={reason}");
        }
        catch (Exception exception)
        {
            Log.LogWarning(
                $"Capture remains at partial path {_partialPath}; finalize failed: {exception.GetType().Name}");
        }
        finally
        {
            _partialPath = null;
            _finalPath = null;
            _sampleCount = 0;
            _clientFixedTick = 0;
            _captureErrorCount = 0;
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
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value))).ToLowerInvariant();
    }

    private static string HashBytes(byte[] value)
    {
        return Convert.ToHexString(SHA256.HashData(value)).ToLowerInvariant();
    }

    private static unsafe byte[] CopyExactReaderBody(FastBufferReader reader, int expectedLength, string messageName)
    {
        var remaining = reader.Length - reader.Position;
        if (remaining != expectedLength)
        {
            throw new InvalidDataException(
                $"{messageName} body length {remaining} does not match audited length {expectedLength}.");
        }

        var body = new byte[expectedLength];
        Marshal.Copy((IntPtr)reader.GetUnsafePtrAtCurrentPosition(), body, 0, body.Length);
        return body;
    }

    private static float ReadFloat32(byte[] body, int offset)
    {
        var bits = BinaryPrimitives.ReadInt32LittleEndian(body.AsSpan(offset, sizeof(float)));
        return BitConverter.Int32BitsToSingle(bits);
    }

    private static float ReadFloat32(byte[] body, ref int offset)
    {
        var value = ReadFloat32(body, offset);
        offset += sizeof(float);
        return value;
    }

    private static string? FightPhaseName(byte value) => value switch
    {
        0 => "Idle",
        1 => "RoundActive",
        2 => "RoundEnd",
        3 => "BetweenRounds",
        4 => "FightOver",
        5 => "Setup",
        6 => "Sandbox",
        _ => null,
    };

    private static string? RoundResultName(byte value) => value switch
    {
        0 => "InProgress",
        1 => "WonByPoints",
        2 => "WonByKO",
        3 => "Tie",
        4 => "Redo",
        _ => null,
    };

    private static string? FightResultName(byte value) => value switch
    {
        0 => "InProgress",
        1 => "WonByRounds",
        2 => "WonByTKO",
        _ => null,
    };

    private static string? FightFormatName(byte value) => value switch
    {
        0 => "BestOf3",
        1 => "BestOf5",
        _ => null,
    };

    private static string? RefereeCallName(byte value) => value switch
    {
        0 => "Slip",
        1 => "SlipEStop",
        2 => "Knockdown",
        3 => "BeatCount",
        4 => "Knockout",
        5 => "DoubleKnockdown",
        6 => "DoubleKnockout",
        _ => null,
    };

    private sealed class ScopeSnapshot
    {
        private ScopeSnapshot(bool allowed, string reason)
        {
            Allowed = allowed;
            Reason = reason;
        }

        public bool Allowed { get; }
        public string Reason { get; }
        public FightCoordinator? Coordinator { get; private init; }
        public NetworkSession? Network { get; private init; }
        public Robot? Fighter0 { get; private init; }
        public Robot? Fighter1 { get; private init; }
        public int LocalSlot { get; private init; }
        public int OpponentSlot { get; private init; }
        public int SparringBotNumber { get; private init; }
        public Dictionary<string, object?>? Server { get; private init; }

        public static ScopeSnapshot Denied(string reason) => new(false, reason);

        public static ScopeSnapshot AllowedScope(
            FightCoordinator coordinator,
            NetworkSession network,
            Robot fighter0,
            Robot fighter1,
            int localSlot,
            int opponentSlot,
            int sparringBotNumber,
            Dictionary<string, object?> server) => new(true, "allowed")
        {
            Coordinator = coordinator,
            Network = network,
            Fighter0 = fighter0,
            Fighter1 = fighter1,
            LocalSlot = localSlot,
            OpponentSlot = opponentSlot,
            SparringBotNumber = sparringBotNumber,
            Server = server,
        };
    }

    private readonly record struct BoneSnapshotCursor(int Head, int Count, float ReceivedAt);
}

public sealed class RecorderBehaviour : MonoBehaviour
{
    public RecorderBehaviour(IntPtr pointer) : base(pointer)
    {
    }

    public void FixedUpdate()
    {
        Plugin.Instance?.OnClientFixedUpdate();
    }
}

[HarmonyPatch(typeof(RobotInputController), "SendVelocityCommand")]
internal static class SendVelocityCommandObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(RobotInputController __instance)
    {
        try
        {
            Plugin.Instance?.ObserveVelocityCommandRequest(__instance);
        }
        catch
        {
        }
    }
}

[HarmonyPatch(typeof(RobotInputController), "SendMoveEvent")]
internal static class SendMoveEventObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(RobotInputController __instance)
    {
        try
        {
            Plugin.Instance?.ObserveMoveRequest(__instance);
        }
        catch
        {
        }
    }
}

[HarmonyPatch(typeof(RobotInputController), "SendSpecialEvent")]
internal static class SendSpecialEventObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(RobotInputController __instance)
    {
        try
        {
            Plugin.Instance?.ObserveClientTransportInvocation("SendSpecialEvent");
        }
        catch
        {
        }
    }
}

[HarmonyPatch(typeof(RobotInputController), "SendEStopToggle")]
internal static class SendEStopToggleObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(RobotInputController __instance)
    {
        try
        {
            Plugin.Instance?.ObserveClientTransportInvocation("SendEStopToggle");
        }
        catch
        {
        }
    }
}

[HarmonyPatch(typeof(FightCoordinator), "ApplyFightStateSnapshot")]
internal static class FightStateSnapshotObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(FastBufferReader __0, out ulong __state)
    {
        __state = 0;
        try
        {
            __state = Plugin.Instance?.ObserveRawFightState(__0) ?? 0;
        }
        catch (Exception exception)
        {
            Plugin.Instance?.ObserveHookError("REKApp.FightCoordinator.ApplyFightStateSnapshot:prefix", exception);
        }
    }

    [HarmonyPostfix]
    private static void Postfix(FightCoordinator __instance, ulong __state)
    {
        try
        {
            Plugin.Instance?.ObserveFightStateSnapshot(__instance, __state);
        }
        catch (Exception exception)
        {
            Plugin.Instance?.ObserveHookError("REKApp.FightCoordinator.ApplyFightStateSnapshot:postfix", exception);
        }
    }
}

[HarmonyPatch(typeof(FightCoordinator), "OnScoreReceived")]
internal static class ScoreMessageObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(FastBufferReader __1)
    {
        try
        {
            Plugin.Instance?.ObserveRawScore(__1);
        }
        catch (Exception exception)
        {
            Plugin.Instance?.ObserveHookError("REKApp.FightCoordinator.OnScoreReceived:prefix", exception);
        }
    }
}

[HarmonyPatch(typeof(FightCoordinator), "OnHitReceived")]
internal static class HitMessageObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(FastBufferReader __1)
    {
        try
        {
            Plugin.Instance?.ObserveRawHit(__1);
        }
        catch (Exception exception)
        {
            Plugin.Instance?.ObserveHookError("REKApp.FightCoordinator.OnHitReceived:prefix", exception);
        }
    }
}

[HarmonyPatch(typeof(Robot), "OnBoneMessageReceived")]
internal static class BoneMessageObservationPatch
{
    [HarmonyPrefix]
    private static void Prefix(FastBufferReader __1, out ulong __state)
    {
        __state = 0;
        try
        {
            __state = Plugin.Instance?.ObserveRawBoneMessage(__1) ?? 0;
        }
        catch (Exception exception)
        {
            Plugin.Instance?.ObserveHookError("REKApp.Robot.OnBoneMessageReceived:prefix", exception);
        }
    }

    [HarmonyPostfix]
    private static void Postfix(ulong __state)
    {
        try
        {
            Plugin.Instance?.ObserveBoneMessageReceived(__state);
        }
        catch (Exception exception)
        {
            Plugin.Instance?.ObserveHookError("REKApp.Robot.OnBoneMessageReceived:postfix", exception);
        }
    }
}
