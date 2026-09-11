using System.Text.Json;
using System.Text.Json.Serialization;

namespace RekUiBridgeAgent;

internal static class BridgeScheduleContract
{
    internal const string Schema = "rek.client_fixed.command_schedule.v2";
    internal const string ScheduleId = "rek.private_bot1.baseline.v1";
    internal const string ExpectedSha256 =
        "39aaab9c3156e8f4d114daac4d4328257b81230ec8b8a372ad2739d38754ec0d";
    internal const int UnityFixedRateHz = 500;
    internal const int ScheduleRateHz = 50;
    internal const float ExpectedFixedDeltaTime = 0.002f;
    internal const int FixedSubstepsPerScheduleTick = UnityFixedRateHz / ScheduleRateHz;
    internal const int DurationScheduleTicks = 2601;
    internal const int FinalScheduleTick = DurationScheduleTicks - 1;
    internal const string CanonicalJson =
        "{\"duration_ticks\":2601,\"fixed_substeps_per_tick\":10,\"move_commands\":[{\"move_index\":2,\"tick\":900},{\"move_index\":3,\"tick\":1100},{\"move_index\":4,\"tick\":1300},{\"move_index\":5,\"tick\":1500},{\"move_index\":9,\"tick\":1700},{\"move_index\":10,\"tick\":1900},{\"move_index\":2,\"tick\":2100},{\"move_index\":3,\"tick\":2400}],\"schedule_id\":\"rek.private_bot1.baseline.v1\",\"schedule_rate_hz\":50,\"schema\":\"rek.client_fixed.command_schedule.v2\",\"unity_fixed_rate_hz\":500,\"velocity_component_order\":[\"forward\",\"strafe\",\"yaw\"],\"velocity_segments\":[{\"start\":0,\"stop\":50,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":50,\"stop\":150,\"velocity_command\":[1.0,0.0,0.0]},{\"start\":150,\"stop\":200,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":200,\"stop\":300,\"velocity_command\":[-1.0,0.0,0.0]},{\"start\":300,\"stop\":350,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":350,\"stop\":450,\"velocity_command\":[0.0,-1.0,0.0]},{\"start\":450,\"stop\":500,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":500,\"stop\":600,\"velocity_command\":[0.0,1.0,0.0]},{\"start\":600,\"stop\":650,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":650,\"stop\":750,\"velocity_command\":[0.0,0.0,-1.0]},{\"start\":750,\"stop\":800,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":800,\"stop\":900,\"velocity_command\":[0.0,0.0,1.0]},{\"start\":900,\"stop\":2100,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":2100,\"stop\":2300,\"velocity_command\":[1.0,0.0,0.0]},{\"start\":2300,\"stop\":2400,\"velocity_command\":[0.0,0.0,0.0]},{\"start\":2400,\"stop\":2600,\"velocity_command\":[-1.0,0.0,0.0]},{\"start\":2600,\"stop\":2601,\"velocity_command\":[0.0,0.0,0.0]}]}";
}

internal static class BridgePairingContract
{
    internal const string RequiredPairing = "exact_homogeneous_supported_runtime_pair";
    internal const string T800RobotId = "t800";
    internal const string G1RobotId = "g1";
    internal const string ExactPairingReason = "exact_t800_vs_t800_pairing_proven";
    internal const string ExactG1PairingReason =
        "exact_g1_vs_g1_runtime_pairing_proven_semantic_ids_recorded_not_trusted";
    internal const string T800BoneSignatureSha256 =
        "ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab";
    internal const string G1BoneSignatureSha256 =
        "9d18e697233d9578b398fbe849cd59d65cb27a5c2223b2602db66a82a410e987";

    internal static readonly string[] T800BoneNames =
    {
        "LINK_BASE",
        "LINK_HIP_PITCH_L",
        "LINK_HIP_ROLL_L",
        "LINK_HIP_YAW_L",
        "LINK_KNEE_PITCH_L",
        "LINK_ANKLE_PITCH_L",
        "LINK_ANKLE_ROLL_L",
        "LINK_HIP_PITCH_R",
        "LINK_HIP_ROLL_R",
        "LINK_HIP_YAW_R",
        "LINK_KNEE_PITCH_R",
        "LINK_ANKLE_PITCH_R",
        "LINK_ANKLE_ROLL_R",
        "LINK_WAIST_YAW",
        "LINK_SHOULDER_PITCH_L",
        "LINK_SHOULDER_ROLL_L",
        "LINK_SHOULDER_YAW_L",
        "LINK_ELBOW_PITCH_L",
        "LINK_ELBOW_YAW_L",
        "LINK_SHOULDER_PITCH_R",
        "LINK_SHOULDER_ROLL_R",
        "LINK_SHOULDER_YAW_R",
        "LINK_ELBOW_PITCH_R",
        "LINK_ELBOW_YAW_R",
        "LINK_HEAD_PITCH",
        "LINK_HEAD_YAW",
    };

    internal static readonly string[] G1BoneNames =
    {
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    };

    internal static PairingValidation Validate(
        string? localRobotId,
        IReadOnlyList<string?>? localBoneNames,
        string? opponentRobotId,
        IReadOnlyList<string?>? opponentBoneNames)
    {
        var localSemanticT800 = string.Equals(localRobotId, T800RobotId, StringComparison.Ordinal);
        var opponentSemanticT800 = string.Equals(opponentRobotId, T800RobotId, StringComparison.Ordinal);
        var localSemanticG1 = string.Equals(localRobotId, G1RobotId, StringComparison.Ordinal);
        var opponentSemanticG1 = string.Equals(opponentRobotId, G1RobotId, StringComparison.Ordinal);
        var localExactT800BoneSignature = IsExactT800BoneSignature(localBoneNames);
        var opponentExactT800BoneSignature = IsExactT800BoneSignature(opponentBoneNames);
        var localExactG1BoneSignature = IsExactG1BoneSignature(localBoneNames);
        var opponentExactG1BoneSignature = IsExactG1BoneSignature(opponentBoneNames);
        var localRuntimeModel = ExactRuntimeModel(
            localExactT800BoneSignature,
            localExactG1BoneSignature);
        var opponentRuntimeModel = ExactRuntimeModel(
            opponentExactT800BoneSignature,
            opponentExactG1BoneSignature);

        var reason = localRuntimeModel switch
        {
            T800RobotId when opponentRuntimeModel == T800RobotId => ExactPairingReason,
            G1RobotId when opponentRuntimeModel == G1RobotId => ExactG1PairingReason,
            _ => "supported_runtime_pairing_not_proven",
        };
        if (localBoneNames is null)
            reason = "local_fighter_bones_unavailable";
        else if (localRuntimeModel is null)
            reason = RuntimeSignatureFailureReason("local", localBoneNames.Count);
        else if (opponentBoneNames is null)
            reason = "opponent_fighter_bones_unavailable";
        else if (opponentRuntimeModel is null)
            reason = RuntimeSignatureFailureReason("opponent", opponentBoneNames.Count);
        else if (!string.Equals(localRuntimeModel, opponentRuntimeModel, StringComparison.Ordinal))
            reason = "mixed_supported_runtime_models_rejected";

        var exactT800VersusT800 = string.Equals(
            reason,
            ExactPairingReason,
            StringComparison.Ordinal);
        var exactG1VersusG1 = string.Equals(
            reason,
            ExactG1PairingReason,
            StringComparison.Ordinal);
        var exactSupportedRuntimePairing = exactT800VersusT800 || exactG1VersusG1;
        var runtimeModel = exactSupportedRuntimePairing ? localRuntimeModel : null;

        return new PairingValidation(
            reason,
            exactSupportedRuntimePairing,
            runtimeModel,
            exactT800VersusT800,
            exactG1VersusG1,
            localSemanticT800,
            opponentSemanticT800,
            localSemanticG1,
            opponentSemanticG1,
            localExactT800BoneSignature,
            opponentExactT800BoneSignature,
            localExactG1BoneSignature,
            opponentExactG1BoneSignature,
            ClassifySemanticRuntimeConsistency(localRobotId, runtimeModel),
            ClassifySemanticRuntimeConsistency(opponentRobotId, runtimeModel));
    }

    internal static bool IsExactT800BoneSignature(IReadOnlyList<string?>? actual)
    {
        if (actual is null || actual.Count != T800BoneNames.Length)
            return false;
        for (var index = 0; index < T800BoneNames.Length; index++)
        {
            if (!string.Equals(actual[index], T800BoneNames[index], StringComparison.Ordinal))
                return false;
        }
        return true;
    }

    internal static bool IsExactG1BoneSignature(IReadOnlyList<string?>? actual)
    {
        if (actual is null || actual.Count != G1BoneNames.Length)
            return false;
        for (var index = 0; index < G1BoneNames.Length; index++)
        {
            if (!string.Equals(actual[index], G1BoneNames[index], StringComparison.Ordinal))
                return false;
        }
        return true;
    }

    private static string? ExactRuntimeModel(bool exactT800, bool exactG1) =>
        (exactT800, exactG1) switch
        {
            (true, false) => T800RobotId,
            (false, true) => G1RobotId,
            _ => null,
        };

    private static string RuntimeSignatureFailureReason(string role, int boneCount) =>
        boneCount switch
        {
            26 => $"{role}_fighter_t800_bone_signature_mismatch",
            30 => $"{role}_fighter_g1_bone_signature_mismatch",
            _ => $"{role}_fighter_unsupported_bone_count:{boneCount}",
        };

    private static string ClassifySemanticRuntimeConsistency(
        string? semanticRobotId,
        string? runtimeModel)
    {
        if (runtimeModel is null)
            return "runtime_model_not_proven";
        if (string.IsNullOrWhiteSpace(semanticRobotId))
            return $"semantic_robot_id_unavailable_runtime_{runtimeModel}_exact";
        return string.Equals(semanticRobotId, runtimeModel, StringComparison.Ordinal)
            ? $"semantic_and_runtime_{runtimeModel}_exact"
            : $"semantic_robot_id_mismatch_runtime_{runtimeModel}_exact";
    }
}

internal sealed record PairingValidation(
    string Reason,
    bool ExactSupportedRuntimePairing,
    string? RuntimeModel,
    bool ExactT800VersusT800,
    bool ExactG1VersusG1,
    bool LocalSemanticT800,
    bool OpponentSemanticT800,
    bool LocalSemanticG1,
    bool OpponentSemanticG1,
    bool LocalExactT800BoneSignature,
    bool OpponentExactT800BoneSignature,
    bool LocalExactG1BoneSignature,
    bool OpponentExactG1BoneSignature,
    string LocalSemanticRuntimeConsistency,
    string OpponentSemanticRuntimeConsistency);

internal enum BridgeKey
{
    Left,
    Right,
    Up,
    Down,
    Enter,
    Escape,
    Space,
}

internal enum BridgeCommand
{
    AcquireExclusiveControl,
    ReleaseExclusiveControl,
    ConfirmLoggedIn,
    NavigateFreePlay,
    EnterSolo,
    StartRound,
    ExitUnexpectedPrivateAiSession,
    ExitLostPrivateSession,
    StartMeasuredSchedule,
    StopMeasuredSchedule,
    StartSingleMotionTrial,
    StartContinuousBotController,
    StopContinuousBotController,
    StartAttackZoneTrial,
    StopAttackZoneTrial,
    StartG1HeldInputSchedule,
    StopG1HeldInputSchedule,
}

internal enum RequestKind
{
    GetState,
    Input,
    Command,
}

internal sealed record BridgeRequest(
    long ConnectionId,
    RequestKind Kind,
    string RequestId,
    BridgeKey? Key,
    BridgeCommand? Command,
    string? Selector,
    AttackZoneTrialTarget? AttackZoneTarget = null);

internal sealed record OutboundMessage(long ConnectionId, object Payload);

internal static class BridgeJson
{
    internal static readonly JsonSerializerOptions Options = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        WriteIndented = false,
    };
}

internal static class BridgeProtocol
{
    internal const int MaxLineBytes = 4096;
    private const int MaxRequestIdLength = 64;

    internal static bool TryParse(
        ReadOnlySpan<byte> utf8,
        long connectionId,
        out BridgeRequest? request,
        out string error,
        out string? requestId)
    {
        request = null;
        error = string.Empty;
        requestId = null;

        if (utf8.IsEmpty || utf8.Length > MaxLineBytes)
        {
            error = "invalid_line_length";
            return false;
        }

        try
        {
            using var document = JsonDocument.Parse(utf8.ToArray());
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                error = "request_must_be_object";
                return false;
            }

            string? type = null;
            string? key = null;
            string? command = null;
            string? selector = null;
            AttackZoneTrialTarget? attackZoneTarget = null;
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (!names.Add(property.Name))
                {
                    error = "duplicate_property";
                    return false;
                }

                if (property.Name is not ("type" or "request_id" or "key" or "command" or "selector" or "target"))
                {
                    error = "unknown_property";
                    return false;
                }

                if (property.Name == "target")
                {
                    if (!AttackZoneTrialContract.TryParseTarget(
                            property.Value,
                            out attackZoneTarget,
                            out error))
                    {
                        return false;
                    }
                    continue;
                }

                if (property.Value.ValueKind != JsonValueKind.String)
                {
                    error = "properties_must_be_strings_except_target";
                    return false;
                }

                switch (property.Name)
                {
                    case "type":
                        type = property.Value.GetString();
                        break;
                    case "request_id":
                        requestId = property.Value.GetString();
                        break;
                    case "key":
                        key = property.Value.GetString();
                        break;
                    case "command":
                        command = property.Value.GetString();
                        break;
                    case "selector":
                        selector = property.Value.GetString();
                        break;
                }
            }

            if (!ValidRequestId(requestId))
            {
                requestId = null;
                error = "invalid_request_id";
                return false;
            }

            if (string.Equals(type, "get_state", StringComparison.Ordinal))
            {
                if (key is not null || command is not null || selector is not null ||
                    attackZoneTarget is not null || names.Count != 2)
                {
                    error = "invalid_get_state_shape";
                    return false;
                }
                request = new BridgeRequest(connectionId, RequestKind.GetState, requestId!, null, null, null);
                return true;
            }

            if (string.Equals(type, "input", StringComparison.Ordinal))
            {
                if (key is null || command is not null || selector is not null ||
                    attackZoneTarget is not null || names.Count != 3 ||
                    !Enum.GetNames<BridgeKey>().Contains(key, StringComparer.Ordinal))
                {
                    error = "invalid_or_disallowed_key";
                    return false;
                }
                var parsedKey = Enum.Parse<BridgeKey>(key, ignoreCase: false);
                request = new BridgeRequest(connectionId, RequestKind.Input, requestId!, parsedKey, null, null);
                return true;
            }

            if (string.Equals(type, "command", StringComparison.Ordinal))
            {
                if (command is null || key is not null ||
                    !Enum.GetNames<BridgeCommand>().Contains(command, StringComparer.Ordinal))
                {
                    error = "invalid_or_disallowed_command";
                    return false;
                }
                var parsedCommand = Enum.Parse<BridgeCommand>(command, ignoreCase: false);
                var isSingleMotionTrial = parsedCommand == BridgeCommand.StartSingleMotionTrial;
                var isAttackZoneTrial = parsedCommand == BridgeCommand.StartAttackZoneTrial;
                if (isSingleMotionTrial)
                {
                    if (attackZoneTarget is not null || names.Count != 4 ||
                        !SingleMotionTrialContract.TryGet(selector, out _))
                    {
                        error = "invalid_or_disallowed_selector";
                        return false;
                    }
                }
                else if (isAttackZoneTrial)
                {
                    if (selector is not null || attackZoneTarget is null || names.Count != 4 ||
                        !AttackZoneTrialContract.TryValidateTarget(
                            attackZoneTarget,
                            out _,
                            out error))
                    {
                        if (string.IsNullOrEmpty(error))
                            error = "invalid_attack_zone_target";
                        return false;
                    }
                }
                else if (selector is not null || attackZoneTarget is not null || names.Count != 3)
                {
                    error = "invalid_command_shape";
                    return false;
                }
                request = new BridgeRequest(
                    connectionId,
                    RequestKind.Command,
                    requestId!,
                    null,
                    parsedCommand,
                    selector,
                    attackZoneTarget);
                return true;
            }

            error = "invalid_request_type";
            return false;
        }
        catch (JsonException)
        {
            error = "invalid_json";
            return false;
        }
    }

    private static bool ValidRequestId(string? value)
    {
        if (string.IsNullOrEmpty(value) || value.Length > MaxRequestIdLength)
            return false;
        foreach (var character in value)
        {
            var asciiLetterOrDigit = character is >= 'a' and <= 'z' or >= 'A' and <= 'Z' or >= '0' and <= '9';
            if (!asciiLetterOrDigit && character is not ('.' or '_' or ':' or '-'))
                return false;
        }
        return true;
    }
}
