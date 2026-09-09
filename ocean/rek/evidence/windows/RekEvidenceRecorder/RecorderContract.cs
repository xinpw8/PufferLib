using System.Security.Cryptography;
using System.Text;

namespace RekEvidenceRecorder;

internal static class RecorderContract
{
    internal const string Schema = "rek.private_ai.protocol.v7";
    internal const string PluginVersion = "0.7.2";
    internal const string RequiredPairing = "exact_homogeneous_supported_runtime_pair";
    internal const string T800RobotId = "t800";
    internal const string G1RobotId = "g1";
    internal const string ExactT800PairingReason = "exact_t800_vs_t800_pairing_proven";
    internal const string ExactT800PairingWithSemanticMismatchReason =
        "exact_t800_vs_t800_runtime_pairing_proven_semantic_mismatch_recorded";
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

    internal static PairingValidation ValidatePairing(
        int localSlot,
        string? fighter0RobotId,
        string? fighter0RuntimeObjectName,
        IReadOnlyList<string?>? fighter0BoneNames,
        string? fighter1RobotId,
        string? fighter1RuntimeObjectName,
        IReadOnlyList<string?>? fighter1BoneNames)
    {
        var fighter0SemanticT800 = SemanticIs(fighter0RobotId, T800RobotId);
        var fighter1SemanticT800 = SemanticIs(fighter1RobotId, T800RobotId);
        var fighter0SemanticG1 = SemanticIs(fighter0RobotId, G1RobotId);
        var fighter1SemanticG1 = SemanticIs(fighter1RobotId, G1RobotId);
        var fighter0ExactT800 = IsExactT800BoneSignature(fighter0BoneNames);
        var fighter1ExactT800 = IsExactT800BoneSignature(fighter1BoneNames);
        var fighter0ExactG1 = IsExactG1BoneSignature(fighter0BoneNames);
        var fighter1ExactG1 = IsExactG1BoneSignature(fighter1BoneNames);
        var fighter0RuntimeModel = ExactRuntimeModel(fighter0ExactT800, fighter0ExactG1);
        var fighter1RuntimeModel = ExactRuntimeModel(fighter1ExactT800, fighter1ExactG1);

        var localRobotId = SlotValue(localSlot, fighter0RobotId, fighter1RobotId);
        var opponentRobotId = SlotValue(localSlot, fighter1RobotId, fighter0RobotId);
        var localRuntimeObjectName = SlotValue(
            localSlot,
            fighter0RuntimeObjectName,
            fighter1RuntimeObjectName);
        var opponentRuntimeObjectName = SlotValue(
            localSlot,
            fighter1RuntimeObjectName,
            fighter0RuntimeObjectName);
        var localBoneNames = SlotValue(localSlot, fighter0BoneNames, fighter1BoneNames);
        var opponentBoneNames = SlotValue(localSlot, fighter1BoneNames, fighter0BoneNames);
        var localRuntimeModel = SlotValue(localSlot, fighter0RuntimeModel, fighter1RuntimeModel);
        var opponentRuntimeModel = SlotValue(localSlot, fighter1RuntimeModel, fighter0RuntimeModel);

        var reason = localRuntimeModel switch
        {
            T800RobotId when opponentRuntimeModel == T800RobotId => ExactT800PairingReason,
            G1RobotId when opponentRuntimeModel == G1RobotId => ExactG1PairingReason,
            _ => "supported_runtime_pairing_not_proven",
        };
        if (localSlot is < 0 or > 1)
            reason = "local_slot_invalid";
        else if (string.IsNullOrWhiteSpace(localRuntimeObjectName))
            reason = "local_runtime_object_name_unavailable";
        else if (string.IsNullOrWhiteSpace(opponentRuntimeObjectName))
            reason = "opponent_runtime_object_name_unavailable";
        else if (localBoneNames is null)
            reason = "local_bones_unavailable";
        else if (localRuntimeModel is null)
            reason = RuntimeSignatureFailureReason("local", localBoneNames.Count);
        else if (opponentBoneNames is null)
            reason = "opponent_bones_unavailable";
        else if (opponentRuntimeModel is null)
            reason = RuntimeSignatureFailureReason("opponent", opponentBoneNames.Count);
        else if (!string.Equals(localRuntimeModel, opponentRuntimeModel, StringComparison.Ordinal))
            reason = "mixed_supported_runtime_models_rejected";

        var exactT800 = string.Equals(reason, ExactT800PairingReason, StringComparison.Ordinal);
        var exactG1 = string.Equals(reason, ExactG1PairingReason, StringComparison.Ordinal);
        var exactSupported = exactT800 || exactG1;
        var runtimeModel = exactSupported ? localRuntimeModel : null;
        var localSemanticConsistency = SemanticRuntimeConsistency(localRobotId, runtimeModel);
        var opponentSemanticConsistency = SemanticRuntimeConsistency(opponentRobotId, runtimeModel);
        var localSemanticMismatch = IsSemanticMismatch(localSemanticConsistency);
        var opponentSemanticMismatch = IsSemanticMismatch(opponentSemanticConsistency);
        if (exactT800 && (localSemanticMismatch || opponentSemanticMismatch))
            reason = ExactT800PairingWithSemanticMismatchReason;

        return new PairingValidation(
            reason,
            exactSupported,
            runtimeModel,
            exactT800,
            exactG1,
            localSlot,
            fighter0SemanticT800,
            fighter1SemanticT800,
            fighter0SemanticG1,
            fighter1SemanticG1,
            fighter0ExactT800,
            fighter1ExactT800,
            fighter0ExactG1,
            fighter1ExactG1,
            SlotValue(localSlot, fighter0SemanticT800, fighter1SemanticT800),
            SlotValue(localSlot, fighter1SemanticT800, fighter0SemanticT800),
            SlotValue(localSlot, fighter0SemanticG1, fighter1SemanticG1),
            SlotValue(localSlot, fighter1SemanticG1, fighter0SemanticG1),
            localSemanticMismatch,
            opponentSemanticMismatch,
            localSemanticConsistency,
            opponentSemanticConsistency);
    }

    internal static bool IsExactT800BoneSignature(IReadOnlyList<string?>? actual) =>
        ExactNames(actual, T800BoneNames);

    internal static bool IsExactG1BoneSignature(IReadOnlyList<string?>? actual) =>
        ExactNames(actual, G1BoneNames);

    internal static string BoneSignatureSha256(IReadOnlyList<string?> names)
    {
        var joined = string.Join("\n", names.Select(name => name ?? string.Empty));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(joined))).ToLowerInvariant();
    }

    private static bool ExactNames(IReadOnlyList<string?>? actual, IReadOnlyList<string> expected)
    {
        if (actual is null || actual.Count != expected.Count)
            return false;
        for (var index = 0; index < expected.Count; index++)
        {
            if (!string.Equals(actual[index], expected[index], StringComparison.Ordinal))
                return false;
        }
        return true;
    }

    private static bool SemanticIs(string? actual, string expected) =>
        string.Equals(actual, expected, StringComparison.Ordinal);

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
            26 => $"{role}_t800_bone_signature_mismatch",
            30 => $"{role}_g1_bone_signature_mismatch",
            _ => $"{role}_unsupported_bone_count:{boneCount}",
        };

    private static string SemanticRuntimeConsistency(string? semanticRobotId, string? runtimeModel)
    {
        if (runtimeModel is null)
            return "runtime_model_not_proven";
        if (string.IsNullOrWhiteSpace(semanticRobotId))
            return $"semantic_robot_id_unavailable_runtime_{runtimeModel}_exact";
        return string.Equals(semanticRobotId, runtimeModel, StringComparison.Ordinal)
            ? $"semantic_and_runtime_{runtimeModel}_exact"
            : $"semantic_robot_id_mismatch_runtime_{runtimeModel}_exact";
    }

    private static bool IsSemanticMismatch(string classification) =>
        classification.StartsWith("semantic_robot_id_mismatch_", StringComparison.Ordinal);

    private static T? SlotValue<T>(int localSlot, T slot0, T slot1) => localSlot switch
    {
        0 => slot0,
        1 => slot1,
        _ => default,
    };
}

internal readonly record struct PairingValidation(
    string Reason,
    bool ExactSupportedRuntimePairing,
    string? RuntimeModel,
    bool ExactT800VersusT800,
    bool ExactG1VersusG1,
    int LocalSlot,
    bool Fighter0SemanticT800,
    bool Fighter1SemanticT800,
    bool Fighter0SemanticG1,
    bool Fighter1SemanticG1,
    bool Fighter0ExactT800BoneSignature,
    bool Fighter1ExactT800BoneSignature,
    bool Fighter0ExactG1BoneSignature,
    bool Fighter1ExactG1BoneSignature,
    bool LocalSemanticT800,
    bool OpponentSemanticT800,
    bool LocalSemanticG1,
    bool OpponentSemanticG1,
    bool LocalSemanticRuntimeMismatch,
    bool OpponentSemanticRuntimeMismatch,
    string LocalSemanticRuntimeConsistency,
    string OpponentSemanticRuntimeConsistency);
