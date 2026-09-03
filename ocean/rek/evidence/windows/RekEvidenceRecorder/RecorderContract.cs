using System.Security.Cryptography;
using System.Text;

namespace RekEvidenceRecorder;

internal static class RecorderContract
{
    internal const string Schema = "rek.private_ai.protocol.v6";
    internal const string PluginVersion = "0.6.1";
    internal const string RequiredPairing = "t800_vs_t800";
    internal const string RequiredRobotId = "t800";
    internal const string ExactPairingReason = "exact_t800_vs_t800_pairing_proven";
    internal const string ExactPairingWithOpponentSemanticMismatchReason =
        "exact_t800_vs_t800_runtime_pairing_proven_opponent_semantic_mismatch_recorded";
    internal const string T800BoneSignatureSha256 =
        "ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab";

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

    internal static PairingValidation ValidatePairing(
        int localSlot,
        string? fighter0RobotId,
        string? fighter0RuntimeObjectName,
        IReadOnlyList<string?>? fighter0BoneNames,
        string? fighter1RobotId,
        string? fighter1RuntimeObjectName,
        IReadOnlyList<string?>? fighter1BoneNames)
    {
        var fighter0SemanticT800 = string.Equals(
            fighter0RobotId, RequiredRobotId, StringComparison.Ordinal);
        var fighter1SemanticT800 = string.Equals(
            fighter1RobotId, RequiredRobotId, StringComparison.Ordinal);
        var fighter0ExactT800BoneSignature = IsExactT800BoneSignature(fighter0BoneNames);
        var fighter1ExactT800BoneSignature = IsExactT800BoneSignature(fighter1BoneNames);

        var localSemanticT800 = localSlot switch
        {
            0 => fighter0SemanticT800,
            1 => fighter1SemanticT800,
            _ => false,
        };
        var opponentSemanticT800 = localSlot switch
        {
            0 => fighter1SemanticT800,
            1 => fighter0SemanticT800,
            _ => false,
        };
        var localRuntimeObjectName = localSlot switch
        {
            0 => fighter0RuntimeObjectName,
            1 => fighter1RuntimeObjectName,
            _ => null,
        };
        var opponentRuntimeObjectName = localSlot switch
        {
            0 => fighter1RuntimeObjectName,
            1 => fighter0RuntimeObjectName,
            _ => null,
        };
        var localBoneNames = localSlot switch
        {
            0 => fighter0BoneNames,
            1 => fighter1BoneNames,
            _ => null,
        };
        var opponentBoneNames = localSlot switch
        {
            0 => fighter1BoneNames,
            1 => fighter0BoneNames,
            _ => null,
        };
        var localExactT800BoneSignature = localSlot switch
        {
            0 => fighter0ExactT800BoneSignature,
            1 => fighter1ExactT800BoneSignature,
            _ => false,
        };
        var opponentExactT800BoneSignature = localSlot switch
        {
            0 => fighter1ExactT800BoneSignature,
            1 => fighter0ExactT800BoneSignature,
            _ => false,
        };

        var reason = opponentSemanticT800
            ? ExactPairingReason
            : ExactPairingWithOpponentSemanticMismatchReason;
        if (localSlot is < 0 or > 1)
            reason = "local_slot_invalid";
        else if (string.IsNullOrWhiteSpace(localRuntimeObjectName))
            reason = "local_runtime_object_name_unavailable";
        else if (string.IsNullOrWhiteSpace(opponentRuntimeObjectName))
            reason = "opponent_runtime_object_name_unavailable";
        else if ((localSlot == 0 ? fighter0RobotId : fighter1RobotId) is null)
            reason = "local_robot_id_unavailable";
        else if (!localSemanticT800)
            reason = "local_robot_id_not_t800";
        else if (localBoneNames is null)
            reason = "local_bones_unavailable";
        else if (localBoneNames.Count != T800BoneNames.Length)
            reason = "local_bone_count_not_26";
        else if (!localExactT800BoneSignature)
            reason = "local_t800_bone_signature_mismatch";
        else if (opponentBoneNames is null)
            reason = "opponent_bones_unavailable";
        else if (opponentBoneNames.Count != T800BoneNames.Length)
            reason = "opponent_bone_count_not_26";
        else if (!opponentExactT800BoneSignature)
            reason = "opponent_t800_bone_signature_mismatch";

        var exactT800VersusT800 =
            string.Equals(reason, ExactPairingReason, StringComparison.Ordinal) ||
            string.Equals(
                reason,
                ExactPairingWithOpponentSemanticMismatchReason,
                StringComparison.Ordinal);
        var opponentSemanticRuntimeConsistency = exactT800VersusT800 switch
        {
            false => "pairing_not_proven",
            true when (localSlot == 0 ? fighter1RobotId : fighter0RobotId) is null =>
                "opponent_semantic_identity_unavailable_runtime_t800_exact",
            true when opponentSemanticT800 => "opponent_semantic_and_runtime_t800_exact",
            _ => "opponent_semantic_non_t800_runtime_t800_exact",
        };

        return new PairingValidation(
            reason,
            exactT800VersusT800,
            localSlot,
            fighter0SemanticT800,
            fighter1SemanticT800,
            fighter0ExactT800BoneSignature,
            fighter1ExactT800BoneSignature,
            localSemanticT800,
            opponentSemanticT800,
            opponentExactT800BoneSignature,
            exactT800VersusT800 && !opponentSemanticT800,
            opponentSemanticRuntimeConsistency);
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

    internal static string BoneSignatureSha256(IReadOnlyList<string?> names)
    {
        var joined = string.Join("\n", names.Select(name => name ?? string.Empty));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(joined))).ToLowerInvariant();
    }
}

internal readonly record struct PairingValidation(
    string Reason,
    bool ExactT800VersusT800,
    int LocalSlot,
    bool Fighter0SemanticT800,
    bool Fighter1SemanticT800,
    bool Fighter0ExactT800BoneSignature,
    bool Fighter1ExactT800BoneSignature,
    bool LocalSemanticT800,
    bool OpponentSemanticT800,
    bool OpponentExactT800BoneSignature,
    bool OpponentSemanticRuntimeMismatch,
    string OpponentSemanticRuntimeConsistency);
