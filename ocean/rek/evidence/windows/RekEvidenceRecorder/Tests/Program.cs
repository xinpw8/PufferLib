using RekEvidenceRecorder;

var failures = new List<string>();

void Expect(string name, bool condition)
{
    Console.WriteLine($"{name}: {(condition ? "PASS" : "FAIL")}");
    if (!condition)
        failures.Add(name);
}

var exactBones = RecorderContract.T800BoneNames
    .Select(name => (string?)name)
    .ToArray();
Expect("schema_v6", RecorderContract.Schema == "rek.private_ai.protocol.v6");
Expect("plugin_version_0_6_1", RecorderContract.PluginVersion == "0.6.1");
Expect("t800_bone_count_26", exactBones.Length == 26);
Expect(
    "t800_signature_hash",
    RecorderContract.BoneSignatureSha256(exactBones) ==
    RecorderContract.T800BoneSignatureSha256);

var exact = RecorderContract.ValidatePairing(
    0, "t800", "T800_Local", exactBones, "t800", "T800_Opponent", exactBones);
Expect("exact_t800_pairing", exact.ExactT800VersusT800);
Expect("exact_t800_reason", exact.Reason == RecorderContract.ExactPairingReason);
Expect("exact_local_semantic", exact.LocalSemanticT800);
Expect("exact_opponent_runtime", exact.OpponentExactT800BoneSignature);
Expect("exact_no_semantic_mismatch", !exact.OpponentSemanticRuntimeMismatch);

var staleOpponentSemantic = RecorderContract.ValidatePairing(
    0, "t800", "T800_Local", exactBones, "g1", "T800_Opponent", exactBones);
Expect("stale_opponent_semantic_allowed", staleOpponentSemantic.ExactT800VersusT800);
Expect(
    "stale_opponent_semantic_reason",
    staleOpponentSemantic.Reason ==
    RecorderContract.ExactPairingWithOpponentSemanticMismatchReason);
Expect("stale_opponent_semantic_mismatch_recorded", staleOpponentSemantic.OpponentSemanticRuntimeMismatch);

var localSlotOne = RecorderContract.ValidatePairing(
    1, "g1", "T800_Opponent", exactBones, "t800", "T800_Local", exactBones);
Expect("local_slot_one_stale_opponent_semantic_allowed", localSlotOne.ExactT800VersusT800);
Expect("local_slot_one_is_recorded", localSlotOne.LocalSlot == 1);

var wrongOrder = exactBones.ToArray();
(wrongOrder[1], wrongOrder[2]) = (wrongOrder[2], wrongOrder[1]);
Expect(
    "wrong_order_rejected",
    RecorderContract.ValidatePairing(
        0, "t800", "T800_Local", exactBones, "t800", "T800_Opponent", wrongOrder).Reason ==
    "opponent_t800_bone_signature_mismatch");
Expect(
    "wrong_local_robot_id_rejected",
    RecorderContract.ValidatePairing(
        0, "g1", "T800_Local", exactBones, "t800", "T800_Opponent", exactBones).Reason ==
    "local_robot_id_not_t800");
Expect(
    "robot_id_case_sensitive",
    !RecorderContract.ValidatePairing(
        0, "T800", "T800_Local", exactBones, "t800", "T800_Opponent", exactBones)
        .ExactT800VersusT800);
Expect(
    "missing_identity_rejected",
    RecorderContract.ValidatePairing(
        0, null, "T800_Local", exactBones, "t800", "T800_Opponent", exactBones).Reason ==
    "local_robot_id_unavailable");
Expect(
    "wrong_bone_count_rejected",
    RecorderContract.ValidatePairing(
        0, "t800", "T800_Local", exactBones[..^1], "t800", "T800_Opponent", exactBones).Reason ==
    "local_bone_count_not_26");
Expect(
    "actual_g1_runtime_rejected",
    RecorderContract.ValidatePairing(
        0,
        "t800",
        "T800_Local",
        exactBones,
        "g1",
        "G1_Opponent",
        new string?[] { "g1_root" }).Reason ==
    "opponent_bone_count_not_26");
Expect(
    "missing_runtime_identity_rejected",
    RecorderContract.ValidatePairing(
        0, "t800", "T800_Local", exactBones, "t800", null, exactBones).Reason ==
    "opponent_runtime_object_name_unavailable");
Expect(
    "invalid_local_slot_rejected",
    RecorderContract.ValidatePairing(
        2, "t800", "T800_Local", exactBones, "t800", "T800_Opponent", exactBones).Reason ==
    "local_slot_invalid");

if (failures.Count > 0)
{
    Console.Error.WriteLine($"FAILED: {string.Join(", ", failures)}");
    return 1;
}

Console.WriteLine("all recorder contract tests passed");
return 0;
