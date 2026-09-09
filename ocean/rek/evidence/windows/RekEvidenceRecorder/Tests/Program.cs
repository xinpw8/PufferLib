using RekEvidence;
using RekEvidenceRecorder;

var failures = new List<string>();

void Expect(string name, bool condition)
{
    Console.WriteLine($"{name}: {(condition ? "PASS" : "FAIL")}");
    if (!condition)
        failures.Add(name);
}

var t800Bones = RecorderContract.T800BoneNames
    .Select(name => (string?)name)
    .ToArray();
var g1Bones = RecorderContract.G1BoneNames
    .Select(name => (string?)name)
    .ToArray();
Expect("schema_v7", RecorderContract.Schema == "rek.private_ai.protocol.v7");
Expect("plugin_version_0_7_2", RecorderContract.PluginVersion == "0.7.2");
Expect("t800_bone_count_26", t800Bones.Length == 26);
Expect("g1_bone_count_30", g1Bones.Length == 30);
Expect(
    "t800_signature_hash",
    RecorderContract.BoneSignatureSha256(t800Bones) ==
    RecorderContract.T800BoneSignatureSha256);
Expect(
    "g1_signature_hash",
    RecorderContract.BoneSignatureSha256(g1Bones) ==
    RecorderContract.G1BoneSignatureSha256);

var exactT800 = RecorderContract.ValidatePairing(
    0, "t800", "T800_Local", t800Bones, "t800", "T800_Opponent", t800Bones);
Expect("exact_t800_pairing", exactT800.ExactT800VersusT800);
Expect("exact_t800_supported", exactT800.ExactSupportedRuntimePairing);
Expect("exact_t800_model", exactT800.RuntimeModel == "t800");
Expect("exact_t800_reason", exactT800.Reason == RecorderContract.ExactT800PairingReason);
Expect("exact_t800_local_semantic", exactT800.LocalSemanticT800);
Expect("exact_t800_no_semantic_mismatch", !exactT800.OpponentSemanticRuntimeMismatch);

var staleOpponentSemantic = RecorderContract.ValidatePairing(
    0, "t800", "T800_Local", t800Bones, "g1", "T800_Opponent", t800Bones);
Expect("stale_t800_semantic_pairing_allowed", staleOpponentSemantic.ExactT800VersusT800);
Expect(
    "stale_t800_semantic_reason",
    staleOpponentSemantic.Reason ==
    RecorderContract.ExactT800PairingWithSemanticMismatchReason);
Expect("stale_t800_semantic_mismatch_recorded", staleOpponentSemantic.OpponentSemanticRuntimeMismatch);

var exactG1WithEmptyLocalSemantic = RecorderContract.ValidatePairing(
    0, string.Empty, "G1_Local", g1Bones, "g1", "G1_Opponent", g1Bones);
Expect("exact_g1_pairing", exactG1WithEmptyLocalSemantic.ExactG1VersusG1);
Expect("exact_g1_supported", exactG1WithEmptyLocalSemantic.ExactSupportedRuntimePairing);
Expect("exact_g1_model", exactG1WithEmptyLocalSemantic.RuntimeModel == "g1");
Expect("empty_local_semantic_not_inferred_g1", !exactG1WithEmptyLocalSemantic.LocalSemanticG1);
Expect("empty_local_semantic_not_marked_mismatch", !exactG1WithEmptyLocalSemantic.LocalSemanticRuntimeMismatch);
Expect(
    "empty_local_semantic_recorded_unavailable",
    exactG1WithEmptyLocalSemantic.LocalSemanticRuntimeConsistency ==
    "semantic_robot_id_unavailable_runtime_g1_exact");

var exactG1LocalSlotOne = RecorderContract.ValidatePairing(
    1, "g1", "G1_Opponent", g1Bones, null, "G1_Local", g1Bones);
Expect("g1_local_slot_one_pairing", exactG1LocalSlotOne.ExactG1VersusG1);
Expect("g1_local_slot_one_recorded", exactG1LocalSlotOne.LocalSlot == 1);
Expect("g1_local_slot_one_empty_semantic_not_inferred", !exactG1LocalSlotOne.LocalSemanticG1);

var wrongT800Order = t800Bones.ToArray();
(wrongT800Order[1], wrongT800Order[2]) = (wrongT800Order[2], wrongT800Order[1]);
Expect(
    "wrong_t800_order_rejected",
    RecorderContract.ValidatePairing(
        0, "t800", "T800_Local", t800Bones, "t800", "T800_Opponent", wrongT800Order)
        .Reason == "opponent_t800_bone_signature_mismatch");

var wrongG1Order = g1Bones.ToArray();
(wrongG1Order[1], wrongG1Order[2]) = (wrongG1Order[2], wrongG1Order[1]);
Expect(
    "wrong_g1_order_rejected",
    RecorderContract.ValidatePairing(
        0, "g1", "G1_Local", g1Bones, "g1", "G1_Opponent", wrongG1Order)
        .Reason == "opponent_g1_bone_signature_mismatch");
Expect(
    "mixed_models_rejected",
    RecorderContract.ValidatePairing(
        0, "t800", "T800_Local", t800Bones, "g1", "G1_Opponent", g1Bones)
        .Reason == "mixed_supported_runtime_models_rejected");
Expect(
    "unsupported_bone_count_rejected",
    RecorderContract.ValidatePairing(
        0, "g1", "G1_Local", g1Bones[..^1], "g1", "G1_Opponent", g1Bones)
        .Reason == "local_unsupported_bone_count:29");
Expect(
    "missing_runtime_identity_rejected",
    RecorderContract.ValidatePairing(
        0, "g1", "G1_Local", g1Bones, "g1", null, g1Bones)
        .Reason == "opponent_runtime_object_name_unavailable");
Expect(
    "invalid_local_slot_rejected",
    RecorderContract.ValidatePairing(
        2, "g1", "G1_Local", g1Bones, "g1", "G1_Opponent", g1Bones)
        .Reason == "local_slot_invalid");

var route = new SoloRouteProofTracker();
route.ObserveFindMatch("solo");
route.ObserveConnectToArena("private-arena-a");
route.ObserveEnterChampionship(
    "private-arena-a", "test.invalid", 7777, koth: false, solo: true);
var routeProof = route.SnapshotForArena("private-arena-a");
Expect(
    "solo_route_exact_chain_accepted",
    routeProof.SoloRouteProven &&
    routeProof.ArenaIdentityConsistent &&
    routeProof.Reason == "solo_route_proven");
Expect(
    "solo_route_does_not_claim_server_privacy",
    !routeProof.ServerPrivateProven &&
    routeProof.ServerPrivateStatus == "unknown");
var boundRouteProof = route.SnapshotForRuntimeSession(
    "private-arena-a",
    "test.invalid",
    7777,
    "test.invalid",
    7777,
    runtimeSessionIdentity: 201);
Expect(
    "solo_route_runtime_session_binding_accepted",
    boundRouteProof.SoloRouteProven &&
    boundRouteProof.RuntimeSessionIdentityConsistent);
Expect(
    "solo_route_scope_requires_exact_bot_one_no_human",
    !SoloRouteProofContract.EvaluateScope(
        exactBotOneNoHumanProofEstablished: false,
        boundRouteProof).Allowed);
Expect(
    "solo_route_scope_accepts_exact_bot_one_no_human",
    SoloRouteProofContract.EvaluateScope(
        exactBotOneNoHumanProofEstablished: true,
        boundRouteProof).Allowed);
Expect(
    "solo_route_runtime_session_change_rejected",
    !route.SnapshotForRuntimeSession(
        "private-arena-a",
        "test.invalid",
        7777,
        "test.invalid",
        7777,
        runtimeSessionIdentity: 202).SoloRouteProven);
Expect(
    "solo_route_bound_lifecycle_change_invalidates",
    route.InvalidateIfRuntimeSessionBound(
        "network_session_stopped_after_solo_route_binding") &&
    !route.SnapshotForRuntimeSession(
        "private-arena-a",
        "test.invalid",
        7777,
        "test.invalid",
        7777,
        runtimeSessionIdentity: 201).SoloRouteProven);
Expect(
    "solo_route_arena_change_rejected",
    !route.SnapshotForArena("private-arena-b").SoloRouteProven);

var unboundLifecycleRoute = new SoloRouteProofTracker();
unboundLifecycleRoute.ObserveFindMatch("solo");
unboundLifecycleRoute.ObserveConnectToArena("private-arena-a");
unboundLifecycleRoute.ObserveEnterChampionship(
    "private-arena-a",
    "test.invalid",
    7777,
    koth: false,
    solo: true);
Expect(
    "initial_network_connection_preserves_unbound_route",
    !unboundLifecycleRoute.InvalidateIfRuntimeSessionBound(
        "network_client_connected_after_solo_route_binding") &&
    unboundLifecycleRoute.SnapshotForArena("private-arena-a").SoloRouteProven);

var publicRoute = new SoloRouteProofTracker();
publicRoute.ObserveFindMatch("solo");
publicRoute.ObserveConnectToArena("public-arena-a");
publicRoute.ObserveEnterChampionship(
    "public-arena-a", "test.invalid", 7777, koth: true, solo: false);
Expect(
    "public_koth_enter_rejected",
    !publicRoute.SnapshotForArena("public-arena-a").SoloRouteProven);

var mismatchedRoute = new SoloRouteProofTracker();
mismatchedRoute.ObserveFindMatch("solo");
mismatchedRoute.ObserveConnectToArena("private-arena-a");
mismatchedRoute.ObserveEnterChampionship(
    "private-arena-b", "test.invalid", 7777, koth: false, solo: true);
Expect(
    "mismatched_route_arena_rejected",
    !mismatchedRoute.SnapshotForArena("private-arena-b").SoloRouteProven);

route.ObserveFindMatch("championship");
Expect(
    "new_non_solo_flow_invalidates_prior_route",
    !route.SnapshotForArena("private-arena-a").SoloRouteProven);

if (failures.Count > 0)
{
    Console.Error.WriteLine($"FAILED: {string.Join(", ", failures)}");
    return 1;
}

Console.WriteLine("all recorder contract tests passed");
return 0;
