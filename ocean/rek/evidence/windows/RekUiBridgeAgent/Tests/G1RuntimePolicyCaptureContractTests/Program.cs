using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using RekUiBridgeAgent;

var failures = new List<string>();
var cases = 0;

void Expect(string name, bool condition)
{
    cases++;
    if (!condition)
        failures.Add(name);
}

var config = new RuntimeSonicConfigSnapshot(
    "1.0",
    "g1",
    new RuntimeTimingConfigSnapshot(0.002f, 50),
    new RuntimeActionConfigSnapshot(100.0f),
    new RuntimeEncoderConfigSnapshot(
        2,
        64,
        "encoder_input",
        "encoder_output",
        "raw_int_padded",
        new[]
        {
            new RuntimeEncoderObservationSnapshot(
                "encoder_mode_4",
                4,
                new[] { 0, 1, 2 }),
        }),
    new RuntimeDecoderConfigSnapshot(
        "decoder_input",
        "decoder_output",
        new[]
        {
            new RuntimeDecoderObservationSnapshot(
                "history",
                "history",
                30,
                10,
                1,
                3),
        }),
    new RuntimeVrTrackingConfigSnapshot(
        new[]
        {
            new RuntimeVrTrackingPointSnapshot(
                "left_hand",
                "LeftHand",
                new[] { 1.0f, 0.0f, 0.0f },
                new[] { 0.1f, 0.2f, 0.3f },
                new[] { 0.2f, 0.2f, 0.25f },
                new[] { 1.0f, 0.0f, 0.0f, 0.0f }),
        }),
    new[] { 2, 0, 1 },
    new[] { 1, 2, 0 },
    new[]
    {
        new RuntimeJointConfigSnapshot(
            "left_hip_pitch_joint",
            -0.1f,
            100.0f,
            2.0f,
            88.0f,
            0.25f),
    });

var canonical = G1RuntimePolicyCaptureContract.CanonicalRuntimeConfigBytes(config);
var canonicalAgain = G1RuntimePolicyCaptureContract.CanonicalRuntimeConfigBytes(config);
var canonicalSha256 = G1RuntimePolicyCaptureContract.Sha256(canonical);
Expect("canonical_bytes_deterministic", canonical.AsSpan().SequenceEqual(canonicalAgain));
Expect(
    "canonical_sha256_pinned",
    canonicalSha256 == "62ff8f93535296d8ca6486ee2302e90a04398891ba5087734c2ebe362d00a929");
Expect("canonical_lf_terminated", canonical[^1] == (byte)'\n');

using (var document = JsonDocument.Parse(canonical))
{
    var root = document.RootElement;
    Expect(
        "canonical_root_field_order_and_completeness",
        root.EnumerateObject().Select(property => property.Name).SequenceEqual(new[]
        {
            "schema_version",
            "robot_name",
            "timing",
            "action",
            "encoder",
            "decoder",
            "vr_tracking",
            "isaaclab_to_mujoco",
            "mujoco_to_isaaclab",
            "joints",
        }));
    Expect(
        "canonical_timing_fields",
        root.GetProperty("timing").EnumerateObject().Select(property => property.Name)
            .SequenceEqual(new[] { "physics_dt", "controller_rate_hz" }));
    Expect(
        "canonical_encoder_fields",
        root.GetProperty("encoder").EnumerateObject().Select(property => property.Name)
            .SequenceEqual(new[]
            {
                "mode",
                "token_dimension",
                "input_tensor_name",
                "output_tensor_name",
                "mode_encoding",
                "observations",
            }));
    Expect(
        "canonical_decoder_observation_fields",
        root.GetProperty("decoder").GetProperty("observations")[0]
            .EnumerateObject().Select(property => property.Name).SequenceEqual(new[]
            {
                "name",
                "type",
                "dim",
                "num_frames",
                "step",
                "dim_per_frame",
            }));
    Expect(
        "canonical_vr_tracking_complete",
        root.GetProperty("vr_tracking").GetProperty("points")[0]
            .EnumerateObject().Select(property => property.Name).SequenceEqual(new[]
            {
                "name",
                "xr_node",
                "axis_alignment",
                "offset",
                "robot_neutral_pos",
                "robot_neutral_rot",
            }));
    Expect(
        "canonical_joint_fields_no_derived_index",
        root.GetProperty("joints")[0].EnumerateObject().Select(property => property.Name)
            .SequenceEqual(new[]
            {
                "name",
                "default_pos",
                "kp",
                "kd",
                "effort_limit",
                "action_scale",
            }));
}

using (var nullDocument = JsonDocument.Parse(
           G1RuntimePolicyCaptureContract.CanonicalRuntimeConfigBytes(config with
           {
               VrTracking = null,
               IsaacLabToMujoco = null,
               MujocoToIsaacLab = null,
               Joints = null,
           })))
{
    var root = nullDocument.RootElement;
    Expect("canonical_nulls_are_not_invented_as_empty_arrays",
        root.GetProperty("vr_tracking").ValueKind == JsonValueKind.Null &&
        root.GetProperty("isaaclab_to_mujoco").ValueKind == JsonValueKind.Null &&
        root.GetProperty("mujoco_to_isaaclab").ValueKind == JsonValueKind.Null &&
        root.GetProperty("joints").ValueKind == JsonValueKind.Null);
}

var exactComparison = G1RuntimePolicyCaptureContract.CompareRawConfigRecoveredFields(
    canonical,
    config);
Expect(
    "raw_exact_status",
    exactComparison is
    {
        Status: "raw_config_recovered_fields_exact",
        RecoveredFieldsExact: true,
    });

var rawWithUnknownField = JsonNode.Parse(canonical)!.AsObject();
rawWithUnknownField["unrecovered_future_field"] = 123;
var unknownFieldComparison = G1RuntimePolicyCaptureContract.CompareRawConfigRecoveredFields(
    Encoding.UTF8.GetBytes(rawWithUnknownField.ToJsonString()),
    config);
Expect(
    "raw_unknown_fields_do_not_invent_runtime_fields",
    unknownFieldComparison.RecoveredFieldsExact is true);

var differentRaw = JsonNode.Parse(canonical)!.AsObject();
differentRaw["robot_name"] = "different";
var differentComparison = G1RuntimePolicyCaptureContract.CompareRawConfigRecoveredFields(
    Encoding.UTF8.GetBytes(differentRaw.ToJsonString()),
    config);
Expect(
    "raw_difference_detected",
    differentComparison is
    {
        Status: "raw_config_recovered_fields_differ",
        RecoveredFieldsExact: false,
    });
var invalidComparison = G1RuntimePolicyCaptureContract.CompareRawConfigRecoveredFields(
    Encoding.UTF8.GetBytes("{"),
    config);
Expect(
    "raw_invalid_is_not_exact",
    invalidComparison is
    {
        Status: "raw_config_json_invalid",
        RecoveredFieldsExact: null,
    });

var localInspection = new RuntimeRunnerInspection(
    "local",
    "g1-local",
    false,
    false,
    true,
    true,
    true,
    false,
    true,
    29,
    1504,
    704,
    64,
    null,
    29,
    2,
    "fused",
    false,
    null,
    null,
    null,
    null,
    null,
    "not_attempted",
    null);
var opponentInspection = localInspection with
{
    Role = "opponent",
    RuntimeName = "g1-opponent",
};
var failed = RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
    "local_sonic_runtime_config_unavailable",
    localInspection,
    opponentInspection);
Expect("diagnostic_failure_remains_fail_closed", !failed.Captured);
Expect("diagnostic_failure_reason_retained", failed.Reason == "local_sonic_runtime_config_unavailable");
Expect("diagnostic_failure_payload_typed", failed.Payload is RuntimePolicyCaptureFailurePayload);
var payload = (RuntimePolicyCaptureFailurePayload)failed.Payload!;
Expect("diagnostic_schema", payload.Schema == G1RuntimePolicyCaptureContract.CaptureSchema);
Expect("diagnostic_stage", payload.Stage == "runtime_runner_inspection_complete");
Expect("diagnostic_config_fields", !payload.LocalRunner.ConfigJsonFieldPresent &&
                                   !payload.LocalRunner.LiveConfigFieldPresent);
Expect("diagnostic_lifecycle", payload.LocalRunner.InitComplete && payload.LocalRunner.ActionsReady);
Expect("diagnostic_model_fields", payload.LocalRunner.EncoderOnnxFieldPresent &&
                                        !payload.LocalRunner.DecoderOnnxFieldPresent &&
                                        payload.LocalRunner.FusedOnnxFieldPresent);
Expect("diagnostic_four_live_dimensions", payload.LocalRunner.NumJoints == 29 &&
                                               payload.LocalRunner.EncoderObservationSize == 1504 &&
                                               payload.LocalRunner.DecoderObservationSize == 704 &&
                                               payload.LocalRunner.TokenDimension == 64);
Expect("diagnostic_protocol_excludes_payloads", !payload.ProprietaryPayloadsInProtocol);
var artifactInspection = localInspection with
{
    LiveConfigFieldPresent = true,
    CanonicalLiveConfigCaptured = true,
    CanonicalLiveConfigSchema = G1RuntimePolicyCaptureContract.CanonicalConfigSchema,
    CanonicalLiveConfigSource = G1RuntimePolicyCaptureContract.CanonicalConfigSource,
    CanonicalLiveConfigSha256 = canonicalSha256,
    CanonicalLiveConfigBytes = canonical.LongLength,
    CanonicalLiveConfigRelativePath = $"assets/{canonicalSha256}.json",
};
var artifactFailure = RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
    "local_sonic_model_assets_incomplete",
    artifactInspection,
    opponentInspection);
var artifactPayload = (RuntimePolicyCaptureFailurePayload)artifactFailure.Payload!;
Expect("diagnostic_retains_canonical_artifact_reference",
    artifactPayload.LocalRunner.CanonicalLiveConfigCaptured &&
    artifactPayload.LocalRunner.CanonicalLiveConfigSchema ==
        G1RuntimePolicyCaptureContract.CanonicalConfigSchema &&
    artifactPayload.LocalRunner.CanonicalLiveConfigSource ==
        G1RuntimePolicyCaptureContract.CanonicalConfigSource &&
    artifactPayload.LocalRunner.CanonicalLiveConfigSha256 == canonicalSha256 &&
    artifactPayload.LocalRunner.CanonicalLiveConfigBytes == canonical.LongLength &&
    artifactPayload.LocalRunner.CanonicalLiveConfigRelativePath ==
        $"assets/{canonicalSha256}.json");
Expect("ordinary_preinspection_failure_has_no_payload", RuntimePolicyCaptureResult.Failed("early").Payload is null);
Expect(
    "fused_model_is_runnable",
    G1RuntimePolicyCaptureContract.HasRunnableModelFields(localInspection));
Expect(
    "split_model_is_runnable",
    G1RuntimePolicyCaptureContract.HasRunnableModelFields(localInspection with
    {
        FusedOnnxFieldPresent = false,
        EncoderOnnxFieldPresent = true,
        DecoderOnnxFieldPresent = true,
    }));
Expect(
    "partial_model_fails_closed",
    !G1RuntimePolicyCaptureContract.HasRunnableModelFields(localInspection with
    {
        FusedOnnxFieldPresent = false,
        EncoderOnnxFieldPresent = true,
        DecoderOnnxFieldPresent = false,
    }));

if (failures.Count != 0)
{
    Console.Error.WriteLine($"FAIL cases={cases} canonical_sha256={canonicalSha256} names={string.Join(',', failures)}");
    return 1;
}

Console.WriteLine($"PASS cases={cases} canonical_sha256={canonicalSha256}");
return 0;
