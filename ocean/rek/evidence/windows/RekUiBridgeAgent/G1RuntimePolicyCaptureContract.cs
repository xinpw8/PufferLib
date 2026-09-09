using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace RekUiBridgeAgent;

internal static class G1RuntimePolicyCaptureContract
{
    internal const string CaptureSchema = "rek.g1_runtime_policy_capture.v3";
    internal const string CanonicalConfigSchema = "rek.g1_runtime_sonic_config_snapshot.v1";
    internal const string CanonicalConfigSource = "live_deserialized_sonic_config_typed_dto";

    private static readonly JsonSerializerOptions CanonicalConfigJson = new()
    {
        WriteIndented = true,
    };

    internal static byte[] CanonicalRuntimeConfigBytes(RuntimeSonicConfigSnapshot config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return Encoding.UTF8.GetBytes(
            JsonSerializer.Serialize(config, CanonicalConfigJson) + "\n");
    }

    internal static string Sha256(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    internal static RuntimeRawConfigComparison CompareRawConfigRecoveredFields(
        byte[] rawConfigBytes,
        RuntimeSonicConfigSnapshot liveConfig)
    {
        ArgumentNullException.ThrowIfNull(rawConfigBytes);
        ArgumentNullException.ThrowIfNull(liveConfig);

        RuntimeSonicConfigSnapshot? rawConfig;
        try
        {
            rawConfig = JsonSerializer.Deserialize<RuntimeSonicConfigSnapshot>(
                rawConfigBytes,
                CanonicalConfigJson);
        }
        catch (JsonException)
        {
            return new RuntimeRawConfigComparison("raw_config_json_invalid", null);
        }
        catch (NotSupportedException)
        {
            return new RuntimeRawConfigComparison("raw_config_shape_unsupported", null);
        }

        if (rawConfig is null)
            return new RuntimeRawConfigComparison("raw_config_json_null", null);

        var exact = CanonicalRuntimeConfigBytes(rawConfig)
            .AsSpan()
            .SequenceEqual(CanonicalRuntimeConfigBytes(liveConfig));
        return new RuntimeRawConfigComparison(
            exact
                ? "raw_config_recovered_fields_exact"
                : "raw_config_recovered_fields_differ",
            exact);
    }

    internal static RuntimePolicyCaptureResult FailedWithRunnerDiagnostics(
        string reason,
        RuntimeRunnerInspection localRunner,
        RuntimeRunnerInspection opponentRunner) =>
        new(
            false,
            reason,
            new RuntimePolicyCaptureFailurePayload(
                CaptureSchema,
                "runtime_runner_inspection_complete",
                reason,
                localRunner,
                opponentRunner,
                false));

    internal static bool HasRunnableModelFields(RuntimeRunnerInspection runner) =>
        runner.FusedOnnxFieldPresent ||
        runner.EncoderOnnxFieldPresent && runner.DecoderOnnxFieldPresent;
}

internal sealed record RuntimePolicyCaptureResult(bool Captured, string Reason, object? Payload)
{
    internal static RuntimePolicyCaptureResult Succeeded(object payload) =>
        new(true, "g1_runtime_policy_assets_captured", payload);

    internal static RuntimePolicyCaptureResult Failed(string reason) =>
        new(false, reason, null);

    internal static RuntimePolicyCaptureResult FailedWithRunnerDiagnostics(
        string reason,
        RuntimeRunnerInspection localRunner,
        RuntimeRunnerInspection opponentRunner) =>
        G1RuntimePolicyCaptureContract.FailedWithRunnerDiagnostics(
            reason,
            localRunner,
            opponentRunner);
}

internal sealed record RuntimePolicyCaptureFailurePayload(
    string Schema,
    string Stage,
    string Reason,
    RuntimeRunnerInspection LocalRunner,
    RuntimeRunnerInspection OpponentRunner,
    bool ProprietaryPayloadsInProtocol);

internal sealed record RuntimeRunnerInspection(
    string Role,
    string RuntimeName,
    bool ConfigJsonFieldPresent,
    bool LiveConfigFieldPresent,
    bool InitComplete,
    bool ActionsReady,
    bool EncoderOnnxFieldPresent,
    bool DecoderOnnxFieldPresent,
    bool FusedOnnxFieldPresent,
    int NumJoints,
    int EncoderObservationSize,
    int DecoderObservationSize,
    int? TokenDimension,
    string? TokenDimensionReadFailure,
    int MotionDofs,
    int ActiveEncoderMode,
    string ModelMode,
    bool CanonicalLiveConfigCaptured,
    string? CanonicalLiveConfigSchema,
    string? CanonicalLiveConfigSource,
    string? CanonicalLiveConfigSha256,
    long? CanonicalLiveConfigBytes,
    string? CanonicalLiveConfigRelativePath,
    string RawConfigComparisonStatus,
    bool? RawConfigRecoveredFieldsExact);

internal sealed record RuntimeRawConfigComparison(
    string Status,
    bool? RecoveredFieldsExact);

internal sealed record RuntimeSonicConfigSnapshot(
    [property: JsonPropertyName("schema_version")] string? SchemaVersion,
    [property: JsonPropertyName("robot_name")] string? RobotName,
    [property: JsonPropertyName("timing")] RuntimeTimingConfigSnapshot? Timing,
    [property: JsonPropertyName("action")] RuntimeActionConfigSnapshot? Action,
    [property: JsonPropertyName("encoder")] RuntimeEncoderConfigSnapshot? Encoder,
    [property: JsonPropertyName("decoder")] RuntimeDecoderConfigSnapshot? Decoder,
    [property: JsonPropertyName("vr_tracking")] RuntimeVrTrackingConfigSnapshot? VrTracking,
    [property: JsonPropertyName("isaaclab_to_mujoco")] int[]? IsaacLabToMujoco,
    [property: JsonPropertyName("mujoco_to_isaaclab")] int[]? MujocoToIsaacLab,
    [property: JsonPropertyName("joints")] RuntimeJointConfigSnapshot[]? Joints);

internal sealed record RuntimeTimingConfigSnapshot(
    [property: JsonPropertyName("physics_dt")] float PhysicsDt,
    [property: JsonPropertyName("controller_rate_hz")] int ControllerRateHz);

internal sealed record RuntimeActionConfigSnapshot(
    [property: JsonPropertyName("clip_actions")] float ClipActions);

internal sealed record RuntimeEncoderConfigSnapshot(
    [property: JsonPropertyName("mode")] int Mode,
    [property: JsonPropertyName("token_dimension")] int TokenDimension,
    [property: JsonPropertyName("input_tensor_name")] string? InputTensorName,
    [property: JsonPropertyName("output_tensor_name")] string? OutputTensorName,
    [property: JsonPropertyName("mode_encoding")] string? ModeEncoding,
    [property: JsonPropertyName("observations")] RuntimeEncoderObservationSnapshot[]? Observations);

internal sealed record RuntimeEncoderObservationSnapshot(
    [property: JsonPropertyName("name")] string? Name,
    [property: JsonPropertyName("dim")] int Dimension,
    [property: JsonPropertyName("active_modes")] int[]? ActiveModes);

internal sealed record RuntimeDecoderConfigSnapshot(
    [property: JsonPropertyName("input_tensor_name")] string? InputTensorName,
    [property: JsonPropertyName("output_tensor_name")] string? OutputTensorName,
    [property: JsonPropertyName("observations")] RuntimeDecoderObservationSnapshot[]? Observations);

internal sealed record RuntimeDecoderObservationSnapshot(
    [property: JsonPropertyName("name")] string? Name,
    [property: JsonPropertyName("type")] string? Type,
    [property: JsonPropertyName("dim")] int Dimension,
    [property: JsonPropertyName("num_frames")] int NumFrames,
    [property: JsonPropertyName("step")] int Step,
    [property: JsonPropertyName("dim_per_frame")] int DimensionPerFrame);

internal sealed record RuntimeVrTrackingConfigSnapshot(
    [property: JsonPropertyName("points")] RuntimeVrTrackingPointSnapshot[]? Points);

internal sealed record RuntimeVrTrackingPointSnapshot(
    [property: JsonPropertyName("name")] string? Name,
    [property: JsonPropertyName("xr_node")] string? XrNode,
    [property: JsonPropertyName("axis_alignment")] float[]? AxisAlignment,
    [property: JsonPropertyName("offset")] float[]? Offset,
    [property: JsonPropertyName("robot_neutral_pos")] float[]? RobotNeutralPosition,
    [property: JsonPropertyName("robot_neutral_rot")] float[]? RobotNeutralRotation);

internal sealed record RuntimeJointConfigSnapshot(
    [property: JsonPropertyName("name")] string? Name,
    [property: JsonPropertyName("default_pos")] float DefaultPositionRadians,
    [property: JsonPropertyName("kp")] float Kp,
    [property: JsonPropertyName("kd")] float Kd,
    [property: JsonPropertyName("effort_limit")] float EffortLimitNewtonMetres,
    [property: JsonPropertyName("action_scale")] float ActionScaleRadians);
