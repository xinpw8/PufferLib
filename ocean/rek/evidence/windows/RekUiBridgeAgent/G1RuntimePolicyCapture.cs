using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using BepInEx;
using REKApp;
using UnityEngine;

namespace RekUiBridgeAgent;

internal static class G1RuntimePolicyCapture
{
    internal const string Schema = G1RuntimePolicyCaptureContract.CaptureSchema;

    internal static RuntimePolicyCaptureResult Capture(
        RobotInputController input,
        Robot localRobot,
        Robot opponentRobot,
        string gameAssemblySha256,
        string globalMetadataSha256,
        string sharedAssets0Sha256)
    {
        RuntimeRunnerInspection? localInspection = null;
        RuntimeRunnerInspection? opponentInspection = null;
        try
        {
            if (!IsSha256(gameAssemblySha256) ||
                !IsSha256(globalMetadataSha256) ||
                !IsSha256(sharedAssets0Sha256))
            {
                return RuntimePolicyCaptureResult.Failed("build_identity_unavailable");
            }

            var robotConfig = input.robotConfig;
            if (robotConfig is null)
                return RuntimePolicyCaptureResult.Failed("local_robot_config_unavailable");

            var localRunner = input.sonicRunner ?? localRobot.GetComponent<SonicPolicyRunner>();
            var opponentRunner = opponentRobot.GetComponent<SonicPolicyRunner>();
            if (localRunner is null)
                return RuntimePolicyCaptureResult.Failed("local_sonic_policy_runner_unavailable");
            if (opponentRunner is null)
                return RuntimePolicyCaptureResult.Failed("opponent_sonic_policy_runner_unavailable");

            localInspection = InspectRunner("local", localRunner);
            opponentInspection = InspectRunner("opponent", opponentRunner);
            if (!localInspection.LiveConfigFieldPresent)
            {
                return RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
                    localRobot.IsVisualOnly
                        ? "local_visual_only_sonic_runtime_config_not_instantiated"
                        : "local_sonic_runtime_config_unavailable",
                    localInspection,
                    opponentInspection);
            }
            if (!opponentInspection.LiveConfigFieldPresent)
            {
                return RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
                    opponentRobot.IsVisualOnly
                        ? "opponent_visual_only_sonic_runtime_config_not_instantiated"
                        : "opponent_sonic_runtime_config_unavailable",
                    localInspection,
                    opponentInspection);
            }

            var outputRoot = Path.Combine(
                Paths.GameRootPath,
                "BepInEx",
                "evidence",
                "g1-runtime-policy",
                gameAssemblySha256);
            var assetsRoot = Path.Combine(outputRoot, "assets");
            Directory.CreateDirectory(assetsRoot);

            var assets = new Dictionary<string, RuntimeAssetFile>(StringComparer.Ordinal);
            var localPolicy = CaptureRunner("local", localRunner, assetsRoot, assets);
            localInspection = CompleteInspection(localInspection, localPolicy);
            var opponentPolicy = CaptureRunner("opponent", opponentRunner, assetsRoot, assets);
            opponentInspection = CompleteInspection(opponentInspection, opponentPolicy);
            if (RawConfigCrossCheckFailed(localPolicy))
            {
                return RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
                    $"local_sonic_raw_config_cross_check_failed:{localPolicy.RawConfigComparison.Status}",
                    localInspection,
                    opponentInspection);
            }
            if (RawConfigCrossCheckFailed(opponentPolicy))
            {
                return RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
                    $"opponent_sonic_raw_config_cross_check_failed:{opponentPolicy.RawConfigComparison.Status}",
                    localInspection,
                    opponentInspection);
            }
            if (!G1RuntimePolicyCaptureContract.HasRunnableModelFields(localInspection))
            {
                return RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
                    "local_sonic_model_assets_incomplete",
                    localInspection,
                    opponentInspection);
            }
            if (!G1RuntimePolicyCaptureContract.HasRunnableModelFields(opponentInspection))
            {
                return RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
                    "opponent_sonic_model_assets_incomplete",
                    localInspection,
                    opponentInspection);
            }

            var locomotion = new[]
            {
                CaptureClip("idle", robotConfig.idle, assetsRoot, assets),
                CaptureClip("walk_forward", robotConfig.walkForward, assetsRoot, assets),
                CaptureClip("walk_backward", robotConfig.walkBackward, assetsRoot, assets),
                CaptureClip("strafe_left", robotConfig.strafeLeft, assetsRoot, assets),
                CaptureClip("strafe_right", robotConfig.strafeRight, assetsRoot, assets),
                CaptureClip("turn_left", robotConfig.turnLeft, assetsRoot, assets),
                CaptureClip("turn_right", robotConfig.turnRight, assetsRoot, assets),
            };

            var moves = new List<RuntimeClipManifest>();
            if (robotConfig.moves is not null)
            {
                for (var index = 0; index < robotConfig.moves.Count; index++)
                {
                    moves.Add(CaptureClip(
                        $"move_{index}",
                        robotConfig.moves[index],
                        assetsRoot,
                        assets));
                }
            }

            var recovery = new[]
            {
                CaptureClip("get_up_prone", localRunner.getUpProneClip, assetsRoot, assets),
                CaptureClip("get_up_supine", localRunner.getUpSupineClip, assetsRoot, assets),
            };

            var localAssetSet = PolicyAssetIdentity(localPolicy);
            var opponentAssetSet = PolicyAssetIdentity(opponentPolicy);
            var policyAssetsExact = localAssetSet is not null && opponentAssetSet is not null
                ? string.Equals(localAssetSet, opponentAssetSet, StringComparison.Ordinal)
                : (bool?)null;
            var localRuntimeConfigIdentity = localPolicy.RuntimeConfigCanonical?.Sha256 ??
                throw new InvalidDataException("local_runtime_config_canonical_unavailable");
            var opponentRuntimeConfigIdentity = opponentPolicy.RuntimeConfigCanonical?.Sha256 ??
                throw new InvalidDataException("opponent_runtime_config_canonical_unavailable");
            var manifest = new RuntimePolicyManifest(
                Schema,
                gameAssemblySha256,
                globalMetadataSha256,
                sharedAssets0Sha256,
                localPolicy,
                opponentPolicy,
                policyAssetsExact,
                localAssetSet,
                opponentAssetSet,
                string.Equals(
                    localRuntimeConfigIdentity,
                    opponentRuntimeConfigIdentity,
                    StringComparison.Ordinal),
                localRuntimeConfigIdentity,
                opponentRuntimeConfigIdentity,
                robotConfig.name,
                robotConfig.locomotionTransitionSettle,
                robotConfig.transitionSettlePlanarSpeed,
                robotConfig.transitionSettleYawRate,
                locomotion,
                recovery,
                moves.ToArray(),
                assets.Values.OrderBy(value => value.Sha256, StringComparer.Ordinal).ToArray());

            var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions
            {
                WriteIndented = true,
            }) + "\n";
            var manifestBytes = Encoding.UTF8.GetBytes(json);
            var manifestPath = Path.Combine(outputRoot, "manifest.json");
            WriteOrVerify(manifestPath, manifestBytes);

            return RuntimePolicyCaptureResult.Succeeded(new
            {
                schema = Schema,
                output_root = outputRoot,
                manifest_path = manifestPath,
                manifest_sha256 = Hash(manifestBytes),
                asset_count = assets.Count,
                local_runner_initialized = localRunner.initComplete,
                local_runner_paused = localRunner.paused,
                local_runner_num_joints = localRunner.numJoints,
                local_runner_encoder_observation_size = localRunner.encoderObsSize,
                local_runner_decoder_observation_size = localRunner.decoderObsSize,
                local_runner_token_dimension = localRunner.TokenDimension,
                local_runner_actions_ready = localRunner.actionsReady,
                local_runner_diagnostic = localInspection,
                opponent_runner_diagnostic = opponentInspection,
                local_and_opponent_policy_assets_exact = manifest.LocalAndOpponentPolicyAssetsExact,
                policy_asset_identity_status = manifest.LocalAndOpponentPolicyAssetsExact is null
                    ? "unknown_runtime_model_bytes_unavailable"
                    : "measured_from_runtime_model_bytes",
                local_policy_identity_sha256 = localAssetSet,
                opponent_policy_identity_sha256 = opponentAssetSet,
                local_and_opponent_runtime_config_exact =
                    manifest.LocalAndOpponentRuntimeConfigExact,
                local_runtime_config_identity_sha256 = localRuntimeConfigIdentity,
                opponent_runtime_config_identity_sha256 = opponentRuntimeConfigIdentity,
                proprietary_payloads_in_protocol = false,
            });
        }
        catch (Exception exception)
        {
            var reason = $"runtime_policy_capture_failed:{exception.GetType().Name}";
            return localInspection is not null && opponentInspection is not null
                ? RuntimePolicyCaptureResult.FailedWithRunnerDiagnostics(
                    reason,
                    localInspection,
                    opponentInspection)
                : RuntimePolicyCaptureResult.Failed(reason);
        }
    }

    private static RuntimeRunnerInspection InspectRunner(
        string role,
        SonicPolicyRunner runner)
    {
        int? tokenDimension = null;
        string? tokenDimensionReadFailure = null;
        try
        {
            tokenDimension = runner.TokenDimension;
        }
        catch (Exception exception)
        {
            tokenDimensionReadFailure = exception.GetType().Name;
        }

        var encoderPresent = runner.encoderOnnxBytes is not null;
        var decoderPresent = runner.decoderOnnxBytes is not null;
        var fusedPresent = runner.fusedOnnxBytes is not null;
        var modelMode = fusedPresent
            ? "fused"
            : encoderPresent && decoderPresent
                ? "split_encoder_decoder"
                : runner.initComplete && runner.actionsReady
                    ? "initialized_runtime_model_bytes_unavailable"
                    : "unresolved";
        return new RuntimeRunnerInspection(
            role,
            runner.name,
            runner.configJson is not null,
            runner.config is not null,
            runner.initComplete,
            runner.actionsReady,
            encoderPresent,
            decoderPresent,
            fusedPresent,
            runner.numJoints,
            runner.encoderObsSize,
            runner.decoderObsSize,
            tokenDimension,
            tokenDimensionReadFailure,
            runner.motionDofs,
            runner._activeEncoderMode,
            modelMode,
            false,
            null,
            null,
            null,
            null,
            null,
            "not_attempted",
            null);
    }

    private static RuntimeRunnerInspection CompleteInspection(
        RuntimeRunnerInspection inspection,
        RuntimeRunnerManifest runner) =>
        inspection with
        {
            ConfigJsonFieldPresent = runner.Config is not null,
            LiveConfigFieldPresent = runner.RuntimeConfig is not null,
            InitComplete = runner.RuntimeInitialized,
            ActionsReady = runner.ActionsReady,
            EncoderOnnxFieldPresent = runner.Encoder is not null,
            DecoderOnnxFieldPresent = runner.Decoder is not null,
            FusedOnnxFieldPresent = runner.Fused is not null,
            NumJoints = runner.NumJoints,
            EncoderObservationSize = runner.EncoderObservationSize,
            DecoderObservationSize = runner.DecoderObservationSize,
            TokenDimension = runner.LiveTokenDimension,
            TokenDimensionReadFailure = runner.LiveTokenDimension is null
                ? inspection.TokenDimensionReadFailure
                : null,
            MotionDofs = runner.MotionDofs,
            ActiveEncoderMode = runner.ActiveEncoderMode,
            ModelMode = runner.ModelMode,
            CanonicalLiveConfigCaptured = runner.RuntimeConfigCanonical is not null,
            CanonicalLiveConfigSchema = runner.RuntimeConfigCanonicalSchema,
            CanonicalLiveConfigSource = runner.RuntimeConfigCanonicalSource,
            CanonicalLiveConfigSha256 = runner.RuntimeConfigCanonical?.Sha256,
            CanonicalLiveConfigBytes = runner.RuntimeConfigCanonical?.Bytes,
            CanonicalLiveConfigRelativePath = runner.RuntimeConfigCanonical?.RelativePath,
            RawConfigComparisonStatus = runner.RawConfigComparison.Status,
            RawConfigRecoveredFieldsExact = runner.RawConfigComparison.RecoveredFieldsExact,
        };

    private static bool RawConfigCrossCheckFailed(RuntimeRunnerManifest runner) =>
        runner.Config is not null && runner.RawConfigComparison.RecoveredFieldsExact is not true;

    private static RuntimeRunnerManifest CaptureRunner(
        string role,
        SonicPolicyRunner runner,
        string assetsRoot,
        IDictionary<string, RuntimeAssetFile> assets)
    {
        var config = CaptureAsset($"{role}_config", runner.configJson, ".json", assetsRoot, assets);
        var encoder = CaptureAsset($"{role}_encoder", runner.encoderOnnxBytes, ".onnx", assetsRoot, assets);
        var decoder = CaptureAsset($"{role}_decoder", runner.decoderOnnxBytes, ".onnx", assetsRoot, assets);
        var fused = CaptureAsset($"{role}_fused", runner.fusedOnnxBytes, ".onnx", assetsRoot, assets);
        var motionJointPosition = CaptureAsset(
            $"{role}_motion_joint_position",
            runner.motionJointPosCsv,
            ".csv",
            assetsRoot,
            assets);
        var motionJointVelocity = CaptureAsset(
            $"{role}_motion_joint_velocity",
            runner.motionJointVelCsv,
            ".csv",
            assetsRoot,
            assets);
        var motionBodyQuaternion = CaptureAsset(
            $"{role}_motion_body_quaternion",
            runner.motionBodyQuatCsv,
            ".csv",
            assetsRoot,
            assets);

        var modelMode = fused is not null
            ? "fused"
            : encoder is not null && decoder is not null
                ? "split_encoder_decoder"
                : runner.initComplete && runner.actionsReady
                    ? "initialized_runtime_model_bytes_unavailable"
                    : "unresolved";
        var runtimeConfig = CaptureRuntimeConfig(runner);
        var runtimeConfigCanonical = runtimeConfig is null
            ? null
            : CaptureCanonicalRuntimeConfig(
                role,
                runtimeConfig,
                assetsRoot,
                assets);
        var rawConfigComparison = runner.configJson is null
            ? new RuntimeRawConfigComparison("raw_config_asset_unavailable", null)
            : runtimeConfig is null
                ? new RuntimeRawConfigComparison("live_config_unavailable", null)
                : G1RuntimePolicyCaptureContract.CompareRawConfigRecoveredFields(
                    CopyAssetBytes(runner.configJson, $"{role}_config"),
                    runtimeConfig);
        return new RuntimeRunnerManifest(
            role,
            runner.name,
            runner.config?.schema_version,
            runner.config?.robot_name,
            runner.config?.timing?.physics_dt,
            runner.config?.timing?.controller_rate_hz,
            runner.config?.action?.clip_actions,
            runner.config?.encoder?.mode,
            runner.config?.encoder?.token_dimension,
            runner.config?.encoder?.input_tensor_name,
            runner.config?.encoder?.output_tensor_name,
            modelMode,
            runtimeConfig,
            runtimeConfig is null
                ? null
                : G1RuntimePolicyCaptureContract.CanonicalConfigSchema,
            runtimeConfig is null
                ? null
                : G1RuntimePolicyCaptureContract.CanonicalConfigSource,
            runtimeConfigCanonical,
            rawConfigComparison,
            runner.initComplete,
            runner.actionsReady,
            runner.numJoints,
            runner.encoderObsSize,
            runner.decoderObsSize,
            runtimeConfig is null ? null : runner.TokenDimension,
            runner.motionDofs,
            runner._activeEncoderMode,
            runner.commandLpfCutoffHz,
            runner.motionFrameIdx,
            config,
            encoder,
            decoder,
            fused,
            motionJointPosition,
            motionJointVelocity,
            motionBodyQuaternion);
    }

    private static RuntimeSonicConfigSnapshot? CaptureRuntimeConfig(
        SonicPolicyRunner runner)
    {
        var config = runner.config;
        if (config is null)
            return null;

        RuntimeJointConfigSnapshot[]? joints = null;
        if (config.joints is not null)
        {
            var captured = new List<RuntimeJointConfigSnapshot>();
            for (var index = 0; index < config.joints.Length; index++)
            {
                var joint = config.joints[index];
                if (joint is null)
                    throw new InvalidDataException($"null_runtime_joint_config:{index}");
                captured.Add(new RuntimeJointConfigSnapshot(
                    joint.name,
                    joint.default_pos,
                    joint.kp,
                    joint.kd,
                    joint.effort_limit,
                    joint.action_scale));
            }
            joints = captured.ToArray();
        }

        RuntimeEncoderObservationSnapshot[]? encoderObservations = null;
        if (config.encoder?.observations is not null)
        {
            var captured = new List<RuntimeEncoderObservationSnapshot>();
            for (var index = 0; index < config.encoder.observations.Length; index++)
            {
                var observation = config.encoder.observations[index];
                if (observation is null)
                    throw new InvalidDataException($"null_runtime_encoder_observation:{index}");
                captured.Add(new RuntimeEncoderObservationSnapshot(
                    observation.name,
                    observation.dim,
                    observation.active_modes is null
                        ? null
                        : observation.active_modes.ToArray()));
            }
            encoderObservations = captured.ToArray();
        }

        RuntimeDecoderObservationSnapshot[]? decoderObservations = null;
        if (config.decoder?.observations is not null)
        {
            var captured = new List<RuntimeDecoderObservationSnapshot>();
            for (var index = 0; index < config.decoder.observations.Length; index++)
            {
                var observation = config.decoder.observations[index];
                if (observation is null)
                    throw new InvalidDataException($"null_runtime_decoder_observation:{index}");
                captured.Add(new RuntimeDecoderObservationSnapshot(
                    observation.name,
                    observation.type,
                    observation.dim,
                    observation.num_frames,
                    observation.step,
                    observation.dim_per_frame));
            }
            decoderObservations = captured.ToArray();
        }

        RuntimeVrTrackingPointSnapshot[]? vrTrackingPoints = null;
        if (config.vr_tracking?.points is not null)
        {
            var captured = new List<RuntimeVrTrackingPointSnapshot>();
            for (var index = 0; index < config.vr_tracking.points.Length; index++)
            {
                var point = config.vr_tracking.points[index];
                if (point is null)
                    throw new InvalidDataException($"null_runtime_vr_tracking_point:{index}");
                captured.Add(new RuntimeVrTrackingPointSnapshot(
                    point.name,
                    point.xr_node,
                    point.axis_alignment?.ToArray(),
                    point.offset?.ToArray(),
                    point.robot_neutral_pos?.ToArray(),
                    point.robot_neutral_rot?.ToArray()));
            }
            vrTrackingPoints = captured.ToArray();
        }

        return new RuntimeSonicConfigSnapshot(
            config.schema_version,
            config.robot_name,
            config.timing is null
                ? null
                : new RuntimeTimingConfigSnapshot(
                    config.timing.physics_dt,
                    config.timing.controller_rate_hz),
            config.action is null
                ? null
                : new RuntimeActionConfigSnapshot(config.action.clip_actions),
            config.encoder is null
                ? null
                : new RuntimeEncoderConfigSnapshot(
                    config.encoder.mode,
                    config.encoder.token_dimension,
                    config.encoder.input_tensor_name,
                    config.encoder.output_tensor_name,
                    config.encoder.mode_encoding,
                    encoderObservations),
            config.decoder is null
                ? null
                : new RuntimeDecoderConfigSnapshot(
                    config.decoder.input_tensor_name,
                    config.decoder.output_tensor_name,
                    decoderObservations),
            config.vr_tracking is null
                ? null
                : new RuntimeVrTrackingConfigSnapshot(vrTrackingPoints),
            config.isaaclab_to_mujoco?.ToArray(),
            config.mujoco_to_isaaclab?.ToArray(),
            joints);
    }

    private static RuntimeClipManifest CaptureClip(
        string role,
        MocapClipConfig? clip,
        string assetsRoot,
        IDictionary<string, RuntimeAssetFile> assets)
    {
        if (clip is null)
            return RuntimeClipManifest.Unavailable(role);

        var impacts = new List<RuntimeImpactManifest>();
        if (clip.impactEvents is not null)
        {
            for (var index = 0; index < clip.impactEvents.Count; index++)
            {
                var impact = clip.impactEvents[index];
                if (impact is null)
                    continue;
                impacts.Add(new RuntimeImpactManifest(
                    impact.impactTime,
                    impact.leadTime,
                    impact.releaseTime,
                    impact.gainBoost,
                    (int)impact.limb,
                    impact.limb.ToString()));
            }
        }

        return new RuntimeClipManifest(
            role,
            true,
            clip.name,
            clip.displayName,
            clip.policyProfile,
            clip.startFrame,
            clip.endFrame,
            clip.mirror,
            clip.playbackSpeed,
            clip.loop,
            clip.yawBlend,
            clip.yawForgiveness,
            clip.impactYawForgiveness,
            clip.impactForgivenessDuration,
            clip.blendInTime,
            clip.blendOutTime,
            CaptureAsset($"{role}_npz", clip.npzFile, ".npz", assetsRoot, assets),
            impacts.ToArray());
    }

    private static RuntimeAssetReference? CaptureAsset(
        string role,
        TextAsset? asset,
        string extension,
        string assetsRoot,
        IDictionary<string, RuntimeAssetFile> assets)
    {
        if (asset is null)
            return null;

        return CaptureBytes(
            role,
            asset.name,
            CopyAssetBytes(asset, role),
            extension,
            assetsRoot,
            assets);
    }

    private static RuntimeAssetReference CaptureCanonicalRuntimeConfig(
        string role,
        RuntimeSonicConfigSnapshot config,
        string assetsRoot,
        IDictionary<string, RuntimeAssetFile> assets) =>
        CaptureBytes(
            $"{role}_runtime_config_canonical",
            G1RuntimePolicyCaptureContract.CanonicalConfigSource,
            G1RuntimePolicyCaptureContract.CanonicalRuntimeConfigBytes(config),
            ".json",
            assetsRoot,
            assets);

    private static byte[] CopyAssetBytes(TextAsset asset, string role)
    {
        var il2CppBytes = asset.bytes;
        if (il2CppBytes is null || il2CppBytes.Length == 0)
            throw new InvalidDataException($"empty_runtime_asset:{role}");
        var bytes = new byte[il2CppBytes.Length];
        for (var index = 0; index < bytes.Length; index++)
            bytes[index] = il2CppBytes[index];
        return bytes;
    }

    private static RuntimeAssetReference CaptureBytes(
        string role,
        string runtimeName,
        byte[] bytes,
        string extension,
        string assetsRoot,
        IDictionary<string, RuntimeAssetFile> assets)
    {
        var sha256 = Hash(bytes);
        var relativePath = Path.Combine("assets", sha256 + extension).Replace('\\', '/');
        var fullPath = Path.Combine(assetsRoot, sha256 + extension);
        WriteOrVerify(fullPath, bytes);
        if (!assets.ContainsKey(sha256))
        {
            assets.Add(sha256, new RuntimeAssetFile(
                sha256,
                bytes.LongLength,
                relativePath,
                extension));
        }

        return new RuntimeAssetReference(
            role,
            runtimeName,
            sha256,
            bytes.LongLength,
            relativePath);
    }

    private static void WriteOrVerify(string path, byte[] bytes)
    {
        if (File.Exists(path))
        {
            var info = new FileInfo(path);
            if (info.Length != bytes.LongLength ||
                !string.Equals(HashFile(path), Hash(bytes), StringComparison.Ordinal))
            {
                throw new InvalidDataException($"existing_capture_mismatch:{Path.GetFileName(path)}");
            }
            return;
        }

        var parent = Path.GetDirectoryName(path) ??
            throw new InvalidDataException("capture_parent_unavailable");
        Directory.CreateDirectory(parent);
        var partial = Path.Combine(parent, $".{Path.GetFileName(path)}.partial-{Guid.NewGuid():N}");
        try
        {
            using (var stream = new FileStream(
                       partial,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       1024 * 1024,
                       FileOptions.WriteThrough))
            {
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush(flushToDisk: true);
            }
            File.Move(partial, path, overwrite: false);
        }
        finally
        {
            if (File.Exists(partial))
                File.Delete(partial);
        }
    }

    private static string? PolicyAssetIdentity(RuntimeRunnerManifest runner)
    {
        string[] values;
        if (runner.Fused is not null)
        {
            values = new[] { "fused", runner.Fused.Sha256 };
        }
        else if (runner.Encoder is not null && runner.Decoder is not null)
        {
            values = new[]
            {
                "split_encoder_decoder",
                runner.Encoder.Sha256,
                runner.Decoder.Sha256,
            };
        }
        else
        {
            return null;
        }
        return Hash(Encoding.UTF8.GetBytes(string.Join("\n", values)));
    }

    private static string Hash(byte[] bytes) =>
        Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private static string HashFile(string path)
    {
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read);
        using var sha256 = SHA256.Create();
        return Convert.ToHexString(sha256.ComputeHash(stream)).ToLowerInvariant();
    }

    private static bool IsSha256(string? value) =>
        value is { Length: 64 } && value.All(character =>
            character is >= '0' and <= '9' or >= 'a' and <= 'f');
}

internal sealed record RuntimePolicyManifest(
    string Schema,
    string GameAssemblySha256,
    string GlobalMetadataSha256,
    string SharedAssets0Sha256,
    RuntimeRunnerManifest LocalRunner,
    RuntimeRunnerManifest OpponentRunner,
    bool? LocalAndOpponentPolicyAssetsExact,
    string? LocalPolicyIdentitySha256,
    string? OpponentPolicyIdentitySha256,
    bool LocalAndOpponentRuntimeConfigExact,
    string LocalRuntimeConfigIdentitySha256,
    string OpponentRuntimeConfigIdentitySha256,
    string RobotConfigName,
    bool LocomotionTransitionSettle,
    float TransitionSettlePlanarSpeed,
    float TransitionSettleYawRate,
    RuntimeClipManifest[] LocomotionClips,
    RuntimeClipManifest[] RecoveryClips,
    RuntimeClipManifest[] Moves,
    RuntimeAssetFile[] Assets);

internal sealed record RuntimeRunnerManifest(
    string Role,
    string RuntimeName,
    string? ConfigSchemaVersion,
    string? ConfigRobotName,
    float? PhysicsDt,
    int? ControllerRateHz,
    float? ClipActions,
    int? EncoderMode,
    int? TokenDimension,
    string? EncoderInputTensorName,
    string? EncoderOutputTensorName,
    string ModelMode,
    RuntimeSonicConfigSnapshot? RuntimeConfig,
    string? RuntimeConfigCanonicalSchema,
    string? RuntimeConfigCanonicalSource,
    RuntimeAssetReference? RuntimeConfigCanonical,
    RuntimeRawConfigComparison RawConfigComparison,
    bool RuntimeInitialized,
    bool ActionsReady,
    int NumJoints,
    int EncoderObservationSize,
    int DecoderObservationSize,
    int? LiveTokenDimension,
    int MotionDofs,
    int ActiveEncoderMode,
    float CommandLpfCutoffHz,
    int MotionFrameIndex,
    RuntimeAssetReference? Config,
    RuntimeAssetReference? Encoder,
    RuntimeAssetReference? Decoder,
    RuntimeAssetReference? Fused,
    RuntimeAssetReference? MotionJointPositionCsv,
    RuntimeAssetReference? MotionJointVelocityCsv,
    RuntimeAssetReference? MotionBodyQuaternionCsv);

internal sealed record RuntimeAssetReference(
    string Role,
    string RuntimeName,
    string Sha256,
    long Bytes,
    string RelativePath);

internal sealed record RuntimeAssetFile(
    string Sha256,
    long Bytes,
    string RelativePath,
    string Extension);

internal sealed record RuntimeClipManifest(
    string Role,
    bool Available,
    string? RuntimeName,
    string? DisplayName,
    string? PolicyProfile,
    int? StartFrame,
    int? EndFrame,
    bool? Mirror,
    float? PlaybackSpeed,
    bool? Loop,
    float? YawBlend,
    float? YawForgiveness,
    float? ImpactYawForgiveness,
    float? ImpactForgivenessDuration,
    float? BlendInTime,
    float? BlendOutTime,
    RuntimeAssetReference? Npz,
    RuntimeImpactManifest[] Impacts)
{
    internal static RuntimeClipManifest Unavailable(string role) =>
        new(
            role,
            false,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            Array.Empty<RuntimeImpactManifest>());
}

internal sealed record RuntimeImpactManifest(
    float ImpactTimeSeconds,
    float LeadTimeSeconds,
    float ReleaseTimeSeconds,
    float GainBoost,
    int LimbValue,
    string LimbName);
