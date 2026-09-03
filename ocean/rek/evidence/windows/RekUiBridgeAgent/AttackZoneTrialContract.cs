using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace RekUiBridgeAgent;

internal static class AttackZoneTrialContract
{
    internal const string Schema = "rek.attack_zone_trial.v1";
    internal const string ScheduleSchema = "rek.attack_zone_schedule.v1";
    internal const string AuthorityScope =
        "client_request_edges_and_local_observations_only";
    internal const string AuthorityCaveat =
        "client request edge and local observations only; server acceptance, authoritative execution, and causal hit attribution are unknown";
    internal const string RequiredIsolationProof =
        "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98";
    internal const string RandomizationAlgorithm =
        "sha256_counter_fisher_yates_rejection_v1";
    internal const string BearingDefinition =
        "signed_degrees_in_range_minus_180_inclusive_to_180_exclusive_about_world_positive_y_from_measured_local_root_positive_z_to_local_to_opponent_xz";
    internal const string AcquisitionYawRule =
        "RobotInputController.VelocityCommand.z=clamp(-(measured_bearing_deg-target_bin_center_deg)/35,-1,1); positive z increases target bearing";
    internal const string ExpectedGameAssemblySha256 =
        "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
    internal const string ExpectedGlobalMetadataSha256 =
        "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
    internal const string ExpectedRecorderVersion = "0.6.1";
    internal const string ExpectedRecorderPluginSha256 =
        "24cbea0a149589b71c093e989f43b8dac4862e73d103c323f0f9472a38355e0b";
    internal const int UnityFixedRateHz = 500;
    internal const int ControlRateHz = 50;
    internal const int FixedSubstepsPerControlTick = UnityFixedRateHz / ControlRateHz;
    internal const int TelemetryIntervalTicks = 1;
    internal const int SettleTicks = 15;
    internal const int AcquisitionTimeoutTicks = ControlRateHz * 10;
    internal const int RequestStartTimeoutTicks = ControlRateHz * 5;
    internal const int CompletionTimeoutTicks = ControlRateHz * 15;
    internal const int MinimumIndependentRunsPerCell = 5;
    internal const int RecoveryReadyTicks = 15;
    internal const double BearingErrorLimitDegrees = 3.0;
    internal const double PlanarSpeedLimitMetersPerSecond = 0.15;
    internal const double YawRateLimitRadiansPerSecond = 0.30;
    internal const double QuaternionNormTolerance = 0.001;
    internal const double UnityFixedIntervalToleranceSeconds = 0.000001;
    internal const double MinimumStopwatchIntervalSeconds = 0.0125;
    internal const double MaximumStopwatchIntervalSeconds = 0.0400;
    internal const float ApproachForwardCommand = 0.8f;
    internal const float ApproachBackoffCommand = -0.25f;
    internal const float MaximumYawCommand = 1f;
    internal const float YawCommandScaleDegrees = 35f;
    internal const string ExpectedSha256 =
        "1c55900c766aac8cf3382c389b297be6324b3ca19c4a5de6d25f17a7ee217278";

    internal static readonly AttackZoneBin[] DistanceBins =
    {
        new("d00", 0.25, 0.35, true, false),
        new("d01", 0.35, 0.45, true, false),
        new("d02", 0.45, 0.5180000126361847, true, true),
        new("d03", 0.5180000126361847, 0.60, false, false),
        new("d04", 0.60, 0.75, true, false),
        new("d05", 0.75, 0.90, true, false),
        new("d06", 0.90, 1.10, true, false),
        new("d07", 1.10, 1.50, true, false),
        new("d08", 1.50, 2.00, true, true),
    };

    internal static readonly AttackZoneBin[] BearingBins =
    {
        new("b00", -180.0, -90.0, true, false),
        new("b01", -90.0, -60.0, true, false),
        new("b02", -60.0, -35.0, true, false),
        new("b03", -35.0, -20.0, true, false),
        new("b04", -20.0, -5.0, true, false),
        new("b05", -5.0, 5.0, true, true),
        new("b06", 5.0, 20.0, false, true),
        new("b07", 20.0, 35.0, false, true),
        new("b08", 35.0, 60.0, false, true),
        new("b09", 60.0, 90.0, false, true),
        new("b10", 90.0, 180.0, false, false),
    };

    internal static readonly string[] RequiredEvidenceEvents =
    {
        "target_requested",
        "acquisition_sample",
        "target_acquired",
        "target_not_acquired_unresolved",
        "neutral_command_edge_set",
        "neutral_request_method_returned",
        "local_command_edge_set",
        "client_request_method_returned",
        "local_motion_start_observed",
        "action_sample",
        "local_motion_completion_and_readiness_observed",
        "configured_asset_marker_projected",
        "raw_rek_hit_observed",
        "round_score_delta_observed",
        "fall_observed",
        "recovery_request_observed",
        "recovery_state_observed",
        "trial_censored",
        "trial_interrupted",
    };

    internal static readonly string[] RequiredRecoveryRequestKinds =
    {
        "Dampen",
        "Straighten",
        "GetUpProne",
        "GetUpSupine",
        "fault_estop_toggle_on",
        "fault_estop_toggle_off",
    };

    internal static readonly string CanonicalJson = BuildCanonicalJson();

    internal static string ComputeSha256() => HashUtf8(CanonicalJson);

    internal static bool ValidateEmbeddedContract(out string reason)
    {
        if (!string.Equals(ComputeSha256(), ExpectedSha256, StringComparison.Ordinal))
        {
            reason = "attack_zone_contract_sha256_mismatch";
            return false;
        }
        var fixedValues = new[]
        {
            UnityFixedRateHz,
            ControlRateHz,
            TelemetryIntervalTicks,
            SettleTicks,
            AcquisitionTimeoutTicks,
            MinimumIndependentRunsPerCell,
        };
        if (fixedValues[0] % fixedValues[1] != 0 || fixedValues[2] != 1 ||
            fixedValues[3] != 15 || fixedValues[4] != 500 || fixedValues[5] < 5)
        {
            reason = "attack_zone_contract_rate_or_quota_invalid";
            return false;
        }
        if (DistanceBins.Select(value => value.Id).Distinct(StringComparer.Ordinal).Count() !=
                DistanceBins.Length ||
            BearingBins.Select(value => value.Id).Distinct(StringComparer.Ordinal).Count() !=
                BearingBins.Length ||
            DistanceBins.Any(value => !value.IsValid) ||
            BearingBins.Any(value => !value.IsValid))
        {
            reason = "attack_zone_contract_bins_invalid";
            return false;
        }
        var allowedMoves = ContinuousBotControllerContract.Attacks
            .Select(value => value.MoveIndex)
            .ToArray();
        if (!allowedMoves.SequenceEqual(new[] { 2, 3, 4, 5, 9, 10 }))
        {
            reason = "attack_zone_contract_move_profiles_invalid";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    internal static bool TryParseTarget(
        JsonElement element,
        out AttackZoneTrialTarget target,
        out string reason)
    {
        target = null!;
        reason = "attack_zone_target_invalid";
        try
        {
            if (!HasExactProperties(element, new[]
                {
                    "attack_zone_trial_schema",
                    "protocol_sha256",
                    "controller_contract_sha256",
                    "game_assembly_sha256",
                    "global_metadata_sha256",
                    "recorder_version",
                    "recorder_plugin_sha256",
                    "schedule_schema",
                    "schedule_sha256",
                    "randomization_algorithm",
                    "randomization_seed_hex",
                    "schedule_ordinal",
                    "independent_run_id",
                    "independent_run_ordinal",
                    "required_independent_runs_per_cell",
                    "session_identity_sha256",
                    "round_identity_sha256",
                    "trial_id",
                    "action_sequence",
                    "move_index",
                    "serialized_asset_sha256",
                    "distance_bin",
                    "bearing_bin",
                    "acquisition_timeout_ticks",
                }))
            {
                reason = "invalid_attack_zone_target_shape";
                return false;
            }

            if (!TryParseRequestedBin(
                    element.GetProperty("distance_bin"),
                    "lower_m",
                    "upper_m",
                    "center_m",
                    out var distanceBin) ||
                !TryParseRequestedBin(
                    element.GetProperty("bearing_bin"),
                    "lower_deg",
                    "upper_deg",
                    "center_deg",
                    out var bearingBin))
            {
                reason = "invalid_attack_zone_target_bin_shape";
                return false;
            }

            target = new AttackZoneTrialTarget(
                RequiredString(element, "attack_zone_trial_schema"),
                RequiredString(element, "protocol_sha256"),
                RequiredString(element, "controller_contract_sha256"),
                RequiredString(element, "game_assembly_sha256"),
                RequiredString(element, "global_metadata_sha256"),
                RequiredString(element, "recorder_version"),
                RequiredString(element, "recorder_plugin_sha256"),
                RequiredString(element, "schedule_schema"),
                RequiredString(element, "schedule_sha256"),
                RequiredString(element, "randomization_algorithm"),
                RequiredString(element, "randomization_seed_hex"),
                RequiredInt32(element, "schedule_ordinal"),
                RequiredString(element, "independent_run_id"),
                RequiredInt32(element, "independent_run_ordinal"),
                RequiredInt32(element, "required_independent_runs_per_cell"),
                RequiredString(element, "session_identity_sha256"),
                RequiredString(element, "round_identity_sha256"),
                RequiredString(element, "trial_id"),
                RequiredInt32(element, "action_sequence"),
                RequiredInt32(element, "move_index"),
                RequiredString(element, "serialized_asset_sha256"),
                distanceBin,
                bearingBin,
                RequiredInt32(element, "acquisition_timeout_ticks"));
            return true;
        }
        catch (InvalidOperationException)
        {
            reason = "invalid_attack_zone_target_value_type";
            return false;
        }
        catch (FormatException)
        {
            reason = "invalid_attack_zone_target_value_format";
            return false;
        }
        catch (OverflowException)
        {
            reason = "invalid_attack_zone_target_value_range";
            return false;
        }
    }

    internal static bool TryValidateTarget(
        AttackZoneTrialTarget? target,
        out AttackZoneValidatedTarget validated,
        out string reason)
    {
        validated = null!;
        if (target is null)
        {
            reason = "attack_zone_target_missing";
            return false;
        }
        if (!string.Equals(target.Schema, Schema, StringComparison.Ordinal))
        {
            reason = "attack_zone_target_schema_mismatch";
            return false;
        }
        if (!string.Equals(target.ProtocolSha256, ExpectedSha256, StringComparison.Ordinal))
        {
            reason = "attack_zone_target_protocol_sha256_mismatch";
            return false;
        }
        if (!string.Equals(
                target.ControllerContractSha256,
                ContinuousBotControllerContract.ExpectedSha256,
                StringComparison.Ordinal))
        {
            reason = "attack_zone_target_controller_contract_sha256_mismatch";
            return false;
        }
        if (!string.Equals(
                target.GameAssemblySha256,
                ExpectedGameAssemblySha256,
                StringComparison.Ordinal) ||
            !string.Equals(
                target.GlobalMetadataSha256,
                ExpectedGlobalMetadataSha256,
                StringComparison.Ordinal))
        {
            reason = "attack_zone_target_build_sha256_mismatch";
            return false;
        }
        if (!string.Equals(
                target.RecorderVersion,
                ExpectedRecorderVersion,
                StringComparison.Ordinal) ||
            !string.Equals(
                target.RecorderPluginSha256,
                ExpectedRecorderPluginSha256,
                StringComparison.Ordinal))
        {
            reason = "attack_zone_target_recorder_pin_mismatch";
            return false;
        }
        if (!string.Equals(target.ScheduleSchema, ScheduleSchema, StringComparison.Ordinal) ||
            !ValidSha256(target.ScheduleSha256) ||
            !string.Equals(
                target.RandomizationAlgorithm,
                RandomizationAlgorithm,
                StringComparison.Ordinal) ||
            !ValidLowerHex(target.RandomizationSeedHex, 64) ||
            target.ScheduleOrdinal < 0 || target.IndependentRunOrdinal < 0 ||
            target.RequiredIndependentRunsPerCell != MinimumIndependentRunsPerCell)
        {
            reason = "attack_zone_target_schedule_identity_invalid";
            return false;
        }
        if (!ValidIdentifier(target.IndependentRunId) || !ValidIdentifier(target.TrialId) ||
            !ValidSha256(target.SessionIdentitySha256) ||
            !ValidSha256(target.RoundIdentitySha256) || target.ActionSequence <= 0)
        {
            reason = "attack_zone_target_runtime_identity_invalid";
            return false;
        }
        var attack = ContinuousBotControllerContract.Attacks.FirstOrDefault(
            value => value.MoveIndex == target.MoveIndex);
        if (attack is null || !string.Equals(
                attack.SerializedAssetSha256,
                target.SerializedAssetSha256,
                StringComparison.Ordinal))
        {
            reason = "attack_zone_target_move_profile_mismatch";
            return false;
        }
        if (!TryMatchRequestedBin(target.DistanceBin, DistanceBins, out var distanceBin))
        {
            reason = "attack_zone_target_distance_bin_mismatch";
            return false;
        }
        if (!TryMatchRequestedBin(target.BearingBin, BearingBins, out var bearingBin))
        {
            reason = "attack_zone_target_bearing_bin_mismatch";
            return false;
        }
        if (target.AcquisitionTimeoutTicks != AcquisitionTimeoutTicks)
        {
            reason = "attack_zone_target_acquisition_timeout_mismatch";
            return false;
        }
        validated = new AttackZoneValidatedTarget(target, attack, distanceBin, bearingBin);
        reason = string.Empty;
        return true;
    }

    internal static AttackZoneScopeValidation ValidateScope(
        AttackZoneScopeObservation? scope,
        AttackZoneValidatedTarget target)
    {
        if (scope is null)
            return new(false, "attack_zone_scope_missing");
        if (!scope.IsolatedSparkVerified || !string.Equals(
                scope.IsolationProof,
                RequiredIsolationProof,
                StringComparison.Ordinal))
        {
            return new(false, "verified_isolated_spark_scope_required");
        }
        if (!scope.ExclusiveLeaseHeld || scope.LeaseConnectionId <= 0)
            return new(false, "exclusive_local_control_lease_required");
        if (scope.GlobalInputUsed)
            return new(false, "global_input_observed");
        if (!scope.SemanticCommandSurfaceAvailable)
            return new(false, "semantic_command_surface_unavailable");
        if (!scope.PrivateSessionProven || scope.Ranked || !scope.ExactSparringBotOne ||
            !scope.ActiveRound || scope.FighterCount != 2)
        {
            return new(false, "exact_private_unranked_sparring_bot_1_round_not_proven");
        }
        if (!scope.LocalSemanticT800 || !scope.LocalRuntimeExactT800)
            return new(false, "local_exact_t800_identity_not_proven");
        if (!scope.OpponentRuntimeExactT800 ||
            !ValidSha256(scope.OpponentRuntimeIdentitySha256))
        {
            return new(false, "opponent_measured_runtime_t800_identity_not_proven");
        }
        if (!scope.BuildHashesMatch || !scope.ControllerContractHashMatch ||
            !scope.RecorderPinMatch || !scope.SendBoundaryPatchesVerified)
        {
            return new(false, "build_contract_or_send_boundary_mismatch");
        }
        if (!string.Equals(
                scope.SessionIdentitySha256,
                target.Request.SessionIdentitySha256,
                StringComparison.Ordinal) ||
            !string.Equals(
                scope.RoundIdentitySha256,
                target.Request.RoundIdentitySha256,
                StringComparison.Ordinal))
        {
            return new(false, "target_runtime_identity_changed");
        }
        return new(true, scope.OpponentSemanticRuntimeMismatch
            ? "scope_proven_opponent_semantic_runtime_mismatch_recorded"
            : "scope_proven");
    }

    internal static AttackZoneAcquisitionDecision DecideAcquisition(
        AttackZoneValidatedTarget target,
        PlanarCombatGeometry geometry)
    {
        var forward = 0f;
        if (geometry.DistanceMeters < target.DistanceBin.CentralLower)
            forward = ApproachBackoffCommand;
        else if (geometry.DistanceMeters > target.DistanceBin.CentralUpper)
            forward = ApproachForwardCommand;

        var bearingError = WrapTo180(
            geometry.LocalBearingToOpponentDegrees - target.BearingBin.Center);
        var yaw = Math.Abs(bearingError) <= BearingErrorLimitDegrees
            ? 0f
            : (float)Math.Clamp(
                -bearingError / YawCommandScaleDegrees,
                -MaximumYawCommand,
                MaximumYawCommand);
        var exactNeutral = SameFloatBits(forward, 0f) && SameFloatBits(yaw, 0f);
        return new AttackZoneAcquisitionDecision(
            forward,
            0f,
            yaw,
            exactNeutral,
            exactNeutral ? "target_geometry_ready_for_neutral_settle" :
                forward != 0f && yaw != 0f ? "drive_distance_and_target_bearing" :
                forward != 0f ? "drive_target_distance" : "drive_target_bearing");
    }

    internal static AttackZoneSampleEvaluation EvaluateSettleSample(
        AttackZoneValidatedTarget target,
        AttackZoneControlObservation sample)
    {
        var clockValid = ValidateClock(sample.Clock);
        var rootsValid = sample.LocalRoot.IsFinite && sample.OpponentRoot.IsFinite;
        var geometry = AttackZoneGeometry.Invalid;
        var geometryValid = rootsValid && TryComputeGeometry(
            sample.LocalRoot,
            sample.OpponentRoot,
            out geometry);
        var distanceCentral = geometryValid &&
            geometry.DistanceMeters >= target.DistanceBin.CentralLower &&
            geometry.DistanceMeters <= target.DistanceBin.CentralUpper;
        var bearingInBin = geometryValid && target.BearingBin.Contains(
            geometry.LocalBearingToOpponentDegrees);
        var bearingError = geometryValid
            ? WrapTo180(
                geometry.LocalBearingToOpponentDegrees - target.BearingBin.Center)
            : double.NaN;
        var bearingErrorPass = geometryValid &&
            Math.Abs(bearingError) <= BearingErrorLimitDegrees;
        var localPlanarSpeed = sample.LocalRoot.PlanarSpeedMetersPerSecond;
        var localYawRate = Math.Abs(sample.LocalRoot.AngularVelocityY);
        var localMotionPass = rootsValid &&
            localPlanarSpeed <= PlanarSpeedLimitMetersPerSecond &&
            localYawRate <= YawRateLimitRadiansPerSecond;
        var noPending = !sample.PendingMove && !sample.PendingSpecial &&
            !sample.PendingEStop;
        var localHealthy = !sample.LocalRoot.Falling && !sample.LocalRoot.Fallen &&
            !sample.LocalRoot.Recovering && !sample.LocalRoot.Dampened &&
            !sample.LocalRoot.Resetting && !sample.LocalRoot.MotorShutdown;
        var opponentHealthy = !sample.OpponentRoot.Falling &&
            !sample.OpponentRoot.Fallen && !sample.OpponentRoot.Recovering &&
            !sample.OpponentRoot.Dampened &&
            !sample.OpponentRoot.Resetting && !sample.OpponentRoot.MotorShutdown;
        var animationValid = sample.LocalAnimation.IsValid &&
            sample.OpponentAnimation.IsValid;
        var classification = geometryValid
            ? ClassifyOpponentMotion(sample.LocalRoot, sample.OpponentRoot, geometry)
            : AttackZoneMotionClassification.Unknown;

        var acquisitionPass = clockValid && geometryValid && animationValid &&
            sample.NeutralRequestMethodReturned && sample.VelocityCommandExactNeutral &&
            sample.LocalActionReady && noPending && localHealthy && opponentHealthy &&
            distanceCentral && bearingInBin && bearingErrorPass && localMotionPass;
        return new AttackZoneSampleEvaluation(
            acquisitionPass,
            clockValid,
            rootsValid,
            geometryValid,
            animationValid,
            sample.NeutralRequestMethodReturned,
            sample.VelocityCommandExactNeutral,
            sample.LocalActionReady,
            noPending,
            localHealthy,
            opponentHealthy,
            distanceCentral,
            bearingInBin,
            bearingErrorPass,
            localMotionPass,
            classification.Stationary,
            bearingError,
            localPlanarSpeed,
            localYawRate,
            geometry,
            classification);
    }

    internal static AttackZoneMotionClassification ClassifyOpponentMotion(
        AttackZoneRootObservation local,
        AttackZoneRootObservation opponent,
        AttackZoneGeometry geometry)
    {
        if (!local.IsFinite || !opponent.IsFinite || !geometry.IsValid)
            return AttackZoneMotionClassification.Unknown;

        var opponentSpeed = opponent.PlanarSpeedMetersPerSecond;
        var opponentYawRate = Math.Abs(opponent.AngularVelocityY);
        var relativeX = opponent.LinearVelocityX - local.LinearVelocityX;
        var relativeZ = opponent.LinearVelocityZ - local.LinearVelocityZ;
        var radialClosing = -(
            relativeX * geometry.LocalToOpponentUnitX +
            relativeZ * geometry.LocalToOpponentUnitZ);
        var tangential = Math.Abs(
            relativeX * geometry.LocalToOpponentUnitZ -
            relativeZ * geometry.LocalToOpponentUnitX);
        var stationary = opponentSpeed <= PlanarSpeedLimitMetersPerSecond &&
            opponentYawRate <= YawRateLimitRadiansPerSecond;

        string motion;
        if (stationary)
        {
            motion = "stationary";
        }
        else
        {
            var closing = radialClosing > PlanarSpeedLimitMetersPerSecond;
            var receding = radialClosing < -PlanarSpeedLimitMetersPerSecond;
            var movingTangentially =
                Math.Abs(radialClosing) <= PlanarSpeedLimitMetersPerSecond &&
                tangential > PlanarSpeedLimitMetersPerSecond;
            var turning = opponentYawRate > YawRateLimitRadiansPerSecond;
            var named = (closing ? 1 : 0) + (receding ? 1 : 0) +
                (movingTangentially ? 1 : 0) + (turning ? 1 : 0);
            motion = named == 1
                ? closing ? "closing" : receding ? "receding" :
                    movingTangentially ? "tangential" : "turning"
                : "compound_or_unknown";
        }

        var absoluteOpponentBearing = Math.Abs(geometry.OpponentBearingToLocalDegrees);
        var facing = absoluteOpponentBearing <= 35.0
            ? "opponent_face_on"
            : absoluteOpponentBearing <= 90.0
                ? "opponent_oblique"
                : "opponent_back_turned";
        return new AttackZoneMotionClassification(
            motion,
            facing,
            stationary,
            opponentSpeed,
            opponentYawRate,
            radialClosing,
            tangential);
    }

    internal static IReadOnlyList<AttackZoneScheduleEntry> BuildRandomizedSchedule(
        string independentRunId,
        int independentRunOrdinal,
        string randomizationSeedHex,
        int repetitionsPerCell = 1)
    {
        if (!ValidIdentifier(independentRunId))
            throw new ArgumentException("invalid independent run id", nameof(independentRunId));
        if (independentRunOrdinal < 0)
            throw new ArgumentOutOfRangeException(nameof(independentRunOrdinal));
        if (!ValidLowerHex(randomizationSeedHex, 64))
            throw new ArgumentException("seed must be 64 lowercase hexadecimal characters", nameof(randomizationSeedHex));
        if (repetitionsPerCell <= 0)
            throw new ArgumentOutOfRangeException(nameof(repetitionsPerCell));

        var natural = new List<AttackZoneScheduleEntry>();
        foreach (var attack in ContinuousBotControllerContract.Attacks)
        {
            foreach (var distance in DistanceBins)
            {
                foreach (var bearing in BearingBins)
                {
                    for (var repetition = 0; repetition < repetitionsPerCell; repetition++)
                    {
                        natural.Add(new AttackZoneScheduleEntry(
                            -1,
                            independentRunId,
                            independentRunOrdinal,
                            repetition,
                            attack.MoveIndex,
                            attack.SerializedAssetSha256,
                            distance,
                            bearing));
                    }
                }
            }
        }

        var random = new AttackZoneDeterministicRandom(randomizationSeedHex);
        for (var index = natural.Count - 1; index > 0; index--)
        {
            var selected = random.NextIndex(index + 1);
            (natural[index], natural[selected]) = (natural[selected], natural[index]);
        }
        for (var index = 0; index < natural.Count; index++)
            natural[index] = natural[index] with { ScheduleOrdinal = index };
        return natural;
    }

    internal static string BuildScheduleCanonicalJson(
        string independentRunId,
        int independentRunOrdinal,
        string randomizationSeedHex,
        int repetitionsPerCell,
        IReadOnlyList<AttackZoneScheduleEntry> entries) => JsonSerializer.Serialize(new
        {
            attack_zone_trial_schema = Schema,
            protocol_sha256 = ExpectedSha256,
            schedule_schema = ScheduleSchema,
            randomization_algorithm = RandomizationAlgorithm,
            randomization_seed_hex = randomizationSeedHex,
            independent_run_id = independentRunId,
            independent_run_ordinal = independentRunOrdinal,
            repetitions_per_cell = repetitionsPerCell,
            required_independent_runs_per_cell = MinimumIndependentRunsPerCell,
            entries = entries.Select(entry => new
            {
                schedule_ordinal = entry.ScheduleOrdinal,
                repetition_within_run = entry.RepetitionWithinRun,
                move_index = entry.MoveIndex,
                serialized_asset_sha256 = entry.SerializedAssetSha256,
                distance_bin = DistanceBinPayload(entry.DistanceBin),
                bearing_bin = BearingBinPayload(entry.BearingBin),
            }).ToArray(),
        });

    internal static string SerializeTarget(AttackZoneTrialTarget target) =>
        JsonSerializer.Serialize(TargetPayload(target));

    internal static string ComputeScheduleSha256(
        string independentRunId,
        int independentRunOrdinal,
        string randomizationSeedHex,
        int repetitionsPerCell,
        IReadOnlyList<AttackZoneScheduleEntry> entries) => HashUtf8(
            BuildScheduleCanonicalJson(
                independentRunId,
                independentRunOrdinal,
                randomizationSeedHex,
                repetitionsPerCell,
                entries));

    internal static AttackZoneCoverageValidation ValidateIndependentCoverage(
        IEnumerable<AttackZoneScheduleEntry> entries)
    {
        var coverage = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var attack in ContinuousBotControllerContract.Attacks)
        foreach (var distance in DistanceBins)
        foreach (var bearing in BearingBins)
            coverage[CellKey(attack.MoveIndex, distance.Id, bearing.Id)] =
                new HashSet<string>(StringComparer.Ordinal);

        foreach (var entry in entries)
        {
            var key = CellKey(entry.MoveIndex, entry.DistanceBin.Id, entry.BearingBin.Id);
            if (coverage.TryGetValue(key, out var runs) && ValidIdentifier(entry.IndependentRunId))
                runs.Add(entry.IndependentRunId);
        }
        var missing = coverage
            .Where(value => value.Value.Count < MinimumIndependentRunsPerCell)
            .Select(value => new AttackZoneCellCoverage(
                value.Key,
                value.Value.Count,
                MinimumIndependentRunsPerCell))
            .OrderBy(value => value.CellKey, StringComparer.Ordinal)
            .ToArray();
        return new AttackZoneCoverageValidation(missing.Length == 0, missing);
    }

    internal static AttackZoneTrialTarget CreateTarget(
        AttackZoneScheduleEntry entry,
        string scheduleSha256,
        string randomizationSeedHex,
        string sessionIdentitySha256,
        string roundIdentitySha256,
        string trialId,
        int actionSequence) => new(
            Schema,
            ExpectedSha256,
            ContinuousBotControllerContract.ExpectedSha256,
            ExpectedGameAssemblySha256,
            ExpectedGlobalMetadataSha256,
            ExpectedRecorderVersion,
            ExpectedRecorderPluginSha256,
            ScheduleSchema,
            scheduleSha256,
            RandomizationAlgorithm,
            randomizationSeedHex,
            entry.ScheduleOrdinal,
            entry.IndependentRunId,
            entry.IndependentRunOrdinal,
            MinimumIndependentRunsPerCell,
            sessionIdentitySha256,
            roundIdentitySha256,
            trialId,
            actionSequence,
            entry.MoveIndex,
            entry.SerializedAssetSha256,
            AttackZoneRequestedBin.From(entry.DistanceBin),
            AttackZoneRequestedBin.From(entry.BearingBin),
            AcquisitionTimeoutTicks);

    internal static bool TryComputeGeometry(
        AttackZoneRootObservation local,
        AttackZoneRootObservation opponent,
        out AttackZoneGeometry geometry)
    {
        geometry = AttackZoneGeometry.Invalid;
        if (!local.IsFinite || !opponent.IsFinite ||
            !TryRotateLocalPlusZ(local, out var localForwardX, out var localForwardZ) ||
            !TryRotateLocalPlusZ(opponent, out var opponentForwardX, out var opponentForwardZ))
        {
            return false;
        }
        var deltaX = opponent.PositionX - local.PositionX;
        var deltaZ = opponent.PositionZ - local.PositionZ;
        var distanceSquared = deltaX * deltaX + deltaZ * deltaZ;
        if (!double.IsFinite(distanceSquared) || distanceSquared <= 1e-12)
            return false;
        var distance = Math.Sqrt(distanceSquared);
        var toOpponentX = deltaX / distance;
        var toOpponentZ = deltaZ / distance;
        var localBearing = SignedPlanarAngleDegrees(
            localForwardX,
            localForwardZ,
            toOpponentX,
            toOpponentZ);
        var opponentBearing = SignedPlanarAngleDegrees(
            opponentForwardX,
            opponentForwardZ,
            -toOpponentX,
            -toOpponentZ);
        if (!AllFinite(distance, localBearing, opponentBearing, toOpponentX, toOpponentZ))
            return false;
        geometry = new AttackZoneGeometry(
            true,
            distance,
            localBearing,
            opponentBearing,
            toOpponentX,
            toOpponentZ);
        return true;
    }

    internal static double WrapTo180(double degrees)
    {
        if (!double.IsFinite(degrees))
            return double.NaN;
        var wrapped = degrees % 360.0;
        if (wrapped >= 180.0)
            wrapped -= 360.0;
        if (wrapped < -180.0)
            wrapped += 360.0;
        return wrapped;
    }

    internal static bool ValidateClock(AttackZoneClock clock) =>
        clock.StopwatchTimestampTicks >= 0 && clock.StopwatchFrequencyHz > 0 &&
        DateTimeOffset.TryParse(clock.Utc, out _) && clock.UnityFrame >= 0 &&
        double.IsFinite(clock.UnityTime) && double.IsFinite(clock.UnityFixedTime) &&
        clock.ControlTick >= 0 && clock.ClientFixedSubstep >= 0;

    internal static bool ClocksAreConsecutive(
        AttackZoneClock previous,
        AttackZoneClock current)
    {
        if (!ValidateClock(previous) || !ValidateClock(current) ||
            current.StopwatchFrequencyHz != previous.StopwatchFrequencyHz)
        {
            return false;
        }
        var stopwatchIntervalSeconds =
            (current.StopwatchTimestampTicks - previous.StopwatchTimestampTicks) /
            (double)previous.StopwatchFrequencyHz;
        var unityFixedIntervalSeconds = current.UnityFixedTime - previous.UnityFixedTime;
        return current.ControlTick == previous.ControlTick + 1 &&
               current.ClientFixedSubstep ==
                   previous.ClientFixedSubstep + FixedSubstepsPerControlTick &&
               stopwatchIntervalSeconds >= MinimumStopwatchIntervalSeconds &&
               stopwatchIntervalSeconds <= MaximumStopwatchIntervalSeconds &&
               current.UnityFrame >= previous.UnityFrame &&
               current.UnityTime >= previous.UnityTime &&
               Math.Abs(unityFixedIntervalSeconds - 1.0 / ControlRateHz) <=
                   UnityFixedIntervalToleranceSeconds &&
               DateTimeOffset.Parse(current.Utc) > DateTimeOffset.Parse(previous.Utc);
    }

    internal static AttackZoneCensorDisposition ClassifyCensorDisposition(
        bool localFalling,
        bool localFallen,
        bool localDampened,
        bool localRecoveryArmed,
        bool localGetUpPending,
        bool localResetting,
        bool localMotorShutdown,
        bool localInputRecovering,
        bool opponentUnhealthy)
    {
        if (localFalling || localFallen || localDampened || localRecoveryArmed ||
            localGetUpPending || localResetting || localMotorShutdown ||
            localInputRecovering)
        {
            return AttackZoneCensorDisposition.ContinueLocalRecovery;
        }
        return opponentUnhealthy
            ? AttackZoneCensorDisposition.OpponentOnly
            : AttackZoneCensorDisposition.None;
    }

    internal static string MapRecoveryLifecycleEventName(string reason) =>
        reason is "local_special_command_edge_set" or
            "local_estop_toggle_edge_set" or "client_request_method_returned"
            ? "recovery_request_observed"
            : "recovery_state_observed";

    internal static bool ValidSha256(string? value) => ValidLowerHex(value, 64);

    private static string BuildCanonicalJson() => JsonSerializer.Serialize(new
    {
        acquisition = new
        {
            acquisition_timeout_ticks = AcquisitionTimeoutTicks,
            approach_backoff_command = ApproachBackoffCommand,
            approach_forward_command = ApproachForwardCommand,
            bearing_error_limit_deg = BearingErrorLimitDegrees,
            bearing_definition = BearingDefinition,
            bearing_rule = AcquisitionYawRule,
            distance_rule = "drive_into_central_50_percent_then_exact_neutral",
            no_strafe = true,
            yaw_command_scale_deg = YawCommandScaleDegrees,
        },
        authority_caveat = AuthorityCaveat,
        authority_scope = AuthorityScope,
        bearing_bins = BearingBins.Select(BinPayload).ToArray(),
        build = new
        {
            game_assembly_sha256 = ExpectedGameAssemblySha256,
            global_metadata_sha256 = ExpectedGlobalMetadataSha256,
            recorder_version = ExpectedRecorderVersion,
            recorder_plugin_sha256 = ExpectedRecorderPluginSha256,
        },
        completion_timeout_ticks = CompletionTimeoutTicks,
        control_rate_hz = ControlRateHz,
        distance_bins = DistanceBins.Select(BinPayload).ToArray(),
        events = RequiredEvidenceEvents,
        fixed_substeps_per_control_tick = FixedSubstepsPerControlTick,
        minimum_independent_runs_per_cell = MinimumIndependentRunsPerCell,
        motion = new
        {
            opponent_planar_speed_limit_m_s = PlanarSpeedLimitMetersPerSecond,
            opponent_yaw_rate_limit_rad_s = YawRateLimitRadiansPerSecond,
            strata = new[]
            {
                "stationary", "closing", "receding", "tangential", "turning",
                "compound_or_unknown",
            },
        },
        moves = ContinuousBotControllerContract.Attacks.Select(value => new
        {
            move_index = value.MoveIndex,
            serialized_asset_sha256 = value.SerializedAssetSha256,
        }).ToArray(),
        randomization_algorithm = RandomizationAlgorithm,
        recovery_after_censor = new
        {
            local_fall_recovery_continues_after_terminal_trial_event = true,
            local_upright_readiness_consecutive_ticks = RecoveryReadyTicks,
            normal_recovery_sequence =
                ContinuousBotControllerContract.RecoveryGuardProvenance,
            motor_fault_sequence =
                ContinuousBotControllerContract.FaultEStopProvenance,
            required_recovery_request_kinds = RequiredRecoveryRequestKinds,
            opponent_only_fall_does_not_issue_local_recovery = true,
            attack_requests_allowed_while_recovering = false,
        },
        request_start_timeout_ticks = RequestStartTimeoutTicks,
        required_isolation_proof = RequiredIsolationProof,
        schedule_schema = ScheduleSchema,
        schema = Schema,
        settle = new
        {
            consecutive_ticks = SettleTicks,
            fighter_health_predicates = new[]
            {
                "not_falling", "not_fallen", "not_recovering", "not_dampened",
                "not_resetting", "motor_running",
            },
            local_planar_speed_limit_m_s = PlanarSpeedLimitMetersPerSecond,
            local_yaw_rate_limit_rad_s = YawRateLimitRadiansPerSecond,
            maximum_stopwatch_interval_s = MaximumStopwatchIntervalSeconds,
            minimum_stopwatch_interval_s = MinimumStopwatchIntervalSeconds,
            quaternion_norm_tolerance = QuaternionNormTolerance,
            telemetry_interval_ticks = TelemetryIntervalTicks,
            unity_fixed_interval_s = 1.0 / ControlRateHz,
            unity_fixed_interval_tolerance_s = UnityFixedIntervalToleranceSeconds,
        },
        unity_fixed_rate_hz = UnityFixedRateHz,
    });

    private static object BinPayload(AttackZoneBin value) => new
    {
        id = value.Id,
        lower = value.Lower,
        upper = value.Upper,
        lower_inclusive = value.LowerInclusive,
        upper_inclusive = value.UpperInclusive,
        center = value.Center,
    };

    private static object DistanceBinPayload(AttackZoneBin value) => new
    {
        id = value.Id,
        lower_m = value.Lower,
        upper_m = value.Upper,
        lower_inclusive = value.LowerInclusive,
        upper_inclusive = value.UpperInclusive,
        center_m = value.Center,
    };

    private static object BearingBinPayload(AttackZoneBin value) => new
    {
        id = value.Id,
        lower_deg = value.Lower,
        upper_deg = value.Upper,
        lower_inclusive = value.LowerInclusive,
        upper_inclusive = value.UpperInclusive,
        center_deg = value.Center,
    };

    private static object TargetPayload(AttackZoneTrialTarget target) => new
    {
        attack_zone_trial_schema = target.Schema,
        protocol_sha256 = target.ProtocolSha256,
        controller_contract_sha256 = target.ControllerContractSha256,
        game_assembly_sha256 = target.GameAssemblySha256,
        global_metadata_sha256 = target.GlobalMetadataSha256,
        recorder_version = target.RecorderVersion,
        recorder_plugin_sha256 = target.RecorderPluginSha256,
        schedule_schema = target.ScheduleSchema,
        schedule_sha256 = target.ScheduleSha256,
        randomization_algorithm = target.RandomizationAlgorithm,
        randomization_seed_hex = target.RandomizationSeedHex,
        schedule_ordinal = target.ScheduleOrdinal,
        independent_run_id = target.IndependentRunId,
        independent_run_ordinal = target.IndependentRunOrdinal,
        required_independent_runs_per_cell = target.RequiredIndependentRunsPerCell,
        session_identity_sha256 = target.SessionIdentitySha256,
        round_identity_sha256 = target.RoundIdentitySha256,
        trial_id = target.TrialId,
        action_sequence = target.ActionSequence,
        move_index = target.MoveIndex,
        serialized_asset_sha256 = target.SerializedAssetSha256,
        distance_bin = new
        {
            id = target.DistanceBin.Id,
            lower_m = target.DistanceBin.Lower,
            upper_m = target.DistanceBin.Upper,
            lower_inclusive = target.DistanceBin.LowerInclusive,
            upper_inclusive = target.DistanceBin.UpperInclusive,
            center_m = target.DistanceBin.Center,
        },
        bearing_bin = new
        {
            id = target.BearingBin.Id,
            lower_deg = target.BearingBin.Lower,
            upper_deg = target.BearingBin.Upper,
            lower_inclusive = target.BearingBin.LowerInclusive,
            upper_inclusive = target.BearingBin.UpperInclusive,
            center_deg = target.BearingBin.Center,
        },
        acquisition_timeout_ticks = target.AcquisitionTimeoutTicks,
    };

    private static bool TryParseRequestedBin(
        JsonElement element,
        string lowerName,
        string upperName,
        string centerName,
        out AttackZoneRequestedBin bin)
    {
        bin = null!;
        if (!HasExactProperties(element, new[]
            {
                "id", lowerName, upperName, "lower_inclusive",
                "upper_inclusive", centerName,
            }))
        {
            return false;
        }
        bin = new AttackZoneRequestedBin(
            RequiredString(element, "id"),
            RequiredDouble(element, lowerName),
            RequiredDouble(element, upperName),
            RequiredBoolean(element, "lower_inclusive"),
            RequiredBoolean(element, "upper_inclusive"),
            RequiredDouble(element, centerName));
        return true;
    }

    private static bool HasExactProperties(JsonElement element, IReadOnlyList<string> expected)
    {
        if (element.ValueKind != JsonValueKind.Object)
            return false;
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (!names.Add(property.Name))
                return false;
        }
        return names.Count == expected.Count && expected.All(names.Contains);
    }

    private static string RequiredString(JsonElement parent, string name)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind != JsonValueKind.String)
            throw new InvalidOperationException();
        return value.GetString()!;
    }

    private static int RequiredInt32(JsonElement parent, string name)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out var result))
            throw new InvalidOperationException();
        return result;
    }

    private static double RequiredDouble(JsonElement parent, string name)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind != JsonValueKind.Number)
            throw new InvalidOperationException();
        var result = value.GetDouble();
        if (!double.IsFinite(result))
            throw new InvalidOperationException();
        return result;
    }

    private static bool RequiredBoolean(JsonElement parent, string name)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new InvalidOperationException();
        return value.GetBoolean();
    }

    private static bool TryMatchRequestedBin(
        AttackZoneRequestedBin requested,
        IEnumerable<AttackZoneBin> candidates,
        out AttackZoneBin matched)
    {
        matched = candidates.FirstOrDefault(
            value => string.Equals(value.Id, requested.Id, StringComparison.Ordinal))!;
        return matched is not null && SameDoubleBits(matched.Lower, requested.Lower) &&
            SameDoubleBits(matched.Upper, requested.Upper) &&
            matched.LowerInclusive == requested.LowerInclusive &&
            matched.UpperInclusive == requested.UpperInclusive &&
            SameDoubleBits(matched.Center, requested.Center);
    }

    private static bool TryRotateLocalPlusZ(
        AttackZoneRootObservation root,
        out double forwardX,
        out double forwardZ)
    {
        forwardX = 2.0 * (root.RotationX * root.RotationZ +
            root.RotationW * root.RotationY);
        forwardZ = 1.0 - 2.0 * (
            root.RotationX * root.RotationX + root.RotationY * root.RotationY);
        var lengthSquared = forwardX * forwardX + forwardZ * forwardZ;
        if (!double.IsFinite(lengthSquared) || lengthSquared <= 1e-12)
            return false;
        var length = Math.Sqrt(lengthSquared);
        forwardX /= length;
        forwardZ /= length;
        return AllFinite(forwardX, forwardZ);
    }

    private static double SignedPlanarAngleDegrees(
        double fromX,
        double fromZ,
        double toX,
        double toZ)
    {
        var crossY = fromZ * toX - fromX * toZ;
        var dot = fromX * toX + fromZ * toZ;
        return Math.Atan2(crossY, dot) * (180.0 / Math.PI);
    }

    private static bool ValidIdentifier(string? value)
    {
        if (string.IsNullOrEmpty(value) || value.Length > 64)
            return false;
        return value.All(character =>
            character is >= 'a' and <= 'z' or >= 'A' and <= 'Z' or >= '0' and <= '9' ||
            character is '.' or '_' or ':' or '-');
    }

    private static bool ValidLowerHex(string? value, int length) =>
        value is { } && value.Length == length && value.All(character =>
            character is >= '0' and <= '9' or >= 'a' and <= 'f');

    private static bool SameDoubleBits(double left, double right) =>
        BitConverter.DoubleToInt64Bits(left) == BitConverter.DoubleToInt64Bits(right);

    private static bool SameFloatBits(float left, float right) =>
        BitConverter.SingleToInt32Bits(left) == BitConverter.SingleToInt32Bits(right);

    private static bool AllFinite(params double[] values) => values.All(double.IsFinite);

    private static string HashUtf8(string value) => Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(value))).ToLowerInvariant();

    private static string CellKey(int moveIndex, string distanceId, string bearingId) =>
        $"move-{moveIndex}:{distanceId}:{bearingId}";
}

internal sealed record AttackZoneBin(
    string Id,
    double Lower,
    double Upper,
    bool LowerInclusive,
    bool UpperInclusive)
{
    internal bool IsValid => !string.IsNullOrEmpty(Id) && double.IsFinite(Lower) &&
        double.IsFinite(Upper) && Lower < Upper;

    internal double Center => Lower + (Upper - Lower) / 2.0;

    internal double CentralLower => Lower + (Upper - Lower) / 4.0;

    internal double CentralUpper => Upper - (Upper - Lower) / 4.0;

    internal bool Contains(double value) => double.IsFinite(value) &&
        (value > Lower || LowerInclusive && value == Lower) &&
        (value < Upper || UpperInclusive && value == Upper);
}

internal sealed record AttackZoneRequestedBin(
    string Id,
    double Lower,
    double Upper,
    bool LowerInclusive,
    bool UpperInclusive,
    double Center)
{
    internal static AttackZoneRequestedBin From(AttackZoneBin value) => new(
        value.Id,
        value.Lower,
        value.Upper,
        value.LowerInclusive,
        value.UpperInclusive,
        value.Center);
}

internal sealed record AttackZoneTrialTarget(
    string Schema,
    string ProtocolSha256,
    string ControllerContractSha256,
    string GameAssemblySha256,
    string GlobalMetadataSha256,
    string RecorderVersion,
    string RecorderPluginSha256,
    string ScheduleSchema,
    string ScheduleSha256,
    string RandomizationAlgorithm,
    string RandomizationSeedHex,
    int ScheduleOrdinal,
    string IndependentRunId,
    int IndependentRunOrdinal,
    int RequiredIndependentRunsPerCell,
    string SessionIdentitySha256,
    string RoundIdentitySha256,
    string TrialId,
    int ActionSequence,
    int MoveIndex,
    string SerializedAssetSha256,
    AttackZoneRequestedBin DistanceBin,
    AttackZoneRequestedBin BearingBin,
    int AcquisitionTimeoutTicks);

internal sealed record AttackZoneValidatedTarget(
    AttackZoneTrialTarget Request,
    ContinuousAttackProfile Attack,
    AttackZoneBin DistanceBin,
    AttackZoneBin BearingBin);

internal sealed record AttackZoneScopeObservation(
    bool IsolatedSparkVerified,
    string? IsolationProof,
    bool ExclusiveLeaseHeld,
    long LeaseConnectionId,
    bool GlobalInputUsed,
    bool SemanticCommandSurfaceAvailable,
    bool PrivateSessionProven,
    bool Ranked,
    bool ExactSparringBotOne,
    bool ActiveRound,
    int FighterCount,
    bool LocalSemanticT800,
    bool LocalRuntimeExactT800,
    bool OpponentRuntimeExactT800,
    string? OpponentRuntimeIdentitySha256,
    bool OpponentSemanticRuntimeMismatch,
    bool BuildHashesMatch,
    bool ControllerContractHashMatch,
    bool RecorderPinMatch,
    bool SendBoundaryPatchesVerified,
    string? SessionIdentitySha256,
    string? RoundIdentitySha256);

internal sealed record AttackZoneScopeValidation(bool Accepted, string Reason);

internal sealed record AttackZoneAcquisitionDecision(
    float Forward,
    float Strafe,
    float Yaw,
    bool ExactNeutral,
    string Reason);

internal sealed record AttackZoneClock(
    long StopwatchTimestampTicks,
    long StopwatchFrequencyHz,
    string Utc,
    int UnityFrame,
    double UnityTime,
    double UnityFixedTime,
    int ControlTick,
    int ClientFixedSubstep);

internal sealed record AttackZoneAnimationObservation(
    bool ActionPlaying,
    string? ActiveClip,
    double? ActiveClipFrame,
    double? ActiveClipFps)
{
    internal bool IsValid => ActiveClip is null
        ? !ActionPlaying && ActiveClipFrame is null && ActiveClipFps is null
        : !string.IsNullOrWhiteSpace(ActiveClip) && ActiveClip.Length <= 256 &&
          ActiveClipFrame is { } frame && double.IsFinite(frame) && frame >= 0.0 &&
          ActiveClipFps is { } fps && double.IsFinite(fps) && fps > 0.0;
}

internal sealed record AttackZoneRootObservation(
    double PositionX,
    double PositionY,
    double PositionZ,
    double RotationX,
    double RotationY,
    double RotationZ,
    double RotationW,
    double LinearVelocityX,
    double LinearVelocityY,
    double LinearVelocityZ,
    double AngularVelocityX,
    double AngularVelocityY,
    double AngularVelocityZ,
    bool Falling,
    bool Fallen,
    bool Recovering,
    bool Dampened,
    bool Resetting,
    bool MotorShutdown)
{
    internal bool IsFinite => new[]
    {
        PositionX, PositionY, PositionZ,
        RotationX, RotationY, RotationZ, RotationW,
        LinearVelocityX, LinearVelocityY, LinearVelocityZ,
        AngularVelocityX, AngularVelocityY, AngularVelocityZ,
    }.All(double.IsFinite) &&
        Math.Abs(
            RotationX * RotationX + RotationY * RotationY +
            RotationZ * RotationZ + RotationW * RotationW - 1.0) <=
        AttackZoneTrialContract.QuaternionNormTolerance;

    internal double PlanarSpeedMetersPerSecond =>
        Math.Sqrt(LinearVelocityX * LinearVelocityX +
            LinearVelocityZ * LinearVelocityZ);
}

internal sealed record AttackZoneControlObservation(
    AttackZoneClock Clock,
    AttackZoneRootObservation LocalRoot,
    AttackZoneRootObservation OpponentRoot,
    AttackZoneAnimationObservation LocalAnimation,
    AttackZoneAnimationObservation OpponentAnimation,
    bool NeutralRequestMethodReturned,
    bool VelocityCommandExactNeutral,
    bool LocalActionReady,
    bool PendingMove,
    bool PendingSpecial,
    bool PendingEStop);

internal sealed record AttackZoneGeometry(
    bool IsValid,
    double DistanceMeters,
    double LocalBearingToOpponentDegrees,
    double OpponentBearingToLocalDegrees,
    double LocalToOpponentUnitX,
    double LocalToOpponentUnitZ)
{
    internal static readonly AttackZoneGeometry Invalid = new(
        false,
        double.NaN,
        double.NaN,
        double.NaN,
        double.NaN,
        double.NaN);
}

internal sealed record AttackZoneMotionClassification(
    string MotionStratum,
    string FacingStratum,
    bool Stationary,
    double OpponentPlanarSpeedMetersPerSecond,
    double OpponentYawRateRadiansPerSecond,
    double RadialClosingSpeedMetersPerSecond,
    double TangentialSpeedMetersPerSecond)
{
    internal static readonly AttackZoneMotionClassification Unknown = new(
        "compound_or_unknown",
        "opponent_facing_unknown",
        false,
        double.NaN,
        double.NaN,
        double.NaN,
        double.NaN);
}

internal sealed record AttackZoneSampleEvaluation(
    bool AcquisitionPass,
    bool ClockValid,
    bool RootsFinite,
    bool GeometryValid,
    bool AnimationValid,
    bool NeutralRequestMethodReturned,
    bool VelocityCommandExactNeutral,
    bool LocalActionReady,
    bool NoPendingRequests,
    bool LocalHealthy,
    bool OpponentHealthy,
    bool DistanceCentralPass,
    bool BearingInBinPass,
    bool BearingErrorPass,
    bool LocalMotionPass,
    bool OpponentStationary,
    double BearingErrorDegrees,
    double LocalPlanarSpeedMetersPerSecond,
    double LocalYawRateRadiansPerSecond,
    AttackZoneGeometry Geometry,
    AttackZoneMotionClassification Motion);

internal sealed record AttackZoneScheduleEntry(
    int ScheduleOrdinal,
    string IndependentRunId,
    int IndependentRunOrdinal,
    int RepetitionWithinRun,
    int MoveIndex,
    string SerializedAssetSha256,
    AttackZoneBin DistanceBin,
    AttackZoneBin BearingBin);

internal sealed record AttackZoneCellCoverage(
    string CellKey,
    int IndependentRunCount,
    int RequiredIndependentRunCount);

internal sealed record AttackZoneCoverageValidation(
    bool Complete,
    IReadOnlyList<AttackZoneCellCoverage> MissingCells);

internal enum AttackZoneCensorDisposition
{
    None,
    OpponentOnly,
    ContinueLocalRecovery,
}

internal sealed class AttackZoneDeterministicRandom
{
    private readonly byte[] _seed;
    private ulong _counter;

    internal AttackZoneDeterministicRandom(string seedHex)
    {
        _seed = Convert.FromHexString(seedHex);
    }

    internal int NextIndex(int exclusiveUpperBound)
    {
        if (exclusiveUpperBound <= 0)
            throw new ArgumentOutOfRangeException(nameof(exclusiveUpperBound));
        var bound = (ulong)exclusiveUpperBound;
        var threshold = unchecked(0UL - bound) % bound;
        Span<byte> input = stackalloc byte[40];
        _seed.CopyTo(input);
        Span<byte> digest = stackalloc byte[32];
        while (true)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(input[32..], _counter++);
            SHA256.HashData(input, digest);
            var value = BinaryPrimitives.ReadUInt64LittleEndian(digest);
            if (value >= threshold)
                return (int)(value % bound);
        }
    }
}
