using System.Text.Json;

namespace RekUiBridgeAgent;

internal static class ContinuousBotControllerContract
{
    internal const string Schema = "rek.continuous_private_bot_controller.v1";
    internal const string AuthorityScope = "client_request_edges_and_local_observations_only";
    internal const string AuthorityCaveat =
        "client request edge and local motion lifecycle observations only; server acceptance and authoritative execution are unknown";
    internal const string RangeAngleProvenance =
        "build_pinned_baseline_ai_global_thresholds_projected_per_move_not_runtime_calibrated";
    internal const string FacingYawProvenance =
        "build_pinned_AIOpponentController.ComputeFacingYaw_rva_0x2366e20_AngleToOpponent_rva_0x2366600_half_threshold_deadband_abs_angle_over_45_clamped_times_engage_yaw_speed_negative_bearing_sign";
    internal const string AttackSelectionProvenance =
        "audit_controller_deterministic_round_robin_diverges_from_build_pinned_AIOpponentController_random_category_and_clip_selection";
    internal const string StaticImpactTimingProvenance =
        "build_pinned_serialized_t800_move_asset_metadata_not_measured_runtime_timing";
    internal const string RoundRestartLimitation =
        "build_pinned_post_fight_continue_restarts_only_after_win_and_exits_to_lobby_after_loss";
    internal const string RoundRestartStaticEvidence =
        "GameMenuController.HandlePostFightContinue_rva_0x23aae90_branches_on_postFightIsWinner_false_ExitToLobby_true_SendPostFightIntent_stay_true";
    internal const string RecoveryGuardProvenance =
        "build_pinned_AIOpponentController.DriveRecovery_rva_0x2367430_fallen_not_dampened_Dampen_4_then_Straighten_1_once_then_RecoveryArmed_SuggestedGetUpOrientation_2_or_3";
    internal const string FaultEStopProvenance =
        "build_pinned_AIOpponentController.UpdateFaultEStopCycle_rva_0x23680e0_motorShutdownHold_faultEStopDelay_then_0.5_second_estop_hold";
    internal const string DampenGuard =
        "fallen_and_not_dampened";
    internal const string StraightenGuard =
        "fallen_and_dampened_and_not_already_issued";
    internal const string OpponentRuntimeRequirement =
        "exact_t800_runtime_bone_signature_required_semantic_robot_id_recorded_but_not_trusted";
    internal const string MoveSendFreshGuard =
        "fresh_pre_send_frame_blocks_local_fall_dampen_recovery_reset_motor_or_input_recovery_and_opponent_fall_dampen_recovery_reset_or_motor_state";
    internal const string FaultPreemptionStraightenRule =
        "preserve_straighten_issued_within_existing_recovery_episode_reset_only_for_new_episode_or_verified_upright";
    internal const int UnityFixedRateHz = 500;
    internal const int ControlRateHz = 50;
    internal const int FixedSubstepsPerControlTick = UnityFixedRateHz / ControlRateHz;
    internal const int TelemetryIntervalTicks = 5;
    internal const int SettleTicks = 15;
    internal const int RequestStartTimeoutTicks = 250;
    internal const int ActionCompletionTimeoutTicks = 750;
    internal const int RecoveryRetryTicks = 25;
    internal const int RecoveryObservationTimeoutTicks = 250;
    internal const int FaultEStopDelayTicks = ControlRateHz * 3;
    internal const int FaultEStopHoldTicks = ControlRateHz / 2;
    internal const int RoundStartPromptDelayTicks = 5;
    internal const int RoundStartObservationTimeoutTicks = 1500;
    internal const int TwoMinuteTicks = ControlRateHz * 120;
    internal const float EngageStopDistanceMeters = 0.4180000126361847f;
    internal const float PositionedMarginMeters = 0.1f;
    internal const float MaximumAttackDistanceMeters =
        EngageStopDistanceMeters + PositionedMarginMeters;
    internal const float FacingThresholdDegrees = 35f;
    internal const float FacingDeadbandFactor = 0.5f;
    internal const float FacingYawRampDegrees = 45f;
    internal const float EngageYawCommand = 1.5f;
    internal const float EngageForwardCommand = 0.8f;
    internal const float DownedBackOffCommand = -0.25f;
    internal const float DownedOpponentSpaceMeters = 1.5f;
    internal const string ExpectedSha256 =
        "c19ee1cc02111426db7a58cd648e244e1106842d86caaba3dc729edf4640b92e";

    internal static readonly ContinuousAttackProfile[] Attacks =
    {
        new(2, "skill", "Punch Combo", MaximumAttackDistanceMeters,
            FacingThresholdDegrees,
            "233f952edecb7bf8d1959c6549c0edb95e1833451fff988f57a4b14d92b14dd4",
            new[]
            {
                new ContinuousImpactEvent(0.7599999904632568f, 0.10000000149011612f,
                    0.1899999976158142f, 1, 1f),
                new ContinuousImpactEvent(1.149999976158142f, 0.10000000149011612f,
                    0.10000000149011612f, 1, 1f),
                new ContinuousImpactEvent(1.809999942779541f, 0.10000000149011612f,
                    0.10000000149011612f, 1, 1f),
            }),
        new(3, "youbiantui", "Right Kick", MaximumAttackDistanceMeters,
            FacingThresholdDegrees,
            "70f36a2c7b9b53c10e47cc613d87a770eb86fb2e683ed64ee39efcccf2e75636",
            new[]
            {
                new ContinuousImpactEvent(1.1100000143051147f, 0.25f,
                    0.30000001192092896f, 4, 1f),
            }),
        new(4, "left_light_attack", "Left Punch", MaximumAttackDistanceMeters,
            FacingThresholdDegrees,
            "32081b731a59b7553d94022ebff865764b34c83dbf274aadf26540fb17daad2e",
            new[]
            {
                new ContinuousImpactEvent(0.38999998569488525f, 0.11999999731779099f,
                    0.07999999821186066f, 1, 1f),
            }),
        new(5, "right_light_attack", "Right Punch", MaximumAttackDistanceMeters,
            FacingThresholdDegrees,
            "b1c1b2c000dd612e3eb4c33c5d90e03c2c9306e5cc194747c14248b0d77b7dea",
            new[]
            {
                new ContinuousImpactEvent(0.2199999988079071f, 0.11999999731779099f,
                    0.20000000298023224f, 2, 1f),
            }),
        new(9, "right_shoryuken_lm", "Dragon Punch", MaximumAttackDistanceMeters,
            FacingThresholdDegrees,
            "cc298f53d04ffd56be57ce3049559d3d30c7724fe4d2839a66ea8f3008ca8deb",
            new[]
            {
                new ContinuousImpactEvent(0f, 0f, 0f, 2, 1f),
            }),
        new(10, "front_kick_L", "Left Kick", MaximumAttackDistanceMeters,
            FacingThresholdDegrees,
            "cd5b286f6e4f5c3003cb0f5c9de5e5690ca92ed58e5a1b789f4394e4d7911ee8",
            new[]
            {
                new ContinuousImpactEvent(1.100000023841858f, 0.20000000298023224f,
                    0.15000000596046448f, 3, 1f),
            }),
    };

    internal static readonly string CanonicalJson = BuildCanonicalJson();

    private static string BuildCanonicalJson() => JsonSerializer.Serialize(new
    {
        action_completion_timeout_ticks = ActionCompletionTimeoutTicks,
        attacks = Attacks.Select(attack => new
        {
            display_name = attack.DisplayName,
            maximum_abs_bearing_degrees = attack.MaximumAbsBearingDegrees,
            maximum_distance_m = attack.MaximumDistanceMeters,
            move_index = attack.MoveIndex,
            move_name = attack.MoveName,
            serialized_asset_sha256 = attack.SerializedAssetSha256,
            static_impact_events = attack.StaticImpactEvents.Select(value => new
            {
                gain_boost = value.GainBoost,
                impact_time_s = value.ImpactTimeSeconds,
                lead_time_s = value.LeadTimeSeconds,
                limb = value.Limb,
                release_time_s = value.ReleaseTimeSeconds,
            }).ToArray(),
        }).ToArray(),
        authority_caveat = AuthorityCaveat,
        authority_scope = AuthorityScope,
        attack_selection_provenance = AttackSelectionProvenance,
        control_rate_hz = ControlRateHz,
        downed_back_off_command = DownedBackOffCommand,
        downed_opponent_space_m = DownedOpponentSpaceMeters,
        engage_forward_command = EngageForwardCommand,
        engage_yaw_command = EngageYawCommand,
        engage_stop_distance_m = EngageStopDistanceMeters,
        facing_deadband_factor = FacingDeadbandFactor,
        facing_threshold_degrees = FacingThresholdDegrees,
        facing_yaw_provenance = FacingYawProvenance,
        facing_yaw_ramp_degrees = FacingYawRampDegrees,
        fault_estop_delay_ticks = FaultEStopDelayTicks,
        fault_estop_hold_ticks = FaultEStopHoldTicks,
        fault_estop_provenance = FaultEStopProvenance,
        fixed_substeps_per_control_tick = FixedSubstepsPerControlTick,
        positioned_margin_m = PositionedMarginMeters,
        range_angle_provenance = RangeAngleProvenance,
        recovery_observation_timeout_ticks = RecoveryObservationTimeoutTicks,
        normal_fall_recovery_request_sequence = new[]
        {
            "dampen_while_fallen_and_not_dampened",
            "observe_dampened",
            "straighten_once_while_fallen_and_dampened",
            "observe_recovery_armed",
            "orientation_get_up_only_when_recovery_armed_and_motor_running",
        },
        motor_shutdown_fault_request_sequence = new[]
        {
            "observe_motor_shutdown_hold",
            "wait_fault_estop_delay",
            "estop_toggle_on",
            "observe_motor_shutdown",
            "hold_estop_for_0.5_seconds",
            "estop_toggle_off",
            "observe_motor_running",
        },
        dampen_guard = DampenGuard,
        fault_preemption_straighten_rule = FaultPreemptionStraightenRule,
        move_send_fresh_guard = MoveSendFreshGuard,
        opponent_runtime_requirement = OpponentRuntimeRequirement,
        recovery_guard_provenance = RecoveryGuardProvenance,
        recovery_retry_ticks = RecoveryRetryTicks,
        request_start_timeout_ticks = RequestStartTimeoutTicks,
        round_restart_limitation = RoundRestartLimitation,
        round_restart_static_evidence = RoundRestartStaticEvidence,
        round_start_observation_timeout_ticks = RoundStartObservationTimeoutTicks,
        round_start_prompt_delay_ticks = RoundStartPromptDelayTicks,
        schema = Schema,
        settle_ticks = SettleTicks,
        static_impact_timing_provenance = StaticImpactTimingProvenance,
        straighten_guard = StraightenGuard,
        telemetry_interval_ticks = TelemetryIntervalTicks,
        unity_fixed_rate_hz = UnityFixedRateHz,
    });

    internal static bool ShouldIssueRoundStartRequest(
        bool sameProvenPrivateBotOneSession,
        bool roundInactive,
        bool promptVisible,
        bool promptEnabled,
        bool postFightWinner,
        bool requestAlreadyIssued,
        int inactiveTicks) =>
        sameProvenPrivateBotOneSession &&
        roundInactive &&
        promptVisible &&
        promptEnabled &&
        postFightWinner &&
        !requestAlreadyIssued &&
        inactiveTicks >= RoundStartPromptDelayTicks &&
        inactiveTicks < TwoMinuteTicks;

    internal static bool CanBindRestartedRound(
        bool inactiveTransitionObserved,
        bool ownedStartRequestIssued,
        bool newRoundIdentityObserved) =>
        inactiveTransitionObserved &&
        ownedStartRequestIssued &&
        newRoundIdentityObserved;

    internal static bool IsInputReadyForControl(
        bool isActive,
        bool networkInitialized,
        bool hasPendingEStop,
        bool exactOwnedPendingEStop) =>
        isActive &&
        networkInitialized &&
        (!hasPendingEStop || exactOwnedPendingEStop);

    internal static bool LocalBlocksFreshMoveSend(
        bool falling,
        bool fallen,
        bool dampened,
        bool recoveryArmed,
        bool getUpPending,
        bool resetting,
        bool motorShutdown,
        bool inputRecovering) =>
        falling || fallen || dampened || recoveryArmed || getUpPending ||
        resetting || motorShutdown || inputRecovering;

    internal static bool OpponentBlocksFreshMoveSend(
        bool falling,
        bool fallen,
        bool dampened,
        bool recoveryArmed,
        bool getUpPending,
        bool resetting,
        bool motorShutdown) =>
        falling || fallen || dampened || recoveryArmed || getUpPending ||
        resetting || motorShutdown;

    internal static bool ResolveStraightenIssuedOnFaultEntry(
        bool recoveryEpisodeAlreadyActive,
        bool straightenAlreadyIssued) =>
        recoveryEpisodeAlreadyActive && straightenAlreadyIssued;

    internal static ContinuousSemanticRuntimeConsistency ClassifySemanticRuntimeConsistency(
        bool semanticDeclaresT800,
        bool runtimeIsExactT800)
    {
        var mismatch = semanticDeclaresT800 != runtimeIsExactT800;
        return new ContinuousSemanticRuntimeConsistency(
            mismatch,
            mismatch
                ? "semantic_t800_flag_disagrees_with_runtime_t800_signature"
                : runtimeIsExactT800
                    ? "semantic_and_runtime_both_exact_t800"
                    : "semantic_and_runtime_not_comparable_beyond_t800_signature");
    }

    internal static bool HasRequiredT800Pairing(
        bool localSemanticT800,
        bool localExactT800Runtime,
        bool opponentExactT800Runtime) =>
        localSemanticT800 && localExactT800Runtime && opponentExactT800Runtime;

    internal static ContinuousEStopHandshakeDecision DecideFaultEStopHandshake(
        ContinuousEStopRecoveryStage stage,
        bool motorShutdown,
        int stageElapsedTicks) => stage switch
    {
        ContinuousEStopRecoveryStage.FaultDelay when !motorShutdown =>
            new(
                stage,
                ContinuousEStopHandshakeEdge.FailClosed),
        ContinuousEStopRecoveryStage.FaultDelay
            when stageElapsedTicks >= FaultEStopDelayTicks =>
            new(
                ContinuousEStopRecoveryStage.AwaitMotorShutdown,
                ContinuousEStopHandshakeEdge.RequestToggleOn),
        ContinuousEStopRecoveryStage.FaultDelay =>
            new(stage, ContinuousEStopHandshakeEdge.None),
        ContinuousEStopRecoveryStage.AwaitMotorShutdown when motorShutdown =>
            new(
                ContinuousEStopRecoveryStage.FaultHold,
                ContinuousEStopHandshakeEdge.ObserveMotorShutdown),
        ContinuousEStopRecoveryStage.AwaitMotorShutdown =>
            new(stage, ContinuousEStopHandshakeEdge.None),
        ContinuousEStopRecoveryStage.FaultHold when !motorShutdown =>
            new(
                stage,
                ContinuousEStopHandshakeEdge.FailClosed),
        ContinuousEStopRecoveryStage.FaultHold
            when stageElapsedTicks >= FaultEStopHoldTicks =>
            new(
                ContinuousEStopRecoveryStage.AwaitMotorRunning,
                ContinuousEStopHandshakeEdge.RequestToggleOff),
        ContinuousEStopRecoveryStage.FaultHold =>
            new(stage, ContinuousEStopHandshakeEdge.None),
        ContinuousEStopRecoveryStage.AwaitMotorRunning when !motorShutdown =>
            new(
                ContinuousEStopRecoveryStage.Complete,
                ContinuousEStopHandshakeEdge.ObserveMotorRunning),
        ContinuousEStopRecoveryStage.AwaitMotorRunning =>
            new(stage, ContinuousEStopHandshakeEdge.None),
        _ => new(stage, ContinuousEStopHandshakeEdge.FailClosed),
    };

    internal static bool TryComputePlanarGeometry(
        float localX,
        float localZ,
        float localForwardX,
        float localForwardZ,
        float opponentX,
        float opponentZ,
        float opponentForwardX,
        float opponentForwardZ,
        out PlanarCombatGeometry geometry)
    {
        geometry = default!;
        if (!Finite(localX, localZ, localForwardX, localForwardZ,
                opponentX, opponentZ, opponentForwardX, opponentForwardZ))
        {
            return false;
        }

        var deltaX = opponentX - localX;
        var deltaZ = opponentZ - localZ;
        var distanceSquared = deltaX * deltaX + deltaZ * deltaZ;
        var localForwardSquared = localForwardX * localForwardX + localForwardZ * localForwardZ;
        var opponentForwardSquared =
            opponentForwardX * opponentForwardX + opponentForwardZ * opponentForwardZ;
        if (!float.IsFinite(distanceSquared) || distanceSquared <= 1e-8f ||
            !float.IsFinite(localForwardSquared) || localForwardSquared <= 1e-8f ||
            !float.IsFinite(opponentForwardSquared) || opponentForwardSquared <= 1e-8f)
        {
            return false;
        }

        var distance = MathF.Sqrt(distanceSquared);
        var localForwardLength = MathF.Sqrt(localForwardSquared);
        var opponentForwardLength = MathF.Sqrt(opponentForwardSquared);
        var toOpponentX = deltaX / distance;
        var toOpponentZ = deltaZ / distance;
        var toLocalX = -toOpponentX;
        var toLocalZ = -toOpponentZ;
        var normalizedLocalForwardX = localForwardX / localForwardLength;
        var normalizedLocalForwardZ = localForwardZ / localForwardLength;
        var normalizedOpponentForwardX = opponentForwardX / opponentForwardLength;
        var normalizedOpponentForwardZ = opponentForwardZ / opponentForwardLength;
        var localBearing = SignedPlanarAngleDegrees(
            normalizedLocalForwardX,
            normalizedLocalForwardZ,
            toOpponentX,
            toOpponentZ);
        var opponentBearing = SignedPlanarAngleDegrees(
            normalizedOpponentForwardX,
            normalizedOpponentForwardZ,
            toLocalX,
            toLocalZ);
        var localHeading = HeadingDegrees(normalizedLocalForwardX, normalizedLocalForwardZ);
        var opponentHeading = HeadingDegrees(
            normalizedOpponentForwardX,
            normalizedOpponentForwardZ);
        if (!Finite(distance, localBearing, opponentBearing, localHeading, opponentHeading))
            return false;

        geometry = new PlanarCombatGeometry(
            distance,
            localBearing,
            opponentBearing,
            localHeading,
            opponentHeading,
            toOpponentX,
            toOpponentZ);
        return true;
    }

    internal static ContinuousLocomotionDecision DecideLocomotion(
        PlanarCombatGeometry geometry,
        ContinuousAttackProfile attack,
        bool opponentDown)
    {
        if (opponentDown)
        {
            return geometry.DistanceMeters < DownedOpponentSpaceMeters
                ? new ContinuousLocomotionDecision(
                    DownedBackOffCommand,
                    0f,
                    ComputeFacingYaw(geometry.LocalBearingToOpponentDegrees),
                    AttackWindow: false,
                    "opponent_down_give_room")
                : new ContinuousLocomotionDecision(
                    0f,
                    0f,
                    ComputeFacingYaw(geometry.LocalBearingToOpponentDegrees),
                    AttackWindow: false,
                    "opponent_down_hold_space");
        }

        var absoluteBearing = MathF.Abs(geometry.LocalBearingToOpponentDegrees);
        if (absoluteBearing > attack.MaximumAbsBearingDegrees)
        {
            return new ContinuousLocomotionDecision(
                0f,
                0f,
                ComputeFacingYaw(geometry.LocalBearingToOpponentDegrees),
                AttackWindow: false,
                "turn_to_face_opponent");
        }

        if (geometry.DistanceMeters > attack.MaximumDistanceMeters)
        {
            return new ContinuousLocomotionDecision(
                EngageForwardCommand,
                0f,
                ComputeFacingYaw(geometry.LocalBearingToOpponentDegrees),
                AttackWindow: false,
                "advance_to_attack_range");
        }

        return new ContinuousLocomotionDecision(
            0f,
            0f,
            ComputeFacingYaw(geometry.LocalBearingToOpponentDegrees),
            AttackWindow: true,
            "attack_window_observed");
    }

    internal static ContinuousRecoveryCommand SelectRecoveryCommand(
        bool fallen,
        bool dampened,
        bool recoveryArmed,
        bool motorShutdown,
        bool straightenIssued,
        bool suggestedProne)
    {
        if (!fallen || motorShutdown)
            return ContinuousRecoveryCommand.None;
        if (!dampened)
            return ContinuousRecoveryCommand.Dampen;
        if (!straightenIssued)
            return ContinuousRecoveryCommand.Straighten;
        if (recoveryArmed)
        {
            return suggestedProne
                ? ContinuousRecoveryCommand.GetUpProne
                : ContinuousRecoveryCommand.GetUpSupine;
        }
        return ContinuousRecoveryCommand.WaitForDampenedOrRecoveryArmed;
    }

    internal static float ComputeFacingYaw(float bearingDegrees)
    {
        var absoluteBearing = MathF.Abs(bearingDegrees);
        if (absoluteBearing <= FacingThresholdDegrees * FacingDeadbandFactor)
            return 0f;
        var magnitude = Math.Clamp(absoluteBearing / FacingYawRampDegrees, 0f, 1f) *
                        EngageYawCommand;
        return bearingDegrees > 0f ? -magnitude : magnitude;
    }

    private static float HeadingDegrees(float forwardX, float forwardZ) =>
        MathF.Atan2(forwardX, forwardZ) * (180f / MathF.PI);

    private static float SignedPlanarAngleDegrees(
        float fromX,
        float fromZ,
        float toX,
        float toZ)
    {
        var crossY = fromZ * toX - fromX * toZ;
        var dot = fromX * toX + fromZ * toZ;
        return MathF.Atan2(crossY, dot) * (180f / MathF.PI);
    }

    private static bool Finite(params float[] values) =>
        values.All(float.IsFinite);
}

internal sealed record ContinuousAttackProfile(
    int MoveIndex,
    string MoveName,
    string DisplayName,
    float MaximumDistanceMeters,
    float MaximumAbsBearingDegrees,
    string SerializedAssetSha256,
    IReadOnlyList<ContinuousImpactEvent> StaticImpactEvents);

internal sealed record ContinuousImpactEvent(
    float ImpactTimeSeconds,
    float LeadTimeSeconds,
    float ReleaseTimeSeconds,
    int Limb,
    float GainBoost);

internal sealed record PlanarCombatGeometry(
    float DistanceMeters,
    float LocalBearingToOpponentDegrees,
    float OpponentBearingToLocalDegrees,
    float LocalHeadingDegrees,
    float OpponentHeadingDegrees,
    float LocalToOpponentUnitX,
    float LocalToOpponentUnitZ);

internal sealed record ContinuousLocomotionDecision(
    float Forward,
    float Strafe,
    float Yaw,
    bool AttackWindow,
    string Reason);

internal sealed record ContinuousEStopHandshakeDecision(
    ContinuousEStopRecoveryStage NextStage,
    ContinuousEStopHandshakeEdge Edge);

internal sealed record ContinuousSemanticRuntimeConsistency(
    bool Mismatch,
    string Classification);

internal enum ContinuousEStopRecoveryStage
{
    FaultDelay,
    AwaitMotorShutdown,
    FaultHold,
    AwaitMotorRunning,
    Complete,
}

internal enum ContinuousEStopHandshakeEdge
{
    None,
    RequestToggleOn,
    ObserveMotorShutdown,
    RequestToggleOff,
    ObserveMotorRunning,
    FailClosed,
}

internal enum ContinuousRecoveryCommand
{
    None,
    Dampen,
    Straighten,
    WaitForDampenedOrRecoveryArmed,
    GetUpProne,
    GetUpSupine,
}
