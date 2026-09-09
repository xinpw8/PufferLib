using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace RekUiBridgeAgent;

[Flags]
internal enum G1HeldMask : byte
{
    None = 0,
    W = 1 << 0,
    S = 1 << 1,
    A = 1 << 2,
    D = 1 << 3,
    Q = 1 << 4,
    E = 1 << 5,
}

internal enum G1KickProbeKind
{
    TranslationHeld,
    YawPreempted,
}

internal enum G1KickObservationBoundaryDecision
{
    ContinueObservation,
    CompleteObservation,
    FailPartial,
}

internal sealed record G1HeldCondition(
    int Ordinal,
    string Label,
    int StartTick,
    int StopTick,
    G1HeldMask HeldMask,
    sbyte Forward,
    sbyte Strafe,
    sbyte RawYaw);

internal sealed record G1KickProbe(
    int Ordinal,
    G1KickProbeKind Kind,
    string Label,
    int StartTick,
    int EdgeTick,
    int? TranslationReleaseTick,
    int StopTick,
    G1HeldMask DesiredHeldMask,
    int MoveIndex,
    sbyte Forward,
    sbyte Strafe,
    sbyte RawYawBeforeEdge);

internal sealed record G1KickAsset(
    int MoveIndex,
    string RuntimeName,
    string NpzSha256,
    int ControllerTicks);

internal sealed record G1HeldScheduleFrame(
    int Tick,
    G1HeldMask DesiredHeldMask,
    G1HeldMask EffectiveHeldMask,
    sbyte Forward,
    sbyte Strafe,
    sbyte RawYaw,
    string Phase,
    int? HeldConditionOrdinal,
    G1KickProbe? KickProbe,
    bool KickEdge,
    bool YawPreempted,
    bool TranslationReleased);

internal readonly record struct G1KeyboardYawState(float Ramp, float Sign);

internal readonly record struct G1KeyboardYawStep(
    G1KeyboardYawState State,
    float RawYaw,
    float EffectiveYaw);

internal readonly record struct G1KickDispatchTracker(
    bool Armed,
    int ArmedFixedSubstep,
    bool LateUpdateOpen,
    int LateUpdateOpportunities,
    bool MatchingPendingAtPrefix,
    bool SendPrefixSeen,
    bool SendPostfixSeen,
    bool MissingDispatchAfterCompletedOpportunity)
{
    internal static G1KickDispatchTracker Arm(int fixedSubstep) => new(
        true,
        fixedSubstep,
        false,
        0,
        false,
        false,
        false,
        false);

    internal G1KickDispatchTracker OnFixedSubstep() => this;

    internal G1KickDispatchTracker OnLateUpdatePrefix(bool matchingPending) => this with
    {
        LateUpdateOpen = true,
        LateUpdateOpportunities = LateUpdateOpportunities + 1,
        MatchingPendingAtPrefix = matchingPending,
    };

    internal G1KickDispatchTracker OnSendPrefix() => this with
    {
        SendPrefixSeen = true,
    };

    internal G1KickDispatchTracker OnSendPostfix() => this with
    {
        SendPostfixSeen = true,
    };

    internal G1KickDispatchTracker OnLateUpdatePostfix()
    {
        if (!LateUpdateOpen)
            return this;
        return this with
        {
            LateUpdateOpen = false,
            MissingDispatchAfterCompletedOpportunity =
                MatchingPendingAtPrefix && !SendPrefixSeen,
        };
    }
}

internal sealed record G1TranslationProbeClassification(
    string LocalGateResult,
    string BehavioralResult,
    string ResponseClassification,
    string Criteria);

internal static class G1HeldInputScheduleContract
{
    internal const string Schema = "rek.g1_held_input_schedule.v2";
    internal const string ScheduleId = "rek.private_bot1.g1_held_input.v2";
    internal const string AuthorityScope =
        "client_request_edges_local_returns_and_visual_only_client_diagnostics_only";
    internal const string AuthorityCaveat =
        "server acceptance and authoritative execution are unknown; physical response and kick timing require separately captured recorder bone-trajectory correlation";
    internal const string RequiredIsolationProof =
        "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98";
    internal const string PoseResponseSource =
        "RekEvidenceRecorder rek.private_ai.protocol.v7 correlated by the 500 Hz client-fixed clock";
    internal const string KickRetryPolicy = "one_edge_no_queue_no_retry";
    internal const string MotionIdentityStatus =
        "unknown_visual_only_client_runner_has_no_requested_mocap_asset_identity";
    internal const string KeyboardYawRampProvenance =
        "RobotInputControllerCommand.RampKeyboardYaw_rva_0x2269990_rendered_Update_deltaTime";
    internal const string TranslationBehaviorCriteria =
        "local gate uses ExecuteMove return; physical behavior remains unknown until recorder bone-trajectory correlation";
    internal const string TransitionSettledProvenance =
        "build_pinned_public_RobotInputController.TransitionSettled_rva_0x226f6f0_direction_specific_with_internal_runner_and_robot_guards";
    internal const string TransitionSettleBaseVelocityProvenance =
        "Robot.TryGetBaseVelocityLocal_rva_0x23db6f0_components_are_independent_diagnostics_not_the_TransitionSettled_predicate";
    internal const int UnityFixedRateHz = 500;
    internal const int ScheduleRateHz = 50;
    internal const int FixedSubstepsPerScheduleTick = UnityFixedRateHz / ScheduleRateHz;
    internal const float ExpectedFixedDeltaTime = 0.002f;
    internal const float ExpectedYawSpeed = 1f;
    internal const float ExpectedKeyboardYawRampTimeSeconds = 0.5f;
    internal const float ExpectedTransitionSettlePlanarSpeed = 0.03f;
    internal const float ExpectedTransitionSettleYawRate = 0.03f;
    internal const int HeldDurationTicks = 100;
    internal const int NeutralGapTicks = 50;
    internal const int KickEdgeOffsetTicks = 50;
    internal const int TranslationReleaseOffsetTicks = 5;
    internal const int MaxRecoveredKickControllerTicks = 159;
    internal const int KickDispatchStartMarginTicks = 40;
    internal const int KickObservationTicks = 200;
    internal const float RoundCapacitySafetySeconds = 10f;
    internal const int DurationScheduleTicks = 4551;
    internal const int FinalScheduleTick = DurationScheduleTicks - 1;
    internal const byte ValidHeldMask = (byte)(
        G1HeldMask.W | G1HeldMask.S | G1HeldMask.A |
        G1HeldMask.D | G1HeldMask.Q | G1HeldMask.E);
    internal const byte TranslationMask = (byte)(
        G1HeldMask.W | G1HeldMask.S | G1HeldMask.A | G1HeldMask.D);
    internal const byte YawMask = (byte)(G1HeldMask.Q | G1HeldMask.E);
    internal const string ExpectedSha256 =
        "0e28e089c73e603c7ce1d9cd5e6de4dd7f6f6017ce3bf4bcae2a90366b1adc2b";

    internal static double RequiredRunSeconds =>
        DurationScheduleTicks / (double)ScheduleRateHz;

    internal static double RequiredRoundCapacitySeconds =>
        RequiredRunSeconds + RoundCapacitySafetySeconds;

    internal static readonly int[] KickMoveIndices = { 6, 7, 8, 9 };

    internal static readonly G1KickAsset[] KickAssets =
    {
        new(6, "left_side_kick_processed",
            "854e5bbf91b325288ed0f63e43f65355aa5b1a15fe43a8fd940b4efb6ed6a465", 158),
        new(7, "left_front_kick_processed",
            "3db29034517acc376c16ce5bc04f923803a48367ad58a5c65a314123639815c1", 146),
        new(8, "right_side_kick_processed",
            "4d16c2d07848713713f8252e21fc2893462052f6ee3e157e9c364c03769de6a6", 159),
        new(9, "right_knee_processed",
            "08bba86c004d99e61d53997cd31223b8cfde67c019506267339e146dd771097f", 140),
    };

    internal static readonly G1HeldCondition[] HeldConditions =
    {
        new(0, "W", 50, 150, G1HeldMask.W, 1, 0, 0),
        new(1, "S", 200, 300, G1HeldMask.S, -1, 0, 0),
        new(2, "A", 350, 450, G1HeldMask.A, 0, 1, 0),
        new(3, "D", 500, 600, G1HeldMask.D, 0, -1, 0),
        new(4, "Q", 650, 750, G1HeldMask.Q, 0, 0, 1),
        new(5, "E", 800, 900, G1HeldMask.E, 0, 0, -1),
        new(6, "W+Q", 950, 1050, G1HeldMask.W | G1HeldMask.Q, 1, 0, 1),
        new(7, "W+E", 1100, 1200, G1HeldMask.W | G1HeldMask.E, 1, 0, -1),
        new(8, "S+Q", 1250, 1350, G1HeldMask.S | G1HeldMask.Q, -1, 0, 1),
        new(9, "S+E", 1400, 1500, G1HeldMask.S | G1HeldMask.E, -1, 0, -1),
        new(10, "A+Q", 1550, 1650, G1HeldMask.A | G1HeldMask.Q, 0, 1, 1),
        new(11, "A+E", 1700, 1800, G1HeldMask.A | G1HeldMask.E, 0, 1, -1),
        new(12, "D+Q", 1850, 1950, G1HeldMask.D | G1HeldMask.Q, 0, -1, 1),
        new(13, "D+E", 2000, 2100, G1HeldMask.D | G1HeldMask.E, 0, -1, -1),
    };

    internal static readonly G1KickProbe[] KickProbes =
    {
        new(0, G1KickProbeKind.TranslationHeld, "W+kick-6", 2150, 2200, 2205, 2400,
            G1HeldMask.W, 6, 1, 0, 0),
        new(1, G1KickProbeKind.TranslationHeld, "S+kick-7", 2450, 2500, 2505, 2700,
            G1HeldMask.S, 7, -1, 0, 0),
        new(2, G1KickProbeKind.TranslationHeld, "A+kick-8", 2750, 2800, 2805, 3000,
            G1HeldMask.A, 8, 0, 1, 0),
        new(3, G1KickProbeKind.TranslationHeld, "D+kick-9", 3050, 3100, 3105, 3300,
            G1HeldMask.D, 9, 0, -1, 0),
        new(4, G1KickProbeKind.YawPreempted, "Q+kick-6", 3350, 3400, null, 3600,
            G1HeldMask.Q, 6, 0, 0, 1),
        new(5, G1KickProbeKind.YawPreempted, "E+kick-7", 3650, 3700, null, 3900,
            G1HeldMask.E, 7, 0, 0, -1),
        new(6, G1KickProbeKind.YawPreempted, "Q+kick-8", 3950, 4000, null, 4200,
            G1HeldMask.Q, 8, 0, 0, 1),
        new(7, G1KickProbeKind.YawPreempted, "E+kick-9", 4250, 4300, null, 4500,
            G1HeldMask.E, 9, 0, 0, -1),
    };

    internal static readonly string CanonicalJson = BuildCanonicalJson();

    internal static G1HeldScheduleFrame FrameAtTick(int tick)
    {
        if (tick is < 0 or >= DurationScheduleTicks)
            throw new ArgumentOutOfRangeException(nameof(tick));

        foreach (var condition in HeldConditions)
        {
            if (tick >= condition.StartTick && tick < condition.StopTick)
            {
                return new G1HeldScheduleFrame(
                    tick, condition.HeldMask, condition.HeldMask,
                    condition.Forward, condition.Strafe, condition.RawYaw,
                    "held_condition", condition.Ordinal, null, false, false, false);
            }
        }

        foreach (var probe in KickProbes)
        {
            if (tick < probe.StartTick || tick >= probe.StopTick)
                continue;
            var kickEdge = tick == probe.EdgeTick;
            var yawPreempted =
                probe.Kind == G1KickProbeKind.YawPreempted && tick >= probe.EdgeTick;
            var translationReleased =
                probe.Kind == G1KickProbeKind.TranslationHeld &&
                probe.TranslationReleaseTick is int releaseTick && tick >= releaseTick;
            var desiredHeldMask = translationReleased
                ? G1HeldMask.None
                : probe.DesiredHeldMask;
            var effectiveHeldMask = yawPreempted
                ? desiredHeldMask & ~(G1HeldMask.Q | G1HeldMask.E)
                : desiredHeldMask;
            return new G1HeldScheduleFrame(
                tick, desiredHeldMask, effectiveHeldMask,
                translationReleased ? (sbyte)0 : probe.Forward,
                translationReleased ? (sbyte)0 : probe.Strafe,
                yawPreempted ? (sbyte)0 : probe.RawYawBeforeEdge,
                probe.Kind == G1KickProbeKind.TranslationHeld
                    ? translationReleased
                        ? "translation_kick_released_observation"
                        : "translation_kick_held"
                    : yawPreempted ? "yaw_kick_preempted" : "yaw_kick_pre_roll",
                null, probe, kickEdge, yawPreempted, translationReleased);
        }

        return new G1HeldScheduleFrame(
            tick, G1HeldMask.None, G1HeldMask.None, 0, 0, 0,
            tick == FinalScheduleTick ? "final_neutral" : "neutral_gap",
            null, null, false, false, false);
    }

    internal static bool IsValidHeldMask(G1HeldMask heldMask)
    {
        var raw = (byte)heldMask;
        if ((raw & ~ValidHeldMask) != 0)
            return false;
        var oppositeForward = heldMask.HasFlag(G1HeldMask.W) && heldMask.HasFlag(G1HeldMask.S);
        var oppositeStrafe = heldMask.HasFlag(G1HeldMask.A) && heldMask.HasFlag(G1HeldMask.D);
        var oppositeYaw = heldMask.HasFlag(G1HeldMask.Q) && heldMask.HasFlag(G1HeldMask.E);
        return !oppositeForward && !oppositeStrafe && !oppositeYaw;
    }

    internal static string ClassifyKickEdge(
        bool executeMoveReturned,
        bool pendingMoveAfterCall,
        int pendingMoveIndexAfterCall,
        int expectedMoveIndex)
    {
        if (pendingMoveAfterCall && pendingMoveIndexAfterCall != expectedMoveIndex)
            return "conflicting_pending_move_after_call";
        if (executeMoveReturned && pendingMoveAfterCall)
            return "accepted_locally_and_armed";
        if (executeMoveReturned)
            return "accepted_locally_without_pending_observation";
        if (pendingMoveAfterCall)
            return "returned_false_but_armed";
        return "rejected_locally";
    }

    internal static G1TranslationProbeClassification ClassifyTranslationProbe(
        bool observationWindowComplete,
        bool? executeMoveReturned,
        string? localClassification,
        bool requestSent,
        bool actionResponseObserved,
        bool motionResponseObserved)
    {
        var response = requestSent
            ? actionResponseObserved || motionResponseObserved
                ? "request_sent_local_diagnostic_response_observed"
                : "request_sent_no_local_diagnostic_response_observed"
            : actionResponseObserved || motionResponseObserved
                ? "no_request_local_diagnostic_response_observed"
                : "no_request_no_local_diagnostic_response_observed";
        var localGate = !observationWindowComplete || executeMoveReturned is null
            ? "unknown"
            : localClassification == "rejected_locally" && !requestSent
                ? "supported"
                : executeMoveReturned == true || requestSent
                    ? "contradicted"
                    : "unknown";
        return new G1TranslationProbeClassification(
            localGate, "unknown", response, TranslationBehaviorCriteria);
    }

    internal static G1KeyboardYawStep AdvanceKeyboardYaw(
        G1KeyboardYawState state,
        float rawYaw,
        float renderedDeltaTime,
        float keyboardYawRampTime,
        float yawSpeed)
    {
        if (!float.IsFinite(rawYaw) || !float.IsFinite(renderedDeltaTime) ||
            !float.IsFinite(keyboardYawRampTime) || !float.IsFinite(yawSpeed) ||
            renderedDeltaTime < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(rawYaw));
        }
        if (keyboardYawRampTime <= 0f || rawYaw == 0f)
        {
            return new G1KeyboardYawStep(
                new G1KeyboardYawState(0f, 0f), rawYaw, rawYaw * yawSpeed);
        }

        var sign = rawYaw > 0f ? 1f : -1f;
        var ramp = state.Sign == sign ? state.Ramp : 0f;
        ramp = MathF.Min(1f, ramp + renderedDeltaTime / keyboardYawRampTime);
        return new G1KeyboardYawStep(
            new G1KeyboardYawState(ramp, sign), rawYaw, ramp * sign * yawSpeed);
    }

    internal static bool HasRoundCapacity(float roundDuration, float timeRemaining) =>
        float.IsFinite(roundDuration) && float.IsFinite(timeRemaining) &&
        roundDuration >= RequiredRoundCapacitySeconds &&
        timeRemaining >= RequiredRoundCapacitySeconds &&
        timeRemaining <= roundDuration;

    internal static int? FixedSubstepsFromSend(
        int? sendPrefixFixedSubstep,
        int? observedFixedSubstep) =>
        sendPrefixFixedSubstep is int send && observedFixedSubstep is int observed &&
        observed >= send
            ? observed - send
            : null;

    internal static int KickObservationStartFixedSubstep(G1KickProbe probe) =>
        probe.EdgeTick * FixedSubstepsPerScheduleTick;

    internal static int KickObservationStopFixedSubstep(G1KickProbe probe) =>
        probe.StopTick * FixedSubstepsPerScheduleTick;

    internal static bool IsKickLifecycleInsideObservationWindow(
        G1KickProbe probe,
        int fixedSubstep) =>
        fixedSubstep >= KickObservationStartFixedSubstep(probe) &&
        fixedSubstep < KickObservationStopFixedSubstep(probe);

    internal static G1KickObservationBoundaryDecision EvaluateKickObservationBoundary(
        G1KickProbe probe,
        int fixedSubstep,
        bool terminalOutcomeObserved)
    {
        if (fixedSubstep < KickObservationStopFixedSubstep(probe))
            return G1KickObservationBoundaryDecision.ContinueObservation;
        return terminalOutcomeObserved
            ? G1KickObservationBoundaryDecision.CompleteObservation
            : G1KickObservationBoundaryDecision.FailPartial;
    }

    internal static string ClassifyTranslationSendTiming(
        int? sendPrefixFixedSubstep,
        int releaseFixedSubstep,
        int? firstTransitionSettledFixedSubstep)
    {
        if (sendPrefixFixedSubstep is not int send)
            return "no_request_send_observed";
        if (send < releaseFixedSubstep)
            return "request_sent_while_translation_held";
        if (firstTransitionSettledFixedSubstep is int settled && send >= settled)
            return "request_sent_after_transition_settled";
        return "request_sent_after_release_before_transition_settled";
    }

    internal static G1KickAsset AssetForMove(int moveIndex) =>
        KickAssets.Single(value => value.MoveIndex == moveIndex);

    internal static string[] HeldNames(G1HeldMask heldMask)
    {
        var result = new List<string>();
        foreach (var value in new[]
                 {
                     G1HeldMask.W, G1HeldMask.S, G1HeldMask.A,
                     G1HeldMask.D, G1HeldMask.Q, G1HeldMask.E,
                 })
        {
            if ((heldMask & value) != 0)
                result.Add(value.ToString());
        }
        return result.ToArray();
    }

    internal static string TranslationOutgoingLocomotion(G1HeldMask heldMask) => heldMask switch
    {
        G1HeldMask.W => "Forward",
        G1HeldMask.S => "Backward",
        G1HeldMask.A => "StrafeLeft",
        G1HeldMask.D => "StrafeRight",
        _ => throw new ArgumentOutOfRangeException(
            nameof(heldMask),
            heldMask,
            "translation probe must map to one exact locomotion direction"),
    };

    internal static string ComputeCanonicalSha256() => Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(CanonicalJson))).ToLowerInvariant();

    private static string BuildCanonicalJson() => JsonSerializer.Serialize(new
    {
        authority_caveat = AuthorityCaveat,
        authority_scope = AuthorityScope,
        duration_ticks = DurationScheduleTicks,
        f_binding_included = false,
        fixed_substeps_per_tick = FixedSubstepsPerScheduleTick,
        held_conditions = HeldConditions.Select(value => new
        {
            held = HeldNames(value.HeldMask),
            held_mask = (byte)value.HeldMask,
            label = value.Label,
            ordinal = value.Ordinal,
            raw_controller_target = new[] { value.Forward, value.Strafe, value.RawYaw },
            start_tick = value.StartTick,
            stop_tick = value.StopTick,
        }).ToArray(),
        held_duration_ticks = HeldDurationTicks,
        keyboard_yaw_ramp = new
        {
            provenance = KeyboardYawRampProvenance,
            ramp_time_seconds = ExpectedKeyboardYawRampTimeSeconds,
            update_phase = "each_rendered_RobotInputController.LateUpdate_prefix",
            yaw_speed = ExpectedYawSpeed,
        },
        kick_assets = KickAssets.Select(value => new
        {
            controller_ticks = value.ControllerTicks,
            move_index = value.MoveIndex,
            npz_sha256 = value.NpzSha256,
            runtime_name = value.RuntimeName,
        }).ToArray(),
        kick_dispatch_start_margin_ticks = KickDispatchStartMarginTicks,
        kick_move_indices = KickMoveIndices,
        kick_observation_ticks = KickObservationTicks,
        kick_retry_policy = KickRetryPolicy,
        max_recovered_kick_controller_ticks = MaxRecoveredKickControllerTicks,
        motion_identity_status = MotionIdentityStatus,
        neutral_gap_ticks = NeutralGapTicks,
        pose_response_source = PoseResponseSource,
        post_release_settled_kick_control_included = false,
        probes = KickProbes.Select(value => new
        {
            desired_held = HeldNames(value.DesiredHeldMask),
            desired_held_mask = (byte)value.DesiredHeldMask,
            edge_tick = value.EdgeTick,
            kind = value.Kind == G1KickProbeKind.TranslationHeld
                ? "translation_held" : "yaw_preempted",
            label = value.Label,
            move_index = value.MoveIndex,
            ordinal = value.Ordinal,
            raw_controller_target_before_edge =
                new[] { value.Forward, value.Strafe, value.RawYawBeforeEdge },
            start_tick = value.StartTick,
            stop_tick = value.StopTick,
            translation_release_tick = value.TranslationReleaseTick,
            transition_settled_outgoing_locomotion =
                value.Kind == G1KickProbeKind.TranslationHeld
                    ? TranslationOutgoingLocomotion(value.DesiredHeldMask)
                    : null,
        }).ToArray(),
        required_isolation_proof = RequiredIsolationProof,
        required_round_capacity_seconds = RequiredRoundCapacitySeconds,
        required_runtime_pairing = "exact_g1_vs_g1_bone_signatures",
        round_capacity_safety_seconds = RoundCapacitySafetySeconds,
        schedule_id = ScheduleId,
        schedule_rate_hz = ScheduleRateHz,
        schema = Schema,
        sonic_action_composer_lifecycle_used = false,
        translation_behavior_criteria = TranslationBehaviorCriteria,
        translation_release_offset_ticks = TranslationReleaseOffsetTicks,
        transition_settle_base_velocity_provenance =
            TransitionSettleBaseVelocityProvenance,
        transition_settle_planar_speed_m_s = ExpectedTransitionSettlePlanarSpeed,
        transition_settle_yaw_rate_rad_s = ExpectedTransitionSettleYawRate,
        transition_settled_provenance = TransitionSettledProvenance,
        unity_fixed_rate_hz = UnityFixedRateHz,
    });
}
