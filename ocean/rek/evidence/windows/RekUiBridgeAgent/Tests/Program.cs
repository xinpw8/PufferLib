using System.IO.Pipes;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using RekEvidence;
using RekUiBridgeAgent;

var failures = new ConcurrentQueue<string>();
var protocolCases = 0;

void Expect(string name, bool condition)
{
    protocolCases++;
    if (!condition)
        failures.Enqueue(name);
}

void ExpectInvalidData(string name, Action action)
{
    protocolCases++;
    try
    {
        action();
        failures.Enqueue(name);
    }
    catch (InvalidDataException)
    {
    }
    catch (Exception exception)
    {
        failures.Enqueue($"{name}:unexpected_{exception.GetType().Name}");
    }
}

Expect("unity_fixed_rate", BridgeScheduleContract.UnityFixedRateHz == 500);
Expect("schedule_rate", BridgeScheduleContract.ScheduleRateHz == 50);
Expect("schedule_decimation", BridgeScheduleContract.FixedSubstepsPerScheduleTick == 10);
Expect(
    "schedule_duration",
    BridgeScheduleContract.DurationScheduleTicks / (double)BridgeScheduleContract.ScheduleRateHz == 52.02);
Expect("final_schedule_tick", BridgeScheduleContract.FinalScheduleTick == 2600);
Expect(
    "fixed_delta_time",
    BridgeScheduleContract.ExpectedFixedDeltaTime * BridgeScheduleContract.UnityFixedRateHz == 1.0f);
Expect(
    "schedule_sha256",
    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        BridgeScheduleContract.CanonicalJson))).ToLowerInvariant() ==
        BridgeScheduleContract.ExpectedSha256);
Expect(
    "unproven_unexpected_round_reset_not_exposed",
    !Enum.GetNames<BridgeCommand>().Contains(
        "StartUnexpectedPrivateAiRound",
        StringComparer.Ordinal));

Expect("g1_held_schema", G1HeldInputScheduleContract.Schema == "rek.g1_held_input_schedule.v2");
Expect("g1_held_fixed_rate", G1HeldInputScheduleContract.UnityFixedRateHz == 500);
Expect("g1_held_rate", G1HeldInputScheduleContract.ScheduleRateHz == 50);
Expect("g1_held_decimation", G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick == 10);
Expect("g1_held_duration", G1HeldInputScheduleContract.DurationScheduleTicks == 4551);
Expect("g1_held_final_tick", G1HeldInputScheduleContract.FinalScheduleTick == 4550);
Expect("g1_held_condition_count", G1HeldInputScheduleContract.HeldConditions.Length == 14);
Expect("g1_held_each_condition_two_seconds", G1HeldInputScheduleContract.HeldConditions.All(value =>
    value.StopTick - value.StartTick == 100));
Expect("g1_held_condition_labels", G1HeldInputScheduleContract.HeldConditions
    .Select(value => value.Label).SequenceEqual(new[]
    {
        "W", "S", "A", "D", "Q", "E", "W+Q", "W+E", "S+Q", "S+E",
        "A+Q", "A+E", "D+Q", "D+E",
    }));
Expect("g1_held_translation_yaw_overlap_count", G1HeldInputScheduleContract.HeldConditions.Count(value =>
    (((byte)value.HeldMask & G1HeldInputScheduleContract.TranslationMask) != 0) &&
    (((byte)value.HeldMask & G1HeldInputScheduleContract.YawMask) != 0)) == 8);
Expect("g1_held_kicks_only", G1HeldInputScheduleContract.KickMoveIndices.SequenceEqual(
    new[] { 6, 7, 8, 9 }));
Expect("g1_held_no_f_binding", G1HeldInputScheduleContract.HeldNames(
    (G1HeldMask)G1HeldInputScheduleContract.ValidHeldMask).All(value => value != "F"));
Expect("g1_held_invalid_unknown_bit", !G1HeldInputScheduleContract.IsValidHeldMask((G1HeldMask)64));
Expect("g1_held_probe_count", G1HeldInputScheduleContract.KickProbes.Length == 8);
Expect("g1_held_translation_probe_count", G1HeldInputScheduleContract.KickProbes.Count(value =>
    value.Kind == G1KickProbeKind.TranslationHeld) == 4);
Expect("g1_held_yaw_probe_count", G1HeldInputScheduleContract.KickProbes.Count(value =>
    value.Kind == G1KickProbeKind.YawPreempted) == 4);
Expect("g1_held_each_move_has_translation_and_control", G1HeldInputScheduleContract.KickMoveIndices.All(move =>
    G1HeldInputScheduleContract.KickProbes.Count(value => value.MoveIndex == move) == 2));
Expect("g1_held_all_windows_cover_max_clip_and_margin",
    G1HeldInputScheduleContract.KickProbes.All(value =>
        value.StopTick - value.EdgeTick >=
        G1HeldInputScheduleContract.MaxRecoveredKickControllerTicks +
        G1HeldInputScheduleContract.KickDispatchStartMarginTicks));
Expect("g1_held_all_windows_exact_200_ticks",
    G1HeldInputScheduleContract.KickProbes.All(value =>
        value.StopTick - value.EdgeTick ==
        G1HeldInputScheduleContract.KickObservationTicks));
Expect("g1_held_translation_release_offset_100ms",
    G1HeldInputScheduleContract.TranslationReleaseOffsetTicks == 5 &&
    G1HeldInputScheduleContract.KickProbes
        .Where(value => value.Kind == G1KickProbeKind.TranslationHeld)
        .All(value => value.TranslationReleaseTick ==
                      value.EdgeTick +
                      G1HeldInputScheduleContract.TranslationReleaseOffsetTicks));
Expect("g1_held_translation_total_hold_1_1s",
    G1HeldInputScheduleContract.KickProbes
        .Where(value => value.Kind == G1KickProbeKind.TranslationHeld)
        .All(value => value.TranslationReleaseTick - value.StartTick == 55));
Expect("g1_held_yaw_has_no_translation_release",
    G1HeldInputScheduleContract.KickProbes
        .Where(value => value.Kind == G1KickProbeKind.YawPreempted)
        .All(value => value.TranslationReleaseTick is null));
Expect("g1_held_exact_translation_direction_mapping",
    G1HeldInputScheduleContract.TranslationOutgoingLocomotion(G1HeldMask.W) == "Forward" &&
    G1HeldInputScheduleContract.TranslationOutgoingLocomotion(G1HeldMask.S) == "Backward" &&
    G1HeldInputScheduleContract.TranslationOutgoingLocomotion(G1HeldMask.A) == "StrafeLeft" &&
    G1HeldInputScheduleContract.TranslationOutgoingLocomotion(G1HeldMask.D) == "StrafeRight");
Expect("g1_held_probe_windows_nonoverlap",
    G1HeldInputScheduleContract.KickProbes.Zip(
        G1HeldInputScheduleContract.KickProbes.Skip(1),
        (left, right) => left.StopTick <= right.StartTick).All(value => value));
Expect("g1_held_recovered_asset_lengths",
    G1HeldInputScheduleContract.KickAssets.Select(value => value.ControllerTicks)
        .SequenceEqual(new[] { 158, 146, 159, 140 }));
Expect("g1_held_translation_edges_remain_held", G1HeldInputScheduleContract.KickProbes
    .Where(value => value.Kind == G1KickProbeKind.TranslationHeld)
    .All(value => G1HeldInputScheduleContract.FrameAtTick(value.EdgeTick).EffectiveHeldMask ==
                  value.DesiredHeldMask));
Expect("g1_held_translation_releases_to_neutral_after_edge",
    G1HeldInputScheduleContract.KickProbes
        .Where(value => value.Kind == G1KickProbeKind.TranslationHeld)
        .All(value =>
        {
            var releaseTick = value.TranslationReleaseTick!.Value;
            var before = G1HeldInputScheduleContract.FrameAtTick(releaseTick - 1);
            var released = G1HeldInputScheduleContract.FrameAtTick(releaseTick);
            return before.DesiredHeldMask == value.DesiredHeldMask &&
                   !before.TranslationReleased &&
                   released.DesiredHeldMask == G1HeldMask.None &&
                   released.EffectiveHeldMask == G1HeldMask.None &&
                   released.Forward == 0 && released.Strafe == 0 &&
                   released.TranslationReleased &&
                   released.Phase == "translation_kick_released_observation";
        }));
Expect("g1_held_yaw_edges_are_preempted", G1HeldInputScheduleContract.KickProbes
    .Where(value => value.Kind == G1KickProbeKind.YawPreempted)
    .All(value =>
    {
        var frame = G1HeldInputScheduleContract.FrameAtTick(value.EdgeTick);
        return frame.DesiredHeldMask == value.DesiredHeldMask &&
               frame.EffectiveHeldMask == G1HeldMask.None && frame.RawYaw == 0 &&
               frame.KickEdge && frame.YawPreempted;
    }));
Expect("g1_held_final_neutral", G1HeldInputScheduleContract.FrameAtTick(
    G1HeldInputScheduleContract.FinalScheduleTick) is
    { DesiredHeldMask: G1HeldMask.None, EffectiveHeldMask: G1HeldMask.None,
      Forward: 0, Strafe: 0, RawYaw: 0, Phase: "final_neutral" });
Expect("g1_held_classifies_local_reject", G1HeldInputScheduleContract.ClassifyKickEdge(
    false, false, 0, 6) == "rejected_locally");
Expect("g1_held_classifies_local_defer", G1HeldInputScheduleContract.ClassifyKickEdge(
    true, true, 6, 6) == "accepted_locally_and_armed");
var rejectedTranslation = G1HeldInputScheduleContract.ClassifyTranslationProbe(
    true, false, "rejected_locally", false, false, false);
Expect("g1_translation_explicit_local_reject_supported",
    rejectedTranslation.LocalGateResult == "supported" &&
    rejectedTranslation.BehavioralResult == "unknown");
var sentTranslation = G1HeldInputScheduleContract.ClassifyTranslationProbe(
    true, true, "accepted_locally_and_armed", true, false, false);
Expect("g1_translation_sent_contradicts_local_gate_only",
    sentTranslation.LocalGateResult == "contradicted" &&
    sentTranslation.BehavioralResult == "unknown" &&
    sentTranslation.ResponseClassification ==
        "request_sent_no_local_diagnostic_response_observed");
var incompleteTranslation = G1HeldInputScheduleContract.ClassifyTranslationProbe(
    false, false, "rejected_locally", false, false, false);
Expect("g1_translation_incomplete_unknown",
    incompleteTranslation.LocalGateResult == "unknown" &&
    incompleteTranslation.BehavioralResult == "unknown");
Expect("g1_translation_send_while_held",
    G1HeldInputScheduleContract.ClassifyTranslationSendTiming(
        22_049, 22_050, 22_100) == "request_sent_while_translation_held");
Expect("g1_translation_send_after_release_before_settle",
    G1HeldInputScheduleContract.ClassifyTranslationSendTiming(
        22_075, 22_050, 22_100) ==
        "request_sent_after_release_before_transition_settled");
Expect("g1_translation_send_after_settle",
    G1HeldInputScheduleContract.ClassifyTranslationSendTiming(
        22_100, 22_050, 22_100) ==
        "request_sent_after_transition_settled");
Expect("g1_translation_no_send_observed",
    G1HeldInputScheduleContract.ClassifyTranslationSendTiming(
        null, 22_050, 22_100) == "no_request_send_observed");
var yawState = new G1KeyboardYawState(0f, 0f);
var yawStep = G1HeldInputScheduleContract.AdvanceKeyboardYaw(
    yawState, 1f, 0.02f, 0.5f, 1f);
Expect("g1_yaw_first_rendered_ramp_step",
    yawStep.State.Sign == 1f && Math.Abs(yawStep.State.Ramp - 0.04f) < 1e-6f &&
    Math.Abs(yawStep.EffectiveYaw - 0.04f) < 1e-6f);
for (var renderedFrame = 1; renderedFrame < 25; renderedFrame++)
{
    yawStep = G1HeldInputScheduleContract.AdvanceKeyboardYaw(
        yawStep.State, 1f, 0.02f, 0.5f, 1f);
}
Expect("g1_yaw_ramp_clamps_at_one",
    yawStep.State.Ramp == 1f && yawStep.EffectiveYaw == 1f);
var reversedYaw = G1HeldInputScheduleContract.AdvanceKeyboardYaw(
    yawStep.State, -1f, 0.02f, 0.5f, 1f);
Expect("g1_yaw_sign_change_resets_then_ramps",
    reversedYaw.State.Sign == -1f &&
    Math.Abs(reversedYaw.State.Ramp - 0.04f) < 1e-6f &&
    Math.Abs(reversedYaw.EffectiveYaw + 0.04f) < 1e-6f);
var releasedYaw = G1HeldInputScheduleContract.AdvanceKeyboardYaw(
    reversedYaw.State, 0f, 0.02f, 0.5f, 1f);
Expect("g1_yaw_release_resets",
    releasedYaw.State == new G1KeyboardYawState(0f, 0f) &&
    releasedYaw.EffectiveYaw == 0f);
var dispatch = G1KickDispatchTracker.Arm(22_000);
for (var fixedUpdate = 0; fixedUpdate < 100; fixedUpdate++)
    dispatch = dispatch.OnFixedSubstep();
Expect("g1_dispatch_many_fixed_updates_do_not_cancel",
    dispatch.Armed && dispatch.LateUpdateOpportunities == 0 &&
    !dispatch.MissingDispatchAfterCompletedOpportunity);
dispatch = dispatch.OnLateUpdatePrefix(matchingPending: true);
Expect("g1_dispatch_not_failed_before_late_update_returns",
    dispatch.LateUpdateOpen && !dispatch.MissingDispatchAfterCompletedOpportunity);
dispatch = dispatch.OnSendPrefix().OnSendPostfix().OnLateUpdatePostfix();
Expect("g1_dispatch_late_send_completes",
    dispatch.SendPrefixSeen && dispatch.SendPostfixSeen &&
    !dispatch.MissingDispatchAfterCompletedOpportunity);
var missingDispatch = G1KickDispatchTracker.Arm(22_000)
    .OnLateUpdatePrefix(matchingPending: true)
    .OnLateUpdatePostfix();
Expect("g1_dispatch_missing_only_after_completed_opportunity",
    missingDispatch.MissingDispatchAfterCompletedOpportunity);
var firstKickProbe = G1HeldInputScheduleContract.KickProbes[0];
var firstKickStartSubstep =
    G1HeldInputScheduleContract.KickObservationStartFixedSubstep(firstKickProbe);
var firstKickStopSubstep =
    G1HeldInputScheduleContract.KickObservationStopFixedSubstep(firstKickProbe);
Expect("g1_kick_lifecycle_window_half_open",
    G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
        firstKickProbe,
        firstKickStartSubstep) &&
    G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
        firstKickProbe,
        firstKickStopSubstep - 1) &&
    !G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
        firstKickProbe,
        firstKickStartSubstep - 1) &&
    !G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
        firstKickProbe,
        firstKickStopSubstep));
Expect("g1_pending_at_observation_boundary_fails_partial",
    G1HeldInputScheduleContract.EvaluateKickObservationBoundary(
        firstKickProbe,
        firstKickStopSubstep,
        terminalOutcomeObserved: false) ==
    G1KickObservationBoundaryDecision.FailPartial);
Expect("g1_pending_before_observation_boundary_keeps_waiting",
    G1HeldInputScheduleContract.EvaluateKickObservationBoundary(
        firstKickProbe,
        firstKickStopSubstep - 1,
        terminalOutcomeObserved: false) ==
    G1KickObservationBoundaryDecision.ContinueObservation);
Expect("g1_terminal_at_observation_boundary_completes",
    G1HeldInputScheduleContract.EvaluateKickObservationBoundary(
        firstKickProbe,
        firstKickStopSubstep,
        terminalOutcomeObserved: true) ==
    G1KickObservationBoundaryDecision.CompleteObservation);
var staleCompleteSummary = new G1HeldEventReconciler();
staleCompleteSummary.ObserveProbeEvent(
    "g1_kick_request_lifecycle:pending_awaiting_render_dispatch_opportunity",
    firstKickProbe.Ordinal);
ExpectInvalidData(
    "g1_client_rejects_complete_summary_before_terminal",
    () => staleCompleteSummary.ObserveSummary(
        firstKickProbe.Ordinal,
        observationWindowComplete: true));
var partialSummaryThenLateDispatch = new G1HeldEventReconciler();
partialSummaryThenLateDispatch.ObserveSummary(
    firstKickProbe.Ordinal,
    observationWindowComplete: false);
ExpectInvalidData(
    "g1_client_rejects_late_render_dispatch_after_partial_summary",
    () => partialSummaryThenLateDispatch.ObserveProbeEvent(
        "g1_kick_request_lifecycle:send_move_invoked",
        firstKickProbe.Ordinal));
var terminalSummaryThenLateDispatch = new G1HeldEventReconciler();
terminalSummaryThenLateDispatch.ObserveTerminal(firstKickProbe.Ordinal);
terminalSummaryThenLateDispatch.ObserveSummary(
    firstKickProbe.Ordinal,
    observationWindowComplete: true);
ExpectInvalidData(
    "g1_client_rejects_dispatch_event_after_complete_summary",
    () => terminalSummaryThenLateDispatch.ObserveProbeEvent(
        "g1_kick_dispatch_opportunity",
        firstKickProbe.Ordinal));
G1HeldEventReconciler.ValidateSendAnchor(
    firstKickProbe,
    firstKickStartSubstep,
    firstKickProbe.EdgeTick);
G1HeldEventReconciler.ValidateSendAnchor(
    firstKickProbe,
    firstKickStartSubstep + G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick,
    firstKickProbe.EdgeTick);
G1HeldEventReconciler.ValidateSendAnchor(
    firstKickProbe,
    firstKickStopSubstep - 1,
    firstKickProbe.StopTick - 1);
Expect("g1_client_accepts_send_anchors_inside_window", true);
ExpectInvalidData(
    "g1_client_rejects_send_anchor_at_stop_boundary",
    () => G1HeldEventReconciler.ValidateSendAnchor(
        firstKickProbe,
        firstKickStopSubstep,
        firstKickProbe.StopTick));
ExpectInvalidData(
    "g1_client_rejects_send_anchor_with_stale_schedule_tick",
    () => G1HeldEventReconciler.ValidateSendAnchor(
        firstKickProbe,
        firstKickStopSubstep - 1,
        firstKickProbe.StopTick));
Expect("g1_timing_is_send_anchored",
    G1HeldInputScheduleContract.FixedSubstepsFromSend(22_007, 22_019) == 12 &&
    G1HeldInputScheduleContract.FixedSubstepsFromSend(null, 22_019) is null &&
    G1HeldInputScheduleContract.FixedSubstepsFromSend(22_020, 22_019) is null);
Expect("g1_round_capacity_preflight",
    G1HeldInputScheduleContract.HasRoundCapacity(120f, 110f) &&
    !G1HeldInputScheduleContract.HasRoundCapacity(120f, 100f));
using (var binary32RoundTrip = JsonDocument.Parse(JsonSerializer.Serialize(new
       {
           fixed_delta_time = G1HeldInputScheduleContract.ExpectedFixedDeltaTime,
           transition_settle_planar_speed_m_s =
               G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed,
           transition_settle_yaw_rate_rad_s =
               G1HeldInputScheduleContract.ExpectedTransitionSettleYawRate,
           keyboard_yaw_ramp_time_seconds =
               G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds,
           keyboard_yaw_speed = G1HeldInputScheduleContract.ExpectedYawSpeed,
           raw_yaw_target = -1f,
           round_capacity_safety_seconds =
               G1HeldInputScheduleContract.RoundCapacitySafetySeconds,
       }, BridgeJson.Options)))
{
    var root = binary32RoundTrip.RootElement;
    var roundTripAccepted = true;
    try
    {
        JsonBinary32Contract.RequireExact(
            root,
            "fixed_delta_time",
            G1HeldInputScheduleContract.ExpectedFixedDeltaTime);
        JsonBinary32Contract.RequireExact(
            root,
            "transition_settle_planar_speed_m_s",
            G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed);
        JsonBinary32Contract.RequireExact(
            root,
            "transition_settle_yaw_rate_rad_s",
            G1HeldInputScheduleContract.ExpectedTransitionSettleYawRate);
        JsonBinary32Contract.RequireExact(
            root,
            "keyboard_yaw_ramp_time_seconds",
            G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds);
        JsonBinary32Contract.RequireExact(
            root,
            "keyboard_yaw_speed",
            G1HeldInputScheduleContract.ExpectedYawSpeed);
        JsonBinary32Contract.RequireExact(root, "raw_yaw_target", -1f);
        JsonBinary32Contract.RequireExact(
            root,
            "round_capacity_safety_seconds",
            G1HeldInputScheduleContract.RoundCapacitySafetySeconds);
    }
    catch
    {
        roundTripAccepted = false;
    }
    Expect("g1_json_binary32_actual_serializer_roundtrip", roundTripAccepted);
    var planarToken = root.GetProperty("transition_settle_planar_speed_m_s");
    Expect("g1_json_binary32_reproduces_old_double_promotion_mismatch",
        planarToken.GetDouble() !=
            (double)G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed &&
        BitConverter.SingleToInt32Bits(planarToken.GetSingle()) ==
            BitConverter.SingleToInt32Bits(
                G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed));
}
var adjacentPlanarSpeed = BitConverter.Int32BitsToSingle(
    BitConverter.SingleToInt32Bits(
        G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed) + 1);
using (var adjacentBinary32 = JsonDocument.Parse(JsonSerializer.Serialize(new
       {
           transition_settle_planar_speed_m_s = adjacentPlanarSpeed,
       }, BridgeJson.Options)))
{
    ExpectInvalidData(
        "g1_json_binary32_rejects_adjacent_float",
        () => JsonBinary32Contract.RequireExact(
            adjacentBinary32.RootElement,
            "transition_settle_planar_speed_m_s",
            G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed));
}
Expect("g1_held_sha256", G1HeldInputScheduleContract.ComputeCanonicalSha256() ==
    G1HeldInputScheduleContract.ExpectedSha256);
using (var g1HeldManifest = JsonDocument.Parse(G1HeldInputScheduleContract.CanonicalJson))
{
    var root = g1HeldManifest.RootElement;
    Expect("g1_held_manifest_no_f", !root.GetProperty("f_binding_included").GetBoolean());
    Expect("g1_held_manifest_no_composer",
        !root.GetProperty("sonic_action_composer_lifecycle_used").GetBoolean());
    Expect("g1_held_manifest_no_retry", root.GetProperty("kick_retry_policy").GetString() ==
        "one_edge_no_queue_no_retry");
    Expect("g1_held_manifest_isolation", root.GetProperty("required_isolation_proof").GetString() ==
        G1HeldInputScheduleContract.RequiredIsolationProof);
}

var trialSelectors = SingleMotionTrialContract.Selectors;
Expect("single_trial_schema", SingleMotionTrialContract.Schema == "rek.single_motion_trial.v1");
Expect("single_trial_authority_scope", SingleMotionTrialContract.AuthorityScope == "client_request_edges_only");
Expect(
    "single_trial_authority_caveat",
    SingleMotionTrialContract.AuthorityCaveat ==
    "client request edge observed; server acceptance and authoritative execution are unknown");
Expect("single_trial_fixed_rate", SingleMotionTrialContract.UnityFixedRateHz == 500);
Expect("single_trial_rate", SingleMotionTrialContract.TrialRateHz == 50);
Expect("single_trial_decimation", SingleMotionTrialContract.FixedSubstepsPerTrialTick == 10);
Expect("single_trial_pre_roll", SingleMotionTrialContract.NeutralPreRollTicks == 50);
Expect("single_trial_action_tick", SingleMotionTrialContract.ActionTick == 50);
Expect("single_trial_release_tick", SingleMotionTrialContract.LocomotionReleaseTick == 100);
Expect("single_trial_duration", SingleMotionTrialContract.DurationTrialTicks == 250);
Expect("single_trial_final_tick", SingleMotionTrialContract.FinalTrialTick == 249);
Expect(
    "single_trial_sha256",
    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        SingleMotionTrialContract.CanonicalJson))).ToLowerInvariant() ==
        SingleMotionTrialContract.ExpectedSha256);
Expect("single_trial_selector_count", trialSelectors.Length == 12);
Expect(
    "single_trial_selector_order",
    trialSelectors.Select(value => value.Selector).SequenceEqual(new[]
    {
        "forward", "backward", "strafe-left", "strafe-right", "yaw-left", "yaw-right",
        "move-2", "move-3", "move-4", "move-5", "move-9", "move-10",
    }));
Expect("single_trial_selector_unique", trialSelectors.Select(value => value.Selector).Distinct(StringComparer.Ordinal).Count() == 12);
Expect("single_trial_try_get_null_rejected", !SingleMotionTrialContract.TryGet(null, out _));
Expect("single_trial_try_get_empty_rejected", !SingleMotionTrialContract.TryGet(string.Empty, out _));
Expect("single_trial_locomotion_count", trialSelectors.Count(value => value.IsLocomotion) == 6);
Expect(
    "single_trial_move_slots",
    trialSelectors.Where(value => !value.IsLocomotion).Select(value => value.MoveIndex).SequenceEqual(
        new int?[] { 2, 3, 4, 5, 9, 10 }));
Expect("single_trial_vectors_nonzero", trialSelectors.Where(value => value.IsLocomotion).All(value =>
    Math.Abs(value.Forward) + Math.Abs(value.Strafe) + Math.Abs(value.Yaw) == 1f));
Expect("single_trial_move_vectors_neutral", trialSelectors.Where(value => !value.IsLocomotion).All(value =>
    value.Forward == 0f && value.Strafe == 0f && value.Yaw == 0f));
Expect("single_trial_strafe_left_native_positive", trialSelectors.Single(value =>
    value.Selector == "strafe-left").Strafe == 1f);
Expect("single_trial_strafe_right_native_negative", trialSelectors.Single(value =>
    value.Selector == "strafe-right").Strafe == -1f);
Expect("single_trial_yaw_left_native_positive", trialSelectors.Single(value =>
    value.Selector == "yaw-left").Yaw == 1f);
Expect("single_trial_yaw_right_native_negative", trialSelectors.Single(value =>
    value.Selector == "yaw-right").Yaw == -1f);
using (var trialDocument = JsonDocument.Parse(SingleMotionTrialContract.CanonicalJson))
{
    var root = trialDocument.RootElement;
    Expect("single_trial_manifest_duration", root.GetProperty("duration_ticks").GetInt32() == 250);
    Expect("single_trial_manifest_caveat", root.GetProperty("authority_caveat").GetString() ==
        SingleMotionTrialContract.AuthorityCaveat);
    var manifestSelectors = root.GetProperty("selectors").EnumerateArray().ToArray();
    Expect("single_trial_manifest_selector_count", manifestSelectors.Length == trialSelectors.Length);
    for (var index = 0; index < Math.Min(manifestSelectors.Length, trialSelectors.Length); index++)
    {
        var manifest = manifestSelectors[index];
        var selector = trialSelectors[index];
        Expect($"single_trial_manifest_selector_{index}",
            manifest.GetProperty("selector").GetString() == selector.Selector &&
            manifest.GetProperty("kind").GetString() == selector.Kind &&
            manifest.GetProperty("command_identity").GetString() == selector.CommandIdentity);
    }
}

var continuousAttacks = ContinuousBotControllerContract.Attacks;
var g1ContinuousAttacks = ContinuousBotControllerContract.G1Attacks;
Expect(
    "continuous_schema",
    ContinuousBotControllerContract.Schema == "rek.continuous_private_bot_controller.v1");
var soloRoute = new SoloRouteProofTracker();
var initialSoloRoute = soloRoute.SnapshotForArena("arena-private-1");
Expect(
    "solo_route_initially_unproven_and_server_privacy_unknown",
    !initialSoloRoute.SoloRouteProven &&
    !initialSoloRoute.ServerPrivateProven &&
    initialSoloRoute.ServerPrivateStatus == "unknown");
soloRoute.ObserveFindMatch("solo");
soloRoute.ObserveConnectToArena("arena-private-1");
soloRoute.ObserveEnterChampionship(
    "arena-private-1", "test.invalid", 7777, koth: false, solo: true);
var provenSoloRoute = soloRoute.SnapshotForArena("arena-private-1");
Expect(
    "solo_route_exact_chain_proven_without_server_privacy_claim",
    provenSoloRoute.SoloRouteProven &&
    provenSoloRoute.FlowSoloObserved &&
    provenSoloRoute.ConnectToArenaObserved &&
    provenSoloRoute.EnterChampionshipObserved &&
    provenSoloRoute.EnterChampionshipKoth == false &&
    provenSoloRoute.EnterChampionshipSolo == true &&
    provenSoloRoute.ArenaIdentityConsistent &&
    !provenSoloRoute.ServerPrivateProven &&
    provenSoloRoute.ServerPrivateStatus == "unknown" &&
    provenSoloRoute.Reason == SoloRouteProofContract.ProvenReason);
var boundSoloRoute = soloRoute.SnapshotForRuntimeSession(
    "arena-private-1",
    "test.invalid",
    7777,
    "test.invalid",
    7777,
    runtimeSessionIdentity: 101);
Expect(
    "solo_route_runtime_session_binding_proven",
    boundSoloRoute.SoloRouteProven &&
    boundSoloRoute.RuntimeSessionIdentityConsistent);
Expect(
    "solo_route_scope_rejects_unbound_runtime_session",
    !SoloRouteProofContract.EvaluateScope(
        exactBotOneNoHumanProofEstablished: true,
        provenSoloRoute).Allowed);
Expect(
    "solo_route_scope_requires_exact_bot_one_no_human_proof",
    !SoloRouteProofContract.EvaluateScope(
        exactBotOneNoHumanProofEstablished: false,
        boundSoloRoute).Allowed);
Expect(
    "solo_route_scope_accepts_exact_chain_and_bot_one_no_human_proof",
    SoloRouteProofContract.EvaluateScope(
        exactBotOneNoHumanProofEstablished: true,
        boundSoloRoute).Allowed);
Expect(
    "solo_route_runtime_session_replacement_rejected",
    !soloRoute.SnapshotForRuntimeSession(
        "arena-private-1",
        "test.invalid",
        7777,
        "test.invalid",
        7777,
        runtimeSessionIdentity: 102).SoloRouteProven);
Expect(
    "solo_route_runtime_endpoint_change_rejected",
    !soloRoute.SnapshotForRuntimeSession(
        "arena-private-1",
        "test.invalid",
        7777,
        "changed.invalid",
        7777,
        runtimeSessionIdentity: 101).SoloRouteProven);
Expect(
    "solo_route_bound_network_lifecycle_change_invalidates",
    soloRoute.InvalidateIfRuntimeSessionBound(
        "network_client_disconnected_after_solo_route_binding") &&
    !soloRoute.SnapshotForRuntimeSession(
        "arena-private-1",
        "test.invalid",
        7777,
        "test.invalid",
        7777,
        runtimeSessionIdentity: 101).SoloRouteProven);
Expect(
    "solo_route_current_arena_change_rejected",
    !soloRoute.SnapshotForArena("arena-private-2").SoloRouteProven);

var unboundLifecycleRoute = new SoloRouteProofTracker();
unboundLifecycleRoute.ObserveFindMatch("solo");
unboundLifecycleRoute.ObserveConnectToArena("arena-private-1");
unboundLifecycleRoute.ObserveEnterChampionship(
    "arena-private-1",
    "test.invalid",
    7777,
    koth: false,
    solo: true);
Expect(
    "initial_network_connection_does_not_invalidate_unbound_route",
    !unboundLifecycleRoute.InvalidateIfRuntimeSessionBound(
        "network_client_connected_after_solo_route_binding") &&
    unboundLifecycleRoute.SnapshotForArena("arena-private-1").SoloRouteProven);
Expect(
    "transient_startup_scope_failure_does_not_erase_unbound_route",
    !unboundLifecycleRoute.InvalidateIfRuntimeSessionBound(
        "client_only_connected_network_session_not_proven") &&
    unboundLifecycleRoute.SnapshotForArena("arena-private-1").SoloRouteProven);

var publicFlow = new SoloRouteProofTracker();
publicFlow.ObserveFindMatch("koth");
publicFlow.ObserveConnectToArena("arena-public-1");
publicFlow.ObserveEnterChampionship(
    "arena-public-1", "test.invalid", 7777, koth: true, solo: false);
Expect(
    "non_solo_flow_rejected",
    !publicFlow.SnapshotForArena("arena-public-1").SoloRouteProven);

var missingConnect = new SoloRouteProofTracker();
missingConnect.ObserveFindMatch("solo");
missingConnect.ObserveEnterChampionship(
    "arena-private-1", "test.invalid", 7777, koth: false, solo: true);
Expect(
    "solo_route_without_connect_rejected",
    !missingConnect.SnapshotForArena("arena-private-1").SoloRouteProven);

var mismatchedArena = new SoloRouteProofTracker();
mismatchedArena.ObserveFindMatch("solo");
mismatchedArena.ObserveConnectToArena("arena-private-1");
mismatchedArena.ObserveEnterChampionship(
    "arena-private-2", "test.invalid", 7777, koth: false, solo: true);
Expect(
    "solo_route_mismatched_arena_rejected",
    !mismatchedArena.SnapshotForArena("arena-private-2").SoloRouteProven);

var wrongMode = new SoloRouteProofTracker();
wrongMode.ObserveFindMatch("solo");
wrongMode.ObserveConnectToArena("arena-private-1");
wrongMode.ObserveEnterChampionship(
    "arena-private-1", "test.invalid", 7777, koth: true, solo: true);
Expect(
    "solo_route_koth_flag_rejected",
    !wrongMode.SnapshotForArena("arena-private-1").SoloRouteProven);
wrongMode.ObserveEnterChampionship(
    "arena-private-1", "test.invalid", 7777, koth: false, solo: true);
Expect(
    "invalid_enter_requires_fresh_find_and_connect",
    !wrongMode.SnapshotForArena("arena-private-1").SoloRouteProven);
wrongMode.ObserveFindMatch("solo");
wrongMode.ObserveConnectToArena("arena-private-1");
wrongMode.ObserveEnterChampionship(
    "arena-private-1", "test.invalid", 7777, koth: false, solo: false);
Expect(
    "solo_route_missing_solo_flag_rejected",
    !wrongMode.SnapshotForArena("arena-private-1").SoloRouteProven);

soloRoute.ObserveFindMatch("Solo");
Expect(
    "solo_route_flow_is_case_sensitive_and_new_flow_invalidates_prior_proof",
    !soloRoute.SnapshotForArena("arena-private-1").SoloRouteProven);
Expect(
    "continuous_authority_scope",
    ContinuousBotControllerContract.AuthorityScope ==
    "client_request_edges_and_local_observations_only");
Expect(
    "continuous_request_only_caveat",
    ContinuousBotControllerContract.AuthorityCaveat.Contains(
        "server acceptance and authoritative execution are unknown",
        StringComparison.Ordinal));
Expect(
    "continuous_client_default_bounded_run_mode",
    ControllerRunModeContract.TryParse(null, out var defaultControllerRunMode) &&
    !defaultControllerRunMode.UntilEnded &&
    defaultControllerRunMode.RunSeconds == 120);
Expect(
    "continuous_client_explicit_bounded_run_mode",
    ControllerRunModeContract.TryParse("600", out var boundedControllerRunMode) &&
    !boundedControllerRunMode.UntilEnded &&
    boundedControllerRunMode.RunSeconds == 600);
Expect(
    "continuous_client_persistent_until_ended_mode",
    ControllerRunModeContract.TryParse(
        "until-ended",
        out var persistentControllerRunMode) &&
    persistentControllerRunMode.UntilEnded &&
    persistentControllerRunMode.RunSeconds == 0);
Expect(
    "continuous_client_run_mode_rejects_out_of_range_or_ambiguous_tokens",
    !ControllerRunModeContract.TryParse("0", out _) &&
    !ControllerRunModeContract.TryParse("601", out _) &&
    !ControllerRunModeContract.TryParse("Until-Ended", out _));
var allowedLostExit = LostSessionExitModeContract.Evaluate(
    exactPrivateBotOneProven: true,
    roundActive: false,
    postFightPrompt: true,
    postFightWinner: false,
    scheduleRunning: false,
    singleTrialRunning: false,
    continuousControllerRunning: false,
    attackZoneTrialRunning: false,
    attackZoneRecoveryOnlyRunning: false);
Expect("exit_lost_mode_exact_guard_accepts", allowedLostExit.Allowed);
Expect(
    "exit_lost_mode_rejects_active_round",
    !LostSessionExitModeContract.Evaluate(
        true, true, true, false, false, false, false, false, false).Allowed);
Expect(
    "exit_lost_mode_rejects_winner",
    !LostSessionExitModeContract.Evaluate(
        true, false, true, true, false, false, false, false, false).Allowed);
Expect(
    "exit_lost_mode_rejects_missing_prompt_or_scope",
    !LostSessionExitModeContract.Evaluate(
        true, false, false, false, false, false, false, false, false).Allowed &&
    !LostSessionExitModeContract.Evaluate(
        false, false, true, false, false, false, false, false, false).Allowed);
Expect(
    "exit_lost_mode_rejects_each_running_controller",
    !LostSessionExitModeContract.Evaluate(
        true, false, true, false, true, false, false, false, false).Allowed &&
    !LostSessionExitModeContract.Evaluate(
        true, false, true, false, false, true, false, false, false).Allowed &&
    !LostSessionExitModeContract.Evaluate(
        true, false, true, false, false, false, true, false, false).Allowed &&
    !LostSessionExitModeContract.Evaluate(
        true, false, true, false, false, false, false, true, false).Allowed &&
    !LostSessionExitModeContract.Evaluate(
        true, false, true, false, false, false, false, false, true).Allowed);
var computedContinuousContractSha256 = Convert.ToHexString(
    SHA256.HashData(Encoding.UTF8.GetBytes(
        ContinuousBotControllerContract.CanonicalJson))).ToLowerInvariant();
Expect(
    $"continuous_contract_sha256:{computedContinuousContractSha256}",
    computedContinuousContractSha256 == ContinuousBotControllerContract.ExpectedSha256);
Expect("continuous_attack_count", continuousAttacks.Length == 6);
Expect(
    "continuous_attack_round_robin_order",
    continuousAttacks.Select(value => value.MoveIndex)
        .SequenceEqual(new[] { 2, 3, 4, 5, 9, 10 }));
Expect("continuous_g1_kick_only_count", g1ContinuousAttacks.Length == 4);
Expect(
    "continuous_g1_kick_only_indices",
    g1ContinuousAttacks.Select(value => value.MoveIndex)
        .SequenceEqual(new[] { 6, 7, 8, 9 }));
Expect(
    "continuous_g1_kick_only_names",
    g1ContinuousAttacks.Select(value => value.MoveName).SequenceEqual(new[]
    {
        "left_side_kick_processed",
        "left_front_kick_processed",
        "right_side_kick_processed",
        "right_knee_processed",
    }));
Expect(
    "continuous_g1_profiles_exclude_punches",
    g1ContinuousAttacks.All(value =>
        !value.MoveName.Contains("punch", StringComparison.OrdinalIgnoreCase) &&
        !value.DisplayName.Contains("punch", StringComparison.OrdinalIgnoreCase) &&
        !value.MoveName.Contains("jab", StringComparison.OrdinalIgnoreCase) &&
        !value.DisplayName.Contains("jab", StringComparison.OrdinalIgnoreCase) &&
        !value.MoveName.Contains("hook", StringComparison.OrdinalIgnoreCase) &&
        !value.DisplayName.Contains("hook", StringComparison.OrdinalIgnoreCase)));
Expect(
    "continuous_g1_profile_hashes_pinned",
    g1ContinuousAttacks.Select(value => value.SerializedAssetSha256).SequenceEqual(new[]
    {
        "fb5c3938396c789020003634b0dc76d2319ea3df33ec2fa2927a2e4e24a070a0",
        "5f61681420dbd84ce046f68770b73d2ed9e845d38cacd4e37d1f704bcb5f9241",
        "6c4a414aa3c6860f30b28d65e6ef7c18d5e6bf9331ab69215ded03cf82c560ac",
        "04dbcb4b28f617912c7ac5c452a23969c5bc17ac745571c35d29ebea03b2733b",
    }));
Expect(
    "continuous_runtime_model_selects_g1_kicks",
    ReferenceEquals(
        ContinuousBotControllerContract.AttacksForRuntimeModel("g1"),
        ContinuousBotControllerContract.G1Attacks));
Expect(
    "continuous_runtime_model_selects_t800_profile",
    ReferenceEquals(
        ContinuousBotControllerContract.AttacksForRuntimeModel("t800"),
        ContinuousBotControllerContract.Attacks));
Expect(
    "continuous_unknown_runtime_model_has_no_attacks",
    ContinuousBotControllerContract.AttacksForRuntimeModel(null) is null &&
    ContinuousBotControllerContract.AttacksForRuntimeModel("G1") is null);
Expect(
    "continuous_attack_round_robin_labeled_audit_divergence",
    ContinuousBotControllerContract.AttackSelectionProvenance.Contains(
        "audit_controller_deterministic_round_robin_diverges",
        StringComparison.Ordinal));
Expect(
    "continuous_projected_range_all_moves",
    continuousAttacks.All(value =>
        BitConverter.SingleToInt32Bits(value.MaximumDistanceMeters) ==
        BitConverter.SingleToInt32Bits(
            ContinuousBotControllerContract.MaximumAttackDistanceMeters)));
Expect(
    "continuous_static_impact_events_pinned",
    continuousAttacks.Select(value => value.StaticImpactEvents.Count)
        .SequenceEqual(new[] { 3, 1, 1, 1, 1, 1 }));
Expect(
    "continuous_static_impact_provenance_not_runtime",
    ContinuousBotControllerContract.StaticImpactTimingProvenance.Contains(
        "not_measured_input_to_completion_timing",
        StringComparison.Ordinal));
Expect(
    "continuous_translation_settle_native_thresholds_pinned",
    ContinuousBotControllerContract.TranslationSettleProvenance.Contains(
        "TransitionSettled_rva_0x226f6f0_strict_less_than",
        StringComparison.Ordinal) &&
    BitConverter.SingleToInt32Bits(
        ContinuousBotControllerContract
            .G1LocomotionTransitionVelocityThresholdMetersPerSecond) ==
        BitConverter.SingleToInt32Bits(0.03f) &&
    BitConverter.SingleToInt32Bits(
        ContinuousBotControllerContract
            .T800LocomotionTransitionVelocityThresholdMetersPerSecond) ==
        BitConverter.SingleToInt32Bits(0.15f));
Expect(
    "continuous_yaw_attack_preemption_is_labeled_human_observation",
    ContinuousBotControllerContract.YawAttackPreemptionRule.Contains(
        "human_observed_g1",
        StringComparison.Ordinal) &&
    ContinuousBotControllerContract.YawAttackPreemptionRule.Contains(
        "pending_instrumented_confirmation",
        StringComparison.Ordinal));
Expect(
    "continuous_move_profiles_bound_to_pinned_asset_container",
    ContinuousBotControllerContract.MoveProfileBindingRule.Contains(
        "sharedassets0",
        StringComparison.Ordinal) &&
    ContinuousBotControllerContract.SharedAssets0Sha256.Length == 64);
Expect(
    "continuous_round_restart_loss_limitation_pinned",
    ContinuousBotControllerContract.RoundRestartLimitation.Contains(
        "exits_to_lobby_after_loss",
        StringComparison.Ordinal) &&
    ContinuousBotControllerContract.RoundRestartStaticEvidence.Contains(
        "HandlePostFightContinue_rva_0x23aae90",
        StringComparison.Ordinal));
Expect(
    "continuous_recovery_guard_native_evidence_pinned",
    ContinuousBotControllerContract.RecoveryGuardProvenance.Contains(
        "DriveRecovery_rva_0x2367430",
        StringComparison.Ordinal) &&
    ContinuousBotControllerContract.DampenGuard ==
        "fallen_and_not_dampened" &&
    ContinuousBotControllerContract.StraightenGuard ==
        "fallen_and_dampened_and_not_already_issued");
using (var continuousManifest = JsonDocument.Parse(
           ContinuousBotControllerContract.CanonicalJson))
{
    var manifestG1Indices = continuousManifest.RootElement
        .GetProperty("g1_attacks")
        .EnumerateArray()
        .Select(value => value.GetProperty("move_index").GetInt32())
        .ToArray();
    Expect(
        "continuous_manifest_g1_kick_only_indices",
        manifestG1Indices.SequenceEqual(new[] { 6, 7, 8, 9 }));
    var normalFallSequence = continuousManifest.RootElement
        .GetProperty("normal_fall_recovery_request_sequence")
        .EnumerateArray()
        .Select(value => value.GetString() ?? string.Empty)
        .ToArray();
    var faultSequence = continuousManifest.RootElement
        .GetProperty("motor_shutdown_fault_request_sequence")
        .EnumerateArray()
        .Select(value => value.GetString() ?? string.Empty)
        .ToArray();
    Expect(
        "continuous_normal_fall_recovery_excludes_estop",
        normalFallSequence.SequenceEqual(new[]
        {
            "dampen_while_fallen_and_not_dampened",
            "observe_dampened",
            "straighten_once_while_fallen_and_dampened",
            "observe_recovery_armed",
            "orientation_get_up_only_when_recovery_armed_and_motor_running",
        }) &&
        normalFallSequence.All(value =>
            !value.Contains("estop", StringComparison.Ordinal)));
    Expect(
        "continuous_estop_is_motor_shutdown_fault_only",
        faultSequence.Contains("wait_fault_estop_delay", StringComparer.Ordinal) &&
        faultSequence.Contains("estop_toggle_on", StringComparer.Ordinal) &&
        faultSequence.Contains("estop_toggle_off", StringComparer.Ordinal));
}
Expect(
    "continuous_facing_yaw_native_evidence_pinned",
    ContinuousBotControllerContract.FacingYawProvenance.Contains(
        "ComputeFacingYaw_rva_0x2366e20",
        StringComparison.Ordinal) &&
    ContinuousBotControllerContract.FacingYawProvenance.Contains(
        "AngleToOpponent_rva_0x2366600",
        StringComparison.Ordinal));
Expect(
    "continuous_facing_yaw_deadband_boundaries",
    ContinuousBotControllerContract.ComputeFacingYaw(17.5f) == 0f &&
    ContinuousBotControllerContract.ComputeFacingYaw(-17.5f) == 0f &&
    ContinuousBotControllerContract.ComputeFacingYaw(
        MathF.BitIncrement(17.5f)) < 0f &&
    ContinuousBotControllerContract.ComputeFacingYaw(
        MathF.BitDecrement(-17.5f)) > 0f);
Expect(
    "continuous_facing_yaw_right_negative_left_positive",
    ContinuousBotControllerContract.ComputeFacingYaw(30f) < 0f &&
    ContinuousBotControllerContract.ComputeFacingYaw(-30f) > 0f);
Expect(
    "continuous_facing_yaw_saturates_at_native_scale",
    ContinuousBotControllerContract.ComputeFacingYaw(45f) == -1.5f &&
    ContinuousBotControllerContract.ComputeFacingYaw(90f) == -1.5f &&
    ContinuousBotControllerContract.ComputeFacingYaw(-45f) == 1.5f &&
    ContinuousBotControllerContract.ComputeFacingYaw(-90f) == 1.5f);
Expect(
    "continuous_fresh_move_send_blocks_each_local_down_or_recovery_state",
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(true, false, false, false, false, false, false, false) &&
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, true, false, false, false, false, false, false) &&
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, false, true, false, false, false, false, false) &&
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, false, false, true, false, false, false, false) &&
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, false, false, false, true, false, false, false) &&
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, false, false, false, false, true, false, false) &&
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, false, false, false, false, false, true, false) &&
    ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, false, false, false, false, false, false, true) &&
    !ContinuousBotControllerContract.LocalBlocksFreshMoveSend(false, false, false, false, false, false, false, false));
Expect(
    "continuous_fresh_move_send_blocks_each_opponent_down_or_recovery_state",
    ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(true, false, false, false, false, false, false) &&
    ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(false, true, false, false, false, false, false) &&
    ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(false, false, true, false, false, false, false) &&
    ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(false, false, false, true, false, false, false) &&
    ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(false, false, false, false, true, false, false) &&
    ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(false, false, false, false, false, true, false) &&
    ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(false, false, false, false, false, false, true) &&
    !ContinuousBotControllerContract.OpponentBlocksFreshMoveSend(false, false, false, false, false, false, false));

Expect(
    "continuous_geometry_right_bearing",
    ContinuousBotControllerContract.TryComputePlanarGeometry(
        0f, 0f, 0f, 1f,
        1f, 0f, -1f, 0f,
        out var rightGeometry) &&
    Math.Abs(rightGeometry.DistanceMeters - 1f) < 1e-6f &&
    Math.Abs(rightGeometry.LocalBearingToOpponentDegrees - 90f) < 1e-4f &&
    Math.Abs(rightGeometry.OpponentBearingToLocalDegrees) < 1e-4f);
Expect(
    "continuous_geometry_rejects_coincident_roots",
    !ContinuousBotControllerContract.TryComputePlanarGeometry(
        0f, 0f, 0f, 1f,
        0f, 0f, 0f, -1f,
        out _));
_ = ContinuousBotControllerContract.TryComputePlanarGeometry(
    0f, 0f, 0f, 1f,
    0f, 0.4f, 0f, -1f,
    out var attackWindowGeometry);
var attackWindowDecision = ContinuousBotControllerContract.DecideLocomotion(
    attackWindowGeometry,
    continuousAttacks[0],
    opponentDown: false);
Expect(
    "continuous_face_then_attack_window",
    attackWindowDecision.AttackWindow &&
    attackWindowDecision.Forward == 0f &&
    attackWindowDecision.Yaw == 0f);
var faceFirstDecision = ContinuousBotControllerContract.DecideLocomotion(
    rightGeometry,
    continuousAttacks[0],
    opponentDown: false);
Expect(
    "continuous_faces_before_attack",
    !faceFirstDecision.AttackWindow &&
    faceFirstDecision.Forward == 0f &&
    faceFirstDecision.Yaw == -1.5f);
var giveRoomDecision = ContinuousBotControllerContract.DecideLocomotion(
    rightGeometry,
    continuousAttacks[0],
    opponentDown: true);
Expect(
    "continuous_downed_opponent_backoff",
    !giveRoomDecision.AttackWindow &&
    giveRoomDecision.Forward == ContinuousBotControllerContract.DownedBackOffCommand);
Expect(
    "continuous_held_translation_released_and_settled_before_attack",
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0.8f, 0f, -1.5f, "g1", ContinuousTranslationSettleAxes.None,
        0f, 0f, 100, 0) ==
    ContinuousAttackMotionGate.ReleaseTranslationAndSettle &&
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0f, 1f, 0f, "g1", ContinuousTranslationSettleAxes.None,
        0f, 0f, 100, 0) ==
    ContinuousAttackMotionGate.ReleaseTranslationAndSettle);
Expect(
    "continuous_yaw_release_and_attack_share_one_control_tick",
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0f, 0f, 1.5f, "g1", ContinuousTranslationSettleAxes.None,
        0f, 0f, 100, 0) ==
    ContinuousAttackMotionGate.ReleaseYawAndAttack &&
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0f, 0f, 0f, "g1", ContinuousTranslationSettleAxes.None,
        0f, 0f, 101, 0) ==
    ContinuousAttackMotionGate.Ready);
Expect(
    "continuous_translation_release_obeys_time_and_measured_motion_gates",
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0f, 0f, 0f, "g1", ContinuousTranslationSettleAxes.Forward,
        0f, 0f, 100, 115) ==
    ContinuousAttackMotionGate.Settling &&
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0f, 0f, 0f, "g1", ContinuousTranslationSettleAxes.Forward,
        0.031f, 0f, 115, 115) ==
    ContinuousAttackMotionGate.Settling &&
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0f, 0f, 0f, "g1", ContinuousTranslationSettleAxes.Forward,
        0.029f, 0f, 115, 115) ==
    ContinuousAttackMotionGate.Ready);
Expect(
    "continuous_repositioning_and_nonfinite_gates",
    ContinuousBotControllerContract.DecideAttackMotionGate(
        false, 1f, 0f, 1f, null, ContinuousTranslationSettleAxes.None,
        0f, 0f, 100, 0) ==
    ContinuousAttackMotionGate.Repositioning &&
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, float.NaN, 0f, 0f, "g1", ContinuousTranslationSettleAxes.None,
        0f, 0f, 100, 0) ==
    ContinuousAttackMotionGate.Invalid &&
    ContinuousBotControllerContract.DecideAttackMotionGate(
        true, 0f, 0f, 0f, "g1", ContinuousTranslationSettleAxes.Forward,
        float.PositiveInfinity, 0f, 100, 0) ==
    ContinuousAttackMotionGate.Invalid);
Expect(
    "continuous_translation_settle_uses_build_pinned_strict_axis_thresholds",
    ContinuousBotControllerContract.IsTranslationSettled(
        "g1", ContinuousTranslationSettleAxes.None, 10f, -10f) &&
    ContinuousBotControllerContract.IsTranslationSettled(
        "g1", ContinuousTranslationSettleAxes.Forward, 0.029f, 10f) &&
    !ContinuousBotControllerContract.IsTranslationSettled(
        "g1", ContinuousTranslationSettleAxes.Forward, 0.03f, 0f) &&
    ContinuousBotControllerContract.IsTranslationSettled(
        "g1", ContinuousTranslationSettleAxes.Strafe, 10f, -0.029f) &&
    !ContinuousBotControllerContract.IsTranslationSettled(
        "g1", ContinuousTranslationSettleAxes.Strafe, 0f, -0.03f) &&
    ContinuousBotControllerContract.IsTranslationSettled(
        "t800", ContinuousTranslationSettleAxes.Planar, 0.149f, -0.149f) &&
    !ContinuousBotControllerContract.IsTranslationSettled(
        "t800", ContinuousTranslationSettleAxes.Forward, 0.15f, 0f) &&
    !ContinuousBotControllerContract.IsTranslationSettled(
        "unknown", ContinuousTranslationSettleAxes.None, 0f, 0f) &&
    !ContinuousBotControllerContract.IsTranslationSettled(
        "g1", ContinuousTranslationSettleAxes.Forward, float.NaN, 0f));
Expect(
    "continuous_translation_release_axes_preserved_before_neutral_write",
    ContinuousBotControllerContract.TranslationAxesForVelocity(0.8f, 0f) ==
        ContinuousTranslationSettleAxes.Forward &&
    ContinuousBotControllerContract.TranslationAxesForVelocity(-0.8f, 0f) ==
        ContinuousTranslationSettleAxes.Forward &&
    ContinuousBotControllerContract.TranslationAxesForVelocity(0f, 1f) ==
        ContinuousTranslationSettleAxes.Strafe &&
    ContinuousBotControllerContract.TranslationAxesForVelocity(0f, -1f) ==
        ContinuousTranslationSettleAxes.Strafe &&
    ContinuousBotControllerContract.TranslationAxesForVelocity(1f, -1f) ==
        ContinuousTranslationSettleAxes.Planar &&
    ContinuousBotControllerContract.TranslationAxesForVelocity(0f, -0f) ==
        ContinuousTranslationSettleAxes.None);
Expect(
    "continuous_held_velocity_boundary_accepts_exact_owned_value",
    ContinuousBotControllerContract.DecideVelocityBoundary(
        0.8f, 0f, -1.5f, 0.8f, 0f, -1.5f) ==
    ContinuousVelocityBoundaryDecision.OwnedExact);
Expect(
    "continuous_held_velocity_boundary_reasserts_only_exact_native_neutral",
    ContinuousBotControllerContract.DecideVelocityBoundary(
        0f, 0f, 0f, 0.8f, 0f, -1.5f) ==
    ContinuousVelocityBoundaryDecision.ReassertNativeNeutralOverwrite);
Expect(
    "continuous_held_velocity_boundary_rejects_non_neutral_drift",
    ContinuousBotControllerContract.DecideVelocityBoundary(
        0f, 0f, 1f, 0.8f, 0f, -1.5f) ==
    ContinuousVelocityBoundaryDecision.RejectUnownedNonNeutral);
Expect(
    "continuous_held_velocity_boundary_rejects_nonfinite_values",
    ContinuousBotControllerContract.DecideVelocityBoundary(
        float.NaN, 0f, 0f, 0.8f, 0f, -1.5f) ==
    ContinuousVelocityBoundaryDecision.Invalid);

var staleOpponentSemanticT800 =
    ContinuousBotControllerContract.ClassifySemanticRuntimeConsistency(
        semanticRobotId: "t800",
        exactRuntimeModel: "g1");
var staleOpponentSemanticG1 =
    ContinuousBotControllerContract.ClassifySemanticRuntimeConsistency(
        semanticRobotId: "g1",
        exactRuntimeModel: "t800");
var unavailableLocalSemanticG1 =
    ContinuousBotControllerContract.ClassifySemanticRuntimeConsistency(
        semanticRobotId: string.Empty,
        exactRuntimeModel: "g1");
Expect(
    "continuous_exposes_stale_t800_semantic_against_g1_runtime",
    staleOpponentSemanticT800.Mismatch);
Expect(
    "continuous_exposes_stale_g1_semantic_against_t800_runtime",
    staleOpponentSemanticG1.Mismatch);
Expect(
    "continuous_empty_semantic_id_remains_unavailable_not_inferred",
    !unavailableLocalSemanticG1.Mismatch &&
    unavailableLocalSemanticG1.Classification ==
        "semantic_robot_id_unavailable_runtime_g1_exact");
Expect(
    "continuous_rejects_unproven_runtime_pairing",
    !ContinuousBotControllerContract.HasRequiredRuntimePairing(
        exactSupportedRuntimePairing: false,
        exactRuntimeModel: "g1"));
Expect(
    "continuous_accepts_exact_g1_runtime_pairing",
    staleOpponentSemanticG1.Mismatch &&
    ContinuousBotControllerContract.HasRequiredRuntimePairing(
        exactSupportedRuntimePairing: true,
        exactRuntimeModel: "g1"));
Expect(
    "continuous_accepts_exact_t800_runtime_pairing",
    ContinuousBotControllerContract.HasRequiredRuntimePairing(
        exactSupportedRuntimePairing: true,
        exactRuntimeModel: "t800"));
Expect(
    "continuous_t800_autonomous_recovery_retained",
    ContinuousBotControllerContract.SupportsAutonomousRecovery("t800") &&
    ContinuousBotControllerContract.RecoveryModeForRuntimeModel("t800") ==
        ContinuousBotControllerContract.T800RecoveryMode);
Expect(
    "continuous_g1_recovery_fails_closed",
    !ContinuousBotControllerContract.SupportsAutonomousRecovery("g1") &&
    ContinuousBotControllerContract.RecoveryModeForRuntimeModel("g1") ==
        ContinuousBotControllerContract.G1RecoveryMode);
Expect(
    "continuous_unknown_recovery_fails_closed",
    !ContinuousBotControllerContract.SupportsAutonomousRecovery(null) &&
    ContinuousBotControllerContract.RecoveryModeForRuntimeModel(null) is null);
Expect(
    "continuous_rejects_unknown_runtime_model",
    !ContinuousBotControllerContract.HasRequiredRuntimePairing(
        exactSupportedRuntimePairing: true,
        exactRuntimeModel: "unknown"));

Expect(
    "continuous_recovery_prone_after_straighten_when_armed",
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: true,
        recoveryArmed: true,
        motorShutdown: false,
        straightenIssued: true,
        suggestedProne: true) == ContinuousRecoveryCommand.GetUpProne);
Expect(
    "continuous_recovery_supine_after_straighten_when_armed",
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: true,
        recoveryArmed: true,
        motorShutdown: false,
        straightenIssued: true,
        suggestedProne: false) == ContinuousRecoveryCommand.GetUpSupine);
Expect(
    "continuous_recovery_dampens_first",
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: false,
        recoveryArmed: false,
        motorShutdown: false,
        straightenIssued: false,
        suggestedProne: true) ==
    ContinuousRecoveryCommand.Dampen);
Expect(
    "continuous_recovery_straighten_on_pinned_dampened_guard",
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: true,
        recoveryArmed: false,
        motorShutdown: false,
        straightenIssued: false,
        suggestedProne: true) == ContinuousRecoveryCommand.Straighten);
Expect(
    "continuous_recovery_straighten_is_one_shot",
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: true,
        recoveryArmed: false,
        motorShutdown: false,
        straightenIssued: true,
        suggestedProne: true) ==
    ContinuousRecoveryCommand.WaitForDampenedOrRecoveryArmed);
var straightenAfterFaultPreemption =
    ContinuousBotControllerContract.ResolveStraightenIssuedOnFaultEntry(
        recoveryEpisodeAlreadyActive: true,
        straightenAlreadyIssued: true);
Expect(
    "continuous_fault_preemption_preserves_straighten_one_shot",
    straightenAfterFaultPreemption &&
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: true,
        recoveryArmed: false,
        motorShutdown: false,
        straightenIssued: straightenAfterFaultPreemption,
        suggestedProne: true) ==
    ContinuousRecoveryCommand.WaitForDampenedOrRecoveryArmed &&
    !ContinuousBotControllerContract.ResolveStraightenIssuedOnFaultEntry(
        recoveryEpisodeAlreadyActive: false,
        straightenAlreadyIssued: true));
Expect(
    "continuous_recovery_straightens_before_armed_getup",
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: true,
        recoveryArmed: true,
        motorShutdown: false,
        straightenIssued: false,
        suggestedProne: true) == ContinuousRecoveryCommand.Straighten);
Expect(
    "continuous_normal_recovery_does_not_command_during_motor_shutdown",
    ContinuousBotControllerContract.SelectRecoveryCommand(
        fallen: true,
        dampened: false,
        recoveryArmed: false,
        motorShutdown: true,
        straightenIssued: false,
        suggestedProne: true) == ContinuousRecoveryCommand.None);

var recoveryHandshake = ContinuousBotControllerContract.DecideFaultEStopHandshake(
    ContinuousEStopRecoveryStage.FaultDelay,
    motorShutdown: true,
    stageElapsedTicks: ContinuousBotControllerContract.FaultEStopDelayTicks - 1);
Expect(
    "continuous_fault_estop_waits_pinned_delay",
    recoveryHandshake.Edge == ContinuousEStopHandshakeEdge.None &&
    recoveryHandshake.NextStage == ContinuousEStopRecoveryStage.FaultDelay);
recoveryHandshake = ContinuousBotControllerContract.DecideFaultEStopHandshake(
    recoveryHandshake.NextStage,
    motorShutdown: true,
    stageElapsedTicks: ContinuousBotControllerContract.FaultEStopDelayTicks);
Expect(
    "continuous_fault_estop_toggle_on_after_delay",
    recoveryHandshake.Edge == ContinuousEStopHandshakeEdge.RequestToggleOn &&
    recoveryHandshake.NextStage == ContinuousEStopRecoveryStage.AwaitMotorShutdown);
Expect(
    "continuous_owned_pending_estop_keeps_input_scope_ready",
    ContinuousBotControllerContract.IsInputReadyForControl(
        isActive: true,
        networkInitialized: true,
        hasPendingEStop: true,
        exactOwnedPendingEStop: true));
Expect(
    "continuous_unowned_pending_estop_fails_input_scope",
    !ContinuousBotControllerContract.IsInputReadyForControl(
        isActive: true,
        networkInitialized: true,
        hasPendingEStop: true,
        exactOwnedPendingEStop: false));
recoveryHandshake = ContinuousBotControllerContract.DecideFaultEStopHandshake(
    recoveryHandshake.NextStage,
    motorShutdown: true,
    stageElapsedTicks: 0);
Expect(
    "continuous_fault_estop_observed_shutdown",
    recoveryHandshake.Edge == ContinuousEStopHandshakeEdge.ObserveMotorShutdown &&
    recoveryHandshake.NextStage == ContinuousEStopRecoveryStage.FaultHold);
recoveryHandshake = ContinuousBotControllerContract.DecideFaultEStopHandshake(
    recoveryHandshake.NextStage,
    motorShutdown: true,
    stageElapsedTicks: ContinuousBotControllerContract.FaultEStopHoldTicks - 1);
Expect(
    "continuous_fault_estop_holds_for_half_second",
    recoveryHandshake.Edge == ContinuousEStopHandshakeEdge.None &&
    recoveryHandshake.NextStage == ContinuousEStopRecoveryStage.FaultHold);
recoveryHandshake = ContinuousBotControllerContract.DecideFaultEStopHandshake(
    recoveryHandshake.NextStage,
    motorShutdown: true,
    stageElapsedTicks: ContinuousBotControllerContract.FaultEStopHoldTicks);
Expect(
    "continuous_fault_estop_toggle_off_after_hold",
    recoveryHandshake.Edge == ContinuousEStopHandshakeEdge.RequestToggleOff &&
    recoveryHandshake.NextStage == ContinuousEStopRecoveryStage.AwaitMotorRunning);
recoveryHandshake = ContinuousBotControllerContract.DecideFaultEStopHandshake(
    recoveryHandshake.NextStage,
    motorShutdown: false,
    stageElapsedTicks: 0);
Expect(
    "continuous_fault_estop_observed_running",
    recoveryHandshake.Edge == ContinuousEStopHandshakeEdge.ObserveMotorRunning &&
    recoveryHandshake.NextStage == ContinuousEStopRecoveryStage.Complete);

Expect(
    "continuous_round_prompt_gate_allows",
    ContinuousBotControllerContract.ShouldIssueRoundStartRequest(
        true, true, true, true, true, false,
        ContinuousBotControllerContract.RoundStartPromptDelayTicks));
Expect(
    "continuous_round_prompt_one_shot",
    !ContinuousBotControllerContract.ShouldIssueRoundStartRequest(
        true, true, true, true, true, true,
        ContinuousBotControllerContract.RoundStartPromptDelayTicks));
Expect(
    "continuous_round_prompt_loss_exits_in_current_build",
    !ContinuousBotControllerContract.ShouldIssueRoundStartRequest(
        true, true, true, true, false, false,
        ContinuousBotControllerContract.RoundStartPromptDelayTicks));
Expect(
    "continuous_round_prompt_not_home",
    !ContinuousBotControllerContract.ShouldIssueRoundStartRequest(
        true, true, false, false, true, false,
        ContinuousBotControllerContract.RoundStartPromptDelayTicks));
Expect(
    "continuous_round_prompt_not_public_or_unproven",
    !ContinuousBotControllerContract.ShouldIssueRoundStartRequest(
        false, true, true, true, true, false,
        ContinuousBotControllerContract.RoundStartPromptDelayTicks));
Expect(
    "continuous_round_prompt_before_two_minute_limit",
    ContinuousBotControllerContract.RoundStartPromptDelayTicks <
        ContinuousBotControllerContract.TwoMinuteTicks &&
    ContinuousBotControllerContract.RoundStartObservationTimeoutTicks <
        ContinuousBotControllerContract.TwoMinuteTicks);
Expect(
    "continuous_round_prompt_rejects_two_minute_boundary",
    !ContinuousBotControllerContract.ShouldIssueRoundStartRequest(
        true, true, true, true, true, false,
        ContinuousBotControllerContract.TwoMinuteTicks));
Expect(
    "continuous_round_rebind_requires_owned_request",
    ContinuousBotControllerContract.CanBindRestartedRound(true, true, true) &&
    !ContinuousBotControllerContract.CanBindRestartedRound(true, false, true));
Expect(
    "continuous_round_rebind_rejects_reuse",
    !ContinuousBotControllerContract.CanBindRestartedRound(true, true, false));

var markerSpecs = RenderedCommandMarkerContract.Specs;
Expect("rendered_marker_schema", RenderedCommandMarkerContract.Schema == "rek.rendered_command_marker.v1");
Expect(
    "rendered_marker_binding",
    RenderedCommandMarkerContract.RenderBinding ==
    "first_post_marker_frame_is_first_rendered_frame_after_command_edge");
Expect("rendered_marker_transition", RenderedCommandMarkerContract.Transition == "persistent_exact_rgb_rising_edge");
Expect("rendered_marker_count", markerSpecs.Length == 24);
Expect("rendered_marker_indices", markerSpecs.Select(value => value.Index).SequenceEqual(Enumerable.Range(0, 24)));
Expect("rendered_marker_selectors_unique", markerSpecs.Select(value => value.Selector).Distinct(StringComparer.Ordinal).Count() == 24);
Expect("rendered_marker_regions_unique", markerSpecs.Select(value => (value.X, value.Y, value.Width, value.Height)).Distinct().Count() == 24);
Expect("rendered_marker_regions_8px", markerSpecs.All(value => value.Width == 8 && value.Height == 8));
Expect("rendered_marker_first_edge", markerSpecs[0].ScheduleTick == 50 && markerSpecs[0].Selector == "walk_forward.press.1");
Expect("rendered_marker_last_edge", markerSpecs[^1].ScheduleTick == 2600 && markerSpecs[^1].Selector == "walk_backward.release.2");
Expect("rendered_marker_tick_900_count", markerSpecs.Count(value => value.ScheduleTick == 900) == 2);
Expect("rendered_marker_tick_2100_count", markerSpecs.Count(value => value.ScheduleTick == 2100) == 2);
Expect("rendered_marker_tick_2400_count", markerSpecs.Count(value => value.ScheduleTick == 2400) == 2);
Expect("rendered_marker_pre_rgb", RenderedCommandMarkerContract.PreRgb.SequenceEqual(new[] { 0, 0, 0 }));
Expect("rendered_marker_post_rgb", RenderedCommandMarkerContract.PostRgb.SequenceEqual(new[] { 255, 0, 255 }));

var exactT800Bones = BridgePairingContract.T800BoneNames
    .Select(name => (string?)name)
    .ToArray();
Expect("t800_bone_count", exactT800Bones.Length == 26);
Expect("t800_bone_root", exactT800Bones[0] == "LINK_BASE");
Expect("t800_bone_tail", exactT800Bones[^1] == "LINK_HEAD_YAW");
Expect(
    "t800_bone_signature_sha256",
    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        string.Join("\n", exactT800Bones)))).ToLowerInvariant() ==
        BridgePairingContract.T800BoneSignatureSha256);

var exactG1Bones = BridgePairingContract.G1BoneNames
    .Select(name => (string?)name)
    .ToArray();
Expect("g1_bone_count", exactG1Bones.Length == 30);
Expect("g1_bone_root", exactG1Bones[0] == "pelvis");
Expect("g1_bone_tail", exactG1Bones[^1] == "right_wrist_yaw_link");
Expect(
    "g1_bone_signature_sha256",
    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        string.Join("\n", exactG1Bones)))).ToLowerInvariant() ==
        BridgePairingContract.G1BoneSignatureSha256);

var exactPairing = BridgePairingContract.Validate(
    "t800",
    exactT800Bones,
    "t800",
    exactT800Bones);
Expect("exact_t800_pairing_accepted", exactPairing.ExactT800VersusT800);
Expect(
    "exact_t800_pairing_reason",
    exactPairing.Reason == BridgePairingContract.ExactPairingReason);
Expect("exact_t800_supported_runtime_pairing", exactPairing.ExactSupportedRuntimePairing);
Expect("exact_t800_runtime_model", exactPairing.RuntimeModel == "t800");

var exactG1PairingWithEmptyLocalSemantic = BridgePairingContract.Validate(
    string.Empty,
    exactG1Bones,
    "g1",
    exactG1Bones);
Expect("exact_g1_pairing_accepted", exactG1PairingWithEmptyLocalSemantic.ExactG1VersusG1);
Expect("exact_g1_supported_runtime_pairing", exactG1PairingWithEmptyLocalSemantic.ExactSupportedRuntimePairing);
Expect("exact_g1_runtime_model", exactG1PairingWithEmptyLocalSemantic.RuntimeModel == "g1");
Expect("empty_local_semantic_not_inferred_as_g1", !exactG1PairingWithEmptyLocalSemantic.LocalSemanticG1);
Expect(
    "empty_local_semantic_recorded_unavailable",
    exactG1PairingWithEmptyLocalSemantic.LocalSemanticRuntimeConsistency ==
        "semantic_robot_id_unavailable_runtime_g1_exact");

var wrongOrderBones = exactT800Bones.ToArray();
(wrongOrderBones[1], wrongOrderBones[2]) = (wrongOrderBones[2], wrongOrderBones[1]);
Expect(
    "local_bone_order_mismatch_rejected",
    BridgePairingContract.Validate("t800", wrongOrderBones, "t800", exactT800Bones).Reason ==
    "local_fighter_t800_bone_signature_mismatch");
Expect(
    "opponent_bone_order_mismatch_rejected",
    BridgePairingContract.Validate("t800", exactT800Bones, "t800", wrongOrderBones).Reason ==
    "opponent_fighter_t800_bone_signature_mismatch");
Expect(
    "local_bone_count_mismatch_rejected",
    BridgePairingContract.Validate("t800", exactT800Bones[..^1], "t800", exactT800Bones).Reason ==
    "local_fighter_unsupported_bone_count:25");
Expect(
    "mixed_t800_g1_runtime_pair_rejected",
    BridgePairingContract.Validate(
        "t800",
        exactT800Bones,
        "g1",
        exactG1Bones).Reason ==
    "mixed_supported_runtime_models_rejected");
Expect(
    "empty_semantic_does_not_block_exact_runtime_t800_pair",
    BridgePairingContract.Validate(null, exactT800Bones, "t800", exactT800Bones)
        .ExactT800VersusT800);
Expect(
    "stale_opponent_semantic_does_not_override_exact_runtime_t800_pair",
    BridgePairingContract.Validate("t800", exactT800Bones, "g1", exactT800Bones)
        .ExactT800VersusT800);
Expect(
    "semantic_id_is_case_sensitive_and_non_authoritative",
    !BridgePairingContract.Validate("T800", exactT800Bones, "t800", exactT800Bones)
        .LocalSemanticT800 &&
    BridgePairingContract.Validate("T800", exactT800Bones, "t800", exactT800Bones)
        .ExactT800VersusT800);
var wrongG1Order = exactG1Bones.ToArray();
(wrongG1Order[1], wrongG1Order[2]) = (wrongG1Order[2], wrongG1Order[1]);
Expect(
    "g1_wrong_order_rejected",
    BridgePairingContract.Validate("g1", exactG1Bones, "g1", wrongG1Order).Reason ==
        "opponent_fighter_g1_bone_signature_mismatch");

void ExpectParse(
    string name,
    string json,
    bool expected,
    BridgeKey? expectedKey = null,
    BridgeCommand? expectedCommand = null,
    string? expectedSelector = null)
{
    var actual = BridgeProtocol.TryParse(
        Encoding.UTF8.GetBytes(json),
        7,
        out var request,
        out _,
        out _);
    if (actual != expected ||
        (expectedKey is not null && request?.Key != expectedKey) ||
        (expectedCommand is not null && request?.Command != expectedCommand) ||
        (expectedSelector is not null && request?.Selector != expectedSelector))
        failures.Enqueue(name);
}

ExpectParse("get_state", "{\"type\":\"get_state\",\"request_id\":\"r-1\"}", true);
foreach (var key in Enum.GetValues<BridgeKey>())
{
    ExpectParse(
        $"input_{key}",
        $"{{\"type\":\"input\",\"request_id\":\"r-{key}\",\"key\":\"{key}\"}}",
        true,
        key);
}
foreach (var command in Enum.GetValues<BridgeCommand>())
{
    if (command is BridgeCommand.StartSingleMotionTrial or BridgeCommand.StartAttackZoneTrial)
        continue;
    ExpectParse(
        $"command_{command}",
        $"{{\"type\":\"command\",\"request_id\":\"r-{command}\",\"command\":\"{command}\"}}",
        true,
        expectedCommand: command);
}
var attackZoneSeed = new string('a', 64);
var attackZoneEntries = AttackZoneTrialContract.BuildRandomizedSchedule(
    "protocol-test-run",
    0,
    attackZoneSeed);
var attackZoneScheduleSha256 = AttackZoneTrialContract.ComputeScheduleSha256(
    "protocol-test-run",
    0,
    attackZoneSeed,
    1,
    attackZoneEntries);
var attackZoneTarget = AttackZoneTrialContract.CreateTarget(
    attackZoneEntries[0],
    attackZoneScheduleSha256,
    attackZoneSeed,
    new string('b', 64),
    new string('c', 64),
    "protocol-test-trial",
    1);
var attackZoneTargetJson = AttackZoneTrialContract.SerializeTarget(attackZoneTarget);
var attackZoneRequestJson =
    $"{{\"type\":\"command\",\"request_id\":\"r-attack-zone\",\"command\":\"StartAttackZoneTrial\",\"target\":{attackZoneTargetJson}}}";
var parsedAttackZone = BridgeProtocol.TryParse(
    Encoding.UTF8.GetBytes(attackZoneRequestJson),
    7,
    out var parsedAttackZoneRequest,
    out _,
    out _);
Expect(
    "attack_zone_target_parsed_exactly",
    parsedAttackZone &&
    parsedAttackZoneRequest?.Command == BridgeCommand.StartAttackZoneTrial &&
    parsedAttackZoneRequest.AttackZoneTarget == attackZoneTarget);
ExpectParse(
    "attack_zone_missing_target",
    "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartAttackZoneTrial\"}",
    false);
ExpectParse(
    "attack_zone_target_on_stop_rejected",
    $"{{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StopAttackZoneTrial\",\"target\":{attackZoneTargetJson}}}",
    false);
ExpectParse(
    "attack_zone_target_unknown_field_rejected",
    attackZoneRequestJson.Replace(
        "\"acquisition_timeout_ticks\":500",
        "\"acquisition_timeout_ticks\":500,\"extra\":true",
        StringComparison.Ordinal),
    false);
foreach (var selector in trialSelectors)
{
    ExpectParse(
        $"single_trial_{selector.Selector}",
        $"{{\"type\":\"command\",\"request_id\":\"r-{selector.Selector}\",\"command\":\"StartSingleMotionTrial\",\"selector\":\"{selector.Selector}\"}}",
        true,
        expectedCommand: BridgeCommand.StartSingleMotionTrial,
        expectedSelector: selector.Selector);
}
ExpectParse("single_trial_missing_selector", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartSingleMotionTrial\"}", false);
ExpectParse("single_trial_unknown_selector", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartSingleMotionTrial\",\"selector\":\"jump\"}", false);
ExpectParse("single_trial_case_sensitive_selector", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartSingleMotionTrial\",\"selector\":\"Forward\"}", false);
ExpectParse("single_trial_empty_selector", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartSingleMotionTrial\",\"selector\":\"\"}", false);
ExpectParse("single_trial_duplicate_selector", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartSingleMotionTrial\",\"selector\":\"forward\",\"selector\":\"forward\"}", false);
ExpectParse("single_trial_non_string_selector", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartSingleMotionTrial\",\"selector\":1}", false);
ExpectParse("selector_on_other_command", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"StartMeasuredSchedule\",\"selector\":\"forward\"}", false);
ExpectParse("lowercase_key", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"left\"}", false);
ExpectParse("numeric_key_zero", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"0\"}", false);
ExpectParse("numeric_key_defined", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"6\"}", false);
ExpectParse("unknown_key", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"F1\"}", false);
ExpectParse("lowercase_command", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"navigatefreeplay\"}", false);
ExpectParse("numeric_command_zero", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"0\"}", false);
ExpectParse("numeric_command_defined", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"6\"}", false);
ExpectParse("unknown_command", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"LaunchLiveArena\"}", false);
ExpectParse("unknown_property", "{\"type\":\"get_state\",\"request_id\":\"r\",\"extra\":\"x\"}", false);
ExpectParse("duplicate_property", "{\"type\":\"get_state\",\"request_id\":\"r\",\"request_id\":\"r2\"}", false);
ExpectParse("invalid_id", "{\"type\":\"get_state\",\"request_id\":\"space id\"}", false);
ExpectParse("bad_shape", "{\"type\":\"get_state\",\"request_id\":\"r\",\"key\":\"Left\"}", false);
ExpectParse("mixed_command_input", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"NavigateFreePlay\",\"key\":\"Left\"}", false);
ExpectParse("invalid_json", "{", false);

var pipeName = $"rek-ui-bridge-test-{Environment.ProcessId}-{Guid.NewGuid():N}";
var accepted = new TaskCompletionSource<BridgeRequest>(TaskCreationOptions.RunContinuationsAsynchronously);
using (var server = new LocalPipeServer(
           pipeName,
           request => accepted.TrySetResult(request),
           info => Console.WriteLine($"PIPE_INFO {info}"),
           warning =>
           {
               Console.WriteLine($"PIPE_WARNING {warning}");
               failures.Enqueue($"pipe_warning:{warning}");
           }))
{
    server.Start();
    using var client = new NamedPipeClientStream(
        ".",
        pipeName,
        PipeDirection.InOut,
        PipeOptions.Asynchronous);
    using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
    await client.ConnectAsync(timeout.Token);
    using var reader = new StreamReader(client, Encoding.UTF8, false, 4096, leaveOpen: true);
    var helloLine = await reader.ReadLineAsync().WaitAsync(timeout.Token);
    if (helloLine is null)
    {
        failures.Enqueue("missing_hello");
    }
    else
    {
        using var hello = JsonDocument.Parse(helloLine);
        if (hello.RootElement.GetProperty("event").GetString() != "hello" ||
            !hello.RootElement.GetProperty("current_user_only").GetBoolean() ||
            !hello.RootElement.GetProperty("local_computer_verified").GetBoolean() ||
            string.IsNullOrWhiteSpace(
                hello.RootElement.GetProperty("local_client_verification").GetString()) ||
            !hello.RootElement.GetProperty("capabilities")
                .GetProperty("exclusive_control_lease_required").GetBoolean() ||
            hello.RootElement.GetProperty("capabilities")
                .GetProperty("semantic_commands").GetArrayLength() != Enum.GetValues<BridgeCommand>().Length ||
            hello.RootElement.GetProperty("capabilities")
                .GetProperty("rendered_command_marker_schema").GetString() != RenderedCommandMarkerContract.Schema ||
            hello.RootElement.GetProperty("capabilities")
                .GetProperty("rendered_command_marker_count").GetInt32() != markerSpecs.Length)
        {
            failures.Enqueue("invalid_hello");
        }
        var capabilities = hello.RootElement.GetProperty("capabilities");
        if (capabilities.GetProperty("private_ai_proof_basis").GetString() !=
                "build_pinned_REK_FindMatch_solo_ConnectToArena_EnterChampionship_non_koth_solo_same_runtime_session" ||
            capabilities.GetProperty("solo_route_required_flow").GetString() != "solo" ||
            capabilities.GetProperty("solo_route_arena_identifier_recorded").GetBoolean() ||
            capabilities.GetProperty("solo_route_connection_ticket_recorded").GetBoolean() ||
            capabilities.GetProperty("solo_route_endpoint_recorded").GetBoolean() ||
            capabilities.GetProperty("server_private_proven").GetBoolean() ||
            capabilities.GetProperty("server_private_status").GetString() != "unknown")
        {
            failures.Enqueue("invalid_solo_route_hello");
        }
        if (capabilities.GetProperty("single_motion_trial_schema").GetString() !=
                SingleMotionTrialContract.Schema ||
            capabilities.GetProperty("single_motion_trial_sha256").GetString() !=
                SingleMotionTrialContract.ExpectedSha256 ||
            capabilities.GetProperty("single_motion_trial_authority_scope").GetString() !=
                SingleMotionTrialContract.AuthorityScope ||
            capabilities.GetProperty("single_motion_trial_authority_caveat").GetString() !=
                SingleMotionTrialContract.AuthorityCaveat ||
            capabilities.GetProperty("single_motion_trial_unity_fixed_rate_hz").GetInt32() != 500 ||
            capabilities.GetProperty("single_motion_trial_rate_hz").GetInt32() != 50 ||
            capabilities.GetProperty("single_motion_trial_fixed_substeps_per_tick").GetInt32() != 10 ||
            capabilities.GetProperty("single_motion_trial_neutral_pre_roll_ticks").GetInt32() != 50 ||
            capabilities.GetProperty("single_motion_trial_action_tick").GetInt32() != 50 ||
            capabilities.GetProperty("single_motion_trial_locomotion_release_tick").GetInt32() != 100 ||
            capabilities.GetProperty("single_motion_trial_duration_ticks").GetInt32() != 250 ||
            capabilities.GetProperty("single_motion_trial_selectors").GetArrayLength() !=
                trialSelectors.Length ||
            !capabilities.GetProperty("single_motion_trial_selectors").EnumerateArray()
                .Select(value => value.GetString())
                .SequenceEqual(trialSelectors.Select(value => (string?)value.Selector)))
        {
            failures.Enqueue("invalid_single_motion_trial_hello");
        }
        if (capabilities.GetProperty("continuous_controller_schema").GetString() !=
                ContinuousBotControllerContract.Schema ||
            !capabilities.GetProperty("autonomous_semantic_controller").GetBoolean() ||
            capabilities.GetProperty("continuous_controller_sha256").GetString() !=
                ContinuousBotControllerContract.ExpectedSha256 ||
            capabilities.GetProperty("continuous_controller_authority_scope").GetString() !=
                ContinuousBotControllerContract.AuthorityScope ||
            capabilities.GetProperty("continuous_controller_authority_caveat").GetString() !=
                ContinuousBotControllerContract.AuthorityCaveat ||
            capabilities.GetProperty("continuous_controller_range_angle_provenance").GetString() !=
                ContinuousBotControllerContract.RangeAngleProvenance ||
            capabilities.GetProperty("continuous_controller_facing_yaw_provenance").GetString() !=
                ContinuousBotControllerContract.FacingYawProvenance ||
            capabilities.GetProperty("continuous_controller_attack_selection_provenance").GetString() !=
                ContinuousBotControllerContract.AttackSelectionProvenance ||
            capabilities.GetProperty("continuous_controller_static_impact_timing_provenance").GetString() !=
                ContinuousBotControllerContract.StaticImpactTimingProvenance ||
            capabilities.GetProperty("continuous_controller_round_restart_limitation").GetString() !=
                ContinuousBotControllerContract.RoundRestartLimitation ||
            capabilities.GetProperty("continuous_controller_round_restart_static_evidence").GetString() !=
                ContinuousBotControllerContract.RoundRestartStaticEvidence ||
            capabilities.GetProperty("continuous_controller_recovery_guard_provenance").GetString() !=
                ContinuousBotControllerContract.RecoveryGuardProvenance ||
            capabilities.GetProperty("continuous_controller_fault_estop_provenance").GetString() !=
                ContinuousBotControllerContract.FaultEStopProvenance ||
            capabilities.GetProperty("continuous_controller_dampen_guard").GetString() !=
                ContinuousBotControllerContract.DampenGuard ||
            capabilities.GetProperty("continuous_controller_straighten_guard").GetString() !=
                ContinuousBotControllerContract.StraightenGuard ||
            capabilities.GetProperty("continuous_controller_opponent_runtime_requirement").GetString() !=
                ContinuousBotControllerContract.OpponentRuntimeRequirement ||
            capabilities.GetProperty("continuous_controller_facing_deadband_factor").GetSingle() !=
                ContinuousBotControllerContract.FacingDeadbandFactor ||
            capabilities.GetProperty("continuous_controller_facing_threshold_degrees").GetSingle() !=
                ContinuousBotControllerContract.FacingThresholdDegrees ||
            capabilities.GetProperty("continuous_controller_facing_yaw_ramp_degrees").GetSingle() !=
                ContinuousBotControllerContract.FacingYawRampDegrees ||
            capabilities.GetProperty("continuous_controller_engage_yaw_command").GetSingle() !=
                ContinuousBotControllerContract.EngageYawCommand ||
            capabilities.GetProperty("continuous_controller_fault_estop_delay_ticks").GetInt32() !=
                ContinuousBotControllerContract.FaultEStopDelayTicks ||
            capabilities.GetProperty("continuous_controller_fault_estop_hold_ticks").GetInt32() !=
                ContinuousBotControllerContract.FaultEStopHoldTicks ||
            capabilities.GetProperty("continuous_controller_unity_fixed_rate_hz").GetInt32() != 500 ||
            capabilities.GetProperty("continuous_controller_rate_hz").GetInt32() != 50 ||
            capabilities.GetProperty("continuous_controller_fixed_substeps_per_tick").GetInt32() != 10 ||
            capabilities.GetProperty("continuous_controller_recovery_observation_timeout_ticks").GetInt32() !=
                ContinuousBotControllerContract.RecoveryObservationTimeoutTicks ||
            capabilities.GetProperty("continuous_controller_round_start_prompt_delay_ticks").GetInt32() !=
                ContinuousBotControllerContract.RoundStartPromptDelayTicks ||
            capabilities.GetProperty("continuous_controller_round_start_observation_timeout_ticks").GetInt32() !=
                ContinuousBotControllerContract.RoundStartObservationTimeoutTicks ||
            capabilities.GetProperty("continuous_controller_two_minute_limit_ticks").GetInt32() !=
                ContinuousBotControllerContract.TwoMinuteTicks ||
            capabilities.GetProperty("continuous_controller_round_start_semantic_method").GetString() !=
                "GameMenuController.HandlePostFightContinue" ||
            capabilities.GetProperty("continuous_controller_global_space_input_emitted").GetBoolean() ||
            capabilities.GetProperty("continuous_controller_opponent_semantic_robot_id_used_for_acceptance").GetBoolean() ||
            !capabilities.GetProperty("continuous_controller_supported_runtime_models")
                .EnumerateArray().Select(value => value.GetString())
                .SequenceEqual(new string?[] { "t800", "g1" }) ||
            capabilities.GetProperty("continuous_controller_t800_recovery_mode").GetString() !=
                ContinuousBotControllerContract.T800RecoveryMode ||
            capabilities.GetProperty("continuous_controller_g1_recovery_mode").GetString() !=
                ContinuousBotControllerContract.G1RecoveryMode ||
            capabilities.GetProperty("continuous_controller_move_indices").GetArrayLength() !=
                continuousAttacks.Length ||
            capabilities.GetProperty("continuous_controller_g1_move_indices").GetArrayLength() !=
                ContinuousBotControllerContract.G1Attacks.Length ||
            capabilities.GetProperty("continuous_controller_attack_profiles").GetArrayLength() !=
                continuousAttacks.Length ||
            capabilities.GetProperty("continuous_controller_g1_attack_profiles").GetArrayLength() !=
                ContinuousBotControllerContract.G1Attacks.Length)
        {
            failures.Enqueue("invalid_continuous_controller_hello");
        }
        if (capabilities.GetProperty("attack_zone_trial_schema").GetString() !=
                AttackZoneTrialContract.Schema ||
            capabilities.GetProperty("attack_zone_trial_sha256").GetString() !=
                AttackZoneTrialContract.ExpectedSha256 ||
            capabilities.GetProperty("attack_zone_trial_authority_scope").GetString() !=
                AttackZoneTrialContract.AuthorityScope ||
            capabilities.GetProperty("attack_zone_trial_authority_caveat").GetString() !=
                AttackZoneTrialContract.AuthorityCaveat ||
            capabilities.GetProperty("attack_zone_trial_required_isolation_proof").GetString() !=
                AttackZoneTrialContract.RequiredIsolationProof ||
            capabilities.GetProperty("attack_zone_trial_control_rate_hz").GetInt32() != 50 ||
            capabilities.GetProperty("attack_zone_trial_fixed_substeps_per_tick").GetInt32() != 10 ||
            capabilities.GetProperty("attack_zone_trial_settle_ticks").GetInt32() != 15 ||
            capabilities.GetProperty("attack_zone_trial_action_sample_rate_hz").GetInt32() != 50 ||
            capabilities.GetProperty("attack_zone_trial_recovery_ready_ticks").GetInt32() != 15 ||
            capabilities.GetProperty("attack_zone_trial_acquisition_timeout_ticks").GetInt32() != 500 ||
            capabilities.GetProperty("attack_zone_trial_minimum_independent_runs_per_cell").GetInt32() != 5 ||
            capabilities.GetProperty("attack_zone_trial_recorder_version").GetString() != "0.7.2" ||
            capabilities.GetProperty("attack_zone_trial_recorder_plugin_sha256").GetString() !=
                AttackZoneTrialContract.ExpectedRecorderPluginSha256 ||
            capabilities.GetProperty("attack_zone_trial_global_input_emitted").GetBoolean())
        {
            failures.Enqueue("invalid_attack_zone_trial_hello");
        }
        if (capabilities.GetProperty("g1_held_schedule_schema").GetString() !=
                G1HeldInputScheduleContract.Schema ||
            capabilities.GetProperty("g1_held_schedule_id").GetString() !=
                G1HeldInputScheduleContract.ScheduleId ||
            capabilities.GetProperty("g1_held_schedule_sha256").GetString() !=
                G1HeldInputScheduleContract.ExpectedSha256 ||
            capabilities.GetProperty("g1_held_schedule_required_isolation_proof").GetString() !=
                G1HeldInputScheduleContract.RequiredIsolationProof ||
            capabilities.GetProperty("g1_held_schedule_unity_fixed_rate_hz").GetInt32() != 500 ||
            capabilities.GetProperty("g1_held_schedule_rate_hz").GetInt32() != 50 ||
            capabilities.GetProperty("g1_held_schedule_fixed_substeps_per_tick").GetInt32() != 10 ||
            capabilities.GetProperty("g1_held_schedule_duration_ticks").GetInt32() != 4551 ||
            capabilities.GetProperty("g1_held_schedule_kick_observation_ticks").GetInt32() != 200 ||
            capabilities.GetProperty("g1_held_schedule_translation_release_offset_ticks").GetInt32() != 5 ||
            capabilities.GetProperty("g1_held_schedule_transition_settled_provenance").GetString() !=
                G1HeldInputScheduleContract.TransitionSettledProvenance ||
            capabilities.GetProperty("g1_held_schedule_transition_settle_base_velocity_provenance").GetString() !=
                G1HeldInputScheduleContract.TransitionSettleBaseVelocityProvenance ||
            capabilities.GetProperty("g1_held_schedule_transition_settle_planar_speed_m_s").GetSingle() !=
                G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed ||
            capabilities.GetProperty("g1_held_schedule_transition_settle_yaw_rate_rad_s").GetSingle() !=
                G1HeldInputScheduleContract.ExpectedTransitionSettleYawRate ||
            capabilities.GetProperty("g1_held_schedule_post_release_settled_kick_control_included").GetBoolean() ||
            capabilities.GetProperty("g1_held_schedule_lifecycle_observation_rate_hz").GetInt32() != 500 ||
            capabilities.GetProperty("g1_held_schedule_held_condition_count").GetInt32() != 14 ||
            !capabilities.GetProperty("g1_held_schedule_kick_move_indices")
                .EnumerateArray().Select(value => value.GetInt32()).SequenceEqual(new[] { 6, 7, 8, 9 }) ||
            capabilities.GetProperty("g1_held_schedule_f_binding_included").GetBoolean() ||
            capabilities.GetProperty("g1_held_schedule_queue_or_retry_used").GetBoolean() ||
            capabilities.GetProperty("g1_held_schedule_sonic_action_composer_lifecycle_used").GetBoolean() ||
            capabilities.GetProperty("g1_held_schedule_global_input_emitted").GetBoolean())
        {
            failures.Enqueue("invalid_g1_held_schedule_hello");
        }
    }

    if (client.IsConnected)
    {
        try
        {
            var bytes = Encoding.UTF8.GetBytes("{\"type\":\"get_state\",\"request_id\":\"pipe-test\"}\n");
            await client.WriteAsync(bytes.AsMemory(), timeout.Token);
            await client.FlushAsync(timeout.Token);
            var request = await accepted.Task.WaitAsync(timeout.Token);
            if (request.Kind != RequestKind.GetState || request.RequestId != "pipe-test")
                failures.Enqueue("pipe_request_mismatch");
            server.Send(request.ConnectionId, new { @event = "test_ack", request_id = request.RequestId });
            var ackLine = await reader.ReadLineAsync().WaitAsync(timeout.Token);
            if (ackLine is null || !ackLine.Contains("test_ack", StringComparison.Ordinal))
                failures.Enqueue("missing_pipe_ack");
        }
        catch (Exception exception)
        {
            failures.Enqueue($"pipe_roundtrip_exception:{exception.GetType().Name}");
        }
    }
}

if (!failures.IsEmpty)
{
    Console.Error.WriteLine($"FAIL {failures.Count}: {string.Join(",", failures)}");
    return 1;
}

Console.WriteLine(
    $"PASS protocol_cases={protocolCases} " +
    "local_pipe_roundtrip=true");
return 0;
