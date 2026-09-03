using System.IO.Pipes;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using RekUiBridgeAgent;

var failures = new ConcurrentQueue<string>();
var protocolCases = 0;

void Expect(string name, bool condition)
{
    protocolCases++;
    if (!condition)
        failures.Enqueue(name);
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
Expect(
    "continuous_schema",
    ContinuousBotControllerContract.Schema == "rek.continuous_private_bot_controller.v1");
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
Expect(
    "continuous_contract_sha256",
    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        ContinuousBotControllerContract.CanonicalJson))).ToLowerInvariant() ==
    ContinuousBotControllerContract.ExpectedSha256);
Expect("continuous_attack_count", continuousAttacks.Length == 6);
Expect(
    "continuous_attack_round_robin_order",
    continuousAttacks.Select(value => value.MoveIndex)
        .SequenceEqual(new[] { 2, 3, 4, 5, 9, 10 }));
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
        "not_measured_runtime_timing",
        StringComparison.Ordinal));
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

var staleOpponentSemanticT800 =
    ContinuousBotControllerContract.ClassifySemanticRuntimeConsistency(
        semanticDeclaresT800: true,
        runtimeIsExactT800: false);
var staleOpponentSemanticG1 =
    ContinuousBotControllerContract.ClassifySemanticRuntimeConsistency(
        semanticDeclaresT800: false,
        runtimeIsExactT800: true);
Expect(
    "continuous_exposes_stale_t800_semantic_against_g1_runtime",
    staleOpponentSemanticT800.Mismatch);
Expect(
    "continuous_exposes_stale_g1_semantic_against_t800_runtime",
    staleOpponentSemanticG1.Mismatch);
Expect(
    "continuous_runtime_identity_classification_independent_of_semantic_id",
    staleOpponentSemanticT800.Classification ==
    staleOpponentSemanticG1.Classification);
Expect(
    "continuous_rejects_non_t800_opponent_runtime",
    !ContinuousBotControllerContract.HasRequiredT800Pairing(
        localSemanticT800: true,
        localExactT800Runtime: true,
        opponentExactT800Runtime: false));
Expect(
    "continuous_accepts_stale_semantic_g1_with_exact_t800_opponent_runtime",
    staleOpponentSemanticG1.Mismatch &&
    ContinuousBotControllerContract.HasRequiredT800Pairing(
        localSemanticT800: true,
        localExactT800Runtime: true,
        opponentExactT800Runtime: true));

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

var exactPairing = BridgePairingContract.Validate(
    "t800",
    exactT800Bones,
    "t800",
    exactT800Bones);
Expect("exact_t800_pairing_accepted", exactPairing.ExactT800VersusT800);
Expect(
    "exact_t800_pairing_reason",
    exactPairing.Reason == BridgePairingContract.ExactPairingReason);

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
    "local_fighter_bone_count_not_26");
Expect(
    "opponent_g1_layout_rejected",
    BridgePairingContract.Validate(
        "t800",
        exactT800Bones,
        "t800",
        Enumerable.Repeat<string?>("pelvis", 30).ToArray()).Reason ==
    "opponent_fighter_bone_count_not_26");
Expect(
    "local_semantic_id_required",
    BridgePairingContract.Validate(null, exactT800Bones, "t800", exactT800Bones).Reason ==
    "local_fighter_robot_id_unavailable");
Expect(
    "opponent_semantic_t800_required",
    BridgePairingContract.Validate("t800", exactT800Bones, "g1", exactT800Bones).Reason ==
    "opponent_fighter_robot_id_not_t800");
Expect(
    "semantic_id_is_exact_case_sensitive",
    !BridgePairingContract.Validate("T800", exactT800Bones, "t800", exactT800Bones)
        .ExactT800VersusT800);

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
            capabilities.GetProperty("continuous_controller_move_indices").GetArrayLength() !=
                continuousAttacks.Length ||
            capabilities.GetProperty("continuous_controller_attack_profiles").GetArrayLength() !=
                continuousAttacks.Length)
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
            capabilities.GetProperty("attack_zone_trial_recorder_version").GetString() != "0.6.1" ||
            capabilities.GetProperty("attack_zone_trial_recorder_plugin_sha256").GetString() !=
                AttackZoneTrialContract.ExpectedRecorderPluginSha256 ||
            capabilities.GetProperty("attack_zone_trial_global_input_emitted").GetBoolean())
        {
            failures.Enqueue("invalid_attack_zone_trial_hello");
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
