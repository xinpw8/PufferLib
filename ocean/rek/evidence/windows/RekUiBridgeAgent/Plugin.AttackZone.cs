using System.Buffers.Binary;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using HarmonyLib;
using REKApp;
using Unity.Netcode;
using UnityEngine;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private readonly HashSet<string> _attackZoneConsumedTrialIds = new(StringComparer.Ordinal);
    private readonly HashSet<string> _attackZoneContaminatedRounds = new(StringComparer.Ordinal);
    private readonly List<AttackZoneRawHitReference> _attackZoneUnassociatedRawHits = new();
    private bool _attackZoneTrialRunning;
    private bool _attackZoneRecoveryOnlyRunning;
    private AttackZoneValidatedTarget? _attackZoneTarget;
    private AttackZoneSettleTracker? _attackZoneSettleTracker;
    private int _attackZoneAcquisitionStartTick;
    private bool _attackZoneNeutralRequestMethodReturned;
    private bool _attackZoneAttackIssued;
    private bool _attackZoneCompletionObserved;
    private string _attackZonePhase = "inactive";
    private string? _attackZoneLastOutcome;
    private string? _attackZoneRecorderObservedVersion;
    private string? _attackZoneRecorderObservedSha256;
    private AttackZoneClock? _attackZoneTargetRequestedClock;
    private AttackZoneClock? _attackZoneActionRequestedClock;
    private AttackZoneClock? _attackZoneActionStartedClock;
    private AttackZoneClock? _attackZoneActionCompletedClock;
    private AttackZoneGeometry? _attackZoneRequestEdgeGeometry;
    private AttackZoneClock? _attackZoneLastActionSampleClock;
    private int _attackZoneActionSampleSequence;
    private int _attackZoneRecoveryReadyTicks;
    private long _attackZoneRawHitSequence;

    private CommandResult StartAttackZoneTrial(AttackZoneTrialTarget? requestedTarget)
    {
        if (!RequireBackgroundControl(out var reason))
            return CommandResult.Rejected(reason);
        if (_scheduleRunning || _singleMotionTrialRunning || _continuousControllerRunning)
            return CommandResult.Rejected("another_control_mode_already_running");
        if (!SameFloatBits(Time.fixedDeltaTime, BridgeScheduleContract.ExpectedFixedDeltaTime))
            return CommandResult.Rejected($"unexpected_fixed_delta_time:{Time.fixedDeltaTime:R}");
        if (!_sendBoundaryPatchesVerified || !_trialIsolationPatchesVerified)
            return CommandResult.Rejected("attack_zone_send_or_observation_boundary_patches_not_verified");
        if (!AttackZoneTrialContract.TryValidateTarget(
                requestedTarget,
                out var target,
                out reason))
        {
            return CommandResult.Rejected(reason);
        }
        if (_attackZoneConsumedTrialIds.Contains(target.Request.TrialId))
            return CommandResult.Rejected("attack_zone_trial_id_already_consumed");
        if (!TryVerifyAttackZoneRecorder(
                out _attackZoneRecorderObservedVersion,
                out _attackZoneRecorderObservedSha256,
                out reason))
        {
            return CommandResult.Rejected(reason);
        }
        if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out reason))
            return CommandResult.Rejected(reason);
        var measuredPairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        var measuredPairingPayload = MeasuredPairingPayload(measuredPairing);
        if (!TryValidateContinuousPairing(measuredPairing, out reason))
        {
            return CommandResult.Rejected(
                $"attack_zone_runtime_pairing_not_proven:{reason}",
                measuredPairingPayload);
        }
        if (scope.Input is null || !TryValidateContinuousMoveMap(scope.Input, out reason))
            return CommandResult.Rejected(reason, measuredPairingPayload);
        if (scope.Input.hasPendingMove || scope.Input.hasPendingSpecial || scope.Input.hasPendingEStop)
            return CommandResult.Rejected("attack_zone_initial_pending_command", measuredPairingPayload);
        if (!Finite(scope.Input.VelocityCommand) ||
            !VelocityEquals(scope.Input.VelocityCommand, Vector3.zero))
        {
            return CommandResult.Rejected(
                "attack_zone_initial_velocity_not_exact_neutral",
                measuredPairingPayload);
        }
        if (!TryCreateTrialRoundIdentity(scope, out var roundIdentity, out reason))
            return CommandResult.Rejected(reason, measuredPairingPayload);
        var sessionIdentity = roundIdentity.SessionIdentity;
        var sessionIdentitySha256 = HashTrialSessionIdentity(sessionIdentity);
        var roundIdentitySha256 = HashTrialRoundIdentity(roundIdentity);
        if (_attackZoneContaminatedRounds.Contains(roundIdentitySha256))
            return CommandResult.Rejected("attack_zone_round_excluded_after_post_start_fall", measuredPairingPayload);
        if (!TryCaptureContinuousFrame(scope, measuredPairing, out var frame, out reason))
            return CommandResult.Rejected(reason, measuredPairingPayload);
        if (!ContinuousLocalActionReady(frame) || AttackZoneOpponentUnhealthy(frame))
            return CommandResult.Rejected("attack_zone_initial_fighter_readiness_not_proven", measuredPairingPayload);

        var scopeValidation = AttackZoneTrialContract.ValidateScope(
            BuildAttackZoneScopeObservation(
                scope,
                measuredPairing,
                frame,
                sessionIdentitySha256,
                roundIdentitySha256),
            target);
        if (!scopeValidation.Accepted)
            return CommandResult.Rejected(scopeValidation.Reason, measuredPairingPayload);

        _attackZoneConsumedTrialIds.Add(target.Request.TrialId);
        _attackZoneTarget = target;
        _attackZoneSettleTracker = new AttackZoneSettleTracker(target);
        _attackZoneTrialRunning = true;
        _attackZoneRecoveryOnlyRunning = false;
        _attackZonePhase = "target_acquisition";
        _attackZoneLastOutcome = null;
        _attackZoneNeutralRequestMethodReturned = false;
        _attackZoneAttackIssued = false;
        _attackZoneCompletionObserved = false;
        _attackZoneTargetRequestedClock = CaptureAttackZoneClock();
        _attackZoneActionRequestedClock = null;
        _attackZoneActionStartedClock = null;
        _attackZoneActionCompletedClock = null;
        _attackZoneRequestEdgeGeometry = null;
        _attackZoneLastActionSampleClock = null;
        _attackZoneActionSampleSequence = 0;
        _attackZoneRecoveryReadyTicks = 0;
        _attackZoneUnassociatedRawHits.Clear();

        _continuousControllerRunId = target.Request.IndependentRunId;
        _continuousControllerSessionIdentity = sessionIdentity;
        _continuousControllerRoundIdentity = RuntimeIdentity.From(scope);
        _continuousControllerRoundIdentitySha256 = roundIdentitySha256;
        _continuousControllerLastRoundIdentitySha256 = roundIdentitySha256;
        _continuousControllerInput = frame.Input;
        _continuousControllerInputPointer = NativePointer(frame.Input);
        _continuousControllerVelocity = Vector3.zero;
        _continuousControllerFixedSubstep = 0;
        _continuousControllerTick = 0;
        _continuousControllerRoundTick = 0;
        _continuousControllerRoundSequence = 1;
        _continuousControllerTelemetrySequence = 0;
        _continuousControllerPhase = "attack_zone_target_acquisition";
        _continuousControllerSuspendReason = null;
        _continuousControllerActionSequence = target.Request.ActionSequence - 1;
        _continuousControllerNextAttackIndex = Array.FindIndex(
            ContinuousBotControllerContract.Attacks,
            value => value.MoveIndex == target.Request.MoveIndex);
        _continuousControllerLastFrame = frame;
        _continuousControllerLastRoundMetrics = frame.RoundMetrics;
        _continuousControllerRecoveryEpisodeActive = false;
        _continuousControllerRecoveryStage = "inactive";
        _continuousControllerRecoveryStageTick = 0;
        _continuousControllerRecoverySequence = 0;
        _continuousControllerStraightenIssued = false;
        ClearContinuousActionState();
        ClearContinuousPendingRequestState();
        _continuousControllerAuthorizedWhileBackground = true;
        _continuousControllerRunning = true;
        _attackZoneAcquisitionStartTick = 0;

        EmitAttackZoneEvent(
            "target_requested",
            scopeValidation.Reason,
            frame,
            new
            {
                target = AttackZoneTargetPayload(target.Request),
                measured_pairing = measuredPairingPayload,
                recorder_observed_version = _attackZoneRecorderObservedVersion,
                recorder_observed_plugin_sha256 = _attackZoneRecorderObservedSha256,
                target_request_clock = _attackZoneTargetRequestedClock,
                global_input_used = false,
            },
            _attackZoneTargetRequestedClock);
        ForceAttackZoneNeutralRequest(frame.Input, "attack_zone_initial_neutral_settle");
        return CommandResult.AppliedResult(
            "attack_zone_target_acquisition_started",
            measuredPairingPayload);
    }

    private CommandResult StopAttackZoneTrial()
    {
        if (_attackZoneRecoveryOnlyRunning)
        {
            StopContinuousController("attack_zone_recovery_only_requested_stop");
            return CommandResult.AppliedResult("attack_zone_recovery_only_stopped");
        }
        if (!_attackZoneTrialRunning)
            return CommandResult.Rejected("attack_zone_trial_not_running");
        StopAttackZoneTrialInternal("requested", "trial_interrupted");
        return CommandResult.AppliedResult("attack_zone_trial_stopped");
    }

    private void AdvanceAttackZoneTrial()
    {
        if (!_attackZoneTrialRunning || _attackZoneTarget is null ||
            _attackZoneSettleTracker is null)
        {
            StopAttackZoneTrialInternal("attack_zone_runtime_state_missing", "trial_interrupted");
            return;
        }
        if (!RequireOwnedContinuousControl(out var controlReason))
        {
            StopAttackZoneTrialInternal(controlReason, "trial_interrupted");
            return;
        }

        var fixedSubstep = _continuousControllerFixedSubstep++;
        if (fixedSubstep % AttackZoneTrialContract.FixedSubstepsPerControlTick != 0)
            return;
        _continuousControllerTick =
            fixedSubstep / AttackZoneTrialContract.FixedSubstepsPerControlTick;
        _continuousControllerRoundTick++;

        if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var reason))
        {
            StopAttackZoneTrialInternal($"attack_zone_scope_lost:{reason}", "trial_interrupted");
            return;
        }
        if (!TryCreateTrialRoundIdentity(scope, out var trialRoundIdentity, out reason))
        {
            StopAttackZoneTrialInternal(reason, "trial_interrupted");
            return;
        }
        var sessionIdentitySha256 = HashTrialSessionIdentity(trialRoundIdentity.SessionIdentity);
        var roundIdentitySha256 = HashTrialRoundIdentity(trialRoundIdentity);
        if (_continuousControllerRoundIdentity is null ||
            !_continuousControllerRoundIdentity.Equals(RuntimeIdentity.From(scope)) ||
            !string.Equals(
                _continuousControllerRoundIdentitySha256,
                roundIdentitySha256,
                StringComparison.Ordinal))
        {
            StopAttackZoneTrialInternal(
                "attack_zone_session_or_round_identity_changed",
                "trial_interrupted");
            return;
        }
        var pairing = ReadMeasuredPairing(scope.Coordinator, scope.LocalSlot, scope.OpponentSlot);
        if (!TryCaptureContinuousFrame(scope, pairing, out var frame, out reason))
        {
            StopAttackZoneTrialInternal(reason, "trial_interrupted");
            return;
        }
        if (!ContinuousRuntimeIdentityMatches(frame, _continuousControllerLastFrame))
        {
            StopAttackZoneTrialInternal(
                "attack_zone_runtime_fighter_identity_changed",
                "trial_interrupted");
            return;
        }
        if (_continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(frame.Input) != _continuousControllerInputPointer ||
            !VelocityEquals(frame.Input.VelocityCommand, _continuousControllerVelocity))
        {
            StopAttackZoneTrialInternal(
                "attack_zone_input_binding_or_velocity_changed_outside_owned_edge",
                "trial_interrupted");
            return;
        }
        if ((frame.Input.hasPendingMove &&
             (_continuousControllerMoveAwaitingSend is null ||
              frame.Input.pendingMoveIndex != _continuousControllerMoveAwaitingSend.Value)) ||
            frame.Input.hasPendingSpecial || frame.Input.hasPendingEStop)
        {
            StopAttackZoneTrialInternal(
                "attack_zone_unowned_pending_command_observed",
                "trial_interrupted");
            return;
        }
        var scopeValidation = AttackZoneTrialContract.ValidateScope(
            BuildAttackZoneScopeObservation(
                scope,
                pairing,
                frame,
                sessionIdentitySha256,
                roundIdentitySha256),
            _attackZoneTarget);
        if (!scopeValidation.Accepted)
        {
            StopAttackZoneTrialInternal(scopeValidation.Reason, "trial_interrupted");
            return;
        }

        var previousMetrics = _continuousControllerLastRoundMetrics;
        _continuousControllerLastFrame = frame;
        ObserveAttackZoneRoundDelta(frame, previousMetrics);
        _continuousControllerLastRoundMetrics = frame.RoundMetrics;

        if (_attackZoneAttackIssued && !EmitAttackZoneActionSample(frame))
            return;

        if (AttackZoneAnyFighterUnhealthy(frame))
        {
            ObserveAttackZoneFallOrRecovery(frame);
            if (_continuousControllerActionStartedObserved &&
                _continuousControllerRoundIdentitySha256 is not null)
            {
                _attackZoneContaminatedRounds.Add(_continuousControllerRoundIdentitySha256);
            }
            var disposition = AttackZoneTrialContract.ClassifyCensorDisposition(
                frame.LocalFalling,
                frame.LocalFallen,
                frame.LocalDampened,
                frame.LocalRecoveryArmed,
                frame.LocalGetUpPending,
                frame.LocalResetting,
                frame.LocalMotorShutdown,
                frame.InputRecovering,
                AttackZoneOpponentUnhealthy(frame));
            StopAttackZoneTrialInternal(
                _continuousControllerActionStartedObserved
                    ? "post_start_fall_or_recovery_censored_remainder_of_round"
                    : "pre_start_fall_or_recovery_contamination",
                "trial_censored",
                continueLocalRecovery:
                    disposition == AttackZoneCensorDisposition.ContinueLocalRecovery);
            return;
        }

        if (_attackZoneAttackIssued)
        {
            ObserveAttackZoneActionLifecycle(frame);
            return;
        }

        var decision = AttackZoneTrialContract.DecideAcquisition(
            _attackZoneTarget,
            frame.Geometry);
        AttackZoneSettleUpdate? update = null;
        if (!decision.ExactNeutral)
        {
            _attackZoneNeutralRequestMethodReturned = false;
            _attackZoneSettleTracker.Reset();
            SetContinuousVelocity(
                frame.Input,
                new Vector3(decision.Forward, decision.Strafe, decision.Yaw),
                "attack_zone_acquisition_motion");
        }
        else
        {
            if (!VelocityEquals(_continuousControllerVelocity, Vector3.zero))
            {
                _attackZoneNeutralRequestMethodReturned = false;
                SetContinuousVelocity(frame.Input, Vector3.zero, "attack_zone_neutral_settle");
            }
            var sample = BuildAttackZoneControlObservation(frame);
            update = _attackZoneSettleTracker.Observe(sample);
        }

        EmitAttackZoneEvent(
            "acquisition_sample",
            decision.Reason,
            frame,
            new
            {
                target = AttackZoneTargetPayload(_attackZoneTarget.Request),
                acquisition_decision = new
                {
                    forward_command = decision.Forward,
                    strafe_command = decision.Strafe,
                    yaw_command = decision.Yaw,
                    exact_neutral = decision.ExactNeutral,
                    yaw_rule = AttackZoneTrialContract.AcquisitionYawRule,
                },
                settle = update is null ? null : AttackZoneSettleUpdatePayload(update),
            });

        if (_attackZoneSettleTracker.AcquisitionTimedOut(
                _continuousControllerTick - _attackZoneAcquisitionStartTick))
        {
            EmitAttackZoneEvent(
                "target_not_acquired_unresolved",
                "target_acquisition_timeout",
                frame,
                new
                {
                    elapsed_control_ticks =
                        _continuousControllerTick - _attackZoneAcquisitionStartTick,
                    settle_digest = _attackZoneSettleTracker.AcquiredDigest,
                    attack_requested = false,
                });
            StopAttackZoneTrialInternal(
                "target_not_acquired_unresolved",
                "trial_censored");
            return;
        }
        if (update?.Acquired == true)
        {
            _attackZonePhase = "target_acquired";
            _continuousControllerPhase = "attack_zone_target_acquired";
            EmitAttackZoneEvent(
                "target_acquired",
                update.Reason,
                frame,
                new
                {
                    settle_digest = AttackZoneSettleDigestPayload(update.Digest!),
                    primary_stationary_stratum = update.Digest!.AllOpponentStationary,
                    opponent_motion_stratum = update.Digest.OpponentMotionStratum,
                    opponent_facing_stratum = update.Digest.OpponentFacingStratum,
                });
            StartAttackZoneAction(frame);
        }
    }

    private void StartAttackZoneAction(ContinuousFrame frame)
    {
        var target = _attackZoneTarget;
        if (!_attackZoneTrialRunning || target is null || _attackZoneAttackIssued)
            return;
        if (!_attackZoneNeutralRequestMethodReturned ||
            _continuousControllerVelocityPurposeAwaitingSend is not null ||
            !VelocityEquals(_continuousControllerVelocity, Vector3.zero) ||
            !VelocityEquals(frame.Input.VelocityCommand, Vector3.zero) ||
            !ContinuousLocalActionReady(frame))
        {
            StopAttackZoneTrialInternal(
                "attack_zone_attack_edge_without_completed_neutral_settle",
                "trial_interrupted");
            return;
        }
        _attackZoneAttackIssued = true;
        _attackZoneActionRequestedClock = CaptureAttackZoneClock();
        _attackZoneRequestEdgeGeometry = BuildAttackZoneGeometry(frame);
        _continuousControllerPhase = "attack_zone_action_requested";
        TryStartContinuousAttack(frame, target.Attack);
        if (!_attackZoneTrialRunning)
            return;
        if (_continuousControllerActionSequence != target.Request.ActionSequence ||
            _continuousControllerActiveAttack?.MoveIndex != target.Request.MoveIndex)
        {
            StopAttackZoneTrialInternal(
                "attack_zone_action_identity_mismatch_after_local_edge",
                "trial_interrupted");
        }
    }

    private bool EmitAttackZoneActionSample(ContinuousFrame frame)
    {
        var clock = CaptureAttackZoneClock();
        if (_attackZoneLastActionSampleClock is not null &&
            !AttackZoneTrialContract.ClocksAreConsecutive(
                _attackZoneLastActionSampleClock,
                clock))
        {
            StopAttackZoneTrialInternal(
                "attack_zone_action_sample_cadence_not_measured_50hz",
                "trial_interrupted");
            return false;
        }
        _attackZoneLastActionSampleClock = clock;
        _attackZoneActionSampleSequence++;
        EmitAttackZoneEvent(
            "action_sample",
            "periodic_measured_action_state",
            frame,
            new
            {
                action_sample_sequence = _attackZoneActionSampleSequence,
                expected_control_rate_hz = AttackZoneTrialContract.ControlRateHz,
                geometry = BuildAttackZoneGeometry(frame),
                active_clip_name = frame.ActiveActionClipName,
                active_clip_frame = frame.ActionClipFrame,
                active_clip_fps = frame.ActionClipFps,
                composer_action_playing = frame.ComposerActionPlaying,
                composer_busy = frame.ComposerBusy,
                input_punching = frame.InputPunching,
                input_recovering = frame.InputRecovering,
                request_relative_seconds = AttackZoneElapsedSeconds(
                    _attackZoneActionRequestedClock,
                    clock),
                start_relative_seconds = AttackZoneElapsedSeconds(
                    _attackZoneActionStartedClock,
                    clock),
                round_metrics = new
                {
                    local_clean_hits = frame.RoundMetrics.LocalCleanHits,
                    opponent_clean_hits = frame.RoundMetrics.OpponentCleanHits,
                    local_falls = frame.RoundMetrics.LocalFalls,
                    opponent_falls = frame.RoundMetrics.OpponentFalls,
                },
                fall_and_recovery = new
                {
                    local_falling = frame.LocalFalling,
                    local_fallen = frame.LocalFallen,
                    local_dampened = frame.LocalDampened,
                    local_recovery_armed = frame.LocalRecoveryArmed,
                    local_get_up_pending = frame.LocalGetUpPending,
                    local_resetting = frame.LocalResetting,
                    local_motor_shutdown = frame.LocalMotorShutdown,
                    opponent_falling = frame.OpponentFalling,
                    opponent_fallen = frame.OpponentFallen,
                    opponent_dampened = frame.OpponentDampened,
                    opponent_recovery_armed = frame.OpponentRecoveryArmed,
                    opponent_get_up_pending = frame.OpponentGetUpPending,
                    opponent_resetting = frame.OpponentResetting,
                    opponent_motor_shutdown = frame.OpponentMotorShutdown,
                },
                full_measured_frame_in_event_envelope = true,
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
            },
            clock);
        return _attackZoneTrialRunning;
    }

    private static double? AttackZoneElapsedSeconds(
        AttackZoneClock? start,
        AttackZoneClock end) =>
        start is null || start.StopwatchFrequencyHz != end.StopwatchFrequencyHz
            ? null
            : (end.StopwatchTimestampTicks - start.StopwatchTimestampTicks) /
              (double)end.StopwatchFrequencyHz;

    private void ObserveAttackZoneActionLifecycle(ContinuousFrame frame)
    {
        if (_continuousControllerActiveAttack is null || _attackZoneTarget is null)
        {
            StopAttackZoneTrialInternal(
                "attack_zone_active_action_state_missing",
                "trial_interrupted");
            return;
        }
        if (!_continuousControllerMoveRequestObserved)
        {
            if (_continuousControllerTick - _continuousControllerActionRequestTick >
                AttackZoneTrialContract.RequestStartTimeoutTicks)
            {
                StopAttackZoneTrialInternal(
                    "attack_zone_move_client_request_method_return_timeout",
                    "trial_interrupted");
            }
            return;
        }
        if (!_continuousControllerActionStartedObserved)
        {
            if (frame.ComposerActionPlaying)
            {
                if (frame.ActiveActionClipPointer == IntPtr.Zero ||
                    frame.ActiveActionClipPointer != _continuousControllerActiveClipPointer ||
                    !string.Equals(
                        frame.ActiveActionClipName,
                        _attackZoneTarget.Attack.MoveName,
                        StringComparison.Ordinal))
                {
                    StopAttackZoneTrialInternal(
                        "attack_zone_unexpected_local_action_clip_started",
                        "trial_interrupted");
                    return;
                }
                _continuousControllerActionStartedObserved = true;
                _continuousControllerActionStartTick = _continuousControllerTick;
                _attackZoneActionStartedClock = CaptureAttackZoneClock();
                _attackZonePhase = "local_motion_observed";
                _continuousControllerPhase = "attack_zone_local_motion_observed";
                EmitAttackZoneEvent(
                    "local_motion_start_observed",
                    "exact_selected_clip_started_locally",
                    frame,
                    AttackZoneActionLifecyclePayload("local_motion_start_observed"),
                    _attackZoneActionStartedClock);
                EmitAttackZoneStaticMarkerProjections(frame);
                return;
            }
            if (_continuousControllerTick - _continuousControllerActionRequestTick >
                AttackZoneTrialContract.RequestStartTimeoutTicks)
            {
                StopAttackZoneTrialInternal(
                    "attack_zone_local_motion_start_observation_timeout",
                    "trial_interrupted");
            }
            return;
        }
        if (frame.ComposerActionPlaying &&
            frame.ActiveActionClipPointer != _continuousControllerActiveClipPointer)
        {
            StopAttackZoneTrialInternal(
                "attack_zone_local_action_clip_changed_during_lifecycle",
                "trial_interrupted");
            return;
        }
        if (!frame.ComposerActionPlaying && !frame.ComposerBusy &&
            !frame.InputPunching && !frame.InputRecovering &&
            !frame.Input.hasPendingMove && !frame.Input.hasPendingSpecial &&
            !frame.Input.hasPendingEStop)
        {
            _attackZoneCompletionObserved = true;
            _attackZoneActionCompletedClock = CaptureAttackZoneClock();
            EmitAttackZoneEvent(
                "local_motion_completion_and_readiness_observed",
                "selected_local_motion_completed_and_readiness_returned",
                frame,
                AttackZoneActionLifecyclePayload(
                    "local_motion_completion_and_readiness_observed"),
                _attackZoneActionCompletedClock);
            StopAttackZoneTrialInternal("completed", "trial_completed");
            return;
        }
        if (_continuousControllerTick - _continuousControllerActionStartTick >
            AttackZoneTrialContract.CompletionTimeoutTicks)
        {
            StopAttackZoneTrialInternal(
                "attack_zone_local_motion_completion_timeout",
                "trial_censored");
        }
    }

    private void ForceAttackZoneNeutralRequest(RobotInputController input, string purpose)
    {
        if (!_attackZoneTrialRunning || _continuousControllerVelocityPurposeAwaitingSend is not null)
        {
            StopAttackZoneTrialInternal(
                "attack_zone_neutral_request_could_not_be_armed",
                "trial_interrupted");
            return;
        }
        _continuousControllerVelocity = Vector3.zero;
        _continuousControllerVelocityPurposeAwaitingSend = purpose;
        _continuousControllerVelocityInvocationObserved = false;
        _attackZoneNeutralRequestMethodReturned = false;
        if (!SetVelocityExact(input, Vector3.zero))
        {
            StopAttackZoneTrialInternal(
                "attack_zone_neutral_velocity_readback_mismatch",
                "trial_interrupted");
            return;
        }
        EmitAttackZoneEvent(
            "neutral_command_edge_set",
            purpose,
            _continuousControllerLastFrame,
            new
            {
                lifecycle_stage = "local_velocity_command_edge_set",
                purpose,
                velocity_command_xyz = new[] { 0f, 0f, 0f },
            });
    }

    private void OnAttackZoneVelocityRequestReturned(string purpose)
    {
        if (!_attackZoneTrialRunning)
            return;
        if (purpose.Contains("neutral", StringComparison.Ordinal) &&
            VelocityEquals(_continuousControllerVelocity, Vector3.zero))
        {
            _attackZoneNeutralRequestMethodReturned = true;
        }
    }

    private static string MapAttackZoneEventName(string eventName, string reason)
    {
        if (string.Equals(eventName, "continuous_velocity_lifecycle", StringComparison.Ordinal))
        {
            if (string.Equals(reason, "local_velocity_command_edge_set", StringComparison.Ordinal))
                return "acquisition_velocity_command_edge_set";
            if (string.Equals(reason, "client_request_method_returned", StringComparison.Ordinal))
                return "acquisition_velocity_request_method_returned";
        }
        if (string.Equals(eventName, "continuous_action_lifecycle", StringComparison.Ordinal) &&
            reason is "local_command_edge_set" or "client_request_method_returned" or
                "local_motion_start_observed" or
                "local_motion_completion_and_readiness_observed" or "interrupted")
        {
            return reason == "interrupted" ? "trial_interrupted" : reason;
        }
        if (string.Equals(eventName, "continuous_recovery_lifecycle", StringComparison.Ordinal))
            return AttackZoneTrialContract.MapRecoveryLifecycleEventName(reason);
        return eventName;
    }

    private AttackZoneScopeObservation BuildAttackZoneScopeObservation(
        PrivateAiContext scope,
        MeasuredPairing pairing,
        ContinuousFrame frame,
        string sessionIdentitySha256,
        string roundIdentitySha256)
    {
        var isolationVerified = TryVerifyExplicitIsolatedSession(out var isolationProof);
        return new AttackZoneScopeObservation(
            isolationVerified,
            isolationProof,
            _leaseConnectionId != 0 && _leaseConnectionId == (_pipe?.CurrentConnectionId ?? 0),
            _leaseConnectionId,
            GlobalInputUsed: false,
            SemanticCommandSurfaceAvailable: true,
            PrivateSessionProven: true,
            Ranked: scope.Context.IsRanked || scope.Coordinator.IsRankedArena,
            ExactSparringBotOne:
                scope.Coordinator.SparringBotNumber == 1 && scope.Coordinator.OpponentIsAI,
            ActiveRound: scope.RoundActive &&
                         scope.Coordinator.CurrentPhase == FightPhase.RoundActive,
            FighterCount: scope.Coordinator.Fighters?.Length ?? 0,
            LocalSemanticT800: pairing.Validation.LocalSemanticT800,
            LocalRuntimeExactT800: pairing.Validation.LocalExactT800BoneSignature,
            OpponentRuntimeExactT800: pairing.Validation.OpponentExactT800BoneSignature,
            OpponentRuntimeIdentitySha256: frame.OpponentRuntimeIdentitySha256,
            OpponentSemanticRuntimeMismatch: frame.OpponentSemanticRuntimeMismatch,
            BuildHashesMatch:
                string.Equals(
                    _gameAssemblySha256,
                    AttackZoneTrialContract.ExpectedGameAssemblySha256,
                    StringComparison.OrdinalIgnoreCase) &&
                string.Equals(
                    _metadataSha256,
                    AttackZoneTrialContract.ExpectedGlobalMetadataSha256,
                    StringComparison.OrdinalIgnoreCase),
            ControllerContractHashMatch:
                string.Equals(
                    _continuousControllerSha256,
                    ContinuousBotControllerContract.ExpectedSha256,
                    StringComparison.Ordinal) &&
                string.Equals(
                    _attackZoneContractSha256,
                    AttackZoneTrialContract.ExpectedSha256,
                    StringComparison.Ordinal),
            RecorderPinMatch:
                string.Equals(
                    _attackZoneRecorderObservedVersion,
                    AttackZoneTrialContract.ExpectedRecorderVersion,
                    StringComparison.Ordinal) &&
                string.Equals(
                    _attackZoneRecorderObservedSha256,
                    AttackZoneTrialContract.ExpectedRecorderPluginSha256,
                    StringComparison.Ordinal),
            SendBoundaryPatchesVerified:
                _sendBoundaryPatchesVerified && _trialIsolationPatchesVerified,
            SessionIdentitySha256: sessionIdentitySha256,
            RoundIdentitySha256: roundIdentitySha256);
    }

    private AttackZoneControlObservation BuildAttackZoneControlObservation(
        ContinuousFrame frame) => new(
        CaptureAttackZoneClock(),
        new AttackZoneRootObservation(
            frame.LocalPosition.x,
            frame.LocalPosition.y,
            frame.LocalPosition.z,
            frame.LocalRotation.x,
            frame.LocalRotation.y,
            frame.LocalRotation.z,
            frame.LocalRotation.w,
            frame.LocalLinearVelocity.x,
            frame.LocalLinearVelocity.y,
            frame.LocalLinearVelocity.z,
            frame.LocalAngularVelocity.x,
            frame.LocalAngularVelocity.y,
            frame.LocalAngularVelocity.z,
            frame.LocalFalling,
            frame.LocalFallen,
            frame.InputRecovering || frame.LocalRecoveryArmed || frame.LocalGetUpPending,
            frame.LocalDampened,
            frame.LocalResetting,
            frame.LocalMotorShutdown),
        new AttackZoneRootObservation(
            frame.OpponentPosition.x,
            frame.OpponentPosition.y,
            frame.OpponentPosition.z,
            frame.OpponentRotation.x,
            frame.OpponentRotation.y,
            frame.OpponentRotation.z,
            frame.OpponentRotation.w,
            frame.OpponentLinearVelocity.x,
            frame.OpponentLinearVelocity.y,
            frame.OpponentLinearVelocity.z,
            frame.OpponentAngularVelocity.x,
            frame.OpponentAngularVelocity.y,
            frame.OpponentAngularVelocity.z,
            frame.OpponentFalling,
            frame.OpponentFallen,
            frame.OpponentRecoveryArmed || frame.OpponentGetUpPending,
            frame.OpponentDampened,
            frame.OpponentResetting,
            frame.OpponentMotorShutdown),
        new AttackZoneAnimationObservation(
            frame.ComposerActionPlaying,
            frame.ActiveActionClipName,
            frame.ActiveActionClipName is null ? null : frame.ActionClipFrame,
            frame.ActiveActionClipName is null ? null : frame.ActionClipFps),
        new AttackZoneAnimationObservation(false, null, null, null),
        _attackZoneNeutralRequestMethodReturned,
        VelocityEquals(frame.Input.VelocityCommand, Vector3.zero),
        ContinuousLocalActionReady(frame),
        frame.Input.hasPendingMove,
        frame.Input.hasPendingSpecial,
        frame.Input.hasPendingEStop);

    private void ObserveAttackZoneRoundDelta(
        ContinuousFrame frame,
        ContinuousRoundMetrics? previous)
    {
        if (previous is null || previous.Equals(frame.RoundMetrics))
            return;
        var localCleanHitDelta = frame.RoundMetrics.LocalCleanHits - previous.LocalCleanHits;
        var opponentCleanHitDelta =
            frame.RoundMetrics.OpponentCleanHits - previous.OpponentCleanHits;
        var localFallDelta = frame.RoundMetrics.LocalFalls - previous.LocalFalls;
        var opponentFallDelta = frame.RoundMetrics.OpponentFalls - previous.OpponentFalls;
        var isolatedActionInterval =
            _attackZoneAttackIssued && _continuousControllerActionStartedObserved &&
            !_attackZoneCompletionObserved;
        var eligibleRawHits = _attackZoneUnassociatedRawHits
            .Where(value =>
                _attackZoneActionStartedClock is not null &&
                value.Clock.StopwatchTimestampTicks >=
                    _attackZoneActionStartedClock.StopwatchTimestampTicks)
            .ToArray();
        var associationStatus = localCleanHitDelta > 0 && isolatedActionInterval &&
                                eligibleRawHits.Length == 1
            ? "selected_local_action_temporal_association_confirmed"
            : localCleanHitDelta > 0 && isolatedActionInterval
                ? "ambiguous_multiple_or_missing_raw_hit_observations"
                : "unknown_or_not_local_selected_action";
        EmitAttackZoneEvent(
            "round_score_delta_observed",
            "separate_round_metric_delta_observed",
            frame,
            new
            {
                local_clean_hit_delta = localCleanHitDelta,
                opponent_clean_hit_delta = opponentCleanHitDelta,
                local_fall_delta = localFallDelta,
                opponent_fall_delta = opponentFallDelta,
                prior = previous,
                current = frame.RoundMetrics,
                isolated_selected_local_action_interval = isolatedActionInterval,
                raw_hit_sequences = eligibleRawHits.Select(value => value.Sequence).ToArray(),
                association_status = associationStatus,
                raw_rek_hit_alone_used_for_attribution = false,
                body_zone_claimed = false,
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
            });
        if (localCleanHitDelta != 0 || opponentCleanHitDelta != 0)
            _attackZoneUnassociatedRawHits.Clear();
    }

    private void ObserveAttackZoneFallOrRecovery(ContinuousFrame frame)
    {
        if (frame.LocalFalling || frame.LocalFallen ||
            frame.OpponentFalling || frame.OpponentFallen)
        {
            EmitAttackZoneEvent(
                "fall_observed",
                "fighter_fall_state_observed",
                frame,
                new
                {
                    local_falling = frame.LocalFalling,
                    local_fallen = frame.LocalFallen,
                    opponent_falling = frame.OpponentFalling,
                    opponent_fallen = frame.OpponentFallen,
                    post_start = _continuousControllerActionStartedObserved,
                    remainder_of_round_excluded = _continuousControllerActionStartedObserved,
                });
        }
        if (frame.InputRecovering || frame.LocalRecoveryArmed || frame.LocalGetUpPending ||
            frame.LocalDampened || frame.LocalResetting || frame.LocalMotorShutdown ||
            frame.OpponentRecoveryArmed || frame.OpponentGetUpPending ||
            frame.OpponentDampened || frame.OpponentResetting ||
            frame.OpponentMotorShutdown)
        {
            EmitAttackZoneEvent(
                "recovery_state_observed",
                "fighter_recovery_or_reset_state_observed",
                frame,
                new
                {
                    recovery_request_issued_by_attack_zone_runner = false,
                    post_start = _continuousControllerActionStartedObserved,
                });
        }
    }

    private static bool AttackZoneOpponentUnhealthy(ContinuousFrame frame) =>
        frame.OpponentFalling || frame.OpponentFallen || frame.OpponentDampened ||
        frame.OpponentRecoveryArmed || frame.OpponentGetUpPending ||
        frame.OpponentResetting || frame.OpponentMotorShutdown;

    private static bool AttackZoneAnyFighterUnhealthy(ContinuousFrame frame) =>
        frame.LocalFalling || frame.LocalFallen || frame.InputRecovering ||
        frame.LocalDampened || frame.LocalRecoveryArmed || frame.LocalGetUpPending ||
        frame.LocalResetting || frame.LocalMotorShutdown ||
        AttackZoneOpponentUnhealthy(frame);

    private void EmitAttackZoneStaticMarkerProjections(ContinuousFrame frame)
    {
        if (_attackZoneTarget is null || _attackZoneActionStartedClock is null)
            return;
        foreach (var marker in _attackZoneTarget.Attack.StaticImpactEvents)
        {
            var projectedStopwatchTicks =
                _attackZoneActionStartedClock.StopwatchTimestampTicks +
                (long)Math.Round(
                    marker.ImpactTimeSeconds *
                    _attackZoneActionStartedClock.StopwatchFrequencyHz,
                    MidpointRounding.AwayFromZero);
            var projectedUtc = DateTimeOffset.Parse(_attackZoneActionStartedClock.Utc)
                .AddSeconds(marker.ImpactTimeSeconds);
            EmitAttackZoneEvent(
                "configured_asset_marker_projected",
                "configured_asset_timing_projection_not_observed_contact",
                frame,
                new
                {
                    marker.ImpactTimeSeconds,
                    marker.LeadTimeSeconds,
                    marker.ReleaseTimeSeconds,
                    marker.Limb,
                    projected_stopwatch_timestamp_ticks = projectedStopwatchTicks,
                    projected_utc = projectedUtc,
                    projection_origin = _attackZoneActionStartedClock,
                    observed_contact = false,
                    observed_hit_ownership = false,
                });
        }
    }

    internal unsafe void ObserveAttackZoneRawHit(FastBufferReader reader)
    {
        if (!_attackZoneTrialRunning)
            return;
        var body = CopyAttackZoneReaderBody(reader, 29);
        var sequence = ++_attackZoneRawHitSequence;
        var clock = CaptureAttackZoneClock();
        if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var scopeReason))
        {
            EmitAttackZoneEvent(
                "raw_rek_hit_observed",
                "raw_hit_preserved_but_contemporaneous_scope_unavailable",
                _continuousControllerLastFrame,
                AttackZoneRawHitPayload(body, sequence, clock, null, null, scopeReason),
                clock);
            StopAttackZoneTrialInternal(
                $"raw_hit_context_capture_failed:{scopeReason}",
                "trial_censored");
            return;
        }
        var pairing = ReadMeasuredPairing(scope.Coordinator, scope.LocalSlot, scope.OpponentSlot);
        if (!TryCaptureContinuousFrame(scope, pairing, out var frame, out var frameReason))
        {
            EmitAttackZoneEvent(
                "raw_rek_hit_observed",
                "raw_hit_preserved_but_contemporaneous_frame_unavailable",
                _continuousControllerLastFrame,
                AttackZoneRawHitPayload(body, sequence, clock, null, null, frameReason),
                clock);
            StopAttackZoneTrialInternal(
                $"raw_hit_context_capture_failed:{frameReason}",
                "trial_censored");
            return;
        }
        var opponentContext = CaptureAttackZoneOpponentContactContext(frame.OpponentRobot);
        _attackZoneUnassociatedRawHits.Add(new AttackZoneRawHitReference(sequence, clock));
        EmitAttackZoneEvent(
            "raw_rek_hit_observed",
            "read_only_rek_hit_packet_with_contemporaneous_opponent_context",
            frame,
            AttackZoneRawHitPayload(
                body,
                sequence,
                clock,
                ContinuousFramePayload(frame),
                opponentContext,
                null),
            clock);
    }

    private object AttackZoneRawHitPayload(
        byte[] body,
        long sequence,
        AttackZoneClock clock,
        object? frame,
        object? opponentContactContext,
        string? failureReason) => new
    {
        raw_hit_sequence = sequence,
        wire_body_bytes = body.Length,
        wire_body_sha256 = Convert.ToHexString(SHA256.HashData(body)).ToLowerInvariant(),
        wire_body_base64 = Convert.ToBase64String(body),
        decoded = new
        {
            world_position_xyz_m = new[]
            {
                ReadAttackZoneFloat32(body, 0),
                ReadAttackZoneFloat32(body, 4),
                ReadAttackZoneFloat32(body, 8),
            },
            world_surface_normal_xyz = new[]
            {
                ReadAttackZoneFloat32(body, 12),
                ReadAttackZoneFloat32(body, 16),
                ReadAttackZoneFloat32(body, 20),
            },
            relative_speed = ReadAttackZoneFloat32(body, 24),
            is_kick_raw_byte = body[28],
        },
        observation_clock = clock,
        contemporaneous_control_frame = frame,
        contemporaneous_opponent_root_bones_and_colliders = opponentContactContext,
        context_capture_failure_reason = failureReason,
        raw_packet_contains_fighter_identity = false,
        raw_packet_contains_move_identity = false,
        hit_ownership = "unknown",
        body_zone = "unknown_requires_downstream_opponent_local_projection",
        association_status = "unknown_at_raw_observation_requires_separate_local_clean_hit_delta_and_isolated_action_interval",
        server_acceptance_observed = false,
        authoritative_execution_observed = false,
    };

    private static object CaptureAttackZoneOpponentContactContext(Robot opponent)
    {
        var root = opponent.RootTransform;
        var bones = new List<object>();
        var boneTransforms = opponent.boneTransforms;
        if (boneTransforms is not null)
        {
            for (var index = 0; index < boneTransforms.Length; index++)
            {
                var bone = boneTransforms[index];
                if (bone is null)
                    continue;
                bones.Add(new
                {
                    index,
                    name = bone.name,
                    world_position_xyz_m = VectorPayload(bone.position),
                    world_rotation_xyzw = QuaternionPayload(bone.rotation),
                });
            }
        }
        var colliders = new List<object>();
        var colliderArray = opponent.GetComponentsInChildren<Collider>(includeInactive: true);
        for (var index = 0; index < colliderArray.Length; index++)
        {
            var collider = colliderArray[index];
            if (collider is null)
                continue;
            var bounds = collider.bounds;
            colliders.Add(new
            {
                index,
                name = collider.name,
                collider_type = collider.GetType().FullName,
                game_object_path = collider.gameObject is null
                    ? null
                    : GameObjectPath(collider.gameObject),
                enabled = collider.enabled,
                is_trigger = collider.isTrigger,
                world_bounds_center_xyz_m = VectorPayload(bounds.center),
                world_bounds_size_xyz_m = VectorPayload(bounds.size),
                world_transform_position_xyz_m = VectorPayload(collider.transform.position),
                world_transform_rotation_xyzw = QuaternionPayload(collider.transform.rotation),
            });
        }
        return new
        {
            root_position_xyz_m = root is null ? null : VectorPayload(root.position),
            root_rotation_xyzw = root is null ? null : QuaternionPayload(root.rotation),
            root_linear_velocity_xyz_m_s = VectorPayload(opponent.RootLinearVelocity),
            root_angular_velocity_xyz_rad_s = VectorPayload(opponent.RootAngularVelocity),
            bones,
            colliders,
            coordinates = "measured_world_space_at_REKApp.FightCoordinator.OnHitReceived_prefix",
        };
    }

    private void AdvanceAttackZoneRecoveryOnly()
    {
        if (!_attackZoneRecoveryOnlyRunning || _attackZoneTarget is null)
        {
            StopContinuousController("attack_zone_recovery_only_state_missing");
            return;
        }
        if (!RequireOwnedContinuousControl(out var controlReason))
        {
            StopContinuousController(controlReason);
            return;
        }

        var fixedSubstep = _continuousControllerFixedSubstep++;
        if (fixedSubstep % AttackZoneTrialContract.FixedSubstepsPerControlTick != 0)
            return;
        _continuousControllerTick =
            fixedSubstep / AttackZoneTrialContract.FixedSubstepsPerControlTick;
        _continuousControllerRoundTick++;

        if (!TryGetPrivateAiContext(
                requireActiveRound: true,
                out var scope,
                out var reason,
                allowOwnedPendingEStop: _continuousControllerEStopAwaitingSend) ||
            !TryCreateTrialRoundIdentity(scope, out var roundIdentity, out reason))
        {
            StopContinuousController($"attack_zone_recovery_scope_lost:{reason}");
            return;
        }
        var roundIdentitySha256 = HashTrialRoundIdentity(roundIdentity);
        if (_continuousControllerRoundIdentity is null ||
            !_continuousControllerRoundIdentity.Equals(RuntimeIdentity.From(scope)) ||
            !string.Equals(
                _continuousControllerRoundIdentitySha256,
                roundIdentitySha256,
                StringComparison.Ordinal))
        {
            StopContinuousController("attack_zone_recovery_session_or_round_identity_changed");
            return;
        }
        var pairing = ReadMeasuredPairing(scope.Coordinator, scope.LocalSlot, scope.OpponentSlot);
        if (!TryValidateContinuousPairing(pairing, out reason) ||
            !TryCaptureContinuousFrame(scope, pairing, out var frame, out reason))
        {
            StopContinuousController($"attack_zone_recovery_frame_unproven:{reason}");
            return;
        }
        if (!ContinuousRuntimeIdentityMatches(frame, _continuousControllerLastFrame) ||
            _continuousControllerInputPointer == IntPtr.Zero ||
            NativePointer(frame.Input) != _continuousControllerInputPointer ||
            !VelocityEquals(frame.Input.VelocityCommand, _continuousControllerVelocity))
        {
            StopContinuousController("attack_zone_recovery_runtime_or_input_identity_changed");
            return;
        }
        if ((frame.Input.hasPendingEStop && !_continuousControllerEStopAwaitingSend) ||
            frame.Input.hasPendingMove ||
            (frame.Input.hasPendingSpecial &&
             (_continuousControllerSpecialAwaitingSend is null ||
              frame.Input.pendingSpecialCommand !=
                  (int)_continuousControllerSpecialAwaitingSend.Value)))
        {
            StopContinuousController("attack_zone_recovery_unowned_pending_command");
            return;
        }

        _continuousControllerLastFrame = frame;
        _continuousControllerLastRoundMetrics = frame.RoundMetrics;
        SetContinuousVelocity(frame.Input, Vector3.zero, "attack_zone_recovery_only_neutral");
        if (!_continuousControllerRunning)
            return;

        var localPlanarSpeed = Math.Sqrt(
            frame.LocalLinearVelocity.x * frame.LocalLinearVelocity.x +
            frame.LocalLinearVelocity.z * frame.LocalLinearVelocity.z);
        var localYawRate = Math.Abs(frame.LocalAngularVelocity.y);
        var localHealthReady =
            !frame.LocalFalling && !frame.LocalFallen && !frame.LocalDampened &&
            !frame.LocalRecoveryArmed && !frame.LocalGetUpPending &&
            !frame.LocalResetting && !frame.LocalMotorShutdown &&
            !frame.InputRecovering && !frame.InputPunching &&
            !frame.ComposerActionPlaying && !frame.ComposerBusy &&
            !frame.Input.hasPendingMove && !frame.Input.hasPendingSpecial &&
            !frame.Input.hasPendingEStop &&
            _continuousControllerSpecialAwaitingSend is null &&
            !_continuousControllerEStopAwaitingSend &&
            _continuousControllerVelocityPurposeAwaitingSend is null &&
            VelocityEquals(frame.Input.VelocityCommand, Vector3.zero) &&
            localPlanarSpeed <= AttackZoneTrialContract.PlanarSpeedLimitMetersPerSecond &&
            localYawRate <= AttackZoneTrialContract.YawRateLimitRadiansPerSecond;
        _attackZoneRecoveryReadyTicks = localHealthReady
            ? _attackZoneRecoveryReadyTicks + 1
            : 0;

        EmitAttackZoneEvent(
            "recovery_state_observed",
            "recovery_only_measured_state",
            frame,
            new
            {
                recovery_only = true,
                attack_requested = false,
                local_health_ready = localHealthReady,
                upright_readiness_consecutive_ticks = _attackZoneRecoveryReadyTicks,
                required_upright_readiness_ticks = AttackZoneTrialContract.RecoveryReadyTicks,
                local_planar_speed_m_s = localPlanarSpeed,
                local_abs_yaw_rate_rad_s = localYawRate,
                recovery_stage = _continuousControllerRecoveryStage,
                recovery_sequence = _continuousControllerRecoverySequence,
            });

        if (_attackZoneRecoveryReadyTicks >= AttackZoneTrialContract.RecoveryReadyTicks)
        {
            EmitAttackZoneEvent(
                "recovery_state_observed",
                "local_upright_readiness_proven_after_censored_trial",
                frame,
                new
                {
                    recovery_only = true,
                    attack_requested = false,
                    upright_readiness_consecutive_ticks = _attackZoneRecoveryReadyTicks,
                    required_upright_readiness_ticks = AttackZoneTrialContract.RecoveryReadyTicks,
                    next_trial_may_revalidate_fresh_scope = true,
                });
            _attackZoneRecoveryOnlyRunning = false;
            _attackZonePhase = "inactive";
            _attackZoneTarget = null;
            _attackZoneSettleTracker = null;
            StopContinuousController("attack_zone_recovery_only_upright_readiness_proven");
            return;
        }

        if (frame.LocalResetting)
        {
            _continuousControllerPhase = "attack_zone_recovery_only_resetting";
            return;
        }
        if (frame.LocalMotorShutdown)
        {
            _continuousControllerPhase = "attack_zone_recovery_only_motor_shutdown_fault";
            DriveContinuousFaultEStopCycle(frame);
            return;
        }
        if (_continuousControllerRecoveryEpisodeActive &&
            IsContinuousFaultEStopStage(_continuousControllerRecoveryStage))
        {
            _continuousControllerPhase = "attack_zone_recovery_only_motor_shutdown_fault";
            DriveContinuousFaultEStopCycle(frame);
            return;
        }
        if (frame.LocalFalling && !frame.LocalFallen)
        {
            _continuousControllerPhase = "attack_zone_recovery_only_await_fallen";
            return;
        }
        if (frame.LocalFallen)
        {
            _continuousControllerPhase = "attack_zone_recovery_only_normal_recovery";
            DriveContinuousRecovery(frame);
            return;
        }
        if (_continuousControllerSpecialAwaitingSend is not null ||
            frame.Input.hasPendingSpecial || frame.LocalGetUpPending ||
            frame.InputRecovering || frame.LocalRecoveryArmed || frame.LocalDampened)
        {
            _continuousControllerPhase = "attack_zone_recovery_only_pending_upright";
            return;
        }
        _continuousControllerPhase = "attack_zone_recovery_only_upright_settle";
    }

    private void StopAttackZoneTrialInternal(
        string reason,
        string finalEvent,
        bool continueLocalRecovery = false)
    {
        if (!_attackZoneTrialRunning)
            return;
        var frame = _continuousControllerLastFrame;
        var finalClock = CaptureAttackZoneClock();
        _attackZoneLastOutcome = reason;
        EmitAttackZoneEvent(
            finalEvent,
            reason,
            frame,
            new
            {
                target_request_clock = _attackZoneTargetRequestedClock,
                action_request_clock = _attackZoneActionRequestedClock,
                action_start_clock = _attackZoneActionStartedClock,
                action_completion_clock = _attackZoneActionCompletedClock,
                final_clock = finalClock,
                target_acquired = _attackZoneSettleTracker?.Acquired ?? false,
                attack_request_issued = _attackZoneAttackIssued,
                client_move_request_method_returned = _continuousControllerMoveRequestObserved,
                local_motion_start_observed = _continuousControllerActionStartedObserved,
                local_motion_completion_observed = _attackZoneCompletionObserved,
                action_sample_count = _attackZoneActionSampleSequence,
                last_action_sample_clock = _attackZoneLastActionSampleClock,
                request_edge_geometry = _attackZoneRequestEdgeGeometry,
                settle_digest = _attackZoneSettleTracker?.AcquiredDigest is null
                    ? null
                    : AttackZoneSettleDigestPayload(_attackZoneSettleTracker.AcquiredDigest),
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
            },
            finalClock);

        TryCancelExactOwnedContinuousPendingRequests();
        TryNeutralOwnedContinuousController();
        ClearContinuousActionState();
        ClearContinuousPendingRequestState();
        _continuousControllerVelocity = Vector3.zero;
        _attackZoneTrialRunning = false;
        _attackZoneUnassociatedRawHits.Clear();
        if (continueLocalRecovery)
        {
            _attackZoneRecoveryOnlyRunning = true;
            _attackZoneRecoveryReadyTicks = 0;
            _attackZonePhase = "recovery_only";
            _continuousControllerPhase = "attack_zone_recovery_only";
            EmitAttackZoneEvent(
                "recovery_state_observed",
                "terminal_trial_persisted_before_local_recovery_continuation",
                frame,
                new
                {
                    terminal_trial_event = finalEvent,
                    terminal_trial_reason = reason,
                    recovery_only = true,
                    attack_requests_allowed = false,
                    required_upright_readiness_ticks =
                        AttackZoneTrialContract.RecoveryReadyTicks,
                });
            return;
        }
        _attackZoneRecoveryOnlyRunning = false;
        _attackZonePhase = "inactive";
        _attackZoneTarget = null;
        _attackZoneSettleTracker = null;
        StopContinuousController($"attack_zone:{reason}");
    }

    private void EmitAttackZoneEvent(
        string eventName,
        string reason,
        ContinuousFrame? frame,
        object? detail,
        AttackZoneClock? eventClock = null)
    {
        try
        {
            var clock = eventClock ?? CaptureAttackZoneClock();
            var target = _attackZoneTarget?.Request;
            var payload = new
            {
                @event = eventName,
                protocol = "rek.ui_bridge.v1",
                attack_zone_schema = AttackZoneTrialContract.Schema,
                attack_zone_protocol_sha256 = _attackZoneContractSha256,
                continuous_controller_sha256 = _continuousControllerSha256,
                authority_scope = AttackZoneTrialContract.AuthorityScope,
                authority_caveat = AttackZoneTrialContract.AuthorityCaveat,
                isolated_spark_proof = AttackZoneTrialContract.RequiredIsolationProof,
                schedule_schema = target?.ScheduleSchema,
                schedule_sha256 = target?.ScheduleSha256,
                randomization_seed_hex = target?.RandomizationSeedHex,
                schedule_ordinal = target?.ScheduleOrdinal,
                independent_run_id = target?.IndependentRunId,
                independent_run_ordinal = target?.IndependentRunOrdinal,
                session_identity_sha256 = target?.SessionIdentitySha256,
                round_identity_sha256 = target?.RoundIdentitySha256,
                trial_id = target?.TrialId,
                action_sequence = target?.ActionSequence,
                move_index = target?.MoveIndex,
                serialized_asset_sha256 = target?.SerializedAssetSha256,
                requested_distance_bin = target?.DistanceBin,
                requested_bearing_bin = target?.BearingBin,
                controller_phase = _attackZonePhase,
                controller_reason = reason,
                measured_state = frame is null ? null : ContinuousFramePayload(frame),
                detail,
                clocks = clock,
                stopwatch_timestamp_ticks = clock.StopwatchTimestampTicks,
                stopwatch_frequency_hz = clock.StopwatchFrequencyHz,
                utc = clock.Utc,
                unity_frame = clock.UnityFrame,
                unity_time = clock.UnityTime,
                unity_fixed_time = clock.UnityFixedTime,
                client_control_tick = clock.ControlTick,
                client_fixed_substep = clock.ClientFixedSubstep,
                fixed_substeps_per_control_tick =
                    AttackZoneTrialContract.FixedSubstepsPerControlTick,
                global_input_used = false,
                client_request_observation_only = true,
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
            };
            _pipe?.Send(_leaseConnectionId, payload);
            Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        }
        catch (Exception exception)
        {
            Log.LogError($"Attack-zone event emission failed: {exception.GetType().Name}");
            TryCancelExactOwnedContinuousPendingRequests();
            TryNeutralOwnedContinuousController();
            _attackZoneTrialRunning = false;
            _attackZoneRecoveryOnlyRunning = false;
            _continuousControllerRunning = false;
            _continuousControllerAuthorizedWhileBackground = false;
            _attackZonePhase = "inactive";
            _attackZoneLastOutcome = $"attack_zone_event_emit_failed:{exception.GetType().Name}";
        }
    }

    private AttackZoneClock CaptureAttackZoneClock() => new(
        System.Diagnostics.Stopwatch.GetTimestamp(),
        System.Diagnostics.Stopwatch.Frequency,
        DateTimeOffset.UtcNow.ToString("O"),
        Time.frameCount,
        Time.timeAsDouble,
        Time.fixedTimeAsDouble,
        _continuousControllerTick,
        _continuousControllerFixedSubstep);

    private object CaptureAttackZoneAvailability()
    {
        try
        {
            if (_attackZoneTrialRunning || _attackZoneRecoveryOnlyRunning ||
                _continuousControllerRunning || _singleMotionTrialRunning || _scheduleRunning)
            {
                return new
                {
                    available = false,
                    reason = _attackZoneRecoveryOnlyRunning
                        ? "attack_zone_recovery_only_not_complete"
                        : "another_control_mode_already_running",
                    session_identity_sha256 = (string?)null,
                    round_identity_sha256 = (string?)null,
                };
            }
            if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var reason) ||
                !TryCreateTrialRoundIdentity(scope, out var roundIdentity, out reason))
            {
                return new
                {
                    available = false,
                    reason,
                    session_identity_sha256 = (string?)null,
                    round_identity_sha256 = (string?)null,
                };
            }
            var pairing = ReadMeasuredPairing(
                scope.Coordinator,
                scope.LocalSlot,
                scope.OpponentSlot);
            if (!TryValidateContinuousPairing(pairing, out reason) ||
                !TryCaptureContinuousFrame(scope, pairing, out var frame, out reason) ||
                !ContinuousLocalActionReady(frame) || AttackZoneOpponentUnhealthy(frame))
            {
                return new
                {
                    available = false,
                    reason = string.IsNullOrEmpty(reason)
                        ? "attack_zone_fighter_readiness_not_proven"
                        : reason,
                    session_identity_sha256 = (string?)null,
                    round_identity_sha256 = (string?)null,
                };
            }
            return new
            {
                available = true,
                reason = "exact_private_bot1_t800_runtime_scope_available",
                session_identity_sha256 = HashTrialSessionIdentity(roundIdentity.SessionIdentity),
                round_identity_sha256 = HashTrialRoundIdentity(roundIdentity),
            };
        }
        catch (Exception exception)
        {
            return new
            {
                available = false,
                reason = $"attack_zone_availability_probe_failed:{exception.GetType().Name}",
                session_identity_sha256 = (string?)null,
                round_identity_sha256 = (string?)null,
            };
        }
    }

    private static string HashTrialSessionIdentity(TrialSessionIdentity identity)
    {
        var canonical = JsonSerializer.Serialize(new
        {
            coordinator_pointer = identity.CoordinatorPointer.ToInt64().ToString("x16"),
            network_pointer = identity.NetworkPointer.ToInt64().ToString("x16"),
            context_pointer = identity.ContextPointer.ToInt64().ToString("x16"),
            identity.LocalSlot,
            identity.OpponentSlot,
            arena_id_sha256 = HashText(identity.ArenaId),
            endpoint_sha256 = HashText(identity.Endpoint),
        }, BridgeJson.Options);
        return HashText(canonical);
    }

    private static bool TryVerifyAttackZoneRecorder(
        out string? observedVersion,
        out string? observedSha256,
        out string reason)
    {
        observedVersion = null;
        observedSha256 = null;
        try
        {
            var assemblies = AppDomain.CurrentDomain.GetAssemblies()
                .Where(value => string.Equals(
                    value.GetName().Name,
                    "RekEvidenceRecorder",
                    StringComparison.Ordinal))
                .ToArray();
            if (assemblies.Length != 1)
            {
                reason = $"attack_zone_expected_exactly_one_loaded_recorder_assembly:observed_{assemblies.Length}";
                return false;
            }
            var assembly = assemblies[0];
            observedSha256 = HashFile(assembly.Location);
            observedVersion = assembly.GetType("RekEvidenceRecorder.Plugin")?
                .GetField(
                    "PluginVersion",
                    BindingFlags.Public | BindingFlags.Static)?
                .GetRawConstantValue()?.ToString();
            if (!string.Equals(
                    observedVersion,
                    AttackZoneTrialContract.ExpectedRecorderVersion,
                    StringComparison.Ordinal) ||
                !string.Equals(
                    observedSha256,
                    AttackZoneTrialContract.ExpectedRecorderPluginSha256,
                    StringComparison.Ordinal))
            {
                reason = "attack_zone_loaded_recorder_version_or_sha256_mismatch";
                return false;
            }
            reason = string.Empty;
            return true;
        }
        catch (Exception exception)
        {
            reason = $"attack_zone_loaded_recorder_verification_failed:{exception.GetType().Name}";
            return false;
        }
    }

    private object AttackZoneActionLifecyclePayload(string stage) => new
    {
        lifecycle_stage = stage,
        target = _attackZoneTarget is null
            ? null
            : AttackZoneTargetPayload(_attackZoneTarget.Request),
        target_request_clock = _attackZoneTargetRequestedClock,
        action_request_clock = _attackZoneActionRequestedClock,
        action_start_clock = _attackZoneActionStartedClock,
        action_completion_clock = _attackZoneActionCompletedClock,
        request_edge_geometry = _attackZoneRequestEdgeGeometry,
        move_profile = _attackZoneTarget is null
            ? null
            : ContinuousAttackProfilePayload(_attackZoneTarget.Attack),
        send_method = "RobotInputController.SendMoveEvent",
        server_acceptance_observed = false,
        authoritative_execution_observed = false,
    };

    private static object AttackZoneSettleUpdatePayload(AttackZoneSettleUpdate update) => new
    {
        update.Acquired,
        update.ConsecutiveTicks,
        update.StreakReset,
        update.Reason,
        digest = update.Digest is null ? null : AttackZoneSettleDigestPayload(update.Digest),
        current_evaluation = update.CurrentEvidence?.Evaluation,
        current_source_clock = update.CurrentEvidence?.Sample.Clock,
    };

    private static object AttackZoneSettleDigestPayload(AttackZoneSettleDigest digest) => new
    {
        digest.SampleCount,
        digest.FirstClock,
        digest.LastClock,
        distance_m = new { min = digest.MinimumDistanceMeters, max = digest.MaximumDistanceMeters },
        local_bearing_deg = new
        {
            min = digest.MinimumLocalBearingDegrees,
            max = digest.MaximumLocalBearingDegrees,
        },
        bearing_error_deg = new
        {
            min = digest.MinimumBearingErrorDegrees,
            max = digest.MaximumBearingErrorDegrees,
        },
        local_planar_speed_m_s = new
        {
            min = digest.MinimumLocalPlanarSpeedMetersPerSecond,
            max = digest.MaximumLocalPlanarSpeedMetersPerSecond,
        },
        local_yaw_rate_rad_s = new
        {
            min = digest.MinimumLocalYawRateRadiansPerSecond,
            max = digest.MaximumLocalYawRateRadiansPerSecond,
        },
        opponent_planar_speed_m_s = new
        {
            min = digest.MinimumOpponentPlanarSpeedMetersPerSecond,
            max = digest.MaximumOpponentPlanarSpeedMetersPerSecond,
        },
        opponent_yaw_rate_rad_s = new
        {
            min = digest.MinimumOpponentYawRateRadiansPerSecond,
            max = digest.MaximumOpponentYawRateRadiansPerSecond,
        },
        radial_closing_speed_m_s = new
        {
            min = digest.MinimumRadialClosingSpeedMetersPerSecond,
            max = digest.MaximumRadialClosingSpeedMetersPerSecond,
        },
        tangential_speed_m_s = new
        {
            min = digest.MinimumTangentialSpeedMetersPerSecond,
            max = digest.MaximumTangentialSpeedMetersPerSecond,
        },
        predicates = new
        {
            digest.AllClocksValid,
            digest.AllRootsFinite,
            digest.AllGeometryValid,
            digest.AllAnimationValid,
            digest.AllNeutralRequestMethodReturned,
            digest.AllVelocityCommandsExactNeutral,
            digest.AllLocalActionReady,
            digest.AllNoPendingRequests,
            digest.AllLocalHealthy,
            digest.AllOpponentHealthy,
            digest.AllDistanceCentralPass,
            digest.AllBearingInBinPass,
            digest.AllBearingErrorPass,
            digest.AllLocalMotionPass,
            digest.AllOpponentStationary,
        },
        digest.OpponentMotionStratum,
        digest.OpponentFacingStratum,
    };

    private static object AttackZoneTargetPayload(AttackZoneTrialTarget target)
    {
        using var document = JsonDocument.Parse(AttackZoneTrialContract.SerializeTarget(target));
        return document.RootElement.Clone();
    }

    private static AttackZoneGeometry BuildAttackZoneGeometry(ContinuousFrame frame) => new(
        true,
        frame.Geometry.DistanceMeters,
        frame.Geometry.LocalBearingToOpponentDegrees,
        frame.Geometry.OpponentBearingToLocalDegrees,
        frame.Geometry.LocalToOpponentUnitX,
        frame.Geometry.LocalToOpponentUnitZ);

    private static float[] VectorPayload(Vector3 value) =>
        new[] { value.x, value.y, value.z };

    private static float[] QuaternionPayload(Quaternion value) =>
        new[] { value.x, value.y, value.z, value.w };

    private static unsafe byte[] CopyAttackZoneReaderBody(
        FastBufferReader reader,
        int expectedLength)
    {
        var remaining = reader.Length - reader.Position;
        if (remaining != expectedLength)
        {
            throw new InvalidDataException(
                $"REK_Hit body length {remaining} does not match audited length {expectedLength}.");
        }
        var body = new byte[expectedLength];
        Marshal.Copy((IntPtr)reader.GetUnsafePtrAtCurrentPosition(), body, 0, body.Length);
        return body;
    }

    private static float ReadAttackZoneFloat32(byte[] body, int offset)
    {
        var bits = BinaryPrimitives.ReadInt32LittleEndian(body.AsSpan(offset, sizeof(float)));
        return BitConverter.Int32BitsToSingle(bits);
    }

    private sealed record AttackZoneRawHitReference(long Sequence, AttackZoneClock Clock);
}

[HarmonyPatch(typeof(FightCoordinator), "OnHitReceived")]
internal static class AttackZoneHitObservationPatch
{
    [HarmonyPrefix]
    [HarmonyPriority(Priority.First)]
    private static void Prefix(FastBufferReader __1)
    {
        try
        {
            Plugin.Instance?.ObserveAttackZoneRawHit(__1);
        }
        catch (Exception exception)
        {
            Plugin.Instance?.StopAttackZoneAfterRawHitFailure(exception);
        }
    }
}

public sealed partial class Plugin
{
    internal void StopAttackZoneAfterRawHitFailure(Exception exception)
    {
        if (_attackZoneTrialRunning)
        {
            StopAttackZoneTrialInternal(
                $"raw_hit_observation_failed:{exception.GetType().Name}",
                "trial_censored");
        }
    }
}
