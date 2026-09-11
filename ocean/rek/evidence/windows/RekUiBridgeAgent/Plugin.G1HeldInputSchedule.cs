using System.Diagnostics;
using System.Text.Json;
using REKApp;
using UnityEngine;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private bool _g1HeldScheduleRunning;
    private bool _g1HeldScheduleAuthorizedWhileBackground;
    private string? _g1HeldScheduleRunId;
    private string? _g1HeldScheduleFreshRoundRequestId;
    private string? _g1HeldScheduleRoundIdentitySha256;
    private RuntimeIdentity? _g1HeldScheduleIdentity;
    private RobotInputController? _g1HeldScheduleInput;
    private IntPtr _g1HeldScheduleInputPointer;
    private int _g1HeldScheduleTick;
    private int _g1HeldScheduleFixedSubstep;
    private int _g1HeldEventSequence;
    private G1HeldMask _g1DesiredHeldMask;
    private G1HeldMask _g1EffectiveHeldMask;
    private float _g1RawYawTarget;
    private G1KeyboardYawState _g1KeyboardYawState;
    private Vector3 _g1HeldScheduleVelocity = Vector3.zero;
    private bool _g1VelocityInvocationObserved;
    private int _g1VelocityInvocationTick;
    private int _g1VelocityInvocationSubstep;
    private bool _g1FinalNeutralSendReturned;
    private float _g1HeldRoundDurationSeconds;
    private float _g1HeldInitialTimeRemainingSeconds;
    private G1KickAttemptRuntime? _g1KickAttempt;
    private readonly int[] _g1HeldConditionObservedTicks =
        new int[G1HeldInputScheduleContract.HeldConditions.Length];
    private readonly bool[] _g1KickEdgeObserved =
        new bool[G1HeldInputScheduleContract.KickProbes.Length];
    private readonly bool[] _g1KickTerminalOutcomeObserved =
        new bool[G1HeldInputScheduleContract.KickProbes.Length];
    private readonly bool[] _g1YawPreemptionObserved =
        new bool[G1HeldInputScheduleContract.KickProbes.Length];
    private readonly bool[] _g1TranslationReleaseObserved =
        new bool[G1HeldInputScheduleContract.KickProbes.Length];
    private readonly G1KickMeasurementRuntime?[] _g1KickMeasurements =
        new G1KickMeasurementRuntime?[G1HeldInputScheduleContract.KickProbes.Length];

    private CommandResult StartG1HeldInputSchedule()
    {
        if (!RequireBackgroundControl(out var isolationReason))
            return CommandResult.Rejected(isolationReason);
        if (!TryVerifyExplicitIsolatedSession(out var isolationProof) ||
            !string.Equals(
                isolationProof,
                G1HeldInputScheduleContract.RequiredIsolationProof,
                StringComparison.Ordinal))
        {
            return CommandResult.Rejected("exact_isolated_spark_marker_not_proven");
        }
        if (_g1HeldScheduleRunning)
            return CommandResult.Rejected("g1_held_input_schedule_already_running");
        if (_scheduleRunning || _singleMotionTrialRunning || _continuousControllerRunning ||
            _attackZoneTrialRunning || _attackZoneRecoveryOnlyRunning)
        {
            return CommandResult.Rejected("another_control_mode_already_running");
        }
        if (!SameFloatBits(
                Time.fixedDeltaTime,
                G1HeldInputScheduleContract.ExpectedFixedDeltaTime))
        {
            return CommandResult.Rejected($"unexpected_fixed_delta_time:{Time.fixedDeltaTime:R}");
        }
        if (!string.Equals(
                G1HeldInputScheduleContract.ComputeCanonicalSha256(),
                G1HeldInputScheduleContract.ExpectedSha256,
                StringComparison.Ordinal))
        {
            return CommandResult.Rejected("g1_held_input_schedule_contract_hash_mismatch");
        }
        if (!_sendBoundaryPatchesVerified)
            return CommandResult.Rejected("g1_held_input_send_boundary_patches_not_verified");
        if (!TryGetPrivateAiContext(requireActiveRound: true, out var scope, out var reason))
            return CommandResult.Rejected(reason);

        var arm = _freshRoundArm;
        if (arm is null)
            return CommandResult.Rejected("g1_held_fresh_round_not_armed");
        _freshRoundArm = null;
        if (arm.ConnectionId != _leaseConnectionId ||
            arm.ConnectionId != (_pipe?.CurrentConnectionId ?? 0))
        {
            return CommandResult.Rejected("g1_held_fresh_round_arm_connection_changed");
        }
        if (arm.InvalidReason is not null)
            return CommandResult.Rejected($"g1_held_fresh_round_arm_invalid:{arm.InvalidReason}");
        var sessionIdentity = TrialSessionIdentity.From(scope);
        if (!sessionIdentity.IsComplete || !sessionIdentity.Equals(arm.SessionIdentity))
            return CommandResult.Rejected("g1_held_fresh_round_session_scope_changed");
        if (!TryCreateTrialRoundIdentity(scope, out var roundIdentity, out reason))
            return CommandResult.Rejected($"g1_held_fresh_round_identity_rejected:{reason}");
        if (roundIdentity.RoundPointer == arm.PriorRoundPointer ||
            (roundIdentity.FightEpoch == arm.PriorFightEpoch &&
             roundIdentity.RoundNumber == arm.PriorRoundNumber))
        {
            return CommandResult.Rejected("g1_held_fresh_round_identity_not_unique");
        }
        var roundIdentitySha256 = HashTrialRoundIdentity(roundIdentity);
        if (_consumedTrialRounds.Contains(roundIdentitySha256))
            return CommandResult.Rejected("g1_held_round_already_consumed");
        if (_consumedTrialRounds.Count >= MaxConsumedTrialRounds)
            return CommandResult.Rejected("controlled_round_history_capacity_reached");

        var measuredPairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        var measuredPairingPayload = MeasuredPairingPayload(measuredPairing);
        if (!measuredPairing.Validation.ExactG1VersusG1)
        {
            return CommandResult.Rejected(
                $"required_exact_g1_vs_g1_pairing_not_proven:{measuredPairing.Validation.Reason}",
                measuredPairingPayload);
        }
        if (scope.Input is null || !scope.Input.IsActive || !scope.Input.networkInitialized)
            return CommandResult.Rejected("g1_local_input_controller_not_active_and_network_initialized");
        if (scope.Input.hasPendingMove || scope.Input.hasPendingSpecial || scope.Input.hasPendingEStop)
            return CommandResult.Rejected("g1_local_input_controller_has_pending_command");
        if (scope.Input.IsPunching || scope.Input.IsRecovering)
            return CommandResult.Rejected("g1_local_input_controller_action_not_settled");
        var robotConfig = scope.Input.robotConfig;
        if (robotConfig is null)
            return CommandResult.Rejected("g1_robot_config_unavailable");
        if (!SameFloatBits(robotConfig.yawSpeed, G1HeldInputScheduleContract.ExpectedYawSpeed) ||
            !SameFloatBits(
                robotConfig.keyboardYawRampTime,
                G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds) ||
            !SameFloatBits(
                robotConfig.transitionSettlePlanarSpeed,
                G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed) ||
            !SameFloatBits(
                robotConfig.transitionSettleYawRate,
                G1HeldInputScheduleContract.ExpectedTransitionSettleYawRate))
        {
            return CommandResult.Rejected(
                "g1_input_runtime_config_mismatch");
        }
        if (!TryValidateG1KickAssetBindings(scope.Input, out reason))
            return CommandResult.Rejected(reason);
        var round = scope.Round;
        if (round is null || !round.IsActive || round.IsRedo || round.KnockoutOccurred)
            return CommandResult.Rejected("g1_fresh_round_runtime_state_not_active");
        var cleanHits = round.CleanHits;
        var falls = round.Falls;
        if (cleanHits is null || cleanHits.Length != 2 ||
            falls is null || falls.Length != 2 ||
            cleanHits[0] != 0 || cleanHits[1] != 0 ||
            falls[0] != 0 || falls[1] != 0)
        {
            return CommandResult.Rejected("g1_fresh_round_score_state_not_clean");
        }
        if (!G1HeldInputScheduleContract.HasRoundCapacity(
                round.RoundDuration,
                round.TimeRemaining))
        {
            return CommandResult.Rejected(
                $"g1_fresh_round_capacity_insufficient_or_unknown:" +
                $"duration={round.RoundDuration:R}:remaining={round.TimeRemaining:R}:" +
                $"required={G1HeldInputScheduleContract.RequiredRoundCapacitySeconds:R}");
        }
        if (!SetVelocityExact(scope.Input, Vector3.zero))
            return CommandResult.Rejected("g1_initial_neutral_velocity_readback_mismatch");
        if (!_consumedTrialRounds.Add(roundIdentitySha256))
            return CommandResult.Rejected("g1_held_round_already_consumed");

        _g1HeldScheduleRunId = Guid.NewGuid().ToString("N");
        _g1HeldScheduleFreshRoundRequestId = arm.RequestId;
        _g1HeldScheduleRoundIdentitySha256 = roundIdentitySha256;
        _g1HeldScheduleIdentity = RuntimeIdentity.From(scope);
        _g1HeldScheduleInput = scope.Input;
        _g1HeldScheduleInputPointer = NativePointer(scope.Input);
        _g1HeldScheduleTick = 0;
        _g1HeldScheduleFixedSubstep = 0;
        _g1HeldEventSequence = 0;
        _g1DesiredHeldMask = G1HeldMask.None;
        _g1EffectiveHeldMask = G1HeldMask.None;
        _g1RawYawTarget = 0f;
        _g1KeyboardYawState = new G1KeyboardYawState(0f, 0f);
        _g1HeldScheduleVelocity = Vector3.zero;
        _g1VelocityInvocationObserved = false;
        _g1VelocityInvocationTick = 0;
        _g1VelocityInvocationSubstep = 0;
        _g1FinalNeutralSendReturned = false;
        _g1HeldRoundDurationSeconds = round.RoundDuration;
        _g1HeldInitialTimeRemainingSeconds = round.TimeRemaining;
        _g1KickAttempt = null;
        Array.Clear(_g1HeldConditionObservedTicks);
        Array.Clear(_g1KickEdgeObserved);
        Array.Clear(_g1KickTerminalOutcomeObserved);
        Array.Clear(_g1YawPreemptionObserved);
        Array.Clear(_g1TranslationReleaseObserved);
        Array.Clear(_g1KickMeasurements);
        _g1HeldScheduleAuthorizedWhileBackground = true;
        _g1HeldScheduleRunning = true;

        return CommandResult.AppliedResult(
            "g1_held_input_schedule_started",
            measuredPairingPayload);
    }

    private CommandResult StopG1HeldInputSchedule()
    {
        if (!_g1HeldScheduleRunning)
            return CommandResult.Rejected("g1_held_input_schedule_not_running");
        StopG1HeldSchedule("requested");
        return CommandResult.AppliedResult("g1_held_input_schedule_stopped");
    }

    private void AdvanceG1HeldInputSchedule()
    {
        if (!_g1HeldScheduleRunning)
            return;
        if (!RequireOwnedG1HeldScheduleControl(out var controlReason))
        {
            StopG1HeldSchedule(controlReason);
            return;
        }
        if (!TryGetExactG1HeldScope(out var scope, out var scopeReason))
        {
            StopG1HeldSchedule($"g1_held_scope_lost:{scopeReason}");
            return;
        }
        var input = scope.Input!;
        _g1HeldScheduleTick =
            _g1HeldScheduleFixedSubstep /
            G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
        if (_g1KickAttempt is not null && !_g1KickAttempt.Terminal)
            _g1KickAttempt.DispatchTracker =
                _g1KickAttempt.DispatchTracker.OnFixedSubstep();
        ObserveG1KickMeasurements(input);
        if (!_g1HeldScheduleRunning)
            return;

        var requiredFixedSubsteps =
            G1HeldInputScheduleContract.DurationScheduleTicks *
            G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
        if (_g1HeldScheduleFixedSubstep >= requiredFixedSubsteps - 1)
        {
            if (!_g1FinalNeutralSendReturned)
                return;
            var finalCoverage = BuildG1HeldCoverage();
            StopG1HeldSchedule(finalCoverage.Complete ? "complete" : "coverage_incomplete");
            return;
        }

        var atScheduleBoundary =
            _g1HeldScheduleFixedSubstep %
            G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick == 0;
        if (!atScheduleBoundary)
        {
            _g1HeldScheduleFixedSubstep++;
            return;
        }

        if (input.hasPendingSpecial || input.hasPendingEStop)
        {
            StopG1HeldSchedule("g1_held_unowned_special_or_estop_at_50hz_boundary");
            return;
        }
        if (input.hasPendingMove &&
            (_g1KickAttempt is null || _g1KickAttempt.Terminal ||
             input.pendingMoveIndex != _g1KickAttempt.MoveIndex))
        {
            StopG1HeldSchedule("g1_held_unowned_pending_move_at_50hz_boundary");
            return;
        }

        var frame = G1HeldInputScheduleContract.FrameAtTick(_g1HeldScheduleTick);
        if (!G1HeldInputScheduleContract.IsValidHeldMask(frame.DesiredHeldMask) ||
            !G1HeldInputScheduleContract.IsValidHeldMask(frame.EffectiveHeldMask))
        {
            StopG1HeldSchedule("g1_held_contract_produced_invalid_mask");
            return;
        }
        if (frame.KickProbe is { } startingProbe &&
            _g1HeldScheduleTick == startingProbe.StartTick &&
            !TryRequirePreviousG1ProbeSettled(startingProbe.Ordinal, input, out var settleReason))
        {
            StopG1HeldSchedule($"g1_prior_probe_not_settled:{settleReason}");
            return;
        }

        var previousDesiredHeldMask = _g1DesiredHeldMask;
        var previousEffectiveHeldMask = _g1EffectiveHeldMask;
        var previousRawYawTarget = _g1RawYawTarget;
        _g1DesiredHeldMask = frame.DesiredHeldMask;
        _g1EffectiveHeldMask = frame.EffectiveHeldMask;
        _g1RawYawTarget = frame.RawYaw;
        var yawResetAtKickEdge = frame.KickEdge && frame.YawPreempted;
        if (yawResetAtKickEdge)
        {
            _g1KeyboardYawState = new G1KeyboardYawState(0f, 0f);
            _g1HeldScheduleVelocity.z = 0f;
        }
        _g1HeldScheduleVelocity.x = frame.Forward;
        _g1HeldScheduleVelocity.y = frame.Strafe;
        var velocityReadbackExact = SetVelocityExact(input, _g1HeldScheduleVelocity);
        if (!velocityReadbackExact)
        {
            StopG1HeldSchedule("g1_held_velocity_readback_mismatch");
            return;
        }

        if (frame.HeldConditionOrdinal is int conditionOrdinal)
            _g1HeldConditionObservedTicks[conditionOrdinal]++;

        var localLifecycle = CaptureG1LocalLifecycle(input);
        EmitG1HeldEvent("g1_held_schedule_tick", new
        {
            phase = frame.Phase,
            desired_held_mask = (byte)frame.DesiredHeldMask,
            desired_held = G1HeldInputScheduleContract.HeldNames(frame.DesiredHeldMask),
            effective_held_mask = (byte)frame.EffectiveHeldMask,
            effective_held = G1HeldInputScheduleContract.HeldNames(frame.EffectiveHeldMask),
            desired_pressed_mask = (byte)(frame.DesiredHeldMask & ~previousDesiredHeldMask),
            desired_released_mask = (byte)(previousDesiredHeldMask & ~frame.DesiredHeldMask),
            effective_pressed_mask = (byte)(frame.EffectiveHeldMask & ~previousEffectiveHeldMask),
            effective_released_mask = (byte)(previousEffectiveHeldMask & ~frame.EffectiveHeldMask),
            desired_raw_controller_target_xyz = new[]
            {
                (float)frame.Forward,
                (float)frame.Strafe,
                (float)frame.RawYaw,
            },
            previous_raw_yaw_target = previousRawYawTarget,
            effective_controller_vector_xyz = new[]
            {
                _g1HeldScheduleVelocity.x,
                _g1HeldScheduleVelocity.y,
                _g1HeldScheduleVelocity.z,
            },
            yaw_ramp = _g1KeyboardYawState.Ramp,
            yaw_sign = _g1KeyboardYawState.Sign,
            yaw_update_phase = yawResetAtKickEdge
                ? "fixed_boundary_kick_preemption_reset"
                : "next_rendered_late_update",
            keyboard_yaw_ramp_provenance =
                G1HeldInputScheduleContract.KeyboardYawRampProvenance,
            velocity_property_write_returned = true,
            velocity_readback_exact = velocityReadbackExact,
            held_condition_ordinal = frame.HeldConditionOrdinal,
            kick_probe_ordinal = frame.KickProbe?.Ordinal,
            kick_edge = frame.KickEdge,
            yaw_preempted = frame.YawPreempted,
            translation_released = frame.TranslationReleased,
            local_lifecycle = G1LocalLifecyclePayload(localLifecycle),
        });

        if (frame.KickProbe is { Kind: G1KickProbeKind.TranslationHeld } releaseProbe &&
            releaseProbe.TranslationReleaseTick == _g1HeldScheduleTick)
        {
            RecordG1TranslationRelease(input, releaseProbe);
        }

        if (frame.KickEdge)
            ExecuteG1KickEdge(input, frame);
        if (!_g1HeldScheduleRunning)
            return;
        _g1HeldScheduleFixedSubstep++;
    }

    private void ExecuteG1KickEdge(
        RobotInputController input,
        G1HeldScheduleFrame frame)
    {
        var probe = frame.KickProbe ??
            throw new InvalidOperationException("g1_kick_edge_has_no_probe");
        if (_g1KickAttempt is not null && !_g1KickAttempt.Terminal)
        {
            StopG1HeldSchedule("g1_previous_kick_attempt_not_terminal");
            return;
        }
        if (input.hasPendingMove || input.hasPendingSpecial || input.hasPendingEStop)
        {
            StopG1HeldSchedule("g1_kick_edge_started_with_pending_command");
            return;
        }
        if (!TryRequirePreviousG1ProbeSettled(probe.Ordinal, input, out var settleReason))
        {
            StopG1HeldSchedule($"g1_kick_edge_prior_probe_not_settled:{settleReason}");
            return;
        }
        var edgeLifecycle = CaptureG1LocalLifecycle(input);
        if (edgeLifecycle.ActionBusy || edgeLifecycle.RunnerIsRecovering == true)
        {
            StopG1HeldSchedule("g1_kick_edge_local_action_not_settled");
            return;
        }

        if (probe.Kind == G1KickProbeKind.YawPreempted)
        {
            _g1YawPreemptionObserved[probe.Ordinal] = true;
            EmitG1HeldEvent("g1_yaw_kick_preemption", new
            {
                probe_ordinal = probe.Ordinal,
                probe_label = probe.Label,
                probe_kind = G1ProbeKindName(probe.Kind),
                move_index = probe.MoveIndex,
                desired_held_mask = (byte)frame.DesiredHeldMask,
                desired_held = G1HeldInputScheduleContract.HeldNames(frame.DesiredHeldMask),
                effective_held_mask = (byte)frame.EffectiveHeldMask,
                effective_held = G1HeldInputScheduleContract.HeldNames(frame.EffectiveHeldMask),
                raw_yaw_before_edge = probe.RawYawBeforeEdge,
                raw_yaw_at_edge = frame.RawYaw,
                effective_yaw_at_edge = _g1HeldScheduleVelocity.z,
                yaw_neutralized_before_execute_move =
                    frame.RawYaw == 0 && _g1HeldScheduleVelocity.z == 0f &&
                    _g1KeyboardYawState == new G1KeyboardYawState(0f, 0f),
                velocity_property_write_returned = true,
                velocity_readback_exact = VelocityEquals(input.VelocityCommand, Vector3.zero),
                preemption_persists_until_tick = probe.StopTick - 1,
            });
        }

        var beforeCallLifecycle = edgeLifecycle;
        var requestedAsset = G1HeldInputScheduleContract.AssetForMove(probe.MoveIndex);
        var requestedClip = input.robotConfig.moves[probe.MoveIndex];
        var attempt = new G1KickAttemptRuntime(
            probe.Ordinal,
            probe.Kind,
            probe.Label,
            probe.MoveIndex,
            _g1HeldScheduleTick,
            _g1HeldEventSequence,
            _g1HeldScheduleFixedSubstep);
        _g1KickAttempt = attempt;
        var measurement = new G1KickMeasurementRuntime(
            probe.Ordinal,
            probe.Kind,
            probe.Label,
            probe.MoveIndex,
            probe.EdgeTick,
            probe.TranslationReleaseTick,
            probe.StopTick,
            probe.DesiredHeldMask,
            beforeCallLifecycle,
            requestedAsset,
            requestedClip is null ? IntPtr.Zero : NativePointer(requestedClip));
        _g1KickMeasurements[probe.Ordinal] = measurement;
        var pendingBefore = input.hasPendingMove;
        var pendingIndexBefore = input.pendingMoveIndex;
        bool executeMoveReturned;
        try
        {
            executeMoveReturned = input.ExecuteMoveByIndex(probe.MoveIndex);
        }
        catch (Exception exception)
        {
            EmitG1HeldEvent("g1_kick_request_lifecycle", new
            {
                probe_ordinal = probe.Ordinal,
                probe_label = probe.Label,
                probe_kind = G1ProbeKindName(probe.Kind),
                move_index = probe.MoveIndex,
                lifecycle_stage = "execute_move_threw",
                execute_move_returned = (bool?)null,
                exception_type = exception.GetType().Name,
                retry_scheduled = false,
                queue_owned_by_schedule = false,
            });
            StopG1HeldSchedule($"g1_execute_move_failed:{exception.GetType().Name}");
            return;
        }

        attempt.ExecuteMoveReturned = executeMoveReturned;
        attempt.PendingAfterCall = input.hasPendingMove;
        attempt.PendingMoveIndexAfterCall = input.pendingMoveIndex;
        attempt.LocalClassification = G1HeldInputScheduleContract.ClassifyKickEdge(
            executeMoveReturned,
            input.hasPendingMove,
            input.pendingMoveIndex,
            probe.MoveIndex);
        _g1KickEdgeObserved[probe.Ordinal] = true;
        measurement.ExecuteMoveReturned = executeMoveReturned;
        measurement.LocalClassification = attempt.LocalClassification;

        EmitG1HeldEvent("g1_kick_request_lifecycle", new
        {
            probe_ordinal = probe.Ordinal,
            probe_label = probe.Label,
            probe_kind = G1ProbeKindName(probe.Kind),
            move_index = probe.MoveIndex,
            lifecycle_stage = "execute_move_returned",
            execute_move_returned = executeMoveReturned,
            local_classification = attempt.LocalClassification,
            pending_move_before_call = pendingBefore,
            pending_move_index_before_call = pendingIndexBefore,
            pending_move_after_call = input.hasPendingMove,
            pending_move_index_after_call = input.pendingMoveIndex,
            translation_held_at_edge =
                ((byte)frame.EffectiveHeldMask & G1HeldInputScheduleContract.TranslationMask) != 0,
            yaw_desired_at_edge =
                ((byte)frame.DesiredHeldMask & G1HeldInputScheduleContract.YawMask) != 0,
            yaw_effective_at_edge = _g1HeldScheduleVelocity.z,
            retry_scheduled = false,
            queue_owned_by_schedule = false,
            attempt_count = 1,
            local_lifecycle_before_call = G1LocalLifecyclePayload(beforeCallLifecycle),
            local_lifecycle_after_call = G1LocalLifecyclePayload(
                CaptureG1LocalLifecycle(input)),
        });

        ObserveG1KickMeasurements(input);

        if (input.hasPendingMove && input.pendingMoveIndex != probe.MoveIndex)
        {
            StopG1HeldSchedule("g1_conflicting_pending_move_after_execute");
            return;
        }
        if (input.hasPendingMove)
        {
            attempt.DispatchTracker =
                G1KickDispatchTracker.Arm(_g1HeldScheduleFixedSubstep);
            EmitG1HeldEvent("g1_kick_request_lifecycle", new
            {
                probe_ordinal = probe.Ordinal,
                probe_label = probe.Label,
                probe_kind = G1ProbeKindName(probe.Kind),
                move_index = probe.MoveIndex,
                lifecycle_stage = "pending_awaiting_render_dispatch_opportunity",
                execute_move_returned = executeMoveReturned,
                armed_fixed_substep = _g1HeldScheduleFixedSubstep,
                pending_move = true,
                pending_move_index = input.pendingMoveIndex,
                cancelled_at_fixed_boundary = false,
                retry_scheduled = false,
                queue_owned_by_schedule = false,
            });
            return;
        }
        if (!executeMoveReturned)
        {
            MarkG1KickAttemptTerminal(attempt, "rejected_locally_no_retry");
            return;
        }
        if (!input.hasPendingMove && !attempt.SendMethodReturned)
            MarkG1KickAttemptTerminal(
                attempt,
                "accepted_return_without_pending_or_send_observation");
    }

    private void RecordG1TranslationRelease(
        RobotInputController input,
        G1KickProbe probe)
    {
        var measurement = _g1KickMeasurements[probe.Ordinal];
        if (measurement is null || measurement.TranslationReleaseFixedSubstep is not null)
        {
            StopG1HeldSchedule("g1_translation_release_without_unique_measurement");
            return;
        }

        var sample = CaptureG1TranslationSettle(input, probe.DesiredHeldMask);
        measurement.TranslationReleaseFixedSubstep = _g1HeldScheduleFixedSubstep;
        measurement.TranslationReleaseQpcTicks = Stopwatch.GetTimestamp();
        measurement.TranslationReleaseSample = sample;
        if (sample.MethodReturned && sample.TransitionSettled)
        {
            measurement.FirstTransitionSettledFixedSubstep =
                _g1HeldScheduleFixedSubstep;
        }
        _g1TranslationReleaseObserved[probe.Ordinal] = true;
        measurement.TranslationReleaseEventSequence = EmitG1HeldEvent(
            "g1_translation_release",
            new
            {
                probe_ordinal = probe.Ordinal,
                probe_label = probe.Label,
                probe_kind = G1ProbeKindName(probe.Kind),
                move_index = probe.MoveIndex,
                release_tick = probe.TranslationReleaseTick,
                release_fixed_substep = _g1HeldScheduleFixedSubstep,
                fixed_substeps_since_edge = _g1HeldScheduleFixedSubstep -
                    probe.EdgeTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick,
                desired_held_mask = (byte)_g1DesiredHeldMask,
                effective_held_mask = (byte)_g1EffectiveHeldMask,
                effective_controller_vector_xyz = new[]
                {
                    _g1HeldScheduleVelocity.x,
                    _g1HeldScheduleVelocity.y,
                    _g1HeldScheduleVelocity.z,
                },
                release_qpc_ticks = measurement.TranslationReleaseQpcTicks,
                qpc_frequency_hz = Stopwatch.Frequency,
                velocity_property_write_returned = true,
                velocity_readback_exact = VelocityEquals(
                    input.VelocityCommand,
                    Vector3.zero),
                transition_settled_diagnostic =
                    G1TranslationSettlePayload(sample),
                retry_scheduled = false,
                second_kick_edge_scheduled = false,
            });
    }

    private void MarkG1KickAttemptTerminal(G1KickAttemptRuntime attempt, string stage)
    {
        if (attempt.Terminal)
            return;
        var measurement = _g1KickMeasurements[attempt.ProbeOrdinal];
        var probe = G1HeldInputScheduleContract.KickProbes[attempt.ProbeOrdinal];
        if (!G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
                probe,
                _g1HeldScheduleFixedSubstep))
        {
            StopG1HeldSchedule("g1_kick_terminal_outside_observation_window");
            return;
        }
        if (measurement is { SummaryEmitted: true })
        {
            StopG1HeldSchedule("g1_kick_terminal_lifecycle_after_measurement_summary");
            return;
        }
        attempt.Terminal = true;
        attempt.TerminalStage = stage;
        _g1KickTerminalOutcomeObserved[attempt.ProbeOrdinal] = true;
        EmitG1HeldEvent("g1_kick_request_lifecycle", new
        {
            probe_ordinal = attempt.ProbeOrdinal,
            probe_label = attempt.ProbeLabel,
            probe_kind = G1ProbeKindName(attempt.Kind),
            move_index = attempt.MoveIndex,
            lifecycle_stage = stage,
            execute_move_returned = attempt.ExecuteMoveReturned,
            local_classification = attempt.LocalClassification,
            send_invoked = attempt.SendInvoked,
            send_method_returned = attempt.SendMethodReturned,
            send_method_local_return_value = attempt.SendMethodReturned
                ? "void_returned_normally"
                : null,
            yaw_neutral_velocity_return_sequence = attempt.YawNeutralVelocityReturnSequence,
            move_send_invoked_sequence = attempt.MoveSendInvokedSequence,
            dispatch_armed_fixed_substep = attempt.DispatchTracker.Armed
                ? attempt.DispatchTracker.ArmedFixedSubstep
                : (int?)null,
            late_update_opportunities = attempt.DispatchTracker.LateUpdateOpportunities,
            send_prefix_fixed_substep =
                _g1KickMeasurements[attempt.ProbeOrdinal]?.SendPrefixFixedSubstep,
            send_postfix_fixed_substep =
                _g1KickMeasurements[attempt.ProbeOrdinal]?.SendPostfixFixedSubstep,
            yaw_neutralization_preceded_move_send =
                attempt.Kind != G1KickProbeKind.YawPreempted ||
                (attempt.YawNeutralVelocityReturnSequence is int velocitySequence &&
                 attempt.MoveSendInvokedSequence is int moveSequence &&
                 velocitySequence < moveSequence),
            retry_scheduled = false,
            queue_owned_by_schedule = false,
            pending_cancelled_by_schedule = false,
            attempt_count = 1,
        });
    }

    private void ObserveG1KickMeasurements(RobotInputController input)
    {
        foreach (var measurement in _g1KickMeasurements)
        {
            if (measurement is null || measurement.ObservationWindowComplete)
                continue;
            var observationStartSubstep =
                measurement.EdgeTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
            if (_g1HeldScheduleFixedSubstep < observationStartSubstep)
                continue;
            var boundaryDecision =
                G1HeldInputScheduleContract.EvaluateKickObservationBoundary(
                    G1HeldInputScheduleContract.KickProbes[measurement.ProbeOrdinal],
                    _g1HeldScheduleFixedSubstep,
                    _g1KickTerminalOutcomeObserved[measurement.ProbeOrdinal]);
            if (boundaryDecision == G1KickObservationBoundaryDecision.FailPartial)
            {
                var attempt = _g1KickAttempt;
                var matchingOwnedPending =
                    attempt is { Terminal: false } &&
                    attempt.ProbeOrdinal == measurement.ProbeOrdinal &&
                    input.hasPendingMove && input.pendingMoveIndex == attempt.MoveIndex;
                StopG1HeldSchedule(matchingOwnedPending
                    ? "g1_owned_pending_move_crossed_kick_observation_boundary"
                    : "g1_kick_request_not_terminal_before_observation_boundary");
                return;
            }
            if (boundaryDecision == G1KickObservationBoundaryDecision.CompleteObservation)
            {
                measurement.ObservationWindowComplete = true;
                EmitG1KickMeasurementSummary(measurement);
                continue;
            }
            if (measurement.LastObservedFixedSubstep == _g1HeldScheduleFixedSubstep)
                continue;

            var snapshot = CaptureG1LocalLifecycle(input);
            var stateChanged = !measurement.HasLastSnapshot ||
                               !snapshot.Equals(measurement.LastSnapshot);
            var translationReleased =
                measurement.Kind == G1KickProbeKind.TranslationHeld &&
                measurement.TranslationReleaseFixedSubstep is int releaseSubstep &&
                _g1HeldScheduleFixedSubstep >= releaseSubstep;
            G1TranslationSettleSnapshot? transitionSettle = null;
            if (translationReleased)
            {
                transitionSettle = CaptureG1TranslationSettle(
                    input,
                    measurement.TranslationHeldMask);
                measurement.LastTranslationSettleSample = transitionSettle.Value;
                if (transitionSettle.Value.MethodReturned &&
                    transitionSettle.Value.TransitionSettled &&
                    measurement.FirstTransitionSettledFixedSubstep is null)
                {
                    measurement.FirstTransitionSettledFixedSubstep =
                        _g1HeldScheduleFixedSubstep;
                }
            }
            measurement.LastObservedFixedSubstep = _g1HeldScheduleFixedSubstep;
            measurement.FixedObservations++;
            EmitG1HeldEvent("g1_kick_fixed_observation", new
            {
                probe_ordinal = measurement.ProbeOrdinal,
                probe_label = measurement.ProbeLabel,
                probe_kind = G1ProbeKindName(measurement.Kind),
                move_index = measurement.MoveIndex,
                edge_fixed_substep = observationStartSubstep,
                fixed_substeps_since_edge =
                    _g1HeldScheduleFixedSubstep - observationStartSubstep,
                fixed_substeps_since_send = measurement.SendPrefixFixedSubstep is int sendSubstep
                    ? _g1HeldScheduleFixedSubstep - sendSubstep
                    : (int?)null,
                send_anchor_observed = measurement.SendPrefixFixedSubstep is not null,
                translation_release_tick = measurement.TranslationReleaseTick,
                translation_released = measurement.Kind == G1KickProbeKind.TranslationHeld
                    ? translationReleased
                    : (bool?)null,
                fixed_substeps_since_translation_release = translationReleased
                    ? _g1HeldScheduleFixedSubstep -
                      measurement.TranslationReleaseFixedSubstep!.Value
                    : (int?)null,
                transition_settled_diagnostic = transitionSettle is { } settle
                    ? G1TranslationSettlePayload(settle)
                    : null,
                pending_move = input.hasPendingMove,
                pending_move_index = input.pendingMoveIndex,
                local_visual_only_diagnostic = G1LocalLifecyclePayload(snapshot),
                recorder_correlation = G1RecorderCorrelation(),
                server_acceptance = "unknown",
                authoritative_execution_observed = false,
            });
            if (stateChanged)
            {
                EmitG1HeldEvent("g1_kick_local_state_transition", new
                {
                    probe_ordinal = measurement.ProbeOrdinal,
                    probe_label = measurement.ProbeLabel,
                    probe_kind = G1ProbeKindName(measurement.Kind),
                    move_index = measurement.MoveIndex,
                    edge_tick = measurement.EdgeTick,
                    edge_fixed_substep = observationStartSubstep,
                    fixed_substeps_since_edge =
                        _g1HeldScheduleFixedSubstep - observationStartSubstep,
                    fixed_substeps_since_send =
                        measurement.SendPrefixFixedSubstep is int transitionSendSubstep
                            ? _g1HeldScheduleFixedSubstep - transitionSendSubstep
                            : (int?)null,
                    previous = measurement.HasLastSnapshot
                        ? G1LocalLifecyclePayload(measurement.LastSnapshot)
                        : null,
                    current = G1LocalLifecyclePayload(snapshot),
                    local_visual_only_diagnostic = true,
                    server_acceptance = "unknown",
                });
            }

            if (measurement.ActionBusyEntryFixedSubstep is null &&
                !measurement.Baseline.ActionBusy && snapshot.ActionBusy)
            {
                measurement.ActionBusyEntryFixedSubstep = _g1HeldScheduleFixedSubstep;
            }
            if (measurement.ActionBusyEntryFixedSubstep is not null &&
                measurement.ActionBusyExitFixedSubstep is null &&
                _g1HeldScheduleFixedSubstep > measurement.ActionBusyEntryFixedSubstep.Value &&
                !snapshot.ActionBusy)
            {
                measurement.ActionBusyExitFixedSubstep = _g1HeldScheduleFixedSubstep;
            }

            var runnerEntry = snapshot.RunnerAvailable && (
                measurement.Baseline.RunnerIsDone == true && snapshot.RunnerIsDone == false ||
                snapshot.CurrentMotionPointer != IntPtr.Zero &&
                snapshot.CurrentMotionPointer != measurement.Baseline.CurrentMotionPointer ||
                snapshot.MotionFrameIndex >= 0 && measurement.Baseline.MotionFrameIndex >= 0 &&
                snapshot.MotionFrameIndex < measurement.Baseline.MotionFrameIndex);
            if (measurement.RunnerMotionEntryFixedSubstep is null && runnerEntry)
            {
                measurement.RunnerMotionEntryFixedSubstep = _g1HeldScheduleFixedSubstep;
                measurement.EntryMotionPointer = snapshot.CurrentMotionPointer;
            }
            if (measurement.RunnerMotionEntryFixedSubstep is not null &&
                measurement.RunnerMotionCompletionFixedSubstep is null &&
                _g1HeldScheduleFixedSubstep >
                    measurement.RunnerMotionEntryFixedSubstep.Value &&
                snapshot.RunnerIsDone == true && !snapshot.InputIsPunching)
            {
                measurement.RunnerMotionCompletionFixedSubstep =
                    _g1HeldScheduleFixedSubstep;
            }

            measurement.LastSnapshot = snapshot;
            measurement.HasLastSnapshot = true;
        }
    }

    private void EmitG1KickMeasurementSummary(G1KickMeasurementRuntime measurement)
    {
        if (measurement.SummaryEmitted)
            return;
        measurement.SummaryEmitted = true;
        var translationClassification =
            G1HeldInputScheduleContract.ClassifyTranslationProbe(
                measurement.ObservationWindowComplete,
                measurement.ExecuteMoveReturned,
                measurement.LocalClassification,
                measurement.MoveSendInvoked,
                measurement.ActionBusyEntryFixedSubstep is not null,
                measurement.RunnerMotionEntryFixedSubstep is not null);
        var translationSendTiming = measurement.Kind == G1KickProbeKind.TranslationHeld &&
                                    measurement.TranslationReleaseFixedSubstep is int release
            ? G1HeldInputScheduleContract.ClassifyTranslationSendTiming(
                measurement.SendPrefixFixedSubstep,
                release,
                measurement.FirstTransitionSettledFixedSubstep)
            : null;
        EmitG1HeldEvent("g1_kick_measurement_summary", new
        {
            probe_ordinal = measurement.ProbeOrdinal,
            probe_label = measurement.ProbeLabel,
            probe_kind = G1ProbeKindName(measurement.Kind),
            move_index = measurement.MoveIndex,
            edge_tick = measurement.EdgeTick,
            observation_stop_tick = measurement.StopTick,
            observation_window_complete = measurement.ObservationWindowComplete,
            observation_ticks_scheduled = measurement.StopTick - measurement.EdgeTick,
            fixed_observations_expected =
                G1HeldInputScheduleContract.KickObservationTicks *
                G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick,
            fixed_observations_observed = measurement.FixedObservations,
            lifecycle_observation_rate_hz = G1HeldInputScheduleContract.UnityFixedRateHz,
            execute_move_returned = measurement.ExecuteMoveReturned,
            local_classification = measurement.LocalClassification,
            baseline_neutral_settled = measurement.BaselineNeutralSettled,
            translation_release_tick = measurement.TranslationReleaseTick,
            translation_release_fixed_substep =
                measurement.TranslationReleaseFixedSubstep,
            translation_release_event_sequence =
                measurement.TranslationReleaseEventSequence,
            translation_release_qpc_ticks = measurement.TranslationReleaseQpcTicks,
            translation_release_qpc_frequency_hz =
                measurement.TranslationReleaseQpcTicks is not null
                    ? Stopwatch.Frequency
                    : (long?)null,
            translation_release_observed = measurement.Kind ==
                G1KickProbeKind.TranslationHeld
                    ? measurement.TranslationReleaseFixedSubstep is not null
                    : (bool?)null,
            translation_first_transition_settled_fixed_substep =
                measurement.FirstTransitionSettledFixedSubstep,
            translation_fixed_substeps_release_to_settled =
                measurement.TranslationReleaseFixedSubstep is int releasedAt &&
                measurement.FirstTransitionSettledFixedSubstep is int settledAt
                    ? settledAt - releasedAt
                    : (int?)null,
            translation_request_send_timing_classification =
                translationSendTiming,
            translation_release_settle_diagnostic =
                measurement.TranslationReleaseSample is { } releaseSample
                    ? G1TranslationSettlePayload(releaseSample)
                    : null,
            translation_last_settle_diagnostic =
                measurement.LastTranslationSettleSample is { } lastSettleSample
                    ? G1TranslationSettlePayload(lastSettleSample)
                    : null,
            translation_post_release_settled_kick_control_included =
                measurement.Kind == G1KickProbeKind.TranslationHeld
                    ? false
                    : (bool?)null,
            translation_post_release_settled_kick_remaining_unknown =
                measurement.Kind == G1KickProbeKind.TranslationHeld
                    ? "single_edge_no_retry_schedule_observes_the_original_request_only"
                    : null,
            move_send_invoked = measurement.MoveSendInvoked,
            move_send_method_returned = measurement.MoveSendMethodReturned,
            yaw_neutral_velocity_return_sequence =
                measurement.YawNeutralVelocityReturnSequence,
            move_send_invoked_sequence = measurement.MoveSendInvokedSequence,
            yaw_neutralization_preceded_move_send =
                measurement.YawNeutralizationPrecededMoveSend,
            requested_move_asset = new
            {
                move_index = measurement.MoveIndex,
                runtime_name = measurement.RequestedAsset.RuntimeName,
                npz_sha256 = measurement.RequestedAsset.NpzSha256,
                recovered_controller_ticks = measurement.RequestedAsset.ControllerTicks,
                mocap_clip_config_pointer = measurement.RequestedClipPointer == IntPtr.Zero
                    ? null
                    : $"0x{measurement.RequestedClipPointer.ToInt64():x}",
            },
            outgoing_projection = new
            {
                method = "RobotInputController.SendMoveEvent",
                move_index = measurement.MoveIndex,
                send_invoked = measurement.MoveSendInvoked,
                method_returned = measurement.MoveSendMethodReturned,
                request_only = true,
                server_acceptance = "unknown",
            },
            send_prefix_anchor = measurement.SendPrefixFixedSubstep is null
                ? null
                : new
                {
                    client_fixed_substep = measurement.SendPrefixFixedSubstep,
                    schedule_tick = measurement.SendPrefixScheduleTick,
                    unity_frame = measurement.SendPrefixUnityFrame,
                    unity_fixed_time = measurement.SendPrefixUnityFixedTime,
                    qpc_ticks = measurement.SendPrefixQpcTicks,
                    qpc_frequency_hz = Stopwatch.Frequency,
                },
            send_postfix_anchor = measurement.SendPostfixFixedSubstep is null
                ? null
                : new
                {
                    client_fixed_substep = measurement.SendPostfixFixedSubstep,
                    schedule_tick = measurement.SendPostfixScheduleTick,
                    unity_frame = measurement.SendPostfixUnityFrame,
                    unity_fixed_time = measurement.SendPostfixUnityFixedTime,
                    qpc_ticks = measurement.SendPostfixQpcTicks,
                    qpc_frequency_hz = Stopwatch.Frequency,
                },
            local_action_busy_entry_fixed_substep =
                measurement.ActionBusyEntryFixedSubstep,
            local_action_busy_exit_fixed_substep =
                measurement.ActionBusyExitFixedSubstep,
            local_action_input_delay_from_send_fixed_substeps =
                G1HeldInputScheduleContract.FixedSubstepsFromSend(
                    measurement.SendPrefixFixedSubstep,
                    measurement.ActionBusyEntryFixedSubstep),
            local_action_busy_duration_fixed_substeps =
                measurement.ActionBusyEntryFixedSubstep is int entry &&
                measurement.ActionBusyExitFixedSubstep is int exit
                    ? exit - entry
                    : (int?)null,
            local_runner_motion_entry_fixed_substep =
                measurement.RunnerMotionEntryFixedSubstep,
            local_runner_motion_completion_fixed_substep =
                measurement.RunnerMotionCompletionFixedSubstep,
            local_runner_entry_delay_from_send_fixed_substeps =
                G1HeldInputScheduleContract.FixedSubstepsFromSend(
                    measurement.SendPrefixFixedSubstep,
                    measurement.RunnerMotionEntryFixedSubstep),
            local_runner_diagnostic_entry_to_completion_fixed_substeps =
                measurement.RunnerMotionEntryFixedSubstep is int motionEntry &&
                measurement.RunnerMotionCompletionFixedSubstep is int motionCompletion
                    ? motionCompletion - motionEntry
                    : (int?)null,
            motion_identity_status = G1HeldInputScheduleContract.MotionIdentityStatus,
            requested_asset_to_runner_motion_identity_proven = false,
            certified_input_delay_ticks = (int?)null,
            certified_entry_to_completion_ticks = (int?)null,
            certified_duration_available = false,
            timing_certification_reason =
                "visual_only_client_diagnostics_are_not_authoritative; correlate recorder local-slot bones",
            translation_local_gate_result = measurement.Kind ==
                G1KickProbeKind.TranslationHeld
                    ? translationClassification.LocalGateResult
                    : null,
            translation_behavior_result = measurement.Kind ==
                G1KickProbeKind.TranslationHeld
                    ? translationClassification.BehavioralResult
                    : null,
            translation_response_classification = measurement.Kind ==
                G1KickProbeKind.TranslationHeld
                    ? translationClassification.ResponseClassification
                    : null,
            translation_result_criteria = measurement.Kind ==
                G1KickProbeKind.TranslationHeld
                    ? translationClassification.Criteria
                    : null,
            physical_response_result = "unknown",
            physical_response_result_criteria =
                "recorder bone-trajectory correlation required",
            baseline = G1LocalLifecyclePayload(measurement.Baseline),
            final = measurement.HasLastSnapshot
                ? G1LocalLifecyclePayload(measurement.LastSnapshot)
                : null,
            entry_motion_pointer = measurement.EntryMotionPointer == IntPtr.Zero
                ? null
                : $"0x{measurement.EntryMotionPointer.ToInt64():x}",
            local_diagnostics_only = true,
            recorder_correlation = G1RecorderCorrelation(),
            request_only = true,
            server_acceptance = "unknown",
        });
    }

    private static G1LocalLifecycleSnapshot CaptureG1LocalLifecycle(
        RobotInputController input)
    {
        try
        {
            var runner = input.sonicRunner;
            if (runner is null)
            {
                return new G1LocalLifecycleSnapshot(
                    input.IsPunching,
                    input.IsRecovering,
                    input.IsPunching || input.IsRecovering,
                    false,
                    null,
                    null,
                    IntPtr.Zero,
                    -1,
                    "sonic_policy_runner_unavailable");
            }
            var motion = runner.currentMotion;
            var motionPointer = motion is null ? IntPtr.Zero : NativePointer(motion);
            var runnerIsDone = runner.IsDone;
            var runnerIsRecovering = runner.IsRecovering;
            return new G1LocalLifecycleSnapshot(
                input.IsPunching,
                input.IsRecovering,
                input.IsPunching || input.IsRecovering,
                true,
                runnerIsDone,
                runnerIsRecovering,
                motionPointer,
                runner.motionFrameIdx,
                null);
        }
        catch (Exception exception)
        {
            return new G1LocalLifecycleSnapshot(
                input.IsPunching,
                input.IsRecovering,
                input.IsPunching || input.IsRecovering,
                false,
                null,
                null,
                IntPtr.Zero,
                -1,
                $"sonic_policy_runner_probe_failed:{exception.GetType().Name}");
        }
    }

    private static object G1LocalLifecyclePayload(G1LocalLifecycleSnapshot snapshot) => new
    {
        input_is_punching = snapshot.InputIsPunching,
        input_is_recovering = snapshot.InputIsRecovering,
        action_busy = snapshot.ActionBusy,
        sonic_policy_runner_available = snapshot.RunnerAvailable,
        sonic_policy_runner_is_done = snapshot.RunnerIsDone,
        sonic_policy_runner_is_recovering = snapshot.RunnerIsRecovering,
        sonic_current_motion_pointer = snapshot.CurrentMotionPointer == IntPtr.Zero
            ? null
            : $"0x{snapshot.CurrentMotionPointer.ToInt64():x}",
        sonic_motion_frame_index = snapshot.MotionFrameIndex,
        probe_reason = snapshot.ProbeReason,
        sonic_action_composer_used = false,
    };

    private static G1TranslationSettleSnapshot CaptureG1TranslationSettle(
        RobotInputController input,
        G1HeldMask heldMask)
    {
        var outgoing = G1TranslationLocomotionDirection(heldMask);
        var outgoingName = G1HeldInputScheduleContract.TranslationOutgoingLocomotion(
            heldMask);
        var config = input.robotConfig;
        if (config is null)
        {
            return G1TranslationSettleSnapshot.Unavailable(
                outgoingName,
                methodInvoked: false,
                "robot_config_unavailable");
        }

        bool transitionSettled;
        try
        {
            transitionSettled = input.TransitionSettled(outgoing);
        }
        catch (Exception exception)
        {
            return G1TranslationSettleSnapshot.Unavailable(
                outgoingName,
                methodInvoked: true,
                $"transition_settled_call_failed:{exception.GetType().Name}",
                config.transitionSettlePlanarSpeed,
                config.transitionSettleYawRate);
        }

        var robot = input.Robot;
        var baseAngularVelocity = Vector3.zero;
        var baseLinearVelocity = Vector3.zero;
        var baseVelocityAvailable = false;
        string? baseVelocityReason = null;
        try
        {
            baseVelocityAvailable = robot is not null &&
                robot.TryGetBaseVelocityLocal(
                    out baseAngularVelocity,
                    out baseLinearVelocity) &&
                Finite(baseAngularVelocity) && Finite(baseLinearVelocity);
            if (!baseVelocityAvailable)
                baseVelocityReason = "base_velocity_local_unavailable_or_invalid";
        }
        catch (Exception exception)
        {
            baseVelocityReason =
                $"base_velocity_local_probe_failed:{exception.GetType().Name}";
        }
        return new G1TranslationSettleSnapshot(
            true,
            true,
            transitionSettled,
            outgoingName,
            baseVelocityAvailable,
            baseVelocityAvailable ? baseLinearVelocity : Vector3.zero,
            baseVelocityAvailable ? baseAngularVelocity : Vector3.zero,
            config.transitionSettlePlanarSpeed,
            config.transitionSettleYawRate,
            baseVelocityReason);
    }

    private static LocomotionDir G1TranslationLocomotionDirection(
        G1HeldMask heldMask) => heldMask switch
    {
        G1HeldMask.W => LocomotionDir.Forward,
        G1HeldMask.S => LocomotionDir.Backward,
        G1HeldMask.A => LocomotionDir.StrafeLeft,
        G1HeldMask.D => LocomotionDir.StrafeRight,
        _ => throw new ArgumentOutOfRangeException(
            nameof(heldMask),
            heldMask,
            "translation probe must map to one exact locomotion direction"),
    };

    private static object G1TranslationSettlePayload(
        G1TranslationSettleSnapshot snapshot) => new
    {
        transition_settled_method_invoked = snapshot.MethodInvoked,
        transition_settled_method_returned = snapshot.MethodReturned,
        transition_settled = snapshot.MethodReturned
            ? snapshot.TransitionSettled
            : (bool?)null,
        outgoing_locomotion = snapshot.OutgoingLocomotion,
        base_velocity_available = snapshot.BaseVelocityAvailable,
        base_linear_velocity_local_m_s = snapshot.BaseVelocityAvailable
            ? new[]
            {
                snapshot.BaseLinearVelocity.x,
                snapshot.BaseLinearVelocity.y,
                snapshot.BaseLinearVelocity.z,
            }
            : null,
        base_angular_velocity_local_rad_s = snapshot.BaseVelocityAvailable
            ? new[]
            {
                snapshot.BaseAngularVelocity.x,
                snapshot.BaseAngularVelocity.y,
                snapshot.BaseAngularVelocity.z,
            }
            : null,
        transition_settle_planar_speed_m_s = snapshot.PlanarThreshold,
        transition_settle_yaw_rate_rad_s = snapshot.YawRateThreshold,
        predicate_mirrored_from_pinned_native_code = false,
        provenance = G1HeldInputScheduleContract.TransitionSettledProvenance,
        base_velocity_provenance =
            G1HeldInputScheduleContract.TransitionSettleBaseVelocityProvenance,
        probe_reason = snapshot.ProbeReason,
    };

    private object G1RecorderCorrelation() => new
    {
        recorder_schema = "rek.private_ai.protocol.v7",
        clock = "client_fixed_tick_500hz",
        client_fixed_substep = _g1HeldScheduleFixedSubstep,
        schedule_tick = _g1HeldScheduleTick,
        unity_fixed_time = Time.fixedTimeAsDouble,
        pose_payload_in_pipe = false,
        pose_response_source = G1HeldInputScheduleContract.PoseResponseSource,
    };

    private static bool TryValidateG1KickAssetBindings(
        RobotInputController input,
        out string reason)
    {
        try
        {
            var moves = input.robotConfig?.moves;
            if (moves is null || moves.Count <= G1HeldInputScheduleContract.KickMoveIndices.Max())
            {
                reason = "g1_kick_move_map_unavailable";
                return false;
            }
            foreach (var expected in G1HeldInputScheduleContract.KickAssets)
            {
                var clip = moves[expected.MoveIndex];
                if (clip is null || clip.npzFile is null ||
                    !string.Equals(clip.name, expected.RuntimeName, StringComparison.Ordinal))
                {
                    reason = $"g1_kick_move_asset_binding_mismatch:{expected.MoveIndex}";
                    return false;
                }
            }
            reason = string.Empty;
            return true;
        }
        catch (Exception exception)
        {
            reason = $"g1_kick_move_asset_binding_probe_failed:{exception.GetType().Name}";
            return false;
        }
    }

    private bool TryRequirePreviousG1ProbeSettled(
        int probeOrdinal,
        RobotInputController input,
        out string reason)
    {
        var current = CaptureG1LocalLifecycle(input);
        if (current.ActionBusy || current.RunnerIsRecovering == true)
        {
            reason = "current_local_action_or_runner_recovery_active";
            return false;
        }
        if (probeOrdinal <= 0)
        {
            reason = string.Empty;
            return true;
        }

        if (_g1KickAttempt is { Terminal: false } || input.hasPendingMove)
        {
            reason = "previous_request_lifecycle_or_pending_move_not_settled";
            return false;
        }

        var previous = _g1KickMeasurements[probeOrdinal - 1];
        if (previous is null || !previous.ObservationWindowComplete)
        {
            reason = "previous_observation_window_incomplete";
            return false;
        }
        if (previous.ActionBusyEntryFixedSubstep is not null &&
            previous.ActionBusyExitFixedSubstep is null)
        {
            reason = "previous_local_action_busy_response_not_settled";
            return false;
        }
        if (previous.Kind == G1KickProbeKind.TranslationHeld &&
            previous.FirstTransitionSettledFixedSubstep is null)
        {
            reason = "previous_translation_release_never_reached_transition_settled";
            return false;
        }
        // The visual-only client runner cannot bind currentMotion to the requested
        // MocapClipConfig. Its transitions are recorded, but they are not used as
        // an action-overlap authority.
        reason = string.Empty;
        return true;
    }

    private bool TryGetExactG1HeldScope(out PrivateAiContext scope, out string reason)
    {
        scope = null!;
        if (!TryVerifyExplicitIsolatedSession(out var isolationProof) ||
            isolationProof != G1HeldInputScheduleContract.RequiredIsolationProof)
        {
            reason = "exact_isolated_spark_marker_not_proven";
            return false;
        }
        if (!TryGetPrivateAiContext(requireActiveRound: true, out scope, out reason))
            return false;
        if (_g1HeldScheduleIdentity is null ||
            !_g1HeldScheduleIdentity.Equals(RuntimeIdentity.From(scope)))
        {
            reason = "g1_held_runtime_identity_changed";
            return false;
        }
        var pairing = ReadMeasuredPairing(
            scope.Coordinator,
            scope.LocalSlot,
            scope.OpponentSlot);
        if (!pairing.Validation.ExactG1VersusG1)
        {
            reason = $"exact_g1_pairing_lost:{pairing.Validation.Reason}";
            return false;
        }
        if (scope.Input is null || _g1HeldScheduleInputPointer == IntPtr.Zero ||
            NativePointer(scope.Input) != _g1HeldScheduleInputPointer)
        {
            reason = "g1_held_input_controller_changed";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    private bool RequireOwnedG1HeldScheduleControl(out string reason)
    {
        var connectionId = _pipe?.CurrentConnectionId ?? 0;
        if (!_g1HeldScheduleRunning || !_g1HeldScheduleAuthorizedWhileBackground)
        {
            reason = "g1_held_schedule_not_authorized_while_background";
            return false;
        }
        if (_leaseConnectionId == 0 || connectionId != _leaseConnectionId)
        {
            reason = "exclusive_g1_held_schedule_lease_not_owned";
            return false;
        }
        if (!RequireBackgroundControl(out reason))
            return false;
        reason = string.Empty;
        return true;
    }

    private void OnG1HeldInputLateUpdatePrefix(RobotInputController input)
    {
        try
        {
            if (!RequireOwnedG1HeldScheduleControl(out var controlReason))
            {
                StopG1HeldSchedule(controlReason);
                return;
            }
            if (!TryGetExactG1HeldScope(out _, out var scopeReason))
            {
                StopG1HeldSchedule($"g1_held_late_update_scope_lost:{scopeReason}");
                return;
            }
            if (input.hasPendingSpecial || input.hasPendingEStop)
            {
                StopG1HeldSchedule("g1_held_unexpected_special_or_estop_at_late_update");
                return;
            }
            if (input.hasPendingMove &&
                (_g1KickAttempt is null || _g1KickAttempt.Terminal ||
                 input.pendingMoveIndex != _g1KickAttempt.MoveIndex))
            {
                StopG1HeldSchedule("g1_held_unowned_pending_move_at_late_update");
                return;
            }
            var attempt = _g1KickAttempt;
            if (attempt is { Terminal: false })
            {
                var probe = G1HeldInputScheduleContract.KickProbes[attempt.ProbeOrdinal];
                if (!G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
                        probe,
                        _g1HeldScheduleFixedSubstep))
                {
                    StopG1HeldSchedule(input.hasPendingMove &&
                                       input.pendingMoveIndex == attempt.MoveIndex
                        ? "g1_owned_pending_move_crossed_kick_observation_boundary"
                        : "g1_kick_request_not_terminal_before_observation_boundary");
                    return;
                }
            }
            var robotConfig = input.robotConfig;
            if (robotConfig is null ||
                !SameFloatBits(
                    robotConfig.yawSpeed,
                    G1HeldInputScheduleContract.ExpectedYawSpeed) ||
                !SameFloatBits(
                    robotConfig.keyboardYawRampTime,
                    G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds))
            {
                StopG1HeldSchedule("g1_keyboard_yaw_runtime_config_changed");
                return;
            }
            var beforeYawState = _g1KeyboardYawState;
            var yawStep = G1HeldInputScheduleContract.AdvanceKeyboardYaw(
                beforeYawState,
                _g1RawYawTarget,
                Time.deltaTime,
                robotConfig.keyboardYawRampTime,
                robotConfig.yawSpeed);
            _g1KeyboardYawState = yawStep.State;
            _g1HeldScheduleVelocity.z = yawStep.EffectiveYaw;
            if (!SetVelocityExact(input, _g1HeldScheduleVelocity))
            {
                StopG1HeldSchedule("g1_held_velocity_late_update_readback_mismatch");
                return;
            }
            EmitG1HeldEvent("g1_yaw_ramp_update", new
            {
                phase = "RobotInputController.LateUpdate.prefix",
                desired_key_mask = (byte)_g1DesiredHeldMask,
                effective_held_mask = (byte)_g1EffectiveHeldMask,
                raw_yaw_target = _g1RawYawTarget,
                rendered_delta_time = Time.deltaTime,
                keyboard_yaw_ramp_time = robotConfig.keyboardYawRampTime,
                yaw_speed = robotConfig.yawSpeed,
                ramp_before = beforeYawState.Ramp,
                sign_before = beforeYawState.Sign,
                ramp_after = yawStep.State.Ramp,
                sign_after = yawStep.State.Sign,
                effective_yaw = yawStep.EffectiveYaw,
                effective_controller_vector_xyz = new[]
                {
                    _g1HeldScheduleVelocity.x,
                    _g1HeldScheduleVelocity.y,
                    _g1HeldScheduleVelocity.z,
                },
                qpc_ticks = Stopwatch.GetTimestamp(),
                qpc_frequency_hz = Stopwatch.Frequency,
                provenance = G1HeldInputScheduleContract.KeyboardYawRampProvenance,
            });

            attempt = _g1KickAttempt;
            if (attempt is null || !attempt.DispatchTracker.Armed ||
                attempt.DispatchTracker.SendPrefixSeen)
            {
                return;
            }
            if (!input.hasPendingMove || input.pendingMoveIndex != attempt.MoveIndex)
            {
                EmitG1HeldEvent("g1_kick_dispatch_opportunity", new
                {
                    probe_ordinal = attempt.ProbeOrdinal,
                    probe_label = attempt.ProbeLabel,
                    probe_kind = G1ProbeKindName(attempt.Kind),
                    move_index = attempt.MoveIndex,
                    stage = "pending_missing_before_first_matching_late_update",
                    cancelled_by_schedule = false,
                    late_update_opportunities =
                        attempt.DispatchTracker.LateUpdateOpportunities,
                });
                StopG1HeldSchedule("g1_owned_pending_move_lost_before_dispatch_opportunity");
                return;
            }
            attempt.DispatchTracker =
                attempt.DispatchTracker.OnLateUpdatePrefix(matchingPending: true);
            EmitG1HeldEvent("g1_kick_dispatch_opportunity", new
            {
                probe_ordinal = attempt.ProbeOrdinal,
                probe_label = attempt.ProbeLabel,
                probe_kind = G1ProbeKindName(attempt.Kind),
                move_index = attempt.MoveIndex,
                stage = "matching_late_update_prefix_entered",
                pending_move = input.hasPendingMove,
                pending_move_index = input.pendingMoveIndex,
                late_update_opportunities =
                    attempt.DispatchTracker.LateUpdateOpportunities,
                cancelled_by_schedule = false,
                fixed_updates_since_arm =
                    _g1HeldScheduleFixedSubstep - attempt.DispatchTracker.ArmedFixedSubstep,
                qpc_ticks = Stopwatch.GetTimestamp(),
                qpc_frequency_hz = Stopwatch.Frequency,
            });
        }
        catch (Exception exception)
        {
            StopG1HeldSchedule($"g1_held_late_update_failed:{exception.GetType().Name}");
        }
    }

    private void OnG1HeldInputLateUpdatePostfix(RobotInputController input)
    {
        if (!_g1HeldScheduleRunning ||
            NativePointer(input) != _g1HeldScheduleInputPointer)
        {
            return;
        }
        var attempt = _g1KickAttempt;
        if (attempt is null || !attempt.DispatchTracker.LateUpdateOpen)
            return;
        attempt.DispatchTracker = attempt.DispatchTracker.OnLateUpdatePostfix();
        EmitG1HeldEvent("g1_kick_dispatch_opportunity", new
        {
            probe_ordinal = attempt.ProbeOrdinal,
            probe_label = attempt.ProbeLabel,
            probe_kind = G1ProbeKindName(attempt.Kind),
            move_index = attempt.MoveIndex,
            stage = "matching_late_update_postfix_completed",
            late_update_opportunities = attempt.DispatchTracker.LateUpdateOpportunities,
            send_prefix_seen = attempt.DispatchTracker.SendPrefixSeen,
            send_postfix_seen = attempt.DispatchTracker.SendPostfixSeen,
            pending_move_after_late_update = input.hasPendingMove,
            pending_move_index_after_late_update = input.pendingMoveIndex,
            missing_dispatch_after_completed_opportunity =
                attempt.DispatchTracker.MissingDispatchAfterCompletedOpportunity,
            cancelled_by_schedule = false,
            qpc_ticks = Stopwatch.GetTimestamp(),
            qpc_frequency_hz = Stopwatch.Frequency,
        });
        if (attempt.DispatchTracker.MissingDispatchAfterCompletedOpportunity)
        {
            StopG1HeldSchedule("g1_matching_late_update_completed_without_move_dispatch");
        }
    }

    private void OnG1HeldVelocityPrefix(RobotInputController input)
    {
        try
        {
            var scopeReason = string.Empty;
            if (!RequireOwnedG1HeldScheduleControl(out var controlReason) ||
                !TryGetExactG1HeldScope(out _, out scopeReason))
            {
                StopG1HeldSchedule(string.IsNullOrEmpty(controlReason)
                    ? $"g1_velocity_scope_lost:{scopeReason}"
                    : controlReason);
                return;
            }
            if (!SetVelocityExact(input, _g1HeldScheduleVelocity))
            {
                StopG1HeldSchedule("g1_velocity_send_readback_mismatch");
                return;
            }
            _g1VelocityInvocationObserved = true;
            _g1VelocityInvocationTick = _g1HeldScheduleTick;
            _g1VelocityInvocationSubstep = _g1HeldScheduleFixedSubstep;
            EmitG1HeldEvent("g1_velocity_request_lifecycle", new
            {
                lifecycle_stage = "send_velocity_invoked",
                method = "RobotInputController.SendVelocityCommand",
                velocity_command_xyz = new[]
                {
                    _g1HeldScheduleVelocity.x,
                    _g1HeldScheduleVelocity.y,
                    _g1HeldScheduleVelocity.z,
                },
                desired_held_mask = (byte)_g1DesiredHeldMask,
                effective_held_mask = (byte)_g1EffectiveHeldMask,
                raw_yaw_target = _g1RawYawTarget,
                method_returned = false,
            });
        }
        catch (Exception exception)
        {
            StopG1HeldSchedule($"g1_velocity_prefix_failed:{exception.GetType().Name}");
        }
    }

    private void OnG1HeldVelocityPostfix(RobotInputController input)
    {
        if (!_g1VelocityInvocationObserved ||
            NativePointer(input) != _g1HeldScheduleInputPointer)
        {
            return;
        }
        _g1VelocityInvocationObserved = false;
        var eventSequence = EmitG1HeldEvent("g1_velocity_request_lifecycle", new
        {
            lifecycle_stage = "client_request_method_returned",
            method = "RobotInputController.SendVelocityCommand",
            invocation_schedule_tick = _g1VelocityInvocationTick,
            invocation_fixed_substep = _g1VelocityInvocationSubstep,
            velocity_command_xyz = new[]
            {
                _g1HeldScheduleVelocity.x,
                _g1HeldScheduleVelocity.y,
                _g1HeldScheduleVelocity.z,
            },
            desired_held_mask = (byte)_g1DesiredHeldMask,
            effective_held_mask = (byte)_g1EffectiveHeldMask,
            raw_yaw_target = _g1RawYawTarget,
            method_returned = true,
            local_return_value = "void_returned_normally",
        });
        if (_g1HeldScheduleTick == G1HeldInputScheduleContract.FinalScheduleTick &&
            VelocityEquals(_g1HeldScheduleVelocity, Vector3.zero))
        {
            _g1FinalNeutralSendReturned = true;
        }
        var attempt = _g1KickAttempt;
        if (attempt is not null && !attempt.Terminal &&
            attempt.Kind == G1KickProbeKind.YawPreempted &&
            _g1HeldScheduleTick >= attempt.EdgeTick &&
            VelocityEquals(_g1HeldScheduleVelocity, Vector3.zero))
        {
            attempt.YawNeutralVelocityReturnSequence ??= eventSequence;
            var measurement = _g1KickMeasurements[attempt.ProbeOrdinal];
            if (measurement is not null)
            {
                measurement.YawNeutralVelocityReturnSequence ??= eventSequence;
            }
        }
    }

    private bool OnG1HeldMovePrefix(RobotInputController input)
    {
        var attempt = _g1KickAttempt;
        if (attempt is null || attempt.Terminal)
        {
            StopG1HeldSchedule("g1_unowned_move_send_invocation");
            return false;
        }
        var probe = G1HeldInputScheduleContract.KickProbes[attempt.ProbeOrdinal];
        if (!G1HeldInputScheduleContract.IsKickLifecycleInsideObservationWindow(
                probe,
                _g1HeldScheduleFixedSubstep))
        {
            StopG1HeldSchedule("g1_move_send_outside_kick_observation_window");
            return false;
        }
        try
        {
            var scopeReason = string.Empty;
            if (!RequireOwnedG1HeldScheduleControl(out var controlReason) ||
                !TryGetExactG1HeldScope(out _, out scopeReason))
            {
                StopG1HeldSchedule(string.IsNullOrEmpty(controlReason)
                    ? $"g1_move_scope_lost:{scopeReason}"
                    : controlReason);
                return false;
            }
            if (!input.hasPendingMove || input.pendingMoveIndex != attempt.MoveIndex)
            {
                StopG1HeldSchedule("g1_move_send_pending_state_mismatch");
                return false;
            }
            attempt.SendInvoked = true;
            attempt.DispatchTracker = attempt.DispatchTracker.OnSendPrefix();
            var sendPrefixQpc = Stopwatch.GetTimestamp();
            var sendPrefixUnityFrame = Time.frameCount;
            var sendPrefixUnityFixedTime = Time.fixedTimeAsDouble;
            attempt.MoveSendInvokedSequence = EmitG1HeldEvent(
                "g1_kick_request_lifecycle",
                new
                {
                    probe_ordinal = attempt.ProbeOrdinal,
                    probe_label = attempt.ProbeLabel,
                    probe_kind = G1ProbeKindName(attempt.Kind),
                    move_index = attempt.MoveIndex,
                    lifecycle_stage = "send_move_invoked",
                    execute_move_returned = attempt.ExecuteMoveReturned,
                    pending_move = input.hasPendingMove,
                    pending_move_index = input.pendingMoveIndex,
                    send_prefix_fixed_substep = _g1HeldScheduleFixedSubstep,
                    send_prefix_schedule_tick = _g1HeldScheduleTick,
                    send_prefix_unity_frame = sendPrefixUnityFrame,
                    send_prefix_unity_fixed_time = sendPrefixUnityFixedTime,
                    send_prefix_qpc_ticks = sendPrefixQpc,
                    qpc_frequency_hz = Stopwatch.Frequency,
                    late_update_opportunities =
                        attempt.DispatchTracker.LateUpdateOpportunities,
                    retry_scheduled = false,
                    queue_owned_by_schedule = false,
                    attempt_count = 1,
                });
            var measurement = _g1KickMeasurements[attempt.ProbeOrdinal];
            if (measurement is not null)
            {
                measurement.MoveSendInvoked = true;
                measurement.MoveSendInvokedSequence = attempt.MoveSendInvokedSequence;
                measurement.SendPrefixFixedSubstep = _g1HeldScheduleFixedSubstep;
                measurement.SendPrefixScheduleTick = _g1HeldScheduleTick;
                measurement.SendPrefixUnityFrame = sendPrefixUnityFrame;
                measurement.SendPrefixUnityFixedTime = sendPrefixUnityFixedTime;
                measurement.SendPrefixQpcTicks = sendPrefixQpc;
                measurement.YawNeutralVelocityReturnSequence =
                    attempt.YawNeutralVelocityReturnSequence;
            }
            return true;
        }
        catch (Exception exception)
        {
            StopG1HeldSchedule($"g1_move_prefix_failed:{exception.GetType().Name}");
            return false;
        }
    }

    private void OnG1HeldMovePostfix(RobotInputController input)
    {
        var attempt = _g1KickAttempt;
        if (attempt is null || attempt.Terminal || !attempt.SendInvoked ||
            NativePointer(input) != _g1HeldScheduleInputPointer)
        {
            return;
        }
        attempt.SendMethodReturned = true;
        attempt.DispatchTracker = attempt.DispatchTracker.OnSendPostfix();
        var sendPostfixQpc = Stopwatch.GetTimestamp();
        var measurement = _g1KickMeasurements[attempt.ProbeOrdinal];
        if (measurement is not null)
        {
            measurement.MoveSendMethodReturned = true;
            measurement.SendPostfixFixedSubstep = _g1HeldScheduleFixedSubstep;
            measurement.SendPostfixScheduleTick = _g1HeldScheduleTick;
            measurement.SendPostfixUnityFrame = Time.frameCount;
            measurement.SendPostfixUnityFixedTime = Time.fixedTimeAsDouble;
            measurement.SendPostfixQpcTicks = sendPostfixQpc;
        }
        MarkG1KickAttemptTerminal(attempt, "client_request_method_returned");
    }

    private void OnG1HeldMoveFailure(RobotInputController input, Exception exception)
    {
        var attempt = _g1KickAttempt;
        if (attempt is null || attempt.Terminal ||
            NativePointer(input) != _g1HeldScheduleInputPointer)
        {
            return;
        }
        EmitG1HeldEvent("g1_kick_request_lifecycle", new
        {
            probe_ordinal = attempt.ProbeOrdinal,
            probe_label = attempt.ProbeLabel,
            probe_kind = G1ProbeKindName(attempt.Kind),
            move_index = attempt.MoveIndex,
            lifecycle_stage = "send_move_threw",
            exception_type = exception.GetType().Name,
            retry_scheduled = false,
            queue_owned_by_schedule = false,
        });
        StopG1HeldSchedule($"g1_move_send_failed:{exception.GetType().Name}");
    }

    private bool RejectUnexpectedG1SpecialOrEStop(
        RobotInputController input,
        string requestKind)
    {
        if (!_g1HeldScheduleRunning || _g1HeldScheduleInputPointer == IntPtr.Zero ||
            NativePointer(input) != _g1HeldScheduleInputPointer)
        {
            return false;
        }
        EmitG1HeldEvent("g1_unexpected_local_request_blocked", new
        {
            request_kind = requestKind,
            original_method_skipped = true,
        });
        StopG1HeldSchedule($"g1_unexpected_{requestKind}_request_blocked");
        return true;
    }

    private void StopG1HeldSchedule(string requestedReason)
    {
        if (!_g1HeldScheduleRunning)
            return;
        var runId = _g1HeldScheduleRunId;
        var authorizedWhileBackground = _g1HeldScheduleAuthorizedWhileBackground;
        var input = _g1HeldScheduleInput;
        var attempt = _g1KickAttempt;
        var ownedPendingClearedOnStop = false;
        if (input is not null && _g1HeldScheduleInputPointer != IntPtr.Zero &&
            NativePointer(input) == _g1HeldScheduleInputPointer)
        {
            try
            {
                if (attempt is not null && !attempt.Terminal && input.hasPendingMove &&
                    input.pendingMoveIndex == attempt.MoveIndex)
                {
                    input.hasPendingMove = false;
                    ownedPendingClearedOnStop = true;
                }
                input.VelocityCommand = Vector3.zero;
            }
            catch
            {
            }
        }

        foreach (var measurement in _g1KickMeasurements)
        {
            if (measurement is not null && !measurement.SummaryEmitted)
                EmitG1KickMeasurementSummary(measurement);
        }
        var coverage = BuildG1HeldCoverage();
        var reason = requestedReason == "complete" && !coverage.Complete
            ? "coverage_incomplete"
            : requestedReason;
        var complete = reason == "complete";
        _g1HeldScheduleRunning = false;
        _g1HeldScheduleAuthorizedWhileBackground = false;
        _g1VelocityInvocationObserved = false;
        _g1KickAttempt = null;

        try
        {
            var payload = new
            {
                @event = "g1_held_schedule_end",
                protocol = "rek.ui_bridge.v1",
                g1_held_schedule_schema = G1HeldInputScheduleContract.Schema,
                g1_held_schedule_id = G1HeldInputScheduleContract.ScheduleId,
                g1_held_schedule_sha256 = G1HeldInputScheduleContract.ExpectedSha256,
                g1_held_schedule_run_id = runId,
                fresh_round_request_id = _g1HeldScheduleFreshRoundRequestId,
                round_identity_sha256 = _g1HeldScheduleRoundIdentitySha256,
                schedule_tick = _g1HeldScheduleTick,
                client_fixed_substep = _g1HeldScheduleFixedSubstep,
                reason,
                complete,
                experiment_coverage_complete = coverage.Complete,
                partial_coverage = !complete,
                authorized_while_background = authorizedWhileBackground,
                final_neutral_send_method_returned = _g1FinalNeutralSendReturned,
                owned_pending_cleared_only_during_stop = ownedPendingClearedOnStop,
                round_capacity_preflight = new
                {
                    round_duration_seconds = _g1HeldRoundDurationSeconds,
                    initial_time_remaining_seconds = _g1HeldInitialTimeRemainingSeconds,
                    required_run_seconds = G1HeldInputScheduleContract.RequiredRunSeconds,
                    safety_seconds = G1HeldInputScheduleContract.RoundCapacitySafetySeconds,
                    required_capacity_seconds =
                        G1HeldInputScheduleContract.RequiredRoundCapacitySeconds,
                    capacity_proven = G1HeldInputScheduleContract.HasRoundCapacity(
                        _g1HeldRoundDurationSeconds,
                        _g1HeldInitialTimeRemainingSeconds),
                },
                coverage,
                pose_response_source = G1HeldInputScheduleContract.PoseResponseSource,
                pose_response_in_pipe_transcript = false,
                sonic_action_composer_lifecycle_used = false,
                global_input_emitted = false,
                request_only = true,
                server_acceptance = "unknown",
                server_acceptance_observed = false,
                authoritative_execution_observed = false,
                unity_frame = Time.frameCount,
                unity_fixed_time = Time.fixedTimeAsDouble,
            };
            _pipe?.Send(_leaseConnectionId, payload);
            Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        }
        catch
        {
        }

        _g1HeldScheduleInput = null;
        _g1HeldScheduleInputPointer = IntPtr.Zero;
        _g1HeldScheduleIdentity = null;
        _g1DesiredHeldMask = G1HeldMask.None;
        _g1EffectiveHeldMask = G1HeldMask.None;
        _g1RawYawTarget = 0f;
        _g1KeyboardYawState = new G1KeyboardYawState(0f, 0f);
        _g1HeldScheduleVelocity = Vector3.zero;
    }

    private G1HeldCoverage BuildG1HeldCoverage()
    {
        var heldConditionDurationsExact = _g1HeldConditionObservedTicks.All(
            value => value == G1HeldInputScheduleContract.HeldDurationTicks);
        var translationProbeOrdinals = G1HeldInputScheduleContract.KickProbes
            .Where(value => value.Kind == G1KickProbeKind.TranslationHeld)
            .Select(value => value.Ordinal)
            .ToArray();
        var yawProbeOrdinals = G1HeldInputScheduleContract.KickProbes
            .Where(value => value.Kind == G1KickProbeKind.YawPreempted)
            .Select(value => value.Ordinal)
            .ToArray();
        var allEdges = _g1KickEdgeObserved.All(value => value);
        var allTerminal = _g1KickTerminalOutcomeObserved.All(value => value);
        var allYawPreemptions = yawProbeOrdinals.All(value => _g1YawPreemptionObserved[value]);
        var allTranslationReleases = translationProbeOrdinals.All(
            value => _g1TranslationReleaseObserved[value]);
        var allObservationWindowsComplete = _g1KickMeasurements.All(
            value => value is { ObservationWindowComplete: true });
        var expectedFixedObservations =
            G1HeldInputScheduleContract.KickObservationTicks *
            G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
        var allFixedObservationCountsExact = _g1KickMeasurements.All(
            value => value?.FixedObservations == expectedFixedObservations);
        var complete = heldConditionDurationsExact && allEdges && allTerminal &&
                       allTranslationReleases &&
                       allYawPreemptions && allObservationWindowsComplete &&
                       allFixedObservationCountsExact && _g1FinalNeutralSendReturned;
        return new G1HeldCoverage(
            complete,
            G1HeldInputScheduleContract.HeldConditions.Length,
            _g1HeldConditionObservedTicks.Count(value =>
                value == G1HeldInputScheduleContract.HeldDurationTicks),
            _g1HeldConditionObservedTicks.ToArray(),
            translationProbeOrdinals.Length,
            translationProbeOrdinals.Count(value => _g1KickEdgeObserved[value]),
            translationProbeOrdinals.Count(value => _g1KickTerminalOutcomeObserved[value]),
            translationProbeOrdinals.Count(value => _g1TranslationReleaseObserved[value]),
            yawProbeOrdinals.Length,
            yawProbeOrdinals.Count(value => _g1KickEdgeObserved[value]),
            yawProbeOrdinals.Count(value => _g1KickTerminalOutcomeObserved[value]),
            yawProbeOrdinals.Count(value => _g1YawPreemptionObserved[value]),
            _g1KickMeasurements.Count(value => value is { SummaryEmitted: true }),
            _g1KickMeasurements.Count(value => value is { ObservationWindowComplete: true }),
            _g1KickMeasurements.Count(value =>
                value?.FixedObservations == expectedFixedObservations),
            G1HeldInputScheduleContract.KickProbes
                .Select(G1KickCoveragePayload)
                .ToArray(),
            G1HeldInputScheduleContract.KickMoveIndices,
            FBindingIncluded: false,
            QueueOrRetryUsed: false,
            SonicActionComposerLifecycleUsed: false,
            PhysicalBehaviorCertified: false,
            PoseResponseSource: G1HeldInputScheduleContract.PoseResponseSource);
    }

    private object G1KickCoveragePayload(G1KickProbe probe)
    {
        var measurement = _g1KickMeasurements[probe.Ordinal];
        return new
        {
            probe_ordinal = probe.Ordinal,
            probe_label = probe.Label,
            probe_kind = G1ProbeKindName(probe.Kind),
            move_index = probe.MoveIndex,
            edge_tick = probe.EdgeTick,
            translation_release_tick = probe.TranslationReleaseTick,
            observation_stop_tick = probe.StopTick,
            edge_observed = _g1KickEdgeObserved[probe.Ordinal],
            terminal_outcome_observed = _g1KickTerminalOutcomeObserved[probe.Ordinal],
            translation_release_observed =
                _g1TranslationReleaseObserved[probe.Ordinal],
            yaw_preemption_observed = _g1YawPreemptionObserved[probe.Ordinal],
            measurement_started = measurement is not null,
            summary_emitted = measurement?.SummaryEmitted ?? false,
            observation_window_complete = measurement?.ObservationWindowComplete ?? false,
            execute_move_returned = measurement?.ExecuteMoveReturned,
            local_classification = measurement?.LocalClassification,
            baseline_neutral_settled = measurement?.BaselineNeutralSettled ?? false,
            move_send_invoked = measurement?.MoveSendInvoked ?? false,
            move_send_method_returned = measurement?.MoveSendMethodReturned ?? false,
            translation_release_fixed_substep =
                measurement?.TranslationReleaseFixedSubstep,
            translation_first_transition_settled_fixed_substep =
                measurement?.FirstTransitionSettledFixedSubstep,
            yaw_neutralization_preceded_move_send =
                measurement?.YawNeutralizationPrecededMoveSend ?? false,
            fixed_observations = measurement?.FixedObservations ?? 0,
            fixed_observations_expected =
                G1HeldInputScheduleContract.KickObservationTicks *
                G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick,
            local_action_busy_entry_fixed_substep =
                measurement?.ActionBusyEntryFixedSubstep,
            local_action_busy_exit_fixed_substep =
                measurement?.ActionBusyExitFixedSubstep,
            local_runner_motion_entry_fixed_substep =
                measurement?.RunnerMotionEntryFixedSubstep,
            local_runner_motion_completion_fixed_substep =
                measurement?.RunnerMotionCompletionFixedSubstep,
            motion_identity_status = G1HeldInputScheduleContract.MotionIdentityStatus,
            certified_duration_available = false,
            physical_behavior_result = "unknown",
            request_only = true,
            server_acceptance = "unknown",
        };
    }

    private int EmitG1HeldEvent(string eventName, object detail)
    {
        var sequence = ++_g1HeldEventSequence;
        var payload = new
        {
            @event = eventName,
            protocol = "rek.ui_bridge.v1",
            g1_held_schedule_schema = G1HeldInputScheduleContract.Schema,
            g1_held_schedule_id = G1HeldInputScheduleContract.ScheduleId,
            g1_held_schedule_sha256 = G1HeldInputScheduleContract.ExpectedSha256,
            g1_held_schedule_run_id = _g1HeldScheduleRunId,
            fresh_round_request_id = _g1HeldScheduleFreshRoundRequestId,
            round_identity_sha256 = _g1HeldScheduleRoundIdentitySha256,
            event_sequence = sequence,
            schedule_tick = _g1HeldScheduleTick,
            client_fixed_substep = _g1HeldScheduleFixedSubstep,
            fixed_substeps_per_schedule_tick =
                G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick,
            schedule_rate_hz = G1HeldInputScheduleContract.ScheduleRateHz,
            unity_fixed_rate_hz = G1HeldInputScheduleContract.UnityFixedRateHz,
            detail,
            authority_scope = G1HeldInputScheduleContract.AuthorityScope,
            authority_caveat = G1HeldInputScheduleContract.AuthorityCaveat,
            recorder_correlation = G1RecorderCorrelation(),
            request_only = true,
            server_acceptance = "unknown",
            server_acceptance_observed = false,
            authoritative_execution_observed = false,
            global_input_emitted = false,
            unity_frame = Time.frameCount,
            unity_fixed_time = Time.fixedTimeAsDouble,
        };
        _pipe?.Send(_leaseConnectionId, payload);
        Log.LogInfo(JsonSerializer.Serialize(payload, BridgeJson.Options));
        return sequence;
    }

    private static string G1ProbeKindName(G1KickProbeKind kind) =>
        kind == G1KickProbeKind.TranslationHeld
            ? "translation_held"
            : "yaw_preempted";

    private sealed class G1KickAttemptRuntime
    {
        internal G1KickAttemptRuntime(
            int probeOrdinal,
            G1KickProbeKind kind,
            string probeLabel,
            int moveIndex,
            int edgeTick,
            int edgeSequence,
            int edgeFixedSubstep)
        {
            ProbeOrdinal = probeOrdinal;
            Kind = kind;
            ProbeLabel = probeLabel;
            MoveIndex = moveIndex;
            EdgeTick = edgeTick;
            EdgeSequence = edgeSequence;
            EdgeFixedSubstep = edgeFixedSubstep;
        }

        internal int ProbeOrdinal { get; }
        internal G1KickProbeKind Kind { get; }
        internal string ProbeLabel { get; }
        internal int MoveIndex { get; }
        internal int EdgeTick { get; }
        internal int EdgeSequence { get; }
        internal int EdgeFixedSubstep { get; }
        internal bool? ExecuteMoveReturned { get; set; }
        internal bool PendingAfterCall { get; set; }
        internal int PendingMoveIndexAfterCall { get; set; }
        internal string? LocalClassification { get; set; }
        internal bool SendInvoked { get; set; }
        internal bool SendMethodReturned { get; set; }
        internal G1KickDispatchTracker DispatchTracker { get; set; }
        internal int? YawNeutralVelocityReturnSequence { get; set; }
        internal int? MoveSendInvokedSequence { get; set; }
        internal bool Terminal { get; set; }
        internal string? TerminalStage { get; set; }
    }

    private sealed record G1LocalLifecycleSnapshot(
        bool InputIsPunching,
        bool InputIsRecovering,
        bool ActionBusy,
        bool RunnerAvailable,
        bool? RunnerIsDone,
        bool? RunnerIsRecovering,
        IntPtr CurrentMotionPointer,
        int MotionFrameIndex,
        string? ProbeReason);

    private readonly record struct G1TranslationSettleSnapshot(
        bool MethodInvoked,
        bool MethodReturned,
        bool TransitionSettled,
        string OutgoingLocomotion,
        bool BaseVelocityAvailable,
        Vector3 BaseLinearVelocity,
        Vector3 BaseAngularVelocity,
        float PlanarThreshold,
        float YawRateThreshold,
        string? ProbeReason)
    {
        internal static G1TranslationSettleSnapshot Unavailable(
            string outgoingLocomotion,
            bool methodInvoked,
            string reason,
            float planarThreshold = 0f,
            float yawRateThreshold = 0f) => new(
            methodInvoked,
            false,
            false,
            outgoingLocomotion,
            false,
            Vector3.zero,
            Vector3.zero,
            planarThreshold,
            yawRateThreshold,
            reason);
    }

    private sealed class G1KickMeasurementRuntime
    {
        internal G1KickMeasurementRuntime(
            int probeOrdinal,
            G1KickProbeKind kind,
            string probeLabel,
            int moveIndex,
            int edgeTick,
            int? translationReleaseTick,
            int stopTick,
            G1HeldMask translationHeldMask,
            G1LocalLifecycleSnapshot baseline,
            G1KickAsset requestedAsset,
            IntPtr requestedClipPointer)
        {
            ProbeOrdinal = probeOrdinal;
            Kind = kind;
            ProbeLabel = probeLabel;
            MoveIndex = moveIndex;
            EdgeTick = edgeTick;
            TranslationReleaseTick = translationReleaseTick;
            StopTick = stopTick;
            TranslationHeldMask = translationHeldMask;
            Baseline = baseline;
            RequestedAsset = requestedAsset;
            RequestedClipPointer = requestedClipPointer;
            LastSnapshot = baseline;
            HasLastSnapshot = true;
            LastObservedFixedSubstep = -1;
        }

        internal int ProbeOrdinal { get; }
        internal G1KickProbeKind Kind { get; }
        internal string ProbeLabel { get; }
        internal int MoveIndex { get; }
        internal int EdgeTick { get; }
        internal int? TranslationReleaseTick { get; }
        internal int StopTick { get; }
        internal G1HeldMask TranslationHeldMask { get; }
        internal G1LocalLifecycleSnapshot Baseline { get; }
        internal G1KickAsset RequestedAsset { get; }
        internal IntPtr RequestedClipPointer { get; }
        internal bool HasLastSnapshot { get; set; }
        internal G1LocalLifecycleSnapshot LastSnapshot { get; set; }
        internal bool? ExecuteMoveReturned { get; set; }
        internal string? LocalClassification { get; set; }
        internal bool MoveSendInvoked { get; set; }
        internal bool MoveSendMethodReturned { get; set; }
        internal int? MoveSendInvokedSequence { get; set; }
        internal int? YawNeutralVelocityReturnSequence { get; set; }
        internal int? SendPrefixFixedSubstep { get; set; }
        internal int? SendPrefixScheduleTick { get; set; }
        internal int? SendPrefixUnityFrame { get; set; }
        internal double? SendPrefixUnityFixedTime { get; set; }
        internal long? SendPrefixQpcTicks { get; set; }
        internal int? SendPostfixFixedSubstep { get; set; }
        internal int? SendPostfixScheduleTick { get; set; }
        internal int? SendPostfixUnityFrame { get; set; }
        internal double? SendPostfixUnityFixedTime { get; set; }
        internal long? SendPostfixQpcTicks { get; set; }
        internal int? TranslationReleaseFixedSubstep { get; set; }
        internal int? TranslationReleaseEventSequence { get; set; }
        internal long? TranslationReleaseQpcTicks { get; set; }
        internal G1TranslationSettleSnapshot? TranslationReleaseSample { get; set; }
        internal G1TranslationSettleSnapshot? LastTranslationSettleSample { get; set; }
        internal int? FirstTransitionSettledFixedSubstep { get; set; }
        internal int? ActionBusyEntryFixedSubstep { get; set; }
        internal int? ActionBusyExitFixedSubstep { get; set; }
        internal int? RunnerMotionEntryFixedSubstep { get; set; }
        internal int? RunnerMotionCompletionFixedSubstep { get; set; }
        internal IntPtr EntryMotionPointer { get; set; }
        internal int LastObservedFixedSubstep { get; set; }
        internal int FixedObservations { get; set; }
        internal bool ObservationWindowComplete { get; set; }
        internal bool SummaryEmitted { get; set; }

        internal bool BaselineNeutralSettled =>
            !Baseline.ActionBusy && Baseline.RunnerAvailable &&
            Baseline.RunnerIsRecovering != true && !Baseline.InputIsPunching &&
            !Baseline.InputIsRecovering;

        internal bool YawNeutralizationPrecededMoveSend =>
            Kind != G1KickProbeKind.YawPreempted ||
            YawNeutralVelocityReturnSequence is int velocitySequence &&
            MoveSendInvokedSequence is int moveSequence &&
            velocitySequence < moveSequence;

    }

    private sealed record G1HeldCoverage(
        bool Complete,
        int HeldConditionsExpected,
        int HeldConditionsObservedForExactDuration,
        int[] HeldConditionObservedTicks,
        int TranslationKickProbesExpected,
        int TranslationKickEdgesObserved,
        int TranslationKickTerminalOutcomesObserved,
        int TranslationReleasesObserved,
        int YawKickProbesExpected,
        int YawKickEdgesObserved,
        int YawKickTerminalOutcomesObserved,
        int YawPreemptionsObserved,
        int KickMeasurementSummariesObserved,
        int KickObservationWindowsComplete,
        int KickFixedObservationWindowsComplete,
        object[] KickMeasurements,
        int[] KickMoveIndices,
        bool FBindingIncluded,
        bool QueueOrRetryUsed,
        bool SonicActionComposerLifecycleUsed,
        bool PhysicalBehaviorCertified,
        string PoseResponseSource);
}
