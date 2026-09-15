using System.Diagnostics;
using REKApp;
using UnityEngine;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private bool _g1PolicyRunning;
    private RobotInputController? _g1PolicyInput;
    private TrialRoundIdentity? _g1PolicyIdentity;
    private string? _g1PolicyRound;
    private long _g1PolicySequence, _g1PolicyConsumed, _g1PolicyLastAction;
    private int _g1PolicyDesiredAction = 1, _g1PolicyLastPublishedFrame = -1;
    private long _g1PolicyLastPublishedQpc;
    private bool _g1PolicyReceivedAction;
    private readonly Dictionary<long, long> _g1PolicyObservations = new();
    private G1HeldMask _g1PolicyHeld, _g1PolicyOutgoing;
    private G1KeyboardYawState _g1PolicyYaw = new(0, 0);
    private Vector3 _g1PolicyVelocity;
    private int? _g1PolicyPendingMove;
    private int? _g1PolicyRequestedMove;
    private bool _g1PolicyMoveInFlight, _g1PolicyBusySeen;
    private bool _g1PolicyMoveSendReturned;
    private long _g1PolicyMoveObservation;
    private string? _g1PolicyMoveRequestId;
    private long _g1PolicyMoveAt;

    private bool TryPolicyScope(out PrivateAiContext scope, out string reason, bool active = true)
    {
        scope = null!;
        if (!RequireBackgroundControl(out reason) ||
            !TryVerifyExplicitIsolatedSession(out var proof) ||
            proof != G1HeldInputScheduleContract.RequiredIsolationProof)
        { reason = "exact_isolated_spark_marker_not_proven"; return false; }
        if (!TryGetPrivateAiContext(active, out scope, out reason)) return false;
        var pair = ReadMeasuredPairing(scope.Coordinator, scope.LocalSlot, scope.OpponentSlot);
        if (!pair.Validation.ExactG1VersusG1)
        { reason = "exact_g1_pairing_not_proven"; return false; }
        if (_g1PolicyRunning)
        {
            if (_leaseConnectionId == 0 || _leaseConnectionId != (_pipe?.CurrentConnectionId ?? 0))
            { reason = "policy_lease_lost"; return false; }
            if (active && (!TryCreateTrialRoundIdentity(scope, out var identity, out reason) ||
                identity != _g1PolicyIdentity))
            { reason = "policy_round_identity_changed"; return false; }
        }
        return true;
    }

    private CommandResult StartG1PolicyStream()
    {
        if (_g1PolicyRunning || _scheduleRunning || _singleMotionTrialRunning ||
            _continuousControllerRunning || _g1HeldScheduleRunning ||
            _attackZoneTrialRunning || _attackZoneRecoveryOnlyRunning)
            return CommandResult.Rejected("another_control_mode_already_running");
        if (!SameFloatBits(Time.fixedDeltaTime, 0.002f) || !_sendBoundaryPatchesVerified)
            return CommandResult.Rejected("policy_fixed_rate_or_send_hooks_not_verified");
        if (!TryPolicyScope(out var scope, out var reason)) return CommandResult.Rejected(reason);
        var input = scope.Input!;
        if (input.hasPendingMove || input.hasPendingSpecial || input.hasPendingEStop ||
            input.IsPunching || input.IsRecovering || !VelocityEquals(input.VelocityCommand, Vector3.zero))
            return CommandResult.Rejected("policy_start_requires_neutral_pending_free_controller");
        if (!TryCreateTrialRoundIdentity(scope, out var identity, out reason))
            return CommandResult.Rejected(reason);
        if (!TryValidateG1KickAssetBindings(input, out reason)) return CommandResult.Rejected(reason);
        var moves = input.robotConfig?.moves;
        if (moves is null || moves.Count != 17 || Enumerable.Range(0, 17).Any(i => moves[i] is null))
            return CommandResult.Rejected("g1_17_move_map_not_available");
        _g1PolicyInput = input; _g1PolicyIdentity = identity;
        _g1PolicyRound = HashTrialRoundIdentity(identity);
        _g1PolicyObservations.Clear(); _g1PolicyConsumed = _g1PolicySequence;
        _g1PolicyLastAction = Stopwatch.GetTimestamp(); _g1PolicyReceivedAction = false;
        _g1PolicyLastPublishedQpc = 0; _g1PolicyLastPublishedFrame = -1;
        _g1PolicyHeld = _g1PolicyOutgoing = G1HeldMask.None;
        _g1PolicyDesiredAction = 1; _g1PolicyVelocity = Vector3.zero;
        _g1PolicyYaw = new(0, 0); _g1PolicyPendingMove = null;
        _g1PolicyMoveInFlight = _g1PolicyBusySeen = _g1PolicyMoveSendReturned = false;
        _g1PolicyRequestedMove = null; _g1PolicyMoveAt = 0;
        _freshRoundArm = null; _g1PolicyRunning = true;
        return CommandResult.AppliedResult("g1_policy_stream_started_neutral");
    }

    private CommandResult StopG1PolicyStreamCommand()
    {
        StopG1PolicyStream("requested_stop");
        return CommandResult.AppliedResult("g1_policy_stream_stopped");
    }

    private void StopG1PolicyStream(string reason)
    {
        if (!_g1PolicyRunning) return;
        _g1PolicyRunning = false;
        var neutral = false; var sent = false; var pendingCleared = false;
        try
        {
            var input = _g1PolicyInput;
            if (input is not null && NativePointer(input) == _g1PolicyIdentity?.ControllerPointer)
            {
                if (_g1PolicyPendingMove is int move && input.hasPendingMove && input.pendingMoveIndex == move)
                { input.hasPendingMove = false; pendingCleared = true; }
                neutral = SetVelocityExact(input, Vector3.zero);
                // Only an owned local controller, still in the isolated execution surface.
                var manager = Unity.Netcode.NetworkManager.Singleton?.CustomMessagingManager;
                if (neutral && RequireBackgroundControl(out _) && input.networkInitialized && manager is not null)
                { input.SendVelocityCommand(manager); sent = true; }
            }
        }
        catch (Exception e) { reason += ":neutralization_" + e.GetType().Name; }
        _pipe?.Send(_leaseConnectionId, new {
            @event = "g1_policy_end", protocol = "rek.ui_bridge.v1", schema = G1PolicyStreamContract.Schema,
            round_identity_sha256 = _g1PolicyRound, reason,
            owned_velocity_neutralized = neutral, neutral_send_method_returned = sent,
            owned_pending_move_cleared = pendingCleared, global_input_emitted = false,
            server_acceptance = "unknown", clock = PolicyClock() });
        _g1PolicyInput = null; _g1PolicyIdentity = null; _g1PolicyPendingMove = null;
        _g1PolicyHeld = G1HeldMask.None; _g1PolicyVelocity = Vector3.zero;
        _g1PolicyObservations.Clear();
    }

    private void AdvanceG1PolicyStream()
    {
        if (!TryPolicyScope(out var scope, out var reason))
        {
            PublishG1PolicyState(_leaseConnectionId, null);
            StopG1PolicyStream(reason); return;
        }
        var now = Stopwatch.GetTimestamp();
        var input = scope.Input!;
        if (input.hasPendingSpecial || input.hasPendingEStop ||
            (input.hasPendingMove && input.pendingMoveIndex != _g1PolicyPendingMove))
        { StopG1PolicyStream("unowned_pending_command"); return; }
        var busy = input.IsPunching || input.IsRecovering;
        if (_g1PolicyMoveInFlight)
        {
            if (input.Robot.IsVisualOnly)
            {
                if (G1PolicyStreamContract.VisualTransportComplete(true, _g1PolicyMoveSendReturned, input.hasPendingMove))
                { _g1PolicyMoveInFlight = false; _g1PolicyPendingMove = null; }
                else if (!input.hasPendingMove && !_g1PolicyMoveSendReturned)
                { StopG1PolicyStream("owned_pending_move_lost_without_send_return"); return; }
            }
            else
            {
                _g1PolicyBusySeen |= busy;
                if (_g1PolicyBusySeen && !busy && !input.hasPendingMove)
                    _g1PolicyMoveInFlight = false;
                else if (!_g1PolicyBusySeen && (now - _g1PolicyMoveAt) / (double)Stopwatch.Frequency > 2)
                { StopG1PolicyStream("requested_move_local_lifecycle_not_observed"); return; }
            }
        }
        UpdateG1PolicyVelocity(input, advanceYaw: false);
    }

    private void CheckG1PolicyWatchdog()
    {
        // Update has already consumed queued commands. Fixed-step catch-up must not
        // expire a command that is waiting for its first main-thread dispatch.
        if (_g1PolicyRunning && G1PolicyStreamContract.WatchdogExpired(_g1PolicyReceivedAction,
                _g1PolicyLastAction, Stopwatch.GetTimestamp(), Stopwatch.Frequency))
            StopG1PolicyStream("policy_action_watchdog_expired");
    }

    private void PublishG1PolicyFrame()
    {
        var now = Stopwatch.GetTimestamp();
        var frame = Time.frameCount;
        if (!_g1PolicyRunning || !G1PolicyStreamContract.ShouldPublish(frame,
                _g1PolicyLastPublishedFrame, now, _g1PolicyLastPublishedQpc, Stopwatch.Frequency)) return;
        _g1PolicyLastPublishedFrame = frame; _g1PolicyLastPublishedQpc = now;
        PublishG1PolicyState(_leaseConnectionId, null);
    }

    private void ApplyG1PolicyAction(BridgeRequest request)
    {
        var action = request.PolicyAction!; bool applied = false; bool? moveReturned = null;
        string reason; var now = Stopwatch.GetTimestamp();
        try
        {
            if (!_g1PolicyRunning || request.ConnectionId != _leaseConnectionId)
                reason = "policy_stream_not_owned";
            else if (!TryPolicyScope(out var scope, out reason)) StopG1PolicyStream(reason);
            else
            {
                _g1PolicyObservations.TryGetValue(action.ObservationSequence, out var produced);
                var rejected = G1PolicyStreamContract.RejectReason(action, _g1PolicyRound,
                    _g1PolicyConsumed, produced, now, Stopwatch.Frequency);
                if (rejected is not null) reason = rejected;
                else
                {
                    _g1PolicyConsumed = action.ObservationSequence;
                    _g1PolicyLastAction = now;
                    _g1PolicyReceivedAction = true;
                    var input = scope.Input!;
                    var mask = PolicyMask(scope);
                    if (!mask[action.Action]) reason = "rejected_client_transport_or_command_gate_no_queue";
                    else if (action.Action == 0) { applied = true; reason = "held_state_unchanged"; }
                    else if (action.Action < 16)
                    {
                        var desired = G1PolicyStreamContract.Held[action.Action];
                        var outgoing = G1PolicyStreamContract.Translation(_g1PolicyHeld);
                        if (outgoing != G1HeldMask.None) _g1PolicyOutgoing = outgoing;
                        _g1PolicyHeld = desired; _g1PolicyDesiredAction = action.Action;
                        UpdateG1PolicyVelocity(input, advanceYaw: false);
                        applied = true; reason = "held_state_applied_locally";
                    }
                    else
                    {
                        var move = G1PolicyStreamContract.MoveOrder[action.Action - 16];
                        _g1PolicyVelocity = Vector3.zero; _g1PolicyYaw = new(0, 0);
                        if (!SetVelocityExact(input, Vector3.zero)) throw new InvalidDataException("neutral_readback_failed");
                        _g1PolicyPendingMove = move;
                        _g1PolicyMoveSendReturned = false;
                        _g1PolicyMoveObservation = action.ObservationSequence;
                        _g1PolicyMoveRequestId = request.RequestId;
                        moveReturned = input.ExecuteMoveByIndex(move);
                        if (input.hasPendingMove && input.pendingMoveIndex != move)
                            throw new InvalidDataException("conflicting_pending_move");
                        applied = moveReturned == true;
                        reason = G1HeldInputScheduleContract.ClassifyKickEdge(moveReturned.Value,
                            input.hasPendingMove, input.pendingMoveIndex, move);
                        if (applied || input.hasPendingMove)
                        {
                            _g1PolicyMoveInFlight = true; _g1PolicyBusySeen = input.IsPunching || input.IsRecovering;
                            _g1PolicyMoveAt = now; _g1PolicyRequestedMove = move;
                        }
                        else _g1PolicyPendingMove = null;
                    }
                }
            }
        }
        catch (Exception e) { reason = "policy_action_failed:" + e.GetType().Name; StopG1PolicyStream(reason); }
        _pipe?.Send(request.ConnectionId, new {
            @event = "g1_policy_action", protocol = "rek.ui_bridge.v1", schema = G1PolicyStreamContract.Schema,
            request_id = request.RequestId, round_identity_sha256 = action.RoundIdentity,
            observation_sequence = action.ObservationSequence, action = action.Action,
            applied, reason, execute_move_returned = moveReturned,
            execution_status = "unknown", completion_status = "unknown",
            pending_move_index = _g1PolicyInput?.hasPendingMove == true ? _g1PolicyInput.pendingMoveIndex : (int?)null,
            server_acceptance = "unknown", global_input_emitted = false, clock = PolicyClock() });
    }

    private bool[] PolicyMask(PrivateAiContext scope)
    {
        var input = scope.Input ?? scope.Coordinator.robotInput;
        var mask = new bool[33]; mask[0] = mask[1] = true;
        if (scope.Round?.IsActive != true || scope.Coordinator.CurrentPhase != FightPhase.RoundActive) return mask;
        if (input is null || !input.IsActive || !input.networkInitialized || input.hasPendingSpecial || input.hasPendingEStop)
            return mask;
        var robot = input.Robot;
        if (robot is null || robot.IsFallen || robot.IsResetting || robot.IsMotorShutdown) return mask;
        mask[6] = mask[7] = true;
        if (input.IsPunching || input.IsRecovering || input.hasPendingMove || _g1PolicyMoveInFlight) return mask;
        for (var i = 2; i < 16; i++) mask[i] = true;
        var velocity = input.VelocityCommand;
        if (!Finite(velocity) || velocity.x != 0 || velocity.y != 0) return mask;
        bool settled;
        if (_g1PolicyOutgoing == G1HeldMask.None)
            settled = input.TransitionSettled(LocomotionDir.Forward) && input.TransitionSettled(LocomotionDir.Backward) &&
                input.TransitionSettled(LocomotionDir.StrafeLeft) && input.TransitionSettled(LocomotionDir.StrafeRight);
        else settled = input.TransitionSettled(G1TranslationLocomotionDirection(_g1PolicyOutgoing));
        var moves = input.robotConfig?.moves;
        if (settled && moves is not null)
            for (var i = 16; i < 33; i++)
            { var move = G1PolicyStreamContract.MoveOrder[i - 16]; mask[i] = move < moves.Count && moves[move] is not null; }
        return mask;
    }

    private void UpdateG1PolicyVelocity(RobotInputController input, bool advanceYaw)
    {
        var busy = _g1PolicyMoveInFlight || input.IsPunching || input.IsRecovering;
        var rawYaw = busy ? 0 : G1PolicyStreamContract.Yaw(_g1PolicyHeld);
        if (advanceYaw)
        {
            var config = input.robotConfig ?? throw new InvalidDataException("robot_config_missing");
            var step = G1HeldInputScheduleContract.AdvanceKeyboardYaw(_g1PolicyYaw, rawYaw,
                Time.deltaTime, config.keyboardYawRampTime, config.yawSpeed);
            _g1PolicyYaw = step.State; _g1PolicyVelocity.z = step.EffectiveYaw;
        }
        else if (rawYaw == 0) { _g1PolicyYaw = new(0, 0); _g1PolicyVelocity.z = 0; }
        _g1PolicyVelocity.x = busy ? 0 : G1PolicyStreamContract.Forward(_g1PolicyHeld);
        _g1PolicyVelocity.y = busy ? 0 : G1PolicyStreamContract.Strafe(_g1PolicyHeld);
        if (!SetVelocityExact(input, _g1PolicyVelocity)) throw new InvalidDataException("velocity_readback_mismatch");
    }

    private bool OwnsPolicyInput(RobotInputController input) => _g1PolicyRunning &&
        _g1PolicyInput is not null && NativePointer(input) == NativePointer(_g1PolicyInput);

    private void OnG1PolicyLateUpdate(RobotInputController input)
    {
        try
        {
            if (!TryPolicyScope(out _, out var reason)) { StopG1PolicyStream(reason); return; }
            UpdateG1PolicyVelocity(input, advanceYaw: true);
        }
        catch (Exception e) { StopG1PolicyStream("policy_late_update:" + e.GetType().Name); }
    }

    private bool OnG1PolicyMovePrefix(RobotInputController input)
    {
        if (!TryPolicyScope(out _, out var reason) || _g1PolicyPendingMove is not int move ||
            !input.hasPendingMove || input.pendingMoveIndex != move)
        { StopG1PolicyStream("policy_move_send_guard:" + reason); return false; }
        return true;
    }

    private void OnG1PolicyMovePostfix(RobotInputController input)
    {
        _g1PolicyMoveSendReturned = true;
        _pipe?.Send(_leaseConnectionId, new {
            @event = "g1_policy_dispatch", protocol = "rek.ui_bridge.v1", schema = G1PolicyStreamContract.Schema,
            request_id = _g1PolicyMoveRequestId, observation_sequence = _g1PolicyMoveObservation,
            round_identity_sha256 = _g1PolicyRound, move_index = _g1PolicyPendingMove,
            send_method = "RobotInputController.SendMoveEvent", send_method_returned = true,
            visual_only = input.Robot.IsVisualOnly, execution_status = "unknown", completion_status = "unknown",
            server_acceptance = "unknown", global_input_emitted = false, clock = PolicyClock() });
        if (!input.Robot.IsVisualOnly) _g1PolicyPendingMove = null;
    }

    private void PublishG1PolicyState(long connectionId, string? requestId)
    {
        string? diagnostic = null;
        try
        {
            if (!TryPolicyScope(out var scope, out var reason, active: false))
            {
                diagnostic = reason;
                throw new InvalidDataException(reason);
            }
            var input = scope.Coordinator.robotInput ?? throw new InvalidDataException("input_missing");
            var round = scope.Round ?? throw new InvalidDataException("round_missing");
            var fight = scope.Coordinator.Fight ?? throw new InvalidDataException("fight_missing");
            var fighters = scope.Coordinator.Fighters;
            if (fighters is null || fighters.Length != 2) throw new InvalidDataException("fighters_missing");
            var activeScope = scope with { Input = input };
            if (!TryCreateTrialRoundIdentity(activeScope, out var identity, out reason))
            { diagnostic = reason; throw new InvalidDataException(reason); }
            var hash = HashTrialRoundIdentity(identity);
            var sequence = ++_g1PolicySequence; var now = Stopwatch.GetTimestamp();
            if (_g1PolicyRunning && hash == _g1PolicyRound && connectionId == _leaseConnectionId)
            {
                _g1PolicyObservations[sequence] = now;
                foreach (var old in _g1PolicyObservations.Keys.Where(k => k < sequence - 32).ToArray())
                    _g1PolicyObservations.Remove(old);
            }
            if (!float.IsFinite(round.RoundDuration) || !float.IsFinite(round.TimeRemaining))
                throw new InvalidDataException("round_time_nonfinite");
            _pipe?.Send(connectionId, new {
                @event = "g1_policy_state", protocol = "rek.ui_bridge.v1", schema = G1PolicyStreamContract.Schema,
                request_id = requestId, observation_sequence = sequence, round_identity_sha256 = hash,
                local_slot = scope.LocalSlot, phase = (int)scope.Coordinator.CurrentPhase,
                stream_active = _g1PolicyRunning, authority_scope = "client_replicated_and_local_observations",
                global_input_emitted = false, server_acceptance = "unknown", clock = PolicyClock(now),
                fighters = new[] { PolicyFighter(fighters[0], input), PolicyFighter(fighters[1], input) },
                input = new {
                    active = input.IsActive, punching = input.IsPunching, recovering = input.IsRecovering,
                    action_busy = input.Robot.IsVisualOnly ? (bool?)null : input.IsPunching || input.IsRecovering,
                    action_busy_source = input.Robot.IsVisualOnly ? "unavailable_visual_only_client" : "local_controller_flags",
                    allow_move_interrupt = input.AllowMoveInterrupt,
                    velocity_command_xyz = PolicyVector(input.VelocityCommand), pending_move = input.hasPendingMove,
                    pending_move_index = input.hasPendingMove ? input.pendingMoveIndex : (int?)null,
                    pending_special = input.hasPendingSpecial, pending_estop = input.hasPendingEStop,
                    desired_action = _g1PolicyRunning ? _g1PolicyDesiredAction : (int?)null,
                    requested_move_index = _g1PolicyRunning ? _g1PolicyRequestedMove : null,
                    requested_move_qpc_ticks = _g1PolicyRunning && _g1PolicyRequestedMove is not null ? _g1PolicyMoveAt : (long?)null,
                    move_request_pending_transport = _g1PolicyMoveInFlight,
                    move_send_method_returned = _g1PolicyMoveSendReturned,
                    native_transition_settled = new {
                        forward = input.TransitionSettled(LocomotionDir.Forward),
                        backward = input.TransitionSettled(LocomotionDir.Backward),
                        strafe_left = input.TransitionSettled(LocomotionDir.StrafeLeft),
                        strafe_right = input.TransitionSettled(LocomotionDir.StrafeRight) } },
                round = new { number = round.RoundNumber, duration = round.RoundDuration,
                    time_remaining = round.TimeRemaining, active = round.IsActive, redo = round.IsRedo,
                    clean_hits = round.CleanHits?.ToArray(), falls = round.Falls?.ToArray(),
                    result = round.Result.ToString(), result_value = (int)round.Result,
                    winner_index = round.WinnerIndex, knockout = round.KnockoutOccurred },
                fight = new { current_round = fight.CurrentRoundNumber, rounds_won = fight.RoundsWon?.ToArray(),
                    result = fight.Result.ToString(), result_value = (int)fight.Result, winner_index = fight.WinnerIndex },
                referee = (object?)null, action_mask = PolicyMask(activeScope),
                action_mask_source = "native_client_transport_and_owned_command_gates_server_readiness_unknown" });
        }
        catch (Exception e)
        {
            // Only our fixed diagnostic names and scope decisions may enter the transcript.
            // Interop, IO and arbitrary exception messages are never forwarded.
            if (diagnostic is null && e is InvalidDataException && e.Message is
                "input_missing" or "round_missing" or "fight_missing" or "fighters_missing" or
                "round_time_nonfinite" or "root_missing" or "bones_missing" or "g1_bone_count_mismatch" or
                "bone_missing" or "g1_bone_order_mismatch" or "tilt_nonfinite" or "vector_nonfinite" or "quaternion_nonfinite")
                diagnostic = e.Message;
            _pipe?.Send(connectionId, new { @event = "error", protocol = "rek.ui_bridge.v1", request_id = requestId,
                reason = "policy_state_unavailable:" + (diagnostic ?? e.GetType().Name) });
            if (_g1PolicyRunning) StopG1PolicyStream("policy_state_unavailable:" + e.GetType().Name);
        }
    }

    private static object PolicyFighter(Robot robot, RobotInputController input)
    {
        var root = robot.RootTransform ?? throw new InvalidDataException("root_missing");
        var bones = robot.boneTransforms ?? throw new InvalidDataException("bones_missing");
        if (bones.Length != 30) throw new InvalidDataException("g1_bone_count_mismatch");
        var names = new string[30]; var rotations = new float[30][]; var positions = new float[30][];
        for (var i = 0; i < 30; i++)
        { var bone = bones[i] ?? throw new InvalidDataException("bone_missing"); names[i] = bone.name;
          rotations[i] = PolicyQuaternion(bone.localRotation); positions[i] = PolicyVector(bone.position); }
        if (!BridgePairingContract.IsExactG1BoneSignature(names)) throw new InvalidDataException("g1_bone_order_mismatch");
        var baseAvailable = robot.TryGetBaseVelocityLocal(out var angular, out var linear) && Finite(angular) && Finite(linear);
        var runner = robot.GetComponent<SonicPolicyRunner>();
        var motion = runner?.currentMotion;
        if (!float.IsFinite(robot.TiltAngle)) throw new InvalidDataException("tilt_nonfinite");
        return new {
            root_position_xyz = PolicyVector(root.position), root_rotation_xyzw = PolicyQuaternion(root.rotation),
            root_linear_velocity_xyz = PolicyVector(robot.RootLinearVelocity),
            root_angular_velocity_xyz = PolicyVector(robot.RootAngularVelocity),
            visual_only = robot.IsVisualOnly, player_controlled = robot.IsPlayerControlled,
            falling = robot.IsFalling, fallen = robot.IsFallen, dampened = robot.IsDampened,
            resetting = robot.IsResetting, motor_shutdown = robot.IsMotorShutdown,
            tilt_angle = robot.TiltAngle, floor_contact_count = robot.FloorContactCount,
            bone_names = names, bone_local_rotations_xyzw = rotations, bone_world_positions_xyz = positions,
            base_linear_velocity_local_xyz = baseAvailable ? PolicyVector(linear) : null,
            base_angular_velocity_local_xyz = baseAvailable ? PolicyVector(angular) : null,
            joint_positions = (object?)null, joint_velocities = (object?)null, last_hit = (object?)null,
            runner = new { available = runner is not null, is_done = runner?.IsDone,
                is_recovering = runner?.IsRecovering, current_motion_name = (string?)null,
                motion_frame_index = runner is not null ? runner.motionFrameIdx : (int?)null,
                current_move_index = (int?)null,
                motion_identity_status = "runtime_MotionSequence_has_no_proven_clip_identity" } };
    }

    private static float[] PolicyVector(Vector3 value) => Finite(value)
        ? new[] { value.x, value.y, value.z } : throw new InvalidDataException("vector_nonfinite");
    private static float[] PolicyQuaternion(Quaternion value) => Finite(value)
        ? new[] { value.x, value.y, value.z, value.w } : throw new InvalidDataException("quaternion_nonfinite");
    private static object PolicyClock(long? qpc = null) => new { utc = DateTimeOffset.UtcNow,
        unity_frame = Time.frameCount, unity_time = Time.timeAsDouble, unity_fixed_time = Time.fixedTimeAsDouble,
        qpc_ticks = qpc ?? Stopwatch.GetTimestamp(), qpc_frequency_hz = Stopwatch.Frequency };
}
