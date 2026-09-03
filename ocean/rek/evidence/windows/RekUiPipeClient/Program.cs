using System.IO.Pipes;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

const string PipeName = "rek-ui-bridge-v1";
const string Protocol = "rek.ui_bridge.v1";
const string ExpectedApplicationVersion = "0.0.119";
const string ExpectedUnityVersion = "6000.5.8f1";
const string ExpectedGameAssemblySha256 =
    "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
const string ExpectedMetadataSha256 =
    "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
const string ExpectedBridgeVersion = "0.4.0";
const string ExpectedBridgeSha256 =
    "fb9e3c0a4994eafc6a45f83a32c907f5eee5f9a4d6997d5bf10d863f27faab55";
const string ScheduleId = "rek.private_bot1.baseline.v1";
const string ScheduleSchema = "rek.client_fixed.command_schedule.v2";
const string ScheduleSha256 =
    "39aaab9c3156e8f4d114daac4d4328257b81230ec8b8a372ad2739d38754ec0d";
const string T800BoneSignatureSha256 =
    "ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab";
const string MarkerSchema = "rek.rendered_command_marker.v1";
const string MarkerRenderBinding =
    "first_post_marker_frame_is_first_rendered_frame_after_command_edge";
const string MarkerTransition = "persistent_exact_rgb_rising_edge";
const int ExpectedMarkerCount = 24;
const string TrialSchema = "rek.single_motion_trial.v1";
const string TrialSha256 =
    "f00348f6f10fa706d5e48e8f31a0cbbdee1512564f819c82d7525637d68de99b";
const string TrialAuthorityScope = "client_request_edges_only";
const string TrialAuthorityCaveat =
    "client request edge observed; server acceptance and authoritative execution are unknown";
const string ContinuousSchema = "rek.continuous_private_bot_controller.v1";
const string ContinuousSha256 =
    "c19ee1cc02111426db7a58cd648e244e1106842d86caaba3dc729edf4640b92e";
const string ContinuousAuthorityScope =
    "client_request_edges_and_local_observations_only";
const string ContinuousAuthorityCaveat =
    "client request edge and local motion lifecycle observations only; server acceptance and authoritative execution are unknown";
const string ContinuousRangeAngleProvenance =
    "build_pinned_baseline_ai_global_thresholds_projected_per_move_not_runtime_calibrated";
const string ContinuousFacingYawProvenance =
    "build_pinned_AIOpponentController.ComputeFacingYaw_rva_0x2366e20_AngleToOpponent_rva_0x2366600_half_threshold_deadband_abs_angle_over_45_clamped_times_engage_yaw_speed_negative_bearing_sign";
const string ContinuousAttackSelectionProvenance =
    "audit_controller_deterministic_round_robin_diverges_from_build_pinned_AIOpponentController_random_category_and_clip_selection";
const string ContinuousStaticImpactTimingProvenance =
    "build_pinned_serialized_t800_move_asset_metadata_not_measured_runtime_timing";
const string ContinuousRoundRestartLimitation =
    "build_pinned_post_fight_continue_restarts_only_after_win_and_exits_to_lobby_after_loss";
const string ContinuousRoundRestartStaticEvidence =
    "GameMenuController.HandlePostFightContinue_rva_0x23aae90_branches_on_postFightIsWinner_false_ExitToLobby_true_SendPostFightIntent_stay_true";
const string ContinuousRecoveryGuardProvenance =
    "build_pinned_AIOpponentController.DriveRecovery_rva_0x2367430_fallen_not_dampened_Dampen_4_then_Straighten_1_once_then_RecoveryArmed_SuggestedGetUpOrientation_2_or_3";
const string ContinuousFaultEStopProvenance =
    "build_pinned_AIOpponentController.UpdateFaultEStopCycle_rva_0x23680e0_motorShutdownHold_faultEStopDelay_then_0.5_second_estop_hold";
const string ContinuousDampenGuard = "fallen_and_not_dampened";
const string ContinuousStraightenGuard =
    "fallen_and_dampened_and_not_already_issued";
const string ContinuousOpponentRuntimeRequirement =
    "exact_t800_runtime_bone_signature_required_semantic_robot_id_recorded_but_not_trusted";
const string AttackZoneSchema = "rek.attack_zone_trial.v1";
const string AttackZoneSha256 =
    "1c55900c766aac8cf3382c389b297be6324b3ca19c4a5de6d25f17a7ee217278";
const string AttackZoneAuthorityScope =
    "client_request_edges_and_local_observations_only";
const string AttackZoneAuthorityCaveat =
    "client request edge and local observations only; server acceptance, authoritative execution, and causal hit attribution are unknown";
const string AttackZoneIsolationProof =
    "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98";
const string AttackZoneRecorderVersion = "0.6.1";
const string AttackZoneRecorderSha256 =
    "24cbea0a149589b71c093e989f43b8dac4862e73d103c323f0f9472a38355e0b";
const int TrialFixedSubstepsPerTick = 10;
const int TrialNeutralPreRollTicks = 50;
const int TrialActionTick = 50;
const int TrialLocomotionReleaseTick = 100;
const int TrialDurationTicks = 250;
const int TrialFinalTick = TrialDurationTicks - 1;

var expectedSteps = new[]
{
    new ExpectedStep(0, "neutral", 0f, 0f, 0f, null),
    new ExpectedStep(50, "forward_1", 1f, 0f, 0f, null),
    new ExpectedStep(150, "neutral", 0f, 0f, 0f, null),
    new ExpectedStep(200, "backward_1", -1f, 0f, 0f, null),
    new ExpectedStep(300, "neutral", 0f, 0f, 0f, null),
    new ExpectedStep(350, "strafe_left_1", 0f, -1f, 0f, null),
    new ExpectedStep(450, "neutral", 0f, 0f, 0f, null),
    new ExpectedStep(500, "strafe_right_1", 0f, 1f, 0f, null),
    new ExpectedStep(600, "neutral", 0f, 0f, 0f, null),
    new ExpectedStep(650, "yaw_left_1", 0f, 0f, -1f, null),
    new ExpectedStep(750, "neutral", 0f, 0f, 0f, null),
    new ExpectedStep(800, "yaw_right_1", 0f, 0f, 1f, null),
    new ExpectedStep(900, "move_2_punch_combo", 0f, 0f, 0f, 2),
    new ExpectedStep(1100, "move_3_right_kick", 0f, 0f, 0f, 3),
    new ExpectedStep(1300, "move_4_left_punch", 0f, 0f, 0f, 4),
    new ExpectedStep(1500, "move_5_right_punch", 0f, 0f, 0f, 5),
    new ExpectedStep(1700, "move_9_right_shoryuken_lm_dragon_punch", 0f, 0f, 0f, 9),
    new ExpectedStep(1900, "move_10_front_kick_L_left_kick", 0f, 0f, 0f, 10),
    new ExpectedStep(2100, "forward_1_move_2", 1f, 0f, 0f, 2),
    new ExpectedStep(2300, "neutral", 0f, 0f, 0f, null),
    new ExpectedStep(2400, "backward_1_move_3", -1f, 0f, 0f, 3),
    new ExpectedStep(2600, "neutral_complete", 0f, 0f, 0f, null),
};

var expectedMarkers = new[]
{
    new ExpectedMarker(0, 50, "walk_forward.press.1", "walk_forward:press:v1"),
    new ExpectedMarker(1, 150, "walk_forward.release.1", "walk_forward:release:v1"),
    new ExpectedMarker(2, 200, "walk_backward.press.1", "walk_backward:press:v1"),
    new ExpectedMarker(3, 300, "walk_backward.release.1", "walk_backward:release:v1"),
    new ExpectedMarker(4, 350, "strafe_left.press.1", "strafe_left:press:v1"),
    new ExpectedMarker(5, 450, "strafe_left.release.1", "strafe_left:release:v1"),
    new ExpectedMarker(6, 500, "strafe_right.press.1", "strafe_right:press:v1"),
    new ExpectedMarker(7, 600, "strafe_right.release.1", "strafe_right:release:v1"),
    new ExpectedMarker(8, 650, "yaw_left.press.1", "yaw_left:press:v1"),
    new ExpectedMarker(9, 750, "yaw_left.release.1", "yaw_left:release:v1"),
    new ExpectedMarker(10, 800, "yaw_right.press.1", "yaw_right:press:v1"),
    new ExpectedMarker(11, 900, "yaw_right.release.1", "yaw_right:release:v1"),
    new ExpectedMarker(12, 900, "move_index_2.press.1", "move_index_2:press:v1"),
    new ExpectedMarker(13, 1100, "move_index_3.press.1", "move_index_3:press:v1"),
    new ExpectedMarker(14, 1300, "move_index_4.press.1", "move_index_4:press:v1"),
    new ExpectedMarker(15, 1500, "move_index_5.press.1", "move_index_5:press:v1"),
    new ExpectedMarker(16, 1700, "move_index_9.press.1", "move_index_9:press:v1"),
    new ExpectedMarker(17, 1900, "move_index_10.press.1", "move_index_10:press:v1"),
    new ExpectedMarker(18, 2100, "walk_forward.press.2", "walk_forward:press:v1"),
    new ExpectedMarker(19, 2100, "move_index_2.press.2", "move_index_2:press:v1"),
    new ExpectedMarker(20, 2300, "walk_forward.release.2", "walk_forward:release:v1"),
    new ExpectedMarker(21, 2400, "walk_backward.press.2", "walk_backward:press:v1"),
    new ExpectedMarker(22, 2400, "move_index_3.press.2", "move_index_3:press:v1"),
    new ExpectedMarker(23, 2600, "walk_backward.release.2", "walk_backward:release:v1"),
};

var expectedTrialSelectors = new[]
{
    new ExpectedTrialSelector("forward", "locomotion", 1f, 0f, 0f, null,
        "RobotInputController.VelocityCommand:[1,0,0]"),
    new ExpectedTrialSelector("backward", "locomotion", -1f, 0f, 0f, null,
        "RobotInputController.VelocityCommand:[-1,0,0]"),
    new ExpectedTrialSelector("strafe-left", "locomotion", 0f, 1f, 0f, null,
        "RobotInputController.VelocityCommand:[0,1,0]"),
    new ExpectedTrialSelector("strafe-right", "locomotion", 0f, -1f, 0f, null,
        "RobotInputController.VelocityCommand:[0,-1,0]"),
    new ExpectedTrialSelector("yaw-left", "locomotion", 0f, 0f, 1f, null,
        "RobotInputController.VelocityCommand:[0,0,1]"),
    new ExpectedTrialSelector("yaw-right", "locomotion", 0f, 0f, -1f, null,
        "RobotInputController.VelocityCommand:[0,0,-1]"),
    new ExpectedTrialSelector("move-2", "move", 0f, 0f, 0f, 2,
        "RobotInputController.ExecuteMoveByIndex:2"),
    new ExpectedTrialSelector("move-3", "move", 0f, 0f, 0f, 3,
        "RobotInputController.ExecuteMoveByIndex:3"),
    new ExpectedTrialSelector("move-4", "move", 0f, 0f, 0f, 4,
        "RobotInputController.ExecuteMoveByIndex:4"),
    new ExpectedTrialSelector("move-5", "move", 0f, 0f, 0f, 5,
        "RobotInputController.ExecuteMoveByIndex:5"),
    new ExpectedTrialSelector("move-9", "move", 0f, 0f, 0f, 9,
        "RobotInputController.ExecuteMoveByIndex:9"),
    new ExpectedTrialSelector("move-10", "move", 0f, 0f, 0f, 10,
        "RobotInputController.ExecuteMoveByIndex:10"),
};

if (args.Length < 1 ||
    args[0] is not ("state" or "enter-private" or "exit-lost" or "schedule" or "trial" or "controller"))
{
    Console.Error.WriteLine(
        "usage: RekUiPipeClient state [output.jsonl] [timeout_seconds] | " +
        "enter-private|exit-lost|schedule output.jsonl [timeout_seconds] | " +
        "controller output.jsonl [run_seconds|until-ended] | " +
        "trial selector output.jsonl [timeout_seconds]");
    return 2;
}

var mode = args[0];
string? trialSelectorName = null;
string? outputPath;
string? timeoutArgument;
if (mode == "trial")
{
    if (args.Length is < 3 or > 4)
    {
        Console.Error.WriteLine("trial requires selector output.jsonl [timeout_seconds]");
        return 2;
    }
    trialSelectorName = args[1];
    outputPath = args[2];
    timeoutArgument = args.Length == 4 ? args[3] : null;
    if (!expectedTrialSelectors.Any(value =>
            string.Equals(value.Selector, trialSelectorName, StringComparison.Ordinal)))
    {
        Console.Error.WriteLine(
            $"selector must be one of: {string.Join(", ", expectedTrialSelectors.Select(value => value.Selector))}");
        return 2;
    }
}
else
{
    if (args.Length is < 1 or > 3)
    {
        Console.Error.WriteLine("invalid argument count");
        return 2;
    }
    outputPath = args.Length >= 2 ? args[1] : null;
    timeoutArgument = args.Length == 3 ? args[2] : null;
}
if (mode != "state" && string.IsNullOrWhiteSpace(outputPath))
{
    Console.Error.WriteLine($"{mode} requires a new output.jsonl transcript path");
    return 2;
}
var controllerRunMode = default(ControllerRunModeSpec);
if (mode == "controller" &&
    !ControllerRunModeContract.TryParse(timeoutArgument, out controllerRunMode))
{
    Console.Error.WriteLine(
        "run_seconds must be an integer between 1 and 600 or until-ended");
    return 2;
}
var controllerUntilEnded = mode == "controller" && controllerRunMode.UntilEnded;
if (mode != "controller" && timeoutArgument is not null &&
    !int.TryParse(timeoutArgument, out _))
{
    Console.Error.WriteLine("timeout_seconds must be an integer between 1 and 600");
    return 2;
}
var timeoutSeconds = mode == "controller"
    ? controllerRunMode.RunSeconds
    : timeoutArgument is not null
        ? int.Parse(timeoutArgument)
    : mode == "schedule" ? 90 : mode is "enter-private" or "exit-lost" or "trial" ? 60 :
        15;
if (!controllerUntilEnded && timeoutSeconds is < 1 or > 600)
{
    Console.Error.WriteLine("timeout_seconds must be between 1 and 600");
    return 2;
}

StreamWriter? transcript = null;
string? transcriptFinalPath = null;
string? transcriptPartialPath = null;
if (outputPath is not null)
{
    var fullPath = Path.GetFullPath(outputPath);
    if (File.Exists(fullPath))
        throw new IOException($"transcript destination already exists: {fullPath}");
    Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
    transcriptFinalPath = fullPath;
    transcriptPartialPath = $"{fullPath}.partial-{Environment.ProcessId}-{Guid.NewGuid():N}";
    transcript = new StreamWriter(
        new FileStream(transcriptPartialPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read),
        new UTF8Encoding(false))
    {
        AutoFlush = true,
        NewLine = "\n",
    };
}

var operationTimeoutSeconds = mode == "controller" && !controllerUntilEnded
    ? Math.Min(timeoutSeconds + 30, 630)
    : timeoutSeconds;
using var deadline = controllerUntilEnded
    ? new CancellationTokenSource()
    : new CancellationTokenSource(TimeSpan.FromSeconds(operationTimeoutSeconds));
using var externalControllerStop = new CancellationTokenSource();
ConsoleCancelEventHandler? cancelHandler = null;
if (controllerUntilEnded)
{
    cancelHandler = (_, eventArgs) =>
    {
        eventArgs.Cancel = true;
        externalControllerStop.Cancel();
        deadline.Cancel();
    };
    Console.CancelKeyPress += cancelHandler;
}
await using var pipe = new NamedPipeClientStream(
    ".",
    PipeName,
    PipeDirection.InOut,
    PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
StreamReader? reader = null;
StreamWriter? writer = null;
var leaseHeld = false;
var connectionId = 0L;

try
{
    await pipe.ConnectAsync(deadline.Token);
    var pipeServer = ValidatePipeServer(pipe);
    reader = new StreamReader(pipe, Encoding.UTF8, false, 65_536, leaveOpen: true);
    writer = new StreamWriter(pipe, new UTF8Encoding(false), 65_536, leaveOpen: true)
    {
        AutoFlush = true,
        NewLine = "\n",
    };
    if (transcript is not null)
    {
        await transcript.WriteLineAsync(JsonSerializer.Serialize(new
        {
            @event = "client_pipe_server_proof",
            observed_utc = DateTimeOffset.UtcNow,
            process_id = pipeServer.ProcessId,
            executable = pipeServer.Executable,
        }));
    }

    using (var hello = await ReadUntilHello(deadline.Token))
    {
        ValidateHello(hello.RootElement, expectedTrialSelectors);
        connectionId = RequireInt64Value(hello.RootElement, "connection_id");
        if (connectionId <= 0)
            throw new InvalidDataException("hello connection_id was not positive");
    }

    string resultJson;
    string? completedScheduleRunId = null;
    string? completedTrialRunId = null;
    string? completedTrialRoundIdentitySha256 = null;
    string? completedTrialInitialStateSha256 = null;
    string? completedControllerRunId = null;
    if (mode == "state")
    {
        using var state = await RequestState(deadline.Token);
        ValidatePinnedState(state.RootElement, requireLease: false, connectionId);
        resultJson = state.RootElement.GetRawText();
    }
    else
    {
        using (var preflight = await RequestState(deadline.Token))
            ValidatePinnedState(preflight.RootElement, requireLease: false, connectionId);
        using (var acquire = await RequireAcceptedCommand(
                   "AcquireExclusiveControl",
                   "exclusive_control_lease_acquired",
                   expectedApplied: true,
                   expectedRequestIssued: false,
                   deadline.Token))
        {
            RequireInt64(acquire.RootElement, "lease_connection_id", connectionId);
        }
        leaseHeld = true;

        if (mode == "exit-lost")
        {
            using (var state = await RequestState(deadline.Token))
            {
                ValidatePinnedState(state.RootElement, requireLease: true, connectionId);
                ValidatePrivateBotOne(state.RootElement, requireActiveRound: false);
                var privateAi = state.RootElement.GetProperty("private_ai");
                var control = state.RootElement.GetProperty("control");
                var decision = LostSessionExitModeContract.Evaluate(
                    exactPrivateBotOneProven:
                        RequireBooleanValue(privateAi, "proven") &&
                        RequireBooleanValue(privateAi, "exact_sparring_bot_1"),
                    roundActive: RequireBooleanValue(privateAi, "round_active"),
                    postFightPrompt: RequireBooleanValue(privateAi, "post_fight_prompt"),
                    postFightWinner:
                        RequireNullableBooleanValue(privateAi, "post_fight_is_winner") ??
                        throw new InvalidDataException("post_fight_is_winner was unavailable"),
                    scheduleRunning: RequireBooleanValue(control, "schedule_running"),
                    singleTrialRunning:
                        RequireBooleanValue(control, "single_motion_trial_running"),
                    continuousControllerRunning:
                        RequireBooleanValue(control, "continuous_controller_running"),
                    attackZoneTrialRunning:
                        RequireBooleanValue(control, "attack_zone_trial_running"),
                    attackZoneRecoveryOnlyRunning:
                        RequireBooleanValue(control, "attack_zone_recovery_only_running"));
                if (!decision.Allowed)
                    throw new InvalidDataException(decision.Reason);
            }

            using (var exit = await RequireAcceptedCommand(
                       "ExitLostPrivateSession",
                       "post_fight_loser_exit_request_issued",
                       expectedApplied: false,
                       expectedRequestIssued: true,
                       deadline.Token))
            {
                RequireInt64(exit.RootElement, "lease_connection_id", connectionId);
            }
            while (true)
            {
                await Task.Delay(50, deadline.Token);
                using var state = await RequestState(deadline.Token);
                ValidatePinnedState(state.RootElement, requireLease: true, connectionId);
                if (RequireNonemptyString(state.RootElement, "scene") != "Lobby")
                    continue;
                var privateAi = state.RootElement.GetProperty("private_ai");
                RequireFalse(privateAi, "proven");
                var control = state.RootElement.GetProperty("control");
                RequireFalse(control, "schedule_running");
                RequireFalse(control, "single_motion_trial_running");
                RequireFalse(control, "continuous_controller_running");
                RequireFalse(control, "attack_zone_trial_running");
                RequireFalse(control, "attack_zone_recovery_only_running");
                resultJson = state.RootElement.GetRawText();
                break;
            }
        }
        else if (mode == "enter-private")
        {
            while (true)
            {
                using var state = await RequestState(deadline.Token);
                ValidatePinnedState(state.RootElement, requireLease: true, connectionId);
                RequireString(state.RootElement, "scene", "Lobby");
                var screen = RequireNonemptyString(state.RootElement, "lobby_screen");
                if (screen == "FreePlay")
                    break;
                if (screen == "Intro")
                {
                    await Task.Delay(250, deadline.Token);
                    continue;
                }
                if (screen == "Login")
                {
                    using var login = await RequireAcceptedCommand(
                        "ConfirmLoggedIn",
                        "home_screen_observed_after_lets_go",
                        expectedApplied: true,
                        expectedRequestIssued: false,
                        deadline.Token);
                    continue;
                }
                if (screen == "Home")
                {
                    using var freePlay = await RequireAcceptedCommand(
                        "NavigateFreePlay",
                        "free_play_screen_observed",
                        expectedApplied: true,
                        expectedRequestIssued: false,
                        deadline.Token);
                    continue;
                }
                throw new InvalidDataException(
                    $"private-practice route does not accept lobby screen {screen}");
            }

            using var request = await RequireAcceptedCommand(
                "EnterSolo",
                "private_practice_reservation_requested",
                expectedApplied: false,
                expectedRequestIssued: true,
                deadline.Token);
            RequireInt64(request.RootElement, "lease_connection_id", connectionId);

            while (true)
            {
                await Task.Delay(250, deadline.Token);
                using var observed = await RequestState(deadline.Token);
                ValidatePinnedState(observed.RootElement, requireLease: true, connectionId);
                if (!OptionalTrue(observed.RootElement.GetProperty("private_ai"), "proven"))
                    continue;
                ValidatePrivateBotOne(observed.RootElement, requireActiveRound: false);
                resultJson = observed.RootElement.GetRawText();
                break;
            }
        }
        else if (mode == "schedule")
        {
            await EnsureActivePrivateRound(connectionId, deadline.Token);

            string scheduleRunId;
            using (var start = await RequireAcceptedCommand(
                       "StartMeasuredSchedule",
                       "measured_schedule_started",
                       expectedApplied: true,
                       expectedRequestIssued: false,
                       deadline.Token))
            {
                RequireInt64(start.RootElement, "lease_connection_id", connectionId);
                ValidateScheduleIdentity(start.RootElement);
                scheduleRunId = RequireNonemptyString(start.RootElement, "schedule_run_id");
                if (scheduleRunId.Length != 32 ||
                    scheduleRunId.Any(character => !Uri.IsHexDigit(character)))
                {
                    throw new InvalidDataException("schedule_run_id was not 32 hexadecimal characters");
                }
                ValidateMeasuredPairing(start.RootElement.GetProperty("measured_pairing"));
                completedScheduleRunId = scheduleRunId;
            }

            var nextStep = 0;
            var nextMarker = 0;
            while (true)
            {
                using var message = await ReadMessage(deadline.Token);
                var eventName = OptionalString(message.RootElement, "event");
                if (eventName == "schedule_step")
                {
                    if (nextStep >= expectedSteps.Length)
                        throw new InvalidDataException("received an extra schedule_step");
                    ValidateScheduleStep(message.RootElement, scheduleRunId, expectedSteps[nextStep]);
                    nextStep++;
                    continue;
                }
                if (eventName == "rendered_command_marker_edge")
                {
                    if (nextMarker >= expectedMarkers.Length)
                        throw new InvalidDataException("received an extra rendered_command_marker_edge");
                    ValidateRenderedMarker(
                        message.RootElement,
                        scheduleRunId,
                        expectedMarkers[nextMarker]);
                    nextMarker++;
                    continue;
                }
                if (eventName != "schedule_end")
                    continue;

                if (nextStep != expectedSteps.Length)
                    throw new InvalidDataException(
                        $"schedule ended after {nextStep} of {expectedSteps.Length} expected steps");
                if (nextMarker != expectedMarkers.Length)
                    throw new InvalidDataException(
                        $"schedule ended after {nextMarker} of {expectedMarkers.Length} expected rendered markers");
                ValidateScheduleEnd(message.RootElement, scheduleRunId);
                resultJson = message.RootElement.GetRawText();
                break;
            }
        }
        else if (mode == "controller")
        {
            await EnsureActivePrivateRound(
                connectionId,
                deadline.Token,
                requireStrictParityPairing: false);
            using (var start = await RequireAcceptedCommand(
                       "StartContinuousBotController",
                       "continuous_private_bot_controller_started",
                       expectedApplied: true,
                       expectedRequestIssued: false,
                       deadline.Token))
            {
                RequireInt64(start.RootElement, "lease_connection_id", connectionId);
                RequireTrue(start.RootElement, "continuous_controller_running");
                RequireFalse(start.RootElement, "server_acceptance_observed");
                RequireFalse(start.RootElement, "authoritative_execution_observed");
                completedControllerRunId = RequireHexString(
                    start.RootElement,
                    "continuous_controller_run_id",
                    32);
                _ = RequireHexString(
                    start.RootElement,
                    "continuous_controller_round_identity_sha256",
                    64);
                ValidateContinuousMeasuredPairing(
                    start.RootElement.GetProperty("measured_pairing"));
                resultJson = start.RootElement.GetRawText();
            }

            if (controllerUntilEnded)
            {
                try
                {
                    while (true)
                    {
                        using var message = await ReadMessage(externalControllerStop.Token);
                        var eventName = OptionalString(message.RootElement, "event");
                        if (eventName == "continuous_controller_end")
                        {
                            ValidateContinuousEnd(
                                message.RootElement,
                                completedControllerRunId,
                                expectedReason: null);
                            resultJson = message.RootElement.GetRawText();
                            break;
                        }
                        if (eventName?.StartsWith("continuous_", StringComparison.Ordinal) == true)
                        {
                            ValidateContinuousEvent(
                                message.RootElement,
                                completedControllerRunId);
                            resultJson = message.RootElement.GetRawText();
                        }
                    }
                }
                catch (OperationCanceledException) when (
                    externalControllerStop.IsCancellationRequested)
                {
                    using var stopDeadline = new CancellationTokenSource(
                        TimeSpan.FromSeconds(30));
                    using var stop = await RequireAcceptedControllerStop(
                        completedControllerRunId,
                        stopDeadline.Token);
                    RequireFalse(stop.RootElement, "continuous_controller_running");
                    resultJson = stop.RootElement.GetRawText();
                }
            }
            else
            {
                using (var runWindow = new CancellationTokenSource(
                           TimeSpan.FromSeconds(timeoutSeconds)))
                using (var runStop = CancellationTokenSource.CreateLinkedTokenSource(
                           deadline.Token,
                           runWindow.Token))
                {
                    try
                    {
                        while (true)
                        {
                            using var message = await ReadMessage(runStop.Token);
                            var eventName = OptionalString(message.RootElement, "event");
                            if (eventName == "continuous_controller_end")
                            {
                                ValidateContinuousEnd(
                                    message.RootElement,
                                    completedControllerRunId,
                                    expectedReason: null);
                                throw new InvalidDataException(
                                    $"continuous controller ended before requested duration: " +
                                    RequireNonemptyString(message.RootElement, "reason"));
                            }
                            if (eventName?.StartsWith("continuous_", StringComparison.Ordinal) == true)
                            {
                                ValidateContinuousEvent(
                                    message.RootElement,
                                    completedControllerRunId);
                                resultJson = message.RootElement.GetRawText();
                            }
                        }
                    }
                    catch (OperationCanceledException) when (
                        runWindow.IsCancellationRequested && !deadline.IsCancellationRequested)
                    {
                    }
                }

                using var stop = await RequireAcceptedControllerStop(
                    completedControllerRunId,
                    deadline.Token);
                RequireFalse(stop.RootElement, "continuous_controller_running");
                RequireString(
                    stop.RootElement,
                    "continuous_controller_run_id",
                    completedControllerRunId);
                resultJson = stop.RootElement.GetRawText();
            }
        }
        else
        {
            var selector = expectedTrialSelectors.Single(value =>
                string.Equals(value.Selector, trialSelectorName, StringComparison.Ordinal));
            var freshRoundRequestId = await StartFreshPrivateRound(connectionId, deadline.Token);

            string trialRunId;
            string roundIdentitySha256;
            string initialStateSha256;
            using (var start = await RequireAcceptedCommand(
                       "StartSingleMotionTrial",
                       "single_motion_trial_started",
                       expectedApplied: true,
                       expectedRequestIssued: false,
                       deadline.Token,
                       selector.Selector))
            {
                RequireInt64(start.RootElement, "lease_connection_id", connectionId);
                ValidateTrialIdentity(start.RootElement);
                RequireString(start.RootElement, "single_motion_trial_selector", selector.Selector);
                RequireString(start.RootElement, "fresh_round_request_id", freshRoundRequestId);
                RequireFalse(start.RootElement, "fresh_round_armed");
                RequireNull(start.RootElement, "fresh_round_invalid_reason");
                RequireFalse(start.RootElement, "authoritative_execution_observed");
                trialRunId = RequireHexString(
                    start.RootElement,
                    "single_motion_trial_run_id",
                    32);
                roundIdentitySha256 = RequireHexString(
                    start.RootElement,
                    "single_motion_trial_round_identity_sha256",
                    64);
                initialStateSha256 = RequireHexString(
                    start.RootElement,
                    "single_motion_trial_initial_state_sha256",
                    64);
                ValidateMeasuredPairing(start.RootElement.GetProperty("measured_pairing"));
                ValidateInitialState(
                    start.RootElement.GetProperty("single_motion_trial_initial_state"),
                    roundIdentitySha256,
                    initialStateSha256);
                completedTrialRunId = trialRunId;
                completedTrialRoundIdentitySha256 = roundIdentitySha256;
                completedTrialInitialStateSha256 = initialStateSha256;
            }

            var expectedEdge = 0;
            var expectedRequest = 0;
            var edgePhases = selector.IsLocomotion
                ? new[] { "action", "release" }
                : new[] { "action" };
            var requestPhases = selector.IsLocomotion
                ? new[] { "neutral_pre_roll", "action", "release" }
                : new[] { "neutral_pre_roll", "action" };
            while (true)
            {
                using var message = await ReadMessage(deadline.Token);
                var eventName = OptionalString(message.RootElement, "event");
                if (eventName == "single_motion_trial_command_edge")
                {
                    if (expectedEdge >= edgePhases.Length)
                        throw new InvalidDataException("received an extra single-motion command edge");
                    ValidateTrialCommandEdge(
                        message.RootElement,
                        trialRunId,
                        freshRoundRequestId,
                        selector,
                        roundIdentitySha256,
                        initialStateSha256,
                        edgePhases[expectedEdge]);
                    expectedEdge++;
                    continue;
                }
                if (eventName == "single_motion_trial_client_request")
                {
                    if (expectedRequest >= requestPhases.Length)
                        throw new InvalidDataException("received an extra single-motion client request");
                    ValidateTrialClientRequest(
                        message.RootElement,
                        trialRunId,
                        freshRoundRequestId,
                        selector,
                        roundIdentitySha256,
                        initialStateSha256,
                        requestPhases[expectedRequest]);
                    expectedRequest++;
                    continue;
                }
                if (eventName != "single_motion_trial_end")
                    continue;

                if (expectedEdge != edgePhases.Length)
                {
                    throw new InvalidDataException(
                        $"trial ended after {expectedEdge} of {edgePhases.Length} command edges");
                }
                if (expectedRequest != requestPhases.Length)
                {
                    throw new InvalidDataException(
                        $"trial ended after {expectedRequest} of {requestPhases.Length} client requests");
                }
                ValidateTrialEnd(
                    message.RootElement,
                    trialRunId,
                    freshRoundRequestId,
                    roundIdentitySha256,
                    initialStateSha256,
                    selector);
                resultJson = message.RootElement.GetRawText();
                break;
            }
        }

        await ReleaseLease();
        leaseHeld = false;
        using var releasedStateDeadline = controllerUntilEnded
            ? new CancellationTokenSource(TimeSpan.FromSeconds(5))
            : null;
        using var releasedState = await RequestState(
            releasedStateDeadline?.Token ?? deadline.Token);
        ValidatePinnedState(releasedState.RootElement, requireLease: false, connectionId);
        var releasedControl = releasedState.RootElement.GetProperty("control");
        RequireFalse(releasedControl, "lease_held");
        RequireFalse(releasedControl, "schedule_running");
        RequireFalse(releasedControl, "schedule_authorized_while_background");
        RequireFalse(releasedControl, "single_motion_trial_running");
        RequireFalse(releasedControl, "single_motion_trial_authorized_while_background");
        RequireFalse(releasedControl, "continuous_controller_running");
        RequireFalse(releasedControl, "continuous_controller_authorized_while_background");
        RequireFalse(releasedControl, "attack_zone_trial_running");
        RequireFalse(releasedControl, "attack_zone_recovery_only_running");
        if (mode == "exit-lost")
        {
            RequireString(releasedState.RootElement, "scene", "Lobby");
            RequireFalse(releasedState.RootElement.GetProperty("private_ai"), "proven");
        }
        if (mode == "schedule")
        {
            RequireString(
                releasedControl,
                "schedule_run_id",
                completedScheduleRunId ?? throw new InvalidDataException(
                    "completed schedule run ID was unavailable"));
            RequireInt32(releasedControl, "schedule_tick", 2600);
            RequireInt32(releasedControl, "client_fixed_substep", 26009);
            RequireTrue(releasedControl, "rendered_command_markers_visible");
            RequireInt32(
                releasedControl,
                "rendered_command_markers_post_count",
                ExpectedMarkerCount);
        }
        else if (mode == "trial")
        {
            RequireString(
                releasedControl,
                "single_motion_trial_run_id",
                completedTrialRunId ?? throw new InvalidDataException(
                    "completed trial run ID was unavailable"));
            RequireString(
                releasedControl,
                "single_motion_trial_selector",
                trialSelectorName!);
            RequireString(
                releasedControl,
                "single_motion_trial_round_identity_sha256",
                completedTrialRoundIdentitySha256 ?? throw new InvalidDataException(
                    "completed trial round identity was unavailable"));
            RequireString(
                releasedControl,
                "single_motion_trial_initial_state_sha256",
                completedTrialInitialStateSha256 ?? throw new InvalidDataException(
                    "completed trial initial-state identity was unavailable"));
            RequireInt32(releasedControl, "single_motion_trial_tick", TrialFinalTick);
            RequireInt32(
                releasedControl,
                "single_motion_trial_client_fixed_substep",
                TrialDurationTicks * TrialFixedSubstepsPerTick - 1);
        }
        else if (mode == "controller")
        {
            RequireString(
                releasedControl,
                "continuous_controller_run_id",
                completedControllerRunId ?? throw new InvalidDataException(
                    "completed continuous controller run ID was unavailable"));
            RequireString(releasedControl, "continuous_controller_phase", "inactive");
        }
    }

    await WriteClientResult("complete", null);
    await PublishTranscript();
    Console.WriteLine(resultJson);
    return 0;
}
catch (Exception exception)
{
    if (leaseHeld && pipe.IsConnected && reader is not null && writer is not null)
    {
        try
        {
            await ReleaseLease();
            leaseHeld = false;
        }
        catch (Exception releaseException)
        {
            await WriteClientResult(
                "release_failed",
                $"{releaseException.GetType().Name}:{releaseException.Message}");
        }
    }
    await WriteClientResult("failed", $"{exception.GetType().Name}:{exception.Message}");
    await FlushTranscript();
    Console.Error.WriteLine($"{exception.GetType().Name}: {exception.Message}");
    if (transcriptPartialPath is not null)
        Console.Error.WriteLine($"failure transcript preserved at {transcriptPartialPath}");
    return 1;
}
finally
{
    if (cancelHandler is not null)
        Console.CancelKeyPress -= cancelHandler;
    if (writer is not null)
        await writer.DisposeAsync();
    reader?.Dispose();
    if (transcript is not null)
    {
        await FlushTranscript();
        await transcript.DisposeAsync();
        transcript = null;
    }
}

async Task<JsonDocument> RequestState(CancellationToken cancellationToken)
{
    var requestId = NewRequestId("state");
    await SendRequest(new { type = "get_state", request_id = requestId });
    return await ReadMatching("state", requestId, cancellationToken);
}

async Task<JsonDocument> RequireAcceptedCommand(
    string command,
    string reason,
    bool expectedApplied,
    bool expectedRequestIssued,
    CancellationToken cancellationToken,
    string? selector = null)
{
    var requestId = NewRequestId("command");
    if (selector is null)
        await SendRequest(new { type = "command", request_id = requestId, command });
    else
        await SendRequest(new { type = "command", request_id = requestId, command, selector });
    var response = await ReadMatching("ack", requestId, cancellationToken);
    try
    {
        RequireString(response.RootElement, "protocol", Protocol);
        RequireString(response.RootElement, "command", command);
        if (selector is null)
            RequireNull(response.RootElement, "selector");
        else
            RequireString(response.RootElement, "selector", selector);
        RequireString(response.RootElement, "status", "accepted");
        RequireString(response.RootElement, "reason", reason);
        RequireBool(response.RootElement, "applied", expectedApplied);
        RequireBool(response.RootElement, "client_request_issued", expectedRequestIssued);
        RequireFalse(response.RootElement, "server_acceptance_observed");
        RequireFalse(response.RootElement, "authoritative_execution_observed");
        ValidateScheduleIdentity(response.RootElement);
        ValidateTrialIdentity(response.RootElement);
        ValidateContinuousContractIdentity(response.RootElement);
        ValidateAttackZoneAckIdentity(response.RootElement);
        return response;
    }
    catch
    {
        response.Dispose();
        throw;
    }
}

static void ValidateAttackZoneAckIdentity(JsonElement value)
{
    RequireString(value, "attack_zone_trial_schema", AttackZoneSchema);
    RequireString(value, "attack_zone_trial_sha256", AttackZoneSha256);
    _ = RequireBooleanValue(value, "attack_zone_trial_running");
    _ = RequireBooleanValue(value, "attack_zone_recovery_only_running");
    var readyTicks = RequireInt32Value(value, "attack_zone_recovery_ready_ticks");
    if (readyTicks is < 0 or > 15)
        throw new InvalidDataException("attack-zone recovery-ready tick count was invalid");
    _ = RequireNonemptyString(value, "attack_zone_trial_phase");
}

async Task<JsonDocument> RequireAcceptedControllerStop(
    string runId,
    CancellationToken cancellationToken)
{
    var requestId = NewRequestId("command");
    await SendRequest(new
    {
        type = "command",
        request_id = requestId,
        command = "StopContinuousBotController",
    });
    var endSeen = false;
    while (true)
    {
        var response = await ReadMessage(cancellationToken);
        var eventName = OptionalString(response.RootElement, "event");
        if (eventName == "error" &&
            OptionalString(response.RootElement, "request_id") == requestId)
        {
            var reason = OptionalString(response.RootElement, "reason") ?? "unspecified";
            response.Dispose();
            throw new InvalidDataException($"bridge rejected request {requestId}: {reason}");
        }
        if (eventName == "continuous_controller_end")
        {
            if (endSeen)
            {
                response.Dispose();
                throw new InvalidDataException("duplicate continuous_controller_end");
            }
            ValidateContinuousEnd(response.RootElement, runId, "requested");
            endSeen = true;
            response.Dispose();
            continue;
        }
        if (eventName != "ack" ||
            OptionalString(response.RootElement, "request_id") != requestId)
        {
            response.Dispose();
            continue;
        }
        try
        {
            if (!endSeen)
                throw new InvalidDataException("controller stop ack preceded continuous_controller_end");
            RequireString(response.RootElement, "protocol", Protocol);
            RequireString(response.RootElement, "command", "StopContinuousBotController");
            RequireNull(response.RootElement, "selector");
            RequireString(response.RootElement, "status", "accepted");
            RequireString(
                response.RootElement,
                "reason",
                "continuous_private_bot_controller_stopped");
            RequireTrue(response.RootElement, "applied");
            RequireFalse(response.RootElement, "client_request_issued");
            RequireFalse(response.RootElement, "server_acceptance_observed");
            RequireFalse(response.RootElement, "authoritative_execution_observed");
            ValidateScheduleIdentity(response.RootElement);
            ValidateTrialIdentity(response.RootElement);
            ValidateContinuousContractIdentity(response.RootElement);
            return response;
        }
        catch
        {
            response.Dispose();
            throw;
        }
    }
}

async Task<string> StartFreshPrivateRound(
    long expectedConnectionId,
    CancellationToken cancellationToken)
{
    using (var initial = await RequestState(cancellationToken))
    {
        ValidatePinnedState(initial.RootElement, requireLease: true, expectedConnectionId);
        ValidatePrivateBotOne(initial.RootElement, requireActiveRound: false);
        var privateAi = initial.RootElement.GetProperty("private_ai");
        if (RequireBooleanValue(privateAi, "round_active"))
            throw new InvalidDataException("single-motion trial requires an inactive fresh-round start");
        var control = initial.RootElement.GetProperty("control");
        RequireFalse(control, "fresh_round_armed");
        RequireNull(control, "fresh_round_request_id");
        RequireNull(control, "fresh_round_invalid_reason");
    }

    var requestId = NewRequestId("fresh-round");
    await SendRequest(new { type = "command", request_id = requestId, command = "StartRound" });
    using (var response = await ReadMatching("ack", requestId, cancellationToken))
    {
        RequireString(response.RootElement, "protocol", Protocol);
        RequireString(response.RootElement, "command", "StartRound");
        RequireNull(response.RootElement, "selector");
        RequireString(response.RootElement, "status", "accepted");
        RequireFalse(response.RootElement, "server_acceptance_observed");
        RequireFalse(response.RootElement, "authoritative_execution_observed");
        RequireInt64(response.RootElement, "lease_connection_id", expectedConnectionId);
        ValidateScheduleIdentity(response.RootElement);
        ValidateTrialIdentity(response.RootElement);
        RequireTrue(response.RootElement, "fresh_round_armed");
        RequireString(response.RootElement, "fresh_round_request_id", requestId);
        RequireNull(response.RootElement, "fresh_round_invalid_reason");
        var reason = RequireNonemptyString(response.RootElement, "reason");
        if (reason is "post_fight_continue_request_issued" or "remote_ready_request_issued")
        {
            RequireFalse(response.RootElement, "applied");
            RequireTrue(response.RootElement, "client_request_issued");
        }
        else if (reason == "native_start_fight_coroutine_observed")
        {
            RequireTrue(response.RootElement, "applied");
            RequireFalse(response.RootElement, "client_request_issued");
        }
        else
        {
            throw new InvalidDataException($"unexpected accepted StartRound reason {reason}");
        }
    }

    while (true)
    {
        await Task.Delay(50, cancellationToken);
        using var state = await RequestState(cancellationToken);
        ValidatePinnedState(state.RootElement, requireLease: true, expectedConnectionId);
        ValidatePrivateBotOne(state.RootElement, requireActiveRound: false);
        var control = state.RootElement.GetProperty("control");
        RequireTrue(control, "fresh_round_armed");
        RequireString(control, "fresh_round_request_id", requestId);
        RequireNull(control, "fresh_round_invalid_reason");
        var privateAi = state.RootElement.GetProperty("private_ai");
        if (!OptionalTrue(privateAi, "active_gameplay_proven") ||
            !OptionalTrue(privateAi, "round_active"))
        {
            continue;
        }
        ValidatePrivateBotOne(state.RootElement, requireActiveRound: true);
        return requestId;
    }
}

async Task EnsureActivePrivateRound(
    long expectedConnectionId,
    CancellationToken cancellationToken,
    bool requireStrictParityPairing = true)
{
    var requestRoundStart = false;
    using (var initial = await RequestState(cancellationToken))
    {
        ValidatePinnedState(initial.RootElement, requireLease: true, expectedConnectionId);
        ValidatePrivateBotOne(initial.RootElement, requireActiveRound: false);
        var privateAi = initial.RootElement.GetProperty("private_ai");
        if (OptionalTrue(privateAi, "active_gameplay_proven") &&
            OptionalTrue(privateAi, "round_active"))
        {
            ValidatePrivateBotOne(
                initial.RootElement,
                requireActiveRound: true,
                requireStrictParityPairing: requireStrictParityPairing);
            return;
        }
        requestRoundStart = !RequireBooleanValue(privateAi, "round_active");
    }

    if (requestRoundStart)
    {
        var requestId = NewRequestId("command");
        await SendRequest(new { type = "command", request_id = requestId, command = "StartRound" });
        using var response = await ReadMatching("ack", requestId, cancellationToken);
        RequireString(response.RootElement, "protocol", Protocol);
        RequireString(response.RootElement, "command", "StartRound");
        RequireString(response.RootElement, "status", "accepted");
        RequireFalse(response.RootElement, "server_acceptance_observed");
        RequireInt64(response.RootElement, "lease_connection_id", expectedConnectionId);
        ValidateScheduleIdentity(response.RootElement);
        var reason = RequireNonemptyString(response.RootElement, "reason");
        if (reason is "post_fight_continue_request_issued" or "remote_ready_request_issued")
        {
            RequireFalse(response.RootElement, "applied");
            RequireTrue(response.RootElement, "client_request_issued");
        }
        else if (reason == "native_start_fight_coroutine_observed")
        {
            RequireTrue(response.RootElement, "applied");
            RequireFalse(response.RootElement, "client_request_issued");
        }
        else
        {
            throw new InvalidDataException($"unexpected accepted StartRound reason {reason}");
        }
    }

    while (true)
    {
        await Task.Delay(100, cancellationToken);
        using var state = await RequestState(cancellationToken);
        ValidatePinnedState(state.RootElement, requireLease: true, expectedConnectionId);
        ValidatePrivateBotOne(state.RootElement, requireActiveRound: false);
        var privateAi = state.RootElement.GetProperty("private_ai");
        if (!OptionalTrue(privateAi, "active_gameplay_proven") ||
            !OptionalTrue(privateAi, "round_active"))
        {
            continue;
        }
        ValidatePrivateBotOne(
            state.RootElement,
            requireActiveRound: true,
            requireStrictParityPairing: requireStrictParityPairing);
        return;
    }
}

async Task ReleaseLease()
{
    using var releaseDeadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
    using var release = await RequireAcceptedCommand(
        "ReleaseExclusiveControl",
        "exclusive_control_lease_released",
        expectedApplied: true,
        expectedRequestIssued: false,
        releaseDeadline.Token);
    RequireNull(release.RootElement, "lease_connection_id");
}

async Task<JsonDocument> ReadUntilHello(CancellationToken cancellationToken)
{
    while (true)
    {
        var message = await ReadMessage(cancellationToken);
        if (OptionalString(message.RootElement, "event") == "hello")
            return message;
        message.Dispose();
    }
}

async Task<JsonDocument> ReadMatching(
    string expectedEvent,
    string requestId,
    CancellationToken cancellationToken)
{
    while (true)
    {
        var message = await ReadMessage(cancellationToken);
        var root = message.RootElement;
        if (OptionalString(root, "event") == "error" &&
            OptionalString(root, "request_id") == requestId)
        {
            var reason = OptionalString(root, "reason") ?? "unspecified";
            message.Dispose();
            throw new InvalidDataException($"bridge rejected request {requestId}: {reason}");
        }
        if (OptionalString(root, "event") == expectedEvent &&
            OptionalString(root, "request_id") == requestId)
            return message;
        message.Dispose();
    }
}

async Task<JsonDocument> ReadMessage(CancellationToken cancellationToken)
{
    if (reader is null)
        throw new InvalidOperationException("pipe reader is unavailable");
    var line = await reader.ReadLineAsync(cancellationToken);
    if (line is null)
        throw new EndOfStreamException("bridge closed the pipe");
    if (Encoding.UTF8.GetByteCount(line) > 1_048_576)
        throw new InvalidDataException("bridge response exceeded 1 MiB");
    if (transcript is not null)
        await transcript.WriteLineAsync(line);
    return JsonDocument.Parse(line);
}

async Task SendRequest(object request)
{
    if (writer is null)
        throw new InvalidOperationException("pipe writer is unavailable");
    var line = JsonSerializer.Serialize(request);
    if (transcript is not null)
    {
        await transcript.WriteLineAsync(JsonSerializer.Serialize(new
        {
            @event = "client_request",
            observed_utc = DateTimeOffset.UtcNow,
            request,
        }));
    }
    await writer.WriteLineAsync(line);
}

async Task WriteClientResult(string status, string? error)
{
    if (transcript is null)
        return;
    await transcript.WriteLineAsync(JsonSerializer.Serialize(new
    {
        @event = "client_result",
        observed_utc = DateTimeOffset.UtcNow,
        mode,
        status,
        error,
        lease_held = leaseHeld,
    }));
}

async Task FlushTranscript()
{
    if (transcript is null)
        return;
    await transcript.FlushAsync();
    if (transcript.BaseStream is FileStream stream)
        stream.Flush(flushToDisk: true);
}

async Task PublishTranscript()
{
    if (transcript is null)
        return;
    if (transcriptFinalPath is null || transcriptPartialPath is null)
        throw new InvalidOperationException("transcript paths are unavailable");
    await FlushTranscript();
    await transcript.DisposeAsync();
    transcript = null;
    File.Move(transcriptPartialPath, transcriptFinalPath, overwrite: false);
}

static PipeServerProof ValidatePipeServer(NamedPipeClientStream pipe)
{
    if (!NativeMethods.GetNamedPipeServerProcessId(
            pipe.SafePipeHandle.DangerousGetHandle(),
            out var processId))
    {
        throw new Win32Exception(
            Marshal.GetLastWin32Error(),
            "GetNamedPipeServerProcessId failed");
    }
    if (processId == 0 || processId > int.MaxValue || processId == Environment.ProcessId)
        throw new InvalidDataException("named-pipe server process ID was invalid");

    try
    {
        using var process = Process.GetProcessById((int)processId);
        if (process.HasExited)
            throw new InvalidDataException("named-pipe server process was not live");
        var executable = process.MainModule?.FileName;
        if (string.IsNullOrWhiteSpace(executable) ||
            !string.Equals(Path.GetFileName(executable), "REK.exe", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("named-pipe server executable was not REK.exe");
        }
        return new PipeServerProof(processId, executable);
    }
    catch (InvalidDataException)
    {
        throw;
    }
    catch (Exception exception)
    {
        throw new InvalidDataException(
            "named-pipe server process was not resolvable on the local host",
            exception);
    }
}

static void ValidateHello(
    JsonElement hello,
    IReadOnlyList<ExpectedTrialSelector> expectedTrialSelectors)
{
    RequireString(hello, "event", "hello");
    RequireString(hello, "protocol", Protocol);
    RequireString(hello, "pipe", PipeName);
    RequireTrue(hello, "current_user_only");
    RequireTrue(hello, "local_computer_verified");
    var verification = RequireNonemptyString(hello, "local_client_verification");
    if (verification is not (
            "local_computer_verified" or
            "local_pipe_verified_win32_error_pipe_local" or
            "local_process_id_verified_after_computer_name_api_unavailable"))
    {
        throw new InvalidDataException("unrecognized local_client_verification");
    }
    var capabilities = hello.GetProperty("capabilities");
    RequireTrue(capabilities, "state");
    RequireFalse(capabilities, "input_available");
    RequireStringArray(
        capabilities,
        "parsed_but_rejected_input",
        new[] { "Left", "Right", "Up", "Down", "Enter", "Escape", "Space" });
    RequireString(
        capabilities,
        "input_unavailable_reason",
        "verified_process_targeted_unity_input_delivery_not_implemented");
    RequireStringArray(
        capabilities,
        "semantic_commands",
        new[]
        {
            "AcquireExclusiveControl",
            "ReleaseExclusiveControl",
            "ConfirmLoggedIn",
            "NavigateFreePlay",
            "EnterSolo",
            "StartRound",
            "ExitUnexpectedPrivateAiSession",
            "ExitLostPrivateSession",
            "StartMeasuredSchedule",
            "StopMeasuredSchedule",
            "StartSingleMotionTrial",
            "StartContinuousBotController",
            "StopContinuousBotController",
            "StartAttackZoneTrial",
            "StopAttackZoneTrial",
        });
    RequireTrue(capabilities, "exclusive_control_lease_required");
    RequireFalse(capabilities, "autonomous_input");
    RequireTrue(capabilities, "autonomous_semantic_controller");
    RequireString(capabilities, "rendered_command_marker_schema", MarkerSchema);
    RequireString(
        capabilities,
        "rendered_command_marker_render_binding",
        MarkerRenderBinding);
    RequireInt32(capabilities, "rendered_command_marker_count", ExpectedMarkerCount);
    RequireString(capabilities, "single_motion_trial_schema", TrialSchema);
    RequireString(capabilities, "single_motion_trial_sha256", TrialSha256);
    RequireString(
        capabilities,
        "single_motion_trial_authority_scope",
        TrialAuthorityScope);
    RequireString(
        capabilities,
        "single_motion_trial_authority_caveat",
        TrialAuthorityCaveat);
    RequireStringArray(
        capabilities,
        "single_motion_trial_selectors",
        expectedTrialSelectors.Select(value => value.Selector).ToArray());
    RequireInt32(capabilities, "single_motion_trial_unity_fixed_rate_hz", 500);
    RequireInt32(capabilities, "single_motion_trial_rate_hz", 50);
    RequireInt32(
        capabilities,
        "single_motion_trial_fixed_substeps_per_tick",
        TrialFixedSubstepsPerTick);
    RequireInt32(
        capabilities,
        "single_motion_trial_neutral_pre_roll_ticks",
        TrialNeutralPreRollTicks);
    RequireInt32(capabilities, "single_motion_trial_action_tick", TrialActionTick);
    RequireInt32(
        capabilities,
        "single_motion_trial_locomotion_release_tick",
        TrialLocomotionReleaseTick);
    RequireInt32(capabilities, "single_motion_trial_duration_ticks", TrialDurationTicks);
    RequireString(capabilities, "continuous_controller_schema", ContinuousSchema);
    RequireString(capabilities, "continuous_controller_sha256", ContinuousSha256);
    RequireString(
        capabilities,
        "continuous_controller_authority_scope",
        ContinuousAuthorityScope);
    RequireString(
        capabilities,
        "continuous_controller_authority_caveat",
        ContinuousAuthorityCaveat);
    RequireString(
        capabilities,
        "continuous_controller_range_angle_provenance",
        ContinuousRangeAngleProvenance);
    RequireString(
        capabilities,
        "continuous_controller_facing_yaw_provenance",
        ContinuousFacingYawProvenance);
    RequireString(
        capabilities,
        "continuous_controller_attack_selection_provenance",
        ContinuousAttackSelectionProvenance);
    RequireString(
        capabilities,
        "continuous_controller_static_impact_timing_provenance",
        ContinuousStaticImpactTimingProvenance);
    RequireString(
        capabilities,
        "continuous_controller_round_restart_limitation",
        ContinuousRoundRestartLimitation);
    RequireString(
        capabilities,
        "continuous_controller_round_restart_static_evidence",
        ContinuousRoundRestartStaticEvidence);
    RequireInt32(capabilities, "continuous_controller_unity_fixed_rate_hz", 500);
    RequireInt32(capabilities, "continuous_controller_rate_hz", 50);
    RequireInt32(capabilities, "continuous_controller_fixed_substeps_per_tick", 10);
    RequireIntArray(
        capabilities,
        "continuous_controller_move_indices",
        new[] { 2, 3, 4, 5, 9, 10 });
    RequireString(
        capabilities,
        "continuous_controller_recovery_guard_provenance",
        ContinuousRecoveryGuardProvenance);
    RequireString(
        capabilities,
        "continuous_controller_fault_estop_provenance",
        ContinuousFaultEStopProvenance);
    RequireString(
        capabilities,
        "continuous_controller_dampen_guard",
        ContinuousDampenGuard);
    RequireString(
        capabilities,
        "continuous_controller_straighten_guard",
        ContinuousStraightenGuard);
    RequireString(
        capabilities,
        "continuous_controller_opponent_runtime_requirement",
        ContinuousOpponentRuntimeRequirement);
    RequireFiniteExactNumber(
        capabilities,
        "continuous_controller_facing_deadband_factor",
        0.5);
    RequireFiniteExactNumber(
        capabilities,
        "continuous_controller_facing_threshold_degrees",
        35.0);
    RequireFiniteExactNumber(
        capabilities,
        "continuous_controller_facing_yaw_ramp_degrees",
        45.0);
    RequireFiniteExactNumber(
        capabilities,
        "continuous_controller_engage_yaw_command",
        1.5);
    RequireInt32(capabilities, "continuous_controller_fault_estop_delay_ticks", 150);
    RequireInt32(capabilities, "continuous_controller_fault_estop_hold_ticks", 25);
    RequireInt32(
        capabilities,
        "continuous_controller_recovery_observation_timeout_ticks",
        250);
    RequireInt32(
        capabilities,
        "continuous_controller_round_start_prompt_delay_ticks",
        5);
    RequireInt32(
        capabilities,
        "continuous_controller_round_start_observation_timeout_ticks",
        1500);
    RequireInt32(
        capabilities,
        "continuous_controller_two_minute_limit_ticks",
        6000);
    RequireString(
        capabilities,
        "continuous_controller_round_start_semantic_method",
        "GameMenuController.HandlePostFightContinue");
    RequireFalse(
        capabilities,
        "continuous_controller_global_space_input_emitted");
    RequireFalse(
        capabilities,
        "continuous_controller_opponent_semantic_robot_id_used_for_acceptance");
    RequireString(capabilities, "attack_zone_trial_schema", AttackZoneSchema);
    RequireString(capabilities, "attack_zone_trial_sha256", AttackZoneSha256);
    RequireString(
        capabilities,
        "attack_zone_trial_authority_scope",
        AttackZoneAuthorityScope);
    RequireString(
        capabilities,
        "attack_zone_trial_authority_caveat",
        AttackZoneAuthorityCaveat);
    RequireString(
        capabilities,
        "attack_zone_trial_required_isolation_proof",
        AttackZoneIsolationProof);
    RequireInt32(capabilities, "attack_zone_trial_control_rate_hz", 50);
    RequireInt32(capabilities, "attack_zone_trial_fixed_substeps_per_tick", 10);
    RequireInt32(capabilities, "attack_zone_trial_settle_ticks", 15);
    RequireInt32(capabilities, "attack_zone_trial_action_sample_rate_hz", 50);
    RequireInt32(capabilities, "attack_zone_trial_recovery_ready_ticks", 15);
    RequireInt32(capabilities, "attack_zone_trial_acquisition_timeout_ticks", 500);
    RequireInt32(
        capabilities,
        "attack_zone_trial_minimum_independent_runs_per_cell",
        5);
    RequireString(
        capabilities,
        "attack_zone_trial_recorder_version",
        AttackZoneRecorderVersion);
    RequireString(
        capabilities,
        "attack_zone_trial_recorder_plugin_sha256",
        AttackZoneRecorderSha256);
    RequireFalse(capabilities, "attack_zone_trial_global_input_emitted");
    var attackProfiles = capabilities.GetProperty("continuous_controller_attack_profiles");
    if (attackProfiles.ValueKind != JsonValueKind.Array || attackProfiles.GetArrayLength() != 6)
        throw new InvalidDataException("expected six continuous_controller_attack_profiles");
    var expectedProfiles = new[]
    {
        (2, "skill", "Punch Combo",
            "233f952edecb7bf8d1959c6549c0edb95e1833451fff988f57a4b14d92b14dd4",
            new[] { (0.76f, 0.1f, 0.19f, 1), (1.15f, 0.1f, 0.1f, 1), (1.81f, 0.1f, 0.1f, 1) }),
        (3, "youbiantui", "Right Kick",
            "70f36a2c7b9b53c10e47cc613d87a770eb86fb2e683ed64ee39efcccf2e75636",
            new[] { (1.11f, 0.25f, 0.3f, 4) }),
        (4, "left_light_attack", "Left Punch",
            "32081b731a59b7553d94022ebff865764b34c83dbf274aadf26540fb17daad2e",
            new[] { (0.39f, 0.12f, 0.08f, 1) }),
        (5, "right_light_attack", "Right Punch",
            "b1c1b2c000dd612e3eb4c33c5d90e03c2c9306e5cc194747c14248b0d77b7dea",
            new[] { (0.22f, 0.12f, 0.2f, 2) }),
        (9, "right_shoryuken_lm", "Dragon Punch",
            "cc298f53d04ffd56be57ce3049559d3d30c7724fe4d2839a66ea8f3008ca8deb",
            new[] { (0f, 0f, 0f, 2) }),
        (10, "front_kick_L", "Left Kick",
            "cd5b286f6e4f5c3003cb0f5c9de5e5690ca92ed58e5a1b789f4394e4d7911ee8",
            new[] { (1.1f, 0.2f, 0.15f, 3) }),
    };
    for (var profileIndex = 0; profileIndex < expectedProfiles.Length; profileIndex++)
    {
        var expected = expectedProfiles[profileIndex];
        var actual = attackProfiles[profileIndex];
        RequireInt32(actual, "move_index", expected.Item1);
        RequireString(actual, "move_name", expected.Item2);
        RequireString(actual, "display_name", expected.Item3);
        RequireString(actual, "serialized_asset_sha256", expected.Item4);
        if (!SameFloatBits(
                actual.GetProperty("maximum_distance_m").GetSingle(),
                0.5180000126361847f) ||
            !SameFloatBits(
                actual.GetProperty("maximum_abs_bearing_degrees").GetSingle(),
                35f))
        {
            throw new InvalidDataException("continuous attack range or angle mismatch");
        }
        var impacts = actual.GetProperty("static_impact_events");
        if (impacts.ValueKind != JsonValueKind.Array ||
            impacts.GetArrayLength() != expected.Item5.Length)
        {
            throw new InvalidDataException("continuous static impact event count mismatch");
        }
        for (var impactIndex = 0; impactIndex < expected.Item5.Length; impactIndex++)
        {
            var expectedImpact = expected.Item5[impactIndex];
            var actualImpact = impacts[impactIndex];
            if (!SameFloatBits(actualImpact.GetProperty("impact_time_s").GetSingle(), expectedImpact.Item1) ||
                !SameFloatBits(actualImpact.GetProperty("lead_time_s").GetSingle(), expectedImpact.Item2) ||
                !SameFloatBits(actualImpact.GetProperty("release_time_s").GetSingle(), expectedImpact.Item3) ||
                !SameFloatBits(actualImpact.GetProperty("gain_boost").GetSingle(), 1f) ||
                actualImpact.GetProperty("limb").GetInt32() != expectedImpact.Item4)
            {
                throw new InvalidDataException("continuous static impact event mismatch");
            }
        }
    }
}

static void ValidatePinnedState(JsonElement state, bool requireLease, long connectionId)
{
    RequireString(state, "protocol", Protocol);
    RequireString(state, "application_version", ExpectedApplicationVersion);
    RequireString(state, "unity_version", ExpectedUnityVersion);
    var build = state.GetProperty("build");
    RequireString(build, "game_assembly_sha256", ExpectedGameAssemblySha256);
    RequireString(build, "global_metadata_sha256", ExpectedMetadataSha256);
    RequireString(build, "plugin_version", ExpectedBridgeVersion);
    RequireString(build, "plugin_sha256", ExpectedBridgeSha256);

    var foreground = state.GetProperty("foreground");
    RequireTrue(foreground, "mutation_allowed");
    RequireTrue(foreground, "isolated_session_verified");
    RequireString(
        foreground,
        "isolated_session_proof",
        "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98");

    var control = state.GetProperty("control");
    RequireTrue(control, "semantic_available");
    RequireTrue(control, "exclusive_lease_required");
    ValidateScheduleIdentity(control);
    ValidateTrialIdentity(control);
    ValidateContinuousIdentity(control);
    ValidateAttackZoneIdentity(control);
    RequireTrue(control, "send_boundary_patches_verified");
    RequireTrue(control, "trial_isolation_patches_verified");
    RequireString(control, "rendered_command_marker_schema", MarkerSchema);
    RequireString(
        control,
        "rendered_command_marker_render_binding",
        MarkerRenderBinding);
    RequireInt32(control, "rendered_command_marker_count", ExpectedMarkerCount);
    RequireInt32(control, "fixed_substeps_per_schedule_tick", 10);
    RequireInt32(
        control,
        "single_motion_trial_fixed_substeps_per_tick",
        TrialFixedSubstepsPerTick);
    RequireInt32(
        control,
        "single_motion_trial_neutral_pre_roll_ticks",
        TrialNeutralPreRollTicks);
    RequireInt32(control, "single_motion_trial_action_tick", TrialActionTick);
    RequireInt32(
        control,
        "single_motion_trial_locomotion_release_tick",
        TrialLocomotionReleaseTick);
    RequireInt32(control, "single_motion_trial_duration_ticks", TrialDurationTicks);
    RequireInt32(control, "single_motion_trial_history_capacity", 128);
    var roundsConsumed = RequireInt32Value(control, "single_motion_trial_rounds_consumed");
    if (roundsConsumed is < 0 or > 128)
        throw new InvalidDataException("single_motion_trial_rounds_consumed was outside [0,128]");
    RequireBool(control, "lease_held", requireLease);
    if (requireLease)
        RequireInt64(control, "lease_connection_id", connectionId);
    else
        RequireNull(control, "lease_connection_id");

    var input = state.GetProperty("input");
    RequireFalse(input, "global_input_available");
    RequireTrue(input, "semantic_commands_available");
    RequireBool(input, "autonomous", requireLease);
    RequireString(
        input,
        "global_input_unavailable_reason",
        "global_keyboard_mouse_and_gamepad_injection_deliberately_unavailable");
}

static void ValidateAttackZoneIdentity(JsonElement value)
{
    RequireString(value, "attack_zone_trial_schema", AttackZoneSchema);
    RequireString(value, "attack_zone_trial_sha256", AttackZoneSha256);
    RequireString(
        value,
        "attack_zone_trial_authority_scope",
        AttackZoneAuthorityScope);
    RequireString(
        value,
        "attack_zone_trial_authority_caveat",
        AttackZoneAuthorityCaveat);
    RequireString(
        value,
        "attack_zone_trial_recorder_version",
        AttackZoneRecorderVersion);
    RequireString(
        value,
        "attack_zone_trial_recorder_plugin_sha256",
        AttackZoneRecorderSha256);
    _ = RequireBooleanValue(value, "attack_zone_trial_running");
    _ = RequireBooleanValue(value, "attack_zone_recovery_only_running");
    var readyTicks = RequireInt32Value(value, "attack_zone_recovery_ready_ticks");
    if (readyTicks is < 0 or > 15)
        throw new InvalidDataException("attack_zone_recovery_ready_ticks was outside [0,15]");
    _ = RequireNonemptyString(value, "attack_zone_trial_phase");
    var availability = value.GetProperty("attack_zone_trial_availability");
    var available = RequireBooleanValue(availability, "available");
    _ = RequireNonemptyString(availability, "reason");
    if (available)
    {
        _ = RequireHexString(availability, "session_identity_sha256", 64);
        _ = RequireHexString(availability, "round_identity_sha256", 64);
    }
    else
    {
        RequireNull(availability, "session_identity_sha256");
        RequireNull(availability, "round_identity_sha256");
    }
}

static void ValidateContinuousIdentity(JsonElement value)
{
    ValidateContinuousContractIdentity(value);
    RequireString(
        value,
        "continuous_controller_static_impact_timing_provenance",
        ContinuousStaticImpactTimingProvenance);
    _ = RequireNonemptyString(value, "continuous_controller_phase");
    _ = RequireInt32Value(value, "continuous_controller_tick");
    _ = RequireInt32Value(value, "continuous_controller_round_tick");
    _ = RequireInt32Value(value, "continuous_controller_round_sequence");
    _ = RequireInt32Value(value, "continuous_controller_next_attack_index");
    _ = RequireInt32Value(value, "continuous_controller_action_sequence");
    _ = RequireInt32Value(value, "continuous_controller_recovery_sequence");
    _ = RequireNonemptyString(value, "continuous_controller_recovery_stage");
    RequireString(
        value,
        "continuous_controller_recovery_guard_provenance",
        ContinuousRecoveryGuardProvenance);
    RequireString(
        value,
        "continuous_controller_fault_estop_provenance",
        ContinuousFaultEStopProvenance);
    RequireString(
        value,
        "continuous_controller_dampen_guard",
        ContinuousDampenGuard);
    RequireString(
        value,
        "continuous_controller_straighten_guard",
        ContinuousStraightenGuard);
    RequireString(
        value,
        "continuous_controller_opponent_runtime_requirement",
        ContinuousOpponentRuntimeRequirement);
    _ = RequireBooleanValue(value, "continuous_controller_straighten_issued");
}

static void ValidateContinuousContractIdentity(JsonElement value)
{
    RequireString(value, "continuous_controller_schema", ContinuousSchema);
    RequireString(value, "continuous_controller_sha256", ContinuousSha256);
    RequireString(
        value,
        "continuous_controller_authority_scope",
        ContinuousAuthorityScope);
    RequireString(
        value,
        "continuous_controller_authority_caveat",
        ContinuousAuthorityCaveat);
    RequireString(
        value,
        "continuous_controller_range_angle_provenance",
        ContinuousRangeAngleProvenance);
    RequireString(
        value,
        "continuous_controller_facing_yaw_provenance",
        ContinuousFacingYawProvenance);
    RequireString(
        value,
        "continuous_controller_attack_selection_provenance",
        ContinuousAttackSelectionProvenance);
    RequireString(
        value,
        "continuous_controller_static_impact_timing_provenance",
        ContinuousStaticImpactTimingProvenance);
    RequireString(
        value,
        "continuous_controller_round_restart_limitation",
        ContinuousRoundRestartLimitation);
    RequireString(
        value,
        "continuous_controller_recovery_guard_provenance",
        ContinuousRecoveryGuardProvenance);
    RequireString(
        value,
        "continuous_controller_fault_estop_provenance",
        ContinuousFaultEStopProvenance);
    RequireString(
        value,
        "continuous_controller_dampen_guard",
        ContinuousDampenGuard);
    RequireString(
        value,
        "continuous_controller_straighten_guard",
        ContinuousStraightenGuard);
    RequireString(
        value,
        "continuous_controller_opponent_runtime_requirement",
        ContinuousOpponentRuntimeRequirement);
}

static void ValidatePrivateBotOne(
    JsonElement state,
    bool requireActiveRound,
    bool requireStrictParityPairing = true)
{
    var privateAi = state.GetProperty("private_ai");
    RequireTrue(privateAi, "proven");
    RequireTrue(privateAi, "network_client_only");
    RequireTrue(privateAi, "context_is_solo");
    RequireTrue(privateAi, "multiplayer_session_privacy_known");
    RequireTrue(privateAi, "multiplayer_session_is_private");
    RequireTrue(privateAi, "opponent_is_ai");
    RequireTrue(privateAi, "opponent_slot_is_ai");
    RequireFalse(privateAi, "human_in_opponent_slot");
    RequireTrue(privateAi, "opponent_slot_client_known");
    RequireFalse(privateAi, "opponent_slot_has_client");
    RequireFalse(privateAi, "opponent_human_bit_set");
    RequireInt32(privateAi, "client_ai_difficulty", 0);
    RequireInt32(privateAi, "sparring_bot_number", 1);
    RequireTrue(privateAi, "exact_sparring_bot_1");
    if (requireActiveRound)
    {
        RequireTrue(privateAi, "active_gameplay_proven");
        RequireTrue(privateAi, "round_active");
        if (requireStrictParityPairing)
            ValidateMeasuredPairing(state.GetProperty("measured_pairing"));
        else
            ValidateContinuousMeasuredPairing(state.GetProperty("measured_pairing"));
    }
}

static void ValidateMeasuredPairing(JsonElement pairing)
{
    RequireTrue(pairing, "exact_t800_vs_t800");
    RequireString(pairing, "required_pairing", "t800_vs_t800");
    RequireString(pairing, "required_robot_id", "t800");
    RequireInt32(pairing, "required_t800_bone_count", 26);
    RequireString(pairing, "required_t800_bone_signature_sha256", T800BoneSignatureSha256);
}

static void ValidateContinuousMeasuredPairing(JsonElement pairing)
{
    _ = RequireBooleanValue(pairing, "exact_t800_vs_t800");
    RequireString(pairing, "required_pairing", "t800_vs_t800");
    RequireString(pairing, "required_robot_id", "t800");
    RequireInt32(pairing, "required_t800_bone_count", 26);
    RequireString(pairing, "required_t800_bone_signature_sha256", T800BoneSignatureSha256);
    var local = pairing.GetProperty("local_fighter");
    RequireString(local, "semantic_robot_id", "t800");
    RequireTrue(local, "semantic_t800");
    RequireTrue(local, "exact_t800_bone_signature");
    RequireInt32(local, "bone_count", 26);
    RequireString(local, "runtime_bone_signature_sha256", T800BoneSignatureSha256);
    RequireFalse(local, "semantic_robot_id_used_for_continuous_acceptance");
    var localBoneNames = RequireNonemptyUniqueStringArray(local, "bone_names", 26);
    if (!string.Equals(
            HashUtf8Text(string.Join("\n", localBoneNames)),
            T800BoneSignatureSha256,
            StringComparison.Ordinal))
    {
        throw new InvalidDataException("continuous local runtime bone payload hash mismatch");
    }

    var opponent = pairing.GetProperty("opponent_fighter");
    _ = RequireNonemptyString(opponent, "runtime_object_name");
    var opponentBoneCount = RequireInt32Value(opponent, "bone_count");
    if (opponentBoneCount != 26)
        throw new InvalidDataException("continuous opponent runtime T800 bone count was not exact");
    var opponentBoneSignature = RequireHexString(
        opponent,
        "runtime_bone_signature_sha256",
        64);
    var opponentBoneNames = RequireNonemptyUniqueStringArray(
        opponent,
        "bone_names",
        opponentBoneCount);
    if (!string.Equals(
            HashUtf8Text(string.Join("\n", opponentBoneNames)),
            opponentBoneSignature,
            StringComparison.Ordinal))
    {
        throw new InvalidDataException("continuous opponent runtime bone payload hash mismatch");
    }
    var opponentSemanticT800 = RequireBooleanValue(opponent, "semantic_t800");
    var opponentExactT800 = RequireBooleanValue(opponent, "exact_t800_bone_signature");
    var derivedSemanticT800 = string.Equals(
        OptionalString(opponent, "semantic_robot_id"),
        "t800",
        StringComparison.Ordinal);
    var derivedExactT800 = opponentBoneCount == 26 && string.Equals(
        opponentBoneSignature,
        T800BoneSignatureSha256,
        StringComparison.Ordinal);
    if (opponentSemanticT800 != derivedSemanticT800 ||
        opponentExactT800 != derivedExactT800 ||
        RequireBooleanValue(opponent, "semantic_runtime_mismatch") !=
        (derivedSemanticT800 != derivedExactT800))
    {
        throw new InvalidDataException(
            "continuous opponent semantic/runtime classification mismatch");
    }
    if (!opponentExactT800 || !derivedExactT800)
        throw new InvalidDataException("continuous opponent runtime T800 signature was not exact");
    RequireFalse(opponent, "semantic_robot_id_used_for_continuous_acceptance");
}

static void ValidateContinuousEvent(JsonElement value, string runId)
{
    var eventName = RequireNonemptyString(value, "event");
    if (eventName is not (
            "continuous_controller_start" or
            "continuous_controller_round_bound" or
            "continuous_controller_suspend" or
            "continuous_controller_resume" or
            "continuous_controller_telemetry" or
            "continuous_round_observation" or
            "continuous_round_start_lifecycle" or
            "continuous_velocity_lifecycle" or
            "continuous_action_lifecycle" or
            "continuous_recovery_lifecycle"))
    {
        throw new InvalidDataException($"unexpected continuous event {eventName}");
    }
    RequireString(value, "protocol", Protocol);
    RequireString(value, "continuous_controller_schema", ContinuousSchema);
    RequireString(value, "continuous_controller_sha256", ContinuousSha256);
    RequireString(value, "authority_scope", ContinuousAuthorityScope);
    RequireString(value, "authority_caveat", ContinuousAuthorityCaveat);
    RequireString(value, "range_angle_provenance", ContinuousRangeAngleProvenance);
    RequireString(value, "facing_yaw_provenance", ContinuousFacingYawProvenance);
    RequireString(
        value,
        "attack_selection_provenance",
        ContinuousAttackSelectionProvenance);
    RequireString(
        value,
        "static_impact_timing_provenance",
        ContinuousStaticImpactTimingProvenance);
    RequireString(
        value,
        "round_restart_limitation",
        ContinuousRoundRestartLimitation);
    RequireString(
        value,
        "recovery_guard_provenance",
        ContinuousRecoveryGuardProvenance);
    RequireString(value, "fault_estop_provenance", ContinuousFaultEStopProvenance);
    RequireString(value, "dampen_guard", ContinuousDampenGuard);
    RequireString(value, "straighten_guard", ContinuousStraightenGuard);
    RequireString(
        value,
        "opponent_runtime_requirement",
        ContinuousOpponentRuntimeRequirement);
    RequireString(value, "continuous_controller_run_id", runId);
    _ = RequireNonemptyString(value, "controller_phase");
    _ = RequireNonemptyString(value, "controller_reason");
    _ = RequireInt32Value(value, "round_sequence");
    _ = RequireInt32Value(value, "client_control_tick");
    _ = RequireInt32Value(value, "client_fixed_substep");
    RequireInt32(value, "fixed_substeps_per_control_tick", 10);
    RequireTrue(value, "client_request_observation_only");
    RequireFalse(value, "server_acceptance_observed");
    RequireFalse(value, "authoritative_execution_observed");
    _ = RequireNonemptyString(value, "utc");
    if (RequireInt64Value(value, "stopwatch_timestamp_ticks") <= 0 ||
        RequireInt64Value(value, "stopwatch_frequency_hz") <= 0)
    {
        throw new InvalidDataException("continuous event stopwatch clock invalid");
    }
    _ = RequireInt32Value(value, "unity_frame");
    RequireFiniteNumber(value, "unity_time");
    RequireFiniteNumber(value, "unity_fixed_time");

    var measured = value.GetProperty("measured_state");
    if (!value.TryGetProperty("detail", out var detail) ||
        detail.ValueKind is not (JsonValueKind.Null or JsonValueKind.Object))
    {
        throw new InvalidDataException("continuous event detail was not object or null");
    }
    if (measured.ValueKind == JsonValueKind.Null)
        return;
    if (measured.ValueKind != JsonValueKind.Object)
        throw new InvalidDataException("continuous measured_state was not object or null");
    var local = measured.GetProperty("local_identity");
    RequireString(local, "semantic_robot_id", "t800");
    _ = RequireNonemptyString(local, "runtime_object_name");
    RequireInt32(local, "runtime_bone_count", 26);
    RequireString(local, "runtime_bone_signature_sha256", T800BoneSignatureSha256);
    RequireTrue(local, "exact_local_t800_proven");
    var opponent = measured.GetProperty("opponent_identity");
    var opponentRuntimeName = RequireNonemptyString(opponent, "runtime_object_name");
    var opponentBoneCount = RequireInt32Value(opponent, "runtime_bone_count");
    if (opponentBoneCount <= 0)
        throw new InvalidDataException("continuous measured opponent bone count not positive");
    var opponentBoneSignature = RequireHexString(
        opponent,
        "runtime_bone_signature_sha256",
        64);
    RequireString(
        opponent,
        "runtime_identity_sha256",
        HashUtf8Text($"{opponentRuntimeName}\n{opponentBoneSignature}"));
    var semanticDeclaresT800 = string.Equals(
        OptionalString(opponent, "semantic_robot_id_untrusted_for_runtime_acceptance"),
        "t800",
        StringComparison.Ordinal);
    var runtimeIsExactT800 = opponentBoneCount == 26 && string.Equals(
        opponentBoneSignature,
        T800BoneSignatureSha256,
        StringComparison.Ordinal);
    if (RequireBooleanValue(opponent, "semantic_runtime_mismatch") !=
        (semanticDeclaresT800 != runtimeIsExactT800))
    {
        throw new InvalidDataException(
            "continuous measured opponent semantic/runtime mismatch flag invalid");
    }
    RequireFalse(opponent, "semantic_robot_id_used_for_acceptance");
    var expectedConsistency = semanticDeclaresT800 != runtimeIsExactT800
        ? "semantic_t800_flag_disagrees_with_runtime_t800_signature"
        : runtimeIsExactT800
            ? "semantic_and_runtime_both_exact_t800"
            : "semantic_and_runtime_not_comparable_beyond_t800_signature";
    RequireString(opponent, "semantic_runtime_consistency", expectedConsistency);
    var geometry = measured.GetProperty("geometry");
    RequirePositiveFiniteNumber(geometry, "planar_distance_m");
    RequireFiniteNumber(geometry, "local_bearing_to_opponent_deg");
    RequireFiniteNumber(geometry, "opponent_bearing_to_local_deg");
    RequireFiniteNumber(geometry, "local_heading_deg");
    RequireFiniteNumber(geometry, "opponent_heading_deg");
    var localRoot = measured.GetProperty("local_root");
    RequireFiniteNumberArray(localRoot, "position_xyz_m", 3, "local_root");
    RequireFiniteNumberArray(localRoot, "rotation_xyzw", 4, "local_root");
    RequireFiniteNumberArray(localRoot, "forward_xyz", 3, "local_root");
    var opponentRoot = measured.GetProperty("opponent_root");
    RequireFiniteNumberArray(opponentRoot, "position_xyz_m", 3, "opponent_root");
    RequireFiniteNumberArray(opponentRoot, "rotation_xyzw", 4, "opponent_root");
    RequireFiniteNumberArray(opponentRoot, "forward_xyz", 3, "opponent_root");
    var localState = measured.GetProperty("local_state");
    _ = RequireBooleanValue(localState, "falling");
    _ = RequireBooleanValue(localState, "fallen");
    _ = RequireBooleanValue(localState, "dampened");
    _ = RequireBooleanValue(localState, "recovery_armed");
    _ = RequireBooleanValue(localState, "get_up_pending");
    _ = RequireBooleanValue(localState, "resetting");
    _ = RequireBooleanValue(localState, "motor_shutdown");
    _ = RequireNonemptyString(localState, "suggested_get_up_orientation");
    _ = RequireInt32Value(localState, "suggested_get_up_orientation_value");
    var opponentState = measured.GetProperty("opponent_state");
    _ = RequireBooleanValue(opponentState, "falling");
    _ = RequireBooleanValue(opponentState, "fallen");
    var inputState = measured.GetProperty("input_state");
    RequireFiniteNumberArray(inputState, "velocity_command_xyz", 3, "input_state");
    _ = RequireBooleanValue(inputState, "punching");
    _ = RequireBooleanValue(inputState, "recovering");
    _ = RequireBooleanValue(inputState, "allow_move_interrupt");
    _ = RequireBooleanValue(inputState, "pending_move");
    _ = RequireInt32Value(inputState, "pending_move_index");
    _ = RequireBooleanValue(inputState, "pending_special");
    _ = RequireInt32Value(inputState, "pending_special_command");
    _ = RequireBooleanValue(inputState, "pending_estop");
    var localMotion = measured.GetProperty("local_motion");
    _ = RequireBooleanValue(localMotion, "action_playing");
    _ = RequireBooleanValue(localMotion, "busy");
    RequireNullOrString(localMotion, "active_action_clip");
    _ = RequireInt32Value(localMotion, "current_move_id");
    _ = RequireInt32Value(localMotion, "action_clip_frame");
    RequireFiniteNumber(localMotion, "action_clip_fps");
}

static void ValidateContinuousEnd(
    JsonElement value,
    string runId,
    string? expectedReason)
{
    RequireString(value, "event", "continuous_controller_end");
    RequireString(value, "protocol", Protocol);
    RequireString(value, "continuous_controller_schema", ContinuousSchema);
    RequireString(value, "continuous_controller_sha256", ContinuousSha256);
    RequireString(value, "authority_scope", ContinuousAuthorityScope);
    RequireString(value, "authority_caveat", ContinuousAuthorityCaveat);
    RequireString(value, "continuous_controller_run_id", runId);
    if (value.GetProperty("round_identity_sha256").ValueKind != JsonValueKind.Null)
        _ = RequireHexString(value, "round_identity_sha256", 64);
    var reason = RequireNonemptyString(value, "reason");
    if (expectedReason is not null && reason != expectedReason)
        throw new InvalidDataException($"expected continuous end reason {expectedReason}");
    RequireTrue(value, "authorized_while_background");
    RequireTrue(value, "client_request_observation_mode");
    RequireFalse(value, "server_acceptance_observed");
    RequireFalse(value, "authoritative_execution_observed");
    ValidateClock(value, requireFixedTime: true);
}

static void ValidateScheduleIdentity(JsonElement value)
{
    RequireString(value, "schedule_id", ScheduleId);
    RequireString(value, "command_sequence_schema", ScheduleSchema);
    RequireString(value, "command_sequence_sha256", ScheduleSha256);
}

static void ValidateTrialIdentity(JsonElement value)
{
    RequireString(value, "single_motion_trial_schema", TrialSchema);
    RequireString(value, "single_motion_trial_sha256", TrialSha256);
    RequireString(
        value,
        "single_motion_trial_authority_scope",
        TrialAuthorityScope);
    RequireString(
        value,
        "single_motion_trial_authority_caveat",
        TrialAuthorityCaveat);
}

static void ValidateScheduleStep(JsonElement value, string runId, ExpectedStep expected)
{
    RequireString(value, "protocol", Protocol);
    ValidateScheduleIdentity(value);
    RequireString(value, "schedule_run_id", runId);
    RequireInt32(value, "schedule_tick", expected.Tick);
    RequireInt32(value, "client_fixed_substep", expected.Tick * 10);
    RequireInt32(value, "fixed_substeps_per_schedule_tick", 10);
    RequireString(value, "label", expected.Label);
    RequireTrue(value, "move_accepted_locally");
    RequireFalse(value, "server_acceptance_observed");
    var velocity = value.GetProperty("velocity_command_xyz");
    if (velocity.ValueKind != JsonValueKind.Array || velocity.GetArrayLength() != 3 ||
        velocity[0].GetSingle() != expected.Forward ||
        velocity[1].GetSingle() != expected.Strafe ||
        velocity[2].GetSingle() != expected.Yaw)
    {
        throw new InvalidDataException($"schedule_step {expected.Tick} velocity mismatch");
    }
    if (expected.MoveIndex is null)
        RequireNull(value, "move_index");
    else
        RequireInt32(value, "move_index", expected.MoveIndex.Value);
}

static void ValidateRenderedMarker(
    JsonElement value,
    string runId,
    ExpectedMarker expected)
{
    RequireString(value, "protocol", Protocol);
    ValidateScheduleIdentity(value);
    RequireString(value, "marker_schema", MarkerSchema);
    RequireString(value, "render_binding", MarkerRenderBinding);
    RequireString(value, "transition", MarkerTransition);
    RequireString(value, "schedule_run_id", runId);
    RequireInt32(value, "schedule_tick", expected.Tick);
    RequireInt32(value, "client_fixed_substep", expected.Tick * 10);
    RequireString(value, "selector", expected.Selector);
    RequireString(value, "command_identity", expected.CommandIdentity);
    RequireString(value, "marker_state", "post");
    RequireTrue(value, "marker_persists_after_edge");
    RequireFalse(value, "server_acceptance_observed");
    var region = value.GetProperty("region_px");
    RequireInt32(region, "x", 8 + expected.Index * 10);
    RequireInt32(region, "y", 8);
    RequireInt32(region, "width", 8);
    RequireInt32(region, "height", 8);
    RequireRgb(value.GetProperty("pre_rgb"), 0, 0, 0, "pre_rgb");
    RequireRgb(value.GetProperty("post_rgb"), 255, 0, 255, "post_rgb");
}

static void RequireRgb(JsonElement value, int red, int green, int blue, string name)
{
    if (value.ValueKind != JsonValueKind.Array || value.GetArrayLength() != 3 ||
        !value[0].TryGetInt32(out var actualRed) || actualRed != red ||
        !value[1].TryGetInt32(out var actualGreen) || actualGreen != green ||
        !value[2].TryGetInt32(out var actualBlue) || actualBlue != blue)
    {
        throw new InvalidDataException($"expected exact {name} [{red},{green},{blue}]");
    }
}

static void ValidateScheduleEnd(JsonElement value, string runId)
{
    RequireString(value, "protocol", Protocol);
    ValidateScheduleIdentity(value);
    RequireString(value, "schedule_run_id", runId);
    RequireTrue(value, "complete");
    RequireString(value, "reason", "complete");
    RequireInt32(value, "schedule_tick", 2600);
    RequireInt32(value, "client_fixed_substep", 26009);
    RequireInt32(value, "move_send_completed_count", 8);
    RequireTrue(value, "final_neutral_send_observed");
    RequireTrue(value, "authorized_while_background");
    RequireFalse(value, "server_acceptance_observed");
}

static void ValidateInitialState(
    JsonElement value,
    string roundIdentitySha256,
    string initialStateSha256)
{
    if (value.ValueKind != JsonValueKind.Object)
        throw new InvalidDataException("single-motion initial state was not an object");
    RequireString(value, "schema", "rek.single_motion_initial_state.v1");
    RequireString(value, "round_identity_sha256", roundIdentitySha256);
    _ = RequireHexString(value, "session_id_sha256", 64);
    _ = RequireHexString(value, "endpoint_sha256", 64);
    _ = RequireInt32Value(value, "fight_epoch");

    var round = value.GetProperty("round");
    _ = RequireInt32Value(round, "number");
    RequirePositiveFiniteNumber(round, "duration");
    RequirePositiveFiniteNumber(round, "time_remaining");
    RequireTrue(round, "active");
    RequireFalse(round, "redo");
    RequireIntArray(round, "clean_hits", new[] { 0, 0 });
    RequireIntArray(round, "falls", new[] { 0, 0 });
    _ = RequireNonemptyString(round, "result");
    _ = RequireInt32Value(round, "result_value");
    _ = RequireInt32Value(round, "winner_index");
    RequireFalse(round, "knockout");

    var fight = value.GetProperty("fight");
    _ = RequireNonemptyString(fight, "format");
    _ = RequireInt32Value(fight, "format_value");
    _ = RequireInt32Value(fight, "current_round");
    RequireIntArrayLength(fight, "rounds_won", 2);
    _ = RequireNonemptyString(fight, "result");
    _ = RequireInt32Value(fight, "result_value");
    _ = RequireInt32Value(fight, "winner_index");

    var input = value.GetProperty("input");
    _ = RequireInt32Value(input, "network_index");
    RequireTrue(input, "network_initialized");
    RequireTrue(input, "active");
    RequireFalse(input, "punching");
    RequireFalse(input, "recovering");
    RequireFloatVector(input, "velocity_command_xyz", 0f, 0f, 0f);
    RequireFalse(input, "pending_move");
    RequireFalse(input, "pending_special");
    RequireFalse(input, "pending_estop");
    RequireFalse(input, "action_playing");
    if (!input.TryGetProperty("action_clip", out var actionClip) ||
        actionClip.ValueKind is not (JsonValueKind.Null or JsonValueKind.String))
    {
        throw new InvalidDataException("initial action_clip was neither null nor a string");
    }
    RequireFiniteNumber(input, "action_clip_frame");
    RequireFiniteNumber(input, "action_clip_fps");

    ValidateInitialFighter(value.GetProperty("fighter_0"), "fighter_0");
    ValidateInitialFighter(value.GetProperty("fighter_1"), "fighter_1");
    ValidateClock(value, requireFixedTime: true);

    var observedHash = Convert.ToHexString(SHA256.HashData(
        Encoding.UTF8.GetBytes(value.GetRawText()))).ToLowerInvariant();
    if (!string.Equals(observedHash, initialStateSha256, StringComparison.Ordinal))
        throw new InvalidDataException("single-motion initial-state SHA-256 mismatch");
}

static void ValidateInitialFighter(JsonElement value, string name)
{
    RequireTrue(value, "visual_only");
    _ = RequireBooleanValue(value, "player_controlled");
    _ = RequireBooleanValue(value, "falling");
    _ = RequireBooleanValue(value, "fallen");
    _ = RequireBooleanValue(value, "dampened");
    _ = RequireBooleanValue(value, "resetting");
    _ = RequireBooleanValue(value, "motor_shutdown");
    RequireFiniteNumber(value, "tilt_angle");
    _ = RequireInt32Value(value, "floor_contact_count");
    RequireFiniteNumberArray(value, "root_position_xyz", 3, name);
    RequireFiniteNumberArray(value, "root_rotation_xyzw", 4, name);
    RequireFiniteNumberArray(value, "root_linear_velocity_xyz", 3, name);
    RequireFiniteNumberArray(value, "root_angular_velocity_xyz", 3, name);
}

static void ValidateTrialCommandEdge(
    JsonElement value,
    string runId,
    string freshRoundRequestId,
    ExpectedTrialSelector selector,
    string roundIdentitySha256,
    string initialStateSha256,
    string phase)
{
    ValidateTrialEventIdentity(
        value,
        runId,
        freshRoundRequestId,
        selector,
        roundIdentitySha256,
        initialStateSha256);
    RequireString(value, "edge", phase);
    var expectedTick = phase == "action" ? TrialActionTick : TrialLocomotionReleaseTick;
    RequireInt32(value, "trial_tick", expectedTick);
    RequireInt32(
        value,
        "client_fixed_substep",
        expectedTick * TrialFixedSubstepsPerTick);
    if (phase == "action")
        RequireFloatVector(value, "velocity_command_xyz", selector.Forward, selector.Strafe, selector.Yaw);
    else
        RequireFloatVector(value, "velocity_command_xyz", 0f, 0f, 0f);
    if (selector.MoveIndex is null)
        RequireNull(value, "move_index");
    else
        RequireInt32(value, "move_index", selector.MoveIndex.Value);
    RequireTrue(value, "local_command_value_set");
    RequireFalse(value, "client_request_edge_observed");
    ValidateClock(value, requireFixedTime: true);
}

static void ValidateTrialClientRequest(
    JsonElement value,
    string runId,
    string freshRoundRequestId,
    ExpectedTrialSelector selector,
    string roundIdentitySha256,
    string initialStateSha256,
    string phase)
{
    ValidateTrialEventIdentity(
        value,
        runId,
        freshRoundRequestId,
        selector,
        roundIdentitySha256,
        initialStateSha256);
    var isMove = !selector.IsLocomotion && phase == "action";
    RequireString(value, "request_kind", isMove ? "move" : "velocity");
    RequireString(value, "request_phase", phase);
    var commandTick = phase switch
    {
        "neutral_pre_roll" => 0,
        "action" => TrialActionTick,
        "release" => TrialLocomotionReleaseTick,
        _ => throw new InvalidDataException($"unknown trial request phase {phase}"),
    };
    RequireInt32(value, "command_edge_trial_tick", commandTick);
    var observedTick = RequireInt32Value(value, "observed_trial_tick");
    var observedSubstep = RequireInt32Value(value, "observed_client_fixed_substep");
    if (observedTick != observedSubstep / TrialFixedSubstepsPerTick ||
        observedSubstep < 0 ||
        observedSubstep >= TrialDurationTicks * TrialFixedSubstepsPerTick)
    {
        throw new InvalidDataException("trial client-request observation was off the fixed-step grid");
    }
    var validPhaseWindow = phase switch
    {
        "neutral_pre_roll" => observedTick is >= 0 and < TrialActionTick,
        "action" when selector.IsLocomotion =>
            observedTick is >= TrialActionTick and < TrialLocomotionReleaseTick,
        "action" => observedTick is >= TrialActionTick and <= TrialFinalTick,
        "release" => observedTick is >= TrialLocomotionReleaseTick and <= TrialFinalTick,
        _ => false,
    };
    if (!validPhaseWindow)
        throw new InvalidDataException($"trial {phase} request was outside its allowed phase window");

    if (phase == "action")
        RequireFloatVector(value, "velocity_command_xyz", selector.Forward, selector.Strafe, selector.Yaw);
    else
        RequireFloatVector(value, "velocity_command_xyz", 0f, 0f, 0f);
    if (isMove)
        RequireInt32(value, "move_index", selector.MoveIndex!.Value);
    else
        RequireNull(value, "move_index");
    RequireString(
        value,
        "send_method",
        isMove
            ? "RobotInputController.SendMoveEvent"
            : "RobotInputController.SendVelocityCommand");
    RequireTrue(value, "send_method_returned");
    RequireTrue(value, "client_request_edge_observed");
    ValidateClock(value, requireFixedTime: false);
}

static void ValidateTrialEnd(
    JsonElement value,
    string runId,
    string freshRoundRequestId,
    string roundIdentitySha256,
    string initialStateSha256,
    ExpectedTrialSelector selector)
{
    ValidateTrialEventIdentity(
        value,
        runId,
        freshRoundRequestId,
        selector,
        roundIdentitySha256,
        initialStateSha256);
    ValidateInitialState(
        value.GetProperty("initial_state"),
        roundIdentitySha256,
        initialStateSha256);
    RequireInt32(value, "trial_tick", TrialFinalTick);
    RequireInt32(
        value,
        "client_fixed_substep",
        TrialDurationTicks * TrialFixedSubstepsPerTick - 1);
    RequireInt32(
        value,
        "fixed_substeps_per_trial_tick",
        TrialFixedSubstepsPerTick);
    RequireTrue(value, "neutral_pre_roll_send_observed");
    RequireInt32(value, "non_neutral_edge_count", 1);
    RequireInt32(value, "release_edge_count", selector.IsLocomotion ? 1 : 0);
    RequireInt32(
        value,
        "velocity_press_send_completed_count",
        selector.IsLocomotion ? 1 : 0);
    RequireInt32(
        value,
        "velocity_release_send_completed_count",
        selector.IsLocomotion ? 1 : 0);
    RequireInt32(value, "move_send_completed_count", selector.IsLocomotion ? 0 : 1);
    RequireTrue(value, "round_consumed");
    RequireTrue(value, "complete");
    RequireString(value, "reason", "complete");
    RequireTrue(value, "authorized_while_background");
    RequireTrue(value, "client_request_edges_observed");
    ValidateClock(value, requireFixedTime: true);
}

static void ValidateTrialEventIdentity(
    JsonElement value,
    string runId,
    string freshRoundRequestId,
    ExpectedTrialSelector selector,
    string? roundIdentitySha256 = null,
    string? initialStateSha256 = null)
{
    RequireString(value, "protocol", Protocol);
    RequireString(value, "single_motion_trial_schema", TrialSchema);
    RequireString(value, "single_motion_trial_sha256", TrialSha256);
    RequireString(value, "authority_scope", TrialAuthorityScope);
    RequireString(value, "authority_caveat", TrialAuthorityCaveat);
    RequireString(value, "single_motion_trial_run_id", runId);
    RequireString(value, "fresh_round_request_id", freshRoundRequestId);
    RequireString(value, "selector", selector.Selector);
    RequireString(value, "selector_kind", selector.Kind);
    RequireString(value, "command_identity", selector.CommandIdentity);
    if (roundIdentitySha256 is null)
        _ = RequireHexString(value, "round_identity_sha256", 64);
    else
        RequireString(value, "round_identity_sha256", roundIdentitySha256);
    if (initialStateSha256 is null)
        _ = RequireHexString(value, "initial_state_sha256", 64);
    else
        RequireString(value, "initial_state_sha256", initialStateSha256);
    RequireFalse(value, "server_acceptance_observed");
    RequireFalse(value, "authoritative_execution_observed");
}

static void ValidateClock(JsonElement value, bool requireFixedTime)
{
    var utc = RequireNonemptyString(value, "utc");
    if (!DateTimeOffset.TryParse(utc, out _))
        throw new InvalidDataException("utc was not a valid timestamp");
    if (RequireInt64Value(value, "stopwatch_timestamp_ticks") < 0 ||
        RequireInt64Value(value, "stopwatch_frequency_hz") <= 0)
    {
        throw new InvalidDataException("Stopwatch clock fields were invalid");
    }
    if (RequireInt32Value(value, "unity_frame") < 0)
        throw new InvalidDataException("unity_frame was negative");
    RequireFiniteNumber(value, requireFixedTime ? "unity_fixed_time" : "unity_time");
}

static string NewRequestId(string prefix) => $"{prefix}-{Guid.NewGuid():N}";

static string? OptionalString(JsonElement parent, string name) =>
    parent.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
        ? value.GetString()
        : null;

static bool OptionalTrue(JsonElement parent, string name) =>
    parent.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.True;

static string RequireNonemptyString(JsonElement parent, string name)
{
    var value = OptionalString(parent, name);
    if (string.IsNullOrEmpty(value))
        throw new InvalidDataException($"expected nonempty string {name}");
    return value;
}

static void RequireString(JsonElement parent, string name, string expected)
{
    if (OptionalString(parent, name) != expected)
        throw new InvalidDataException($"expected {name}={expected}");
}

static void RequireTrue(JsonElement parent, string name) => RequireBool(parent, name, true);

static void RequireFalse(JsonElement parent, string name) => RequireBool(parent, name, false);

static void RequireBool(JsonElement parent, string name, bool expected)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != (expected ? JsonValueKind.True : JsonValueKind.False))
    {
        throw new InvalidDataException($"expected {name}={expected.ToString().ToLowerInvariant()}");
    }
}

static bool RequireBooleanValue(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
    {
        throw new InvalidDataException($"expected boolean {name}");
    }
    return value.GetBoolean();
}

static bool? RequireNullableBooleanValue(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value))
        throw new InvalidDataException($"expected nullable boolean {name}");
    if (value.ValueKind == JsonValueKind.Null)
        return null;
    if (value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        throw new InvalidDataException($"expected nullable boolean {name}");
    return value.GetBoolean();
}

static long RequireInt64Value(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) || !value.TryGetInt64(out var actual))
        throw new InvalidDataException($"expected integer {name}");
    return actual;
}

static int RequireInt32Value(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) || !value.TryGetInt32(out var actual))
        throw new InvalidDataException($"expected 32-bit integer {name}");
    return actual;
}

static string RequireHexString(JsonElement parent, string name, int length)
{
    var value = RequireNonemptyString(parent, name);
    if (value.Length != length || value.Any(character =>
            character is not (>= '0' and <= '9' or >= 'a' and <= 'f')))
    {
        throw new InvalidDataException($"expected {name} to be {length} lowercase hexadecimal characters");
    }
    return value;
}

static void RequireStringArray(JsonElement parent, string name, IReadOnlyList<string> expected)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Array ||
        value.GetArrayLength() != expected.Count)
    {
        throw new InvalidDataException($"expected exact string array {name}");
    }
    for (var index = 0; index < expected.Count; index++)
    {
        if (value[index].ValueKind != JsonValueKind.String ||
            value[index].GetString() != expected[index])
        {
            throw new InvalidDataException($"expected exact string array {name}");
        }
    }
}

static string[] RequireNonemptyUniqueStringArray(
    JsonElement parent,
    string name,
    int expectedLength)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Array ||
        value.GetArrayLength() != expectedLength)
    {
        throw new InvalidDataException(
            $"expected string array {name} length {expectedLength}");
    }
    var result = new string[expectedLength];
    var distinct = new HashSet<string>(StringComparer.Ordinal);
    for (var index = 0; index < expectedLength; index++)
    {
        if (value[index].ValueKind != JsonValueKind.String ||
            string.IsNullOrWhiteSpace(value[index].GetString()))
        {
            throw new InvalidDataException($"expected nonempty string array {name}");
        }
        result[index] = value[index].GetString()!;
        if (!distinct.Add(result[index]))
            throw new InvalidDataException($"expected unique string array {name}");
    }
    return result;
}

static string HashUtf8Text(string value) => Convert.ToHexString(
    SHA256.HashData(Encoding.UTF8.GetBytes(value))).ToLowerInvariant();

static void RequireIntArray(JsonElement parent, string name, IReadOnlyList<int> expected)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Array ||
        value.GetArrayLength() != expected.Count)
    {
        throw new InvalidDataException($"expected exact integer array {name}");
    }
    for (var index = 0; index < expected.Count; index++)
    {
        if (!value[index].TryGetInt32(out var actual) || actual != expected[index])
            throw new InvalidDataException($"expected exact integer array {name}");
    }
}

static void RequireIntArrayLength(JsonElement parent, string name, int expectedLength)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Array ||
        value.GetArrayLength() != expectedLength)
    {
        throw new InvalidDataException($"expected integer array {name} length {expectedLength}");
    }
    foreach (var element in value.EnumerateArray())
    {
        if (!element.TryGetInt32(out _))
            throw new InvalidDataException($"expected integer array {name}");
    }
}

static void RequireFiniteNumberArray(
    JsonElement parent,
    string name,
    int expectedLength,
    string context)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Array ||
        value.GetArrayLength() != expectedLength)
    {
        throw new InvalidDataException($"expected {context}.{name} length {expectedLength}");
    }
    foreach (var element in value.EnumerateArray())
    {
        if (element.ValueKind != JsonValueKind.Number || !double.IsFinite(element.GetDouble()))
            throw new InvalidDataException($"expected finite {context}.{name}");
    }
}

static void RequireFloatVector(
    JsonElement parent,
    string name,
    float expectedX,
    float expectedY,
    float expectedZ)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Array || value.GetArrayLength() != 3 ||
        !SameFloatBits(value[0].GetSingle(), expectedX) ||
        !SameFloatBits(value[1].GetSingle(), expectedY) ||
        !SameFloatBits(value[2].GetSingle(), expectedZ))
    {
        throw new InvalidDataException(
            $"expected exact {name} [{expectedX:R},{expectedY:R},{expectedZ:R}]");
    }
}

static bool SameFloatBits(float left, float right) =>
    BitConverter.SingleToInt32Bits(left) == BitConverter.SingleToInt32Bits(right);

static void RequirePositiveFiniteNumber(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Number ||
        !double.IsFinite(value.GetDouble()) || value.GetDouble() <= 0.0)
    {
        throw new InvalidDataException($"expected positive finite number {name}");
    }
}

static void RequireFiniteNumber(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Number || !double.IsFinite(value.GetDouble()))
    {
        throw new InvalidDataException($"expected finite number {name}");
    }
}

static void RequireFiniteExactNumber(JsonElement parent, string name, double expected)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind != JsonValueKind.Number ||
        !double.IsFinite(value.GetDouble()) || value.GetDouble() != expected)
    {
        throw new InvalidDataException($"expected {name}={expected:R}");
    }
}

static void RequireInt64(JsonElement parent, string name, long expected)
{
    if (RequireInt64Value(parent, name) != expected)
        throw new InvalidDataException($"expected {name}={expected}");
}

static void RequireInt32(JsonElement parent, string name, int expected)
{
    if (!parent.TryGetProperty(name, out var value) ||
        !value.TryGetInt32(out var actual) || actual != expected)
    {
        throw new InvalidDataException($"expected {name}={expected}");
    }
}

static void RequireNull(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Null)
        throw new InvalidDataException($"expected {name}=null");
}

static void RequireNullOrString(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind is not (JsonValueKind.Null or JsonValueKind.String))
    {
        throw new InvalidDataException($"expected {name} to be null or string");
    }
}

internal sealed record ExpectedStep(
    int Tick,
    string Label,
    float Forward,
    float Strafe,
    float Yaw,
    int? MoveIndex);

internal sealed record ExpectedMarker(
    int Index,
    int Tick,
    string Selector,
    string CommandIdentity);

internal sealed record ExpectedTrialSelector(
    string Selector,
    string Kind,
    float Forward,
    float Strafe,
    float Yaw,
    int? MoveIndex,
    string CommandIdentity)
{
    internal bool IsLocomotion => MoveIndex is null;
}

internal sealed record PipeServerProof(uint ProcessId, string Executable);

internal static class NativeMethods
{
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    internal static extern bool GetNamedPipeServerProcessId(
        IntPtr pipe,
        out uint serverProcessId);
}
