using System.IO.Pipes;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using RekUiBridgeAgent;

const string PipeName = "rek-ui-bridge-v1";
const string Protocol = "rek.ui_bridge.v1";
const string ExpectedApplicationVersion = "0.0.119";
const string ExpectedUnityVersion = "6000.5.8f1";
const string ExpectedGameAssemblySha256 =
    "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
const string ExpectedMetadataSha256 =
    "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
const string ExpectedSharedAssets0Sha256 =
    "37f7a476c56caae37f5a04d4fa1acf5954fdc2b90f20f521830369ecff05f355";
const string ExpectedBridgeVersion = "0.4.6";
const string ExpectedBridgeSha256 =
    "6a46af475041f7fc42bb38d51e294699bcc6f98b823e56fea356878adcde3149";
const string ScheduleId = "rek.private_bot1.baseline.v1";
const string ScheduleSchema = "rek.client_fixed.command_schedule.v2";
const string ScheduleSha256 =
    "39aaab9c3156e8f4d114daac4d4328257b81230ec8b8a372ad2739d38754ec0d";
const string T800BoneSignatureSha256 =
    "ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab";
const string G1BoneSignatureSha256 =
    "9d18e697233d9578b398fbe849cd59d65cb27a5c2223b2602db66a82a410e987";
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
    "7254c2e9291520a7d78967193f05c3b8d1fce32afa63926d2388e51ed6764ca0";
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
    "build_pinned_serialized_robot_move_asset_metadata_not_measured_input_to_completion_timing";
const string ContinuousRoundRestartLimitation =
    "build_pinned_post_fight_continue_restarts_only_after_win_and_exits_to_lobby_after_loss";
const string ContinuousRoundRestartStaticEvidence =
    "GameMenuController.HandlePostFightContinue_rva_0x23aae90_branches_on_postFightIsWinner_false_ExitToLobby_true_SendPostFightIntent_stay_true";
const string ContinuousRecoveryGuardProvenance =
    "build_pinned_t800_AIOpponentController.DriveRecovery_rva_0x2367430_fallen_not_dampened_Dampen_4_then_Straighten_1_once_then_RecoveryArmed_SuggestedGetUpOrientation_2_or_3_unitree_g1_recovery_unproven_fail_closed";
const string ContinuousFaultEStopProvenance =
    "build_pinned_t800_AIOpponentController.UpdateFaultEStopCycle_rva_0x23680e0_motorShutdownHold_faultEStopDelay_then_0.5_second_estop_hold_unitree_g1_hasEStop_false_fail_closed";
const string ContinuousDampenGuard = "fallen_and_not_dampened";
const string ContinuousStraightenGuard =
    "fallen_and_dampened_and_not_already_issued";
const string ContinuousOpponentRuntimeRequirement =
    "exact_homogeneous_t800_or_unitree_g1_runtime_bone_signatures_required_semantic_robot_ids_recorded_but_not_trusted";
const string ContinuousT800RecoveryMode =
    "build_pinned_t800_special_commands_and_fault_estop_enabled";
const string ContinuousG1RecoveryMode =
    "unitree_g1_hasSpecialCommands_false_hasEStop_false_recovery_semantics_unproven_fail_closed";
const string AttackZoneSchema = "rek.attack_zone_trial.v1";
const string AttackZoneSha256 =
    "195ff18ba30097fa5575b36c6d9fdd4a4a2499e73803d0e04c7e53203cb530cf";
const string AttackZoneAuthorityScope =
    "client_request_edges_and_local_observations_only";
const string AttackZoneAuthorityCaveat =
    "client request edge and local observations only; server acceptance, authoritative execution, and causal hit attribution are unknown";
const string AttackZoneIsolationProof =
    "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98";
const string AttackZoneRecorderVersion = "0.7.2";
const string AttackZoneRecorderSha256 =
    "a19f619c83eeecf9c6ccf79adf339be1f7f1cca8e3cd622f80616f268aaffa95";
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
    args[0] is not ("state" or "enter-private" or "exit-lost" or "schedule" or "trial" or "controller" or "g1-held"))
{
    Console.Error.WriteLine(
        "usage: RekUiPipeClient state [output.jsonl] [timeout_seconds] | " +
        "enter-private|exit-lost|schedule|g1-held output.jsonl [timeout_seconds] | " +
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
    : mode == "g1-held" ? 150 : mode == "schedule" ? 90 :
        mode is "enter-private" or "exit-lost" or "trial" ? 60 :
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
    string? completedControllerRuntimeModel = null;
    string? completedG1HeldRunId = null;
    string? completedG1HeldFreshRoundRequestId = null;
    string? completedG1HeldRoundIdentitySha256 = null;
    int? completedG1HeldFinalTick = null;
    int? completedG1HeldFinalSubstep = null;
    string? g1HeldPartialReason = null;
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
        else if (mode == "g1-held")
        {
            var freshRoundRequestId = await StartFreshPrivateRound(
                connectionId,
                deadline.Token,
                requireStrictParityPairing: false);
            string g1HeldRunId;
            string g1HeldRoundIdentitySha256;
            using (var start = await RequireAcceptedCommand(
                       "StartG1HeldInputSchedule",
                       "g1_held_input_schedule_started",
                       expectedApplied: true,
                       expectedRequestIssued: false,
                       deadline.Token))
            {
                RequireInt64(start.RootElement, "lease_connection_id", connectionId);
                ValidateG1HeldIdentity(start.RootElement);
                RequireTrue(start.RootElement, "g1_held_schedule_running");
                RequireTrue(start.RootElement, "g1_held_schedule_round_capacity_proven");
                var roundDuration = start.RootElement.GetProperty(
                    "g1_held_schedule_round_duration_seconds").GetSingle();
                var timeRemaining = start.RootElement.GetProperty(
                    "g1_held_schedule_initial_time_remaining_seconds").GetSingle();
                if (!G1HeldInputScheduleContract.HasRoundCapacity(
                        roundDuration,
                        timeRemaining))
                {
                    throw new InvalidDataException(
                        "G1 held-input start did not prove fresh-round capacity");
                }
                RequireFalse(start.RootElement, "fresh_round_armed");
                RequireNull(start.RootElement, "fresh_round_invalid_reason");
                g1HeldRunId = RequireHexString(
                    start.RootElement,
                    "g1_held_schedule_run_id",
                    32);
                RequireString(
                    start.RootElement,
                    "g1_held_schedule_fresh_round_request_id",
                    freshRoundRequestId);
                g1HeldRoundIdentitySha256 = RequireHexString(
                    start.RootElement,
                    "g1_held_schedule_round_identity_sha256",
                    64);
                if (ValidateSupportedMeasuredPairing(
                        start.RootElement.GetProperty("measured_pairing")) != "g1")
                {
                    throw new InvalidDataException(
                        "G1 held-input schedule requires exact G1/G1 runtime pairing");
                }
                completedG1HeldRunId = g1HeldRunId;
                completedG1HeldFreshRoundRequestId = freshRoundRequestId;
                completedG1HeldRoundIdentitySha256 = g1HeldRoundIdentitySha256;
            }

            var nextEventSequence = 1;
            var nextScheduleTick = 0;
            var kickExecuteReturnSeen = new bool[G1HeldInputScheduleContract.KickProbes.Length];
            var kickTerminalSeen = new bool[G1HeldInputScheduleContract.KickProbes.Length];
            var kickSummarySeen = new bool[G1HeldInputScheduleContract.KickProbes.Length];
            var yawPreemptionSeen = new bool[G1HeldInputScheduleContract.KickProbes.Length];
            var translationReleaseSeen =
                new bool[G1HeldInputScheduleContract.KickProbes.Length];
            var kickFixedObservationCounts =
                new int[G1HeldInputScheduleContract.KickProbes.Length];
            var kickEventReconciler = new G1HeldEventReconciler();
            while (true)
            {
                using var message = await ReadMessage(deadline.Token);
                var eventName = OptionalString(message.RootElement, "event");
                if (eventName == "g1_held_schedule_end")
                {
                    for (var ordinal = 0;
                         ordinal < G1HeldInputScheduleContract.KickProbes.Length;
                         ordinal++)
                    {
                        if (kickEventReconciler.TerminalSeen(ordinal) !=
                                kickTerminalSeen[ordinal] ||
                            kickEventReconciler.SummarySeen(ordinal) !=
                                kickSummarySeen[ordinal])
                        {
                            throw new InvalidDataException(
                                "G1 event reconciliation differed from validated coverage");
                        }
                    }
                    var endComplete = ValidateG1HeldEnd(
                        message.RootElement,
                        g1HeldRunId,
                        freshRoundRequestId,
                        g1HeldRoundIdentitySha256,
                        nextScheduleTick,
                        kickExecuteReturnSeen,
                        kickTerminalSeen,
                        kickSummarySeen,
                        yawPreemptionSeen,
                        translationReleaseSeen,
                        kickFixedObservationCounts);
                    completedG1HeldFinalTick = RequireInt32Value(
                        message.RootElement,
                        "schedule_tick");
                    completedG1HeldFinalSubstep = RequireInt32Value(
                        message.RootElement,
                        "client_fixed_substep");
                    if (!endComplete)
                    {
                        g1HeldPartialReason = RequireNonemptyString(
                            message.RootElement,
                            "reason");
                    }
                    resultJson = message.RootElement.GetRawText();
                    break;
                }
                if (eventName?.StartsWith("g1_", StringComparison.Ordinal) != true)
                    continue;

                ValidateG1HeldEventIdentity(
                    message.RootElement,
                    g1HeldRunId,
                    freshRoundRequestId,
                    g1HeldRoundIdentitySha256,
                    nextEventSequence);
                nextEventSequence++;
                switch (eventName)
                {
                    case "g1_held_schedule_tick":
                        ValidateG1HeldTick(message.RootElement, nextScheduleTick);
                        nextScheduleTick++;
                        break;
                    case "g1_yaw_kick_preemption":
                        kickEventReconciler.ObserveProbeEvent(
                            eventName,
                            RequireG1Probe(
                                message.RootElement.GetProperty("detail"),
                                G1KickProbeKind.YawPreempted).Ordinal);
                        ValidateG1YawPreemption(
                            message.RootElement,
                            yawPreemptionSeen);
                        break;
                    case "g1_translation_release":
                        kickEventReconciler.ObserveProbeEvent(
                            eventName,
                            RequireG1Probe(
                                message.RootElement.GetProperty("detail"),
                                G1KickProbeKind.TranslationHeld).Ordinal);
                        ValidateG1TranslationRelease(
                            message.RootElement,
                            translationReleaseSeen);
                        break;
                    case "g1_kick_request_lifecycle":
                    {
                        var probeOrdinal = RequireG1Probe(
                            message.RootElement.GetProperty("detail"),
                            null).Ordinal;
                        kickEventReconciler.ObserveProbeEvent(eventName, probeOrdinal);
                        var terminalWasSeen = kickTerminalSeen[probeOrdinal];
                        ValidateG1KickRequestLifecycle(
                            message.RootElement,
                            kickExecuteReturnSeen,
                            kickTerminalSeen);
                        if (!terminalWasSeen && kickTerminalSeen[probeOrdinal])
                            kickEventReconciler.ObserveTerminal(probeOrdinal);
                        break;
                    }
                    case "g1_kick_local_state_transition":
                        kickEventReconciler.ObserveProbeEvent(
                            eventName,
                            RequireG1Probe(
                                message.RootElement.GetProperty("detail"),
                                null).Ordinal);
                        ValidateG1KickLocalStateTransition(message.RootElement);
                        break;
                    case "g1_kick_fixed_observation":
                        kickEventReconciler.ObserveProbeEvent(
                            eventName,
                            RequireG1Probe(
                                message.RootElement.GetProperty("detail"),
                                null).Ordinal);
                        ValidateG1KickFixedObservation(
                            message.RootElement,
                            kickFixedObservationCounts);
                        break;
                    case "g1_kick_measurement_summary":
                    {
                        var detail = message.RootElement.GetProperty("detail");
                        var probeOrdinal = RequireG1Probe(detail, null).Ordinal;
                        var observationWindowComplete = RequireBooleanValue(
                            detail,
                            "observation_window_complete");
                        ValidateG1KickMeasurementSummary(
                            message.RootElement,
                            kickSummarySeen,
                            kickFixedObservationCounts);
                        kickEventReconciler.ObserveSummary(
                            probeOrdinal,
                            observationWindowComplete);
                        break;
                    }
                    case "g1_yaw_ramp_update":
                        ValidateG1YawRampUpdate(message.RootElement);
                        break;
                    case "g1_kick_dispatch_opportunity":
                        kickEventReconciler.ObserveProbeEvent(
                            eventName,
                            RequireG1Probe(
                                message.RootElement.GetProperty("detail"),
                                null).Ordinal);
                        ValidateG1KickDispatchOpportunity(message.RootElement);
                        break;
                    case "g1_velocity_request_lifecycle":
                        ValidateG1VelocityRequestLifecycle(message.RootElement);
                        break;
                    case "g1_unexpected_local_request_blocked":
                        ValidateG1UnexpectedRequestBlocked(message.RootElement);
                        break;
                    default:
                        throw new InvalidDataException(
                            $"unrecognized G1 held-input event {eventName}");
                }
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
                completedControllerRuntimeModel = ValidateContinuousMeasuredPairing(
                    start.RootElement.GetProperty("measured_pairing"));
                RequireString(
                    start.RootElement,
                    "continuous_controller_runtime_model",
                    completedControllerRuntimeModel);
                RequireString(
                    start.RootElement,
                    "continuous_controller_recovery_mode",
                    RecoveryModeForRuntimeModel(completedControllerRuntimeModel));
                RequireIntArray(
                    start.RootElement,
                    "continuous_controller_attack_move_indices",
                    AttackMoveIndicesForRuntimeModel(completedControllerRuntimeModel));
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
                                completedControllerRunId,
                                completedControllerRuntimeModel);
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
                                    completedControllerRunId,
                                    completedControllerRuntimeModel);
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
        RequireFalse(releasedControl, "g1_held_schedule_running");
        RequireFalse(releasedControl, "g1_held_schedule_authorized_while_background");
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
        else if (mode == "g1-held")
        {
            RequireString(
                releasedControl,
                "g1_held_schedule_run_id",
                completedG1HeldRunId ?? throw new InvalidDataException(
                    "completed G1 held-input run ID was unavailable"));
            RequireString(
                releasedControl,
                "g1_held_schedule_fresh_round_request_id",
                completedG1HeldFreshRoundRequestId ?? throw new InvalidDataException(
                    "completed G1 held-input fresh-round request ID was unavailable"));
            RequireString(
                releasedControl,
                "g1_held_schedule_round_identity_sha256",
                completedG1HeldRoundIdentitySha256 ?? throw new InvalidDataException(
                    "completed G1 held-input round identity was unavailable"));
            RequireInt32(
                releasedControl,
                "g1_held_schedule_tick",
                completedG1HeldFinalTick ?? throw new InvalidDataException(
                    "completed G1 held-input final tick was unavailable"));
            RequireInt32(
                releasedControl,
                "g1_held_schedule_client_fixed_substep",
                completedG1HeldFinalSubstep ?? throw new InvalidDataException(
                    "completed G1 held-input final substep was unavailable"));
        }
    }

    if (g1HeldPartialReason is not null)
    {
        await WriteClientResult("partial", g1HeldPartialReason);
        await PublishTranscript();
        Console.WriteLine(resultJson);
        Console.Error.WriteLine(
            $"G1 held-input schedule ended with partial coverage: {g1HeldPartialReason}");
        return 1;
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
        ValidateG1HeldAckIdentity(response.RootElement);
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

static void ValidateG1HeldIdentity(JsonElement value)
{
    RequireString(
        value,
        "g1_held_schedule_schema",
        G1HeldInputScheduleContract.Schema);
    RequireString(
        value,
        "g1_held_schedule_id",
        G1HeldInputScheduleContract.ScheduleId);
    RequireString(
        value,
        "g1_held_schedule_sha256",
        G1HeldInputScheduleContract.ExpectedSha256);
    RequireString(
        value,
        "g1_held_schedule_authority_scope",
        G1HeldInputScheduleContract.AuthorityScope);
    RequireString(
        value,
        "g1_held_schedule_authority_caveat",
        G1HeldInputScheduleContract.AuthorityCaveat);
}

static void ValidateG1HeldAckIdentity(JsonElement value)
{
    ValidateG1HeldIdentity(value);
    _ = RequireBooleanValue(value, "g1_held_schedule_running");
    ValidateOptionalHex(value, "g1_held_schedule_run_id", 32);
    ValidateOptionalString(value, "g1_held_schedule_fresh_round_request_id");
    ValidateOptionalHex(value, "g1_held_schedule_round_identity_sha256", 64);
    var tick = RequireInt32Value(value, "g1_held_schedule_tick");
    var substep = RequireInt32Value(value, "g1_held_schedule_client_fixed_substep");
    if (tick is < 0 or > G1HeldInputScheduleContract.FinalScheduleTick || substep < 0)
        throw new InvalidDataException("G1 held-input ack counters were invalid");
    RequireFiniteNumber(value, "g1_held_schedule_round_duration_seconds");
    RequireFiniteNumber(value, "g1_held_schedule_initial_time_remaining_seconds");
    RequireFiniteExactNumber(
        value,
        "g1_held_schedule_required_run_seconds",
        G1HeldInputScheduleContract.RequiredRunSeconds);
    RequireFiniteExactSingle(
        value,
        "g1_held_schedule_round_capacity_safety_seconds",
        G1HeldInputScheduleContract.RoundCapacitySafetySeconds);
    RequireFiniteExactNumber(
        value,
        "g1_held_schedule_required_round_capacity_seconds",
        G1HeldInputScheduleContract.RequiredRoundCapacitySeconds);
    _ = RequireBooleanValue(value, "g1_held_schedule_round_capacity_proven");
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
    CancellationToken cancellationToken,
    bool requireStrictParityPairing = true)
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
        ValidateG1HeldAckIdentity(response.RootElement);
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
        ValidatePrivateBotOne(
            state.RootElement,
            requireActiveRound: true,
            requireStrictParityPairing: requireStrictParityPairing);
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

static void ValidateG1HeldEventIdentity(
    JsonElement value,
    string runId,
    string freshRoundRequestId,
    string roundIdentitySha256,
    int expectedEventSequence)
{
    RequireString(value, "protocol", Protocol);
    RequireString(value, "g1_held_schedule_schema", G1HeldInputScheduleContract.Schema);
    RequireString(value, "g1_held_schedule_id", G1HeldInputScheduleContract.ScheduleId);
    RequireString(
        value,
        "g1_held_schedule_sha256",
        G1HeldInputScheduleContract.ExpectedSha256);
    RequireString(value, "g1_held_schedule_run_id", runId);
    RequireString(value, "fresh_round_request_id", freshRoundRequestId);
    RequireString(value, "round_identity_sha256", roundIdentitySha256);
    RequireInt32(value, "event_sequence", expectedEventSequence);
    var scheduleTick = RequireInt32Value(value, "schedule_tick");
    var fixedSubstep = RequireInt32Value(value, "client_fixed_substep");
    if (scheduleTick is < 0 or > G1HeldInputScheduleContract.FinalScheduleTick ||
        fixedSubstep < scheduleTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick ||
        fixedSubstep > (scheduleTick + 1) * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick)
    {
        throw new InvalidDataException("G1 held-input event clock was inconsistent");
    }
    RequireInt32(
        value,
        "fixed_substeps_per_schedule_tick",
        G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    RequireInt32(value, "schedule_rate_hz", G1HeldInputScheduleContract.ScheduleRateHz);
    RequireInt32(value, "unity_fixed_rate_hz", G1HeldInputScheduleContract.UnityFixedRateHz);
    RequireString(value, "authority_scope", G1HeldInputScheduleContract.AuthorityScope);
    RequireString(value, "authority_caveat", G1HeldInputScheduleContract.AuthorityCaveat);
    RequireTrue(value, "request_only");
    RequireString(value, "server_acceptance", "unknown");
    RequireFalse(value, "server_acceptance_observed");
    RequireFalse(value, "authoritative_execution_observed");
    RequireFalse(value, "global_input_emitted");
    ValidateG1RecorderCorrelation(
        value.GetProperty("recorder_correlation"),
        scheduleTick,
        fixedSubstep);
}

static void ValidateG1RecorderCorrelation(
    JsonElement value,
    int scheduleTick,
    int fixedSubstep)
{
    RequireString(value, "recorder_schema", "rek.private_ai.protocol.v7");
    RequireString(value, "clock", "client_fixed_tick_500hz");
    RequireInt32(value, "client_fixed_substep", fixedSubstep);
    RequireInt32(value, "schedule_tick", scheduleTick);
    if (value.GetProperty("unity_fixed_time").ValueKind != JsonValueKind.Number)
        throw new InvalidDataException("G1 recorder correlation fixed time was unavailable");
    RequireFalse(value, "pose_payload_in_pipe");
    RequireString(
        value,
        "pose_response_source",
        G1HeldInputScheduleContract.PoseResponseSource);
}

static void ValidateG1HeldTick(JsonElement value, int expectedTick)
{
    RequireInt32(value, "schedule_tick", expectedTick);
    RequireInt32(
        value,
        "client_fixed_substep",
        expectedTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    var expected = G1HeldInputScheduleContract.FrameAtTick(expectedTick);
    var previous = expectedTick == 0
        ? G1HeldInputScheduleContract.FrameAtTick(0) with
        {
            DesiredHeldMask = G1HeldMask.None,
            EffectiveHeldMask = G1HeldMask.None,
        }
        : G1HeldInputScheduleContract.FrameAtTick(expectedTick - 1);
    var detail = value.GetProperty("detail");
    RequireString(detail, "phase", expected.Phase);
    RequireInt32(detail, "desired_held_mask", (byte)expected.DesiredHeldMask);
    RequireStringArray(
        detail,
        "desired_held",
        G1HeldInputScheduleContract.HeldNames(expected.DesiredHeldMask));
    RequireInt32(detail, "effective_held_mask", (byte)expected.EffectiveHeldMask);
    RequireStringArray(
        detail,
        "effective_held",
        G1HeldInputScheduleContract.HeldNames(expected.EffectiveHeldMask));
    RequireInt32(
        detail,
        "desired_pressed_mask",
        (byte)(expected.DesiredHeldMask & ~previous.DesiredHeldMask));
    RequireInt32(
        detail,
        "desired_released_mask",
        (byte)(previous.DesiredHeldMask & ~expected.DesiredHeldMask));
    RequireInt32(
        detail,
        "effective_pressed_mask",
        (byte)(expected.EffectiveHeldMask & ~previous.EffectiveHeldMask));
    RequireInt32(
        detail,
        "effective_released_mask",
        (byte)(previous.EffectiveHeldMask & ~expected.EffectiveHeldMask));
    RequireFloatVector(
        detail,
        "desired_raw_controller_target_xyz",
        expected.Forward,
        expected.Strafe,
        expected.RawYaw);
    RequireFiniteExactSingle(detail, "previous_raw_yaw_target", previous.RawYaw);
    var velocity = detail.GetProperty("effective_controller_vector_xyz");
    if (velocity.ValueKind != JsonValueKind.Array || velocity.GetArrayLength() != 3 ||
        !SameFloatBits(velocity[0].GetSingle(), expected.Forward) ||
        !SameFloatBits(velocity[1].GetSingle(), expected.Strafe) ||
        !float.IsFinite(velocity[2].GetSingle()) ||
        Math.Abs(velocity[2].GetSingle()) > G1HeldInputScheduleContract.ExpectedYawSpeed ||
        expected.KickEdge && expected.YawPreempted &&
        !SameFloatBits(velocity[2].GetSingle(), 0f))
    {
        throw new InvalidDataException(
            "G1 held-input effective controller vector differed from its contract");
    }
    RequireFiniteNumber(detail, "yaw_ramp");
    RequireFiniteNumber(detail, "yaw_sign");
    var yawUpdatePhase = RequireNonemptyString(detail, "yaw_update_phase");
    if (yawUpdatePhase is not (
            "fixed_boundary_kick_preemption_reset" or
            "next_rendered_late_update"))
    {
        throw new InvalidDataException("unknown G1 yaw update phase");
    }
    RequireString(
        detail,
        "keyboard_yaw_ramp_provenance",
        G1HeldInputScheduleContract.KeyboardYawRampProvenance);
    RequireTrue(detail, "velocity_property_write_returned");
    RequireTrue(detail, "velocity_readback_exact");
    RequireNullableInt32(detail, "held_condition_ordinal", expected.HeldConditionOrdinal);
    RequireNullableInt32(detail, "kick_probe_ordinal", expected.KickProbe?.Ordinal);
    RequireBool(detail, "kick_edge", expected.KickEdge);
    RequireBool(detail, "yaw_preempted", expected.YawPreempted);
    RequireBool(detail, "translation_released", expected.TranslationReleased);
    ValidateG1LocalLifecycle(detail.GetProperty("local_lifecycle"));
}

static void ValidateG1TranslationRelease(JsonElement value, bool[] seen)
{
    var detail = value.GetProperty("detail");
    var probe = RequireG1Probe(detail, G1KickProbeKind.TranslationHeld);
    if (seen[probe.Ordinal])
        throw new InvalidDataException("duplicate G1 translation release event");
    seen[probe.Ordinal] = true;
    if (probe.TranslationReleaseTick is not int releaseTick)
        throw new InvalidDataException("G1 translation probe had no release tick");
    RequireInt32(value, "schedule_tick", releaseTick);
    RequireInt32(detail, "release_tick", releaseTick);
    RequireInt32(
        detail,
        "release_fixed_substep",
        releaseTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    RequireInt32(
        detail,
        "fixed_substeps_since_edge",
        (releaseTick - probe.EdgeTick) *
        G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    RequireInt32(detail, "desired_held_mask", 0);
    RequireInt32(detail, "effective_held_mask", 0);
    RequireFloatVector(detail, "effective_controller_vector_xyz", 0f, 0f, 0f);
    if (RequireInt64Value(detail, "release_qpc_ticks") <= 0 ||
        RequireInt64Value(detail, "qpc_frequency_hz") <= 0)
    {
        throw new InvalidDataException("invalid G1 translation release QPC anchor");
    }
    RequireTrue(detail, "velocity_property_write_returned");
    RequireTrue(detail, "velocity_readback_exact");
    ValidateG1TranslationSettleDiagnostic(
        detail.GetProperty("transition_settled_diagnostic"),
        G1HeldInputScheduleContract.TranslationOutgoingLocomotion(
            probe.DesiredHeldMask));
    RequireFalse(detail, "retry_scheduled");
    RequireFalse(detail, "second_kick_edge_scheduled");
}

static void ValidateG1YawPreemption(JsonElement value, bool[] seen)
{
    var detail = value.GetProperty("detail");
    var probe = RequireG1Probe(detail, G1KickProbeKind.YawPreempted);
    if (seen[probe.Ordinal])
        throw new InvalidDataException("duplicate G1 yaw preemption event");
    seen[probe.Ordinal] = true;
    RequireInt32(value, "schedule_tick", probe.EdgeTick);
    RequireInt32(detail, "desired_held_mask", (byte)probe.DesiredHeldMask);
    RequireInt32(detail, "effective_held_mask", 0);
    RequireInt32(detail, "raw_yaw_before_edge", probe.RawYawBeforeEdge);
    RequireInt32(detail, "raw_yaw_at_edge", 0);
    RequireFiniteExactSingle(detail, "effective_yaw_at_edge", 0f);
    RequireTrue(detail, "yaw_neutralized_before_execute_move");
    RequireTrue(detail, "velocity_property_write_returned");
    RequireTrue(detail, "velocity_readback_exact");
    RequireInt32(detail, "preemption_persists_until_tick", probe.StopTick - 1);
}

static void ValidateG1KickRequestLifecycle(
    JsonElement value,
    bool[] executeReturnSeen,
    bool[] terminalSeen)
{
    var detail = value.GetProperty("detail");
    var probe = RequireG1Probe(detail, null);
    var stage = RequireNonemptyString(detail, "lifecycle_stage");
    if (stage == "execute_move_returned")
    {
        if (executeReturnSeen[probe.Ordinal])
            throw new InvalidDataException("duplicate G1 ExecuteMove return event");
        executeReturnSeen[probe.Ordinal] = true;
        _ = RequireBooleanValue(detail, "execute_move_returned");
        _ = RequireBooleanValue(detail, "pending_move_before_call");
        _ = RequireBooleanValue(detail, "pending_move_after_call");
        _ = RequireInt32Value(detail, "pending_move_index_before_call");
        _ = RequireInt32Value(detail, "pending_move_index_after_call");
        var classification = RequireNonemptyString(detail, "local_classification");
        if (classification is not (
                "conflicting_pending_move_after_call" or
                "accepted_locally_and_armed" or
                "accepted_locally_without_pending_observation" or
                "returned_false_but_armed" or
                "rejected_locally"))
        {
            throw new InvalidDataException("unknown G1 local kick classification");
        }
        RequireFalse(detail, "retry_scheduled");
        RequireFalse(detail, "queue_owned_by_schedule");
        RequireInt32(detail, "attempt_count", 1);
        ValidateG1LocalLifecycle(detail.GetProperty("local_lifecycle_before_call"));
        ValidateG1LocalLifecycle(detail.GetProperty("local_lifecycle_after_call"));
        return;
    }
    if (stage == "send_move_invoked")
    {
        RequireTrue(detail, "pending_move");
        RequireInt32(detail, "pending_move_index", probe.MoveIndex);
        var sendPrefixFixedSubstep = RequireInt32Value(
            detail,
            "send_prefix_fixed_substep");
        var sendPrefixScheduleTick = RequireInt32Value(
            detail,
            "send_prefix_schedule_tick");
        G1HeldEventReconciler.ValidateSendAnchor(
            probe,
            sendPrefixFixedSubstep,
            sendPrefixScheduleTick);
        if (RequireInt32Value(detail, "send_prefix_unity_frame") < 0 ||
            RequireInt64Value(detail, "send_prefix_qpc_ticks") <= 0 ||
            RequireInt64Value(detail, "qpc_frequency_hz") <= 0 ||
            RequireInt32Value(detail, "late_update_opportunities") < 1)
        {
            throw new InvalidDataException("invalid G1 SendMoveEvent prefix anchor");
        }
        RequireFiniteNumber(detail, "send_prefix_unity_fixed_time");
        RequireFalse(detail, "retry_scheduled");
        RequireFalse(detail, "queue_owned_by_schedule");
        RequireInt32(detail, "attempt_count", 1);
        return;
    }
    if (stage == "pending_awaiting_render_dispatch_opportunity")
    {
        RequireTrue(detail, "pending_move");
        RequireInt32(detail, "pending_move_index", probe.MoveIndex);
        RequireFalse(detail, "cancelled_at_fixed_boundary");
        RequireFalse(detail, "retry_scheduled");
        RequireFalse(detail, "queue_owned_by_schedule");
        return;
    }
    var terminalStages = new HashSet<string>(StringComparer.Ordinal)
    {
        "rejected_locally_no_retry",
        "accepted_return_without_pending_or_send_observation",
        "client_request_method_returned",
    };
    if (terminalStages.Contains(stage))
    {
        if (terminalSeen[probe.Ordinal])
            throw new InvalidDataException("duplicate G1 terminal kick lifecycle event");
        terminalSeen[probe.Ordinal] = true;
        RequireFalse(detail, "retry_scheduled");
        RequireFalse(detail, "queue_owned_by_schedule");
        RequireFalse(detail, "pending_cancelled_by_schedule");
        RequireInt32(detail, "attempt_count", 1);
        return;
    }
    if (stage is "execute_move_threw" or "send_move_threw")
    {
        RequireFalse(detail, "retry_scheduled");
        RequireFalse(detail, "queue_owned_by_schedule");
        return;
    }
    throw new InvalidDataException($"unknown G1 kick lifecycle stage {stage}");
}

static void ValidateG1KickLocalStateTransition(JsonElement value)
{
    var detail = value.GetProperty("detail");
    var probe = RequireG1Probe(detail, null);
    RequireInt32(detail, "edge_tick", probe.EdgeTick);
    RequireInt32(
        detail,
        "edge_fixed_substep",
        probe.EdgeTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    if (RequireInt32Value(detail, "fixed_substeps_since_edge") < 0)
        throw new InvalidDataException("negative G1 lifecycle transition offset");
    RequireNullOrNonnegativeInt32(detail, "fixed_substeps_since_send");
    if (detail.GetProperty("previous").ValueKind is not (JsonValueKind.Null or JsonValueKind.Object))
        throw new InvalidDataException("invalid prior G1 lifecycle state");
    if (detail.GetProperty("previous").ValueKind == JsonValueKind.Object)
        ValidateG1LocalLifecycle(detail.GetProperty("previous"));
    ValidateG1LocalLifecycle(detail.GetProperty("current"));
    RequireTrue(detail, "local_visual_only_diagnostic");
    RequireString(detail, "server_acceptance", "unknown");
}

static void ValidateG1KickFixedObservation(JsonElement value, int[] counts)
{
    var detail = value.GetProperty("detail");
    var probe = RequireG1Probe(detail, null);
    var expectedSubstep =
        probe.EdgeTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick +
        counts[probe.Ordinal];
    RequireInt32(value, "client_fixed_substep", expectedSubstep);
    RequireInt32(
        detail,
        "edge_fixed_substep",
        probe.EdgeTick * G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    RequireInt32(detail, "fixed_substeps_since_edge", counts[probe.Ordinal]);
    RequireNullOrNonnegativeInt32(detail, "fixed_substeps_since_send");
    _ = RequireBooleanValue(detail, "send_anchor_observed");
    if (probe.Kind == G1KickProbeKind.TranslationHeld)
    {
        if (probe.TranslationReleaseTick is not int releaseTick)
            throw new InvalidDataException("translation probe lacked release tick");
        RequireInt32(detail, "translation_release_tick", releaseTick);
        var releaseSubstep = releaseTick *
            G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
        var released = expectedSubstep > releaseSubstep;
        RequireBool(detail, "translation_released", released);
        if (released)
        {
            RequireInt32(
                detail,
                "fixed_substeps_since_translation_release",
                expectedSubstep - releaseSubstep);
            ValidateG1TranslationSettleDiagnostic(
                detail.GetProperty("transition_settled_diagnostic"),
                G1HeldInputScheduleContract.TranslationOutgoingLocomotion(
                    probe.DesiredHeldMask));
        }
        else
        {
            RequireNull(detail, "fixed_substeps_since_translation_release");
            RequireNull(detail, "transition_settled_diagnostic");
        }
    }
    else
    {
        RequireNull(detail, "translation_release_tick");
        RequireNull(detail, "translation_released");
        RequireNull(detail, "fixed_substeps_since_translation_release");
        RequireNull(detail, "transition_settled_diagnostic");
    }
    _ = RequireBooleanValue(detail, "pending_move");
    _ = RequireInt32Value(detail, "pending_move_index");
    ValidateG1LocalLifecycle(detail.GetProperty("local_visual_only_diagnostic"));
    ValidateG1RecorderCorrelation(
        detail.GetProperty("recorder_correlation"),
        RequireInt32Value(value, "schedule_tick"),
        expectedSubstep);
    RequireString(detail, "server_acceptance", "unknown");
    RequireFalse(detail, "authoritative_execution_observed");
    counts[probe.Ordinal]++;
    var expectedCount =
        G1HeldInputScheduleContract.KickObservationTicks *
        G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
    if (counts[probe.Ordinal] > expectedCount)
        throw new InvalidDataException("extra G1 500 Hz kick observation");
}

static void ValidateG1YawRampUpdate(JsonElement value)
{
    var detail = value.GetProperty("detail");
    RequireString(detail, "phase", "RobotInputController.LateUpdate.prefix");
    var scheduleTick = RequireInt32Value(value, "schedule_tick");
    var frame = G1HeldInputScheduleContract.FrameAtTick(scheduleTick);
    RequireInt32(detail, "desired_key_mask", (byte)frame.DesiredHeldMask);
    RequireInt32(detail, "effective_held_mask", (byte)frame.EffectiveHeldMask);
    RequireFiniteExactSingle(detail, "raw_yaw_target", frame.RawYaw);
    RequireFiniteExactSingle(
        detail,
        "keyboard_yaw_ramp_time",
        G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds);
    RequireFiniteExactSingle(
        detail,
        "yaw_speed",
        G1HeldInputScheduleContract.ExpectedYawSpeed);
    var delta = detail.GetProperty("rendered_delta_time").GetSingle();
    var before = new G1KeyboardYawState(
        detail.GetProperty("ramp_before").GetSingle(),
        detail.GetProperty("sign_before").GetSingle());
    var expected = G1HeldInputScheduleContract.AdvanceKeyboardYaw(
        before,
        frame.RawYaw,
        delta,
        G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds,
        G1HeldInputScheduleContract.ExpectedYawSpeed);
    if (!SameFloatBits(detail.GetProperty("ramp_after").GetSingle(), expected.State.Ramp) ||
        !SameFloatBits(detail.GetProperty("sign_after").GetSingle(), expected.State.Sign) ||
        !SameFloatBits(detail.GetProperty("effective_yaw").GetSingle(), expected.EffectiveYaw))
    {
        throw new InvalidDataException("G1 rendered yaw ramp step differed from native semantics");
    }
    RequireFloatVector(
        detail,
        "effective_controller_vector_xyz",
        frame.Forward,
        frame.Strafe,
        expected.EffectiveYaw);
    if (RequireInt64Value(detail, "qpc_ticks") <= 0 ||
        RequireInt64Value(detail, "qpc_frequency_hz") <= 0)
    {
        throw new InvalidDataException("invalid G1 yaw ramp QPC anchor");
    }
    RequireString(
        detail,
        "provenance",
        G1HeldInputScheduleContract.KeyboardYawRampProvenance);
}

static void ValidateG1KickDispatchOpportunity(JsonElement value)
{
    var detail = value.GetProperty("detail");
    var ordinal = RequireInt32Value(detail, "probe_ordinal");
    if (ordinal is < 0 || ordinal >= G1HeldInputScheduleContract.KickProbes.Length)
        throw new InvalidDataException("invalid G1 dispatch probe ordinal");
    var probe = G1HeldInputScheduleContract.KickProbes[ordinal];
    RequireInt32(detail, "move_index", probe.MoveIndex);
    var stage = RequireNonemptyString(detail, "stage");
    RequireFalse(detail, "cancelled_by_schedule");
    if (stage == "matching_late_update_prefix_entered")
    {
        RequireTrue(detail, "pending_move");
        RequireInt32(detail, "pending_move_index", probe.MoveIndex);
        if (RequireInt32Value(detail, "late_update_opportunities") < 1 ||
            RequireInt32Value(detail, "fixed_updates_since_arm") < 0)
        {
            throw new InvalidDataException("invalid G1 matching dispatch opportunity");
        }
    }
    else if (stage == "matching_late_update_postfix_completed")
    {
        if (RequireInt32Value(detail, "late_update_opportunities") < 1)
            throw new InvalidDataException("invalid G1 completed dispatch opportunity");
        _ = RequireBooleanValue(detail, "send_prefix_seen");
        _ = RequireBooleanValue(detail, "send_postfix_seen");
        _ = RequireBooleanValue(detail, "pending_move_after_late_update");
        _ = RequireInt32Value(detail, "pending_move_index_after_late_update");
        _ = RequireBooleanValue(
            detail,
            "missing_dispatch_after_completed_opportunity");
    }
    else if (stage == "pending_missing_before_first_matching_late_update")
    {
        if (RequireInt32Value(detail, "late_update_opportunities") != 0)
            throw new InvalidDataException("invalid G1 missing-pending opportunity count");
    }
    else
    {
        throw new InvalidDataException($"unknown G1 dispatch opportunity stage {stage}");
    }
}

static void ValidateG1KickMeasurementSummary(
    JsonElement value,
    bool[] seen,
    int[] fixedObservationCounts)
{
    var detail = value.GetProperty("detail");
    var probe = RequireG1Probe(detail, null);
    if (seen[probe.Ordinal])
        throw new InvalidDataException("duplicate G1 kick measurement summary");
    seen[probe.Ordinal] = true;
    RequireInt32(detail, "edge_tick", probe.EdgeTick);
    RequireInt32(detail, "observation_stop_tick", probe.StopTick);
    RequireInt32(
        detail,
        "observation_ticks_scheduled",
        probe.StopTick - probe.EdgeTick);
    var observationComplete = RequireBooleanValue(detail, "observation_window_complete");
    var expectedFixedObservations =
        G1HeldInputScheduleContract.KickObservationTicks *
        G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
    RequireInt32(detail, "fixed_observations_expected", expectedFixedObservations);
    RequireInt32(
        detail,
        "fixed_observations_observed",
        fixedObservationCounts[probe.Ordinal]);
    RequireInt32(
        detail,
        "lifecycle_observation_rate_hz",
        G1HeldInputScheduleContract.UnityFixedRateHz);
    if (observationComplete && fixedObservationCounts[probe.Ordinal] != expectedFixedObservations)
        throw new InvalidDataException("complete G1 kick window lacked exact 500 Hz observations");
    _ = RequireBooleanValue(detail, "baseline_neutral_settled");
    int? firstTransitionSettledSubstep = null;
    if (probe.Kind == G1KickProbeKind.TranslationHeld)
    {
        if (probe.TranslationReleaseTick is not int releaseTick)
            throw new InvalidDataException("translation probe lacked release tick");
        RequireInt32(detail, "translation_release_tick", releaseTick);
        var releaseObserved = RequireBooleanValue(
            detail,
            "translation_release_observed");
        RequireFalse(detail, "translation_post_release_settled_kick_control_included");
        RequireString(
            detail,
            "translation_post_release_settled_kick_remaining_unknown",
            "single_edge_no_retry_schedule_observes_the_original_request_only");
        if (releaseObserved)
        {
            var releaseSubstep = releaseTick *
                G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick;
            RequireInt32(detail, "translation_release_fixed_substep", releaseSubstep);
            if (RequireInt32Value(detail, "translation_release_event_sequence") <= 0 ||
                RequireInt64Value(detail, "translation_release_qpc_ticks") <= 0 ||
                RequireInt64Value(detail, "translation_release_qpc_frequency_hz") <= 0)
            {
                throw new InvalidDataException("invalid G1 translation release anchor");
            }
            ValidateG1TranslationSettleDiagnostic(
                detail.GetProperty("translation_release_settle_diagnostic"),
                G1HeldInputScheduleContract.TranslationOutgoingLocomotion(
                    probe.DesiredHeldMask));
            var settledElement = detail.GetProperty(
                "translation_first_transition_settled_fixed_substep");
            if (settledElement.ValueKind != JsonValueKind.Null)
            {
                firstTransitionSettledSubstep = settledElement.GetInt32();
                if (firstTransitionSettledSubstep < releaseSubstep)
                {
                    throw new InvalidDataException(
                        "G1 transition settled before translation release");
                }
                RequireInt32(
                    detail,
                    "translation_fixed_substeps_release_to_settled",
                    firstTransitionSettledSubstep.Value - releaseSubstep);
            }
            else
            {
                RequireNull(detail, "translation_fixed_substeps_release_to_settled");
            }
            var lastSettle = detail.GetProperty("translation_last_settle_diagnostic");
            if (lastSettle.ValueKind == JsonValueKind.Object)
                ValidateG1TranslationSettleDiagnostic(
                    lastSettle,
                    G1HeldInputScheduleContract.TranslationOutgoingLocomotion(
                        probe.DesiredHeldMask));
            else if (lastSettle.ValueKind != JsonValueKind.Null)
                throw new InvalidDataException("invalid last G1 transition-settle diagnostic");
        }
        else
        {
            RequireNull(detail, "translation_release_fixed_substep");
            RequireNull(detail, "translation_release_event_sequence");
            RequireNull(detail, "translation_release_qpc_ticks");
            RequireNull(detail, "translation_release_qpc_frequency_hz");
            RequireNull(detail, "translation_first_transition_settled_fixed_substep");
            RequireNull(detail, "translation_fixed_substeps_release_to_settled");
            RequireNull(detail, "translation_release_settle_diagnostic");
            RequireNull(detail, "translation_last_settle_diagnostic");
        }
    }
    else
    {
        RequireNull(detail, "translation_release_tick");
        RequireNull(detail, "translation_release_fixed_substep");
        RequireNull(detail, "translation_release_event_sequence");
        RequireNull(detail, "translation_release_qpc_ticks");
        RequireNull(detail, "translation_release_qpc_frequency_hz");
        RequireNull(detail, "translation_release_observed");
        RequireNull(detail, "translation_first_transition_settled_fixed_substep");
        RequireNull(detail, "translation_fixed_substeps_release_to_settled");
        RequireNull(detail, "translation_request_send_timing_classification");
        RequireNull(detail, "translation_release_settle_diagnostic");
        RequireNull(detail, "translation_last_settle_diagnostic");
        RequireNull(detail, "translation_post_release_settled_kick_control_included");
        RequireNull(detail, "translation_post_release_settled_kick_remaining_unknown");
    }
    var moveSendInvoked = RequireBooleanValue(detail, "move_send_invoked");
    var moveSendReturned = RequireBooleanValue(detail, "move_send_method_returned");
    _ = RequireBooleanValue(detail, "yaw_neutralization_preceded_move_send");
    var asset = G1HeldInputScheduleContract.AssetForMove(probe.MoveIndex);
    var requestedAsset = detail.GetProperty("requested_move_asset");
    RequireInt32(requestedAsset, "move_index", asset.MoveIndex);
    RequireString(requestedAsset, "runtime_name", asset.RuntimeName);
    RequireString(requestedAsset, "npz_sha256", asset.NpzSha256);
    RequireInt32(requestedAsset, "recovered_controller_ticks", asset.ControllerTicks);
    _ = RequireNonemptyString(requestedAsset, "mocap_clip_config_pointer");
    var outgoing = detail.GetProperty("outgoing_projection");
    RequireString(outgoing, "method", "RobotInputController.SendMoveEvent");
    RequireInt32(outgoing, "move_index", probe.MoveIndex);
    RequireBool(outgoing, "send_invoked", moveSendInvoked);
    RequireBool(outgoing, "method_returned", moveSendReturned);
    RequireTrue(outgoing, "request_only");
    RequireString(outgoing, "server_acceptance", "unknown");
    ValidateG1SendAnchor(detail.GetProperty("send_prefix_anchor"), moveSendInvoked, probe);
    ValidateG1SendAnchor(detail.GetProperty("send_postfix_anchor"), moveSendReturned, probe);
    if (probe.Kind == G1KickProbeKind.TranslationHeld)
    {
        var releaseFixedSubstep = detail.GetProperty("translation_release_fixed_substep");
        if (releaseFixedSubstep.ValueKind == JsonValueKind.Null)
        {
            RequireNull(detail, "translation_request_send_timing_classification");
        }
        else
        {
            var sendAnchor = detail.GetProperty("send_prefix_anchor");
            var sendSubstep = sendAnchor.ValueKind == JsonValueKind.Object
                ? RequireInt32Value(sendAnchor, "client_fixed_substep")
                : (int?)null;
            RequireString(
                detail,
                "translation_request_send_timing_classification",
                G1HeldInputScheduleContract.ClassifyTranslationSendTiming(
                    sendSubstep,
                    releaseFixedSubstep.GetInt32(),
                    firstTransitionSettledSubstep));
        }
    }
    RequireString(
        detail,
        "motion_identity_status",
        G1HeldInputScheduleContract.MotionIdentityStatus);
    RequireFalse(detail, "requested_asset_to_runner_motion_identity_proven");
    RequireNull(detail, "certified_input_delay_ticks");
    RequireNull(detail, "certified_entry_to_completion_ticks");
    RequireFalse(detail, "certified_duration_available");
    _ = RequireNonemptyString(detail, "timing_certification_reason");
    if (probe.Kind == G1KickProbeKind.TranslationHeld)
    {
        var localGateResult = RequireNonemptyString(
            detail,
            "translation_local_gate_result");
        if (localGateResult is not ("supported" or "contradicted" or "unknown"))
            throw new InvalidDataException("invalid G1 translation local-gate result");
        RequireString(detail, "translation_behavior_result", "unknown");
        _ = RequireNonemptyString(detail, "translation_response_classification");
        RequireString(
            detail,
            "translation_result_criteria",
            G1HeldInputScheduleContract.TranslationBehaviorCriteria);
    }
    else
    {
        RequireNull(detail, "translation_local_gate_result");
        RequireNull(detail, "translation_behavior_result");
        RequireNull(detail, "translation_response_classification");
        RequireNull(detail, "translation_result_criteria");
    }
    RequireString(detail, "physical_response_result", "unknown");
    _ = RequireNonemptyString(detail, "physical_response_result_criteria");
    RequireTrue(detail, "local_diagnostics_only");
    RequireTrue(detail, "request_only");
    RequireString(detail, "server_acceptance", "unknown");
    ValidateG1RecorderCorrelation(
        detail.GetProperty("recorder_correlation"),
        RequireInt32Value(value, "schedule_tick"),
        RequireInt32Value(value, "client_fixed_substep"));
}

static void ValidateG1SendAnchor(
    JsonElement value,
    bool expectedPresent,
    G1KickProbe probe)
{
    if (!expectedPresent)
    {
        if (value.ValueKind != JsonValueKind.Null)
            throw new InvalidDataException("unexpected G1 SendMoveEvent timing anchor");
        return;
    }
    if (value.ValueKind != JsonValueKind.Object)
        throw new InvalidDataException("invalid G1 SendMoveEvent timing anchor");
    var fixedSubstep = RequireInt32Value(value, "client_fixed_substep");
    var scheduleTick = RequireInt32Value(value, "schedule_tick");
    G1HeldEventReconciler.ValidateSendAnchor(probe, fixedSubstep, scheduleTick);
    if (RequireInt32Value(value, "unity_frame") < 0 ||
        RequireInt64Value(value, "qpc_ticks") <= 0 ||
        RequireInt64Value(value, "qpc_frequency_hz") <= 0)
    {
        throw new InvalidDataException("invalid G1 SendMoveEvent timing anchor");
    }
    RequireFiniteNumber(value, "unity_fixed_time");
}

static void ValidateG1VelocityRequestLifecycle(JsonElement value)
{
    var detail = value.GetProperty("detail");
    RequireString(detail, "method", "RobotInputController.SendVelocityCommand");
    var stage = RequireNonemptyString(detail, "lifecycle_stage");
    if (stage == "send_velocity_invoked")
        RequireFalse(detail, "method_returned");
    else if (stage == "client_request_method_returned")
    {
        RequireTrue(detail, "method_returned");
        RequireString(detail, "local_return_value", "void_returned_normally");
    }
    else
        throw new InvalidDataException($"unknown G1 velocity lifecycle stage {stage}");
}

static void ValidateG1UnexpectedRequestBlocked(JsonElement value)
{
    var detail = value.GetProperty("detail");
    var requestKind = RequireNonemptyString(detail, "request_kind");
    if (requestKind is not ("special" or "estop"))
        throw new InvalidDataException("unknown blocked G1 request kind");
    RequireTrue(detail, "original_method_skipped");
}

static void ValidateG1LocalLifecycle(JsonElement value)
{
    _ = RequireBooleanValue(value, "input_is_punching");
    _ = RequireBooleanValue(value, "input_is_recovering");
    _ = RequireBooleanValue(value, "action_busy");
    _ = RequireBooleanValue(value, "sonic_policy_runner_available");
    RequireNullOrBoolean(value, "sonic_policy_runner_is_done");
    RequireNullOrBoolean(value, "sonic_policy_runner_is_recovering");
    RequireNullOrString(value, "sonic_current_motion_pointer");
    if (RequireInt32Value(value, "sonic_motion_frame_index") < -1)
        throw new InvalidDataException("invalid G1 motion frame index");
    RequireNullOrString(value, "probe_reason");
    RequireFalse(value, "sonic_action_composer_used");
}

static void ValidateG1TranslationSettleDiagnostic(
    JsonElement value,
    string expectedOutgoingLocomotion)
{
    var methodInvoked = RequireBooleanValue(
        value,
        "transition_settled_method_invoked");
    var methodReturned = RequireBooleanValue(
        value,
        "transition_settled_method_returned");
    if (methodReturned && !methodInvoked)
        throw new InvalidDataException("G1 TransitionSettled returned without invocation");
    RequireString(value, "outgoing_locomotion", expectedOutgoingLocomotion);
    RequireFalse(value, "predicate_mirrored_from_pinned_native_code");
    RequireString(
        value,
        "provenance",
        G1HeldInputScheduleContract.TransitionSettledProvenance);
    RequireString(
        value,
        "base_velocity_provenance",
        G1HeldInputScheduleContract.TransitionSettleBaseVelocityProvenance);
    if (methodReturned)
        _ = RequireBooleanValue(value, "transition_settled");
    else
        RequireNull(value, "transition_settled");

    var baseVelocityAvailable = RequireBooleanValue(
        value,
        "base_velocity_available");
    if (!baseVelocityAvailable)
    {
        RequireNull(value, "base_linear_velocity_local_m_s");
        RequireNull(value, "base_angular_velocity_local_rad_s");
        _ = RequireNonemptyString(value, "probe_reason");
    }
    else
    {
        ValidateFiniteVector3(value.GetProperty("base_linear_velocity_local_m_s"));
        ValidateFiniteVector3(value.GetProperty("base_angular_velocity_local_rad_s"));
        RequireNull(value, "probe_reason");
    }
    RequireFiniteExactSingle(
        value,
        "transition_settle_planar_speed_m_s",
        G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed);
    RequireFiniteExactSingle(
        value,
        "transition_settle_yaw_rate_rad_s",
        G1HeldInputScheduleContract.ExpectedTransitionSettleYawRate);
}

static void ValidateFiniteVector3(JsonElement value)
{
    if (value.ValueKind != JsonValueKind.Array || value.GetArrayLength() != 3 ||
        value.EnumerateArray().Any(component =>
            component.ValueKind != JsonValueKind.Number ||
            !float.IsFinite(component.GetSingle())))
    {
        throw new InvalidDataException("invalid finite 3-vector");
    }
}

static G1KickProbe RequireG1Probe(JsonElement detail, G1KickProbeKind? expectedKind)
{
    var ordinal = RequireInt32Value(detail, "probe_ordinal");
    if (ordinal is < 0 || ordinal >= G1HeldInputScheduleContract.KickProbes.Length)
        throw new InvalidDataException("G1 kick probe ordinal was invalid");
    var probe = G1HeldInputScheduleContract.KickProbes[ordinal];
    RequireString(detail, "probe_label", probe.Label);
    RequireString(
        detail,
        "probe_kind",
        probe.Kind == G1KickProbeKind.TranslationHeld
            ? "translation_held"
            : "yaw_preempted");
    RequireInt32(detail, "move_index", probe.MoveIndex);
    if (expectedKind is not null && probe.Kind != expectedKind.Value)
        throw new InvalidDataException("G1 kick probe kind was unexpected");
    return probe;
}

static bool ValidateG1HeldEnd(
    JsonElement value,
    string runId,
    string freshRoundRequestId,
    string roundIdentitySha256,
    int observedScheduleTicks,
    bool[] executeReturnSeen,
    bool[] terminalSeen,
    bool[] summarySeen,
    bool[] yawPreemptionSeen,
    bool[] translationReleaseSeen,
    int[] fixedObservationCounts)
{
    RequireString(value, "protocol", Protocol);
    RequireString(value, "g1_held_schedule_schema", G1HeldInputScheduleContract.Schema);
    RequireString(value, "g1_held_schedule_id", G1HeldInputScheduleContract.ScheduleId);
    RequireString(
        value,
        "g1_held_schedule_sha256",
        G1HeldInputScheduleContract.ExpectedSha256);
    RequireString(value, "g1_held_schedule_run_id", runId);
    RequireString(value, "fresh_round_request_id", freshRoundRequestId);
    RequireString(value, "round_identity_sha256", roundIdentitySha256);
    var scheduleTick = RequireInt32Value(value, "schedule_tick");
    var fixedSubstep = RequireInt32Value(value, "client_fixed_substep");
    if (scheduleTick is < 0 or > G1HeldInputScheduleContract.FinalScheduleTick || fixedSubstep < 0)
        throw new InvalidDataException("G1 held-input end clock was invalid");
    var complete = RequireBooleanValue(value, "complete");
    RequireBool(value, "experiment_coverage_complete", complete);
    RequireBool(value, "partial_coverage", !complete);
    RequireTrue(value, "authorized_while_background");
    _ = RequireBooleanValue(value, "final_neutral_send_method_returned");
    _ = RequireBooleanValue(value, "owned_pending_cleared_only_during_stop");
    var capacity = value.GetProperty("round_capacity_preflight");
    RequireFiniteExactNumber(
        capacity,
        "required_run_seconds",
        G1HeldInputScheduleContract.RequiredRunSeconds);
    RequireFiniteExactSingle(
        capacity,
        "safety_seconds",
        G1HeldInputScheduleContract.RoundCapacitySafetySeconds);
    RequireFiniteExactNumber(
        capacity,
        "required_capacity_seconds",
        G1HeldInputScheduleContract.RequiredRoundCapacitySeconds);
    RequireTrue(capacity, "capacity_proven");
    RequireString(
        value,
        "pose_response_source",
        G1HeldInputScheduleContract.PoseResponseSource);
    RequireFalse(value, "pose_response_in_pipe_transcript");
    RequireFalse(value, "sonic_action_composer_lifecycle_used");
    RequireFalse(value, "global_input_emitted");
    RequireTrue(value, "request_only");
    RequireString(value, "server_acceptance", "unknown");
    RequireFalse(value, "server_acceptance_observed");
    RequireFalse(value, "authoritative_execution_observed");

    var coverage = value.GetProperty("coverage");
    RequireBool(coverage, "Complete", complete);
    RequireInt32(
        coverage,
        "HeldConditionsExpected",
        G1HeldInputScheduleContract.HeldConditions.Length);
    var heldObserved = RequireInt32Value(
        coverage,
        "HeldConditionsObservedForExactDuration");
    if (heldObserved < 0 ||
        heldObserved > G1HeldInputScheduleContract.HeldConditions.Length)
        throw new InvalidDataException("G1 held-condition coverage count was invalid");
    var heldTicks = coverage.GetProperty("HeldConditionObservedTicks");
    if (heldTicks.ValueKind != JsonValueKind.Array ||
        heldTicks.GetArrayLength() != G1HeldInputScheduleContract.HeldConditions.Length ||
        heldTicks.EnumerateArray().Any(item =>
            item.GetInt32() is < 0 or > G1HeldInputScheduleContract.HeldDurationTicks))
    {
        throw new InvalidDataException("G1 held-condition coverage vector was invalid");
    }
    RequireInt32(coverage, "TranslationKickProbesExpected", 4);
    RequireInt32(coverage, "YawKickProbesExpected", 4);
    var translationEdges = RequireInt32Value(coverage, "TranslationKickEdgesObserved");
    var translationTerminal = RequireInt32Value(
        coverage,
        "TranslationKickTerminalOutcomesObserved");
    var translationReleases = RequireInt32Value(
        coverage,
        "TranslationReleasesObserved");
    var yawEdges = RequireInt32Value(coverage, "YawKickEdgesObserved");
    var yawTerminal = RequireInt32Value(coverage, "YawKickTerminalOutcomesObserved");
    var yawPreemptions = RequireInt32Value(coverage, "YawPreemptionsObserved");
    var summaries = RequireInt32Value(coverage, "KickMeasurementSummariesObserved");
    var windows = RequireInt32Value(coverage, "KickObservationWindowsComplete");
    var fixedWindows = RequireInt32Value(
        coverage,
        "KickFixedObservationWindowsComplete");
    if (translationEdges is < 0 or > 4 || translationTerminal is < 0 or > 4 ||
        translationReleases is < 0 or > 4 ||
        yawEdges is < 0 or > 4 || yawTerminal is < 0 or > 4 ||
        yawPreemptions is < 0 or > 4 || summaries is < 0 or > 8 ||
        windows is < 0 or > 8 || fixedWindows is < 0 or > 8)
    {
        throw new InvalidDataException("G1 kick coverage count was invalid");
    }
    if (summaries != summarySeen.Count(item => item) ||
        translationTerminal + yawTerminal != terminalSeen.Count(item => item) ||
        yawPreemptions != yawPreemptionSeen.Count(item => item) ||
        translationReleases != translationReleaseSeen.Count(item => item))
    {
        throw new InvalidDataException("G1 bridge and client coverage observations differed");
    }
    RequireIntArray(
        coverage,
        "KickMoveIndices",
        G1HeldInputScheduleContract.KickMoveIndices);
    RequireFalse(coverage, "FBindingIncluded");
    RequireFalse(coverage, "QueueOrRetryUsed");
    RequireFalse(coverage, "SonicActionComposerLifecycleUsed");
    RequireFalse(coverage, "PhysicalBehaviorCertified");
    RequireString(
        coverage,
        "PoseResponseSource",
        G1HeldInputScheduleContract.PoseResponseSource);
    var measurements = coverage.GetProperty("KickMeasurements");
    if (measurements.ValueKind != JsonValueKind.Array || measurements.GetArrayLength() != 8)
        throw new InvalidDataException("G1 kick measurement coverage vector was invalid");
    for (var ordinal = 0; ordinal < measurements.GetArrayLength(); ordinal++)
    {
        var measurement = measurements[ordinal];
        var probe = G1HeldInputScheduleContract.KickProbes[ordinal];
        RequireInt32(measurement, "probe_ordinal", ordinal);
        RequireString(measurement, "probe_label", probe.Label);
        RequireInt32(measurement, "move_index", probe.MoveIndex);
        RequireInt32(measurement, "edge_tick", probe.EdgeTick);
        RequireNullableInt32(
            measurement,
            "translation_release_tick",
            probe.TranslationReleaseTick);
        RequireInt32(measurement, "observation_stop_tick", probe.StopTick);
        RequireBool(measurement, "edge_observed", executeReturnSeen[ordinal]);
        RequireBool(measurement, "terminal_outcome_observed", terminalSeen[ordinal]);
        RequireBool(
            measurement,
            "translation_release_observed",
            translationReleaseSeen[ordinal]);
        RequireBool(
            measurement,
            "yaw_preemption_observed",
            yawPreemptionSeen[ordinal]);
        RequireBool(measurement, "summary_emitted", summarySeen[ordinal]);
        RequireInt32(
            measurement,
            "fixed_observations",
            fixedObservationCounts[ordinal]);
        RequireInt32(
            measurement,
            "fixed_observations_expected",
            G1HeldInputScheduleContract.KickObservationTicks *
            G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
        RequireString(
            measurement,
            "motion_identity_status",
            G1HeldInputScheduleContract.MotionIdentityStatus);
        RequireFalse(measurement, "certified_duration_available");
        RequireString(measurement, "physical_behavior_result", "unknown");
        RequireTrue(measurement, "request_only");
        RequireString(measurement, "server_acceptance", "unknown");
    }

    if (!complete)
    {
        if (RequireNonemptyString(value, "reason") == "complete")
            throw new InvalidDataException("partial G1 end used complete reason");
        if (observedScheduleTicks > G1HeldInputScheduleContract.DurationScheduleTicks)
            throw new InvalidDataException("partial G1 schedule emitted too many ticks");
        return false;
    }

    RequireString(value, "reason", "complete");
    RequireTrue(value, "final_neutral_send_method_returned");
    if (observedScheduleTicks != G1HeldInputScheduleContract.DurationScheduleTicks ||
        scheduleTick != G1HeldInputScheduleContract.FinalScheduleTick ||
        fixedSubstep !=
        G1HeldInputScheduleContract.DurationScheduleTicks *
        G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick - 1 ||
        heldObserved != G1HeldInputScheduleContract.HeldConditions.Length ||
        heldTicks.EnumerateArray().Any(item =>
            item.GetInt32() != G1HeldInputScheduleContract.HeldDurationTicks) ||
        translationEdges != 4 || translationTerminal != 4 || translationReleases != 4 ||
        yawEdges != 4 ||
        yawTerminal != 4 || yawPreemptions != 4 || summaries != 8 || windows != 8 ||
        fixedWindows != 8 || !executeReturnSeen.All(item => item) ||
        !terminalSeen.All(item => item) || !summarySeen.All(item => item) ||
        fixedObservationCounts.Any(item => item !=
            G1HeldInputScheduleContract.KickObservationTicks *
            G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick) ||
        G1HeldInputScheduleContract.KickProbes
            .Where(item => item.Kind == G1KickProbeKind.YawPreempted)
            .Any(item => !yawPreemptionSeen[item.Ordinal]) ||
        G1HeldInputScheduleContract.KickProbes
            .Where(item => item.Kind == G1KickProbeKind.TranslationHeld)
            .Any(item => !translationReleaseSeen[item.Ordinal]))
    {
        throw new InvalidDataException("complete G1 held-input end had incomplete coverage");
    }
    for (var ordinal = 0; ordinal < measurements.GetArrayLength(); ordinal++)
    {
        var measurement = measurements[ordinal];
        RequireTrue(measurement, "measurement_started");
        RequireTrue(measurement, "observation_window_complete");
    }
    return true;
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
    RequireString(
        capabilities,
        "private_ai_proof_basis",
        "build_pinned_REK_FindMatch_solo_ConnectToArena_EnterChampionship_non_koth_solo_same_runtime_session");
    RequireString(capabilities, "solo_route_required_flow", "solo");
    RequireFalse(capabilities, "solo_route_arena_identifier_recorded");
    RequireFalse(capabilities, "solo_route_connection_ticket_recorded");
    RequireFalse(capabilities, "solo_route_endpoint_recorded");
    RequireFalse(capabilities, "server_private_proven");
    RequireString(capabilities, "server_private_status", "unknown");
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
            "StartG1HeldInputSchedule",
            "StopG1HeldInputSchedule",
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
    RequireIntArray(
        capabilities,
        "continuous_controller_g1_move_indices",
        new[] { 6, 7, 8, 9 });
    RequireStringArray(
        capabilities,
        "continuous_controller_supported_runtime_models",
        new[] { "t800", "g1" });
    RequireString(
        capabilities,
        "continuous_controller_t800_recovery_mode",
        ContinuousT800RecoveryMode);
    RequireString(
        capabilities,
        "continuous_controller_g1_recovery_mode",
        ContinuousG1RecoveryMode);
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
    RequireString(
        capabilities,
        "g1_held_schedule_schema",
        G1HeldInputScheduleContract.Schema);
    RequireString(
        capabilities,
        "g1_held_schedule_id",
        G1HeldInputScheduleContract.ScheduleId);
    RequireString(
        capabilities,
        "g1_held_schedule_sha256",
        G1HeldInputScheduleContract.ExpectedSha256);
    RequireString(
        capabilities,
        "g1_held_schedule_authority_scope",
        G1HeldInputScheduleContract.AuthorityScope);
    RequireString(
        capabilities,
        "g1_held_schedule_authority_caveat",
        G1HeldInputScheduleContract.AuthorityCaveat);
    RequireString(
        capabilities,
        "g1_held_schedule_required_isolation_proof",
        G1HeldInputScheduleContract.RequiredIsolationProof);
    RequireString(
        capabilities,
        "g1_held_schedule_pose_response_source",
        G1HeldInputScheduleContract.PoseResponseSource);
    RequireInt32(
        capabilities,
        "g1_held_schedule_unity_fixed_rate_hz",
        G1HeldInputScheduleContract.UnityFixedRateHz);
    RequireInt32(
        capabilities,
        "g1_held_schedule_rate_hz",
        G1HeldInputScheduleContract.ScheduleRateHz);
    RequireInt32(
        capabilities,
        "g1_held_schedule_fixed_substeps_per_tick",
        G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    RequireInt32(
        capabilities,
        "g1_held_schedule_duration_ticks",
        G1HeldInputScheduleContract.DurationScheduleTicks);
    RequireInt32(
        capabilities,
        "g1_held_schedule_kick_observation_ticks",
        G1HeldInputScheduleContract.KickObservationTicks);
    RequireInt32(
        capabilities,
        "g1_held_schedule_translation_release_offset_ticks",
        G1HeldInputScheduleContract.TranslationReleaseOffsetTicks);
    RequireString(
        capabilities,
        "g1_held_schedule_transition_settled_provenance",
        G1HeldInputScheduleContract.TransitionSettledProvenance);
    RequireString(
        capabilities,
        "g1_held_schedule_transition_settle_base_velocity_provenance",
        G1HeldInputScheduleContract.TransitionSettleBaseVelocityProvenance);
    RequireFiniteExactSingle(
        capabilities,
        "g1_held_schedule_transition_settle_planar_speed_m_s",
        G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed);
    RequireFiniteExactSingle(
        capabilities,
        "g1_held_schedule_transition_settle_yaw_rate_rad_s",
        G1HeldInputScheduleContract.ExpectedTransitionSettleYawRate);
    RequireFalse(
        capabilities,
        "g1_held_schedule_post_release_settled_kick_control_included");
    RequireInt32(
        capabilities,
        "g1_held_schedule_lifecycle_observation_rate_hz",
        G1HeldInputScheduleContract.UnityFixedRateHz);
    RequireFiniteExactSingle(
        capabilities,
        "g1_held_schedule_keyboard_yaw_ramp_time_seconds",
        G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds);
    RequireFiniteExactSingle(
        capabilities,
        "g1_held_schedule_keyboard_yaw_speed",
        G1HeldInputScheduleContract.ExpectedYawSpeed);
    RequireFiniteExactNumber(
        capabilities,
        "g1_held_schedule_required_run_seconds",
        G1HeldInputScheduleContract.RequiredRunSeconds);
    RequireFiniteExactSingle(
        capabilities,
        "g1_held_schedule_round_capacity_safety_seconds",
        G1HeldInputScheduleContract.RoundCapacitySafetySeconds);
    RequireFiniteExactNumber(
        capabilities,
        "g1_held_schedule_required_round_capacity_seconds",
        G1HeldInputScheduleContract.RequiredRoundCapacitySeconds);
    RequireInt32(
        capabilities,
        "g1_held_schedule_held_condition_count",
        G1HeldInputScheduleContract.HeldConditions.Length);
    RequireIntArray(
        capabilities,
        "g1_held_schedule_kick_move_indices",
        G1HeldInputScheduleContract.KickMoveIndices);
    RequireFalse(capabilities, "g1_held_schedule_f_binding_included");
    RequireFalse(capabilities, "g1_held_schedule_queue_or_retry_used");
    RequireFalse(capabilities, "g1_held_schedule_sonic_action_composer_lifecycle_used");
    RequireFalse(capabilities, "g1_held_schedule_global_input_emitted");
    ValidateContinuousAttackProfiles(
        capabilities.GetProperty("continuous_controller_attack_profiles"),
        new[]
    {
        (2, "skill", "Punch Combo",
            "233f952edecb7bf8d1959c6549c0edb95e1833451fff988f57a4b14d92b14dd4",
            new[] { (0.76f, 0.1f, 0.19f, 1, 1f), (1.15f, 0.1f, 0.1f, 1, 1f), (1.81f, 0.1f, 0.1f, 1, 1f) }),
        (3, "youbiantui", "Right Kick",
            "70f36a2c7b9b53c10e47cc613d87a770eb86fb2e683ed64ee39efcccf2e75636",
            new[] { (1.11f, 0.25f, 0.3f, 4, 1f) }),
        (4, "left_light_attack", "Left Punch",
            "32081b731a59b7553d94022ebff865764b34c83dbf274aadf26540fb17daad2e",
            new[] { (0.39f, 0.12f, 0.08f, 1, 1f) }),
        (5, "right_light_attack", "Right Punch",
            "b1c1b2c000dd612e3eb4c33c5d90e03c2c9306e5cc194747c14248b0d77b7dea",
            new[] { (0.22f, 0.12f, 0.2f, 2, 1f) }),
        (9, "right_shoryuken_lm", "Dragon Punch",
            "cc298f53d04ffd56be57ce3049559d3d30c7724fe4d2839a66ea8f3008ca8deb",
            new[] { (0f, 0f, 0f, 2, 1f) }),
        (10, "front_kick_L", "Left Kick",
            "cd5b286f6e4f5c3003cb0f5c9de5e5690ca92ed58e5a1b789f4394e4d7911ee8",
            new[] { (1.1f, 0.2f, 0.15f, 3, 1f) }),
    });
    ValidateContinuousAttackProfiles(
        capabilities.GetProperty("continuous_controller_g1_attack_profiles"),
        new[]
    {
        (6, "left_side_kick_processed", "Left Side Kick",
            "fb5c3938396c789020003634b0dc76d2319ea3df33ec2fa2927a2e4e24a070a0",
            new[] { (1.1f, 0.3f, 0.5f, 3, 2f) }),
        (7, "left_front_kick_processed", "Left Front Kick",
            "5f61681420dbd84ce046f68770b73d2ed9e845d38cacd4e37d1f704bcb5f9241",
            new[] { (1f, 0.2f, 0.5f, 3, 2f) }),
        (8, "right_side_kick_processed", "Right Side Kick",
            "6c4a414aa3c6860f30b28d65e6ef7c18d5e6bf9331ab69215ded03cf82c560ac",
            new[] { (1.14f, 0.4f, 0.15f, 4, 2f) }),
        (9, "right_knee_processed", "Right Knee",
            "04dbcb4b28f617912c7ac5c452a23969c5bc17ac745571c35d29ebea03b2733b",
            new[] { (0.65f, 0.4f, 0.15f, 4, 3f) }),
    });
}

static void ValidateContinuousAttackProfiles(
    JsonElement attackProfiles,
    (int MoveIndex, string MoveName, string DisplayName, string AssetSha256,
        (float Impact, float Lead, float Release, int Limb, float Gain)[] Impacts)[]
        expectedProfiles)
{
    if (attackProfiles.ValueKind != JsonValueKind.Array ||
        attackProfiles.GetArrayLength() != expectedProfiles.Length)
    {
        throw new InvalidDataException("continuous attack profile count mismatch");
    }
    for (var profileIndex = 0; profileIndex < expectedProfiles.Length; profileIndex++)
    {
        var expected = expectedProfiles[profileIndex];
        var actual = attackProfiles[profileIndex];
        RequireInt32(actual, "move_index", expected.MoveIndex);
        RequireString(actual, "move_name", expected.MoveName);
        RequireString(actual, "display_name", expected.DisplayName);
        RequireString(actual, "serialized_asset_sha256", expected.AssetSha256);
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
            impacts.GetArrayLength() != expected.Impacts.Length)
        {
            throw new InvalidDataException("continuous static impact event count mismatch");
        }
        for (var impactIndex = 0; impactIndex < expected.Impacts.Length; impactIndex++)
        {
            var expectedImpact = expected.Impacts[impactIndex];
            var actualImpact = impacts[impactIndex];
            if (!SameFloatBits(actualImpact.GetProperty("impact_time_s").GetSingle(), expectedImpact.Impact) ||
                !SameFloatBits(actualImpact.GetProperty("lead_time_s").GetSingle(), expectedImpact.Lead) ||
                !SameFloatBits(actualImpact.GetProperty("release_time_s").GetSingle(), expectedImpact.Release) ||
                !SameFloatBits(actualImpact.GetProperty("gain_boost").GetSingle(), expectedImpact.Gain) ||
                actualImpact.GetProperty("limb").GetInt32() != expectedImpact.Limb)
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
    RequireString(build, "sharedassets0_sha256", ExpectedSharedAssets0Sha256);
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
    ValidateG1HeldStateIdentity(control);
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

static void ValidateG1HeldStateIdentity(JsonElement value)
{
    ValidateG1HeldIdentity(value);
    RequireString(
        value,
        "g1_held_schedule_required_isolation_proof",
        G1HeldInputScheduleContract.RequiredIsolationProof);
    RequireString(
        value,
        "g1_held_schedule_pose_response_source",
        G1HeldInputScheduleContract.PoseResponseSource);
    RequireInt32(
        value,
        "g1_held_schedule_unity_fixed_rate_hz",
        G1HeldInputScheduleContract.UnityFixedRateHz);
    RequireInt32(
        value,
        "g1_held_schedule_rate_hz",
        G1HeldInputScheduleContract.ScheduleRateHz);
    RequireInt32(
        value,
        "g1_held_schedule_fixed_substeps_per_tick",
        G1HeldInputScheduleContract.FixedSubstepsPerScheduleTick);
    RequireInt32(
        value,
        "g1_held_schedule_duration_ticks",
        G1HeldInputScheduleContract.DurationScheduleTicks);
    RequireInt32(
        value,
        "g1_held_schedule_kick_observation_ticks",
        G1HeldInputScheduleContract.KickObservationTicks);
    RequireInt32(
        value,
        "g1_held_schedule_translation_release_offset_ticks",
        G1HeldInputScheduleContract.TranslationReleaseOffsetTicks);
    RequireString(
        value,
        "g1_held_schedule_transition_settled_provenance",
        G1HeldInputScheduleContract.TransitionSettledProvenance);
    RequireString(
        value,
        "g1_held_schedule_transition_settle_base_velocity_provenance",
        G1HeldInputScheduleContract.TransitionSettleBaseVelocityProvenance);
    RequireFiniteExactSingle(
        value,
        "g1_held_schedule_transition_settle_planar_speed_m_s",
        G1HeldInputScheduleContract.ExpectedTransitionSettlePlanarSpeed);
    RequireFiniteExactSingle(
        value,
        "g1_held_schedule_transition_settle_yaw_rate_rad_s",
        G1HeldInputScheduleContract.ExpectedTransitionSettleYawRate);
    RequireFalse(
        value,
        "g1_held_schedule_post_release_settled_kick_control_included");
    RequireInt32(
        value,
        "g1_held_schedule_lifecycle_observation_rate_hz",
        G1HeldInputScheduleContract.UnityFixedRateHz);
    RequireFiniteExactSingle(
        value,
        "g1_held_schedule_keyboard_yaw_ramp_time_seconds",
        G1HeldInputScheduleContract.ExpectedKeyboardYawRampTimeSeconds);
    RequireFiniteExactSingle(
        value,
        "g1_held_schedule_keyboard_yaw_speed",
        G1HeldInputScheduleContract.ExpectedYawSpeed);
    RequireFiniteExactNumber(
        value,
        "g1_held_schedule_required_run_seconds",
        G1HeldInputScheduleContract.RequiredRunSeconds);
    RequireFiniteExactSingle(
        value,
        "g1_held_schedule_round_capacity_safety_seconds",
        G1HeldInputScheduleContract.RoundCapacitySafetySeconds);
    RequireFiniteExactNumber(
        value,
        "g1_held_schedule_required_round_capacity_seconds",
        G1HeldInputScheduleContract.RequiredRoundCapacitySeconds);
    RequireInt32(
        value,
        "g1_held_schedule_held_condition_count",
        G1HeldInputScheduleContract.HeldConditions.Length);
    RequireIntArray(
        value,
        "g1_held_schedule_kick_move_indices",
        G1HeldInputScheduleContract.KickMoveIndices);
    RequireFalse(value, "g1_held_schedule_f_binding_included");
    RequireFalse(value, "g1_held_schedule_queue_or_retry_used");
    RequireFalse(value, "g1_held_schedule_sonic_action_composer_lifecycle_used");
    _ = RequireBooleanValue(value, "g1_held_schedule_running");
    _ = RequireBooleanValue(value, "g1_held_schedule_authorized_while_background");
    ValidateOptionalHex(value, "g1_held_schedule_run_id", 32);
    ValidateOptionalString(value, "g1_held_schedule_fresh_round_request_id");
    ValidateOptionalHex(value, "g1_held_schedule_round_identity_sha256", 64);
    var tick = RequireInt32Value(value, "g1_held_schedule_tick");
    var substep = RequireInt32Value(value, "g1_held_schedule_client_fixed_substep");
    if (tick is < 0 or > G1HeldInputScheduleContract.FinalScheduleTick || substep < 0)
        throw new InvalidDataException("G1 held-input state counters were invalid");
    RequireFiniteNumber(value, "g1_held_schedule_round_duration_seconds");
    RequireFiniteNumber(value, "g1_held_schedule_initial_time_remaining_seconds");
    _ = RequireBooleanValue(value, "g1_held_schedule_round_capacity_proven");
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
    RequireTrue(privateAi, "solo_route_hooks_verified");
    RequireTrue(privateAi, "solo_route_proven");
    RequireString(privateAi, "solo_route_flow", "solo");
    RequireTrue(privateAi, "solo_route_connect_to_arena_observed");
    RequireTrue(privateAi, "solo_route_enter_championship_observed");
    RequireFalse(privateAi, "solo_route_enter_championship_koth");
    RequireTrue(privateAi, "solo_route_enter_championship_solo");
    RequireTrue(privateAi, "solo_route_arena_identity_consistent");
    RequireTrue(privateAi, "solo_route_runtime_session_identity_consistent");
    RequireString(privateAi, "solo_route_reason", "solo_route_proven");
    RequireFalse(privateAi, "server_private_proven");
    RequireString(privateAi, "server_private_status", "unknown");
    RequireString(privateAi, "reason", "solo_route_proven");
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
    if (ValidateSupportedMeasuredPairing(pairing) != "t800")
        throw new InvalidDataException("strict parity mode requires exact T800 runtime pairing");
}

static string ValidateContinuousMeasuredPairing(JsonElement pairing) =>
    ValidateSupportedMeasuredPairing(pairing);

static string ValidateSupportedMeasuredPairing(JsonElement pairing)
{
    RequireString(pairing, "required_pairing", "exact_homogeneous_supported_runtime_pair");
    RequireNull(pairing, "required_robot_id");
    RequireStringArray(pairing, "supported_runtime_models", new[] { "t800", "g1" });
    RequireFalse(pairing, "semantic_robot_id_required_for_acceptance");
    RequireInt32(pairing, "required_t800_bone_count", 26);
    RequireString(pairing, "required_t800_bone_signature_sha256", T800BoneSignatureSha256);
    RequireInt32(pairing, "required_g1_bone_count", 30);
    RequireString(pairing, "required_g1_bone_signature_sha256", G1BoneSignatureSha256);
    RequireTrue(pairing, "exact_supported_runtime_pairing");
    var runtimeModel = RequireNonemptyString(pairing, "runtime_model");
    if (runtimeModel is not ("t800" or "g1"))
        throw new InvalidDataException("unsupported continuous runtime model");
    RequireBool(pairing, "exact_t800_vs_t800", runtimeModel == "t800");
    RequireBool(pairing, "exact_g1_vs_g1", runtimeModel == "g1");
    RequireString(
        pairing,
        "reason",
        runtimeModel == "t800"
            ? "exact_t800_vs_t800_pairing_proven"
            : "exact_g1_vs_g1_runtime_pairing_proven_semantic_ids_recorded_not_trusted");
    ValidateMeasuredFighterIdentity(pairing.GetProperty("local_fighter"), runtimeModel);
    ValidateMeasuredFighterIdentity(pairing.GetProperty("opponent_fighter"), runtimeModel);
    return runtimeModel;
}

static void ValidateMeasuredFighterIdentity(JsonElement fighter, string runtimeModel)
{
    _ = RequireNonemptyString(fighter, "runtime_object_name");
    var boneCount = BoneCountForRuntimeModel(runtimeModel);
    var boneSignature = BoneSignatureForRuntimeModel(runtimeModel);
    RequireInt32(fighter, "bone_count", boneCount);
    RequireString(fighter, "runtime_bone_signature_sha256", boneSignature);
    var boneNames = RequireNonemptyUniqueStringArray(fighter, "bone_names", boneCount);
    if (HashUtf8Text(string.Join("\n", boneNames)) != boneSignature)
        throw new InvalidDataException("continuous runtime bone payload hash mismatch");
    var semanticRobotId = OptionalString(fighter, "semantic_robot_id");
    RequireBool(fighter, "semantic_t800", semanticRobotId == "t800");
    RequireBool(fighter, "semantic_g1", semanticRobotId == "g1");
    RequireBool(fighter, "exact_t800_bone_signature", runtimeModel == "t800");
    RequireBool(fighter, "exact_g1_bone_signature", runtimeModel == "g1");
    ValidateSemanticRuntimeConsistency(fighter, semanticRobotId, runtimeModel);
    RequireFalse(fighter, "semantic_robot_id_used_for_continuous_acceptance");
}

static void ValidateSemanticRuntimeConsistency(
    JsonElement value,
    string? semanticRobotId,
    string runtimeModel)
{
    var unavailable = string.IsNullOrWhiteSpace(semanticRobotId);
    var matches = semanticRobotId == runtimeModel;
    RequireBool(value, "semantic_runtime_mismatch", !unavailable && !matches);
    RequireString(
        value,
        "semantic_runtime_consistency",
        unavailable
            ? $"semantic_robot_id_unavailable_runtime_{runtimeModel}_exact"
            : matches
                ? $"semantic_and_runtime_{runtimeModel}_exact"
                : $"semantic_robot_id_mismatch_runtime_{runtimeModel}_exact");
}

static int BoneCountForRuntimeModel(string runtimeModel) => runtimeModel == "t800" ? 26 : 30;
static string BoneSignatureForRuntimeModel(string runtimeModel) =>
    runtimeModel == "t800" ? T800BoneSignatureSha256 : G1BoneSignatureSha256;
static int[] AttackMoveIndicesForRuntimeModel(string runtimeModel) =>
    runtimeModel == "t800" ? new[] { 2, 3, 4, 5, 9, 10 } : new[] { 6, 7, 8, 9 };
static string RecoveryModeForRuntimeModel(string runtimeModel) =>
    runtimeModel == "t800" ? ContinuousT800RecoveryMode : ContinuousG1RecoveryMode;

static void ValidateContinuousEvent(JsonElement value, string runId, string runtimeModel)
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
    RequireString(value, "runtime_model", runtimeModel);
    RequireString(value, "recovery_mode", RecoveryModeForRuntimeModel(runtimeModel));
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
    RequireString(local, "runtime_model", runtimeModel);
    _ = RequireNonemptyString(local, "runtime_object_name");
    RequireInt32(local, "runtime_bone_count", BoneCountForRuntimeModel(runtimeModel));
    RequireString(
        local,
        "runtime_bone_signature_sha256",
        BoneSignatureForRuntimeModel(runtimeModel));
    RequireTrue(local, "exact_local_supported_runtime_proven");
    RequireBool(local, "exact_local_t800_proven", runtimeModel == "t800");
    RequireBool(local, "exact_local_g1_proven", runtimeModel == "g1");
    ValidateSemanticRuntimeConsistency(
        local,
        OptionalString(local, "semantic_robot_id"),
        runtimeModel);
    RequireFalse(local, "semantic_robot_id_used_for_acceptance");
    var opponent = measured.GetProperty("opponent_identity");
    RequireString(opponent, "runtime_model", runtimeModel);
    var opponentRuntimeName = RequireNonemptyString(opponent, "runtime_object_name");
    RequireInt32(opponent, "runtime_bone_count", BoneCountForRuntimeModel(runtimeModel));
    var opponentBoneSignature = BoneSignatureForRuntimeModel(runtimeModel);
    RequireString(opponent, "runtime_bone_signature_sha256", opponentBoneSignature);
    RequireString(
        opponent,
        "runtime_identity_sha256",
        HashUtf8Text($"{opponentRuntimeName}\n{opponentBoneSignature}"));
    ValidateSemanticRuntimeConsistency(
        opponent,
        OptionalString(opponent, "semantic_robot_id_untrusted_for_runtime_acceptance"),
        runtimeModel);
    RequireFalse(opponent, "semantic_robot_id_used_for_acceptance");
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

static void RequireFiniteExactSingle(JsonElement parent, string name, float expected) =>
    JsonBinary32Contract.RequireExact(parent, name, expected);

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

static void RequireNullOrBoolean(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind is not (
            JsonValueKind.Null or JsonValueKind.True or JsonValueKind.False))
    {
        throw new InvalidDataException($"expected {name} to be null or boolean");
    }
}

static void RequireNullableInt32(JsonElement parent, string name, int? expected)
{
    if (!parent.TryGetProperty(name, out var value))
        throw new InvalidDataException($"expected nullable integer {name}");
    if (expected is null)
    {
        if (value.ValueKind != JsonValueKind.Null)
            throw new InvalidDataException($"expected {name}=null");
        return;
    }
    if (!value.TryGetInt32(out var actual) || actual != expected.Value)
        throw new InvalidDataException($"expected {name}={expected.Value}");
}

static void RequireNullOrNonnegativeInt32(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value))
        throw new InvalidDataException($"expected nullable nonnegative integer {name}");
    if (value.ValueKind == JsonValueKind.Null)
        return;
    if (!value.TryGetInt32(out var actual) || actual < 0)
        throw new InvalidDataException($"expected nullable nonnegative integer {name}");
}

static void ValidateOptionalString(JsonElement parent, string name)
{
    if (!parent.TryGetProperty(name, out var value) ||
        value.ValueKind is not (JsonValueKind.Null or JsonValueKind.String))
    {
        throw new InvalidDataException($"expected optional string {name}");
    }
    if (value.ValueKind == JsonValueKind.String && string.IsNullOrEmpty(value.GetString()))
        throw new InvalidDataException($"expected optional string {name} to be nonempty");
}

static void ValidateOptionalHex(JsonElement parent, string name, int length)
{
    if (!parent.TryGetProperty(name, out var value))
        throw new InvalidDataException($"expected optional hexadecimal string {name}");
    if (value.ValueKind == JsonValueKind.Null)
        return;
    _ = RequireHexString(parent, name, length);
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
