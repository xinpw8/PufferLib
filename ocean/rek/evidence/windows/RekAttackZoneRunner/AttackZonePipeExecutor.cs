using System.IO.Pipes;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using RekUiBridgeAgent;

internal static class AttackZonePipeExecutor
{
    private const string PipeName = "rek-ui-bridge-v1";
    private const string Protocol = "rek.ui_bridge.v1";
    private const string ManifestSchema = "rek.attack_zone_execution_manifest.v1";
    private const string TrialResultSchema = "rek.attack_zone_trial_transcript_result.v1";
    private const string ExpectedApplicationVersion = "0.0.119";
    private const string ExpectedUnityVersion = "6000.5.8f1";
    private const string ExpectedBridgeVersion = "0.4.0";
    internal const string ExpectedBridgeSha256 =
        "fb9e3c0a4994eafc6a45f83a32c907f5eee5f9a4d6997d5bf10d863f27faab55";
    private const string T800BoneSignatureSha256 =
        "ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab";

    internal static async Task<int> ExecuteAsync(
        LoadedArtifact artifact,
        string artifactPath,
        string outputDirectory,
        bool resume,
        int maximumTrials,
        int trialTimeoutSeconds)
    {
        if (!AttackZoneTrialContract.ValidateEmbeddedContract(out var contractReason))
            throw new InvalidDataException(contractReason);
        if (!AttackZoneTrialContract.ValidSha256(ExpectedBridgeSha256))
            throw new InvalidDataException("runner final bridge SHA-256 pin was not installed");
        if (maximumTrials <= 0 || maximumTrials > artifact.Entries.Count)
            throw new ArgumentOutOfRangeException(nameof(maximumTrials));
        if (trialTimeoutSeconds is < 30 or > 600)
            throw new ArgumentOutOfRangeException(nameof(trialTimeoutSeconds));

        var root = Path.GetFullPath(outputDirectory);
        var manifestPath = Path.Combine(root, "execution-manifest.json");
        var runnerPath = NormalizeCurrentRunnerPath(Environment.ProcessPath ??
            throw new InvalidOperationException("runner executable path was unavailable"));
        var runnerSha256 = HashFile(runnerPath);
        ValidateExecutableFileBinding(runnerPath, runnerSha256);
        if (resume)
        {
            ValidateExecutionManifest(
                manifestPath,
                artifact,
                artifactPath,
                runnerPath,
                runnerSha256);
        }
        else
        {
            if (Directory.Exists(root))
                throw new IOException($"new execution output directory already exists: {root}");
            Directory.CreateDirectory(root);
            WriteNewText(
                manifestPath,
                BuildExecutionManifestJson(
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256,
                    ContinuousBotControllerContract.ExpectedSha256,
                    ExpectedBridgeVersion,
                    DateTimeOffset.UtcNow.ToString("O")));
        }

        EnsureNoPartialTranscripts(root);
        var startOrdinal = FindResumeOrdinal(root, artifact, runnerSha256);
        var stopOrdinal = Math.Min(artifact.Entries.Count, startOrdinal + maximumTrials);
        var forbiddenRoundIdentities = new HashSet<string>(StringComparer.Ordinal);
        var completed = 0;
        var censored = 0;
        for (var ordinal = startOrdinal; ordinal < stopOrdinal; ordinal++)
        {
            var entry = artifact.Entries[ordinal];
            if (entry.ScheduleOrdinal != ordinal)
                throw new InvalidDataException("schedule ordinal sequence was not canonical");
            var result = await ExecuteTrialAsync(
                artifact,
                entry,
                root,
                runnerSha256,
                trialTimeoutSeconds,
                forbiddenRoundIdentities);
            if (result == "completed")
                completed++;
            else
                censored++;
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            status = "execution_segment_complete",
            resume,
            schedule_sha256 = artifact.ScheduleSha256,
            first_schedule_ordinal = startOrdinal,
            next_schedule_ordinal = stopOrdinal,
            completed_trial_count = completed,
            censored_trial_count = censored,
            remaining_schedule_entries = artifact.Entries.Count - stopOrdinal,
            output_directory = root,
            global_input_used = false,
            server_execution_claimed = false,
        }));
        return 0;
    }

    internal static void RunFileSafetySelfTest()
    {
        var seed = new string('d', 64);
        var entries = AttackZoneTrialContract.BuildRandomizedSchedule(
            "executor-self-test",
            0,
            seed);
        var scheduleSha256 = AttackZoneTrialContract.ComputeScheduleSha256(
            "executor-self-test",
            0,
            seed,
            1,
            entries);
        var artifact = new LoadedArtifact(
            scheduleSha256,
            seed,
            "executor-self-test",
            0,
            1,
            entries);
        var root = Path.Combine(
            Path.GetTempPath(),
            $"rek-attack-zone-executor-self-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var artifactPath = Path.Combine(root, "schedule.json");
            var runnerPath = Path.Combine(root, "RekAttackZoneRunner.exe");
            WriteNewText(artifactPath, "self-test schedule binding\n");
            WriteNewText(runnerPath, "self-test runner bytes\n");
            var runnerSha256 = HashFile(runnerPath);
            var exactManifest = Path.Combine(root, "execution-manifest.json");
            WriteNewText(
                exactManifest,
                BuildExecutionManifestJson(
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256,
                    ContinuousBotControllerContract.ExpectedSha256,
                    ExpectedBridgeVersion,
                    "2026-09-03T00:00:00.0000000+00:00"));
            ValidateExecutionManifest(
                exactManifest,
                artifact,
                artifactPath,
                runnerPath,
                runnerSha256);

            var wrongRunnerHashManifest = Path.Combine(root, "wrong-runner-hash.json");
            WriteNewText(
                wrongRunnerHashManifest,
                BuildExecutionManifestJson(
                    artifact,
                    artifactPath,
                    runnerPath,
                    new string('e', 64),
                    ContinuousBotControllerContract.ExpectedSha256,
                    ExpectedBridgeVersion,
                    "2026-09-03T00:00:00.0000000+00:00"));
            ExpectInvalidData(
                () => ValidateExecutionManifest(
                    wrongRunnerHashManifest,
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256),
                "resume accepted a different runner executable hash");

            var otherRunnerPath = Path.Combine(root, "other-runner.exe");
            WriteNewText(otherRunnerPath, "other self-test runner bytes\n");
            var wrongRunnerPathManifest = Path.Combine(root, "wrong-runner-path.json");
            WriteNewText(
                wrongRunnerPathManifest,
                BuildExecutionManifestJson(
                    artifact,
                    artifactPath,
                    otherRunnerPath,
                    runnerSha256,
                    ContinuousBotControllerContract.ExpectedSha256,
                    ExpectedBridgeVersion,
                    "2026-09-03T00:00:00.0000000+00:00"));
            ExpectInvalidData(
                () => ValidateExecutionManifest(
                    wrongRunnerPathManifest,
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256),
                "resume accepted a different runner executable path");

            var staleControllerManifest = Path.Combine(root, "stale-controller.json");
            WriteNewText(
                staleControllerManifest,
                BuildExecutionManifestJson(
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256,
                    new string('e', 64),
                    ExpectedBridgeVersion,
                    "2026-09-03T00:00:00.0000000+00:00"));
            ExpectInvalidData(
                () => ValidateExecutionManifest(
                    staleControllerManifest,
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256),
                "resume accepted a stale controller contract");

            var staleBridgeVersionManifest = Path.Combine(root, "stale-bridge-version.json");
            WriteNewText(
                staleBridgeVersionManifest,
                BuildExecutionManifestJson(
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256,
                    ContinuousBotControllerContract.ExpectedSha256,
                    "0.3.9",
                    "2026-09-03T00:00:00.0000000+00:00"));
            ExpectInvalidData(
                () => ValidateExecutionManifest(
                    staleBridgeVersionManifest,
                    artifact,
                    artifactPath,
                    runnerPath,
                    runnerSha256),
                "resume accepted a stale bridge version");

            var first = Path.Combine(root, TrialFileName(entries[0]));
            var completedTranscript = CompletedSelfTestTranscript(
                artifact,
                entries[0],
                runnerSha256);
            WriteNewText(first, completedTranscript);
            if (FindResumeOrdinal(root, artifact, runnerSha256) != 1)
                throw new InvalidDataException("safe resume did not select first absent ordinal");
            var preserved = File.ReadAllText(first);
            var noReplace = false;
            try
            {
                WriteNewText(first, "replacement");
            }
            catch (IOException)
            {
                noReplace = true;
            }
            if (!noReplace || File.ReadAllText(first) != preserved)
                throw new InvalidDataException("completed transcript was overwritten");

            var tampered = Path.Combine(root, "tampered-earlier-event.jsonl");
            var tamperedTranscript = string.Join(
                "\n",
                completedTranscript
                    .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                    .Where(line => !line.Contains(
                        "\"event\":\"target_requested\"",
                        StringComparison.Ordinal))) + "\n";
            WriteNewText(tampered, tamperedTranscript);
            ExpectInvalidData(
                () => ValidateCompletedTranscript(
                    tampered,
                    artifact,
                    entries[0],
                    runnerSha256),
                "resume accepted a transcript with an earlier event removed");

            var third = Path.Combine(root, TrialFileName(entries[2]));
            WriteNewText(
                third,
                CompletedSelfTestTranscript(artifact, entries[2], runnerSha256));
            var gapRejected = false;
            try
            {
                _ = FindResumeOrdinal(root, artifact, runnerSha256);
            }
            catch (InvalidDataException)
            {
                gapRejected = true;
            }
            if (!gapRejected)
                throw new InvalidDataException("non-contiguous resume was accepted");
            File.Delete(third);

            var partial = Path.Combine(root, TrialFileName(entries[1]) + ".partial-test");
            WriteNewText(partial, "partial\n");
            var partialRejected = false;
            try
            {
                EnsureNoPartialTranscripts(root);
            }
            catch (InvalidDataException)
            {
                partialRejected = true;
            }
            if (!partialRejected)
                throw new InvalidDataException("ambiguous partial resume was accepted");

            File.AppendAllText(runnerPath, "mutated\n", new UTF8Encoding(false));
            ExpectInvalidData(
                () => ValidateExecutableFileBinding(runnerPath, runnerSha256),
                "runner executable changed after its hash was observed");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string CompletedSelfTestTranscript(
        LoadedArtifact artifact,
        AttackZoneScheduleEntry entry,
        string runnerSha256)
    {
        var trialId = TrialId(artifact.ScheduleSha256, entry.ScheduleOrdinal);
        var sessionSha256 = new string('a', 64);
        var roundSha256 = new string('b', 64);
        var target = AttackZoneTrialContract.CreateTarget(
            entry,
            artifact.ScheduleSha256,
            artifact.Seed,
            sessionSha256,
            roundSha256,
            trialId,
            entry.ScheduleOrdinal + 1);
        using var targetDocument = JsonDocument.Parse(
            AttackZoneTrialContract.SerializeTarget(target));
        var lines = new[]
        {
            JsonSerializer.Serialize(new
            {
                @event = "runner_trial_binding",
                observed_utc = "2026-09-03T00:00:00.0000000+00:00",
                schedule_sha256 = artifact.ScheduleSha256,
                schedule_ordinal = entry.ScheduleOrdinal,
                independent_run_id = artifact.IndependentRunId,
                independent_run_ordinal = artifact.IndependentRunOrdinal,
                trial_id = trialId,
                move_index = entry.MoveIndex,
                distance_bin_id = entry.DistanceBin.Id,
                bearing_bin_id = entry.BearingBin.Id,
                runner_executable_sha256_observed = runnerSha256,
                expected_bridge_sha256 = ExpectedBridgeSha256,
                attack_zone_trial_sha256 = AttackZoneTrialContract.ExpectedSha256,
                global_input_used = false,
                server_execution_claimed = false,
            }),
            JsonSerializer.Serialize(new
            {
                @event = "client_request",
                observed_utc = "2026-09-03T00:00:00.0100000+00:00",
                request = new
                {
                    type = "command",
                    request_id = "self-test-attack",
                    command = "StartAttackZoneTrial",
                    target = targetDocument.RootElement,
                },
                global_input_used = false,
            }),
            SelfTestAttackEvent(
                artifact,
                entry,
                target,
                "target_requested",
                "target_request_received",
                1),
            SelfTestAttackEvent(
                artifact,
                entry,
                target,
                "trial_censored",
                "target_acquisition_timeout_unresolved_no_attack",
                2),
            JsonSerializer.Serialize(new
            {
                @event = "runner_trial_result",
                result_schema = TrialResultSchema,
                observed_utc = "2026-09-03T00:00:00.0500000+00:00",
                status = "censored",
                terminal_event = "trial_censored",
                terminal_reason = "target_acquisition_timeout_unresolved_no_attack",
                schedule_sha256 = artifact.ScheduleSha256,
                schedule_ordinal = entry.ScheduleOrdinal,
                trial_id = trialId,
                session_identity_sha256 = sessionSha256,
                round_identity_sha256 = roundSha256,
                event_counts = new Dictionary<string, int>(StringComparer.Ordinal)
                {
                    ["target_requested"] = 1,
                    ["trial_censored"] = 1,
                },
                action_sample_count = 0,
                recovery_upright_readiness_proven = false,
                remainder_of_round_excluded = false,
                global_input_used = false,
                server_execution_claimed = false,
            }),
        };
        return string.Join("\n", lines) + "\n";
    }

    private static string SelfTestAttackEvent(
        LoadedArtifact artifact,
        AttackZoneScheduleEntry entry,
        AttackZoneTrialTarget target,
        string eventName,
        string reason,
        int controlTick) => JsonSerializer.Serialize(new
        {
            @event = eventName,
            attack_zone_schema = AttackZoneTrialContract.Schema,
            attack_zone_protocol_sha256 = AttackZoneTrialContract.ExpectedSha256,
            continuous_controller_sha256 = ContinuousBotControllerContract.ExpectedSha256,
            authority_scope = AttackZoneTrialContract.AuthorityScope,
            authority_caveat = AttackZoneTrialContract.AuthorityCaveat,
            isolated_spark_proof = AttackZoneTrialContract.RequiredIsolationProof,
            schedule_sha256 = artifact.ScheduleSha256,
            schedule_ordinal = entry.ScheduleOrdinal,
            independent_run_id = artifact.IndependentRunId,
            trial_id = target.TrialId,
            move_index = entry.MoveIndex,
            serialized_asset_sha256 = entry.SerializedAssetSha256,
            global_input_used = false,
            server_acceptance_observed = false,
            authoritative_execution_observed = false,
            client_request_observation_only = true,
            fixed_substeps_per_control_tick =
                AttackZoneTrialContract.FixedSubstepsPerControlTick,
            session_identity_sha256 = target.SessionIdentitySha256,
            round_identity_sha256 = target.RoundIdentitySha256,
            stopwatch_timestamp_ticks = controlTick * 200_000L,
            stopwatch_frequency_hz = 10_000_000L,
            utc = $"2026-09-03T00:00:00.{controlTick * 2:D2}00000+00:00",
            unity_frame = controlTick,
            unity_time = controlTick * 0.02,
            unity_fixed_time = controlTick * 0.02,
            client_control_tick = controlTick,
            client_fixed_substep =
                controlTick * AttackZoneTrialContract.FixedSubstepsPerControlTick,
            controller_reason = reason,
        });

    private static void ExpectInvalidData(Action action, string failureMessage)
    {
        try
        {
            action();
        }
        catch (InvalidDataException)
        {
            return;
        }
        throw new InvalidDataException(failureMessage);
    }

    private static void EnsureNoPartialTranscripts(string root)
    {
        var partials = Directory.GetFiles(
            root,
            "trial-*.jsonl.partial-*",
            SearchOption.TopDirectoryOnly);
        if (partials.Length != 0)
        {
            throw new InvalidDataException(
                "unresolved partial trial transcript exists; inspect it before an explicit recovery decision");
        }
    }

    private static async Task<string> ExecuteTrialAsync(
        LoadedArtifact artifact,
        AttackZoneScheduleEntry entry,
        string root,
        string runnerSha256,
        int timeoutSeconds,
        HashSet<string> forbiddenRoundIdentities)
    {
        var trialId = TrialId(artifact.ScheduleSha256, entry.ScheduleOrdinal);
        var finalPath = Path.Combine(root, TrialFileName(entry));
        if (File.Exists(finalPath))
            throw new IOException($"trial transcript already exists: {finalPath}");
        var partialPath = finalPath + $".partial-{Environment.ProcessId}-{Guid.NewGuid():N}";
        await using var transcript = new StreamWriter(
            new FileStream(
                partialPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.Read | FileShare.Delete),
            new UTF8Encoding(false))
        {
            AutoFlush = true,
            NewLine = "\n",
        };
        await transcript.WriteLineAsync(JsonSerializer.Serialize(new
        {
            @event = "runner_trial_binding",
            observed_utc = DateTimeOffset.UtcNow.ToString("O"),
            schedule_sha256 = artifact.ScheduleSha256,
            schedule_ordinal = entry.ScheduleOrdinal,
            independent_run_id = artifact.IndependentRunId,
            independent_run_ordinal = artifact.IndependentRunOrdinal,
            trial_id = trialId,
            move_index = entry.MoveIndex,
            distance_bin_id = entry.DistanceBin.Id,
            bearing_bin_id = entry.BearingBin.Id,
            runner_executable_sha256_observed = runnerSha256,
            expected_bridge_sha256 = ExpectedBridgeSha256,
            attack_zone_trial_sha256 = AttackZoneTrialContract.ExpectedSha256,
            global_input_used = false,
            server_execution_claimed = false,
        }));

        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        var validator = new TrialEventValidator(artifact, entry, trialId);
        await using var session = new TrialPipeSession(transcript, validator);
        var leaseHeld = false;
        try
        {
            await session.ConnectAsync(deadline.Token);
            using (var hello = await session.ReadUntilAsync("hello", null, deadline.Token))
                ValidateHello(hello.RootElement);
            var connectionId = session.ConnectionId;
            using (var preflight = await session.StateAsync(deadline.Token))
                ValidatePinnedState(preflight.RootElement, connectionId, requireLease: false);
            using (var acquire = await session.CommandAsync(
                       "AcquireExclusiveControl",
                       deadline.Token))
            {
                ValidateAcceptedAck(
                    acquire.RootElement,
                    "AcquireExclusiveControl",
                    "exclusive_control_lease_acquired",
                    expectedApplied: true,
                    expectedRequestIssued: false,
                    connectionId);
            }
            leaseHeld = true;

            var scope = await EnsureActivePrivateScopeAsync(
                session,
                connectionId,
                forbiddenRoundIdentities,
                deadline.Token);
            var target = AttackZoneTrialContract.CreateTarget(
                entry,
                artifact.ScheduleSha256,
                artifact.Seed,
                scope.SessionIdentitySha256,
                scope.RoundIdentitySha256,
                trialId,
                entry.ScheduleOrdinal + 1);
            if (!AttackZoneTrialContract.TryValidateTarget(target, out _, out var targetReason))
                throw new InvalidDataException(targetReason);
            validator.BindTarget(target);
            using (var start = await session.AttackCommandAsync(target, deadline.Token))
            {
                ValidateAcceptedAck(
                    start.RootElement,
                    "StartAttackZoneTrial",
                    "attack_zone_target_acquisition_started",
                    expectedApplied: true,
                    expectedRequestIssued: false,
                    connectionId);
                RequireTrue(start.RootElement, "attack_zone_trial_running");
                RequireString(start.RootElement, "attack_zone_trial_id", trialId);
                RequireString(
                    start.RootElement,
                    "attack_zone_trial_schedule_sha256",
                    artifact.ScheduleSha256);
                RequireInt32(
                    start.RootElement,
                    "attack_zone_trial_schedule_ordinal",
                    entry.ScheduleOrdinal);
            }

            while (validator.TerminalEvent is null)
            {
                using var message = await session.ReadAsync(deadline.Token);
            }
            if (validator.TerminalEvent == "trial_interrupted")
            {
                throw new InvalidDataException(
                    $"attack-zone trial interrupted: {validator.TerminalReason}");
            }

            await WaitForAttackControllerQuiescenceAsync(
                session,
                connectionId,
                validator,
                deadline.Token);
            if (validator.PostStartRoundExcluded)
            {
                forbiddenRoundIdentities.Add(scope.RoundIdentitySha256);
                await FinishExcludedRoundAsync(
                    session,
                    connectionId,
                    scope.RoundIdentitySha256,
                    deadline.Token);
            }

            using (var release = await session.CommandAsync(
                       "ReleaseExclusiveControl",
                       deadline.Token))
            {
                ValidateAcceptedAck(
                    release.RootElement,
                    "ReleaseExclusiveControl",
                    "exclusive_control_lease_released",
                    expectedApplied: true,
                    expectedRequestIssued: false,
                    expectedConnectionId: null);
                RequireNull(release.RootElement, "lease_connection_id");
            }
            leaseHeld = false;
            validator.ValidateTerminalEvidence();
            var result = validator.TerminalEvent == "trial_completed"
                ? "completed"
                : "censored";
            await transcript.WriteLineAsync(JsonSerializer.Serialize(new
            {
                @event = "runner_trial_result",
                result_schema = TrialResultSchema,
                observed_utc = DateTimeOffset.UtcNow.ToString("O"),
                status = result,
                terminal_event = validator.TerminalEvent,
                terminal_reason = validator.TerminalReason,
                schedule_sha256 = artifact.ScheduleSha256,
                schedule_ordinal = entry.ScheduleOrdinal,
                trial_id = trialId,
                session_identity_sha256 = scope.SessionIdentitySha256,
                round_identity_sha256 = scope.RoundIdentitySha256,
                event_counts = validator.EventCounts,
                action_sample_count = validator.ActionSampleCount,
                recovery_upright_readiness_proven = validator.RecoveryReadinessProven,
                remainder_of_round_excluded = validator.PostStartRoundExcluded,
                global_input_used = false,
                server_execution_claimed = false,
            }));
            await FlushToDiskAsync(transcript);
            File.Move(partialPath, finalPath, overwrite: false);
            return result;
        }
        catch (Exception exception)
        {
            if (leaseHeld)
            {
                try
                {
                    using var releaseDeadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                    using var release = await session.CommandAsync(
                        "ReleaseExclusiveControl",
                        releaseDeadline.Token);
                    leaseHeld = false;
                }
                catch
                {
                }
            }
            await transcript.WriteLineAsync(JsonSerializer.Serialize(new
            {
                @event = "runner_trial_result",
                result_schema = TrialResultSchema,
                observed_utc = DateTimeOffset.UtcNow.ToString("O"),
                status = "failed_partial_preserved",
                error = $"{exception.GetType().Name}:{exception.Message}",
                schedule_sha256 = artifact.ScheduleSha256,
                schedule_ordinal = entry.ScheduleOrdinal,
                trial_id = trialId,
                lease_held = leaseHeld,
                global_input_used = false,
                server_execution_claimed = false,
            }));
            await FlushToDiskAsync(transcript);
            throw new InvalidDataException(
                $"trial {entry.ScheduleOrdinal} failed; partial transcript preserved at {partialPath}",
                exception);
        }
    }

    private static async Task<ScopePins> EnsureActivePrivateScopeAsync(
        TrialPipeSession session,
        long connectionId,
        HashSet<string> forbiddenRoundIdentities,
        CancellationToken cancellationToken)
    {
        for (;;)
        {
            using var state = await session.StateAsync(cancellationToken);
            ValidatePinnedState(state.RootElement, connectionId);
            var scene = RequiredString(state.RootElement, "scene");
            if (scene == "Lobby")
            {
                var screen = RequiredString(state.RootElement, "lobby_screen");
                if (screen == "Intro")
                {
                    await Task.Delay(100, cancellationToken);
                    continue;
                }
                if (screen == "Login")
                {
                    using var ack = await session.CommandAsync("ConfirmLoggedIn", cancellationToken);
                    ValidateAcceptedAck(
                        ack.RootElement,
                        "ConfirmLoggedIn",
                        "home_screen_observed_after_lets_go",
                        true,
                        false,
                        connectionId);
                    continue;
                }
                if (screen == "Home")
                {
                    using var ack = await session.CommandAsync("NavigateFreePlay", cancellationToken);
                    ValidateAcceptedAck(
                        ack.RootElement,
                        "NavigateFreePlay",
                        "free_play_screen_observed",
                        true,
                        false,
                        connectionId);
                    continue;
                }
                if (screen != "FreePlay")
                    throw new InvalidDataException($"unsupported lobby screen {screen}");
                using (var ack = await session.CommandAsync("EnterSolo", cancellationToken))
                {
                    ValidateAcceptedAck(
                        ack.RootElement,
                        "EnterSolo",
                        "private_practice_reservation_requested",
                        false,
                        true,
                        connectionId);
                }
                await Task.Delay(100, cancellationToken);
                continue;
            }

            var privateAi = state.RootElement.GetProperty("private_ai");
            if (!privateAi.TryGetProperty("proven", out var privateProven) ||
                privateProven.ValueKind != JsonValueKind.True)
            {
                await Task.Delay(100, cancellationToken);
                continue;
            }
            ValidatePrivateBotOne(privateAi);
            if (!RequiredBoolean(privateAi, "round_active"))
            {
                if (RequiredBoolean(privateAi, "post_fight_prompt") &&
                    !RequiredNullableBoolean(privateAi, "post_fight_is_winner"))
                {
                    using var exit = await session.CommandAsync(
                        "ExitLostPrivateSession",
                        cancellationToken);
                    ValidateAcceptedAck(
                        exit.RootElement,
                        "ExitLostPrivateSession",
                        "post_fight_loser_exit_request_issued",
                        false,
                        true,
                        connectionId);
                }
                else
                {
                    using var start = await session.CommandAsync("StartRound", cancellationToken);
                    ValidateStartRoundAck(start.RootElement, connectionId);
                }
                await Task.Delay(50, cancellationToken);
                continue;
            }

            RequireTrue(privateAi, "active_gameplay_proven");
            ValidateMeasuredPairing(state.RootElement.GetProperty("measured_pairing"));
            var control = state.RootElement.GetProperty("control");
            var availability = control.GetProperty("attack_zone_trial_availability");
            if (!RequiredBoolean(availability, "available"))
            {
                await Task.Delay(50, cancellationToken);
                continue;
            }
            var sessionHash = RequireSha256(availability, "session_identity_sha256");
            var roundHash = RequireSha256(availability, "round_identity_sha256");
            if (forbiddenRoundIdentities.Contains(roundHash))
            {
                throw new InvalidDataException(
                    "runner refused to relabel or reuse a post-start contaminated round");
            }
            return new ScopePins(sessionHash, roundHash);
        }
    }

    private static async Task WaitForAttackControllerQuiescenceAsync(
        TrialPipeSession session,
        long connectionId,
        TrialEventValidator validator,
        CancellationToken cancellationToken)
    {
        var recoveryWasRunning = false;
        for (;;)
        {
            using var state = await session.StateAsync(cancellationToken);
            ValidatePinnedState(state.RootElement, connectionId);
            var control = state.RootElement.GetProperty("control");
            var trialRunning = RequiredBoolean(control, "attack_zone_trial_running");
            var recoveryRunning = RequiredBoolean(control, "attack_zone_recovery_only_running");
            var continuousRunning = RequiredBoolean(control, "continuous_controller_running");
            recoveryWasRunning |= recoveryRunning;
            if (!trialRunning && !recoveryRunning && !continuousRunning)
                break;
            await Task.Delay(20, cancellationToken);
        }
        if ((recoveryWasRunning || validator.RecoveryContinuationObserved) &&
            !validator.RecoveryReadinessProven)
        {
            using var boundary = await session.StateAsync(cancellationToken);
            ValidatePinnedState(boundary.RootElement, connectionId);
            var privateAi = boundary.RootElement.GetProperty("private_ai");
            var inactiveResetAvailable =
                privateAi.TryGetProperty("proven", out var proven) &&
                proven.ValueKind == JsonValueKind.True &&
                !RequiredBoolean(privateAi, "round_active");
            if (!inactiveResetAvailable)
            {
                throw new InvalidDataException(
                    "recovery-only controller stopped without upright proof or an inactive-round reset boundary");
            }
        }
    }

    private static async Task FinishExcludedRoundAsync(
        TrialPipeSession session,
        long connectionId,
        string excludedRoundIdentitySha256,
        CancellationToken cancellationToken)
    {
        using (var initial = await session.StateAsync(cancellationToken))
        {
            ValidatePinnedState(initial.RootElement, connectionId);
            var privateAi = initial.RootElement.GetProperty("private_ai");
            if (privateAi.TryGetProperty("proven", out var proven) &&
                proven.ValueKind == JsonValueKind.True &&
                !RequiredBoolean(privateAi, "round_active"))
            {
                await HandleInactiveExcludedRoundBoundaryAsync(
                    session,
                    connectionId,
                    privateAi,
                    cancellationToken);
                return;
            }
        }
        using (var start = await session.CommandAsync(
                   "StartContinuousBotController",
                   cancellationToken))
        {
            ValidateAcceptedAck(
                start.RootElement,
                "StartContinuousBotController",
                "continuous_private_bot_controller_started",
                true,
                false,
                connectionId);
        }
        for (;;)
        {
            using var state = await session.StateAsync(cancellationToken);
            ValidatePinnedState(state.RootElement, connectionId);
            var privateAi = state.RootElement.GetProperty("private_ai");
            if (privateAi.TryGetProperty("proven", out var proven) && proven.ValueKind == JsonValueKind.True)
            {
                ValidatePrivateBotOne(privateAi);
                if (!RequiredBoolean(privateAi, "round_active"))
                {
                    var control = state.RootElement.GetProperty("control");
                    if (RequiredBoolean(control, "continuous_controller_running"))
                    {
                        using var stop = await session.CommandAsync(
                            "StopContinuousBotController",
                            cancellationToken);
                        ValidateAcceptedAck(
                            stop.RootElement,
                            "StopContinuousBotController",
                            "continuous_private_bot_controller_stopped",
                            true,
                            false,
                            connectionId);
                    }
                    await HandleInactiveExcludedRoundBoundaryAsync(
                        session,
                        connectionId,
                        privateAi,
                        cancellationToken);
                    return;
                }
            }
            var controlState = state.RootElement.GetProperty("control");
            if (!RequiredBoolean(controlState, "continuous_controller_running"))
            {
                throw new InvalidDataException(
                    $"cleanup controller stopped before excluded round {excludedRoundIdentitySha256} ended");
            }
            await Task.Delay(20, cancellationToken);
        }
    }

    private static async Task HandleInactiveExcludedRoundBoundaryAsync(
        TrialPipeSession session,
        long connectionId,
        JsonElement privateAi,
        CancellationToken cancellationToken)
    {
        if (!RequiredBoolean(privateAi, "post_fight_prompt"))
            throw new InvalidDataException("excluded round ended without a post-fight prompt");
        var winner = RequiredNullableBoolean(privateAi, "post_fight_is_winner");
        if (winner)
            return;
        using var exit = await session.CommandAsync(
            "ExitLostPrivateSession",
            cancellationToken);
        ValidateAcceptedAck(
            exit.RootElement,
            "ExitLostPrivateSession",
            "post_fight_loser_exit_request_issued",
            false,
            true,
            connectionId);
        await ProveLobbyAfterLossAsync(session, connectionId, cancellationToken);
    }

    private static async Task ProveLobbyAfterLossAsync(
        TrialPipeSession session,
        long connectionId,
        CancellationToken cancellationToken)
    {
        for (;;)
        {
            await Task.Delay(50, cancellationToken);
            using var state = await session.StateAsync(cancellationToken);
            ValidatePinnedState(state.RootElement, connectionId);
            if (RequiredString(state.RootElement, "scene") != "Lobby")
                continue;
            var control = state.RootElement.GetProperty("control");
            RequireFalse(control, "continuous_controller_running");
            RequireFalse(control, "attack_zone_trial_running");
            RequireFalse(control, "attack_zone_recovery_only_running");
            return;
        }
    }

    private static void ValidateHello(JsonElement hello)
    {
        RequireString(hello, "event", "hello");
        RequireString(hello, "protocol", Protocol);
        RequireString(hello, "pipe", PipeName);
        RequireTrue(hello, "current_user_only");
        RequireTrue(hello, "local_computer_verified");
        var capabilities = hello.GetProperty("capabilities");
        RequireFalse(capabilities, "input_available");
        RequireTrue(capabilities, "exclusive_control_lease_required");
        RequireTrue(capabilities, "autonomous_semantic_controller");
        RequireString(
            capabilities,
            "attack_zone_trial_schema",
            AttackZoneTrialContract.Schema);
        RequireString(
            capabilities,
            "attack_zone_trial_sha256",
            AttackZoneTrialContract.ExpectedSha256);
        RequireString(
            capabilities,
            "attack_zone_trial_authority_scope",
            AttackZoneTrialContract.AuthorityScope);
        RequireString(
            capabilities,
            "attack_zone_trial_authority_caveat",
            AttackZoneTrialContract.AuthorityCaveat);
        RequireString(
            capabilities,
            "attack_zone_trial_required_isolation_proof",
            AttackZoneTrialContract.RequiredIsolationProof);
        RequireInt32(capabilities, "attack_zone_trial_control_rate_hz", 50);
        RequireInt32(capabilities, "attack_zone_trial_fixed_substeps_per_tick", 10);
        RequireInt32(capabilities, "attack_zone_trial_settle_ticks", 15);
        RequireInt32(capabilities, "attack_zone_trial_action_sample_rate_hz", 50);
        RequireInt32(capabilities, "attack_zone_trial_recovery_ready_ticks", 15);
        RequireInt32(capabilities, "attack_zone_trial_acquisition_timeout_ticks", 500);
        RequireInt32(capabilities, "attack_zone_trial_minimum_independent_runs_per_cell", 5);
        RequireString(
            capabilities,
            "attack_zone_trial_recorder_version",
            AttackZoneTrialContract.ExpectedRecorderVersion);
        RequireString(
            capabilities,
            "attack_zone_trial_recorder_plugin_sha256",
            AttackZoneTrialContract.ExpectedRecorderPluginSha256);
        RequireFalse(capabilities, "attack_zone_trial_global_input_emitted");
        var commands = capabilities.GetProperty("semantic_commands")
            .EnumerateArray()
            .Select(value => value.GetString())
            .ToHashSet(StringComparer.Ordinal);
        foreach (var required in new[]
                 {
                     "AcquireExclusiveControl", "ReleaseExclusiveControl",
                     "ConfirmLoggedIn", "NavigateFreePlay", "EnterSolo", "StartRound",
                     "ExitLostPrivateSession", "StartContinuousBotController",
                     "StopContinuousBotController", "StartAttackZoneTrial",
                     "StopAttackZoneTrial",
                 })
        {
            if (!commands.Contains(required))
                throw new InvalidDataException($"missing semantic command {required}");
        }
    }

    private static void ValidatePinnedState(
        JsonElement state,
        long connectionId,
        bool requireLease = true)
    {
        RequireString(state, "protocol", Protocol);
        RequireString(state, "application_version", ExpectedApplicationVersion);
        RequireString(state, "unity_version", ExpectedUnityVersion);
        var build = state.GetProperty("build");
        RequireString(
            build,
            "game_assembly_sha256",
            AttackZoneTrialContract.ExpectedGameAssemblySha256);
        RequireString(
            build,
            "global_metadata_sha256",
            AttackZoneTrialContract.ExpectedGlobalMetadataSha256);
        RequireString(build, "plugin_version", ExpectedBridgeVersion);
        RequireString(build, "plugin_sha256", ExpectedBridgeSha256);
        var foreground = state.GetProperty("foreground");
        RequireTrue(foreground, "isolated_session_verified");
        RequireTrue(foreground, "mutation_allowed");
        RequireString(
            foreground,
            "isolated_session_proof",
            AttackZoneTrialContract.RequiredIsolationProof);
        var control = state.GetProperty("control");
        RequireTrue(control, "semantic_available");
        RequireBoolean(control, "lease_held", requireLease);
        if (requireLease)
            RequireInt64(control, "lease_connection_id", connectionId);
        else
            RequireNull(control, "lease_connection_id");
        RequireString(control, "attack_zone_trial_schema", AttackZoneTrialContract.Schema);
        RequireString(
            control,
            "attack_zone_trial_sha256",
            AttackZoneTrialContract.ExpectedSha256);
        RequireString(
            control,
            "attack_zone_trial_authority_scope",
            AttackZoneTrialContract.AuthorityScope);
        RequireString(
            control,
            "attack_zone_trial_authority_caveat",
            AttackZoneTrialContract.AuthorityCaveat);
        RequireString(
            control,
            "attack_zone_trial_recorder_version",
            AttackZoneTrialContract.ExpectedRecorderVersion);
        RequireString(
            control,
            "attack_zone_trial_recorder_plugin_sha256",
            AttackZoneTrialContract.ExpectedRecorderPluginSha256);
        var input = state.GetProperty("input");
        RequireFalse(input, "global_input_available");
        RequireTrue(input, "semantic_commands_available");
    }

    private static void ValidatePrivateBotOne(JsonElement value)
    {
        RequireTrue(value, "proven");
        RequireTrue(value, "network_client_only");
        RequireTrue(value, "context_is_solo");
        RequireTrue(value, "multiplayer_session_privacy_known");
        RequireTrue(value, "multiplayer_session_is_private");
        RequireTrue(value, "opponent_is_ai");
        RequireTrue(value, "opponent_slot_is_ai");
        RequireFalse(value, "human_in_opponent_slot");
        RequireTrue(value, "opponent_slot_client_known");
        RequireFalse(value, "opponent_slot_has_client");
        RequireFalse(value, "opponent_human_bit_set");
        RequireInt32(value, "client_ai_difficulty", 0);
        RequireInt32(value, "sparring_bot_number", 1);
        RequireTrue(value, "exact_sparring_bot_1");
    }

    private static void ValidateMeasuredPairing(JsonElement pairing)
    {
        _ = RequiredBoolean(pairing, "exact_t800_vs_t800");
        RequireString(pairing, "required_pairing", "t800_vs_t800");
        RequireString(pairing, "required_robot_id", "t800");
        RequireInt32(pairing, "required_t800_bone_count", 26);
        RequireString(
            pairing,
            "required_t800_bone_signature_sha256",
            T800BoneSignatureSha256);
        var local = pairing.GetProperty("local_fighter");
        RequireString(local, "semantic_robot_id", "t800");
        RequireTrue(local, "semantic_t800");
        RequireTrue(local, "exact_t800_bone_signature");
        RequireInt32(local, "bone_count", 26);
        RequireString(local, "runtime_bone_signature_sha256", T800BoneSignatureSha256);
        var opponent = pairing.GetProperty("opponent_fighter");
        _ = RequiredString(opponent, "runtime_object_name");
        RequireTrue(opponent, "exact_t800_bone_signature");
        RequireInt32(opponent, "bone_count", 26);
        RequireString(opponent, "runtime_bone_signature_sha256", T800BoneSignatureSha256);
        RequireFalse(opponent, "semantic_robot_id_used_for_continuous_acceptance");
    }

    private static void ValidateAcceptedAck(
        JsonElement ack,
        string command,
        string reason,
        bool expectedApplied,
        bool expectedRequestIssued,
        long? expectedConnectionId)
    {
        RequireString(ack, "event", "ack");
        RequireString(ack, "protocol", Protocol);
        RequireString(ack, "command", command);
        RequireString(ack, "status", "accepted");
        RequireString(ack, "reason", reason);
        RequireBoolean(ack, "applied", expectedApplied);
        RequireBoolean(ack, "client_request_issued", expectedRequestIssued);
        RequireFalse(ack, "server_acceptance_observed");
        RequireFalse(ack, "authoritative_execution_observed");
        RequireString(
            ack,
            "attack_zone_trial_schema",
            AttackZoneTrialContract.Schema);
        RequireString(
            ack,
            "attack_zone_trial_sha256",
            AttackZoneTrialContract.ExpectedSha256);
        var build = ack.GetProperty("build");
        RequireString(build, "plugin_sha256", ExpectedBridgeSha256);
        if (expectedConnectionId is null)
            RequireNull(ack, "lease_connection_id");
        else
            RequireInt64(ack, "lease_connection_id", expectedConnectionId.Value);
    }

    private static void ValidateStartRoundAck(JsonElement ack, long connectionId)
    {
        RequireString(ack, "event", "ack");
        RequireString(ack, "protocol", Protocol);
        RequireString(ack, "command", "StartRound");
        RequireString(ack, "status", "accepted");
        RequireInt64(ack, "lease_connection_id", connectionId);
        var reason = RequiredString(ack, "reason");
        if (reason is "post_fight_continue_request_issued" or "remote_ready_request_issued")
        {
            RequireFalse(ack, "applied");
            RequireTrue(ack, "client_request_issued");
        }
        else if (reason == "native_start_fight_coroutine_observed")
        {
            RequireTrue(ack, "applied");
            RequireFalse(ack, "client_request_issued");
        }
        else
        {
            throw new InvalidDataException($"unexpected StartRound reason {reason}");
        }
    }

    private static int FindResumeOrdinal(
        string root,
        LoadedArtifact artifact,
        string runnerSha256)
    {
        var firstMissing = artifact.Entries.Count;
        for (var ordinal = 0; ordinal < artifact.Entries.Count; ordinal++)
        {
            var path = Path.Combine(root, TrialFileName(artifact.Entries[ordinal]));
            if (!File.Exists(path))
            {
                firstMissing = ordinal;
                break;
            }
            ValidateCompletedTranscript(
                path,
                artifact,
                artifact.Entries[ordinal],
                runnerSha256);
        }
        for (var ordinal = firstMissing + 1; ordinal < artifact.Entries.Count; ordinal++)
        {
            if (File.Exists(Path.Combine(root, TrialFileName(artifact.Entries[ordinal]))))
                throw new InvalidDataException("non-contiguous completed trial transcript set");
        }
        return firstMissing;
    }

    private static void ValidateCompletedTranscript(
        string path,
        LoadedArtifact artifact,
        AttackZoneScheduleEntry entry,
        string runnerSha256)
    {
        var trialId = TrialId(artifact.ScheduleSha256, entry.ScheduleOrdinal);
        var validator = new TrialEventValidator(artifact, entry, trialId);
        AttackZoneTrialTarget? target = null;
        var bindingSeen = false;
        var targetCommandSeen = false;
        var resultSeen = false;
        var lineNumber = 0;
        foreach (var line in File.ReadLines(path))
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line))
                throw new InvalidDataException($"blank transcript line {lineNumber} in {path}");
            if (Encoding.UTF8.GetByteCount(line) > 4 * 1024 * 1024)
                throw new InvalidDataException($"oversized transcript line {lineNumber} in {path}");
            if (resultSeen)
                throw new InvalidDataException($"data followed terminal runner result in {path}");
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"non-object transcript line {lineNumber} in {path}");
            var eventName = RequiredString(root, "event");
            switch (eventName)
            {
                case "runner_trial_binding":
                    if (bindingSeen || lineNumber != 1)
                        throw new InvalidDataException($"trial binding was not unique and first in {path}");
                    ValidateTrialBinding(root, artifact, entry, trialId, runnerSha256);
                    bindingSeen = true;
                    break;
                case "client_request":
                    if (!bindingSeen)
                        throw new InvalidDataException($"client request preceded trial binding in {path}");
                    var parsedTarget = ValidateRecordedClientRequest(
                        root,
                        artifact,
                        entry,
                        trialId);
                    if (parsedTarget is not null)
                    {
                        if (targetCommandSeen)
                            throw new InvalidDataException($"duplicate attack target command in {path}");
                        target = parsedTarget;
                        validator.BindTarget(target);
                        targetCommandSeen = true;
                    }
                    break;
                case "runner_trial_result":
                    if (!bindingSeen || !targetCommandSeen || target is null)
                        throw new InvalidDataException($"runner result preceded exact target binding in {path}");
                    validator.ValidateTerminalEvidence();
                    ValidateRunnerTrialResult(root, artifact, entry, trialId, target, validator);
                    resultSeen = true;
                    break;
                default:
                    if (!bindingSeen)
                        throw new InvalidDataException($"bridge data preceded trial binding in {path}");
                    break;
            }
            validator.Observe(root);
        }
        if (lineNumber == 0)
            throw new InvalidDataException($"empty completed transcript {path}");
        if (!bindingSeen || !targetCommandSeen || !resultSeen)
            throw new InvalidDataException($"completed transcript structure was incomplete in {path}");
    }

    private static void ValidateTrialBinding(
        JsonElement root,
        LoadedArtifact artifact,
        AttackZoneScheduleEntry entry,
        string trialId,
        string runnerSha256)
    {
        RequireExactProperties(
            root,
            "event",
            "observed_utc",
            "schedule_sha256",
            "schedule_ordinal",
            "independent_run_id",
            "independent_run_ordinal",
            "trial_id",
            "move_index",
            "distance_bin_id",
            "bearing_bin_id",
            "runner_executable_sha256_observed",
            "expected_bridge_sha256",
            "attack_zone_trial_sha256",
            "global_input_used",
            "server_execution_claimed");
        RequireString(root, "event", "runner_trial_binding");
        RequireUtc(root, "observed_utc");
        RequireString(root, "schedule_sha256", artifact.ScheduleSha256);
        RequireInt32(root, "schedule_ordinal", entry.ScheduleOrdinal);
        RequireString(root, "independent_run_id", artifact.IndependentRunId);
        RequireInt32(root, "independent_run_ordinal", artifact.IndependentRunOrdinal);
        RequireString(root, "trial_id", trialId);
        RequireInt32(root, "move_index", entry.MoveIndex);
        RequireString(root, "distance_bin_id", entry.DistanceBin.Id);
        RequireString(root, "bearing_bin_id", entry.BearingBin.Id);
        RequireString(root, "runner_executable_sha256_observed", runnerSha256);
        RequireString(root, "expected_bridge_sha256", ExpectedBridgeSha256);
        RequireString(
            root,
            "attack_zone_trial_sha256",
            AttackZoneTrialContract.ExpectedSha256);
        RequireFalse(root, "global_input_used");
        RequireFalse(root, "server_execution_claimed");
    }

    private static AttackZoneTrialTarget? ValidateRecordedClientRequest(
        JsonElement root,
        LoadedArtifact artifact,
        AttackZoneScheduleEntry entry,
        string trialId)
    {
        RequireExactProperties(
            root,
            "event",
            "observed_utc",
            "request",
            "global_input_used");
        RequireString(root, "event", "client_request");
        RequireUtc(root, "observed_utc");
        RequireFalse(root, "global_input_used");
        var request = root.GetProperty("request");
        if (request.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("recorded client request was not an object");
        var type = RequiredString(request, "type");
        _ = RequiredString(request, "request_id");
        if (type == "get_state")
        {
            RequireExactProperties(request, "type", "request_id");
            return null;
        }
        if (type != "command")
            throw new InvalidDataException($"unexpected recorded request type {type}");
        var command = RequiredString(request, "command");
        if (command == "StartAttackZoneTrial")
        {
            RequireExactProperties(request, "type", "request_id", "command", "target");
            var targetElement = request.GetProperty("target");
            if (!AttackZoneTrialContract.TryParseTarget(
                    targetElement,
                    out var target,
                    out var parseReason))
            {
                throw new InvalidDataException(
                    $"recorded attack target was invalid: {parseReason}");
            }
            if (!AttackZoneTrialContract.TryValidateTarget(
                    target,
                    out _,
                    out var validationReason))
            {
                throw new InvalidDataException(
                    $"recorded attack target was invalid: {validationReason}");
            }
            var expected = AttackZoneTrialContract.CreateTarget(
                entry,
                artifact.ScheduleSha256,
                artifact.Seed,
                target.SessionIdentitySha256,
                target.RoundIdentitySha256,
                trialId,
                entry.ScheduleOrdinal + 1);
            if (!string.Equals(
                    AttackZoneTrialContract.SerializeTarget(expected),
                    targetElement.GetRawText(),
                    StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    "recorded attack target was not the exact canonical schedule target");
            }
            return target;
        }
        RequireExactProperties(request, "type", "request_id", "command");
        if (command is not (
            "AcquireExclusiveControl" or
            "ReleaseExclusiveControl" or
            "ConfirmLoggedIn" or
            "NavigateFreePlay" or
            "EnterSolo" or
            "StartRound" or
            "ExitLostPrivateSession" or
            "StartContinuousBotController" or
            "StopContinuousBotController"))
        {
            throw new InvalidDataException($"unexpected recorded command {command}");
        }
        return null;
    }

    private static void ValidateRunnerTrialResult(
        JsonElement root,
        LoadedArtifact artifact,
        AttackZoneScheduleEntry entry,
        string trialId,
        AttackZoneTrialTarget target,
        TrialEventValidator validator)
    {
        RequireExactProperties(
            root,
            "event",
            "result_schema",
            "observed_utc",
            "status",
            "terminal_event",
            "terminal_reason",
            "schedule_sha256",
            "schedule_ordinal",
            "trial_id",
            "session_identity_sha256",
            "round_identity_sha256",
            "event_counts",
            "action_sample_count",
            "recovery_upright_readiness_proven",
            "remainder_of_round_excluded",
            "global_input_used",
            "server_execution_claimed");
        RequireString(root, "event", "runner_trial_result");
        RequireString(root, "result_schema", TrialResultSchema);
        RequireUtc(root, "observed_utc");
        var expectedStatus = validator.TerminalEvent == "trial_completed"
            ? "completed"
            : validator.TerminalEvent == "trial_censored"
                ? "censored"
                : throw new InvalidDataException(
                    "completed transcript had a non-completable terminal event");
        RequireString(root, "status", expectedStatus);
        RequireString(
            root,
            "terminal_event",
            validator.TerminalEvent ??
                throw new InvalidDataException("runner result had no replayed terminal event"));
        RequireString(root, "terminal_reason", validator.TerminalReason ?? string.Empty);
        RequireString(root, "schedule_sha256", artifact.ScheduleSha256);
        RequireInt32(root, "schedule_ordinal", entry.ScheduleOrdinal);
        RequireString(root, "trial_id", trialId);
        RequireString(root, "session_identity_sha256", target.SessionIdentitySha256);
        RequireString(root, "round_identity_sha256", target.RoundIdentitySha256);
        RequireInt32(root, "action_sample_count", validator.ActionSampleCount);
        RequireBoolean(
            root,
            "recovery_upright_readiness_proven",
            validator.RecoveryReadinessProven);
        RequireBoolean(
            root,
            "remainder_of_round_excluded",
            validator.PostStartRoundExcluded);
        RequireFalse(root, "global_input_used");
        RequireFalse(root, "server_execution_claimed");
        var eventCounts = root.GetProperty("event_counts");
        if (eventCounts.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("runner result event counts were not an object");
        var observedNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in eventCounts.EnumerateObject())
        {
            if (!observedNames.Add(property.Name) ||
                property.Value.ValueKind != JsonValueKind.Number ||
                !property.Value.TryGetInt32(out var observedCount) ||
                !validator.EventCounts.TryGetValue(property.Name, out var expectedCount) ||
                observedCount != expectedCount)
            {
                throw new InvalidDataException("runner result event counts did not match replay");
            }
        }
        if (observedNames.Count != validator.EventCounts.Count)
            throw new InvalidDataException("runner result omitted replayed event counts");
    }

    private static string BuildExecutionManifestJson(
        LoadedArtifact artifact,
        string artifactPath,
        string runnerPath,
        string runnerSha256,
        string controllerContractSha256,
        string expectedBridgeVersion,
        string createdUtc) => JsonSerializer.Serialize(new
        {
            manifest_schema = ManifestSchema,
            schedule_artifact_path = Path.GetFullPath(artifactPath),
            schedule_sha256 = artifact.ScheduleSha256,
            independent_run_id = artifact.IndependentRunId,
            independent_run_ordinal = artifact.IndependentRunOrdinal,
            repetitions_per_cell = artifact.RepetitionsPerCell,
            schedule_entry_count = artifact.Entries.Count,
            attack_zone_trial_schema = AttackZoneTrialContract.Schema,
            attack_zone_trial_sha256 = AttackZoneTrialContract.ExpectedSha256,
            controller_contract_sha256 = controllerContractSha256,
            expected_bridge_version = expectedBridgeVersion,
            expected_bridge_sha256 = ExpectedBridgeSha256,
            expected_recorder_version = AttackZoneTrialContract.ExpectedRecorderVersion,
            expected_recorder_plugin_sha256 =
                AttackZoneTrialContract.ExpectedRecorderPluginSha256,
            required_isolation_proof = AttackZoneTrialContract.RequiredIsolationProof,
            runner_executable_path = Path.GetFullPath(runnerPath),
            runner_executable_sha256_observed = runnerSha256,
            global_input_used = false,
            server_execution_claimed = false,
            created_utc = createdUtc,
        }) + "\n";

    private static void ValidateExecutionManifest(
        string path,
        LoadedArtifact artifact,
        string artifactPath,
        string runnerPath,
        string runnerSha256)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("execution manifest was missing", path);
        ValidateExecutableFileBinding(runnerPath, runnerSha256);
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        RequireExactProperties(
            root,
            "manifest_schema",
            "schedule_artifact_path",
            "schedule_sha256",
            "independent_run_id",
            "independent_run_ordinal",
            "repetitions_per_cell",
            "schedule_entry_count",
            "attack_zone_trial_schema",
            "attack_zone_trial_sha256",
            "controller_contract_sha256",
            "expected_bridge_version",
            "expected_bridge_sha256",
            "expected_recorder_version",
            "expected_recorder_plugin_sha256",
            "required_isolation_proof",
            "runner_executable_path",
            "runner_executable_sha256_observed",
            "global_input_used",
            "server_execution_claimed",
            "created_utc");
        RequireString(root, "manifest_schema", ManifestSchema);
        RequireString(root, "schedule_artifact_path", Path.GetFullPath(artifactPath));
        RequireString(root, "schedule_sha256", artifact.ScheduleSha256);
        RequireString(root, "independent_run_id", artifact.IndependentRunId);
        RequireInt32(root, "independent_run_ordinal", artifact.IndependentRunOrdinal);
        RequireInt32(root, "repetitions_per_cell", artifact.RepetitionsPerCell);
        RequireInt32(root, "schedule_entry_count", artifact.Entries.Count);
        RequireString(root, "attack_zone_trial_schema", AttackZoneTrialContract.Schema);
        RequireString(root, "attack_zone_trial_sha256", AttackZoneTrialContract.ExpectedSha256);
        RequireString(
            root,
            "controller_contract_sha256",
            ContinuousBotControllerContract.ExpectedSha256);
        RequireString(root, "expected_bridge_version", ExpectedBridgeVersion);
        RequireString(root, "expected_bridge_sha256", ExpectedBridgeSha256);
        RequireString(
            root,
            "expected_recorder_version",
            AttackZoneTrialContract.ExpectedRecorderVersion);
        RequireString(
            root,
            "expected_recorder_plugin_sha256",
            AttackZoneTrialContract.ExpectedRecorderPluginSha256);
        RequireString(
            root,
            "required_isolation_proof",
            AttackZoneTrialContract.RequiredIsolationProof);
        RequireString(root, "runner_executable_path", Path.GetFullPath(runnerPath));
        RequireString(root, "runner_executable_sha256_observed", runnerSha256);
        RequireFalse(root, "global_input_used");
        RequireFalse(root, "server_execution_claimed");
        RequireUtc(root, "created_utc");
    }

    private static string TrialFileName(AttackZoneScheduleEntry entry) =>
        $"trial-{entry.ScheduleOrdinal:D6}-m{entry.MoveIndex:D2}-{entry.DistanceBin.Id}-{entry.BearingBin.Id}.jsonl";

    private static string TrialId(string scheduleSha256, int ordinal) =>
        $"az-{HashText($"{scheduleSha256}:{ordinal}")[..32]}";

    private static string HashText(string text) => Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();

    private static string NormalizeCurrentRunnerPath(string path)
    {
        var fullPath = Path.GetFullPath(path);
        if (!string.Equals(
                Path.GetFileName(fullPath),
                "RekAttackZoneRunner.exe",
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                "execute/resume requires the RekAttackZoneRunner.exe app host");
        }
        return fullPath;
    }

    private static void ValidateExecutableFileBinding(string path, string expectedSha256)
    {
        var fullPath = Path.GetFullPath(path);
        if (!string.Equals(path, fullPath, StringComparison.Ordinal))
            throw new InvalidDataException("runner executable path was not exact and normalized");
        if (!File.Exists(fullPath))
            throw new FileNotFoundException("runner executable was missing", fullPath);
        if ((File.GetAttributes(fullPath) & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("runner executable cannot be a reparse point");
        if (!AttackZoneTrialContract.ValidSha256(expectedSha256))
            throw new InvalidDataException("runner executable SHA-256 was invalid");
        if (!string.Equals(HashFile(fullPath), expectedSha256, StringComparison.Ordinal))
            throw new InvalidDataException("runner executable SHA-256 changed or mismatched");
    }

    private static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static void WriteNewText(string path, string content)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.NewLine = "\n";
        writer.Write(content);
        writer.Flush();
        stream.Flush(flushToDisk: true);
    }

    private static async Task FlushToDiskAsync(StreamWriter writer)
    {
        await writer.FlushAsync();
        if (writer.BaseStream is FileStream stream)
            stream.Flush(flushToDisk: true);
    }

    private static string RequiredString(JsonElement parent, string name)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind != JsonValueKind.String || string.IsNullOrEmpty(value.GetString()))
            throw new InvalidDataException($"expected nonempty string {name}");
        return value.GetString()!;
    }

    private static void RequireUtc(JsonElement parent, string name)
    {
        var value = RequiredString(parent, name);
        if (!DateTimeOffset.TryParse(value, out _))
            throw new InvalidDataException($"expected parseable UTC timestamp {name}");
    }

    private static void RequireExactProperties(
        JsonElement value,
        params string[] expectedProperties)
    {
        if (value.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("expected JSON object");
        var expected = new HashSet<string>(expectedProperties, StringComparer.Ordinal);
        if (expected.Count != expectedProperties.Length)
            throw new InvalidOperationException("duplicate expected JSON property name");
        var observed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in value.EnumerateObject())
        {
            if (!observed.Add(property.Name) || !expected.Contains(property.Name))
                throw new InvalidDataException($"unexpected or duplicate property {property.Name}");
        }
        if (!observed.SetEquals(expected))
            throw new InvalidDataException("required JSON property set was incomplete");
    }

    private static bool RequiredBoolean(JsonElement parent, string name)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new InvalidDataException($"expected boolean {name}");
        return value.GetBoolean();
    }

    private static bool RequiredNullableBoolean(JsonElement parent, string name)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new InvalidDataException($"expected nonnull boolean {name}");
        return value.GetBoolean();
    }

    private static string RequireSha256(JsonElement parent, string name)
    {
        var value = RequiredString(parent, name);
        if (!AttackZoneTrialContract.ValidSha256(value))
            throw new InvalidDataException($"expected lowercase SHA-256 {name}");
        return value;
    }

    private static void RequireString(JsonElement parent, string name, string expected)
    {
        if (!string.Equals(RequiredString(parent, name), expected, StringComparison.Ordinal))
            throw new InvalidDataException($"expected {name}={expected}");
    }

    private static void RequireInt32(JsonElement parent, string name, int expected)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind != JsonValueKind.Number ||
            !value.TryGetInt32(out var actual) || actual != expected)
        {
            throw new InvalidDataException($"expected {name}={expected}");
        }
    }

    private static void RequireInt64(JsonElement parent, string name, long expected)
    {
        var value = parent.GetProperty(name);
        if (value.ValueKind != JsonValueKind.Number ||
            !value.TryGetInt64(out var actual) || actual != expected)
        {
            throw new InvalidDataException($"expected {name}={expected}");
        }
    }

    private static void RequireBoolean(JsonElement parent, string name, bool expected)
    {
        if (RequiredBoolean(parent, name) != expected)
            throw new InvalidDataException($"expected {name}={expected}");
    }

    private static void RequireTrue(JsonElement parent, string name) =>
        RequireBoolean(parent, name, true);

    private static void RequireFalse(JsonElement parent, string name) =>
        RequireBoolean(parent, name, false);

    private static void RequireNull(JsonElement parent, string name)
    {
        if (!parent.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Null)
            throw new InvalidDataException($"expected null {name}");
    }

    private sealed record ScopePins(
        string SessionIdentitySha256,
        string RoundIdentitySha256);

    private sealed class TrialPipeSession : IAsyncDisposable
    {
        private readonly StreamWriter _transcript;
        private readonly TrialEventValidator _validator;
        private readonly NamedPipeClientStream _pipe = new(
            ".",
            PipeName,
            PipeDirection.InOut,
            PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
        private StreamReader? _reader;
        private StreamWriter? _writer;
        private int _requestSequence;

        internal long ConnectionId { get; private set; }

        internal TrialPipeSession(StreamWriter transcript, TrialEventValidator validator)
        {
            _transcript = transcript;
            _validator = validator;
        }

        internal async Task ConnectAsync(CancellationToken cancellationToken)
        {
            await _pipe.ConnectAsync(cancellationToken);
            _reader = new StreamReader(_pipe, Encoding.UTF8, false, 65_536, leaveOpen: true);
            _writer = new StreamWriter(_pipe, new UTF8Encoding(false), 65_536, leaveOpen: true)
            {
                AutoFlush = true,
                NewLine = "\n",
            };
        }

        internal async Task<JsonDocument> StateAsync(CancellationToken cancellationToken)
        {
            var requestId = NextRequestId("state");
            await SendAsync(new { type = "get_state", request_id = requestId });
            return await ReadUntilAsync("state", requestId, cancellationToken);
        }

        internal async Task<JsonDocument> CommandAsync(
            string command,
            CancellationToken cancellationToken)
        {
            var requestId = NextRequestId("command");
            await SendAsync(new { type = "command", request_id = requestId, command });
            return await ReadUntilAsync("ack", requestId, cancellationToken);
        }

        internal async Task<JsonDocument> AttackCommandAsync(
            AttackZoneTrialTarget target,
            CancellationToken cancellationToken)
        {
            var requestId = NextRequestId("attack");
            using var targetDocument = JsonDocument.Parse(
                AttackZoneTrialContract.SerializeTarget(target));
            await SendAsync(new
            {
                type = "command",
                request_id = requestId,
                command = "StartAttackZoneTrial",
                target = targetDocument.RootElement,
            });
            return await ReadUntilAsync("ack", requestId, cancellationToken);
        }

        internal async Task<JsonDocument> ReadUntilAsync(
            string eventName,
            string? requestId,
            CancellationToken cancellationToken)
        {
            for (;;)
            {
                var message = await ReadAsync(cancellationToken);
                var root = message.RootElement;
                if (OptionalString(root, "event") == "hello")
                {
                    ConnectionId = root.GetProperty("connection_id").GetInt64();
                    if (ConnectionId <= 0)
                    {
                        message.Dispose();
                        throw new InvalidDataException("hello connection ID was invalid");
                    }
                }
                if (OptionalString(root, "event") == "error" &&
                    OptionalString(root, "request_id") == requestId)
                {
                    var reason = OptionalString(root, "reason") ?? "unknown";
                    message.Dispose();
                    throw new InvalidDataException($"bridge request rejected: {reason}");
                }
                if (OptionalString(root, "event") == eventName &&
                    (requestId is null || OptionalString(root, "request_id") == requestId))
                {
                    return message;
                }
                message.Dispose();
            }
        }

        internal async Task<JsonDocument> ReadAsync(CancellationToken cancellationToken)
        {
            if (_reader is null)
                throw new InvalidOperationException("pipe reader was unavailable");
            var line = await _reader.ReadLineAsync(cancellationToken) ??
                throw new EndOfStreamException("bridge closed pipe");
            if (Encoding.UTF8.GetByteCount(line) > 4 * 1024 * 1024)
                throw new InvalidDataException("bridge message exceeded 4 MiB");
            await _transcript.WriteLineAsync(line);
            var document = JsonDocument.Parse(line);
            _validator.Observe(document.RootElement);
            return document;
        }

        private async Task SendAsync(object request)
        {
            if (_writer is null)
                throw new InvalidOperationException("pipe writer was unavailable");
            var line = JsonSerializer.Serialize(request);
            using var requestDocument = JsonDocument.Parse(line);
            await _transcript.WriteLineAsync(JsonSerializer.Serialize(new
            {
                @event = "client_request",
                observed_utc = DateTimeOffset.UtcNow.ToString("O"),
                request = requestDocument.RootElement,
                global_input_used = false,
            }));
            await _writer.WriteLineAsync(line);
        }

        private string NextRequestId(string kind) =>
            $"az-{kind}-{Environment.ProcessId}-{++_requestSequence}";

        public async ValueTask DisposeAsync()
        {
            if (_writer is not null)
                await _writer.DisposeAsync();
            _reader?.Dispose();
            await _pipe.DisposeAsync();
        }

        private static string? OptionalString(JsonElement parent, string name) =>
            parent.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
                ? value.GetString()
                : null;
    }

    private sealed class TrialEventValidator
    {
        private readonly LoadedArtifact _artifact;
        private readonly AttackZoneScheduleEntry _entry;
        private readonly string _trialId;
        private AttackZoneTrialTarget? _target;
        private AttackZoneClock? _lastActionSampleClock;
        private bool _targetRequested;
        private bool _targetAcquired;
        private bool _neutralEdge;
        private bool _neutralReturned;
        private bool _actionEdge;
        private bool _actionReturned;
        private bool _motionStarted;
        private bool _motionCompleted;

        internal Dictionary<string, int> EventCounts { get; } = new(StringComparer.Ordinal);
        internal string? TerminalEvent { get; private set; }
        internal string? TerminalReason { get; private set; }
        internal int ActionSampleCount { get; private set; }
        internal bool RecoveryContinuationObserved { get; private set; }
        internal bool RecoveryReadinessProven { get; private set; }
        internal bool PostStartRoundExcluded { get; private set; }

        internal TrialEventValidator(
            LoadedArtifact artifact,
            AttackZoneScheduleEntry entry,
            string trialId)
        {
            _artifact = artifact;
            _entry = entry;
            _trialId = trialId;
        }

        internal void BindTarget(AttackZoneTrialTarget target) => _target = target;

        internal void Observe(JsonElement value)
        {
            if (!value.TryGetProperty("attack_zone_schema", out var schema) ||
                schema.ValueKind != JsonValueKind.String)
            {
                return;
            }
            RequireString(value, "attack_zone_schema", AttackZoneTrialContract.Schema);
            RequireString(
                value,
                "attack_zone_protocol_sha256",
                AttackZoneTrialContract.ExpectedSha256);
            RequireString(
                value,
                "continuous_controller_sha256",
                ContinuousBotControllerContract.ExpectedSha256);
            RequireString(value, "authority_scope", AttackZoneTrialContract.AuthorityScope);
            RequireString(value, "authority_caveat", AttackZoneTrialContract.AuthorityCaveat);
            RequireString(
                value,
                "isolated_spark_proof",
                AttackZoneTrialContract.RequiredIsolationProof);
            RequireString(value, "schedule_sha256", _artifact.ScheduleSha256);
            RequireInt32(value, "schedule_ordinal", _entry.ScheduleOrdinal);
            RequireString(value, "independent_run_id", _artifact.IndependentRunId);
            RequireString(value, "trial_id", _trialId);
            RequireInt32(value, "move_index", _entry.MoveIndex);
            RequireString(value, "serialized_asset_sha256", _entry.SerializedAssetSha256);
            RequireFalse(value, "global_input_used");
            RequireFalse(value, "server_acceptance_observed");
            RequireFalse(value, "authoritative_execution_observed");
            RequireTrue(value, "client_request_observation_only");
            RequireInt32(
                value,
                "fixed_substeps_per_control_tick",
                AttackZoneTrialContract.FixedSubstepsPerControlTick);
            if (_target is null)
                throw new InvalidDataException("attack event arrived before target binding");
            RequireString(value, "session_identity_sha256", _target.SessionIdentitySha256);
            RequireString(value, "round_identity_sha256", _target.RoundIdentitySha256);
            var clock = ReadClock(value);
            if (!AttackZoneTrialContract.ValidateClock(clock))
                throw new InvalidDataException("attack event clock was invalid");

            var eventName = RequiredString(value, "event");
            EventCounts[eventName] = EventCounts.TryGetValue(eventName, out var count)
                ? count + 1
                : 1;
            switch (eventName)
            {
                case "target_requested":
                    _targetRequested = true;
                    break;
                case "neutral_command_edge_set":
                    _neutralEdge = true;
                    break;
                case "neutral_request_method_returned":
                    _neutralReturned = true;
                    break;
                case "target_acquired":
                {
                    var digest = value.GetProperty("detail").GetProperty("settle_digest");
                    RequireInt32(digest, "SampleCount", AttackZoneTrialContract.SettleTicks);
                    _targetAcquired = true;
                    break;
                }
                case "local_command_edge_set":
                    _actionEdge = true;
                    break;
                case "client_request_method_returned":
                    _actionReturned = true;
                    break;
                case "local_motion_start_observed":
                    _motionStarted = true;
                    break;
                case "action_sample":
                    if (_lastActionSampleClock is not null &&
                        !AttackZoneTrialContract.ClocksAreConsecutive(
                            _lastActionSampleClock,
                            clock))
                    {
                        throw new InvalidDataException(
                            "action sample transcript cadence was not consecutive measured 50 Hz");
                    }
                    _lastActionSampleClock = clock;
                    ActionSampleCount++;
                    RequireInt32(
                        value.GetProperty("detail"),
                        "action_sample_sequence",
                        ActionSampleCount);
                    break;
                case "local_motion_completion_and_readiness_observed":
                    _motionCompleted = true;
                    break;
                case "recovery_state_observed":
                {
                    var reason = RequiredString(value, "controller_reason");
                    if (reason ==
                        "terminal_trial_persisted_before_local_recovery_continuation")
                    {
                        RecoveryContinuationObserved = true;
                    }
                    if (reason ==
                        "local_upright_readiness_proven_after_censored_trial")
                    {
                        RecoveryReadinessProven = true;
                    }
                    break;
                }
                case "trial_completed":
                case "trial_censored":
                case "trial_interrupted":
                    if (TerminalEvent is not null)
                        throw new InvalidDataException("duplicate attack trial terminal event");
                    TerminalEvent = eventName;
                    TerminalReason = RequiredString(value, "controller_reason");
                    PostStartRoundExcluded = TerminalReason.Contains(
                        "post_start_fall_or_recovery_censored_remainder_of_round",
                        StringComparison.Ordinal);
                    break;
            }
        }

        private static AttackZoneClock ReadClock(JsonElement value) => new(
            value.GetProperty("stopwatch_timestamp_ticks").GetInt64(),
            value.GetProperty("stopwatch_frequency_hz").GetInt64(),
            RequiredString(value, "utc"),
            value.GetProperty("unity_frame").GetInt32(),
            value.GetProperty("unity_time").GetDouble(),
            value.GetProperty("unity_fixed_time").GetDouble(),
            value.GetProperty("client_control_tick").GetInt32(),
            value.GetProperty("client_fixed_substep").GetInt32());

        internal void ValidateTerminalEvidence()
        {
            if (TerminalEvent is null)
                throw new InvalidDataException("trial terminal event was not observed");
            if (!_targetRequested)
                throw new InvalidDataException("target request evidence was incomplete");
            if (TerminalEvent == "trial_completed" &&
                (!_neutralEdge || !_neutralReturned || !_targetAcquired || !_actionEdge ||
                 !_actionReturned || !_motionStarted ||
                 !_motionCompleted || ActionSampleCount == 0))
            {
                throw new InvalidDataException("completed trial lifecycle evidence was incomplete");
            }
        }
    }
}
