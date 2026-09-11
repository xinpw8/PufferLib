using System.Text.Json;
using RekUiBridgeAgent;

var tests = new (string Name, Action Run)[]
{
    ("canonical contract and exact bins", TestCanonicalContractAndBins),
    ("strict target wire parser", TestStrictTargetWireParser),
    ("scope rejects everything outside exact isolated target", TestScopeValidation),
    ("acquisition may target nonzero bearing", TestTargetBearingAcquisition),
    ("acquisition bearing commands converge from both sides", TestTargetBearingConvergence),
    ("fifteen consecutive 50 Hz settle samples", TestFifteenConsecutiveSettleSamples),
    ("settle streak resets on clock gap and bad predicate", TestSettleReset),
    ("settle rejects dampened opponent", TestDampenedOpponent),
    ("settle clock proves measured 50 Hz cadence", TestMeasuredClockCadence),
    ("animation and quaternion contradictions fail closed", TestObservationValidation),
    ("action sampling is required at consecutive measured cadence", TestActionSamplingCadence),
    ("censored local falls continue recovery without audit attacks", TestRecoveryContinuation),
    ("opponent motion remains a named secondary stratum", TestOpponentMotionStratum),
    ("randomized schedules are deterministic permutations", TestRandomizedSchedule),
    ("five independent runs are required per cell", TestIndependentCoverage),
};

var failures = new List<string>();
foreach (var test in tests)
{
    try
    {
        test.Run();
        Console.WriteLine($"PASS {test.Name}");
    }
    catch (Exception exception)
    {
        failures.Add($"FAIL {test.Name}: {exception.Message}");
    }
}

Console.WriteLine($"attack_zone_contract_sha256={AttackZoneTrialContract.ComputeSha256()}");
if (failures.Count > 0)
{
    foreach (var failure in failures)
        Console.Error.WriteLine(failure);
    return 1;
}
Console.WriteLine($"PASS all {tests.Length} attack-zone contract tests");
return 0;

static void TestCanonicalContractAndBins()
{
    Equal(AttackZoneTrialContract.ExpectedSha256,
        AttackZoneTrialContract.ComputeSha256(), "canonical SHA-256");
    True(AttackZoneTrialContract.ValidateEmbeddedContract(out var reason), reason);
    Equal(9, AttackZoneTrialContract.DistanceBins.Length, "distance bin count");
    Equal(11, AttackZoneTrialContract.BearingBins.Length, "bearing bin count");
    var d02 = AttackZoneTrialContract.DistanceBins.Single(value => value.Id == "d02");
    var d03 = AttackZoneTrialContract.DistanceBins.Single(value => value.Id == "d03");
    True(d02.Contains(0.5180000126361847), "d02 must include pinned boundary");
    False(d03.Contains(0.5180000126361847), "d03 must exclude pinned boundary");
    var b05 = AttackZoneTrialContract.BearingBins.Single(value => value.Id == "b05");
    var b06 = AttackZoneTrialContract.BearingBins.Single(value => value.Id == "b06");
    True(b05.Contains(5.0), "b05 must include +5 degrees");
    False(b06.Contains(5.0), "b06 must exclude +5 degrees");
    Equal(1, AttackZoneTrialContract.TelemetryIntervalTicks, "50 Hz telemetry interval");
    True(AttackZoneTrialContract.RequiredEvidenceEvents.Contains("raw_rek_hit_observed"),
        "raw REK_Hit event required");
    True(AttackZoneTrialContract.RequiredEvidenceEvents.Contains("recovery_state_observed"),
        "recovery observation required");
    True(AttackZoneTrialContract.RequiredEvidenceEvents.Contains("action_sample"),
        "50 Hz action sample event required");
}

static void TestStrictTargetWireParser()
{
    var setup = TargetSetup("wire-run", 0, Seed('1'), 7);
    var wire = AttackZoneTrialContract.SerializeTarget(setup.Target);
    using var document = JsonDocument.Parse(wire);
    True(AttackZoneTrialContract.TryParseTarget(
        document.RootElement, out var parsed, out var parseReason), parseReason);
    True(AttackZoneTrialContract.TryValidateTarget(
        parsed, out var validated, out var validationReason), validationReason);
    Equal(setup.Target.MoveIndex, validated.Attack.MoveIndex, "parsed move index");
    Equal(setup.Entry.DistanceBin.Id, validated.DistanceBin.Id, "parsed distance bin");
    Equal(setup.Entry.BearingBin.Id, validated.BearingBin.Id, "parsed bearing bin");

    var wrongTimeout = wire.Replace(
        "\"acquisition_timeout_ticks\":500",
        "\"acquisition_timeout_ticks\":499",
        StringComparison.Ordinal);
    using var wrongDocument = JsonDocument.Parse(wrongTimeout);
    True(AttackZoneTrialContract.TryParseTarget(
        wrongDocument.RootElement, out var wrong, out _), "timeout shape still parses");
    False(AttackZoneTrialContract.TryValidateTarget(
        wrong, out _, out var wrongReason), "wrong timeout must fail closed");
    Equal("attack_zone_target_acquisition_timeout_mismatch", wrongReason,
        "wrong timeout reason");

    var unknownProperty = wire[..^1] + ",\"unexpected\":true}";
    using var unknownDocument = JsonDocument.Parse(unknownProperty);
    False(AttackZoneTrialContract.TryParseTarget(
        unknownDocument.RootElement, out _, out var unknownReason),
        "unknown target property must fail");
    Equal("invalid_attack_zone_target_shape", unknownReason, "unknown property reason");
}

static void TestScopeValidation()
{
    var setup = TargetSetup("scope-run", 0, Seed('2'), 0);
    True(AttackZoneTrialContract.TryValidateTarget(
        setup.Target, out var target, out var reason), reason);
    var scope = ValidScope(setup.Target) with
    {
        OpponentSemanticRuntimeMismatch = true,
    };
    var accepted = AttackZoneTrialContract.ValidateScope(scope, target);
    True(accepted.Accepted, accepted.Reason);
    Equal("scope_proven_opponent_semantic_runtime_mismatch_recorded",
        accepted.Reason, "semantic mismatch must be recorded, not used as runtime identity");

    var nativeWindows = AttackZoneTrialContract.ValidateScope(scope with
    {
        IsolatedSparkVerified = false,
        IsolationProof = null,
    }, target);
    False(nativeWindows.Accepted, "native Windows must be rejected");
    Equal("verified_isolated_spark_scope_required", nativeWindows.Reason,
        "isolation rejection reason");

    var wrongOpponent = AttackZoneTrialContract.ValidateScope(scope with
    {
        OpponentRuntimeExactT800 = false,
    }, target);
    False(wrongOpponent.Accepted, "runtime-mismatched opponent must be rejected");
    var globalInput = AttackZoneTrialContract.ValidateScope(scope with
    {
        GlobalInputUsed = true,
    }, target);
    False(globalInput.Accepted, "global input must be rejected");
}

static void TestTargetBearingAcquisition()
{
    var setup = TargetCellSetup("bearing-run", 0, Seed('3'), 2, "d01", "b07");
    True(AttackZoneTrialContract.TryValidateTarget(
        setup.Target, out var target, out var reason), reason);
    var faceOpponent = new PlanarCombatGeometry(
        (float)target.DistanceBin.Center,
        0f,
        0f,
        0f,
        0f,
        0f,
        1f);
    var deliberateBearing = AttackZoneTrialContract.DecideAcquisition(target, faceOpponent);
    True(deliberateBearing.Yaw > 0f,
        "positive target bearing must turn away from zero bearing during acquisition");
    False(deliberateBearing.ExactNeutral, "bearing acquisition must remain active");

    var onTarget = faceOpponent with
    {
        LocalBearingToOpponentDegrees = (float)target.BearingBin.Center,
    };
    var neutral = AttackZoneTrialContract.DecideAcquisition(target, onTarget);
    True(neutral.ExactNeutral, "target center must become exact neutral");
    Equal(0f, neutral.Forward, "neutral forward");
    Equal(0f, neutral.Strafe, "neutral strafe");
    Equal(0f, neutral.Yaw, "neutral yaw");
}

static void TestTargetBearingConvergence()
{
    var positiveSetup = TargetCellSetup("bearing-convergence-positive", 0, Seed('8'), 2, "d01", "b07");
    True(
        AttackZoneTrialContract.TryValidateTarget(
            positiveSetup.Target,
            out var positiveTarget,
            out var positiveReason),
        positiveReason);
    var fromBelow = AttackZoneTrialContract.DecideAcquisition(
        positiveTarget,
        Geometry(distanceMeters: 0.60, bearingDegrees: 0.0));
    True(fromBelow.Forward > 0f, "mixed acquisition must approach when too far");
    True(fromBelow.Yaw > 0f, "bearing below a positive target requires positive yaw");

    var negativeSetup = TargetCellSetup("bearing-convergence-negative", 0, Seed('9'), 2, "d01", "b03");
    True(
        AttackZoneTrialContract.TryValidateTarget(
            negativeSetup.Target,
            out var negativeTarget,
            out var negativeReason),
        negativeReason);
    var fromAbove = AttackZoneTrialContract.DecideAcquisition(
        negativeTarget,
        Geometry(distanceMeters: 0.20, bearingDegrees: 0.0));
    True(fromAbove.Forward < 0f, "mixed acquisition must back off when too close");
    True(fromAbove.Yaw < 0f, "bearing above a negative target requires negative yaw");

    var overshotPositive = AttackZoneTrialContract.DecideAcquisition(
        positiveTarget,
        Geometry(distanceMeters: positiveTarget.DistanceBin.Center, bearingDegrees: 50.0));
    True(overshotPositive.Yaw < 0f, "positive bearing overshoot requires negative yaw");
}

static void TestFifteenConsecutiveSettleSamples()
{
    var setup = TargetCellSetup("settle-run", 0, Seed('4'), 4, "d01", "b05");
    True(AttackZoneTrialContract.TryValidateTarget(
        setup.Target, out var target, out var reason), reason);
    var tracker = new AttackZoneSettleTracker(target);
    for (var tick = 0; tick < AttackZoneTrialContract.SettleTicks; tick++)
    {
        var update = tracker.Observe(Sample(tick, target.DistanceBin.Center, 0.0));
        Equal(tick + 1, update.ConsecutiveTicks, "settle count");
        Equal(tick == AttackZoneTrialContract.SettleTicks - 1,
            update.Acquired, "acquisition edge");
    }
    True(tracker.Acquired, "15 samples must acquire");
    var digest = tracker.AcquiredDigest ?? throw new Exception("missing acquired digest");
    Equal(15, digest.SampleCount, "digest sample count");
    Equal(14, digest.LastClock.ControlTick, "digest last control tick");
    True(digest.AllDistanceCentralPass, "distance predicate digest");
    True(digest.AllBearingErrorPass, "bearing predicate digest");
    True(digest.AllOpponentStationary, "stationary digest");
    Equal("stationary", digest.OpponentMotionStratum, "stationary stratum");
}

static void TestSettleReset()
{
    var setup = TargetCellSetup("reset-run", 0, Seed('5'), 5, "d01", "b05");
    True(AttackZoneTrialContract.TryValidateTarget(
        setup.Target, out var target, out var reason), reason);
    var tracker = new AttackZoneSettleTracker(target);
    tracker.Observe(Sample(0, target.DistanceBin.Center, 0.0));
    tracker.Observe(Sample(1, target.DistanceBin.Center, 0.0));
    var gap = tracker.Observe(Sample(3, target.DistanceBin.Center, 0.0));
    True(gap.StreakReset, "clock gap must reset streak");
    Equal(1, gap.ConsecutiveTicks, "gap sample starts a new streak");
    var pending = tracker.Observe(Sample(4, target.DistanceBin.Center, 0.0) with
    {
        PendingMove = true,
    });
    True(pending.StreakReset, "pending command must reset streak");
    Equal(0, pending.ConsecutiveTicks, "bad predicate clears streak");
    Equal("pending_request_observed", pending.Reason, "pending reason");
    False(tracker.AcquisitionTimedOut(499), "timeout one tick early");
    True(tracker.AcquisitionTimedOut(500), "timeout at exact bound");
}

static void TestOpponentMotionStratum()
{
    var setup = TargetCellSetup("motion-run", 0, Seed('6'), 9, "d01", "b05");
    True(AttackZoneTrialContract.TryValidateTarget(
        setup.Target, out var target, out var reason), reason);
    var tracker = new AttackZoneSettleTracker(target);
    for (var tick = 0; tick < 15; tick++)
    {
        var sample = Sample(tick, target.DistanceBin.Center, 0.0) with
        {
            OpponentRoot = Root(
                0.0,
                target.DistanceBin.Center,
                linearZ: -0.20),
        };
        var update = tracker.Observe(sample);
        if (tick == 14)
            True(update.Acquired, "secondary motion stratum remains an acquired trial");
    }
    var digest = tracker.AcquiredDigest ?? throw new Exception("missing moving digest");
    False(digest.AllOpponentStationary, "moving opponent is excluded from primary stratum");
    Equal("closing", digest.OpponentMotionStratum, "closing stratum");
}

static void TestDampenedOpponent()
{
    var setup = TargetCellSetup("dampened-run", 0, Seed('a'), 9, "d01", "b05");
    True(AttackZoneTrialContract.TryValidateTarget(
        setup.Target, out var target, out var reason), reason);
    var sample = Sample(0, target.DistanceBin.Center, 0.0);
    var evaluation = AttackZoneTrialContract.EvaluateSettleSample(target, sample with
    {
        OpponentRoot = sample.OpponentRoot with { Dampened = true },
    });
    False(evaluation.OpponentHealthy, "dampened opponent must be unhealthy");
    False(evaluation.AcquisitionPass, "dampened opponent must reject settle sample");
}

static void TestMeasuredClockCadence()
{
    var first = Sample(0, 0.4, 0.0).Clock;
    var valid = Sample(1, 0.4, 0.0).Clock;
    True(AttackZoneTrialContract.ClocksAreConsecutive(first, valid),
        "20 ms measured interval must be consecutive");
    False(AttackZoneTrialContract.ClocksAreConsecutive(first, valid with
    {
        StopwatchFrequencyHz = first.StopwatchFrequencyHz + 1,
    }), "frequency change must fail");
    False(AttackZoneTrialContract.ClocksAreConsecutive(first, valid with
    {
        StopwatchTimestampTicks = first.StopwatchTimestampTicks + 5_000,
    }), "5 ms monotonic interval must fail");
    False(AttackZoneTrialContract.ClocksAreConsecutive(first, valid with
    {
        UnityFixedTime = first.UnityFixedTime + 0.019,
    }), "non-20 ms Unity fixed interval must fail");
}

static void TestObservationValidation()
{
    False(new AttackZoneAnimationObservation(true, null, null, null).IsValid,
        "playing action without clip/frame/fps must fail");
    False((Root(0.0, 0.0) with { RotationW = 2.0 }).IsFinite,
        "non-unit quaternion must fail");
    True((Root(0.0, 0.0) with { RotationW = Math.Sqrt(1.0005) }).IsFinite,
        "quaternion inside norm tolerance must pass");
}

static void TestActionSamplingCadence()
{
    var clocks = Enumerable.Range(0, 8)
        .Select(tick => Sample(tick, 0.4, 0.0).Clock)
        .ToArray();
    True(clocks.Zip(clocks.Skip(1), AttackZoneTrialContract.ClocksAreConsecutive)
        .All(value => value), "every action sample must be a measured consecutive 50 Hz tick");
    var withGap = clocks.ToArray();
    withGap[4] = Sample(5, 0.4, 0.0).Clock;
    False(withGap.Zip(withGap.Skip(1), AttackZoneTrialContract.ClocksAreConsecutive)
        .All(value => value), "a missing action sample tick must fail closed");
}

static void TestRecoveryContinuation()
{
    Equal(
        AttackZoneCensorDisposition.ContinueLocalRecovery,
        AttackZoneTrialContract.ClassifyCensorDisposition(
            true, false, false, false, false, false, false, false, false),
        "local falling must retain a recovery-only controller");
    Equal(
        AttackZoneCensorDisposition.OpponentOnly,
        AttackZoneTrialContract.ClassifyCensorDisposition(
            false, false, false, false, false, false, false, false, true),
        "opponent-only fall must not issue local recovery");
    Equal(
        ContinuousRecoveryCommand.Dampen,
        ContinuousBotControllerContract.SelectRecoveryCommand(
            fallen: true,
            dampened: false,
            recoveryArmed: false,
            motorShutdown: false,
            straightenIssued: false,
            suggestedProne: true),
        "fallen local must dampen first");
    Equal(
        ContinuousRecoveryCommand.GetUpProne,
        ContinuousBotControllerContract.SelectRecoveryCommand(
            fallen: true,
            dampened: true,
            recoveryArmed: true,
            motorShutdown: false,
            straightenIssued: true,
            suggestedProne: true),
        "prone recovery must use the prone get-up command");
    Equal(
        ContinuousRecoveryCommand.GetUpSupine,
        ContinuousBotControllerContract.SelectRecoveryCommand(
            fallen: true,
            dampened: true,
            recoveryArmed: true,
            motorShutdown: false,
            straightenIssued: true,
            suggestedProne: false),
        "supine recovery must use the supine get-up command");
    Equal(
        ContinuousRecoveryCommand.None,
        ContinuousBotControllerContract.SelectRecoveryCommand(
            fallen: true,
            dampened: true,
            recoveryArmed: true,
            motorShutdown: true,
            straightenIssued: true,
            suggestedProne: true),
        "motor shutdown must not issue normal recovery before the fault EStop cycle");
    Equal(
        "recovery_request_observed",
        AttackZoneTrialContract.MapRecoveryLifecycleEventName(
            "local_special_command_edge_set"),
        "special recovery edge event mapping");
    Equal(
        "recovery_request_observed",
        AttackZoneTrialContract.MapRecoveryLifecycleEventName(
            "local_estop_toggle_edge_set"),
        "fault EStop recovery edge event mapping");
    Equal(
        "recovery_request_observed",
        AttackZoneTrialContract.MapRecoveryLifecycleEventName(
            "client_request_method_returned"),
        "recovery request return event mapping");
    Equal(
        "Dampen|Straighten|GetUpProne|GetUpSupine|fault_estop_toggle_on|fault_estop_toggle_off",
        string.Join('|', AttackZoneTrialContract.RequiredRecoveryRequestKinds),
        "all normal and fault recovery requests must remain observable");
    Equal(15, AttackZoneTrialContract.RecoveryReadyTicks,
        "recovery-only completion requires 15 upright readiness ticks");
}

static void TestRandomizedSchedule()
{
    var first = AttackZoneTrialContract.BuildRandomizedSchedule(
        "random-run", 0, Seed('7'));
    var repeated = AttackZoneTrialContract.BuildRandomizedSchedule(
        "random-run", 0, Seed('7'));
    var other = AttackZoneTrialContract.BuildRandomizedSchedule(
        "random-run", 0, Seed('8'));
    Equal(6 * 9 * 11, first.Count, "full factorial cell count");
    Equal(Signatures(first), Signatures(repeated), "same seed must reproduce order");
    NotEqual(Signatures(first), Signatures(other), "different seed should alter order");
    Equal(first.Count, first.Select(CellSignature).Distinct(StringComparer.Ordinal).Count(),
        "schedule must be a permutation of all move cells");
    True(first.Select((entry, index) => entry.ScheduleOrdinal == index).All(value => value),
        "schedule ordinals must match randomized order");
    var hash1 = AttackZoneTrialContract.ComputeScheduleSha256(
        "random-run", 0, Seed('7'), 1, first);
    var hash2 = AttackZoneTrialContract.ComputeScheduleSha256(
        "random-run", 0, Seed('7'), 1, repeated);
    Equal(hash1, hash2, "deterministic schedule SHA-256");
}

static void TestIndependentCoverage()
{
    var oneRun = AttackZoneTrialContract.BuildRandomizedSchedule(
        "coverage-run-0", 0, Seed('9'));
    var insufficient = AttackZoneTrialContract.ValidateIndependentCoverage(oneRun);
    False(insufficient.Complete, "one run cannot satisfy five-run coverage");
    Equal(6 * 9 * 11, insufficient.MissingCells.Count, "all cells missing run coverage");

    var combined = new List<AttackZoneScheduleEntry>();
    for (var run = 0; run < AttackZoneTrialContract.MinimumIndependentRunsPerCell; run++)
    {
        combined.AddRange(AttackZoneTrialContract.BuildRandomizedSchedule(
            $"coverage-run-{run}", run, run.ToString("x64")));
    }
    var complete = AttackZoneTrialContract.ValidateIndependentCoverage(combined);
    True(complete.Complete, "five distinct run identities must satisfy coverage");
    Equal(0, complete.MissingCells.Count, "complete coverage missing list");
}

static (AttackZoneScheduleEntry Entry, AttackZoneTrialTarget Target) TargetSetup(
    string runId,
    int runOrdinal,
    string seed,
    int scheduleOrdinal)
{
    var schedule = AttackZoneTrialContract.BuildRandomizedSchedule(runId, runOrdinal, seed);
    var entry = schedule[scheduleOrdinal];
    var scheduleSha = AttackZoneTrialContract.ComputeScheduleSha256(
        runId, runOrdinal, seed, 1, schedule);
    var target = AttackZoneTrialContract.CreateTarget(
        entry,
        scheduleSha,
        seed,
        new string('a', 64),
        new string('b', 64),
        $"trial-{scheduleOrdinal}",
        scheduleOrdinal + 1);
    return (entry, target);
}

static (AttackZoneScheduleEntry Entry, AttackZoneTrialTarget Target) TargetCellSetup(
    string runId,
    int runOrdinal,
    string seed,
    int moveIndex,
    string distanceId,
    string bearingId)
{
    var schedule = AttackZoneTrialContract.BuildRandomizedSchedule(runId, runOrdinal, seed);
    var entry = schedule.Single(value => value.MoveIndex == moveIndex &&
        value.DistanceBin.Id == distanceId && value.BearingBin.Id == bearingId);
    return TargetSetup(runId, runOrdinal, seed, entry.ScheduleOrdinal);
}

static AttackZoneScopeObservation ValidScope(AttackZoneTrialTarget target) => new(
    true,
    AttackZoneTrialContract.RequiredIsolationProof,
    true,
    7,
    false,
    true,
    true,
    false,
    true,
    true,
    2,
    true,
    true,
    true,
    new string('c', 64),
    false,
    true,
    true,
    true,
    true,
    target.SessionIdentitySha256,
    target.RoundIdentitySha256);

static AttackZoneControlObservation Sample(
    int tick,
    double distance,
    double bearingDegrees)
{
    var radians = bearingDegrees * Math.PI / 180.0;
    var opponentX = Math.Sin(radians) * distance;
    var opponentZ = Math.Cos(radians) * distance;
    return new AttackZoneControlObservation(
        new AttackZoneClock(
            1_000_000 + tick * 20_000,
            1_000_000,
            DateTimeOffset.UnixEpoch.AddMilliseconds(tick * 20).ToString("o"),
            100 + tick,
            tick * 0.02,
            tick * 0.02,
            tick,
            tick * AttackZoneTrialContract.FixedSubstepsPerControlTick),
        Root(0.0, 0.0),
        Root(opponentX, opponentZ),
        new AttackZoneAnimationObservation(false, null, null, null),
        new AttackZoneAnimationObservation(false, null, null, null),
        true,
        true,
        true,
        false,
        false,
        false);
}

static AttackZoneRootObservation Root(
    double x,
    double z,
    double linearX = 0.0,
    double linearZ = 0.0,
    double angularY = 0.0) => new(
        x, 0.0, z,
        0.0, 0.0, 0.0, 1.0,
        linearX, 0.0, linearZ,
        0.0, angularY, 0.0,
        false, false, false, false, false, false);

static PlanarCombatGeometry Geometry(double distanceMeters, double bearingDegrees)
{
    var radians = bearingDegrees * Math.PI / 180.0;
    return new PlanarCombatGeometry(
        (float)distanceMeters,
        (float)bearingDegrees,
        0f,
        0f,
        0f,
        (float)Math.Sin(radians),
        (float)Math.Cos(radians));
}

static string Seed(char value) => new(value, 64);

static string CellSignature(AttackZoneScheduleEntry entry) =>
    $"{entry.MoveIndex}:{entry.DistanceBin.Id}:{entry.BearingBin.Id}";

static string Signatures(IEnumerable<AttackZoneScheduleEntry> entries) =>
    string.Join('|', entries.Select(CellSignature));

static void True(bool value, string message)
{
    if (!value)
        throw new Exception(message);
}

static void False(bool value, string message) => True(!value, message);

static void Equal<T>(T expected, T actual, string message)
{
    if (!EqualityComparer<T>.Default.Equals(expected, actual))
        throw new Exception($"{message}: expected={expected}, actual={actual}");
}

static void NotEqual<T>(T left, T right, string message)
{
    if (EqualityComparer<T>.Default.Equals(left, right))
        throw new Exception($"{message}: both={left}");
}
