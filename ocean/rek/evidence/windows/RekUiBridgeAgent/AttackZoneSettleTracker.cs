namespace RekUiBridgeAgent;

internal sealed class AttackZoneSettleTracker
{
    private readonly AttackZoneValidatedTarget _target;
    private readonly List<AttackZoneSettleEvidence> _streak = new();
    private AttackZoneSettleDigest? _acquiredDigest;

    internal AttackZoneSettleTracker(AttackZoneValidatedTarget target)
    {
        _target = target ?? throw new ArgumentNullException(nameof(target));
    }

    internal int ConsecutiveTicks => _streak.Count;

    internal bool Acquired => _acquiredDigest is not null;

    internal IReadOnlyList<AttackZoneSettleEvidence> Evidence => _streak;

    internal AttackZoneSettleDigest? AcquiredDigest => _acquiredDigest;

    internal AttackZoneSettleUpdate Observe(AttackZoneControlObservation sample)
    {
        if (_acquiredDigest is not null)
        {
            return new AttackZoneSettleUpdate(
                true,
                _streak.Count,
                false,
                "target_already_acquired",
                _acquiredDigest,
                null);
        }

        var evaluation = AttackZoneTrialContract.EvaluateSettleSample(_target, sample);
        var evidence = new AttackZoneSettleEvidence(sample, evaluation);
        if (!evaluation.AcquisitionPass)
        {
            _streak.Clear();
            return new AttackZoneSettleUpdate(
                false,
                0,
                true,
                FirstFailureReason(evaluation),
                null,
                evidence);
        }

        var reset = false;
        var resetReason = "settle_sample_accepted";
        if (_streak.Count > 0 && !AttackZoneTrialContract.ClocksAreConsecutive(
                _streak[^1].Sample.Clock,
                sample.Clock))
        {
            _streak.Clear();
            reset = true;
            resetReason = "nonconsecutive_50hz_settle_clock";
        }
        _streak.Add(evidence);
        if (_streak.Count == AttackZoneTrialContract.SettleTicks)
        {
            _acquiredDigest = BuildDigest(_streak);
            return new AttackZoneSettleUpdate(
                true,
                _streak.Count,
                reset,
                "target_acquired_after_15_consecutive_50hz_ticks",
                _acquiredDigest,
                evidence);
        }
        return new AttackZoneSettleUpdate(
            false,
            _streak.Count,
            reset,
            resetReason,
            BuildDigest(_streak),
            evidence);
    }

    internal bool AcquisitionTimedOut(int elapsedControlTicks) =>
        !Acquired && elapsedControlTicks >= AttackZoneTrialContract.AcquisitionTimeoutTicks;

    internal void Reset()
    {
        _streak.Clear();
        _acquiredDigest = null;
    }

    private static string FirstFailureReason(AttackZoneSampleEvaluation value)
    {
        if (!value.ClockValid)
            return "settle_clock_invalid";
        if (!value.RootsFinite)
            return "settle_root_observation_invalid";
        if (!value.GeometryValid)
            return "settle_geometry_invalid";
        if (!value.AnimationValid)
            return "settle_animation_observation_invalid";
        if (!value.NeutralRequestMethodReturned)
            return "neutral_request_method_return_not_observed";
        if (!value.VelocityCommandExactNeutral)
            return "velocity_command_not_exact_neutral";
        if (!value.LocalActionReady)
            return "local_action_not_ready";
        if (!value.NoPendingRequests)
            return "pending_request_observed";
        if (!value.LocalHealthy)
            return "local_fall_or_recovery_contamination";
        if (!value.OpponentHealthy)
            return "opponent_fall_or_recovery_contamination";
        if (!value.DistanceCentralPass)
            return "distance_outside_target_central_50_percent";
        if (!value.BearingInBinPass)
            return "bearing_outside_target_bin";
        if (!value.BearingErrorPass)
            return "bearing_target_error_exceeded";
        if (!value.LocalMotionPass)
            return "local_motion_threshold_exceeded";
        return "settle_predicate_failed";
    }

    private static AttackZoneSettleDigest BuildDigest(
        IReadOnlyList<AttackZoneSettleEvidence> evidence)
    {
        if (evidence.Count == 0)
            throw new InvalidOperationException("cannot digest an empty settle streak");
        var first = evidence[0];
        var last = evidence[^1];
        var evaluations = evidence.Select(value => value.Evaluation).ToArray();
        var motionStrata = evaluations.Select(value => value.Motion.MotionStratum)
            .Distinct(StringComparer.Ordinal).ToArray();
        var facingStrata = evaluations.Select(value => value.Motion.FacingStratum)
            .Distinct(StringComparer.Ordinal).ToArray();
        var motion = evaluations.All(value => value.OpponentStationary)
            ? "stationary"
            : motionStrata.Length == 1 ? motionStrata[0] : "compound_or_unknown";
        var facing = facingStrata.Length == 1
            ? facingStrata[0]
            : "opponent_facing_changed_or_unknown";
        return new AttackZoneSettleDigest(
            evidence.Count,
            first.Sample.Clock,
            last.Sample.Clock,
            evaluations.Min(value => value.Geometry.DistanceMeters),
            evaluations.Max(value => value.Geometry.DistanceMeters),
            evaluations.Min(value => value.Geometry.LocalBearingToOpponentDegrees),
            evaluations.Max(value => value.Geometry.LocalBearingToOpponentDegrees),
            evaluations.Min(value => value.BearingErrorDegrees),
            evaluations.Max(value => value.BearingErrorDegrees),
            evaluations.Min(value => value.LocalPlanarSpeedMetersPerSecond),
            evaluations.Max(value => value.LocalPlanarSpeedMetersPerSecond),
            evaluations.Min(value => value.LocalYawRateRadiansPerSecond),
            evaluations.Max(value => value.LocalYawRateRadiansPerSecond),
            evaluations.Min(value => value.Motion.OpponentPlanarSpeedMetersPerSecond),
            evaluations.Max(value => value.Motion.OpponentPlanarSpeedMetersPerSecond),
            evaluations.Min(value => value.Motion.OpponentYawRateRadiansPerSecond),
            evaluations.Max(value => value.Motion.OpponentYawRateRadiansPerSecond),
            evaluations.Min(value => value.Motion.RadialClosingSpeedMetersPerSecond),
            evaluations.Max(value => value.Motion.RadialClosingSpeedMetersPerSecond),
            evaluations.Min(value => value.Motion.TangentialSpeedMetersPerSecond),
            evaluations.Max(value => value.Motion.TangentialSpeedMetersPerSecond),
            evaluations.All(value => value.ClockValid),
            evaluations.All(value => value.RootsFinite),
            evaluations.All(value => value.GeometryValid),
            evaluations.All(value => value.AnimationValid),
            evaluations.All(value => value.NeutralRequestMethodReturned),
            evaluations.All(value => value.VelocityCommandExactNeutral),
            evaluations.All(value => value.LocalActionReady),
            evaluations.All(value => value.NoPendingRequests),
            evaluations.All(value => value.LocalHealthy),
            evaluations.All(value => value.OpponentHealthy),
            evaluations.All(value => value.DistanceCentralPass),
            evaluations.All(value => value.BearingInBinPass),
            evaluations.All(value => value.BearingErrorPass),
            evaluations.All(value => value.LocalMotionPass),
            evaluations.All(value => value.OpponentStationary),
            motion,
            facing);
    }
}

internal sealed record AttackZoneSettleEvidence(
    AttackZoneControlObservation Sample,
    AttackZoneSampleEvaluation Evaluation);

internal sealed record AttackZoneSettleUpdate(
    bool Acquired,
    int ConsecutiveTicks,
    bool StreakReset,
    string Reason,
    AttackZoneSettleDigest? Digest,
    AttackZoneSettleEvidence? CurrentEvidence);

internal sealed record AttackZoneSettleDigest(
    int SampleCount,
    AttackZoneClock FirstClock,
    AttackZoneClock LastClock,
    double MinimumDistanceMeters,
    double MaximumDistanceMeters,
    double MinimumLocalBearingDegrees,
    double MaximumLocalBearingDegrees,
    double MinimumBearingErrorDegrees,
    double MaximumBearingErrorDegrees,
    double MinimumLocalPlanarSpeedMetersPerSecond,
    double MaximumLocalPlanarSpeedMetersPerSecond,
    double MinimumLocalYawRateRadiansPerSecond,
    double MaximumLocalYawRateRadiansPerSecond,
    double MinimumOpponentPlanarSpeedMetersPerSecond,
    double MaximumOpponentPlanarSpeedMetersPerSecond,
    double MinimumOpponentYawRateRadiansPerSecond,
    double MaximumOpponentYawRateRadiansPerSecond,
    double MinimumRadialClosingSpeedMetersPerSecond,
    double MaximumRadialClosingSpeedMetersPerSecond,
    double MinimumTangentialSpeedMetersPerSecond,
    double MaximumTangentialSpeedMetersPerSecond,
    bool AllClocksValid,
    bool AllRootsFinite,
    bool AllGeometryValid,
    bool AllAnimationValid,
    bool AllNeutralRequestMethodReturned,
    bool AllVelocityCommandsExactNeutral,
    bool AllLocalActionReady,
    bool AllNoPendingRequests,
    bool AllLocalHealthy,
    bool AllOpponentHealthy,
    bool AllDistanceCentralPass,
    bool AllBearingInBinPass,
    bool AllBearingErrorPass,
    bool AllLocalMotionPass,
    bool AllOpponentStationary,
    string OpponentMotionStratum,
    string OpponentFacingStratum);
