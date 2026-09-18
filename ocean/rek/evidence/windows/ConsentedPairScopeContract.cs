namespace RekEvidence;

// Pure, inactive by default. No native reads, logging, control or recorder wiring.
// A client-local connection list or lobby occupancy count is not a complete roster.
internal sealed record ConsentedPairGrant(
    bool Enabled, bool BothAccountsConsented, string? ArenaId, string? SessionInstance,
    string? LocalAccountId, string? OtherAccountId, int LocalSlot, long ParticipantEpoch,
    long IssuedQpc, long ExpiresQpc, long QpcFrequency, bool AllowNonexclusiveRoom = false);

internal sealed record ConsentedParticipant(string? AccountId, int FighterSlot);

internal sealed record ConsentedPairObservation(
    string? ArenaId, string? SessionInstance, string? LocalAccountId,
    bool IsolatedAgentClient, bool Unranked, bool ExclusiveRoomProven,
    bool ServerAccountBindingsVerified, bool CompleteRosterKnown, bool RosterHistoryContinuous,
    long ParticipantEpoch, IReadOnlyList<ConsentedParticipant>? Participants,
    long ObservedQpc, long QpcFrequency);

internal readonly record struct ConsentedPairDecision(bool Allowed, string Reason);
internal readonly record struct ConsentedPairCaptureDecision(bool CaptureAllowed, bool ControlAllowed, string Reason);

internal static class ConsentedPairScopeContract
{
    internal const double MaximumConsentSeconds = 900;
    internal const double MaximumPassiveCaptureSeconds = 4 * 60 * 60;
    internal const double MaximumObservationAgeSeconds = 0.250;
    private static bool Known(string? value) => !string.IsNullOrWhiteSpace(value);
    private static bool Same(string? a, string? b) => string.Equals(a, b, StringComparison.Ordinal);
    private static ConsentedPairDecision Denied(string reason) => new(false, reason);

    internal static ConsentedPairDecision Evaluate(
        ConsentedPairGrant? grant, ConsentedPairObservation? observed, long nowQpc) => EvaluateCore(grant, observed, nowQpc, false);

    internal static ConsentedPairCaptureDecision EvaluatePassiveCapture(
        ConsentedPairGrant? grant, ConsentedPairObservation? observed, long nowQpc)
    {
        var result = EvaluateCore(grant, observed, nowQpc, true);
        return new(result.Allowed, false, result.Allowed ? "consented_pair_passive_capture_roster_completeness_unknown" : result.Reason);
    }

    private static ConsentedPairDecision EvaluateCore(
        ConsentedPairGrant? grant, ConsentedPairObservation? observed, long nowQpc, bool passiveCapture)
    {
        if (grant is null || !grant.Enabled) return Denied("consented_pair_inactive");
        if (!grant.BothAccountsConsented) return Denied("two_account_consent_missing");
        if (!Known(grant.ArenaId) || !Known(grant.SessionInstance) ||
            !Known(grant.LocalAccountId) || !Known(grant.OtherAccountId) ||
            Same(grant.LocalAccountId, grant.OtherAccountId) || grant.LocalSlot is < 0 or > 1 ||
            grant.ParticipantEpoch < 0) return Denied("consent_binding_invalid");
        if (grant.QpcFrequency <= 0 || grant.IssuedQpc < 0 || grant.ExpiresQpc <= grant.IssuedQpc ||
            ((decimal)grant.ExpiresQpc - grant.IssuedQpc) / grant.QpcFrequency >
                (decimal)(passiveCapture ? MaximumPassiveCaptureSeconds : MaximumConsentSeconds))
            return Denied("consent_clock_invalid");
        if (nowQpc < grant.IssuedQpc) return Denied("consent_clock_rollback");
        if (nowQpc >= grant.ExpiresQpc) return Denied("consent_expired");
        if (observed is null) return Denied("participant_observation_missing");
        if (observed.QpcFrequency != grant.QpcFrequency || observed.ObservedQpc < grant.IssuedQpc ||
            nowQpc < observed.ObservedQpc ||
            ((decimal)nowQpc - observed.ObservedQpc) / grant.QpcFrequency > (decimal)MaximumObservationAgeSeconds)
            return Denied("participant_observation_stale_or_clock_invalid");
        if (!passiveCapture && !observed.IsolatedAgentClient) return Denied("isolated_agent_client_not_proven");
        if ((!passiveCapture && !observed.Unranked) || (!observed.ExclusiveRoomProven && !grant.AllowNonexclusiveRoom))
            return Denied("consented_unranked_room_access_not_proven");
        if (!Same(grant.ArenaId, observed.ArenaId) || !Same(grant.SessionInstance, observed.SessionInstance))
            return Denied("consented_room_or_session_changed");
        if (!observed.ServerAccountBindingsVerified) return Denied("server_account_bindings_not_proven");
        if (!Same(grant.LocalAccountId, observed.LocalAccountId)) return Denied("local_account_changed");
        if (!passiveCapture && (!observed.CompleteRosterKnown || !observed.RosterHistoryContinuous))
            return Denied("complete_participant_history_not_proven");
        if (observed.ParticipantEpoch != grant.ParticipantEpoch) return Denied("participant_membership_changed");
        var roster = observed.Participants;
        if (roster is null || roster.Count != 2) return Denied("exactly_two_participants_required");
        var expected = new[] { grant.LocalAccountId, grant.OtherAccountId };
        var slots = new[] { grant.LocalSlot, 1 - grant.LocalSlot };
        for (var i = 0; i < 2; i++)
            if (roster.Count(p => p is not null && Same(p.AccountId, expected[i]) && p.FighterSlot == slots[i]) != 1)
                return Denied("consented_fighter_pair_not_proven");
        return new(true, "consented_pair_scope_proven");
    }
}
