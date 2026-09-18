using RekEvidence;

namespace RekEvidenceRecorder;

internal sealed record ConsentedPairCaptureConfig(bool Enabled, string ArenaDisplayName,
    string LocalDisplayName, string OtherDisplayName, DateTimeOffset ExpiresUtc, bool AllowNonexclusiveRoom);
internal sealed record PairCaptureFacts(string ArenaId, string ArenaDisplayName, string SessionInstance,
    string LocalName, string LocalId, string OtherName, string OtherId, int LocalSlot,
    bool BothHuman, bool Unranked, string? ObservedConnectionSet);

internal sealed class ConsentedPairCaptureTracker
{
    private readonly ConsentedPairCaptureConfig _config;
    private readonly long _expiresQpc, _frequency;
    private ConsentedPairGrant? _grant;
    private string? _connections, _stopped;
    internal bool Bound => _grant is not null;
    internal string? LocalId => _grant?.LocalAccountId;
    internal string? OtherId => _grant?.OtherAccountId;
    internal string? ArenaId => _grant?.ArenaId;
    internal ConsentedPairCaptureTracker(ConsentedPairCaptureConfig config, DateTimeOffset nowUtc, long nowQpc, long frequency)
    {
        _config = config; _frequency = frequency;
        var seconds = (config.ExpiresUtc - nowUtc).TotalSeconds;
        if (!config.Enabled || !config.AllowNonexclusiveRoom || seconds <= 0 || seconds > 900 || frequency <= 0 ||
            string.IsNullOrWhiteSpace(config.ArenaDisplayName) || string.IsNullOrWhiteSpace(config.LocalDisplayName) ||
            string.IsNullOrWhiteSpace(config.OtherDisplayName) || config.LocalDisplayName == config.OtherDisplayName)
            throw new InvalidDataException("invalid_consented_pair_capture_configuration");
        _expiresQpc = checked(nowQpc + (long)(seconds * frequency));
    }
    internal void ObserveMembershipChange() { if (Bound) _stopped = "observed_participant_or_session_change"; }
    internal ConsentedPairCaptureDecision Evaluate(PairCaptureFacts f, long nowQpc)
    {
        ConsentedPairCaptureDecision Deny(string reason) => new(false, false, reason);
        if (_stopped is not null) return Deny(_stopped);
        if (nowQpc >= _expiresQpc) return Deny("consent_expired");
        if (!f.BothHuman || f.ArenaDisplayName != _config.ArenaDisplayName ||
            f.LocalName != _config.LocalDisplayName || f.OtherName != _config.OtherDisplayName ||
            string.IsNullOrWhiteSpace(f.LocalId) || string.IsNullOrWhiteSpace(f.OtherId) || f.LocalId == f.OtherId)
        {
            if (Bound) _stopped = "consented_fighter_or_room_changed";
            return Deny(_stopped ?? "consented_named_pair_not_observed");
        }
        if (_grant is null)
        {
            _grant = new(true, true, f.ArenaId, f.SessionInstance, f.LocalId, f.OtherId, f.LocalSlot, 0,
                nowQpc, _expiresQpc, _frequency, true);
            _connections = f.ObservedConnectionSet;
        }
        if (_connections != f.ObservedConnectionSet) { ObserveMembershipChange(); return Deny(_stopped!); }
        var observed = new ConsentedPairObservation(f.ArenaId, f.SessionInstance, f.LocalId, false, f.Unranked, false,
            true, false, false, 0, new[] { new ConsentedParticipant(f.LocalId, f.LocalSlot), new ConsentedParticipant(f.OtherId, 1-f.LocalSlot) },
            nowQpc, _frequency);
        var decision = ConsentedPairScopeContract.EvaluatePassiveCapture(_grant, observed, nowQpc);
        if (!decision.CaptureAllowed) _stopped = decision.Reason;
        return decision;
    }
}
