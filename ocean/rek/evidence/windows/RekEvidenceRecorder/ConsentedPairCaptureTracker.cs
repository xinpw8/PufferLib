using RekEvidence;

namespace RekEvidenceRecorder;

internal sealed record ConsentedPairCaptureConfig(bool Enabled, string ArenaDisplayName,
    string LocalDisplayName, string OtherDisplayName, DateTimeOffset ExpiresUtc, bool AllowNonexclusiveRoom);
internal sealed record PairCaptureFacts(string ArenaId, string ArenaDisplayName, string SessionInstance,
    string LocalName, string LocalId, string OtherName, string OtherId, int LocalSlot,
    bool BothHuman, bool Unranked, string? ObservedConnectionSet);

internal sealed class ConsentedPairCaptureTracker
{
    private ConsentedPairCaptureConfig _config;
    private long _issuedQpc, _expiresQpc;
    private readonly long _frequency;
    private ConsentedPairGrant? _grant;
    private string? _connections, _stopped, _configurationReason;
    internal bool Bound => _grant is not null;
    internal string? LocalId => _grant?.LocalAccountId;
    internal string? OtherId => _grant?.OtherAccountId;
    internal string? ArenaId => _grant?.ArenaId;
    internal static bool UsePairModeAtStartup(bool fileExists, ConsentedPairCaptureConfig? config, bool readFailed) =>
        fileExists && (readFailed || config is null || config.Enabled);
    internal ConsentedPairCaptureTracker(ConsentedPairCaptureConfig config, DateTimeOffset nowUtc, long nowQpc, long frequency)
    {
        _config = config; _frequency = frequency;
        if (ConfigurationReason(config, nowUtc, nowQpc) is not null)
            throw new InvalidDataException("invalid_consented_pair_capture_configuration");
        SetDeadline(config, nowUtc, nowQpc);
    }
    private string? ConfigurationReason(ConsentedPairCaptureConfig? config, DateTimeOffset nowUtc, long nowQpc)
    {
        if (config is null) return "consented_pair_configuration_invalid";
        if (!config.Enabled) return "consented_pair_configuration_disabled";
        var seconds = (config.ExpiresUtc - nowUtc).TotalSeconds;
        if (seconds <= 0) return "consent_expired";
        if (!config.AllowNonexclusiveRoom || seconds > ConsentedPairScopeContract.MaximumPassiveCaptureSeconds ||
            _frequency <= 0 || nowQpc < 0 ||
            string.IsNullOrWhiteSpace(config.ArenaDisplayName) || string.IsNullOrWhiteSpace(config.LocalDisplayName) ||
            string.IsNullOrWhiteSpace(config.OtherDisplayName) || config.LocalDisplayName == config.OtherDisplayName)
            return "consented_pair_configuration_invalid";
        return null;
    }
    private void SetDeadline(ConsentedPairCaptureConfig config, DateTimeOffset nowUtc, long nowQpc)
    {
        var expiresQpc = checked(nowQpc + (long)((decimal)(config.ExpiresUtc - nowUtc).Ticks * _frequency / TimeSpan.TicksPerSecond));
        _issuedQpc = nowQpc; _expiresQpc = expiresQpc; _config = config;
        if (_grant is not null) _grant = _grant with { IssuedQpc = nowQpc, ExpiresQpc = expiresQpc };
    }
    internal void RejectConfiguration(string reason) => _configurationReason = reason;
    internal string Reload(ConsentedPairCaptureConfig? config, DateTimeOffset nowUtc, long nowQpc)
    {
        if (_stopped is not null) return _stopped;
        var reason = ConfigurationReason(config, nowUtc, nowQpc);
        if (reason is not null) { RejectConfiguration(reason); return reason; }
        if (config!.ArenaDisplayName != _config.ArenaDisplayName || config.LocalDisplayName != _config.LocalDisplayName ||
            config.OtherDisplayName != _config.OtherDisplayName || config.AllowNonexclusiveRoom != _config.AllowNonexclusiveRoom)
            return _stopped = "consented_pair_configuration_binding_changed";
        // Polling an unchanged file must never slide its monotonic deadline.
        if (config != _config) SetDeadline(config, nowUtc, nowQpc);
        _configurationReason = nowQpc >= _expiresQpc ? "consent_expired" : null;
        return _configurationReason ?? "consented_pair_configuration_loaded";
    }
    internal void ObserveMembershipChange() { if (Bound) _stopped = "observed_participant_or_session_change"; }
    internal ConsentedPairCaptureDecision Evaluate(PairCaptureFacts f, long nowQpc)
    {
        ConsentedPairCaptureDecision Deny(string reason) => new(false, false, reason);
        if (_stopped is not null) return Deny(_stopped);
        if (!f.BothHuman || f.ArenaDisplayName != _config.ArenaDisplayName ||
            f.LocalName != _config.LocalDisplayName || f.OtherName != _config.OtherDisplayName ||
            string.IsNullOrWhiteSpace(f.LocalId) || string.IsNullOrWhiteSpace(f.OtherId) || f.LocalId == f.OtherId)
        {
            if (Bound) _stopped = "consented_fighter_or_room_changed";
            return Deny(_stopped ?? "consented_named_pair_not_observed");
        }
        // Continue observing bound identity changes while expired or config-disabled.
        if (_grant is not null)
        {
            if (f.ArenaId != _grant.ArenaId || f.SessionInstance != _grant.SessionInstance)
                return Deny(_stopped = "consented_room_or_session_changed");
            if (f.LocalId != _grant.LocalAccountId || f.OtherId != _grant.OtherAccountId || f.LocalSlot != _grant.LocalSlot)
                return Deny(_stopped = "consented_fighter_pair_not_proven");
            if (_connections != f.ObservedConnectionSet) { ObserveMembershipChange(); return Deny(_stopped!); }
        }
        if (_configurationReason is not null) return Deny(_configurationReason);
        if (nowQpc >= _expiresQpc) return Deny("consent_expired");
        if (_grant is null)
        {
            _grant = new(true, true, f.ArenaId, f.SessionInstance, f.LocalId, f.OtherId, f.LocalSlot, 0,
                _issuedQpc, _expiresQpc, _frequency, true);
            _connections = f.ObservedConnectionSet;
        }
        if (_connections != f.ObservedConnectionSet) { ObserveMembershipChange(); return Deny(_stopped!); }
        var observed = new ConsentedPairObservation(f.ArenaId, f.SessionInstance, f.LocalId, false, f.Unranked, false,
            true, false, false, 0, new[] { new ConsentedParticipant(f.LocalId, f.LocalSlot), new ConsentedParticipant(f.OtherId, 1-f.LocalSlot) },
            nowQpc, _frequency);
        var decision = ConsentedPairScopeContract.EvaluatePassiveCapture(_grant, observed, nowQpc);
        if (!decision.CaptureAllowed && decision.Reason != "consent_expired") _stopped = decision.Reason;
        return decision;
    }
}
