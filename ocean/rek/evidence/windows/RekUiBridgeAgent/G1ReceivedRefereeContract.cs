using System.Security.Cryptography;

namespace RekUiBridgeAgent;

public readonly record struct RefereeRuntimeIdentity(long Coordinator, long Network, long Fight, long Round,
    int FightEpoch, int RoundNumber, bool Redo)
{
    public bool Known => Coordinator != 0 && Network != 0 && Fight != 0 && Round != 0 && FightEpoch >= 0 && RoundNumber >= 0;
}
public readonly record struct RefereeClientMirrors(bool SnapshotSeen, byte CallSequence, byte CountMask, int CountSeconds);
public readonly record struct RefereeReceiptClock(long QpcTicks, long QpcFrequency, int UnityFrame,
    double UnityTime, double UnityUnscaledTime);
public sealed record ReceivedRefereePacket(byte Phase, byte RoundNumber, bool RoundActive, bool Redo,
    bool RoundKnockoutOccurred, byte RoundResult, byte CountMask, byte CountSeconds,
    byte CallSequence, byte CallType, sbyte CallFaller, byte CallPoints, string WireSha256, string WireBase64);
public sealed record RefereeReceipt(ReceivedRefereePacket Packet, RefereeRuntimeIdentity Identity,
    RefereeReceiptClock Clock, long ReceiptSequence, long Lifecycle, long? CallObservationSequence,
    string CallTransition, bool CallHistoryCensored);
public sealed record RefereeAvailability(bool Available, string Reason, double? AgeSeconds, RefereeReceipt? Receipt);

public static class G1ReceivedRefereeContract
{
    public const string Schema = "rek.g1_received_referee.v1";
    public const int WireBytes = 33;
    // Client observation freshness budget, not a recovered physics/referee threshold.
    public const double MaximumReceiptAgeSeconds = 0.5;

    public static bool TryDecode(ReadOnlySpan<byte> body, out ReceivedRefereePacket? packet, out string reason)
    {
        packet = null;
        if (body.Length != WireBytes) { reason = "referee_wire_length_not_33"; return false; }
        if (body[0] > 6 || body[2] > 1 || body[3] > 1 || body[12] > 1 || body[13] > 4 || body[25] > 3)
        { reason = "referee_wire_schema_value_invalid"; return false; }
        packet = new(body[0], body[1], body[2] != 0, body[3] != 0, body[12] != 0, body[13],
            body[25], body[26], body[27], body[28], unchecked((sbyte)body[29]), body[30],
            Convert.ToHexString(SHA256.HashData(body)).ToLowerInvariant(), Convert.ToBase64String(body));
        reason = "";
        return true;
    }

    public static string? CallName(byte type) => type switch {
        0 => "Slip", 1 => "SlipEStop", 2 => "Knockdown", 3 => "BeatCount", 4 => "Knockout",
        5 => "DoubleKnockdown", 6 => "DoubleKnockout", _ => null };

    public static bool MirrorsMatch(ReceivedRefereePacket packet, RefereeClientMirrors mirrors) =>
        mirrors.SnapshotSeen && mirrors.CallSequence == packet.CallSequence && mirrors.CountMask == packet.CountMask &&
        mirrors.CountSeconds == (packet.CountMask == 0 ? -1 : packet.CountSeconds);

    public static bool ClockKnown(RefereeReceiptClock clock) => clock.QpcTicks > 0 && clock.QpcFrequency > 0 &&
        clock.UnityFrame >= 0 && double.IsFinite(clock.UnityTime) && clock.UnityTime >= 0 &&
        double.IsFinite(clock.UnityUnscaledTime) && clock.UnityUnscaledTime >= 0;
}

// Binds the last observed packet to its applied native lifecycle. Unknown state
// never comes from zero-initialized native mirrors. Only a copied, decoded,
// successfully applied receipt can establish availability.
public sealed class G1ReceivedRefereeCache
{
    private RefereeReceipt? _receipt;
    private string _reason = "referee_snapshot_not_observed";
    private long _sequence, _lifecycle = 1, _callSequence;

    public void Invalidate(string reason)
    {
        if (_receipt is not null) _lifecycle++;
        _receipt = null;
        _reason = reason;
    }

    public bool Commit(ReceivedRefereePacket packet, RefereeRuntimeIdentity identity,
        RefereeClientMirrors mirrors, RefereeReceiptClock clock, out string reason)
    {
        reason = !identity.Known ? "referee_runtime_identity_unavailable" :
            identity.RoundNumber != packet.RoundNumber || identity.Redo != packet.Redo ? "referee_packet_round_mismatch" :
            !G1ReceivedRefereeContract.ClockKnown(clock) ? "referee_receipt_clock_invalid" :
            !G1ReceivedRefereeContract.MirrorsMatch(packet, mirrors) ? "referee_applied_mirrors_mismatch" : "";
        if (reason.Length != 0) { Invalidate(reason); return false; }
        if (_receipt is { } previousClock && (previousClock.Clock.QpcFrequency != clock.QpcFrequency || clock.QpcTicks < previousClock.Clock.QpcTicks))
        { reason = "referee_receipt_clock_regressed_or_changed"; Invalidate(reason); return false; }
        if (_receipt is { } prior && (prior.Identity != identity ||
            (clock.QpcTicks - prior.Clock.QpcTicks) / (double)clock.QpcFrequency > G1ReceivedRefereeContract.MaximumReceiptAgeSeconds))
            Invalidate("referee_lifecycle_or_receipt_gap");

        var previous = _receipt;
        string transition; bool censored; long? callObservation;
        if (packet.CallSequence == 0)
        { transition = "empty_sequence"; censored = false; callObservation = null; }
        else if (previous is null)
        { transition = "initial_latched_call"; censored = true; callObservation = ++_callSequence; }
        else if (previous.Packet.CallSequence == packet.CallSequence &&
            previous.Packet.CallType == packet.CallType && previous.Packet.CallFaller == packet.CallFaller &&
            previous.Packet.CallPoints == packet.CallPoints)
        { transition = "repeated_latched_call"; censored = previous.CallHistoryCensored; callObservation = previous.CallObservationSequence; }
        else
        {
            var priorSequence = previous.Packet.CallSequence;
            var wrap = priorSequence == 255 && packet.CallSequence == 1;
            var forwardGap = packet.CallSequence > priorSequence + 1;
            censored = priorSequence == packet.CallSequence || forwardGap || (packet.CallSequence < priorSequence && !wrap);
            transition = priorSequence == packet.CallSequence ? "same_sequence_payload_changed_censored" :
                wrap ? "observed_255_to_1_wrap" : forwardGap ? "sequence_gap_censored" :
                censored ? "sequence_decrease_censored" : "changed_received_sequence";
            callObservation = ++_callSequence;
        }
        _receipt = new(packet, identity, clock, ++_sequence, _lifecycle, callObservation, transition, censored);
        _reason = "received_snapshot_applied_and_bound";
        return true;
    }

    public RefereeAvailability Read(RefereeRuntimeIdentity identity, RefereeClientMirrors mirrors,
        long nowQpc, long frequency)
    {
        if (_receipt is not { } receipt) return new(false, _reason, null, null);
        string? failure = identity != receipt.Identity || !identity.Known ? "referee_runtime_identity_changed" :
            !G1ReceivedRefereeContract.MirrorsMatch(receipt.Packet, mirrors) ? "referee_native_mirrors_changed_without_receipt" :
            frequency != receipt.Clock.QpcFrequency || frequency <= 0 || nowQpc < receipt.Clock.QpcTicks ? "referee_receipt_clock_invalid" : null;
        var age = failure is null ? (nowQpc - receipt.Clock.QpcTicks) / (double)frequency : (double?)null;
        if (failure is null && age > G1ReceivedRefereeContract.MaximumReceiptAgeSeconds) failure = "referee_receipt_stale";
        if (failure is not null)
        { Invalidate(failure); return new(false, failure, age, null); }
        return new(true, _reason, age, receipt);
    }
}
