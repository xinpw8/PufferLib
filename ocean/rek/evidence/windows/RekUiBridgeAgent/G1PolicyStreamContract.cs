using System.Text.Json;

namespace RekUiBridgeAgent;

internal sealed record G1PolicyAction(string RoundIdentity, long ObservationSequence, int Action);

internal struct G1PolicyPublicationClock
{
    internal int LastPublishedFrame;
    internal long LastObservedQpc, Frequency;
    internal decimal NextDeadlineQpc;
}

internal static class G1PolicyStreamContract
{
    internal const string Schema = "rek.g1_policy_source.v1";
    internal const double MaximumAgeSeconds = 0.250;
    internal const double StartupGraceSeconds = 1.0;
    internal const int PublishRateHz = 50;
    internal static readonly int[] MoveOrder = { 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16 };
    internal static readonly G1HeldMask[] Held = {
        G1HeldMask.None, G1HeldMask.None, G1HeldMask.W, G1HeldMask.S,
        G1HeldMask.A, G1HeldMask.D, G1HeldMask.Q, G1HeldMask.E,
        G1HeldMask.W|G1HeldMask.Q, G1HeldMask.W|G1HeldMask.E,
        G1HeldMask.S|G1HeldMask.Q, G1HeldMask.S|G1HeldMask.E,
        G1HeldMask.A|G1HeldMask.Q, G1HeldMask.A|G1HeldMask.E,
        G1HeldMask.D|G1HeldMask.Q, G1HeldMask.D|G1HeldMask.E };

    internal static bool IsHash(string? value) => value is { Length: 64 } &&
        value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');

    internal static bool TryParse(JsonElement root, out G1PolicyAction? action,
        out string? requestId, out string error)
    {
        action = null; requestId = null; error = "invalid_policy_action";
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in root.EnumerateObject())
            if (!names.Add(p.Name) || p.Name is not ("type" or "request_id" or
                "round_identity_sha256" or "observation_sequence" or "action")) return false;
        if (names.Count != 5 || !root.TryGetProperty("request_id", out var id) ||
            id.ValueKind != JsonValueKind.String) return false;
        requestId = id.GetString();
        if (requestId is not { Length: >= 1 and <= 64 } ||
            !requestId.All(c => c is >= 'a' and <= 'z' or >= 'A' and <= 'Z' or
                >= '0' and <= '9' or '.' or '_' or ':' or '-')) { requestId = null; return false; }
        if (!root.TryGetProperty("round_identity_sha256", out var round) ||
            round.ValueKind != JsonValueKind.String || !IsHash(round.GetString()) ||
            !root.TryGetProperty("observation_sequence", out var seq) ||
            seq.ValueKind != JsonValueKind.Number || !seq.TryGetInt64(out var sequence) || sequence <= 0 || sequence > 9007199254740991L ||
            !root.TryGetProperty("action", out var a) || a.ValueKind != JsonValueKind.Number || !a.TryGetInt32(out var category) ||
            category is < 0 or > 32) return false;
        action = new G1PolicyAction(round.GetString()!, sequence, category);
        error = string.Empty; return true;
    }

    internal static string? RejectReason(G1PolicyAction action, string? round,
        long lastConsumed, long producedAt, long now, long frequency)
    {
        if (!string.Equals(action.RoundIdentity, round, StringComparison.Ordinal)) return "round_identity_mismatch";
        if (action.ObservationSequence <= lastConsumed) return "observation_already_consumed";
        if (producedAt <= 0 || frequency <= 0 || now < producedAt) return "observation_unknown_or_clock_invalid";
        if ((now - producedAt) / (double)frequency > MaximumAgeSeconds) return "stale_observation";
        return null;
    }

    internal static float Forward(G1HeldMask held) => (held & G1HeldMask.W) != 0 ? 1 : (held & G1HeldMask.S) != 0 ? -1 : 0;
    internal static float Strafe(G1HeldMask held) => (held & G1HeldMask.A) != 0 ? 1 : (held & G1HeldMask.D) != 0 ? -1 : 0;
    internal static float Yaw(G1HeldMask held) => (held & G1HeldMask.Q) != 0 ? 1 : (held & G1HeldMask.E) != 0 ? -1 : 0;
    internal static G1HeldMask Translation(G1HeldMask held) => held & (G1HeldMask.W|G1HeldMask.S|G1HeldMask.A|G1HeldMask.D);

    internal static bool AttackTranslationReady(G1HeldMask desiredHeld, float forward, float strafe) =>
        Translation(desiredHeld) == G1HeldMask.None && forward == 0f && strafe == 0f;

    internal static bool VisualTransportComplete(bool visualOnly, bool sendReturned, bool pending) =>
        visualOnly && sendReturned && !pending;

    internal static bool ShouldPublish(ref G1PolicyPublicationClock clock, int frame, long now, long frequency)
    {
        if (frequency <= 0 || now <= 0 || (clock.Frequency != 0 &&
                (frequency != clock.Frequency || now < clock.LastObservedQpc))) return false;
        clock.LastObservedQpc = now;
        // Decimal preserves integer QPC values and the exact frequency / 50 period.
        var interval = (decimal)frequency / PublishRateHz;
        if (clock.Frequency == 0)
        {
            clock.Frequency = frequency;
            clock.LastPublishedFrame = frame;
            clock.NextDeadlineQpc = now + interval;
            return true;
        }
        if (frame == clock.LastPublishedFrame || now < clock.NextDeadlineQpc) return false;
        // Carry normal frame lateness forward, but discard every missed deadline
        // after a stall. Only this measured frame can produce an observation.
        var missed = decimal.Floor((now - clock.NextDeadlineQpc) / interval);
        clock.NextDeadlineQpc += (missed + 1) * interval;
        clock.LastPublishedFrame = frame;
        return true;
    }

    internal static bool WatchdogExpired(bool receivedAction, long lastAction, long now, long frequency) =>
        frequency <= 0 || now < lastAction ||
        (now - lastAction) / (double)frequency > (receivedAction ? MaximumAgeSeconds : StartupGraceSeconds);
}
