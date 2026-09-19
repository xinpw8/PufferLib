using System.Security.Cryptography;
using System.Text.Json;
using RekUiBridgeAgent;

var checks = 0;
void Check(bool result, string reason) { checks++; if (!result) throw new Exception(reason); }
byte[] Wire(byte mask = 3, byte seconds = 10, byte sequence = 9, byte type = 6, byte faller = 255, byte points = 5)
{
    var body = new byte[33];
    body[0] = 1; body[1] = 2; body[2] = 1;
    body[25] = mask; body[26] = seconds; body[27] = sequence; body[28] = type; body[29] = faller; body[30] = points;
    return body;
}
ReceivedRefereePacket Decode(byte[] body)
{
    Check(G1ReceivedRefereeContract.TryDecode(body, out var packet, out var reason) && reason == "", "valid 33-byte receipt decodes");
    return packet!;
}
var packet = Decode(Wire());
Check(packet.CountMask == 3 && packet.CountSeconds == 10 && packet.CallSequence == 9 && packet.CallType == 6 &&
    packet.CallFaller == -1 && packet.CallPoints == 5, "exact audited offsets 25..30 and signed faller");
Check(!packet.RoundKnockoutOccurred && packet.RoundResult == 0, "DoubleKnockout call does not create a terminal round KO");
Check(Convert.FromBase64String(packet.WireBase64).SequenceEqual(Wire()) &&
    packet.WireSha256 == Convert.ToHexString(SHA256.HashData(Wire())).ToLowerInvariant(), "exact immutable body provenance");
foreach (var length in new[] { 0, 1, 6, 25, 30, 32, 34, 4096 })
    Check(!G1ReceivedRefereeContract.TryDecode(new byte[length], out var absent, out _) && absent is null, "wrong length cannot supply zero referee fields");
foreach (var offset in new[] { 0, 2, 3, 12, 13, 25 })
{
    var body = Wire(); body[offset] = 255;
    Check(!G1ReceivedRefereeContract.TryDecode(body, out var absent, out _) && absent is null, "unknown schema ranges fail closed");
}
for (var value = 0; value < 256; value++)
{
    var decoded = Decode(Wire(seconds: (byte)value, sequence: (byte)value, type: (byte)value, faller: (byte)value, points: (byte)value));
    Check(decoded.CountSeconds == value && decoded.CallSequence == value && decoded.CallType == value &&
        decoded.CallFaller == unchecked((sbyte)value) && decoded.CallPoints == value, "wire bytes preserved without fabricated clamps");
    Check((G1ReceivedRefereeContract.CallName((byte)value) is not null) == (value < 7), "unknown call enum remains unnamed");
}
Check(Enumerable.Range(0, 7).Select(i => G1ReceivedRefereeContract.CallName((byte)i)).SequenceEqual(
    new[] { "Slip", "SlipEStop", "Knockdown", "BeatCount", "Knockout", "DoubleKnockdown", "DoubleKnockout" }), "recovered call names exact");

var id = new RefereeRuntimeIdentity(11, 22, 33, 44, 1, 2, false);
var clock = new RefereeReceiptClock(1000, 1000, 12, 1.5, 1.5);
RefereeClientMirrors Mirrors(ReceivedRefereePacket p) => new(true, p.CallSequence, p.CountMask, p.CountMask == 0 ? -1 : p.CountSeconds);
var cache = new G1ReceivedRefereeCache();
var view = cache.Read(id, Mirrors(packet), 1000, 1000);
Check(!view.Available && view.Receipt is null && view.AgeSeconds is null, "native mirror values alone cannot invent a receipt");
Check(cache.Commit(packet, id, Mirrors(packet), clock, out _), "postfix verified receipt accepted");
view = cache.Read(id, Mirrors(packet), 1500, 1000);
Check(view.Available && view.AgeSeconds == 0.5 && view.Receipt!.CallHistoryCensored, "500ms inclusive freshness and initial left-censor");
var firstCall = view.Receipt!.CallObservationSequence;
Check(cache.Commit(packet, id, Mirrors(packet), clock with { QpcTicks = 1400 }, out _), "repeated packet refreshes measured receipt clock");
view = cache.Read(id, Mirrors(packet), 1500, 1000);
Check(view.Receipt!.CallObservationSequence == firstCall && view.Receipt.CallTransition == "repeated_latched_call" &&
    view.Receipt.CallHistoryCensored, "repeated latched call cannot become a repeated event");
view = cache.Read(id, Mirrors(packet), 1901, 1000);
Check(!view.Available && view.Receipt is null && view.Reason == "referee_receipt_stale", "501ms stale receipt exposes no current measurements");
Check(!cache.Read(id, Mirrors(packet), 1500, 1000).Available, "clock reversal cannot revive stale data");

foreach (var changed in new[] { id with { Coordinator = 12 }, id with { Network = 23 }, id with { Fight = 34 },
    id with { Round = 45 }, id with { FightEpoch = 2 }, id with { RoundNumber = 3 }, id with { Redo = true },
    id with { Coordinator = 0 }, id with { Network = 0 }, id with { Fight = 0 }, id with { Round = 0 } })
{
    cache = new(); cache.Commit(packet, id, Mirrors(packet), clock, out _);
    view = cache.Read(changed, Mirrors(packet), 1100, 1000);
    Check(!view.Available && view.Receipt is null, "new coordinator/network/fight/round/epoch/redo invalidates old call");
    Check(!cache.Read(id, Mirrors(packet), 1100, 1000).Available, "old lifecycle cannot resurrect after mismatch");
}
foreach (var bad in new[] { Mirrors(packet) with { SnapshotSeen = false }, Mirrors(packet) with { CallSequence = 0 },
    Mirrors(packet) with { CountMask = 0 }, Mirrors(packet) with { CountSeconds = -1 } })
{
    cache = new(); Check(!cache.Commit(packet, id, bad, clock, out _), "native mirror mismatch rejects successful apply claim");
    cache.Commit(packet, id, Mirrors(packet), clock, out _);
    Check(!cache.Read(id, bad, 1100, 1000).Available, "native replay reset without packet invalidates receipt");
}
foreach (var bad in new[] { clock with { QpcTicks = 0 }, clock with { QpcFrequency = 0 },
    clock with { UnityFrame = -1 }, clock with { UnityTime = double.NaN }, clock with { UnityUnscaledTime = double.PositiveInfinity } })
{
    cache = new(); Check(!cache.Commit(packet, id, Mirrors(packet), bad, out _), "unknown receipt clocks fail closed");
}
foreach (var bad in new[] { clock with { QpcTicks = 999 }, clock with { QpcFrequency = 999 } })
{
    cache = new(); cache.Commit(packet, id, Mirrors(packet), clock, out _);
    Check(!cache.Commit(packet, id, Mirrors(packet), bad, out _), "regressed/frequency-changed receipt rejected");
}
foreach (var readClock in new[] { (999L, 1000L), (1100L, 0L), (1100L, 999L) })
{
    cache = new(); cache.Commit(packet, id, Mirrors(packet), clock, out _);
    Check(!cache.Read(id, Mirrors(packet), readClock.Item1, readClock.Item2).Available, "invalid read clock fails closed");
}
cache = new(); cache.Commit(packet, id, Mirrors(packet), clock, out _);
cache.Invalidate("referee_native_replay_reset");
Check(!cache.Read(id, Mirrors(packet), 1100, 1000).Available, "explicit native replay reset invalidates cache");
cache.Commit(packet, id, Mirrors(packet), clock with { QpcTicks = 1100 }, out _);
Check(cache.Read(id, Mirrors(packet), 1100, 1000).Receipt!.Lifecycle > 1, "reset namespaces later same sequence");
cache.Invalidate("referee_network_lifecycle_changed");
Check(!cache.Read(id, Mirrors(packet), 1100, 1000).Available, "disconnect invalidates cache independent of object reuse");

cache = new();
var empty = Decode(Wire(mask: 0, seconds: 0, sequence: 0, type: 0, faller: 0, points: 0));
cache.Commit(empty, id, Mirrors(empty), clock, out _);
view = cache.Read(id, Mirrors(empty), 1000, 1000);
Check(view.Available && view.Receipt!.Packet.CallSequence == 0 && view.Receipt.CallObservationSequence is null &&
    view.Receipt.CallTransition == "empty_sequence", "received sequence zero is no call, never an inferred Slip");
var callOne = Decode(Wire(mask: 1, sequence: 1, type: 0, faller: 0, points: 0));
cache.Commit(callOne, id, Mirrors(callOne), clock with { QpcTicks = 1100 }, out _);
view = cache.Read(id, Mirrors(callOne), 1100, 1000);
Check(view.Receipt!.CallTransition == "changed_received_sequence" && !view.Receipt.CallHistoryCensored, "first changed sequence after observed empty is distinguishable");
var last = Decode(Wire(sequence: 255));
cache.Commit(last, id, Mirrors(last), clock with { QpcTicks = 1200 }, out _);
cache.Commit(callOne, id, Mirrors(callOne), clock with { QpcTicks = 1300 }, out _);
Check(cache.Read(id, Mirrors(callOne), 1300, 1000).Receipt!.CallTransition == "observed_255_to_1_wrap", "byte wrap skips zero");
var eight = Decode(Wire(sequence: 8)); var two = Decode(Wire(sequence: 2));
cache.Commit(eight, id, Mirrors(eight), clock with { QpcTicks = 1400 }, out _);
cache.Commit(two, id, Mirrors(two), clock with { QpcTicks = 1500 }, out _);
view = cache.Read(id, Mirrors(two), 1500, 1000);
Check(view.Receipt!.CallHistoryCensored && view.Receipt.CallTransition == "sequence_decrease_censored", "uncertain decreases do not imply a new physical event");
var changedPayload = two with { CallType = 4 };
cache.Commit(changedPayload, id, Mirrors(changedPayload), clock with { QpcTicks = 1600 }, out _);
Check(cache.Read(id, Mirrors(changedPayload), 1600, 1000).Receipt!.CallTransition == "same_sequence_payload_changed_censored", "ambiguous payload sequence is censored");
cache.Commit(changedPayload, id, Mirrors(changedPayload), clock with { QpcTicks = 2101 }, out _);
Check(cache.Read(id, Mirrors(changedPayload), 2101, 1000).Receipt!.CallTransition == "initial_latched_call", "receipt gap cannot preserve complete event history");
Check(!cache.Commit(packet, id with { RoundNumber = 3 }, Mirrors(packet), clock, out _), "packet cannot bind to a different applied round");
Check(!cache.Commit(packet, id with { Redo = true }, Mirrors(packet), clock, out _), "packet cannot bind to a different redo state");
var unavailable = cache.Read(id, Mirrors(packet), 1100, 1000);
Check(JsonSerializer.Serialize(unavailable).Contains("\"Receipt\":null"), "unavailable payload has null data, never default zeros");
foreach (var jump in new[] { (0, 255), (1, 3), (254, 1) })
{
    cache = new();
    var before = Decode(Wire(sequence: (byte)jump.Item1));
    var after = Decode(Wire(sequence: (byte)jump.Item2));
    cache.Commit(before, id, Mirrors(before), clock, out _);
    cache.Commit(after, id, Mirrors(after), clock with { QpcTicks = 1100 }, out _);
    view = cache.Read(id, Mirrors(after), 1100, 1000);
    Check(view.Available && view.Receipt!.CallHistoryCensored && view.Receipt.Packet.CallSequence == jump.Item2,
        "sequence gaps censor history while preserving measured current packet availability");
}
Console.WriteLine(JsonSerializer.Serialize(new { passed = checks, native_game_invocations = 0, global_input_invocations = 0,
    referee_schema = G1ReceivedRefereeContract.Schema, maximum_receipt_age_seconds = G1ReceivedRefereeContract.MaximumReceiptAgeSeconds }));
