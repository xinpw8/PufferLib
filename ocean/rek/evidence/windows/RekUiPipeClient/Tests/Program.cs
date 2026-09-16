using System.Text.Json;
using RekUiBridgeAgent;

var cases = 0;
void Check(bool condition, string name) { cases++; if (!condition) throw new Exception(name); }
void Reject(string json) { cases++; try { PolicyRelay.ValidateRequest(json); } catch { return; } throw new Exception("accepted invalid request"); }
var hash = new string('a', 64);
string Action(int action, long seq = 1) => JsonSerializer.Serialize(new { type = "policy_action", request_id = "a1", round_identity_sha256 = hash, observation_sequence = seq, action });
for (var action = 0; action < 33; action++) { PolicyRelay.ValidateRequest(Action(action)); cases++; }
foreach (var a in new[] { -1, 33, int.MaxValue }) Reject(Action(a));
Reject(Action(1, 0)); Reject(Action(1, 9007199254740992));
Reject(Action(1).Replace("\"action\":1", "\"action\":null"));
Reject(Action(1).Replace("\"action\":1", "\"action\":1.5"));
Reject(Action(1).Replace("\"action\":1", "\"action\":\"1\""));
Reject(Action(1).Replace("\"observation_sequence\":1", "\"observation_sequence\":null"));
Reject(Action(1).Replace("\"action\":1", "\"action\":1,\"action\":1"));
Reject(Action(1).Replace(hash, new string('g', 64)));
Reject(Action(1).Replace("\"action\":1", "\"action\":1,\"opponent\":1"));
Reject("{\"type\":\"input\",\"request_id\":\"x\",\"key\":\"Space\"}");
Reject("{\"type\":\"command\",\"request_id\":\"x\",\"command\":\"StartContinuousBotController\"}");
foreach (var type in new[] { "get_state", "get_policy_state" })
{ PolicyRelay.ValidateRequest(JsonSerializer.Serialize(new { type, request_id = "read1" })); cases++; }
foreach (var command in new[] { "AcquireExclusiveControl", "StartG1PolicyStream", "StopG1PolicyStream", "ReleaseExclusiveControl", "StartRound" })
{ PolicyRelay.ValidateRequest(JsonSerializer.Serialize(new { type = "command", request_id = "c1", command })); cases++; }
var sample = new G1PolicyAction(hash, 10, 17);
Check(G1PolicyStreamContract.RejectReason(sample, hash, 9, 1000, 1250, 1000) is null, "250ms accepted");
Check(G1PolicyStreamContract.RejectReason(sample, hash, 9, 1000, 1251, 1000) == "stale_observation", "251ms rejected");
Check(G1PolicyStreamContract.RejectReason(sample, hash, 10, 1000, 1000, 1000) == "observation_already_consumed", "duplicate rejected");
Check(G1PolicyStreamContract.RejectReason(sample, new string('b',64), 9, 1000, 1000, 1000) == "round_identity_mismatch", "round mismatch rejected");
Check(G1PolicyStreamContract.RejectReason(sample, hash, 9, 0, 1000, 1000) == "observation_unknown_or_clock_invalid", "unknown observation rejected");
Check(G1PolicyStreamContract.RejectReason(sample, hash, 9, 1001, 1000, 1000) == "observation_unknown_or_clock_invalid", "reverse clock rejected");
Check(G1PolicyStreamContract.MoveOrder.SequenceEqual(new[] {6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16}), "move mapping");
Check(G1PolicyStreamContract.Strafe(G1PolicyStreamContract.Held[4]) == 1, "native A sign");
Check(G1PolicyStreamContract.Yaw(G1PolicyStreamContract.Held[6]) == 1, "native Q sign");
Check(G1PolicyStreamContract.Held[8] == (G1HeldMask.W|G1HeldMask.Q), "WQ mapping");
// A visual client's native command can transiently read zero while the owned
// desired translation is still held. The next fixed update restores it.
for (var category = 1; category < 16; category++)
{
    var held = G1PolicyStreamContract.Held[category];
    var translationReleased = category is 1 or 6 or 7;
    Check(G1PolicyStreamContract.AttackTranslationReady(held, 0f, 0f) == translationReleased,
        $"attack eligibility uses desired category {category} during transient native zero");
    Check(G1PolicyStreamContract.AttackTranslationReady(held,
            G1PolicyStreamContract.Forward(held), G1PolicyStreamContract.Strafe(held)) == translationReleased,
        $"attack eligibility is unchanged after native velocity restores category {category}");
}
foreach (var held in new[] { G1HeldMask.None, G1HeldMask.Q, G1HeldMask.E })
{
    foreach (var residual in new[] { -1f, 1f, float.Epsilon, float.NaN, float.PositiveInfinity, float.NegativeInfinity })
    {
        Check(!G1PolicyStreamContract.AttackTranslationReady(held, residual, 0f),
            $"native forward residual still blocks attack after release or yaw {held}");
        Check(!G1PolicyStreamContract.AttackTranslationReady(held, 0f, residual),
            $"native strafe residual still blocks attack after release or yaw {held}");
    }
}
Check(G1PolicyStreamContract.VisualTransportComplete(true, true, false), "visual send completed with pending clear");
Check(!G1PolicyStreamContract.VisualTransportComplete(true, true, true), "visual pending remains owned");
Check(!G1PolicyStreamContract.VisualTransportComplete(true, false, false), "visual clearing alone is not send completion");
Check(!G1PolicyStreamContract.VisualTransportComplete(false, true, false), "nonvisual playback remains separate");
int CountPublications(int framesPerSecond, long frequency = 10_000_000)
{
    var clock = new G1PolicyPublicationClock();
    var count = 0;
    for (var frame = 0; frame < framesPerSecond * 10; frame++)
    {
        var now = frequency + (long)frame * frequency / framesPerSecond;
        var published = G1PolicyStreamContract.ShouldPublish(ref clock, frame, now, frequency);
        Check(!G1PolicyStreamContract.ShouldPublish(ref clock, frame, now, frequency), "same measured frame cannot publish twice");
        if (!published) continue;
        Check(clock.LastObservedQpc == now && clock.NextDeadlineQpc > now, "actual QPC retained and next deadline is in the future");
        count++;
    }
    return count;
}
var publicationCounts = new { fps60 = CountPublications(60), fps100 = CountPublications(100), fps30 = CountPublications(30) };
Check(publicationCounts.fps60 == 500, $"60 FPS produces 500 publications in ten seconds, observed {publicationCounts.fps60}");
Check(publicationCounts.fps100 == 500, "100 FPS produces 500 publications in ten seconds");
Check(publicationCounts.fps30 == 300, "30 FPS publishes every available frame without fabricating observations");
Check(CountPublications(60, 1001) == 500, "fractional QPC tick periods retain 50 Hz phase");
var publishClock = new G1PolicyPublicationClock();
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 10, 1000, 1000), "first rendered source allowed");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 11, 1019, 1000), "first deadline under20ms not due");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 11, 1018, 1000), "clock regression after an unpublished frame rejected");
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 11, 1020, 1000), "first deadline at20ms allowed");
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 12, 3007, 1000), "long stall emits one current source");
Check(publishClock.NextDeadlineQpc == 3020, "long stall discards missed deadlines and retains original phase");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 12, 3007, 1000), "stall cannot duplicate the measured frame");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 13, 3008, 1000), "no next-frame catch-up burst after stall");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 14, 3019, 1000), "no residual catch-up debt after stall");
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 15, 3020, 1000), "normal phase resumes after stall");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 15, 3300, 1000), "same frame never republishes even after multiple deadlines");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 16, 3299, 1000), "backward clock after duplicate-frame callback rejected");
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 16, 3300, 1000), "fresh frame may use the current measured deadline");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 17, 3320, 1001), "frequency changes require an explicit reset");
publishClock = default;
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 16, 100, 1001), "reset clears frame, clock, phase and frequency history");
Check(publishClock.NextDeadlineQpc == 120.02m, "nondivisible clock frequency retains fractional deadline");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 17, 120, 1001), "fractional deadline cannot publish early");
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 17, 121, 1001), "first measured tick after fractional deadline publishes");
publishClock = default;
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 0, 0, 1000), "zero QPC rejected");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 0, -1, 1000), "negative QPC rejected");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 0, 1000, 0), "zero clock frequency rejected");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 0, 1000, -1), "negative clock frequency rejected");
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 0, long.MaxValue - 30, 1000), "large measured QPC preserves integer precision");
Check(!G1PolicyStreamContract.ShouldPublish(ref publishClock, 1, long.MaxValue - 11, 1000), "large QPC deadline is not rounded early");
Check(G1PolicyStreamContract.ShouldPublish(ref publishClock, 1, long.MaxValue - 10, 1000), "large QPC exact deadline publishes");
Check(!G1PolicyStreamContract.WatchdogExpired(false, 1000, 2000, 1000), "startup1s grace");
Check(G1PolicyStreamContract.WatchdogExpired(false, 1000, 2001, 1000), "startup grace bounded");
Check(!G1PolicyStreamContract.WatchdogExpired(true, 1000, 1250, 1000), "established watchdog250ms");
Check(G1PolicyStreamContract.WatchdogExpired(true, 1000, 1251, 1000), "established watchdog expires251ms");
Check(!G1PolicyStreamContract.WatchdogExpired(true, 1299, 1300, 1000), "queued fresh action dispatch refreshes watchdog before check");
var lines = new StringReader("first\r\nsecond\n");
Check(await PolicyRelay.ReadBoundedLine(lines, 10, default) == "first", "buffer first");
Check(await PolicyRelay.ReadBoundedLine(lines, 10, default) == "second", "buffer second retained");
Check(await PolicyRelay.ReadBoundedLine(lines, 10, default) is null, "buffer EOF");
try { await PolicyRelay.ReadBoundedLine(new StringReader("123456\n"), 5, default); throw new Exception("overlength accepted"); } catch (InvalidDataException) { cases++; }
try { await PolicyRelay.ReadBoundedLine(new StringReader("abc"), 5, default); throw new Exception("unterminated accepted"); } catch (InvalidDataException) { cases++; }
Console.WriteLine(JsonSerializer.Serialize(new { passed = cases, publication_counts_per_ten_seconds = publicationCounts, native_game_invocations = 0, global_input_invocations = 0 }));

// The test never connects to a game; connection-only platform API is deliberately unavailable.
internal static class NativeMethods
{
    internal static bool GetNamedPipeServerProcessId(IntPtr handle, out uint pid) => throw new NotSupportedException();
}
