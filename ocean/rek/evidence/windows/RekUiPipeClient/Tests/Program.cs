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
Check(G1PolicyStreamContract.VisualTransportComplete(true, true, false), "visual send completed with pending clear");
Check(!G1PolicyStreamContract.VisualTransportComplete(true, true, true), "visual pending remains owned");
Check(!G1PolicyStreamContract.VisualTransportComplete(true, false, false), "visual clearing alone is not send completion");
Check(!G1PolicyStreamContract.VisualTransportComplete(false, true, false), "nonvisual playback remains separate");
Check(G1PolicyStreamContract.ShouldPublish(10, -1, 1000, 0, 1000), "first rendered source allowed");
Check(!G1PolicyStreamContract.ShouldPublish(10, 10, 1300, 1000, 1000), "same frame never publishes twice even after catch-up");
Check(!G1PolicyStreamContract.ShouldPublish(11, 10, 1019, 1000, 1000), "new frame under20ms throttled");
Check(G1PolicyStreamContract.ShouldPublish(11, 10, 1020, 1000, 1000), "new frame at20ms allowed");
Check(!G1PolicyStreamContract.ShouldPublish(11, 10, 999, 1000, 1000), "reverse publish clock rejected");
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
Console.WriteLine(JsonSerializer.Serialize(new { passed = cases, native_game_invocations = 0, global_input_invocations = 0 }));

// The test never connects to a game; connection-only platform API is deliberately unavailable.
internal static class NativeMethods
{
    internal static bool GetNamedPipeServerProcessId(IntPtr handle, out uint pid) => throw new NotSupportedException();
}
