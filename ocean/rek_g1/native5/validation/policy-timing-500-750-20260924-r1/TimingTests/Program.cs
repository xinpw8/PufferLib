using System.Text.Json;
using RekUiBridgeAgent;

var checks = 0;
void Check(bool value, string label) { checks++; if (!value) throw new Exception(label); }
var round = new string('a', 64);
var otherRound = new string('b', 64);
var action = new G1PolicyAction(round, 10, 6);
foreach (var hz in new long[] { 1000, 10_000_000 })
{
    var produced = hz;
    long At(int ms) => produced + hz * ms / 1000;
    string? RejectAt(int ms) => G1PolicyStreamContract.RejectReason(action, round, 9, produced, At(ms), hz);
    Check(RejectAt(250) is null, "former freshness limit remains accepted");
    Check(RejectAt(500) is null, "exactly500ms accepted");
    Check(RejectAt(501) == "stale_observation", "501ms rejected");
    Check(!G1PolicyStreamContract.WatchdogExpired(true, produced, At(750), hz), "750ms established accepted");
    Check(G1PolicyStreamContract.WatchdogExpired(true, produced, At(751), hz), "751ms established expired");
    Check(!G1PolicyStreamContract.WatchdogExpired(false, produced, At(1000), hz), "1000ms startup accepted");
    Check(G1PolicyStreamContract.WatchdogExpired(false, produced, At(1001), hz), "1001ms startup expired");
    Check(G1PolicyStreamContract.RejectReason(action, otherRound, 9, produced, At(1), hz) == "round_identity_mismatch", "fresh wrong round rejected");
    Check(G1PolicyStreamContract.RejectReason(action, round, 10, produced, At(1), hz) == "observation_already_consumed", "fresh duplicate rejected");
    Check(G1PolicyStreamContract.RejectReason(action, round, 9, 0, At(1), hz) == "observation_unknown_or_clock_invalid", "unknown observation rejected");
    Check(G1PolicyStreamContract.RejectReason(action, round, 9, produced, produced - 1, hz) == "observation_unknown_or_clock_invalid", "reverse source clock rejected");
    Check(G1PolicyStreamContract.WatchdogExpired(true, produced, produced - 1, hz), "reverse watchdog clock expires");

    // CPU model of the unchanged ApplyG1PolicyAction accepted branch. A rejected
    // prediction does not mutate consumed sequence or the last accepted clock.
    var lastAccepted = produced;
    var consumed = 9L;
    foreach (var ms in new[] { 501, 600, 750, 751 })
    {
        var reason = G1PolicyStreamContract.RejectReason(action, round, consumed, produced, At(ms), hz);
        if (reason is null) { consumed = action.ObservationSequence; lastAccepted = At(ms); }
        Check(reason == "stale_observation" && consumed == 9 && lastAccepted == produced, "stale rejection cannot refresh clock");
    }
    Check(G1PolicyStreamContract.WatchdogExpired(true, lastAccepted, At(751), hz), "repeated stale requests cannot extend watchdog");
    var fresh = new G1PolicyAction(round, 11, 7);
    var freshProduced = At(600);
    var freshNow = At(700);
    var freshReason = G1PolicyStreamContract.RejectReason(fresh, round, consumed, freshProduced, freshNow, hz);
    if (freshReason is null) { consumed = fresh.ObservationSequence; lastAccepted = freshNow; }
    Check(freshReason is null && consumed == 11 && lastAccepted == freshNow, "subsequent fresh source resumes before expiry");
    Check(!G1PolicyStreamContract.WatchdogExpired(true, lastAccepted, At(701), hz), "fresh dispatch refreshes before same-frame watchdog");
}
// Exact measured s1101 end: last source, last accepted action, rejected ACK,
// and watchdog check. This tests timing eligibility only, not game execution.
Check(G1PolicyStreamContract.RejectReason(action, round, 9, 9307064245042, 9307066868307, 10_000_000) is null, "recorded262ms action is eligible");
Check(!G1PolicyStreamContract.WatchdogExpired(true, 9307064214604, 9307066999896, 10_000_000), "recorded278ms gap no longer expires");
Console.WriteLine(JsonSerializer.Serialize(new { passed = checks, scope = "pure_contract_and_accepted_clock_model", game_invocations = 0, bridge_connections = 0 }));
