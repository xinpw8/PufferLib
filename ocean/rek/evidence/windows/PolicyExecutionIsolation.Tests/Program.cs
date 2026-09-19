using System.Text.Json;
using RekUiBridgeAgent;

var checks = 0;
void Check(bool value, string label) { checks++; if (!value) throw new Exception(label); }
AuthenticatedContinuationTests.Run(Check);
var good = new WindowsDesktopFacts(true, true, "D21", "WinSta0", "RekPolicyEval", "Default");
Check(PolicyExecutionIsolationContract.SparkProof == G1HeldInputScheduleContract.RequiredIsolationProof,
    "legacy exact Spark schedule proof is unchanged");
Check(G1HeldInputScheduleContract.RequiredIsolationProof != PolicyExecutionIsolationContract.WindowsProof,
    "Windows policy proof cannot authorize legacy G1 held schedules");
foreach (var api in new[] { false, true })
foreach (var native in new[] { false, true })
foreach (var host in new[] { null, "D21", "d21", "spark-4ae3" })
foreach (var station in new[] { null, "WinSta0", "winsta0", "Service-0" })
foreach (var desktop in new[] { null, "Default", "RekPolicyEval", "RekPolicyEval ", "Other" })
foreach (var input in new[] { null, "Default", "RekPolicyEval", "Winlogon", "Other" })
{
    var facts = new WindowsDesktopFacts(api, native, host, station, desktop, input);
    var expected = facts == good;
    Check((PolicyExecutionIsolationContract.WindowsRejectReason(facts) is null) == expected, "all native desktop facts necessary");
    Check(NativeWindowsDesktopIsolation.TryVerify(() => facts, out var proof, out var reason) == expected,
        "reader facts fail closed");
    Check(expected ? proof == PolicyExecutionIsolationContract.WindowsProof && reason == "" : proof is null && reason != "",
        "proof never emitted on failure");
}
foreach (var proof in new string?[] { null, "", "spark-x98", "DISPLAY=:98", "windows_native=1",
    PolicyExecutionIsolationContract.WindowsProof + ";extra=1",
    PolicyExecutionIsolationContract.WindowsProof.Replace("RekPolicyEval", "Default"),
    PolicyExecutionIsolationContract.WindowsProof.Replace("Default", "RekPolicyEval"),
    PolicyExecutionIsolationContract.WindowsProof.Replace("D21", "d21"),
    " " + PolicyExecutionIsolationContract.SparkProof })
    Check(!PolicyExecutionIsolationContract.IsSupportedProof(proof), "noncanonical proof rejected");
Check(PolicyExecutionIsolationContract.IsSupportedProof(PolicyExecutionIsolationContract.WindowsProof), "native proof supported");
Check(PolicyExecutionIsolationContract.IsSupportedProof(PolicyExecutionIsolationContract.SparkProof), "Spark proof supported");
var queue = new Queue<WindowsDesktopFacts>(new[] { good, good with { Desktop = "Default" },
    good with { InputDesktop = "RekPolicyEval" }, good with { ApiSucceeded = false }, good });
foreach (var expected in new[] { true, false, false, false, true })
    Check(NativeWindowsDesktopIsolation.TryVerify(() => queue.Dequeue(), out _, out _) == expected,
        "every verification rereads changing runtime facts");
Check(!NativeWindowsDesktopIsolation.TryVerify(() => throw new InvalidOperationException(), out var failedProof, out _)
    && failedProof is null, "exception cannot preserve prior valid proof");

var account = new WindowsPolicyAccountFacts(true, true, "moogleod", "moogleod", 123);
Check(PolicyExecutionIsolationContract.AccountRejectReason(account, 0, false) is null, "Home evidence may acquire pin");
Check(PolicyExecutionIsolationContract.AccountRejectReason(account, 0, true) is not null, "other mutations require lease pin");
Check(PolicyExecutionIsolationContract.AccountRejectReason(account, 123, true) is null, "Home pinned account accepted");
Check(PolicyExecutionIsolationContract.AccountRejectReason(account with { AtHome = false, HomeDisplayName = null }, 123, true) is null,
    "same token-free context pins account across private scene");
foreach (var invalid in new[] { account with { Available = false }, account with { ContextIdentity = 0 },
    account with { ContextIdentity = 124 }, account with { FighterName = null }, account with { FighterName = "scabnft" },
    account with { FighterName = "Moogleod" }, account with { HomeDisplayName = "scabnft" },
    account with { HomeDisplayName = null } })
    Check(PolicyExecutionIsolationContract.AccountRejectReason(invalid, 123, true) is not null, "account identity change or unknown rejected");
Check(PolicyExecutionIsolationContract.AccountRejectReason(account with { AtHome = false }, 0, false) is not null,
    "cannot acquire first pin outside Home");
foreach (var command in new[] { "AcquireExclusiveControl", "ReleaseExclusiveControl", "NavigateFreePlay", "EnterSolo",
    "ReadyPrivateAiSession", "StartG1PolicyRound", "ExitLostG1PolicySession", "StartG1PolicyStream",
    "StartG1PolicyStreamAnyAi", "StopG1PolicyStream" })
    Check(PolicyExecutionIsolationContract.WindowsCommandAllowed(command), "bounded private G1 command allowed");
foreach (var command in new[] { "ConfirmLoggedIn", "StartRound", "ExitLostPrivateSession", "ExitUnexpectedPrivateAiSession",
    "StartMeasuredSchedule", "StopMeasuredSchedule", "StartSingleMotionTrial", "StartContinuousBotController",
    "StopContinuousBotController", "StartAttackZoneTrial", "StopAttackZoneTrial", "StartG1HeldInputSchedule",
    "StopG1HeldInputSchedule", "SendInput", "NavigateFreePlay " })
    Check(!PolicyExecutionIsolationContract.WindowsCommandAllowed(command), "authentication and legacy commands remain rejected");

// A passive real runtime negative test. No window is created and no process is launched.
var actual = NativeWindowsDesktopIsolation.Read();
var actualReason = PolicyExecutionIsolationContract.WindowsRejectReason(actual);
Check(actualReason is not null, "test runner must not be authorized on the ordinary desktop");
if (OperatingSystem.IsWindows())
    Check(actual.ApiSucceeded && actual.NativeWindows && actual.Desktop == "Default" && actual.InputDesktop == "Default",
        "real native APIs successfully identify Default, which is rejected");
Console.WriteLine(JsonSerializer.Serialize(new { passed = checks, actual_runtime = actual, actual_rejection = actualReason,
    native_game_invocations = 0, global_input_invocations = 0, desktop_switch_invocations = 0 }));
