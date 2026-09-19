using RekUiBridgeAgent;

internal static class AuthenticatedContinuationTests
{
    internal static void Run(Action<bool, string> check)
    {
        var desktop = new WindowsDesktopFacts(true, true, "D21", "WinSta0", "RekPolicyEval", "Default");
        var login = new AuthenticatedContinuationFacts(true, true, true, true, true, true, true, true, true,
            "moogleod", "moogleod", 123, 456);
        var home = new WindowsPolicyAccountFacts(true, true, "moogleod", "moogleod", 123);
        for (var bits = 0; bits < 512; bits++)
        {
            var facts = login with { Available = (bits & 1) != 0, PreLeaseIdle = (bits & 2) != 0,
                AtLogin = (bits & 4) != 0, IsLoggedIn = (bits & 8) != 0, LoggedInShown = (bits & 16) != 0,
                LoggedOutHidden = (bits & 32) != 0, AttemptingHidden = (bits & 64) != 0,
                LetsGoShown = (bits & 128) != 0, LetsGoEnabled = (bits & 256) != 0 };
            check((AuthenticatedContinuationContract.RejectReason(facts) is null) == (bits == 511),
                "every authenticated continuation UI precondition is necessary");
            var calls = 0;
            var success = AuthenticatedContinuationContract.TryContinue(() => desktop, () => facts, () => calls++,
                () => home, out var invoked, out _);
            check(success == (bits == 511) && invoked == success && calls == (success ? 1 : 0),
                "invalid UI facts cannot dispatch continuation");
        }
        foreach (var bad in new[] { login with { ContextIdentity = 0 }, login with { LoginIdentity = 0 },
            login with { ContextFighterName = null }, login with { ContextFighterName = "scabnft" },
            login with { LabelFighterName = null }, login with { LabelFighterName = "scabnft" },
            login with { LabelFighterName = "moogleod@example.invalid" }, login with { LabelFighterName = "Moogleod" } })
            check(AuthenticatedContinuationContract.RejectReason(bad) is not null, "absent or mismatched continuation identity rejected");

        var trace = new List<string>();
        var result = AuthenticatedContinuationContract.TryContinue(
            () => { trace.Add("surface"); return desktop; },
            () => { trace.Add("login"); return login; },
            () => trace.Add("continue"), () => { trace.Add("home"); return home; }, out var wasInvoked, out _);
        check(result && wasInvoked && string.Join(",", trace) == "surface,login,surface,login,surface,continue,surface,home",
            "native proof before reads and immediately before sole callback, then fresh Home postcondition");

        foreach (var bad in new[] { desktop with { ApiSucceeded = false }, desktop with { NativeWindows = false },
            desktop with { Desktop = "Default" }, desktop with { InputDesktop = "RekPolicyEval" },
            desktop with { Host = "spark-4ae3" }, desktop with { Station = "Service" } })
        for (var failAt = 1; failAt <= 3; failAt++)
        {
            var reads = 0; var calls = 0;
            var expectedFailureRead = failAt;
            result = AuthenticatedContinuationContract.TryContinue(
                () => ++reads == expectedFailureRead ? bad : desktop, () => login, () => calls++,
                () => home, out wasInvoked, out _);
            check(!result && !wasInvoked && calls == 0 && reads == expectedFailureRead,
                "desktop change at any pre-dispatch boundary blocks callback");
        }
        foreach (var changed in new[] { login with { IsLoggedIn = false }, login with { AtLogin = false },
            login with { ContextIdentity = 124 }, login with { LoginIdentity = 457 },
            login with { LabelFighterName = "scabnft" }, login with { LetsGoEnabled = false } })
        {
            var reads = 0; var calls = 0;
            result = AuthenticatedContinuationContract.TryContinue(() => desktop, () => ++reads == 1 ? login : changed,
                () => calls++, () => home, out wasInvoked, out _);
            check(!result && !wasInvoked && calls == 0, "Login change between reads cannot use stale authorization");
        }
        foreach (var bad in new[] { home with { AtHome = false }, home with { Available = false },
            home with { ContextIdentity = 124 }, home with { FighterName = "scabnft" },
            home with { HomeDisplayName = "moogleod@example.invalid" }, home with { HomeDisplayName = null } })
        {
            var calls = 0;
            result = AuthenticatedContinuationContract.TryContinue(() => desktop, () => login,
                () => calls++, () => bad, out wasInvoked, out _);
            check(!result && wasInvoked && calls == 1, "callback return cannot fabricate successful Home or account evidence");
        }
        var attempted = false; var oneShotCalls = 0;
        for (var attempt = 0; attempt < 2; attempt++)
        {
            result = AuthenticatedContinuationContract.TryContinue(() => desktop,
                () => login with { PreLeaseIdle = !attempted },
                () => { attempted = true; oneShotCalls++; }, () => home, out wasInvoked, out _);
            check(result == (attempt == 0) && wasInvoked == result && oneShotCalls == 1,
                "continuation is one-shot and never grants a lease");
        }
        var eventCalls = 0;
        result = AuthenticatedContinuationContract.TryContinue(() => throw new Exception(), () => login,
            () => eventCalls++, () => home, out wasInvoked, out _);
        check(!result && !wasInvoked && eventCalls == 0, "native API exception fails before dispatch");
        result = AuthenticatedContinuationContract.TryContinue(() => desktop, () => throw new Exception(),
            () => eventCalls++, () => home, out wasInvoked, out _);
        check(!result && !wasInvoked && eventCalls == 0, "UI read exception fails before dispatch");
        result = AuthenticatedContinuationContract.TryContinue(() => desktop, () => login,
            () => { eventCalls++; throw new Exception(); }, () => home, out wasInvoked, out _);
        check(!result && wasInvoked && eventCalls == 1, "callback exception is reported as attempted, never retried");
        check(PolicyExecutionIsolationContract.AccountRejectReason(home with { AtHome = false }, 0, false) is not null,
            "post-login continuation never broadens the initial Home lease proof");
        check(!PolicyExecutionIsolationContract.WindowsCommandAllowed("ConfirmLoggedIn"),
            "continuation is separate from normal native Windows leased command surface");
    }
}
