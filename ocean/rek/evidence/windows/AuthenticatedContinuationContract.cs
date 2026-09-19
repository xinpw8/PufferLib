namespace RekUiBridgeAgent;

// Post-login UI continuation only. No authentication client or credential data.
public static class AuthenticatedContinuationContract
{
    public static string? RejectReason(AuthenticatedContinuationFacts facts)
    {
        if (!facts.Available) return "authenticated_continuation_ui_unavailable";
        if (!facts.PreLeaseIdle) return "authenticated_continuation_requires_unused_prelease_idle_state";
        if (!facts.AtLogin) return "authenticated_continuation_requires_login_screen";
        if (!facts.IsLoggedIn) return "authenticated_continuation_requires_already_logged_in";
        if (!facts.LoggedInShown || !facts.LoggedOutHidden || !facts.AttemptingHidden)
            return "authenticated_continuation_panes_not_proven";
        if (facts.ContextIdentity == 0 || facts.LoginIdentity == 0 ||
            facts.ContextFighterName != PolicyExecutionIsolationContract.WindowsAccount ||
            facts.LabelFighterName != PolicyExecutionIsolationContract.WindowsAccount)
            return "authenticated_continuation_moogleod_identity_not_proven";
        if (!facts.LetsGoShown || !facts.LetsGoEnabled)
            return "authenticated_continuation_lets_go_not_visible_enabled";
        return null;
    }

    public static bool TryContinue(Func<WindowsDesktopFacts> readDesktop,
        Func<AuthenticatedContinuationFacts> readLogin, Action continueLoggedIn,
        Func<WindowsPolicyAccountFacts> readHome, out bool invoked, out string reason)
    {
        invoked = false;
        try
        {
            reason = PolicyExecutionIsolationContract.WindowsRejectReason(readDesktop()) ?? "";
            if (reason.Length != 0) return false;
            var before = readLogin();
            reason = RejectReason(before) ?? "";
            if (reason.Length != 0) return false;
            // Recheck both the native surface and current UI immediately before
            // the sole callback. A cached valid state cannot authorize it.
            reason = PolicyExecutionIsolationContract.WindowsRejectReason(readDesktop()) ?? "";
            if (reason.Length != 0) return false;
            var current = readLogin();
            reason = RejectReason(current) ?? "";
            if (reason.Length != 0) return false;
            if (current.ContextIdentity != before.ContextIdentity || current.LoginIdentity != before.LoginIdentity)
            { reason = "authenticated_continuation_identity_changed"; return false; }
            reason = PolicyExecutionIsolationContract.WindowsRejectReason(readDesktop()) ?? "";
            if (reason.Length != 0) return false;
            invoked = true;
            continueLoggedIn();
            reason = PolicyExecutionIsolationContract.WindowsRejectReason(readDesktop()) ?? "";
            if (reason.Length != 0) return false;
            var home = readHome();
            reason = PolicyExecutionIsolationContract.AccountRejectReason(home, before.ContextIdentity, requirePin: false) ?? "";
            if (reason.Length != 0) return false;
            if (!home.AtHome || home.HomeDisplayName != PolicyExecutionIsolationContract.WindowsAccount)
            { reason = "authenticated_continuation_home_postcondition_not_observed"; return false; }
            reason = "already_authenticated_moogleod_home_observed";
            return true;
        }
        catch
        {
            reason = invoked ? "authenticated_continuation_callback_or_postcondition_failed" : "authenticated_continuation_probe_failed";
            return false;
        }
    }
}

public readonly record struct AuthenticatedContinuationFacts(bool Available, bool PreLeaseIdle,
    bool AtLogin, bool IsLoggedIn, bool LoggedInShown, bool LoggedOutHidden, bool AttemptingHidden,
    bool LetsGoShown, bool LetsGoEnabled, string? ContextFighterName, string? LabelFighterName,
    long ContextIdentity, long LoginIdentity);
