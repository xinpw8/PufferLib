namespace RekUiBridgeAgent;

// Policy execution only. Legacy schedules retain their exact Spark proof.
public static class PolicyExecutionIsolationContract
{
    public const string SparkProof = "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98";
    public const string WindowsProof = "windows_native=1;station=WinSta0;desktop=RekPolicyEval;input_desktop=Default;host=D21";
    public const string WindowsAccount = "moogleod";

    public static bool IsSupportedProof(string? proof) => proof is SparkProof or WindowsProof;
    public static string? WindowsRejectReason(WindowsDesktopFacts facts)
    {
        if (!facts.ApiSucceeded) return "windows_desktop_api_failure";
        if (!facts.NativeWindows) return "native_windows_required_wine_rejected";
        if (facts.Host != "D21") return "windows_policy_host_mismatch";
        if (facts.Station != "WinSta0") return "windows_policy_station_mismatch";
        if (facts.Desktop != "RekPolicyEval") return "windows_policy_desktop_mismatch";
        if (facts.InputDesktop != "Default") return "windows_policy_input_desktop_mismatch";
        if (facts.Desktop == facts.InputDesktop) return "windows_policy_desktop_receives_input";
        return null;
    }

    public static bool WindowsCommandAllowed(string command) => command is
        "AcquireExclusiveControl" or "ReleaseExclusiveControl" or "NavigateFreePlay" or "EnterSolo" or
        "ReadyPrivateAiSession" or "StartG1PolicyRound" or "ExitLostG1PolicySession" or
        "StartG1PolicyStream" or "StartG1PolicyStreamAnyAi" or "StopG1PolicyStream";

    public static string? AccountRejectReason(WindowsPolicyAccountFacts facts, long pinnedContext,
        bool requirePin)
    {
        if (!facts.Available || facts.ContextIdentity == 0) return "windows_policy_account_context_unavailable";
        if (facts.FighterName != WindowsAccount) return "windows_policy_context_fighter_name_not_moogleod";
        if (pinnedContext != 0 && facts.ContextIdentity != pinnedContext)
            return "windows_policy_account_context_changed";
        if (facts.AtHome && facts.HomeDisplayName != WindowsAccount)
            return "windows_policy_home_account_not_moogleod";
        if (pinnedContext == 0)
        {
            if (!facts.AtHome || facts.HomeDisplayName != WindowsAccount)
                return "windows_policy_initial_home_account_not_proven";
            if (requirePin) return "windows_policy_account_lease_pin_required";
        }
        return null;
    }
}

public readonly record struct WindowsDesktopFacts(bool ApiSucceeded, bool NativeWindows,
    string? Host, string? Station, string? Desktop, string? InputDesktop);
public readonly record struct WindowsPolicyAccountFacts(bool Available, bool AtHome,
    string? HomeDisplayName, string? FighterName, long ContextIdentity);
