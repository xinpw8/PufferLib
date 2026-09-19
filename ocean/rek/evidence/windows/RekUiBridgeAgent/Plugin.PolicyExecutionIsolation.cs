using REKApp;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private long _windowsPolicyAccountContext;
    private IntPtr _windowsPolicyRevokedInput;

    private static bool TryVerifyPolicyIsolatedSession(out string? proof, out string reason)
    {
        if (TryVerifyExplicitIsolatedSession(out proof)) { reason = ""; return true; }
        return NativeWindowsDesktopIsolation.TryVerify(out proof, out reason);
    }

    private static WindowsPolicyAccountFacts ReadWindowsPolicyAccountFacts()
    {
        try
        {
            var context = GameContext.Instance ?? UnityEngine.Object.FindFirstObjectByType<GameContext>();
            var lobby = UnityEngine.Object.FindFirstObjectByType<LobbyShellController>();
            return new(context is not null, lobby?.CurrentScreen == LobbyShellController.Screen.Home,
                lobby?.homeScreen?.userNameLabel?.text, context?.FighterName,
                context is null ? 0 : NativePointer(context).ToInt64());
        }
        catch { return default; }
    }

    private static bool RequirePolicyBackgroundControl(out string reason)
    {
        if (Instance?._windowsPolicyRevokedInput != IntPtr.Zero && Instance is not null)
        { reason = "windows_policy_surface_revoked_restart_required"; return false; }
        if (!TryVerifyPolicyIsolatedSession(out var proof, out reason)) return false;
        if (proof == PolicyExecutionIsolationContract.SparkProof) return true;
        reason = PolicyExecutionIsolationContract.AccountRejectReason(ReadWindowsPolicyAccountFacts(),
            Instance?._windowsPolicyAccountContext ?? 0, requirePin: true) ?? "";
        return reason.Length == 0;
    }

    private bool RequirePolicyCommandSurface(BridgeCommand command, out string reason)
    {
        if (_windowsPolicyRevokedInput != IntPtr.Zero)
        { reason = "windows_policy_surface_revoked_restart_required"; return false; }
        if (!TryVerifyPolicyIsolatedSession(out var proof, out reason)) return false;
        if (proof == PolicyExecutionIsolationContract.SparkProof) return true;
        if (!PolicyExecutionIsolationContract.WindowsCommandAllowed(command.ToString()))
        { reason = "windows_isolated_surface_allows_only_private_g1_policy_commands"; return false; }
        var account = ReadWindowsPolicyAccountFacts();
        reason = PolicyExecutionIsolationContract.AccountRejectReason(account, _windowsPolicyAccountContext,
            requirePin: command != BridgeCommand.AcquireExclusiveControl) ?? "";
        if (reason.Length != 0) return false;
        if (command == BridgeCommand.AcquireExclusiveControl && _windowsPolicyAccountContext == 0)
            _windowsPolicyAccountContext = account.ContextIdentity;
        return true;
    }

    private object WindowsPolicyAccountEvidence()
    {
        var account = ReadWindowsPolicyAccountFacts();
        var reason = PolicyExecutionIsolationContract.AccountRejectReason(account, _windowsPolicyAccountContext,
            requirePin: false);
        return new {
            required_display_name = PolicyExecutionIsolationContract.WindowsAccount,
            context_available = account.Available,
            context_fighter_name = SafeString(account.FighterName, 128),
            at_home = account.AtHome,
            home_display_matches = account.HomeDisplayName == PolicyExecutionIsolationContract.WindowsAccount,
            pinned = _windowsPolicyAccountContext != 0,
            context_matches_pin = _windowsPolicyAccountContext != 0 && account.ContextIdentity == _windowsPolicyAccountContext,
            allowed = reason is null,
            reason = reason ?? "moogleod_token_free_context_proven" };
    }
}
