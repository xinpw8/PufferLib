using REKApp;
using UnityEngine.UIElements;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private bool _nativeAuthenticatedContinuationAttempted;

    private AuthenticatedContinuationFacts ReadAuthenticatedContinuationFacts(out LoginScreenController? login)
    {
        login = null;
        try
        {
            var lobby = UnityEngine.Object.FindFirstObjectByType<LobbyShellController>();
            var context = GameContext.Instance ?? UnityEngine.Object.FindFirstObjectByType<GameContext>();
            login = lobby?.loginScreen;
            var idle = !_nativeAuthenticatedContinuationAttempted && _leaseConnectionId == 0 &&
                _windowsPolicyAccountContext == 0 && _windowsPolicyRevokedInput == IntPtr.Zero &&
                !_g1PolicyRunning && !_scheduleRunning && !_singleMotionTrialRunning &&
                !_continuousControllerRunning && !_g1HeldScheduleRunning &&
                !_attackZoneTrialRunning && !_attackZoneRecoveryOnlyRunning && _freshRoundArm is null &&
                !AnyConnectedSession();
            return new(login is not null && context is not null, idle,
                lobby?.CurrentScreen == LobbyShellController.Screen.Login, login?.isLoggedIn == true,
                ContinuationElementShown(login?.loggedInPane),
                login?.loggedOutPane?.resolvedStyle.display == DisplayStyle.None,
                login?.attemptingPane?.resolvedStyle.display == DisplayStyle.None,
                ContinuationElementShown(login?.letsGoButton), login?.letsGoButton?.enabledInHierarchy == true,
                context?.FighterName, login?.fighterNameLabel?.text,
                context is null ? 0 : NativePointer(context).ToInt64(),
                login is null ? 0 : NativePointer(login).ToInt64());
        }
        catch { login = null; return default; }
    }

    private static bool ContinuationElementShown(VisualElement? element)
    {
        if (element?.panel is null || !element.visible) return false;
        var bounds = element.worldBound;
        if (!float.IsFinite(bounds.width) || !float.IsFinite(bounds.height) || bounds.width <= 0 || bounds.height <= 0)
            return false;
        for (var parent = element; parent is not null; parent = parent.parent)
        {
            var style = parent.resolvedStyle;
            if (style.display != DisplayStyle.Flex || style.visibility != Visibility.Visible ||
                !float.IsFinite(style.opacity) || style.opacity <= 0) return false;
        }
        return true;
    }

    private CommandResult ContinueAlreadyAuthenticatedNativeLogin()
    {
        LoginScreenController? login = null;
        var success = AuthenticatedContinuationContract.TryContinue(NativeWindowsDesktopIsolation.Read,
            () => ReadAuthenticatedContinuationFacts(out login),
            () => {
                _nativeAuthenticatedContinuationAttempted = true;
                // Recovered OnLetsGoPressed -> LobbyShellController.HandleLetsGo
                // -> ShowScreen(Home). No auth APIs or input events are invoked.
                login!.OnLetsGoClicked();
            }, ReadWindowsPolicyAccountFacts, out _, out var reason);
        return success ? CommandResult.AppliedResult(reason) : CommandResult.Rejected(reason);
    }

    private object ReadAuthenticatedContinuationEvidence()
    {
        var facts = ReadAuthenticatedContinuationFacts(out _);
        var isolated = NativeWindowsDesktopIsolation.TryVerify(out _, out var surfaceReason);
        var reason = isolated ? AuthenticatedContinuationContract.RejectReason(facts) : surfaceReason;
        return new {
            available = facts.Available, current_screen_is_login = facts.AtLogin,
            already_logged_in = facts.IsLoggedIn, logged_in_pane_shown = facts.LoggedInShown,
            logged_out_pane_hidden = facts.LoggedOutHidden, attempting_pane_hidden = facts.AttemptingHidden,
            fighter_name_label = SafeString(facts.LabelFighterName, 128),
            context_fighter_name = SafeString(facts.ContextFighterName, 128),
            lets_go_shown = facts.LetsGoShown, lets_go_enabled = facts.LetsGoEnabled,
            prelease_idle = facts.PreLeaseIdle, continuation_attempted = _nativeAuthenticatedContinuationAttempted,
            native_authenticated_continuation_allowed = reason is null,
            reason = reason ?? "already_authenticated_moogleod_continuation_ready",
            authentication_available = false };
    }
}
