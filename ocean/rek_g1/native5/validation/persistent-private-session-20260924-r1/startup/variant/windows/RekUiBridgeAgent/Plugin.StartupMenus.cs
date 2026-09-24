using REKApp;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private bool StartupControlsIdle() => !_g1PolicyRunning && !_scheduleRunning && !_singleMotionTrialRunning &&
        !_continuousControllerRunning && !_g1HeldScheduleRunning && !_attackZoneTrialRunning &&
        !_attackZoneRecoveryOnlyRunning && _freshRoundArm is null;

    private IntroSkipFacts ReadIntroSkipFacts(LobbyShellController? lobby, out IntroScreenController? intro)
    {
        intro = lobby?.introScreen;
        return new(TryVerifyExplicitIsolatedSession(out _), StartupControlsIdle(), !AnyConnectedSession(),
            lobby?.CurrentScreen == LobbyShellController.Screen.Intro, intro is not null,
            lobby?.introActive == true, intro?.finished == true,
            ContinuationElementShown(intro?.skipButton), intro?.skipButton?.enabledInHierarchy == true);
    }

    private object? ReadIntroSkipEvidence(LobbyShellController? lobby)
    {
        if (lobby?.CurrentScreen != LobbyShellController.Screen.Intro) return null;
        try
        {
            var facts = ReadIntroSkipFacts(lobby, out _);
            var reason = StartupMenuContract.IntroRejectReason(facts);
            return new { available = facts.ControllerAvailable, active = facts.IntroActive,
                finished = facts.Finished, skip_shown = facts.SkipShown, skip_enabled = facts.SkipEnabled,
                skip_allowed = reason is null, reason = reason ?? "observed_native_intro_skip_ready" };
        }
        catch { return new { skip_allowed = false, reason = "intro_skip_probe_failed" }; }
    }

    private CommandResult SkipObservedIntro()
    {
        if (!RequireBackgroundControl(out var surfaceReason)) return CommandResult.Rejected(surfaceReason);
        var lobby = UnityEngine.Object.FindFirstObjectByType<LobbyShellController>();
        var facts = ReadIntroSkipFacts(lobby, out var intro);
        var reason = StartupMenuContract.IntroRejectReason(facts);
        if (reason is not null) return CommandResult.Rejected(reason);
        // Invoke the native Skip button callback. No authentication API or input event.
        intro!.OnSkipClicked();
        return intro.finished || lobby!.CurrentScreen != LobbyShellController.Screen.Intro
            ? CommandResult.AppliedResult("native_intro_skip_finished_observed")
            : CommandResult.RequestIssued("native_intro_skip_request_issued");
    }

    private bool TryGetUnsupportedPairingExitScope(out PrivateAiContext scope, out string reason)
    {
        scope = null!;
        if (!TryVerifyExplicitIsolatedSession(out _)) { reason = "unsupported_pair_exit_requires_isolated_spark"; return false; }
        if (!StartupControlsIdle()) { reason = "controlled_run_already_active"; return false; }
        if (!TryGetPrivateAiContext(false, out scope, out reason, bindRuntimeSession: false)) return false;
        var pairing = ReadMeasuredPairing(scope.GameMenu);
        var v = pairing.Validation;
        var fighters = scope.Coordinator.Fighters;
        var visual = fighters is not null && fighters.Length == 2 && fighters[0] is not null &&
            fighters[1] is not null && fighters[0].IsVisualOnly && fighters[1].IsVisualOnly;
        var facts = new UnsupportedPairingExitFacts(true, true, true, scope.GameMenu is not null,
            pairing.LocalSlot == scope.LocalSlot && pairing.OpponentSlot == scope.OpponentSlot, visual,
            v.LocalSemanticG1, v.OpponentSemanticT800, v.LocalExactG1BoneSignature,
            v.OpponentExactT800BoneSignature, v.Reason == "mixed_supported_runtime_models_rejected");
        reason = StartupMenuContract.UnsupportedExitRejectReason(facts) ?? "";
        return reason.Length == 0;
    }

    private static bool UnsupportedForfeitConfirmationShown(GameMenuController menu) =>
        StartupMenuContract.CanConfirmHomeForfeit(new(
            menu.pendingExitScreen == GameContext.LobbyScreen.Home, menu.IsMenuOpen,
            menu.menuView?.CurrentPane == GameMenuView.Pane.Forfeit,
            ContinuationElementShown(menu.menuView?.forfeitConfirmButton),
            menu.menuView?.forfeitConfirmButton?.enabledInHierarchy == true));

    private object? ReadUnsupportedPairingExitEvidence()
    {
        if (_g1PolicyRunning) return null;
        try
        {
            if (!TryGetUnsupportedPairingExitScope(out var scope, out _)) return null;
            return new { available = true, exact_local_g1_opponent_t800 = true, private_bot1_no_human = true,
                confirmation_required = UnsupportedForfeitConfirmationShown(scope.GameMenu!),
                confirmation_target = "Home", native_exit_only = true, process_restart = false };
        }
        catch { return null; }
    }

    private CommandResult ExitUnsupportedPrivateAiPairing()
    {
        if (!RequireBackgroundControl(out var surfaceReason)) return CommandResult.Rejected(surfaceReason);
        if (!TryGetUnsupportedPairingExitScope(out var scope, out var reason)) return CommandResult.Rejected(reason);
        var menu = scope.GameMenu!;
        if (menu.menuView?.CurrentPane == GameMenuView.Pane.Forfeit)
        {
            if (!UnsupportedForfeitConfirmationShown(menu))
                return CommandResult.Rejected("native_home_forfeit_confirmation_not_shown_and_enabled");
            menu.menuView.OnForfeitConfirmInternal();
            return CommandResult.RequestIssued("unsupported_private_pair_native_home_forfeit_confirmed");
        }
        // The native Home callback normally originates from an open menu.
        // ShowForfeitPane changes the pane but does not open the menu root.
        if (!menu.IsMenuOpen) menu.Open();
        menu.HandleExitHomeRequested();
        return CommandResult.RequestIssued("unsupported_private_pair_native_home_exit_requested");
    }
}
