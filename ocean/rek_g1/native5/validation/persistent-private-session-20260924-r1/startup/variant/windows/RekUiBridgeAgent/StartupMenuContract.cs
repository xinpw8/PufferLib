namespace RekUiBridgeAgent;

internal readonly record struct IntroSkipFacts(bool IsolatedSpark, bool IdleControls, bool NoNetwork,
    bool AtIntro, bool ControllerAvailable, bool IntroActive, bool Finished, bool SkipShown, bool SkipEnabled);

internal readonly record struct UnsupportedPairingExitFacts(bool IsolatedSpark, bool IdleControls,
    bool PrivateBotOneNoHumanScope, bool GameMenuAvailable, bool MatchingSlots, bool VisualOnlyPair,
    bool LocalSemanticG1, bool OpponentSemanticT800, bool LocalExactG1Bones,
    bool OpponentExactT800Bones, bool MixedPairReason);
internal readonly record struct HomeForfeitConfirmationFacts(bool TargetHome, bool MenuOpen,
    bool AtForfeitPane, bool ButtonShown, bool ButtonEnabled);

internal static class StartupMenuContract
{
    internal static bool CanConfirmHomeForfeit(HomeForfeitConfirmationFacts f) =>
        f.TargetHome && f.MenuOpen && f.AtForfeitPane && f.ButtonShown && f.ButtonEnabled;

    internal static string? IntroRejectReason(IntroSkipFacts f)
    {
        if (!f.IsolatedSpark) return "intro_skip_requires_isolated_spark";
        if (!f.IdleControls) return "controlled_run_already_active";
        if (!f.NoNetwork) return "network_session_already_connected";
        if (!f.AtIntro) return "intro_screen_not_observed";
        if (!f.ControllerAvailable || !f.IntroActive || f.Finished) return "active_intro_controller_not_observed";
        if (!f.SkipShown || !f.SkipEnabled) return "intro_skip_control_not_shown_and_enabled";
        return null;
    }

    internal static string? UnsupportedExitRejectReason(UnsupportedPairingExitFacts f)
    {
        if (!f.IsolatedSpark) return "unsupported_pair_exit_requires_isolated_spark";
        if (!f.IdleControls) return "controlled_run_already_active";
        if (!f.PrivateBotOneNoHumanScope) return "private_bot1_no_human_scope_not_proven";
        if (!f.GameMenuAvailable || !f.MatchingSlots || !f.VisualOnlyPair) return "bound_client_visual_pair_not_proven";
        if (!f.LocalSemanticG1 || !f.OpponentSemanticT800 || !f.LocalExactG1Bones ||
            !f.OpponentExactT800Bones || !f.MixedPairReason) return "exact_unsupported_local_g1_opponent_t800_not_proven";
        return null;
    }
}
