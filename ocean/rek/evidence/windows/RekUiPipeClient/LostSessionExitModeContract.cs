internal sealed record LostSessionExitDecision(bool Allowed, string Reason);

internal static class LostSessionExitModeContract
{
    internal static LostSessionExitDecision Evaluate(
        bool exactPrivateBotOneProven,
        bool roundActive,
        bool postFightPrompt,
        bool postFightWinner,
        bool scheduleRunning,
        bool singleTrialRunning,
        bool continuousControllerRunning,
        bool attackZoneTrialRunning,
        bool attackZoneRecoveryOnlyRunning)
    {
        if (!exactPrivateBotOneProven)
            return new(false, "exact_private_bot1_session_not_proven");
        if (roundActive)
            return new(false, "round_still_active");
        if (!postFightPrompt)
            return new(false, "post_fight_prompt_not_observed");
        if (postFightWinner)
            return new(false, "winner_must_use_start_round");
        if (scheduleRunning || singleTrialRunning || continuousControllerRunning ||
            attackZoneTrialRunning || attackZoneRecoveryOnlyRunning)
        {
            return new(false, "control_mode_still_running");
        }
        return new(true, "lost_private_session_exit_allowed");
    }
}
