internal readonly record struct ControllerRunModeSpec(bool UntilEnded, int RunSeconds);

internal static class ControllerRunModeContract
{
    internal const string PersistentToken = "until-ended";
    internal const int DefaultRunSeconds = 120;
    internal const int MinimumRunSeconds = 1;
    internal const int MaximumRunSeconds = 600;

    internal static bool TryParse(string? value, out ControllerRunModeSpec mode)
    {
        if (value is null)
        {
            mode = new ControllerRunModeSpec(false, DefaultRunSeconds);
            return true;
        }
        if (string.Equals(value, PersistentToken, StringComparison.Ordinal))
        {
            mode = new ControllerRunModeSpec(true, 0);
            return true;
        }
        if (int.TryParse(value, out var seconds) &&
            seconds is >= MinimumRunSeconds and <= MaximumRunSeconds)
        {
            mode = new ControllerRunModeSpec(false, seconds);
            return true;
        }
        mode = default;
        return false;
    }
}
