internal static class StateReadModeOptions
{
    private const string Option = "--bridge-sha256";

    internal static bool TryExtract(string[] arguments, out string[] remaining,
        out string? bridgeSha256, out string? error)
    {
        remaining = arguments;
        bridgeSha256 = null;
        error = null;
        var kept = new List<string>();
        foreach (var argument in arguments)
        {
            if (!argument.StartsWith(Option, StringComparison.Ordinal))
            {
                kept.Add(argument);
                continue;
            }
            if (arguments.Length == 0 || arguments[0] != "state")
            { error = "bridge_hash_override_requires_state_mode"; return false; }
            if (bridgeSha256 is not null)
            { error = "duplicate_bridge_hash_override"; return false; }
            if (!argument.StartsWith(Option + "=", StringComparison.Ordinal)
                || !IsHash(argument[(Option.Length + 1)..]))
            { error = "bridge_hash_override_requires_64_lowercase_hex"; return false; }
            bridgeSha256 = argument[(Option.Length + 1)..];
        }
        remaining = kept.ToArray();
        return true;
    }

    internal static string ExpectedHash(bool requireLease, string? readOnlyOverride, string legacyHash)
    {
        if (readOnlyOverride is null) return legacyHash;
        if (requireLease || !IsHash(readOnlyOverride))
            throw new InvalidDataException("invalid_read_only_bridge_hash_override");
        return readOnlyOverride;
    }

    private static bool IsHash(string value) => value.Length == 64
        && value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');
}
