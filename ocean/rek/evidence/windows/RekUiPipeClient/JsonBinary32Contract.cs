using System.Text.Json;

namespace RekUiBridgeAgent;

internal static class JsonBinary32Contract
{
    internal static void RequireExact(
        JsonElement parent,
        string name,
        float expected)
    {
        if (!parent.TryGetProperty(name, out var value) ||
            value.ValueKind != JsonValueKind.Number ||
            !value.TryGetSingle(out var actual) ||
            !float.IsFinite(actual) ||
            BitConverter.SingleToInt32Bits(actual) !=
                BitConverter.SingleToInt32Bits(expected))
        {
            throw new InvalidDataException(
                $"expected {name} binary32={expected:R} " +
                $"bits=0x{BitConverter.SingleToInt32Bits(expected):x8}");
        }
    }
}
