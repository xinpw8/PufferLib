using System.Diagnostics;

namespace RekUiBridgeAgent;

internal static class PipeWriteFailureDiagnostic
{
    // Method metadata identifies the failing component without logging payload
    // values, exception messages, file paths, source lines or exception Data.
    internal static string Describe(Exception exception, string stage)
    {
        var frames = new StackTrace(exception, fNeedFileInfo: false).GetFrames();
        var methods = frames is null ? Array.Empty<string>() : frames.Take(12)
            .Select(frame =>
            {
                var method = frame.GetMethod();
                var type = method?.DeclaringType;
                if (type?.IsGenericType == true) type = type.GetGenericTypeDefinition();
                return $"{type?.FullName ?? "unknown"}.{method?.Name ?? "unknown"}@IL_{frame.GetILOffset():x}";
            }).ToArray();
        return $"Pipe write fault context: stage={stage};type={exception.GetType().FullName};methods={string.Join("|", methods)}";
    }
}
