using System.Diagnostics;
using System.IO.Pipes;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using RekUiBridgeAgent;

internal static class PolicyRelay
{
    private static readonly ConditionalWeakTable<TextReader, BufferedLines> Readers = new();
    private static readonly HashSet<string> Commands = new(StringComparer.Ordinal) {
        "AcquireExclusiveControl", "ReleaseExclusiveControl", "StartG1PolicyStream", "StopG1PolicyStream",
        "StartRound", "ConfirmLoggedIn", "NavigateFreePlay", "EnterSolo", "ExitLostPrivateSession",
        "ExitUnexpectedPrivateAiSession", "ReadyPrivateAiSession",
        "StartG1PolicyRound", "ExitLostG1PolicySession", "StartG1PolicyStreamAnyAi",
        "SkipIntro", "ExitUnsupportedPrivateAiPairing" };

    internal static async Task<int> Run(string[] args)
    {
        if (args.Length != 2 || !G1PolicyStreamContract.IsHash(args[1]))
        { Console.Error.WriteLine("usage: RekUiPipeClient policy-relay expected-bridge-sha256"); return 2; }
        try
        {
            using var stop = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; stop.Cancel(); };
            await using var pipe = new NamedPipeClientStream(".", "rek-ui-bridge-v1",
                PipeDirection.InOut, PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
            using var startup = CancellationTokenSource.CreateLinkedTokenSource(stop.Token);
            startup.CancelAfter(TimeSpan.FromSeconds(15));
            await pipe.ConnectAsync(startup.Token);
            if (!NativeMethods.GetNamedPipeServerProcessId(pipe.SafePipeHandle.DangerousGetHandle(), out var pid) ||
                pid == 0 || pid > int.MaxValue || pid == Environment.ProcessId)
                throw new InvalidDataException("pipe_server_pid_invalid");
            using var process = Process.GetProcessById((int)pid);
            if (process.HasExited || !string.Equals(Path.GetFileName(process.MainModule?.FileName), "REK.exe", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("pipe_server_is_not_live_REK");
            using var reader = new StreamReader(pipe, Encoding.UTF8, false, 65536, leaveOpen: true);
            await using var writer = new StreamWriter(pipe, new UTF8Encoding(false), 65536, leaveOpen: true) { AutoFlush = true, NewLine = "\n" };
            var bootstrapId = "relay-bootstrap-" + Guid.NewGuid().ToString("N");
            await writer.WriteLineAsync(JsonSerializer.Serialize(new { type = "get_state", request_id = bootstrapId }));
            var prelude = new List<string>();
            while (true)
            {
                var line = await ReadBoundedLine(reader, 1048576, startup.Token) ?? throw new EndOfStreamException("pipe_closed_before_state");
                using var document = JsonDocument.Parse(line); var root = document.RootElement;
                RequireString(root, "protocol", "rek.ui_bridge.v1");
                if (prelude.Count >= 16) throw new InvalidDataException("bootstrap_state_not_received");
                prelude.Add(line);
                if (root.TryGetProperty("request_id", out var id) && id.ValueKind == JsonValueKind.String && id.GetString() == bootstrapId)
                {
                    RequireString(root, "event", "state");
                    var build = root.GetProperty("build");
                    RequireString(build, "game_assembly_sha256", "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412");
                    RequireString(build, "global_metadata_sha256", "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd");
                    RequireString(build, "sharedassets0_sha256", "37f7a476c56caae37f5a04d4fa1acf5954fdc2b90f20f521830369ecff05f355");
                    RequireString(build, "plugin_version", "0.4.9"); RequireString(build, "plugin_sha256", args[1]);
                    ValidatePolicyIsolation(root);
                    break;
                }
            }
            foreach (var line in prelude) await Console.Out.WriteLineAsync(line);
            await Console.Out.FlushAsync();
            var receive = Task.Run(async () => {
                while (await ReadBoundedLine(reader, 1048576, stop.Token) is { } line)
                {
                    using var document = JsonDocument.Parse(line);
                    RequireString(document.RootElement, "protocol", "rek.ui_bridge.v1");
                    await Console.Out.WriteLineAsync(line); await Console.Out.FlushAsync();
                }
            });
            var transmit = Task.Run(async () => {
                while (await ReadBoundedLine(Console.In, 4096, stop.Token) is { } line)
                {
                    ValidateRequest(line);
                    await writer.WriteLineAsync(line.AsMemory(), stop.Token);
                }
            });
            var completed = await Task.WhenAny(receive, transmit);
            await completed;
            if (completed == transmit && pipe.IsConnected)
            {
                foreach (var command in new[] { "StopG1PolicyStream", "ReleaseExclusiveControl" })
                    await writer.WriteLineAsync(JsonSerializer.Serialize(new { type = "command", request_id = "relay-close-" + Guid.NewGuid().ToString("N"), command }));
                await Task.WhenAny(receive, Task.Delay(500));
            }
            stop.Cancel();
            try { await receive; } catch (OperationCanceledException) { }
            return 0;
        }
        catch (OperationCanceledException) { Console.Error.WriteLine("policy_relay_cancelled_or_connection_timeout"); return 3; }
        catch (Exception e) { Console.Error.WriteLine("policy_relay_failed:" + e.GetType().Name + ":" + e.Message); return 1; }
    }

    internal static void ValidatePolicyIsolation(JsonElement state)
    {
        var foreground = state.GetProperty("foreground");
        if (!foreground.GetProperty("isolated_session_verified").GetBoolean())
            throw new InvalidDataException("isolated_session_not_verified");
        var proof = foreground.GetProperty("isolated_session_proof").GetString();
        if (!PolicyExecutionIsolationContract.IsSupportedProof(proof))
            throw new InvalidDataException("relay_identity_mismatch:isolated_session_proof");
        if (proof == PolicyExecutionIsolationContract.WindowsProof)
        {
            RequireString(foreground, "execution_surface", "native_windows_isolated_desktop");
            var account = foreground.GetProperty("windows_policy_account");
            RequireString(account, "required_display_name", PolicyExecutionIsolationContract.WindowsAccount);
            RequireString(account, "context_fighter_name", PolicyExecutionIsolationContract.WindowsAccount);
            if (!account.GetProperty("allowed").GetBoolean() || !account.GetProperty("context_available").GetBoolean())
                throw new InvalidDataException("windows_policy_account_not_proven");
            if (account.GetProperty("pinned").GetBoolean())
            {
                if (!account.GetProperty("context_matches_pin").GetBoolean())
                    throw new InvalidDataException("windows_policy_account_pin_mismatch");
            }
            else if (!account.GetProperty("at_home").GetBoolean() || !account.GetProperty("home_display_matches").GetBoolean())
                throw new InvalidDataException("windows_policy_initial_home_account_not_proven");
        }
    }

    internal static void ValidateRequest(string line)
    {
        using var doc = JsonDocument.Parse(line); var r = doc.RootElement;
        if (r.ValueKind != JsonValueKind.Object) throw new InvalidDataException("request_must_be_object");
        var names = new HashSet<string>();
        foreach (var p in r.EnumerateObject()) if (!names.Add(p.Name)) throw new InvalidDataException("duplicate_request_key");
        var type = r.GetProperty("type").GetString();
        if (type == "policy_action")
        { if (!G1PolicyStreamContract.TryParse(r, out _, out _, out var error)) throw new InvalidDataException(error); return; }
        var id = r.GetProperty("request_id").GetString();
        if (id is not { Length: >= 1 and <= 64 } || !id.All(c => char.IsAsciiLetterOrDigit(c) || c is '.' or '_' or ':' or '-'))
            throw new InvalidDataException("request_id_invalid");
        if (type is "get_state" or "get_policy_state")
        { if (names.SetEquals(new[] { "type", "request_id" })) return; }
        else if (type == "command" && names.SetEquals(new[] { "type", "request_id", "command" }) && Commands.Contains(r.GetProperty("command").GetString() ?? "")) return;
        throw new InvalidDataException("request_not_allowed_by_policy_relay");
    }

    internal static async Task<string?> ReadBoundedLine(TextReader reader, int limit, CancellationToken token)
        => await Readers.GetValue(reader, _ => new BufferedLines()).Read(reader, limit, token);

    private sealed class BufferedLines
    {
        private readonly char[] _buffer = new char[4096];
        private int _at, _count;
        internal async Task<string?> Read(TextReader reader, int limit, CancellationToken token)
        {
            var line = new StringBuilder();
            while (true)
            {
                if (_at == _count)
                {
                    _count = await reader.ReadAsync(_buffer.AsMemory(), token); _at = 0;
                    if (_count == 0)
                    { if (line.Length != 0) throw new InvalidDataException("unterminated_relay_line"); return null; }
                }
                while (_at < _count)
                {
                    var c = _buffer[_at++];
                    if (c == '\n') return line.ToString().TrimEnd('\r');
                    if (line.Length >= limit) throw new InvalidDataException("relay_line_too_long");
                    line.Append(c);
                }
            }
        }
    }

    private static void RequireString(JsonElement value, string key, string expected)
    {
        if (!value.TryGetProperty(key, out var field) || field.ValueKind != JsonValueKind.String || field.GetString() != expected)
            throw new InvalidDataException("relay_identity_mismatch:" + key);
    }
}
