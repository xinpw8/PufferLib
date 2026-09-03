using System.Collections.Concurrent;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace RekUiBridgeAgent;

internal sealed class LocalPipeServer : IDisposable
{
    private const int ErrorPipeLocal = 229;
    private readonly string _pipeName;
    private readonly Func<BridgeRequest, bool> _acceptRequest;
    private readonly Action<string> _logInfo;
    private readonly Action<string> _logWarning;
    private readonly ConcurrentQueue<OutboundMessage> _outbound = new();
    private readonly SemaphoreSlim _outboundSignal = new(0);
    private readonly CancellationTokenSource _stop = new();
    private readonly object _activePipeLock = new();
    private NamedPipeServerStream? _activePipe;
    private Task? _serverTask;
    private long _nextConnectionId;
    private long _currentConnectionId;

    internal LocalPipeServer(
        string pipeName,
        Func<BridgeRequest, bool> acceptRequest,
        Action<string> logInfo,
        Action<string> logWarning)
    {
        _pipeName = pipeName;
        _acceptRequest = acceptRequest;
        _logInfo = logInfo;
        _logWarning = logWarning;
    }

    internal long CurrentConnectionId => Interlocked.Read(ref _currentConnectionId);

    internal void Start()
    {
        if (_serverTask is not null)
            throw new InvalidOperationException("Pipe server was already started.");
        _serverTask = Task.Run(ServerLoopAsync);
    }

    internal void Send(long connectionId, object payload)
    {
        if (connectionId <= 0)
            return;
        _outbound.Enqueue(new OutboundMessage(connectionId, payload));
        _outboundSignal.Release();
    }

    internal void SendToCurrent(object payload)
    {
        var connectionId = CurrentConnectionId;
        if (connectionId > 0)
            Send(connectionId, payload);
    }

    private async Task ServerLoopAsync()
    {
        while (!_stop.IsCancellationRequested)
        {
            NamedPipeServerStream? pipe = null;
            try
            {
                pipe = CreatePipe();
                lock (_activePipeLock)
                    _activePipe = pipe;

                await pipe.WaitForConnectionAsync(_stop.Token).ConfigureAwait(false);
                if (!TryVerifyLocalClient(pipe, out var locality))
                {
                    _logWarning($"Rejected named-pipe client: {locality}");
                    pipe.Dispose();
                    continue;
                }

                var connectionId = Interlocked.Increment(ref _nextConnectionId);
                Interlocked.Exchange(ref _currentConnectionId, connectionId);
                _logInfo($"Accepted local current-user pipe client, connection {connectionId}.");

                using var connectionStop = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token);
                var writerTask = WriteLoopAsync(pipe, connectionId, connectionStop.Token);
                Send(connectionId, new
                {
                    @event = "hello",
                    protocol = "rek.ui_bridge.v1",
                    connection_id = connectionId,
                    pipe = _pipeName,
                    current_user_only = true,
                    local_computer_verified = true,
                    capabilities = new
                    {
                        state = true,
                        input_available = false,
                        parsed_but_rejected_input = new[] { "Left", "Right", "Up", "Down", "Enter", "Escape", "Space" },
                        input_unavailable_reason = "verified_process_targeted_unity_input_delivery_not_implemented",
                        semantic_commands = Enum.GetNames<BridgeCommand>(),
                        exclusive_control_lease_required = true,
                        autonomous_input = false,
                    },
                });

                try
                {
                    await ReadLoopAsync(pipe, connectionId, connectionStop.Token).ConfigureAwait(false);
                }
                finally
                {
                    connectionStop.Cancel();
                    try
                    {
                        await writerTask.ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                    }
                    if (Interlocked.CompareExchange(ref _currentConnectionId, 0, connectionId) == connectionId)
                        _logInfo($"Pipe client disconnected, connection {connectionId}.");
                }
            }
            catch (OperationCanceledException) when (_stop.IsCancellationRequested)
            {
                break;
            }
            catch (PlatformNotSupportedException exception)
            {
                _logWarning($"Pipe server stopped because current-user-only ACL is unsupported: {exception.GetType().Name}");
                break;
            }
            catch (Exception exception)
            {
                if (!_stop.IsCancellationRequested)
                {
                    _logWarning($"Pipe server connection failed: {exception.GetType().Name}");
                    try
                    {
                        await Task.Delay(500, _stop.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                }
            }
            finally
            {
                lock (_activePipeLock)
                {
                    if (ReferenceEquals(_activePipe, pipe))
                        _activePipe = null;
                }
                pipe?.Dispose();
            }
        }
    }

    private NamedPipeServerStream CreatePipe() => new(
        _pipeName,
        PipeDirection.InOut,
        maxNumberOfServerInstances: 1,
        PipeTransmissionMode.Byte,
        System.IO.Pipes.PipeOptions.Asynchronous | System.IO.Pipes.PipeOptions.CurrentUserOnly,
        inBufferSize: BridgeProtocol.MaxLineBytes + 1,
        outBufferSize: 64 * 1024);

    private async Task ReadLoopAsync(
        NamedPipeServerStream pipe,
        long connectionId,
        CancellationToken cancellationToken)
    {
        var readBuffer = new byte[1024];
        var lineBuffer = new byte[BridgeProtocol.MaxLineBytes + 1];
        var lineLength = 0;

        while (!cancellationToken.IsCancellationRequested && pipe.IsConnected)
        {
            var count = await pipe.ReadAsync(readBuffer.AsMemory(), cancellationToken).ConfigureAwait(false);
            if (count == 0)
                return;

            for (var index = 0; index < count; index++)
            {
                var value = readBuffer[index];
                if (value == (byte)'\n')
                {
                    var payloadLength = lineLength;
                    if (payloadLength > 0 && lineBuffer[payloadLength - 1] == (byte)'\r')
                        payloadLength--;
                    HandleLine(lineBuffer.AsSpan(0, payloadLength), connectionId);
                    lineLength = 0;
                    continue;
                }

                if (lineLength >= BridgeProtocol.MaxLineBytes)
                {
                    Send(connectionId, ErrorPayload(null, "line_too_long"));
                    return;
                }
                lineBuffer[lineLength++] = value;
            }
        }
    }

    private void HandleLine(ReadOnlySpan<byte> line, long connectionId)
    {
        if (!BridgeProtocol.TryParse(line, connectionId, out var request, out var error, out var requestId))
        {
            Send(connectionId, ErrorPayload(requestId, error));
            return;
        }

        if (!_acceptRequest(request!))
            Send(connectionId, ErrorPayload(request!.RequestId, "request_queue_full_or_duplicate"));
    }

    private async Task WriteLoopAsync(
        NamedPipeServerStream pipe,
        long connectionId,
        CancellationToken cancellationToken)
    {
        using var writer = new StreamWriter(
            pipe,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            bufferSize: 64 * 1024,
            leaveOpen: true)
        {
            AutoFlush = true,
            NewLine = "\n",
        };

        while (!cancellationToken.IsCancellationRequested && pipe.IsConnected)
        {
            await _outboundSignal.WaitAsync(cancellationToken).ConfigureAwait(false);
            while (_outbound.TryDequeue(out var message))
            {
                if (message.ConnectionId != connectionId)
                    continue;
                var line = JsonSerializer.Serialize(message.Payload, BridgeJson.Options);
                await writer.WriteLineAsync(line.AsMemory(), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static object ErrorPayload(string? requestId, string reason) => new
    {
        @event = "error",
        protocol = "rek.ui_bridge.v1",
        request_id = requestId,
        reason,
    };

    private static bool TryVerifyLocalClient(NamedPipeServerStream pipe, out string result)
    {
        var name = new StringBuilder(256);
        if (!GetNamedPipeClientComputerNameW(
                pipe.SafePipeHandle.DangerousGetHandle(),
                name,
                (uint)name.Capacity))
        {
            var error = Marshal.GetLastWin32Error();
            if (error == ErrorPipeLocal)
            {
                result = "local_pipe_verified_win32_error_pipe_local";
                return true;
            }
            result = $"client_computer_lookup_failed_win32_{error}";
            return false;
        }

        var client = NormalizeComputerName(name.ToString());
        var local = NormalizeComputerName(Environment.MachineName);
        if (!string.Equals(client, local, StringComparison.OrdinalIgnoreCase))
        {
            result = "client_computer_not_local";
            return false;
        }

        result = "local_computer_verified";
        return true;
    }

    private static string NormalizeComputerName(string value)
    {
        var normalized = value.Trim().TrimStart('\\');
        var dot = normalized.IndexOf('.');
        return dot >= 0 ? normalized[..dot] : normalized;
    }

    public void Dispose()
    {
        if (_stop.IsCancellationRequested)
            return;
        _stop.Cancel();
        lock (_activePipeLock)
        {
            try
            {
                _activePipe?.Dispose();
            }
            catch
            {
            }
        }
        _outboundSignal.Release();
        try
        {
            _serverTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch
        {
        }
        _outboundSignal.Dispose();
        _stop.Dispose();
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetNamedPipeClientComputerNameW(
        IntPtr pipe,
        StringBuilder clientComputerName,
        uint clientComputerNameLength);
}
