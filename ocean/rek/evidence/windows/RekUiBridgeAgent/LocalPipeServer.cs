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
                    local_client_verification = locality,
                    capabilities = new
                    {
                        state = true,
                        input_available = false,
                        parsed_but_rejected_input = new[] { "Left", "Right", "Up", "Down", "Enter", "Escape", "Space" },
                        input_unavailable_reason = "verified_process_targeted_unity_input_delivery_not_implemented",
                        semantic_commands = Enum.GetNames<BridgeCommand>(),
                        exclusive_control_lease_required = true,
                        autonomous_input = false,
                        autonomous_semantic_controller = true,
                        rendered_command_marker_schema = RenderedCommandMarkerContract.Schema,
                        rendered_command_marker_render_binding = RenderedCommandMarkerContract.RenderBinding,
                        rendered_command_marker_count = RenderedCommandMarkerContract.Specs.Length,
                        single_motion_trial_schema = SingleMotionTrialContract.Schema,
                        single_motion_trial_sha256 = SingleMotionTrialContract.ExpectedSha256,
                        single_motion_trial_selectors = SingleMotionTrialContract.Selectors
                            .Select(value => value.Selector)
                            .ToArray(),
                        single_motion_trial_authority_scope = SingleMotionTrialContract.AuthorityScope,
                        single_motion_trial_authority_caveat = SingleMotionTrialContract.AuthorityCaveat,
                        single_motion_trial_unity_fixed_rate_hz = SingleMotionTrialContract.UnityFixedRateHz,
                        single_motion_trial_rate_hz = SingleMotionTrialContract.TrialRateHz,
                        single_motion_trial_fixed_substeps_per_tick =
                            SingleMotionTrialContract.FixedSubstepsPerTrialTick,
                        single_motion_trial_neutral_pre_roll_ticks =
                            SingleMotionTrialContract.NeutralPreRollTicks,
                        single_motion_trial_action_tick = SingleMotionTrialContract.ActionTick,
                        single_motion_trial_locomotion_release_tick =
                            SingleMotionTrialContract.LocomotionReleaseTick,
                        single_motion_trial_duration_ticks = SingleMotionTrialContract.DurationTrialTicks,
                        continuous_controller_schema = ContinuousBotControllerContract.Schema,
                        continuous_controller_sha256 = ContinuousBotControllerContract.ExpectedSha256,
                        continuous_controller_authority_scope =
                            ContinuousBotControllerContract.AuthorityScope,
                        continuous_controller_authority_caveat =
                            ContinuousBotControllerContract.AuthorityCaveat,
                        continuous_controller_range_angle_provenance =
                            ContinuousBotControllerContract.RangeAngleProvenance,
                        continuous_controller_facing_yaw_provenance =
                            ContinuousBotControllerContract.FacingYawProvenance,
                        continuous_controller_attack_selection_provenance =
                            ContinuousBotControllerContract.AttackSelectionProvenance,
                        continuous_controller_static_impact_timing_provenance =
                            ContinuousBotControllerContract.StaticImpactTimingProvenance,
                        continuous_controller_round_restart_limitation =
                            ContinuousBotControllerContract.RoundRestartLimitation,
                        continuous_controller_round_restart_static_evidence =
                            ContinuousBotControllerContract.RoundRestartStaticEvidence,
                        continuous_controller_unity_fixed_rate_hz =
                            ContinuousBotControllerContract.UnityFixedRateHz,
                        continuous_controller_rate_hz =
                            ContinuousBotControllerContract.ControlRateHz,
                        continuous_controller_fixed_substeps_per_tick =
                            ContinuousBotControllerContract.FixedSubstepsPerControlTick,
                        continuous_controller_move_indices = ContinuousBotControllerContract.Attacks
                            .Select(value => value.MoveIndex)
                            .ToArray(),
                        continuous_controller_recovery_guard_provenance =
                            ContinuousBotControllerContract.RecoveryGuardProvenance,
                        continuous_controller_fault_estop_provenance =
                            ContinuousBotControllerContract.FaultEStopProvenance,
                        continuous_controller_dampen_guard =
                            ContinuousBotControllerContract.DampenGuard,
                        continuous_controller_straighten_guard =
                            ContinuousBotControllerContract.StraightenGuard,
                        continuous_controller_opponent_runtime_requirement =
                            ContinuousBotControllerContract.OpponentRuntimeRequirement,
                        continuous_controller_facing_deadband_factor =
                            ContinuousBotControllerContract.FacingDeadbandFactor,
                        continuous_controller_facing_threshold_degrees =
                            ContinuousBotControllerContract.FacingThresholdDegrees,
                        continuous_controller_facing_yaw_ramp_degrees =
                            ContinuousBotControllerContract.FacingYawRampDegrees,
                        continuous_controller_engage_yaw_command =
                            ContinuousBotControllerContract.EngageYawCommand,
                        continuous_controller_fault_estop_delay_ticks =
                            ContinuousBotControllerContract.FaultEStopDelayTicks,
                        continuous_controller_fault_estop_hold_ticks =
                            ContinuousBotControllerContract.FaultEStopHoldTicks,
                        continuous_controller_recovery_observation_timeout_ticks =
                            ContinuousBotControllerContract.RecoveryObservationTimeoutTicks,
                        continuous_controller_round_start_prompt_delay_ticks =
                            ContinuousBotControllerContract.RoundStartPromptDelayTicks,
                        continuous_controller_round_start_observation_timeout_ticks =
                            ContinuousBotControllerContract.RoundStartObservationTimeoutTicks,
                        continuous_controller_two_minute_limit_ticks =
                            ContinuousBotControllerContract.TwoMinuteTicks,
                        continuous_controller_round_start_semantic_method =
                            "GameMenuController.HandlePostFightContinue",
                        continuous_controller_global_space_input_emitted = false,
                        continuous_controller_opponent_semantic_robot_id_used_for_acceptance =
                            false,
                        continuous_controller_attack_profiles =
                            ContinuousBotControllerContract.Attacks.Select(attack => new
                            {
                                move_index = attack.MoveIndex,
                                move_name = attack.MoveName,
                                display_name = attack.DisplayName,
                                maximum_distance_m = attack.MaximumDistanceMeters,
                                maximum_abs_bearing_degrees =
                                    attack.MaximumAbsBearingDegrees,
                                serialized_asset_sha256 = attack.SerializedAssetSha256,
                                static_impact_events = attack.StaticImpactEvents.Select(value => new
                                {
                                    impact_time_s = value.ImpactTimeSeconds,
                                    lead_time_s = value.LeadTimeSeconds,
                                    release_time_s = value.ReleaseTimeSeconds,
                                    limb = value.Limb,
                                    gain_boost = value.GainBoost,
                                }).ToArray(),
                            }).ToArray(),
                        attack_zone_trial_schema = AttackZoneTrialContract.Schema,
                        attack_zone_trial_sha256 = AttackZoneTrialContract.ExpectedSha256,
                        attack_zone_trial_authority_scope = AttackZoneTrialContract.AuthorityScope,
                        attack_zone_trial_authority_caveat = AttackZoneTrialContract.AuthorityCaveat,
                        attack_zone_trial_required_isolation_proof =
                            AttackZoneTrialContract.RequiredIsolationProof,
                        attack_zone_trial_control_rate_hz = AttackZoneTrialContract.ControlRateHz,
                        attack_zone_trial_fixed_substeps_per_tick =
                            AttackZoneTrialContract.FixedSubstepsPerControlTick,
                        attack_zone_trial_settle_ticks = AttackZoneTrialContract.SettleTicks,
                        attack_zone_trial_action_sample_rate_hz =
                            AttackZoneTrialContract.ControlRateHz,
                        attack_zone_trial_recovery_ready_ticks =
                            AttackZoneTrialContract.RecoveryReadyTicks,
                        attack_zone_trial_acquisition_timeout_ticks =
                            AttackZoneTrialContract.AcquisitionTimeoutTicks,
                        attack_zone_trial_minimum_independent_runs_per_cell =
                            AttackZoneTrialContract.MinimumIndependentRunsPerCell,
                        attack_zone_trial_recorder_version =
                            AttackZoneTrialContract.ExpectedRecorderVersion,
                        attack_zone_trial_recorder_plugin_sha256 =
                            AttackZoneTrialContract.ExpectedRecorderPluginSha256,
                        attack_zone_trial_global_input_emitted = false,
                    },
                });
                Interlocked.Exchange(ref _currentConnectionId, connectionId);
                _logInfo($"Accepted local current-user pipe client, connection {connectionId}.");

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
        try
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
        catch (EntryPointNotFoundException)
        {
            return TryVerifyLocalClientProcess(pipe, out result);
        }
    }

    private static bool TryVerifyLocalClientProcess(NamedPipeServerStream pipe, out string result)
    {
        if (!GetNamedPipeClientProcessId(
                pipe.SafePipeHandle.DangerousGetHandle(),
                out var clientProcessId))
        {
            result = $"client_process_lookup_failed_win32_{Marshal.GetLastWin32Error()}";
            return false;
        }
        if (clientProcessId == 0 || clientProcessId > int.MaxValue)
        {
            result = "client_process_id_invalid";
            return false;
        }

        try
        {
            using var clientProcess = System.Diagnostics.Process.GetProcessById((int)clientProcessId);
            if (clientProcess.HasExited)
            {
                result = "client_process_not_live";
                return false;
            }
        }
        catch
        {
            result = "client_process_not_resolvable_on_local_host";
            return false;
        }

        result = "local_process_id_verified_after_computer_name_api_unavailable";
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

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetNamedPipeClientProcessId(
        IntPtr pipe,
        out uint clientProcessId);
}
