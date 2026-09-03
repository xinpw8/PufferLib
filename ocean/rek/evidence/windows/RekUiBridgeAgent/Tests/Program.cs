using System.IO.Pipes;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using RekUiBridgeAgent;

var failures = new ConcurrentQueue<string>();

void Expect(string name, bool condition)
{
    if (!condition)
        failures.Enqueue(name);
}

Expect("unity_fixed_rate", BridgeScheduleContract.UnityFixedRateHz == 500);
Expect("schedule_rate", BridgeScheduleContract.ScheduleRateHz == 50);
Expect("schedule_decimation", BridgeScheduleContract.FixedSubstepsPerScheduleTick == 10);
Expect(
    "schedule_duration",
    BridgeScheduleContract.DurationScheduleTicks / (double)BridgeScheduleContract.ScheduleRateHz == 52.02);
Expect("final_schedule_tick", BridgeScheduleContract.FinalScheduleTick == 2600);
Expect(
    "fixed_delta_time",
    BridgeScheduleContract.ExpectedFixedDeltaTime * BridgeScheduleContract.UnityFixedRateHz == 1.0f);
Expect(
    "schedule_sha256",
    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        BridgeScheduleContract.CanonicalJson))).ToLowerInvariant() ==
        BridgeScheduleContract.ExpectedSha256);
Expect(
    "unproven_unexpected_round_reset_not_exposed",
    !Enum.GetNames<BridgeCommand>().Contains(
        "StartUnexpectedPrivateAiRound",
        StringComparer.Ordinal));

var exactT800Bones = BridgePairingContract.T800BoneNames
    .Select(name => (string?)name)
    .ToArray();
Expect("t800_bone_count", exactT800Bones.Length == 26);
Expect("t800_bone_root", exactT800Bones[0] == "LINK_BASE");
Expect("t800_bone_tail", exactT800Bones[^1] == "LINK_HEAD_YAW");
Expect(
    "t800_bone_signature_sha256",
    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        string.Join("\n", exactT800Bones)))).ToLowerInvariant() ==
        BridgePairingContract.T800BoneSignatureSha256);

var exactPairing = BridgePairingContract.Validate(
    "t800",
    exactT800Bones,
    "t800",
    exactT800Bones);
Expect("exact_t800_pairing_accepted", exactPairing.ExactT800VersusT800);
Expect(
    "exact_t800_pairing_reason",
    exactPairing.Reason == BridgePairingContract.ExactPairingReason);

var wrongOrderBones = exactT800Bones.ToArray();
(wrongOrderBones[1], wrongOrderBones[2]) = (wrongOrderBones[2], wrongOrderBones[1]);
Expect(
    "local_bone_order_mismatch_rejected",
    BridgePairingContract.Validate("t800", wrongOrderBones, "t800", exactT800Bones).Reason ==
    "local_fighter_t800_bone_signature_mismatch");
Expect(
    "opponent_bone_order_mismatch_rejected",
    BridgePairingContract.Validate("t800", exactT800Bones, "t800", wrongOrderBones).Reason ==
    "opponent_fighter_t800_bone_signature_mismatch");
Expect(
    "local_bone_count_mismatch_rejected",
    BridgePairingContract.Validate("t800", exactT800Bones[..^1], "t800", exactT800Bones).Reason ==
    "local_fighter_bone_count_not_26");
Expect(
    "opponent_g1_layout_rejected",
    BridgePairingContract.Validate(
        "t800",
        exactT800Bones,
        "t800",
        Enumerable.Repeat<string?>("pelvis", 30).ToArray()).Reason ==
    "opponent_fighter_bone_count_not_26");
Expect(
    "local_semantic_id_required",
    BridgePairingContract.Validate(null, exactT800Bones, "t800", exactT800Bones).Reason ==
    "local_fighter_robot_id_unavailable");
Expect(
    "opponent_semantic_t800_required",
    BridgePairingContract.Validate("t800", exactT800Bones, "g1", exactT800Bones).Reason ==
    "opponent_fighter_robot_id_not_t800");
Expect(
    "semantic_id_is_exact_case_sensitive",
    !BridgePairingContract.Validate("T800", exactT800Bones, "t800", exactT800Bones)
        .ExactT800VersusT800);

void ExpectParse(
    string name,
    string json,
    bool expected,
    BridgeKey? expectedKey = null,
    BridgeCommand? expectedCommand = null)
{
    var actual = BridgeProtocol.TryParse(
        Encoding.UTF8.GetBytes(json),
        7,
        out var request,
        out _,
        out _);
    if (actual != expected ||
        (expectedKey is not null && request?.Key != expectedKey) ||
        (expectedCommand is not null && request?.Command != expectedCommand))
        failures.Enqueue(name);
}

ExpectParse("get_state", "{\"type\":\"get_state\",\"request_id\":\"r-1\"}", true);
foreach (var key in Enum.GetValues<BridgeKey>())
{
    ExpectParse(
        $"input_{key}",
        $"{{\"type\":\"input\",\"request_id\":\"r-{key}\",\"key\":\"{key}\"}}",
        true,
        key);
}
foreach (var command in Enum.GetValues<BridgeCommand>())
{
    ExpectParse(
        $"command_{command}",
        $"{{\"type\":\"command\",\"request_id\":\"r-{command}\",\"command\":\"{command}\"}}",
        true,
        expectedCommand: command);
}
ExpectParse("lowercase_key", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"left\"}", false);
ExpectParse("numeric_key_zero", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"0\"}", false);
ExpectParse("numeric_key_defined", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"6\"}", false);
ExpectParse("unknown_key", "{\"type\":\"input\",\"request_id\":\"r\",\"key\":\"F1\"}", false);
ExpectParse("lowercase_command", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"navigatefreeplay\"}", false);
ExpectParse("numeric_command_zero", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"0\"}", false);
ExpectParse("numeric_command_defined", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"6\"}", false);
ExpectParse("unknown_command", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"LaunchLiveArena\"}", false);
ExpectParse("unknown_property", "{\"type\":\"get_state\",\"request_id\":\"r\",\"extra\":\"x\"}", false);
ExpectParse("duplicate_property", "{\"type\":\"get_state\",\"request_id\":\"r\",\"request_id\":\"r2\"}", false);
ExpectParse("invalid_id", "{\"type\":\"get_state\",\"request_id\":\"space id\"}", false);
ExpectParse("bad_shape", "{\"type\":\"get_state\",\"request_id\":\"r\",\"key\":\"Left\"}", false);
ExpectParse("mixed_command_input", "{\"type\":\"command\",\"request_id\":\"r\",\"command\":\"NavigateFreePlay\",\"key\":\"Left\"}", false);
ExpectParse("invalid_json", "{", false);

var pipeName = $"rek-ui-bridge-test-{Environment.ProcessId}-{Guid.NewGuid():N}";
var accepted = new TaskCompletionSource<BridgeRequest>(TaskCreationOptions.RunContinuationsAsynchronously);
using (var server = new LocalPipeServer(
           pipeName,
           request => accepted.TrySetResult(request),
           info => Console.WriteLine($"PIPE_INFO {info}"),
           warning =>
           {
               Console.WriteLine($"PIPE_WARNING {warning}");
               failures.Enqueue($"pipe_warning:{warning}");
           }))
{
    server.Start();
    using var client = new NamedPipeClientStream(
        ".",
        pipeName,
        PipeDirection.InOut,
        PipeOptions.Asynchronous);
    using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
    await client.ConnectAsync(timeout.Token);
    using var reader = new StreamReader(client, Encoding.UTF8, false, 4096, leaveOpen: true);
    var helloLine = await reader.ReadLineAsync().WaitAsync(timeout.Token);
    if (helloLine is null)
    {
        failures.Enqueue("missing_hello");
    }
    else
    {
        using var hello = JsonDocument.Parse(helloLine);
        if (hello.RootElement.GetProperty("event").GetString() != "hello" ||
            !hello.RootElement.GetProperty("current_user_only").GetBoolean() ||
            !hello.RootElement.GetProperty("local_computer_verified").GetBoolean() ||
            !hello.RootElement.GetProperty("capabilities")
                .GetProperty("exclusive_control_lease_required").GetBoolean() ||
            hello.RootElement.GetProperty("capabilities")
                .GetProperty("semantic_commands").GetArrayLength() != Enum.GetValues<BridgeCommand>().Length)
        {
            failures.Enqueue("invalid_hello");
        }
    }

    if (client.IsConnected)
    {
        try
        {
            var bytes = Encoding.UTF8.GetBytes("{\"type\":\"get_state\",\"request_id\":\"pipe-test\"}\n");
            await client.WriteAsync(bytes.AsMemory(), timeout.Token);
            await client.FlushAsync(timeout.Token);
            var request = await accepted.Task.WaitAsync(timeout.Token);
            if (request.Kind != RequestKind.GetState || request.RequestId != "pipe-test")
                failures.Enqueue("pipe_request_mismatch");
            server.Send(request.ConnectionId, new { @event = "test_ack", request_id = request.RequestId });
            var ackLine = await reader.ReadLineAsync().WaitAsync(timeout.Token);
            if (ackLine is null || !ackLine.Contains("test_ack", StringComparison.Ordinal))
                failures.Enqueue("missing_pipe_ack");
        }
        catch (Exception exception)
        {
            failures.Enqueue($"pipe_roundtrip_exception:{exception.GetType().Name}");
        }
    }
}

if (!failures.IsEmpty)
{
    Console.Error.WriteLine($"FAIL {failures.Count}: {string.Join(",", failures)}");
    return 1;
}

Console.WriteLine(
    $"PASS protocol_cases={16 + Enum.GetValues<BridgeKey>().Length + Enum.GetValues<BridgeCommand>().Length} " +
    "local_pipe_roundtrip=true");
return 0;
