using System.Diagnostics;
using System.Runtime.InteropServices;
using HarmonyLib;
using REKApp;
using Unity.Netcode;
using UnityEngine;

namespace RekUiBridgeAgent;

public sealed partial class Plugin
{
    private readonly G1ReceivedRefereeCache _receivedReferee = new();
    private bool _refereeObservationPatchesVerified;
    internal sealed record PendingRefereeReceipt(ReceivedRefereePacket Packet, long Coordinator, long Network,
        RefereeReceiptClock Clock);

    internal unsafe PendingRefereeReceipt? ObserveRefereePrefix(FightCoordinator coordinator, FastBufferReader reader)
    {
        try
        {
            if (!TryGetG1PolicyContext(false, out var scope, out _, allowAnyAi: true, bindRuntimeSession: false) ||
                NativePointer(scope.Coordinator) != NativePointer(coordinator))
            { _receivedReferee.Invalidate("referee_private_policy_scope_not_proven"); return null; }
            if (reader.Length - reader.Position != G1ReceivedRefereeContract.WireBytes)
            { _receivedReferee.Invalidate("referee_wire_length_not_33"); return null; }
            var body = new byte[G1ReceivedRefereeContract.WireBytes];
            // Same read-only copy as RekEvidenceRecorder. Never advance the reader.
            var pointer = (IntPtr)reader.GetUnsafePtrAtCurrentPosition();
            if (pointer == IntPtr.Zero)
            { _receivedReferee.Invalidate("referee_wire_pointer_unavailable"); return null; }
            Marshal.Copy(pointer, body, 0, body.Length);
            if (!G1ReceivedRefereeContract.TryDecode(body, out var packet, out var reason))
            { _receivedReferee.Invalidate(reason); return null; }
            return new(packet!, NativePointer(coordinator).ToInt64(), NativePointer(scope.Network).ToInt64(),
                new(Stopwatch.GetTimestamp(), Stopwatch.Frequency, Time.frameCount, Time.timeAsDouble, Time.unscaledTimeAsDouble));
        }
        catch { _receivedReferee.Invalidate("referee_prefix_probe_failed"); return null; }
    }

    internal void ObserveRefereePostfix(FightCoordinator coordinator, PendingRefereeReceipt? pending)
    {
        if (pending is null) return;
        try
        {
            if (!TryGetG1PolicyContext(false, out var scope, out _, allowAnyAi: true, bindRuntimeSession: false) ||
                NativePointer(coordinator).ToInt64() != pending.Coordinator ||
                NativePointer(scope.Coordinator).ToInt64() != pending.Coordinator ||
                NativePointer(scope.Network).ToInt64() != pending.Network)
            { _receivedReferee.Invalidate("referee_scope_changed_during_apply"); return; }
            _receivedReferee.Commit(pending.Packet, RefereeIdentity(scope), RefereeMirrors(coordinator), pending.Clock, out _);
        }
        catch { _receivedReferee.Invalidate("referee_postfix_probe_failed"); }
    }

    internal void InvalidateReceivedReferee(string reason) => _receivedReferee.Invalidate(reason);

    private static RefereeRuntimeIdentity RefereeIdentity(PrivateAiContext scope)
    {
        var fight = scope.Coordinator.Fight;
        var round = scope.Coordinator.CurrentRound;
        return new(NativePointer(scope.Coordinator).ToInt64(), NativePointer(scope.Network).ToInt64(),
            fight is null ? 0 : NativePointer(fight).ToInt64(), round is null ? 0 : NativePointer(round).ToInt64(),
            scope.Coordinator.fightEpoch, round?.RoundNumber ?? -1, round?.IsRedo ?? false);
    }

    private static RefereeClientMirrors RefereeMirrors(FightCoordinator coordinator) =>
        new(coordinator.clientRefSnapshotSeen, coordinator.clientRefCallSeq, coordinator.clientRefCountMask,
            coordinator.clientRefCountSeconds);

    private object ReceivedRefereePayload(PrivateAiContext scope, long now)
    {
        RefereeAvailability view;
        try
        {
            if (!_refereeObservationPatchesVerified)
                _receivedReferee.Invalidate("referee_observation_hooks_not_verified");
            view = _receivedReferee.Read(RefereeIdentity(scope), RefereeMirrors(scope.Coordinator), now, Stopwatch.Frequency);
        }
        catch
        { _receivedReferee.Invalidate("referee_state_probe_failed"); view = new(false, "referee_state_probe_failed", null, null); }
        var receipt = view.Receipt;
        var packet = receipt?.Packet;
        var callAvailable = packet is { CallSequence: > 0 };
        return new {
            schema = G1ReceivedRefereeContract.Schema, available = view.Available, reason = view.Reason,
            source = "received_REK_FightState_33_byte_body",
            provenance = "ApplyFightStateSnapshot_prefix_copy_postfix_client_mirror_verification",
            authority_scope = "server_authored_packet_observed_on_client_not_server_current_state",
            observation_hooks_verified = _refereeObservationPatchesVerified,
            maximum_receipt_age_seconds = G1ReceivedRefereeContract.MaximumReceiptAgeSeconds,
            receipt_age_seconds = view.AgeSeconds,
            receipt_sequence = receipt?.ReceiptSequence, lifecycle = receipt?.Lifecycle,
            receipt_qpc_ticks = receipt?.Clock.QpcTicks, receipt_qpc_frequency_hz = receipt?.Clock.QpcFrequency,
            receipt_unity_frame = receipt?.Clock.UnityFrame, receipt_unity_time = receipt?.Clock.UnityTime,
            receipt_unity_unscaled_time = receipt?.Clock.UnityUnscaledTime,
            wire_body_sha256 = packet?.WireSha256, wire_body_base64 = packet?.WireBase64,
            count_mask = packet?.CountMask, count_seconds = packet?.CountSeconds,
            slot0_count_active = packet is null ? (bool?)null : (packet.CountMask & 1) != 0,
            slot1_count_active = packet is null ? (bool?)null : (packet.CountMask & 2) != 0,
            call_available = callAvailable, call_sequence = packet?.CallSequence,
            call_type = callAvailable ? packet!.CallType : (byte?)null,
            call_name = callAvailable ? G1ReceivedRefereeContract.CallName(packet!.CallType) : null,
            call_faller = callAvailable ? packet!.CallFaller : (sbyte?)null,
            call_points = callAvailable ? packet!.CallPoints : (byte?)null,
            call_observation_sequence = receipt?.CallObservationSequence,
            call_sequence_transition = receipt?.CallTransition,
            call_history_censored = callAvailable ? receipt!.CallHistoryCensored : (bool?)null,
            packet_phase = packet?.Phase, packet_round_number = packet?.RoundNumber,
            packet_round_active = packet?.RoundActive, packet_round_redo = packet?.Redo,
            packet_round_knockout_occurred = packet?.RoundKnockoutOccurred, packet_round_result = packet?.RoundResult,
            server_tick = (long?)null, server_time = (double?)null, server_fight_epoch = (long?)null,
            semantic_limit = "received_call_is_latched_not_an_event_per_observation;Knockout_call_does_not_imply_terminal_round_KO;packet_has_no_server_epoch_or_event_time;no_attacker_or_action_causality" };
    }
}

[HarmonyPatch(typeof(FightCoordinator), "ApplyFightStateSnapshot")]
internal static class G1ReceivedRefereeSnapshotPatch
{
    [HarmonyPrefix]
    [HarmonyPriority(Priority.First)]
    private static void Prefix(FightCoordinator __instance, FastBufferReader __0, out Plugin.PendingRefereeReceipt? __state)
        => __state = Plugin.Instance?.ObserveRefereePrefix(__instance, __0);

    [HarmonyPostfix]
    [HarmonyPriority(Priority.Last)]
    private static void Postfix(FightCoordinator __instance, Plugin.PendingRefereeReceipt? __state)
        => Plugin.Instance?.ObserveRefereePostfix(__instance, __state);

    [HarmonyFinalizer]
    private static void Finalizer(Exception? __exception)
    {
        if (__exception is not null) Plugin.Instance?.InvalidateReceivedReferee("referee_native_apply_failed");
    }
}

[HarmonyPatch(typeof(FightCoordinator), "ResetClientRefereeReplay")]
internal static class G1ReceivedRefereeResetPatch
{
    [HarmonyPrefix]
    private static void Prefix() => Plugin.Instance?.InvalidateReceivedReferee("referee_native_replay_reset");
}
