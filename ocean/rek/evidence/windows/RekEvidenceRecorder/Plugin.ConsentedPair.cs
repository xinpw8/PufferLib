using System.Diagnostics;
using System.Text.Json;
using Il2CppInterop.Runtime;
using RekEvidence;
using REKApp;
using Unity.Netcode;
using UnityEngine;

namespace RekEvidenceRecorder;

public sealed partial class Plugin
{
    private void ReloadConsentedPairConfiguration(bool startup)
    {
        var nowQpc = Stopwatch.GetTimestamp();
        if (!startup && nowQpc < _nextConsentedPairConfigPollQpc) return;
        _nextConsentedPairConfigPollQpc = checked(nowQpc + Stopwatch.Frequency);
        var exists = File.Exists(_consentedPairConfigPath);
        if (startup) _consentedPairConfigPresent = exists;
        if (!_consentedPairConfigPresent) return;
        try
        {
            if (!exists)
            {
                _consentedPairConfigReason = "consented_pair_configuration_missing";
                _consentedPair?.RejectConfiguration(_consentedPairConfigReason);
                return;
            }
            var config = JsonSerializer.Deserialize<ConsentedPairCaptureConfig>(File.ReadAllText(_consentedPairConfigPath));
            if (startup)
                _consentedPairConfigPresent = ConsentedPairCaptureTracker.UsePairModeAtStartup(exists, config, false);
            if (!_consentedPairConfigPresent) return;
            if (_consentedPair is not null)
                _consentedPairConfigReason = _consentedPair.Reload(config, DateTimeOffset.UtcNow, nowQpc);
            else if (config?.Enabled == true)
            {
                _consentedPair = new(config, DateTimeOffset.UtcNow, nowQpc, Stopwatch.Frequency);
                _consentedPairConfigReason = "consented_pair_configuration_loaded";
            }
            else _consentedPairConfigReason = config is null
                ? "consented_pair_configuration_invalid" : "consented_pair_configuration_disabled";
        }
        catch
        {
            _consentedPairConfigReason = "consented_pair_configuration_invalid";
            _consentedPair?.RejectConfiguration(_consentedPairConfigReason);
        }
    }

    private void LogConsentedPairScopeStatus(string reason)
    {
        if (!_consentedPairConfigPresent || reason == _consentedPairScopeStatus) return;
        _consentedPairScopeStatus = reason;
        Log.LogInfo($"Consented pair scope status: {reason}");
    }

    private ScopeSnapshot EvaluateConsentedPairScope()
    {
        if (_consentedPair is null) return ScopeSnapshot.Denied(_consentedPairConfigReason);
        if (Math.Abs((double)Time.fixedDeltaTime - ExpectedFixedDeltaTimeSeconds) > FixedDeltaTimeToleranceSeconds)
            return ScopeSnapshot.Denied("consented_pair_fixed_clock_invalid");
        var coordinator = UnityEngine.Object.FindFirstObjectByType<FightCoordinator>();
        var network = UnityEngine.Object.FindFirstObjectByType<NetworkSession>();
        var context = GameContext.Instance;
        if (coordinator is null || network is null || context is null || !network.IsConnected || !network.IsClient || network.IsServer)
            return ScopeSnapshot.Denied("consented_pair_client_session_missing");
        if (context.Mode != GameContext.SessionMode.Championship || string.IsNullOrWhiteSpace(context.ArenaID))
            return ScopeSnapshot.Denied("consented_pair_arena_identity_missing");
        var local = coordinator.LocalFighterIndex;
        if (local is < 0 or > 1) return ScopeSnapshot.Denied("consented_pair_local_fighter_missing");
        var other = 1 - local;
        var identities = coordinator.fighterIdentities;
        if (identities is null || identities.Length != 2 || identities[local] is null || identities[other] is null)
            return ScopeSnapshot.Denied("consented_pair_fighter_identities_missing");
        var clients = coordinator.slotHasClient;
        var bothHuman = clients is not null && clients.Length >= 2 && clients[0] && clients[1] &&
            coordinator.HumanInSlot(0) && coordinator.HumanInSlot(1) && (coordinator.clientHumanSlotMask & 3) == 3;
        string? connectionSet = null; int? connectionCount = null;
        try
        {
            var ids = NetworkManager.Singleton?.ConnectedClientsIds;
            if (ids is not null)
            {
                var values = new List<ulong>();
                var count = ids.Cast<Il2CppSystem.Collections.Generic.IReadOnlyCollection<ulong>>().Count;
                for (var i = 0; i < count; i++) values.Add(ids[i]);
                connectionCount = values.Count;
                connectionSet = string.Join(",", values.OrderBy(x => x));
            }
        }
        catch { /* Coverage unavailable is recorded, never asserted complete. */ }
        var decision = _consentedPair.Evaluate(new PairCaptureFacts(context.ArenaID, context.ArenaDisplayName,
            IL2CPP.Il2CppObjectBaseToPtr(network).ToString(), identities[local].DisplayName, identities[local].UserID,
            identities[other].DisplayName, identities[other].UserID, local, bothHuman,
            !context.IsRanked && !coordinator.IsRankedArena, connectionSet), Stopwatch.GetTimestamp());
        if (!decision.CaptureAllowed) return ScopeSnapshot.Denied(decision.Reason);
        var fighters = coordinator.Fighters;
        if (fighters is null || fighters.Length != 2 || fighters[0] is null || fighters[1] is null ||
            !fighters[0].IsVisualOnly || !fighters[1].IsVisualOnly || fighters[0].RootTransform is null || fighters[1].RootTransform is null)
            return ScopeSnapshot.Denied("consented_pair_visual_fighters_missing");
        var names0 = BoneNames(fighters[0]); var names1 = BoneNames(fighters[1]);
        var pairing = RecorderContract.ValidatePairing(local, identities[0].RobotID, fighters[0].name, names0,
            identities[1].RobotID, fighters[1].name, names1);
        if (!pairing.ExactSupportedRuntimePairing) return ScopeSnapshot.Denied("consented_pair_supported_runtime_pair_required");
        var round = coordinator.CurrentRound;
        if (round is null || !round.IsActive || coordinator.CurrentPhase != FightPhase.RoundActive)
            return ScopeSnapshot.Denied("consented_pair_round_not_active");
        var camera = Camera.main;
        if (camera is null || !camera.enabled || !camera.gameObject.activeInHierarchy || camera.pixelWidth <= 0 || camera.pixelHeight <= 0)
            return ScopeSnapshot.Denied("consented_pair_camera_missing");
        var scope = ScopeSnapshot.AllowedScope(coordinator, network, context, fighters[0], fighters[1],
            identities[0].RobotID, identities[1].RobotID, names0, names1, pairing, camera, local, other, 0,
            SoloRouteProofSnapshot.Unavailable("consented_nonexclusive_pair_no_private_claim"),
            new SoloRouteScopeDecision(true, decision.Reason), ReadServerIdentity(network));
        scope.ConsentedPair = new Dictionary<string, object?>
        {
            ["allowed"] = true, ["capture_only"] = true, ["control_authorized"] = false,
            ["local_fighter_index"] = local, ["opponent_slot"] = other, ["both_fighters_human"] = true,
            ["arena_display_name"] = context.ArenaDisplayName, ["arena_id_sha256"] = HashText(context.ArenaID),
            ["local_display_name"] = identities[local].DisplayName, ["other_display_name"] = identities[other].DisplayName,
            ["local_user_id_sha256"] = HashText(_consentedPair.LocalId!), ["other_user_id_sha256"] = HashText(_consentedPair.OtherId!),
            ["user_id_provenance"] = "server-authenticated FighterIdentity.UserID received for each fighter slot; pinned after configured names match",
            ["context_is_ranked"] = context.IsRanked, ["coordinator_is_ranked_arena"] = coordinator.IsRankedArena,
            ["server_private_proven"] = false, ["room_exclusive_proven"] = false,
            ["participant_roster_complete"] = false, ["observed_connection_count"] = connectionCount,
            ["participant_change_guard"] = "stop on observed connection-set change, network connection callbacks, or bound fighter/account/arena/session change",
            ["unobserved_participant_changes"] = "unknown; no full spectator identity or complete-roster claim",
            ["data_scope"] = "only the two bound fighters: no spectator account, avatar, chat or identity collection",
        };
        return scope;
    }
}
