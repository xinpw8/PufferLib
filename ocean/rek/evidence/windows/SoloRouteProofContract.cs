using System.Security.Cryptography;
using System.Text;

namespace RekEvidence;

internal readonly record struct SoloRouteProofSnapshot(
    bool SoloRouteProven,
    bool FlowSoloObserved,
    bool ConnectToArenaObserved,
    bool EnterChampionshipObserved,
    bool? EnterChampionshipKoth,
    bool? EnterChampionshipSolo,
    bool ArenaIdentityConsistent,
    bool RuntimeSessionIdentityConsistent,
    bool ServerPrivateProven,
    string ServerPrivateStatus,
    string Reason)
{
    internal static SoloRouteProofSnapshot Unavailable(string reason) => new(
        SoloRouteProven: false,
        FlowSoloObserved: false,
        ConnectToArenaObserved: false,
        EnterChampionshipObserved: false,
        EnterChampionshipKoth: null,
        EnterChampionshipSolo: null,
        ArenaIdentityConsistent: false,
        RuntimeSessionIdentityConsistent: false,
        ServerPrivateProven: false,
        ServerPrivateStatus: SoloRouteProofContract.ServerPrivateStatusUnknown,
        Reason: reason);
}

internal readonly record struct SoloRouteScopeDecision(bool Allowed, string Reason);

internal static class SoloRouteProofContract
{
    internal const string ExactFlow = "solo";
    internal const string ProvenReason = "solo_route_proven";
    internal const string ServerPrivateStatusUnknown = "unknown";

    internal static SoloRouteScopeDecision EvaluateScope(
        bool exactBotOneNoHumanProofEstablished,
        SoloRouteProofSnapshot route)
    {
        if (!exactBotOneNoHumanProofEstablished)
        {
            return new SoloRouteScopeDecision(
                false,
                "exact_sparring_bot_1_no_human_scope_not_proven");
        }

        if (!route.SoloRouteProven)
            return new SoloRouteScopeDecision(false, route.Reason);
        return route.RuntimeSessionIdentityConsistent
            ? new SoloRouteScopeDecision(true, ProvenReason)
            : new SoloRouteScopeDecision(
                false,
                "solo_route_runtime_session_identity_not_proven");
    }
}

internal sealed class SoloRouteProofTracker
{
    private enum RouteStage
    {
        None,
        FlowSolo,
        Connected,
        Entered,
        Terminal,
    }

    private readonly object _gate = new();
    private RouteStage _stage;
    private bool _flowSoloObserved;
    private bool _connectToArenaObserved;
    private bool _enterChampionshipObserved;
    private bool? _enterChampionshipKoth;
    private bool? _enterChampionshipSolo;
    private byte[]? _connectedArenaHash;
    private byte[]? _provenArenaHash;
    private byte[]? _enteredEndpointHash;
    private byte[]? _boundRuntimeSessionHash;
    private string _reason = "solo_route_not_observed";

    internal void ObserveFindMatch(string? flow)
    {
        lock (_gate)
        {
            ResetRoute();
            _flowSoloObserved = string.Equals(
                flow,
                SoloRouteProofContract.ExactFlow,
                StringComparison.Ordinal);
            _stage = _flowSoloObserved
                ? RouteStage.FlowSolo
                : RouteStage.Terminal;
            _reason = _flowSoloObserved
                ? "solo_route_waiting_for_connect"
                : "find_match_flow_not_solo";
        }
    }

    internal void ObserveConnectToArena(string? arenaId)
    {
        lock (_gate)
        {
            _connectToArenaObserved = false;
            _enterChampionshipObserved = false;
            _enterChampionshipKoth = null;
            _enterChampionshipSolo = null;
            _connectedArenaHash = null;
            _provenArenaHash = null;
            _boundRuntimeSessionHash = null;

            if (_stage != RouteStage.FlowSolo)
            {
                MarkAttemptTerminal("solo_route_connect_without_fresh_solo_flow");
                return;
            }
            if (string.IsNullOrWhiteSpace(arenaId))
            {
                MarkAttemptTerminal("solo_route_connect_arena_identity_missing");
                return;
            }

            _connectedArenaHash = HashIdentifier(arenaId);
            _connectToArenaObserved = true;
            _stage = RouteStage.Connected;
            _reason = "solo_route_waiting_for_enter_championship";
        }
    }

    internal void ObserveEnterChampionship(
        string? arenaId,
        string? endpointHost,
        int endpointPort,
        bool koth,
        bool solo)
    {
        lock (_gate)
        {
            _enterChampionshipObserved = true;
            _enterChampionshipKoth = koth;
            _enterChampionshipSolo = solo;
            _provenArenaHash = null;
            _enteredEndpointHash = null;
            _boundRuntimeSessionHash = null;

            if (_stage != RouteStage.Connected)
            {
                MarkAttemptTerminal("enter_championship_without_fresh_solo_connect");
                return;
            }
            if (!_connectToArenaObserved || _connectedArenaHash is null)
            {
                MarkAttemptTerminal("enter_championship_without_solo_connect");
                return;
            }
            if (string.IsNullOrWhiteSpace(arenaId))
            {
                MarkAttemptTerminal("enter_championship_arena_identity_missing");
                return;
            }

            var enteredArenaHash = HashIdentifier(arenaId);
            if (!CryptographicOperations.FixedTimeEquals(
                    enteredArenaHash,
                    _connectedArenaHash))
            {
                MarkAttemptTerminal("solo_route_arena_identity_mismatch");
                return;
            }
            if (koth)
            {
                MarkAttemptTerminal("solo_route_entered_as_koth");
                return;
            }
            if (!solo)
            {
                MarkAttemptTerminal("solo_route_entered_without_solo_flag");
                return;
            }
            if (string.IsNullOrWhiteSpace(endpointHost) || endpointPort <= 0)
            {
                MarkAttemptTerminal("solo_route_entered_endpoint_identity_missing");
                return;
            }

            _provenArenaHash = enteredArenaHash;
            _enteredEndpointHash = HashEndpoint(endpointHost, endpointPort);
            _stage = RouteStage.Entered;
            _reason = SoloRouteProofContract.ProvenReason;
        }
    }

    internal SoloRouteProofSnapshot SnapshotForArena(string? arenaId)
    {
        lock (_gate)
        {
            var arenaIdentityConsistent = false;
            if (_provenArenaHash is not null && !string.IsNullOrWhiteSpace(arenaId))
            {
                arenaIdentityConsistent = CryptographicOperations.FixedTimeEquals(
                    HashIdentifier(arenaId),
                    _provenArenaHash);
            }

            var proven =
                _flowSoloObserved &&
                _connectToArenaObserved &&
                _enterChampionshipObserved &&
                _stage == RouteStage.Entered &&
                _enterChampionshipKoth == false &&
                _enterChampionshipSolo == true &&
                arenaIdentityConsistent;
            var reason = proven
                ? SoloRouteProofContract.ProvenReason
                : _provenArenaHash is not null && !arenaIdentityConsistent
                    ? "solo_route_current_arena_identity_mismatch"
                    : _reason;

            return new SoloRouteProofSnapshot(
                SoloRouteProven: proven,
                FlowSoloObserved: _flowSoloObserved,
                ConnectToArenaObserved: _connectToArenaObserved,
                EnterChampionshipObserved: _enterChampionshipObserved,
                EnterChampionshipKoth: _enterChampionshipKoth,
                EnterChampionshipSolo: _enterChampionshipSolo,
                ArenaIdentityConsistent: arenaIdentityConsistent,
                RuntimeSessionIdentityConsistent: false,
                ServerPrivateProven: false,
                ServerPrivateStatus: SoloRouteProofContract.ServerPrivateStatusUnknown,
                Reason: reason);
        }
    }

    internal SoloRouteProofSnapshot SnapshotForRuntimeSession(
        string? arenaId,
        string? contextEndpointHost,
        int contextEndpointPort,
        string? networkEndpointHost,
        int networkEndpointPort,
        long runtimeSessionIdentity)
    {
        lock (_gate)
        {
            var snapshot = SnapshotForArena(arenaId);
            if (!snapshot.SoloRouteProven)
                return snapshot;
            if (string.IsNullOrWhiteSpace(contextEndpointHost) ||
                contextEndpointPort <= 0 ||
                string.IsNullOrWhiteSpace(networkEndpointHost) ||
                networkEndpointPort <= 0 ||
                runtimeSessionIdentity == 0 ||
                _enteredEndpointHash is null)
            {
                return snapshot with
                {
                    SoloRouteProven = false,
                    Reason = "solo_route_runtime_session_identity_missing",
                };
            }

            var contextEndpointHash = HashEndpoint(
                contextEndpointHost,
                contextEndpointPort);
            var networkEndpointHash = HashEndpoint(
                networkEndpointHost,
                networkEndpointPort);
            if (!CryptographicOperations.FixedTimeEquals(
                    contextEndpointHash,
                    _enteredEndpointHash) ||
                !CryptographicOperations.FixedTimeEquals(
                    networkEndpointHash,
                    _enteredEndpointHash))
            {
                return snapshot with
                {
                    SoloRouteProven = false,
                    Reason = "solo_route_runtime_endpoint_identity_mismatch",
                };
            }

            var runtimeHash = HashIdentifier(
                $"{arenaId}\n{contextEndpointHost}\n{contextEndpointPort}\n{runtimeSessionIdentity}");
            _boundRuntimeSessionHash ??= runtimeHash;
            var runtimeSessionIdentityConsistent = CryptographicOperations.FixedTimeEquals(
                runtimeHash,
                _boundRuntimeSessionHash);
            return snapshot with
            {
                SoloRouteProven = runtimeSessionIdentityConsistent,
                RuntimeSessionIdentityConsistent = runtimeSessionIdentityConsistent,
                Reason = runtimeSessionIdentityConsistent
                    ? SoloRouteProofContract.ProvenReason
                    : "solo_route_runtime_session_identity_mismatch",
            };
        }
    }

    internal void Invalidate(string reason)
    {
        lock (_gate)
        {
            ResetRoute();
            _reason = string.IsNullOrWhiteSpace(reason)
                ? "solo_route_invalidated"
                : reason;
        }
    }

    internal bool InvalidateIfRuntimeSessionBound(string reason)
    {
        lock (_gate)
        {
            if (_boundRuntimeSessionHash is null)
                return false;

            ResetRoute();
            _reason = string.IsNullOrWhiteSpace(reason)
                ? "solo_route_runtime_session_invalidated"
                : reason;
            return true;
        }
    }

    private void ResetRoute()
    {
        _flowSoloObserved = false;
        _connectToArenaObserved = false;
        _enterChampionshipObserved = false;
        _enterChampionshipKoth = null;
        _enterChampionshipSolo = null;
        _connectedArenaHash = null;
        _provenArenaHash = null;
        _enteredEndpointHash = null;
        _boundRuntimeSessionHash = null;
        _stage = RouteStage.None;
    }

    private void MarkAttemptTerminal(string reason)
    {
        _stage = RouteStage.Terminal;
        _provenArenaHash = null;
        _enteredEndpointHash = null;
        _boundRuntimeSessionHash = null;
        _reason = reason;
    }

    private static byte[] HashEndpoint(string host, int port) =>
        HashIdentifier($"{host.Length}:{host}\n{port}");

    private static byte[] HashIdentifier(string value) =>
        SHA256.HashData(Encoding.UTF8.GetBytes(value));
}
