using RekEvidence;
using RekEvidenceRecorder;

var grant = new ConsentedPairGrant(true, true, "arena-a", "connection-a", "account-a", "account-b", 0, 7, 1000, 10000, 1000);
var observation = new ConsentedPairObservation("arena-a", "connection-a", "account-a", true, true, true,
    true, true, true, 7, new[] { new ConsentedParticipant("account-a", 0), new ConsentedParticipant("account-b", 1) }, 1900, 1000);
var checks = 0;
void Expect(bool pass, string name) { checks++; if (!pass) throw new Exception(name); }
void Reject(ConsentedPairGrant? g, ConsentedPairObservation? o, string reason, long now = 2000) {
    var result = ConsentedPairScopeContract.Evaluate(g, o, now);
    Expect(!result.Allowed && result.Reason == reason, reason);
}
Expect(ConsentedPairScopeContract.Evaluate(grant, observation, 2000).Allowed, "explicit_pair_allowed");
Expect(ConsentedPairScopeContract.Evaluate(grant, observation with { Participants = observation.Participants!.Reverse().ToArray() }, 2000).Allowed, "roster_order_irrelevant");
Reject(null, observation, "consented_pair_inactive");
Reject(grant with { Enabled = false }, observation, "consented_pair_inactive");
Reject(grant with { BothAccountsConsented = false }, observation, "two_account_consent_missing");
foreach (var g in new[] { grant with { ArenaId = null }, grant with { SessionInstance = "" }, grant with { LocalAccountId = " " },
    grant with { OtherAccountId = "account-a" }, grant with { LocalSlot = -1 }, grant with { ParticipantEpoch = -1 } })
    Reject(g, observation, "consent_binding_invalid");
foreach (var g in new[] { grant with { QpcFrequency = 0 }, grant with { IssuedQpc = -1 }, grant with { ExpiresQpc = 1000 },
    grant with { ExpiresQpc = 902000 }, grant with { ExpiresQpc = long.MaxValue } })
    Reject(g, observation, "consent_clock_invalid");
Reject(grant, observation, "consent_clock_rollback", 999);
Reject(grant, observation, "consent_expired", 10000);
Reject(grant, null, "participant_observation_missing");
foreach (var o in new[] { observation with { QpcFrequency = 1 }, observation with { ObservedQpc = 999 },
    observation with { ObservedQpc = 2001 }, observation with { ObservedQpc = 1749 } })
    Reject(grant, o, "participant_observation_stale_or_clock_invalid");
Expect(ConsentedPairScopeContract.Evaluate(grant, observation with { ObservedQpc = 1750 }, 2000).Allowed, "freshness_boundary");
Reject(grant, observation with { IsolatedAgentClient = false }, "isolated_agent_client_not_proven");
Reject(grant, observation with { Unranked = false }, "consented_unranked_room_access_not_proven");
Reject(grant, observation with { ExclusiveRoomProven = false }, "consented_unranked_room_access_not_proven");
Expect(ConsentedPairScopeContract.Evaluate(grant with { AllowNonexclusiveRoom = true }, observation with { ExclusiveRoomProven = false }, 2000).Allowed, "explicit_nonexclusive_room_consent");
Reject(grant with { AllowNonexclusiveRoom = true }, observation with { CompleteRosterKnown = false }, "complete_participant_history_not_proven");
Reject(grant, observation with { ArenaId = "arena-b" }, "consented_room_or_session_changed");
Reject(grant, observation with { SessionInstance = "reconnected" }, "consented_room_or_session_changed");
Reject(grant, observation with { ServerAccountBindingsVerified = false }, "server_account_bindings_not_proven");
Reject(grant, observation with { LocalAccountId = "account-c" }, "local_account_changed");
Reject(grant, observation with { CompleteRosterKnown = false }, "complete_participant_history_not_proven");
Reject(grant, observation with { RosterHistoryContinuous = false }, "complete_participant_history_not_proven");
// Even if the same pair returns after a third-party visit, the epoch invalidates consent.
Reject(grant, observation with { ParticipantEpoch = 9 }, "participant_membership_changed");
Reject(grant, observation with { Participants = null }, "exactly_two_participants_required");
Reject(grant, observation with { Participants = observation.Participants!.Append(new ConsentedParticipant("account-c", -1)).ToArray() }, "exactly_two_participants_required");
foreach (var roster in new[] {
    new[] { new ConsentedParticipant("account-a", 0), new ConsentedParticipant("account-c", 1) },
    new[] { new ConsentedParticipant("account-a", 0), new ConsentedParticipant("account-b", -1) },
    new[] { new ConsentedParticipant("account-a", 1), new ConsentedParticipant("account-b", 0) },
    new[] { new ConsentedParticipant("account-a", 0), new ConsentedParticipant("account-a", 0) },
    new[] { new ConsentedParticipant("account-a", 0), new ConsentedParticipant(null, 1) } })
    Reject(grant, observation with { Participants = roster }, "consented_fighter_pair_not_proven");
Console.WriteLine($"{{\"test\":\"consented_pair_scope_contract\",\"checks\":{checks},\"failed\":0,\"native_runtime_exercised\":false}}");
var passive = ConsentedPairScopeContract.EvaluatePassiveCapture(grant with { AllowNonexclusiveRoom = true },
    observation with { IsolatedAgentClient = false, ExclusiveRoomProven = false, CompleteRosterKnown = false, RosterHistoryContinuous = false, Unranked = false }, 2000);
Expect(passive.CaptureAllowed && !passive.ControlAllowed, "passive_unknown_roster_never_authorizes_control");
var config = new ConsentedPairCaptureConfig(true, "Ancient Pit 70", "scabnft", "moogleod", DateTimeOffset.UnixEpoch.AddSeconds(60), true);
var facts = new PairCaptureFacts("arena", "Ancient Pit 70", "session", "scabnft", "stable-a", "moogleod", "stable-b", 0, true, false, "0,1,2");
ConsentedPairCaptureTracker Tracker() => new(config, DateTimeOffset.UnixEpoch, 1000, 1000);
var tracker = Tracker();
Expect(!tracker.Evaluate(facts with { OtherName = "bystander" }, 1100).CaptureAllowed && !tracker.Bound, "unexpected_name_does_not_bind");
Expect(tracker.Evaluate(facts, 1200).CaptureAllowed && tracker.LocalId == "stable-a", "names_bind_server_ids_once");
Expect(!tracker.Evaluate(facts with { OtherId = "different-id" }, 1300).CaptureAllowed, "same_name_different_account_stops");
Expect(!tracker.Evaluate(facts, 1400).CaptureAllowed, "identity_violation_latches");
tracker = Tracker(); tracker.Evaluate(facts, 1200); tracker.ObserveMembershipChange();
Expect(!tracker.Evaluate(facts, 1300).CaptureAllowed, "membership_callback_latches");
tracker = Tracker(); tracker.Evaluate(facts, 1200);
Expect(!tracker.Evaluate(facts with { ObservedConnectionSet = "0,1,2,3" }, 1300).CaptureAllowed, "observed_spectator_change_stops");
tracker = Tracker(); tracker.Evaluate(facts with { ObservedConnectionSet = null }, 1200);
Expect(tracker.Evaluate(facts with { ObservedConnectionSet = null }, 1300).CaptureAllowed, "unknown_roster_is_explicit_passive_only");
Expect(!tracker.Evaluate(facts with { ObservedConnectionSet = null }, 61000).CaptureAllowed, "passive_consent_expires");
Console.WriteLine($"{{\"test\":\"consented_pair_passive_capture\",\"checks_total\":{checks},\"failed\":0,\"control_authorized\":false}}");
