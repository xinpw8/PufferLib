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

DateTimeOffset Utc(long qpc) => DateTimeOffset.UnixEpoch.AddMilliseconds(qpc - 1000);
var extended = config with { ExpiresUtc = DateTimeOffset.UnixEpoch.AddHours(2) };
var longGrant = grant with { ExpiresQpc = 14401000 };
Expect(ConsentedPairScopeContract.EvaluatePassiveCapture(longGrant, observation, 2000).CaptureAllowed,
    "passive_four_hour_limit_allowed");
Expect(!ConsentedPairScopeContract.EvaluatePassiveCapture(longGrant with { ExpiresQpc = 14401001 }, observation, 2000).CaptureAllowed,
    "passive_over_four_hours_rejected");
Reject(longGrant, observation, "consent_clock_invalid");
Expect(!ConsentedPairCaptureTracker.UsePairModeAtStartup(false, null, false), "absent_startup_preserves_ai_default");
Expect(!ConsentedPairCaptureTracker.UsePairModeAtStartup(true, config with { Enabled = false }, false), "disabled_startup_preserves_ai_default");
Expect(ConsentedPairCaptureTracker.UsePairModeAtStartup(true, null, true), "invalid_startup_does_not_fall_into_ai");
Expect(ConsentedPairCaptureTracker.UsePairModeAtStartup(true, config, false), "enabled_startup_uses_pair_mode");

tracker = Tracker(); tracker.Evaluate(facts, 1200);
Expect(tracker.Reload(extended, Utc(11000), 11000) == "consented_pair_configuration_loaded", "explicit_same_pair_renewal_loaded");
var renewed = tracker.Evaluate(facts, 62000);
Expect(renewed.CaptureAllowed && !renewed.ControlAllowed, "renewed_capture_beyond_original_expiry_never_control");
Expect(tracker.LocalId == "stable-a" && tracker.OtherId == "stable-b" && tracker.ArenaId == "arena", "renewal_retains_pinned_ids_arena");
Expect(!tracker.Evaluate(facts with { SessionInstance = "other-session" }, 63000).CaptureAllowed, "renewal_retains_session_binding");
tracker.Reload(extended with { ExpiresUtc = extended.ExpiresUtc.AddMinutes(1) }, Utc(64000), 64000);
Expect(!tracker.Evaluate(facts, 65000).CaptureAllowed, "renewal_cannot_clear_session_change_latch");

tracker = Tracker(); tracker.Evaluate(facts, 1200);
Expect(tracker.Evaluate(facts, 61000).Reason == "consent_expired", "expiry_reported_before_renewal");
tracker.Reload(extended, Utc(62000), 62000);
Expect(tracker.Evaluate(facts, 63000).CaptureAllowed, "same_pair_renewal_recovers_expired_capture");
Expect(!tracker.Evaluate(facts, 7201000).CaptureAllowed, "renewed_capture_expires_at_new_deadline");

tracker = Tracker(); tracker.Evaluate(facts, 1200);
for (var time = 2000L; time < 61000; time += 1000) tracker.Reload(config, Utc(time), time);
Expect(tracker.Evaluate(facts, 61000).Reason == "consent_expired", "one_hz_unchanged_reload_does_not_slide_expiry");
tracker.Reload(config, DateTimeOffset.UnixEpoch, 62000);
Expect(tracker.Evaluate(facts, 63000).Reason == "consent_expired", "wall_clock_rollback_does_not_renew_unchanged_config");

foreach (var stop in new Action<ConsentedPairCaptureTracker>[] {
    t => t.Reload(config with { Enabled = false }, Utc(2000), 2000),
    t => t.Reload(null, Utc(2000), 2000),
    t => t.RejectConfiguration("consented_pair_configuration_missing"),
    t => t.RejectConfiguration("consented_pair_configuration_invalid") })
{
    tracker = Tracker(); tracker.Evaluate(facts, 1200); stop(tracker);
    var denied = tracker.Evaluate(facts, 2100);
    Expect(!denied.CaptureAllowed && !denied.ControlAllowed, "disabled_invalid_missing_config_stops_passive_capture");
    Expect(tracker.LocalId == "stable-a" && tracker.OtherId == "stable-b", "suspension_preserves_bound_ids");
    tracker.Reload(extended, Utc(3000), 3000);
    Expect(tracker.Evaluate(facts, 3100).CaptureAllowed, "same_pair_valid_config_recovers_suspension");
}

foreach (var changedFacts in new[] { facts with { OtherId = "changed" }, facts with { LocalId = "changed" },
    facts with { ArenaId = "changed" }, facts with { SessionInstance = "changed" },
    facts with { LocalSlot = 1 }, facts with { ObservedConnectionSet = "0,1,2,3" } })
{
    tracker = Tracker(); tracker.Evaluate(facts, 1200);
    tracker.Evaluate(changedFacts, 62000);
    tracker.Reload(extended, Utc(63000), 63000);
    Expect(!tracker.Evaluate(facts, 64000).CaptureAllowed, "identity_change_while_expired_survives_renewal");
    tracker = Tracker(); tracker.Evaluate(facts, 1200);
    tracker.Reload(config with { Enabled = false }, Utc(2000), 2000);
    tracker.Evaluate(changedFacts, 2100);
    tracker.Reload(extended, Utc(3000), 3000);
    Expect(!tracker.Evaluate(facts, 3100).CaptureAllowed, "identity_change_while_disabled_survives_renewal");
}
tracker = Tracker(); tracker.Evaluate(facts, 1200); tracker.ObserveMembershipChange();
tracker.Reload(extended, Utc(2000), 2000);
Expect(!tracker.Evaluate(facts, 2100).CaptureAllowed, "renewal_cannot_clear_membership_callback_latch");
foreach (var changedConfig in new[] { extended with { ArenaDisplayName = "different" },
    extended with { LocalDisplayName = "different" }, extended with { OtherDisplayName = "different" } })
{
    tracker = Tracker(); tracker.Evaluate(facts, 1200);
    Expect(tracker.Reload(changedConfig, Utc(2000), 2000) == "consented_pair_configuration_binding_changed", "different_config_pair_rejected");
    tracker.Reload(extended, Utc(3000), 3000);
    Expect(!tracker.Evaluate(facts, 3100).CaptureAllowed, "config_pair_change_cannot_reset_binding");
}
tracker = Tracker(); tracker.Evaluate(facts, 1200);
Expect(tracker.Reload(config with { ExpiresUtc = Utc(2000).AddHours(4).AddTicks(1) }, Utc(2000), 2000) ==
    "consented_pair_configuration_invalid", "renewal_above_four_hours_rejected");
Expect(!tracker.Evaluate(facts, 2100).CaptureAllowed, "invalid_long_renewal_stops_capture");
tracker.Reload(config with { ExpiresUtc = Utc(3000).AddHours(4) }, Utc(3000), 3000);
Expect(tracker.Evaluate(facts, 3100).CaptureAllowed, "renewal_exactly_four_hours_from_load_allowed");
Console.WriteLine($"{{\"test\":\"consented_pair_reload\",\"checks_total\":{checks},\"failed\":0,\"control_authorized\":false,\"native_runtime_exercised\":false}}");
