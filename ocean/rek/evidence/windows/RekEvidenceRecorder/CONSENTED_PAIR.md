# Passive two-account capture

Recorder 0.7.4 adds an explicit two-account capture mode. It observes the local
client's move requests and both fighters' replicated poses, score packets and
hit effects. It does not send inputs or authorize a policy to control a fighter.

At plugin load it reads `BepInEx/config/rek-consented-pair-capture.json`:

```json
{
  "Enabled": true,
  "ArenaDisplayName": "the agreed arena name",
  "LocalDisplayName": "the local account display name",
  "OtherDisplayName": "the other consenting account display name",
  "ExpiresUtc": "replace with a UTC timestamp within the next 15 minutes",
  "AllowNonexclusiveRoom": true
}
```

Names must match exactly. The first matching two-human fighter pair binds the
server-issued `FighterIdentity.UserID` values, local slot, arena ID and local
network-session instance. An observed participant, account, slot or session
change stops capture. Expiry uses the process's monotonic clock after loading.
Missing spectator information is recorded as unknown; the room is not asserted
to be exclusive. No spectator identities or chat are collected.

The current implementation accepts the existing measured homogeneous G1/G1 and
T800/T800 layouts. Mixed rigs remain unsupported. Client fixed-step, camera,
initial-state and transport-packet checks remain in force. Missing or disabled
configuration keeps the previous private-AI scope; invalid enabled configuration
does not fall back to AI capture.

Paired files are separated into `OutputRoot/consented-pair/`, use the
`rek-consented-pair-` filename prefix and declare `rek.consented_pair.protocol.v1`.
They must not be relabeled as private-AI protocol captures or accepted by a
private-AI validator. The attacker-side recorder is needed to observe its exact
requested move ID. The defender does not receive the opponent's move requests.
Requests do not establish server acceptance or execution. A missing score packet
alone does not prove a miss. Hit effects do not uniquely identify the attacker.

Build against the BepInEx core and generated interop assemblies of the actual
target installation. Installing a DLL does not update an already loaded plugin.
Preserve the existing DLL, install while the game is closed, then confirm the
new plugin version and a growing paired capture after relaunch. Do not interrupt
a human-controlled Windows session or use synthetic input to keep it alive.

Offline checks:

```text
dotnet run --project ocean/rek/evidence/windows/ConsentedPairScopeContract.Tests/ConsentedPairScopeContract.Tests.csproj -c Release
dotnet run --project ocean/rek/evidence/windows/RekEvidenceRecorder/Tests/RekEvidenceRecorder.ContractTests.csproj -c Release
```

These checks exercise scope logic and existing recorder contracts. They do not
prove a live capture, contact-model accuracy, or simulator parity.
