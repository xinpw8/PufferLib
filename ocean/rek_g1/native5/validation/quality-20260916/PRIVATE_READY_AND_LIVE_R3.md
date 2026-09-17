# Private readiness bootstrap and authentic rendered-observation trial

Audit date: 2026-09-17 UTC. This report adds documentation only. No game input,
GPU work, or training was performed for the audit.

## Native lifecycle evidence and bounded readiness change

An inactive `Idle` snapshot reporting difficulty 2, round 3, and fight epoch 0,
without a visual fighter pair, does not establish the identity of a newly
spawned opponent. Recovered native code permits retained round and difficulty
fields before the next fight's setup and AI ownership processing:

- `FightCoordinator.ApplyFightStateSnapshot`, RVA `0x2379E00`, constructs a
  missing `currentRound` from a nonzero packet round number independently of
  starting the fight (`FightCoordinator.txt:30416`). It assigns the packet's
  `clientAiDifficultyLevel` at line 32433. The Idle-to-non-Idle transition
  increments `clientFightEpoch` and starts client setup at lines 32918 and
  32944. These fields therefore need not identify a spawned active fighter.
- `ServerSendFightState` exports the current AI difficulty at line 22745.
  `EnsureAiDifficultyOwner`, RVA `0x237EF50`, compares the pilot identity,
  retains difficulty for the same identity, and resets difficulty to zero only
  when the owner changes (lines 66108 to 66130). `PossessSlotIfChanged` invokes
  this before applying difficulty (line 5291). This is a possible lifecycle
  explanation for the earlier Idle observations, not proof that either
  earlier reservation would have spawned Bot1.
- `OnFightButtonNetwork` (line 49191) sends native readiness.
  `OnReadyReceived` (line 6473) checks registered-client/Idle/start-pending
  conditions and schedules the ordinary delayed fight start. The native
  readiness path does not require the client's pre-start difficulty to be zero.

Source is the recovered native disassembly and field/method labels at
`C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp\FightCoordinator.txt`,
SHA-256 `9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5`.
RVAs identify this recovered build; they are not portable across game builds.

The new [ReadyPrivateAiSession handler](../../../../rek/evidence/windows/RekUiBridgeAgent/Plugin.PrivateAiBootstrap.cs)
allows one native ready request per lease only in an explicitly isolated,
proven private solo, client-only, no-human, inactive Idle session with no visual
fighter pair, setup, menu, or concurrent controller. The route probe validates
the same endpoint/runtime identity without prematurely binding it. Existing
bound-identity checks and the default binding behavior remain intact in
[SoloRouteProofContract.cs](../../../../rek/evidence/windows/SoloRouteProofContract.cs).
No difficulty, pose, score, or policy-eligibility field is written.

The [driver](../../live_transfer_run.cjs) waits for a spawned active exact Bot1
and the unchanged active-gameplay proof before starting policy input. Active
Bot3, human/route changes, rejection, or timeout stop the bootstrap. The old
exact-Bot1 `StartRound` behavior remains unchanged.

Offline results: [25 Node tests](private-ready-node-tests.txt),
[4,492 relay assertions](private-ready-relay-tests.txt), and
[275 protocol cases with a local pipe round-trip](private-ready-protocol-tests.txt)
passed. Relay tests recorded zero native-game and global-input invocations.
The DLL build completed with zero warnings. These are offline checks.

Built artifacts, rehashed for this report:

```text
C:\rekagent\work\g1-private-ai-ready-20260916\bridge\RekUiBridgeAgent.dll
2edf75c65e6693db28eb7e87c45d88522668f549c7b457cdf0aa108582667952
C:\rekagent\work\g1-private-ai-ready-20260916\relay\RekUiPipeClient.exe
0438528bbda9ad736b10bd68a1d2dbfee8f1f555f74e1e6b73b4a61799d8a674
```

The [deployment record](private-ready-deployment/stdout.txt) verifies the DLL
replacement after stopping the owned isolated process. Crucially, r3's fresh
reservation already reported Bot1, so it used the old `StartRound` path at
00:30:54.490 UTC. **The new readiness branch was not exercised live.** See the
[command timeline](live-bot1-rendered-r3/stdout.jsonl).

## Completed authentic round

Checkpoint SHA-256:
`f87dae69a777e4ac28782bdee89b30d7434208d773be75bf97f56fba4a52b07e`.
Policy stream: 00:31:03.676 to 00:33:03.261 UTC. The 120 s round ended in a
**4:14 points loss to Bot1**. There were 5,820 predictions, 48.687 decisions/s,
and 78/78 attacks locally accepted. Every selected raw and encoded mask bit
was true. The only rejected action was a held-state action at stream shutdown.
Local acceptance does not establish authoritative server execution.

Sources: [summary](live-bot1-rendered-r3/trial/summary.json),
[provenance](live-bot1-rendered-r3/provenance.json), and
[quality audit](live-bot1-rendered-r3.audit.json). The source JSONL files remain
under `/home/spark-advantage/rek-training/policy-quality-20260916-r1/live-bot1-rendered-r3/trial/`:

```text
relay.stdout.jsonl   8bed2c7d98cd5a57b6ad290099a55232e43144fd6c86c0baa5c66428b6d0378d
encoder.stdout.jsonl 810d8b0151d5c338758e0a5c88f897a2204368cb1bb4287ca0c1963af6f2a2b8
summary.json         7804bf2ecb1611c4c51ee8e024dcd572e750f21bd021c74479751c589c62edcf
```

Right hook, category 23/runtime move 3, accounted for 56/78 attacks. The rest
were left jab 7, left-hook/right-jab 5, left front kick 3, double uppercut 2,
and five other moves once each. Projected busy occupied 64.26% of samples;
held translation occupied 32.15%. At least one attack was legal in 139 samples
(2.388%), with an attack selected in 78 of those samples.

Measured rendered-root facing error was within 0.16 rad in 11.72% of frames.
At attack requests, median absolute error was 0.591 rad; 31/78 exceeded
pi/4 and only 8/78 were within 0.16 rad. Mean request root gap was 0.781 m.
Across all frames, 2.354% of gaps were below the compact model's 0.44 m minimum.

All four own +1 score updates followed a right-hook request:

| Score-update time from first source, s | Score after | Request age, s | Request root gap, m | Request facing error, rad |
| ---: | ---: | ---: | ---: | ---: |
| 59.006 | 1:3 | 0.614 | 0.599 | 0.071 |
| 71.098 | 2:5 | 0.558 | 0.584 | 0.375 |
| 77.591 | 3:8 | 0.582 | 0.638 | 0.745 |
| 90.684 | 4:10 | 0.612 | 0.771 | 0.506 |

These QPC-based request windows are temporal associations. Native clip identity,
accepted-contact attribution, and server playback acceptance were unavailable.

There were no KO-like score/pose/reset sequences in this run: opponent score
increments were only +1 or +2; actor root height stayed at least 0.570 m and
tilt at most 38.79 degrees. The earlier 13:24 cadence trial instead had four
+5 awards accompanied by low poses and bilateral spawn resets, consistent with
the recovered continuing-round KO rule; see [the native KO report](NATIVE_KO_TARGET_GAP.md).
False `knockout`/`fallen` flags and zero `falls` alone cannot establish absence
of KOs. The combined evidence supports ordinary scoring for r3. One round
cannot establish a stability improvement or attribute differences to training.

## Measured locked-action root-motion gap

[fast_runtime.cu](../../fast_runtime.cu), lines 234 to 259, advances the attack
phase and returns before root x/y/yaw integration at line 268. The baked source
root XY span is only `2.256279998e-7 m`, and its provenance explicitly labels
root translation an external approximation
([asset manifest](eval-post-bot1-r1-fixed/stderr.txt)).

In r3, 3,369 consecutive source pairs had the same requested-move QPC, projected
busy true at both samples, and both root gaps greater than 0.5 m. Across those
68.848 s of sampled intervals, measured actor planar speed had median
0.254 m/s and p90 0.580 m/s. These gaps exclude the compact model's ordinary
0.44 m body-separation correction zone.

For the 56 accepted right-hook requests, actor net planar root displacement
from the first source after acknowledgment to the last source within 0.9 s
was median 0.0563 m, p90 0.1614 m, and maximum 0.3733 m.
[encode_live.cpp](../../live_transfer/encode_live.cpp), lines 34 and 192 to 194,
defines that 45-tick/50-Hz candidate window. It is not measured server playback
duration. All timing uses recorded QPC and frequency. Replicated-pose jitter
can inflate finite-difference speed; net displacement is separately reported.

This demonstrates different root-motion behavior over the corresponding
request windows, which changes strike range and facing distributions. It
does not prove the cause of any individual missed strike. A grounded next
diagnostic is matching right-hook request gap/facing bins between native and
saved authentic traces, then comparing root/hand displacement and score yield.
No heuristic motion curve, difficulty override, or physical-parity claim is
justified by these measurements.
