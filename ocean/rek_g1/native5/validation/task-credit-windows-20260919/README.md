# Native Windows evaluation startup, 2026-09-19

This is execution-path evidence. It is not a completed fighting evaluation,
training result, or parity qualification.

## Actual startup r1

- Host: `D21`, native Windows, user `d21\daniel`.
- Direct game child: PID `269136`, started `2026-09-19T20:43:20Z`.
- Executable: `C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\REK.exe`.
- Executable SHA256: `5fe6a5c3da371cb7b75a1795eb073b2e69ec718197cf68826bdcd461dcd986c1`.
- Loaded bridge SHA256: `fe6a42b705cc93749ea7c58033182c274d448152593c07d0c137dec1789f326e`.
- Actual bridge proof on Unity's main thread:
  `windows_native=1;station=WinSta0;desktop=RekPolicyEval;input_desktop=Default;host=D21`.
- Live `GameContext.FighterName`: `moogleod`.
- State advanced from Intro to Login. It did not reach Home during this test.
- No control lease, policy stream, gameplay actions, or global input was issued.
- Recorder startup reported `mode=private_sparring_bot_1`.
- The launcher observed 3,981 samples with zero foreground-window changes and
  zero input-desktop changes. Its polling interval cannot exclude shorter
  transients between samples.

At `2026-09-19T20:47:29Z`, the task stopped only its own idle isolated game
process after rechecking its PID, executable, start time, scene, lease and
desktop proof. The launcher therefore reported `child_exit_code=4294967295`
and `passed=false`. That result is retained unchanged. Desktop isolation was
observed; normal application exit and gameplay were not established.

The hidden process did not remain running when the first test ended. No normal
desktop REK process was terminated.

## Login boundary

Recovered `LoginScreenController.OnLetsGoClicked` dispatches `OnLetsGoPressed`;
the registered `LobbyShellController.HandleLetsGo` opens Home. Authentication
uses separate handlers. However, the first deployed bridge did not expose the
live Login page's authenticated-state fields, so its cached account name could
not distinguish authenticated continuation from a real sign-in prompt.

The initial Home account requirement correctly withheld gameplay. Permission
was requested for a further startup with explicit token-free Login readiness
checks. No password, token, MFA, or authentication handler was used in r1.

## Prepared policy route

Windows child processes successfully started the C++ observation encoder and
native CUDA policy worker through `wsl -e bash -lc 'ssh -T spark ...'`.
Worker readiness reported NVIDIA GB10, 223 observations and 33 actions.
Checkpoint SHA256:
`5bdad2893c5e97e682fdb33ab48a298e2cd2d9c03df724fa2e629cc1c9882246`.

This transport smoke test read readiness messages and closed the worker. It
did not run a fighting round or demonstrate policy strength. The existing
223-feature observation contract was unchanged by the Windows isolation work.

## Artifact locations

Private commands, process metadata, stdout/stderr, state snapshots and backups:

`C:\rekagent\work\task-credit-windows-20260919-r1`

Relevant files:

- `deployment-startup-r1.txt`
- `launch-r1/launch.json`
- `launch-r1/result.json`
- `startup-state-01.json`, `startup-state-02.json`
- `finish-startup-r1.txt`
- `transport-smoke-r1/result.json`
- `prepared-worker-config.json`
- `deployment-backup/`

Raw account-page snapshots and proprietary game/plugin backups stay outside
Git. No video was captured in this startup test.

The 19-file startup snapshot was copied to the physical evidence server and
every copied file's SHA256 was verified:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\task-credit-windows-startup-r1`

The server snapshot includes `snapshot-hashes.json`. The local originals were
preserved.

## Prepared follow-up at the end of startup r1

The prepared r2 bridge added token-free authenticated Login-page readiness and a one-shot
native continuation. It preserves the initial Home account check and does not
provide password, MFA, or sign-in automation. Its runtime validation requires
the requested second startup approval at that point. The authorized execution
is documented below.

Prepared bridge SHA256:
`f868b55ffa6639df32e6c98d42775676a2c641c61047085eebf2f88ffe07b9d5`.

The final source passed 5,930 isolation/readiness checks, 6,447 relay checks,
441 protocol cases with a pipe roundtrip, and 25 live-transfer driver tests.
See the separate authenticated-continuation validation report under
`ocean/rek/evidence/windows/PolicyExecutionIsolation.Tests/`.

At the end of startup r1, no second startup, authenticated continuation,
private-AI round, video, or policy-strength result had been executed. The first
test's r1 bridge was still installed at that point.

The temporary recorder mode change was restored to its original configuration
after the test. Its paired-capture opt-in is expired; a subsequent private-AI
evaluation must explicitly select the private-AI recorder mode again.

## Authorized native execution and authentic results

The user subsequently authorized idle REK restarts without repeated requests.
The idle Home client was closed, r2 was installed, and native REK PID `319504`
ran on `WinSta0\RekPolicyEval`. Fresh native Login evidence proved that moogleod
was already authenticated. `ConfirmLoggedIn` opened Home through the recovered
post-login continuation only. No credentials or authentication handlers were used.

The initial helper reused a state request ID. The bridge deduplicates IDs per
connection, causing a polling timeout before any continuation. Unique poll IDs
fixed this; the one-shot continuation then passed its Home/account postcondition.
`windows_authenticated_continue.cjs` preserves the corrected flow, with 42
mock-transport tests and no automatic retry after an uncertain continuation.

All four trials below used the same task512 checkpoint, native Windows REK,
Spark GB10 CUDA inference, measured local slot 0 and Sparring Bot 1 / difficulty
0. All terminal results were `WonByPoints`. These are awarded point totals,
not counts of strike events.

| Trial | Round | Local:AI points | Observed outcome | Initial delay | Strict full-round policy coverage |
|---|---:|---:|---|---:|---|
| r1 | 1 | 11:5 | Win | 0.20035 s | Pass |
| r2 | 2 | 10:22 | Loss | 1.01777 s | Fail, 0.01777 s beyond the unchanged threshold |
| r3 | 3 | 16:6 | Win after late entry | 14.526794 s | Fail, exclude from full-round policy claims |
| r4, fresh fight | 1 | 4:10 | Loss | 0.13336 s | Pass |

The two strictly covered rounds are one win and one loss. Four observed terminal
outcomes are not four fully controlled rounds and do not establish a population
win rate. This policy can win a real Bot 1 round but is not consistently dominant.
No superhuman performance or simulator parity is established.

The one-second coverage test was unchanged. r2/r3 show a lifecycle problem:
restarting the inference process between automatic rounds can join the next
round late. Startup must be completed before a fresh fight, or inference and
the recurrent-boundary handling must remain available across automatic rounds.
r4 exercised the fresh-fight path with inference ready before the native ready
request. Timing cutoffs must not be relaxed to hide late starts.

### Point and referee evidence

The existing offline analyzer bound r1/r2/r4 native packets to their policy
streams using concurrent Unity frames, QPC and both root positions. All three
native captures completed without errors and all point packets reconcile to the
terminal scoreboard. Their respective matching-anchor counts were 1,057,
1,051 and 1,015. r3's stream-only analysis records its late coverage failure;
its native recording is preserved but was not supplied to that analysis because
the capture began long before policy entry.

- r1: local six one-point hit-associated awards plus one five-point referee
  award; AI one two-point and three one-point awards. The opponent's observed
  slip/countout supplied the five referee points. 109 attack dispatches and ten
  received hit effects are separate measurements, not a causal hit rate.
- r2: local ten one-point awards; AI five one-point, six two-point and one
  five-point referee award. Native packets establish a local slip followed by
  a 3.01384 s countout and a five-point opponent award. All 5,795 bridge samples
  still had `falls:[0,0]`. Other point awards alone totaled 10:17, so the countout
  does not account for the entire loss.
- r4: 4:10, eleven received score packets and eleven hit effects, no five-point
  referee awards. The policy therefore also lost without a countout award.

Same-frame score/hit associations do not identify causal policy requests,
executed attacks, hitboxes or contact regions. No absent event is labeled a miss.

### Remaining observation and training disparity

`Plugin.G1PolicyStream.cs` copies native `round.Falls`; its zeros are not created
by the encoder. Why this native counter remains zero during the observed G1
countout is not established. The policy stream also emits `referee:null`.

`live_transfer/encode_live.cpp` parses raw fallen/falling/tilt/floor and falls
fields, but explicit down/tilt/falling feature slots, falls slots 192–193 and
down-state slots 202–205 are zero. There is no referee count-mask/count-seconds
input. Root orientation, height and joint posture still reach the policy, so it
is not blind to all posture information. Local fallen/increased falls can cancel
projected action busy, but r2's zero counters do not carry its countout.

Five-point awards enter scoreboard features and the generic positive-score
hit-history proxy without their referee cause. Compact `fast_runtime.cu`
explicitly omits balance/fall dynamics and uses upright motion. Longer credit
assignment cannot learn absent transitions. The next parity work must carry
measured referee/countout state and validate corresponding training transitions,
without changing only the live 223-feature contract under this checkpoint.
The hit-scoring deficit also requires work; removing five points arithmetically
is not a causal counterfactual of a round without a countout.

### Execution and storage

The launcher sampled 11,789 times with zero detected foreground changes and
zero desktop changes. As in r1, the task explicitly closed its isolated process,
so `launch-r2/result.json` retains `passed:false` and termination exit code
`4294967295`; no normal-exit claim is made. No global input was emitted.

The evaluation artifacts are under `task512-r1` through `task512-r4` in the
private Windows evidence directory above. `contact-analysis/` holds derived
results; r1/r2 also include explicit referee breakdowns. Compact summaries are
published alongside this report; raw telemetry and proprietary artifacts remain
outside Git. This run is telemetry-only: no video was recorded.

Training was not rerun here. The previously measured task512 training throughput
was 928,776 full-training SPS on Spark; this live-client evaluation does not
measure training SPS and does not improve the training-physics omissions.
