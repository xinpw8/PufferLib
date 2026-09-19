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

## Prepared follow-up, not executed

The r2 bridge adds token-free authenticated Login-page readiness and a one-shot
native continuation. It preserves the initial Home account check and does not
provide password, MFA, or sign-in automation. Its runtime validation requires
the requested second startup approval.

Prepared bridge SHA256:
`f868b55ffa6639df32e6c98d42775676a2c641c61047085eebf2f88ffe07b9d5`.

The final source passed 5,930 isolation/readiness checks, 6,447 relay checks,
441 protocol cases with a pipe roundtrip, and 25 live-transfer driver tests.
See the separate authenticated-continuation validation report under
`ocean/rek/evidence/windows/PolicyExecutionIsolation.Tests/`.

No second startup, authenticated continuation, private-AI round, video, or
completed policy-strength result is claimed here. The first test's r1 bridge
remains installed; r2 is built but has not been deployed.

The temporary recorder mode change was restored to its original configuration
after the test. Its paired-capture opt-in is expired; a subsequent private-AI
evaluation must explicitly select the private-AI recorder mode again.
