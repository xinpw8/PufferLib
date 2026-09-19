# Native Windows isolated policy surface validation

Date: 2026-09-19. This change was built and tested without deploying the bridge,
launching REK, switching desktops, changing foreground focus, or emitting input.

## Scope

Native G1 policy execution accepts only the exact proof:

`windows_native=1;station=WinSta0;desktop=RekPolicyEval;input_desktop=Default;host=D21`

The bridge queries native Windows, the current thread desktop, the process window
station, and the input desktop at mutation boundaries. Wine's native export is
rejected for this surface. Environment variables cannot establish this proof.
API failure, Default execution, an input desktop other than Default, or any
noncanonical host/station/desktop rejects execution.

The first native Windows lease requires the Home display and token-free
`GameContext.FighterName` to both equal `moogleod`. Its native GameContext identity
is pinned across the session. Later mutations recheck that identity and name.
Missing identity is an explicit rejection, not an inferred account. Authentication
commands, legacy schedules, and general StartRound remain unavailable on Windows.
The cached exact Spark proof and Spark-only legacy helpers are unchanged.

Policy scope retains the existing private solo route, no-human occupancy, exact G1
pairing, lease, opponent, round identity, and action-age guards. No observation or
action schema was changed. Foreground evidence adds execution surface and account
diagnostics. Relay bootstrap requires the exact supported proof and, for Windows,
the matching account evidence.

If a running native policy loses its isolation/account proof, cleanup does not
write a neutral command into the now-unverified surface. Its previously owned
controller's velocity and pending move sends are quarantined until restart.
These are client execution guards. They do not establish server acceptance or
revoke commands already received by the server.

## Empirical results

- Bridge build: 0 warnings, 0 errors.
- Relay publish: 0 warnings, 0 errors.
- Isolation tests: 4,860 checks passed, including exhaustive fact combinations,
  malformed proofs, API exceptions, changing proof readings, account changes,
  first-lease pin requirement, and Windows command restrictions.
- Real passive runtime query: native Windows true, API success true, host D21,
  station WinSta0, current desktop Default, input desktop Default. Rejected with
  `windows_policy_desktop_mismatch`.
- Relay tests: 6,447 checks passed, including 14 new bootstrap proof/account
  cases and the existing timing/action/opponent tests.
- Existing protocol tests: 441 cases and local current-user random pipe roundtrip
  passed.
- `git diff --check` passed.

The positive native desktop helper branch uses controlled facts in unit tests.
This subtask did not validate GameContext.FighterName in a running updated bridge
or execute a game policy. The parent task owns isolated startup and authentic
evaluation. Such checks must reject an absent or mismatched name.

## Build artifacts and identity

Output root: `C:\rekagent\tmp\windows-isolation-20260919-r1`.

| Artifact | SHA256 |
| --- | --- |
| `bridge\RekUiBridgeAgent.dll` | `fe6a42b705cc93749ea7c58033182c274d448152593c07d0c137dec1789f326e` |
| `relay\RekUiPipeClient.exe` | `ce4e034a07587dbd618e98615cf2ba603bc9149ecd994c683d37a29034aadcbf` |

Bridge version remains 0.4.9; the new binary SHA256 identifies this implementation.
Relay requires its supplied exact bridge hash as well as the existing game build
hashes. Tested REKApp interop SHA256:
`faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2`.

Pinned GameAssembly:
`6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`.
Pinned global metadata:
`e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd`.
Pinned sharedassets0:
`37f7a476c56caae37f5a04d4fa1acf5954fdc2b90f20f521830369ecff05f355`.

Win32 API contracts checked against Microsoft documentation for
[OpenInputDesktop](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-openinputdesktop)
and [GetUserObjectInformationW](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getuserobjectinformationw).
The input desktop handle is opened with DESKTOP_READOBJECTS, read, and closed.
No API for switching desktops or sending input is present in the helper.
